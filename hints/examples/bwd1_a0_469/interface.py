"""Candidate implementation for causal depthwise conv1d backward."""

from __future__ import annotations

import hashlib
import struct

import torch
import triton
import triton.language as tl
from triton import knobs as _triton_knobs

from candidate.final_reduce_cuda import final_reduce as _cuda_final_reduce
from reference import kernel_fn as reference_kernel_fn


_ZEROED_PARTIAL_CACHE = {}


# Ptxas keeps a 64-bit uniform loop induction for the exact SM90 packed
# specialization even though its static row offset is bounded by 248.  Apply a
# fail-closed cubin peephole that deletes only the dead high-half add/compare
# and fills their slots with two independent tail reductions.  Every
# nonmatching kernel or specialization remains on the unmodified Triton path.
_NARROW_LOOP_TEXT_SECTION = ".text._fused_chunk_bwd_kernel"
_NARROW_LOOP_PARENT_SHA256 = (
    "3ddc4e5bbb3109e21c9adb1e19886d0f341d71bfa357b06ff88880bf43a53768"
)
_NARROW_LOOP_START = 0x1560
_NARROW_LOOP_BRANCH_PC = 0x25B0


def _sm90_instruction(lo, hi):
    return struct.pack("<QQ", lo, hi)


_NARROW_LOOP_HIGH_ADD = _sm90_instruction(
    0x000000053F057290,
    0x000FE200087FE43F,
)
_NARROW_LOOP_HIGH_COMPARE = _sm90_instruction(
    0x0000003F0500728C,
    0x000FE2000BF06100,
)
_NARROW_LOOP_TAIL_DBIAS_W3 = _sm90_instruction(
    0x0000001F11287223,
    0x000FE20000010028,
)
_NARROW_LOOP_TAIL_DWEIGHT = _sm90_instruction(
    0x0000001229187223,
    0x080FE20000010018,
)
_NARROW_LOOP_BRANCH = _sm90_instruction(
    0xFFFFFFEC00E8B947,
    0x000FF0000383FFFF,
)
_NARROW_LOOP_COMPACT_BRANCH = _sm90_instruction(
    0xFFFFFFEC00F0B947,
    0x000FF0000383FFFF,
)
_NARROW_LOOP_NOP = _sm90_instruction(
    0x0000000000007918,
    0x000FC00000000000,
)


def _elf64_sections(blob):
    if blob[:6] != b"\x7fELF\x02\x01":
        return {}
    section_table = struct.unpack_from("<Q", blob, 0x28)[0]
    section_size, section_count, string_index = struct.unpack_from(
        "<HHH", blob, 0x3A
    )
    if section_size != 64 or string_index >= section_count:
        return {}

    headers = []
    for index in range(section_count):
        base = section_table + index * section_size
        if base + section_size > len(blob):
            return {}
        name_offset = struct.unpack_from("<I", blob, base)[0]
        file_offset, byte_count = struct.unpack_from("<QQ", blob, base + 0x18)
        headers.append((name_offset, file_offset, byte_count))

    _, strings_offset, strings_size = headers[string_index]
    strings = blob[strings_offset : strings_offset + strings_size]
    sections = {}
    for name_offset, file_offset, byte_count in headers:
        name_end = strings.find(b"\0", name_offset)
        if name_end < 0:
            return {}
        try:
            name = strings[name_offset:name_end].decode("ascii")
        except UnicodeDecodeError:
            return {}
        sections[name] = (file_offset, byte_count)
    return sections


def _apply_narrow_loop_peephole(cubin):
    sections = _elf64_sections(cubin)
    text_entry = sections.get(_NARROW_LOOP_TEXT_SECTION)
    if text_entry is None:
        return cubin
    text_offset, text_size = text_entry
    if text_size != 0x4A00 or text_offset + text_size > len(cubin):
        return cubin

    recurrent = cubin[
        text_offset + _NARROW_LOOP_START : text_offset + 0x25C0
    ]
    if hashlib.sha256(recurrent).hexdigest() != _NARROW_LOOP_PARENT_SHA256:
        return cubin

    def read_pc(pc):
        return cubin[text_offset + pc : text_offset + pc + 16]

    expected = {
        0x2350: _NARROW_LOOP_HIGH_ADD,
        0x23B0: _NARROW_LOOP_HIGH_COMPARE,
        0x24F0: _NARROW_LOOP_TAIL_DBIAS_W3,
        0x2510: _NARROW_LOOP_TAIL_DWEIGHT,
        _NARROW_LOOP_BRANCH_PC: _NARROW_LOOP_BRANCH,
    }
    if any(read_pc(pc) != instruction for pc, instruction in expected.items()):
        return cubin

    transformed = []
    for pc in range(
        _NARROW_LOOP_START,
        _NARROW_LOOP_BRANCH_PC + 0x10,
        0x10,
    ):
        if pc == 0x2350:
            transformed.append(_NARROW_LOOP_TAIL_DBIAS_W3)
        elif pc == 0x23B0:
            transformed.append(_NARROW_LOOP_TAIL_DWEIGHT)
        elif pc in (0x24F0, 0x2510):
            continue
        else:
            transformed.append(read_pc(pc))
    if len(transformed) != 260 or transformed[-1] != _NARROW_LOOP_BRANCH:
        return cubin
    transformed[-1] = _NARROW_LOOP_COMPACT_BRANCH

    output = bytearray(cubin)
    loop_offset = text_offset + _NARROW_LOOP_START
    output[loop_offset : loop_offset + 260 * 16] = b"".join(transformed)
    output[text_offset + 0x25A0 : text_offset + 0x25C0] = (
        _NARROW_LOOP_NOP + _NARROW_LOOP_NOP
    )
    return bytes(output)


_PREVIOUS_TRITON_STAGE_HOOK = _triton_knobs.runtime.add_stages_inspection_hook


def _install_narrow_loop_stage(backend, stages, options, language, capability):
    if _PREVIOUS_TRITON_STAGE_HOOK is not None:
        _PREVIOUS_TRITON_STAGE_HOOK(
            backend,
            stages,
            options,
            language,
            capability,
        )
    if str(capability) != "90" or "cubin" not in stages:
        return
    make_cubin = stages["cubin"]

    def make_patched_cubin(src, metadata):
        cubin = make_cubin(src, metadata)
        if metadata.get("name") != "_fused_chunk_bwd_kernel":
            return cubin
        cubin = _apply_halo_composition_peephole(cubin)
        cubin = _apply_prologue_bos_shared_peephole(cubin)
        cubin = _apply_narrow_loop_peephole(cubin)
        cubin = _apply_inrange_dx_peephole(cubin)
        cubin = _apply_direct_valid3_peephole(cubin)
        cubin = _apply_predicate_yield_peephole(cubin)
        cubin = _apply_r1_bank_remap_peephole(cubin)
        cubin = _apply_delete_last_src3_fsel_peephole(cubin)
        cubin = _apply_delete_last_src2_fsel_peephole(cubin)
        cubin = _apply_delete_row2_r44_fsel_peephole(cubin)
        cubin = _apply_delete_entry_r57_fsel_peephole(cubin)
        cubin = _apply_delete_p1_r31_fsel_peephole(cubin)
        cubin = _apply_compose_entry_r45_a3_482_peephole(cubin)
        return _apply_compose_entry_r44_a5_440_peephole(cubin)

    stages["cubin"] = make_patched_cubin


_triton_knobs.runtime.add_stages_inspection_hook = _install_narrow_loop_stage
# Source is intentionally unchanged, so bypass any cached unpatched parent
# cubin. In-process JIT specialization caching still applies after compilation.
_triton_knobs.compilation.always_compile = True


@triton.jit
def _silu_backward_factor(z):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 y;
            .reg .f32 e;
            .reg .f32 denom;
            .reg .f32 sig;
            .reg .f32 one_minus;
            .reg .f32 inner;
            fma.rn.ftz.f32 y, $1, 0fBFB8AA3B, 0f00000000;
            ex2.approx.ftz.f32 e, y;
            add.rn.ftz.f32 denom, e, 0f3F800000;
            rcp.approx.ftz.f32 sig, denom;
            sub.rn.ftz.f32 one_minus, 0f3F800000, sig;
            fma.rn.ftz.f32 inner, $1, one_minus, 0f3F800000;
            mul.rn.ftz.f32 $0, sig, inner;
        }
        """,
        constraints="=f,f",
        args=[z],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _silu_backward_apply(z, dout):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 y0;
            .reg .f32 y1;
            .reg .f32 t0;
            .reg .f32 t1;
            .reg .f32 one_minus0;
            .reg .f32 one_minus1;
            .reg .f32 inner0;
            .reg .f32 inner1;
            fma.rn.ftz.f32 y0, $2, 0f3F000000, 0f00000000;
            fma.rn.ftz.f32 y1, $3, 0f3F000000, 0f00000000;
            tanh.approx.f32 t0, y0;
            tanh.approx.f32 t1, y1;
            fma.rn.ftz.f32 $0, t0, 0f3F000000, 0f3F000000;
            fma.rn.ftz.f32 $1, t1, 0f3F000000, 0f3F000000;
            fma.rn.ftz.f32 one_minus0, t0, 0fBF000000, 0f3F000000;
            fma.rn.ftz.f32 inner0, $2, one_minus0, 0f3F800000;
            mul.rn.ftz.f32 $0, $0, inner0;
            fma.rn.ftz.f32 one_minus1, t1, 0fBF000000, 0f3F000000;
            fma.rn.ftz.f32 inner1, $3, one_minus1, 0f3F800000;
            mul.rn.ftz.f32 $1, $1, inner1;
            mul.rn.ftz.f32 $1, $5, $1;
            mul.rn.ftz.f32 $0, $4, $0;
        }
        """,
        constraints="=f,=f,f,f,f,f",
        args=[z, dout],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _load_weight4_bf16(weight_ptr, d, d_mask):
    packed = tl.load(
        weight_ptr.to(tl.pointer_type(tl.uint64)) + d,
        mask=d_mask,
        other=0,
    )
    w0_bits = ((packed >> 0) & 0xFFFF).to(tl.uint32)
    w1_bits = ((packed >> 16) & 0xFFFF).to(tl.uint32)
    w2_bits = ((packed >> 32) & 0xFFFF).to(tl.uint32)
    w3_bits = ((packed >> 48) & 0xFFFF).to(tl.uint32)
    w0 = (w0_bits << 16).to(tl.float32, bitcast=True)
    w1 = (w1_bits << 16).to(tl.float32, bitcast=True)
    w2 = (w2_bits << 16).to(tl.float32, bitcast=True)
    w3 = (w3_bits << 16).to(tl.float32, bitcast=True)
    return w0, w1, w2, w3


@triton.jit
def _load_bias_bf16_u16(bias_ptr, d, d_mask):
    return tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)


@triton.jit
def _prefetch_l2_if(ptr, do_prefetch):
    return tl.inline_asm_elementwise(
        asm="""
        {
            @$2 prefetch.global.L2 [$1];
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,b",
        args=[ptr, do_prefetch],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _opaque_packed_lane_address(ptr):
    """Keep one physical packed-row address per thread in regular registers."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            mov.b64 $0, $1;
        }
        """,
        constraints="=l,l",
        args=[ptr],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _advance_packed_lane_address(addr, byte_step):
    """Advance an opaque packed address without rebuilding it from the lane."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            add.u64 $0, $1, $2;
        }
        """,
        constraints="=l,l,l",
        args=[addr, byte_step],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _load_packed_lane_at_byte_offset(addr, byte_offset):
    """Issue the parent's volatile 64-bit packed load from an opaque base."""
    lo, hi = tl.inline_asm_elementwise(
        asm="""
        {
            ld.volatile.global.v2.u32 {$0, $1}, [$2+$3];
        }
        """,
        constraints="=r,=r,l,n",
        args=[addr, byte_offset],
        dtype=(tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )
    return tl.interleave(lo, hi)


@triton.jit
def _prefetch_l2_lane_at_byte_offset(addr, byte_offset, do_prefetch):
    """Prefetch both packed-row cache lines through their physical lanes."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            @$3 prefetch.global.L2 [$1+$2];
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,n,b",
        args=[addr, byte_offset, do_prefetch],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _stage_bos_chunk_shared_scalar(bos_chunk_ptr, is_first_chunk):
    """Copy a 16-byte BOS prefix plus one 256-byte chunk per CTA."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .shared .align 16 .b8 bos_chunk_smem_scalar[272];
            .reg .u32 smem;
            .reg .u32 lane;
            .reg .u32 off;
            .reg .u32 saddr;
            .reg .u64 off64;
            .reg .u64 gbase;
            .reg .u64 gaddr;
            .reg .pred copy;
            .reg .pred first;
            .reg .pred lane_zero;
            .reg .pred clamp;
            mov.u32 smem, bos_chunk_smem_scalar;
            mov.u32 lane, %tid.x;
            shl.b32 off, lane, 4;
            add.u32 saddr, smem, off;
            cvt.u64.u32 off64, off;
            sub.u64 gbase, $1, 16;
            setp.ne.u32 first, $2, 0;
            setp.eq.u32 lane_zero, lane, 0;
            and.pred clamp, first, lane_zero;
            @clamp mov.u64 gbase, $1;
            add.u64 gaddr, gbase, off64;
            setp.lt.u32 copy, lane, 17;
            @copy cp.async.cg.shared.global [saddr], [gaddr], 16;
            cp.async.commit_group;
            cp.async.wait_group 0;
            bar.sync 0;
            mov.u32 $0, smem;
        }
        """,
        constraints="=r,l,r",
        args=[bos_chunk_ptr, is_first_chunk],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _load_bos_pair_shared_absolute(shared_addr):
    """Load one uniform BOS halfword from an absolute shared address."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b16 pair;
            ld.shared.u16 pair, [$1];
            cvt.u32.u16 $0, pair;
        }
        """,
        constraints="=r,r",
        args=[shared_addr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _unpack_xdout_u32(packed):
    x_bits, dout_bits = tl.inline_asm_elementwise(
        asm="""
        {
            prmt.b32 $0, $2, 0, 0x1044;
            and.b32 $1, $2, 0xffff0000;
        }
        """,
        constraints="=r,=r,r",
        args=[packed],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )
    return x_bits.to(tl.float32, bitcast=True), dout_bits.to(tl.float32, bitcast=True)


@triton.jit
def _unpack_x_u32(packed):
    x_bits = tl.inline_asm_elementwise(
        asm="""
        {
            shl.b32 $0, $1, 16;
        }
        """,
        constraints="=r,r",
        args=[packed],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return x_bits.to(tl.float32, bitcast=True)


@triton.jit
def _conv4_ffma_z(x0, src1, src2, src3, w0, w1, w2, w3, bias):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 acc0;
            .reg .f32 acc1;
            mov.f32 acc0, $18;
            mov.f32 acc1, $19;
            fma.rn.ftz.f32 acc0, $8, $10, acc0;
            fma.rn.ftz.f32 acc1, $9, $11, acc1;
            fma.rn.ftz.f32 acc0, $6, $12, acc0;
            fma.rn.ftz.f32 acc1, $7, $13, acc1;
            fma.rn.ftz.f32 acc0, $4, $14, acc0;
            fma.rn.ftz.f32 acc1, $5, $15, acc1;
            fma.rn.ftz.f32 $0, $2, $16, acc0;
            fma.rn.ftz.f32 $1, $3, $17, acc1;
        }
        """,
        constraints="=f,=f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f",
        args=[x0, src1, src2, src3, w0, w1, w2, w3, bias],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _masked_ffma_acc(acc, a, b, valid):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred p;
            setp.ne.u32 p, $4, 0;
            mov.f32 $0, $1;
            @p fma.rn.ftz.f32 $0, $2, $3, $1;
        }
        """,
        constraints="=f,f,f,f,r",
        args=[acc, a, b, valid.to(tl.int32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _ffma_acc(acc, a, b):
    return tl.inline_asm_elementwise(
        asm="""
        {
            fma.rn.ftz.f32 $0, $2, $3, $1;
        }
        """,
        constraints="=f,f,f,f",
        args=[acc, a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _ffma_one_acc(acc, x):
    return tl.inline_asm_elementwise(
        asm="""
        {
            fma.rn.ftz.f32 $0, $2, 0f3F800000, $1;
        }
        """,
        constraints="=f,f,f",
        args=[acc, x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _store_bf16_trunc(ptr, offsets, value, mask):
    bits = value.to(tl.uint32, bitcast=True)
    half = (bits >> 16).to(tl.uint16)
    tl.store(ptr.to(tl.pointer_type(tl.uint16)) + offsets, half, mask=mask)


@triton.jit
def _masked_ffma3_acc(acc, a0, b0, valid0, a1, b1, valid1, a2, b2, valid2):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred p00;
            .reg .pred p01;
            .reg .pred p10;
            .reg .pred p11;
            .reg .pred p20;
            .reg .pred p21;
            setp.ne.u32 p00, $8, 0;
            setp.ne.u32 p01, $9, 0;
            setp.ne.u32 p10, $14, 0;
            setp.ne.u32 p11, $15, 0;
            setp.ne.u32 p20, $20, 0;
            setp.ne.u32 p21, $21, 0;
            mov.f32 $0, $2;
            mov.f32 $1, $3;
            @p00 fma.rn.ftz.f32 $0, $4, $6, $0;
            @p01 fma.rn.ftz.f32 $1, $5, $7, $1;
            @p20 fma.rn.ftz.f32 $0, $16, $18, $0;
            @p21 fma.rn.ftz.f32 $1, $17, $19, $1;
            @p10 fma.rn.ftz.f32 $0, $10, $12, $0;
            @p11 fma.rn.ftz.f32 $1, $11, $13, $1;
        }
        """,
        constraints="=&f,=&f,f,f,f,f,f,f,r,r,f,f,f,f,r,r,f,f,f,f,r,r",
        args=[
            acc,
            a0,
            b0,
            valid0.to(tl.int32),
            a1,
            b1,
            valid1.to(tl.int32),
            a2,
            b2,
            valid2.to(tl.int32),
        ],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _compute_g_t8_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    initial_ptr,
    bos_ptr,
    dout_ptr,
    g_ptr,
    stage1_clear_ptr,
    dx_valid_ptr,
    dx_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    block_d: tl.constexpr,
    tile_rows_per_batch: tl.constexpr,
    num_d_blocks: tl.constexpr,
    swizzle_group_rows: tl.constexpr,
):
    pid = tl.program_id(0)
    if swizzle_group_rows:
        swizzle_group_size: tl.constexpr = num_d_blocks * swizzle_group_rows
        group = pid // swizzle_group_size
        within_group = pid - group * swizzle_group_size
        pid_d = within_group // swizzle_group_rows
        tile_row = group * swizzle_group_rows + (
            within_group - pid_d * swizzle_group_rows
        )
    else:
        pid_d = pid % num_d_blocks
        tile_row = pid // num_d_blocks
    b = tile_row // tile_rows_per_batch
    tile_t = tile_row - b * tile_rows_per_batch

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim
    w0, w1, w2, w3 = _load_weight4_bf16(weight_ptr, d, d_mask)
    bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    b_i64 = b.to(tl.int64)
    dim_i64 = tl.full((), dim, tl.int64)
    seqlen_i64 = tl.full((), seqlen, tl.int64)
    tile_start = tile_t * 16
    tile_start_i64 = tile_start.to(tl.int64)
    tile_base = (b_i64 * seqlen_i64 + tile_start_i64) * dim_i64 + d.to(tl.int64)

    prev1 = tl.load(
        x_ptr + tile_base - dim_i64,
        mask=d_mask & (tile_start >= 1),
        other=0.0,
    ).to(tl.float32)
    prev2 = tl.load(
        x_ptr + tile_base - 2 * dim_i64,
        mask=d_mask & (tile_start >= 2),
        other=0.0,
    ).to(tl.float32)
    prev3 = tl.load(
        x_ptr + tile_base - 3 * dim_i64,
        mask=d_mask & (tile_start >= 3),
        other=0.0,
    ).to(tl.float32)

    g_m3 = tl.full((block_d,), 0.0, tl.float32)
    g_m2 = tl.full((block_d,), 0.0, tl.float32)
    g_m1 = tl.full((block_d,), 0.0, tl.float32)
    for i in tl.static_range(0, 16):
        t = tile_t * 16 + i
        row_valid = t < seqlen
        base = tile_base + i * dim_i64

        x0 = tl.load(x_ptr + base, mask=d_mask & row_valid, other=0.0).to(tl.float32)

        src1 = tl.where(t >= 1, prev1, 0.0)
        src2 = tl.where(t >= 2, prev2, 0.0)
        src3 = tl.where(t >= 3, prev3, 0.0)

        z = _conv4_ffma_z(x0, src1, src2, src3, w0, w1, w2, w3, bias)
        dout = tl.load(dout_ptr + base, mask=d_mask & row_valid, other=0.0).to(
            tl.float32
        )
        g = dout * _silu_backward_factor(z)
        tl.store(g_ptr + base, g, mask=d_mask & row_valid)

        if i >= 3:
            dx_t = t - 3
            dx_base = tile_base + (i - 3) * dim_i64
            dx = g_m3 * w3 + g_m2 * w2
            dx += g_m1 * w1 + g * w0
            tl.store(dx_ptr + dx_base, dx, mask=d_mask & ((dx_t + 3) < seqlen))

        g_m3 = g_m2
        g_m2 = g_m1
        g_m1 = g
        prev3 = prev2
        prev2 = prev1
        prev1 = x0

    next_g0 = tl.full((block_d,), 0.0, tl.float32)
    next_g1 = tl.full((block_d,), 0.0, tl.float32)
    next_g2 = tl.full((block_d,), 0.0, tl.float32)
    next_prev1 = prev1
    next_prev2 = prev2
    next_prev3 = prev3
    for j in tl.static_range(0, 3):
        t = tile_t * 16 + 16 + j
        row_valid = t < seqlen
        base = tile_base + (16 + j) * dim_i64
        x_next = tl.load(x_ptr + base, mask=d_mask & row_valid, other=0.0).to(
            tl.float32
        )
        src1 = next_prev1
        src2 = next_prev2
        src3 = next_prev3
        z = _conv4_ffma_z(x_next, src1, src2, src3, w0, w1, w2, w3, bias)
        dout = tl.load(dout_ptr + base, mask=d_mask & row_valid, other=0.0).to(
            tl.float32
        )
        g_next = dout * _silu_backward_factor(z)
        if j == 0:
            next_g0 = g_next
        elif j == 1:
            next_g1 = g_next
        else:
            next_g2 = g_next
        next_prev3 = next_prev2
        next_prev2 = next_prev1
        next_prev1 = x_next

    bos_base_i64 = b_i64 * seqlen_i64
    bos14 = tl.load(
        bos_ptr + bos_base_i64 + tile_start_i64 + 14,
        mask=(tile_start + 14) < seqlen,
        other=True,
    )
    bos15 = tl.load(
        bos_ptr + bos_base_i64 + tile_start_i64 + 15,
        mask=(tile_start + 15) < seqlen,
        other=True,
    )
    bos16 = tl.load(
        bos_ptr + bos_base_i64 + tile_start_i64 + 16,
        mask=(tile_start + 16) < seqlen,
        other=True,
    )
    bos17 = tl.load(
        bos_ptr + bos_base_i64 + tile_start_i64 + 17,
        mask=(tile_start + 17) < seqlen,
        other=True,
    )
    bos18 = tl.load(
        bos_ptr + bos_base_i64 + tile_start_i64 + 18,
        mask=(tile_start + 18) < seqlen,
        other=True,
    )

    boundary_base = tile_base + 13 * dim_i64
    v13_1 = ((tile_start + 14) < seqlen) & (~bos14)
    v13_2 = v13_1 & ((tile_start + 15) < seqlen) & (~bos15)
    v13_3 = v13_2 & ((tile_start + 16) < seqlen) & (~bos16)
    dx13 = g_m3 * w3
    dx13 = _masked_ffma3_acc(
        dx13,
        g_m2,
        w2,
        v13_1,
        g_m1,
        w1,
        v13_2,
        next_g0,
        w0,
        v13_3,
    )
    tl.store(dx_ptr + boundary_base, dx13, mask=d_mask & ((tile_start + 13) < seqlen))

    v14_1 = ((tile_start + 15) < seqlen) & (~bos15)
    v14_2 = v14_1 & ((tile_start + 16) < seqlen) & (~bos16)
    v14_3 = v14_2 & ((tile_start + 17) < seqlen) & (~bos17)
    dx14 = g_m2 * w3
    dx14 = _masked_ffma3_acc(
        dx14,
        g_m1,
        w2,
        v14_1,
        next_g0,
        w1,
        v14_2,
        next_g1,
        w0,
        v14_3,
    )
    tl.store(
        dx_ptr + boundary_base + dim_i64,
        dx14,
        mask=d_mask & ((tile_start + 14) < seqlen),
    )

    v15_1 = ((tile_start + 16) < seqlen) & (~bos16)
    v15_2 = v15_1 & ((tile_start + 17) < seqlen) & (~bos17)
    v15_3 = v15_2 & ((tile_start + 18) < seqlen) & (~bos18)
    dx15 = g_m1 * w3
    dx15 = _masked_ffma3_acc(
        dx15,
        next_g0,
        w2,
        v15_1,
        next_g1,
        w1,
        v15_2,
        next_g2,
        w0,
        v15_3,
    )
    tl.store(
        dx_ptr + boundary_base + 2 * dim_i64,
        dx15,
        mask=d_mask & ((tile_start + 15) < seqlen),
    )

    if pid_d == 0:
        row_offsets = tile_t * 16 + tl.arange(0, 16)
        row_valids = row_offsets < seqlen
        bos_base = b * seqlen
        bos_t = tl.load(
            bos_ptr + bos_base + row_offsets,
            mask=row_valids,
            other=0,
        )
        bos_tm1 = tl.load(
            bos_ptr + bos_base + row_offsets - 1,
            mask=row_valids & (row_offsets >= 1),
            other=0,
        )
        bos_tm2 = tl.load(
            bos_ptr + bos_base + row_offsets - 2,
            mask=row_valids & (row_offsets >= 2),
            other=0,
        )

        clear0_vec = row_valids & (~bos_t)
        clear1_vec = clear0_vec & ((row_offsets < 1) | (~bos_tm1))
        clear2_vec = clear1_vec & ((row_offsets < 2) | (~bos_tm2))
        clear_code = (
            tl.where(clear0_vec, 1, 0)
            | tl.where(clear1_vec, 2, 0)
            | tl.where(clear2_vec, 4, 0)
        )
        tl.store(stage1_clear_ptr + bos_base + row_offsets, clear_code, mask=row_valids)

        bos_tp1 = tl.load(
            bos_ptr + bos_base + row_offsets + 1,
            mask=(row_offsets + 1) < seqlen,
            other=0,
        )
        bos_tp2 = tl.load(
            bos_ptr + bos_base + row_offsets + 2,
            mask=(row_offsets + 2) < seqlen,
            other=0,
        )
        bos_tp3 = tl.load(
            bos_ptr + bos_base + row_offsets + 3,
            mask=(row_offsets + 3) < seqlen,
            other=0,
        )
        dx_valid1 = ((row_offsets + 1) < seqlen) & (~bos_tp1)
        dx_valid2 = dx_valid1 & ((row_offsets + 2) < seqlen) & (~bos_tp2)
        dx_valid3 = dx_valid2 & ((row_offsets + 3) < seqlen) & (~bos_tp3)
        dx_valid_code = (
            tl.where(dx_valid1, 1, 0)
            | tl.where(dx_valid2, 2, 0)
            | tl.where(dx_valid3, 4, 0)
        )
        tl.store(dx_valid_ptr + bos_base + row_offsets, dx_valid_code, mask=row_valids)


@triton.jit
def _compute_g_dirty_tile_marker_kernel(
    bos_ptr,
    dirty_ptr,
    stage1_clear_ptr,
    dx_valid_ptr,
    seqlen: tl.constexpr,
    tiles_t: tl.constexpr,
):
    b = tl.program_id(0)
    tile_t = tl.program_id(1)

    row_offsets = tile_t * 16 + tl.arange(0, 16)
    row_valids = row_offsets < seqlen
    bos_base = b * seqlen
    bos_t = tl.load(
        bos_ptr + bos_base + row_offsets,
        mask=row_valids,
        other=0,
    )
    bos_tm1 = tl.load(
        bos_ptr + bos_base + row_offsets - 1,
        mask=row_valids & (row_offsets >= 1),
        other=0,
    )
    bos_tm2 = tl.load(
        bos_ptr + bos_base + row_offsets - 2,
        mask=row_valids & (row_offsets >= 2),
        other=0,
    )

    clear0_vec = row_valids & (~bos_t)
    clear1_vec = clear0_vec & ((row_offsets < 1) | (~bos_tm1))
    clear2_vec = clear1_vec & ((row_offsets < 2) | (~bos_tm2))
    clear_code = (
        tl.where(clear0_vec, 1, 0)
        | tl.where(clear1_vec, 2, 0)
        | tl.where(clear2_vec, 4, 0)
    )
    tl.store(stage1_clear_ptr + bos_base + row_offsets, clear_code, mask=row_valids)

    bos_tp1 = tl.load(
        bos_ptr + bos_base + row_offsets + 1,
        mask=(row_offsets + 1) < seqlen,
        other=0,
    )
    bos_tp2 = tl.load(
        bos_ptr + bos_base + row_offsets + 2,
        mask=(row_offsets + 2) < seqlen,
        other=0,
    )
    bos_tp3 = tl.load(
        bos_ptr + bos_base + row_offsets + 3,
        mask=(row_offsets + 3) < seqlen,
        other=0,
    )
    dx_valid1 = ((row_offsets + 1) < seqlen) & (~bos_tp1)
    dx_valid2 = dx_valid1 & ((row_offsets + 2) < seqlen) & (~bos_tp2)
    dx_valid3 = dx_valid2 & ((row_offsets + 3) < seqlen) & (~bos_tp3)
    dx_valid_code = (
        tl.where(dx_valid1, 1, 0)
        | tl.where(dx_valid2, 2, 0)
        | tl.where(dx_valid3, 4, 0)
    )
    tl.store(dx_valid_ptr + bos_base + row_offsets, dx_valid_code, mask=row_valids)

    dirty_vec = row_valids & (
        (row_offsets < 3) | bos_t | bos_tm1 | bos_tm2
    )

    bit_values = tl.full((16,), 1, tl.int64) << tl.arange(0, 16)
    dirty_bits = tl.sum(tl.where(dirty_vec, bit_values, 0), axis=0)
    clear0_bits = tl.sum(tl.where(clear0_vec, bit_values, 0), axis=0)
    clear1_bits = tl.sum(tl.where(clear1_vec, bit_values, 0), axis=0)
    clear2_bits = tl.sum(tl.where(clear2_vec, bit_values, 0), axis=0)
    packed = (
        dirty_bits
        | (clear0_bits << 16)
        | (clear1_bits << 32)
        | (clear2_bits << 48)
    )
    tl.store(
        dirty_ptr + b * tiles_t + tile_t,
        tl.where(dirty_bits != 0, packed, 0),
    )


@triton.jit
def _compute_g_dirty_compact_repair_t8_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    initial_ptr,
    bos_ptr,
    dout_ptr,
    dirty_idx_ptr,
    dirty_meta_ptr,
    g_ptr,
    dx_valid_ptr,
    dx_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    block_d: tl.constexpr,
    tiles_t: tl.constexpr,
):
    pid_d = tl.program_id(0)
    repair_id = tl.program_id(1)

    b = tl.load(dirty_idx_ptr + repair_id * 2 + 0).to(tl.int64)
    tile_t = tl.load(dirty_idx_ptr + repair_id * 2 + 1).to(tl.int64)

    meta = tl.load(dirty_meta_ptr + b * tiles_t + tile_t).to(tl.int64)
    dirty_bits = (meta & 0xFFFF).to(tl.int32)
    clear0_bits = ((meta >> 16) & 0xFFFF).to(tl.int32)
    clear1_bits = ((meta >> 32) & 0xFFFF).to(tl.int32)
    clear2_bits = ((meta >> 48) & 0xFFFF).to(tl.int32)

    seqlen_i64 = tl.full((), seqlen, tl.int64)
    tile_start = tile_t * 16

    if True:
        d = pid_d * block_d + tl.arange(0, block_d)
        d_mask = d < dim
        w0, w1, w2, w3 = _load_weight4_bf16(weight_ptr, d, d_mask)
        bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

        dim_i64 = tl.full((), dim, tl.int64)
        tile_base = (b * seqlen_i64 + tile_start) * dim_i64 + d.to(tl.int64)

        if tile_start >= 3:
            for i in tl.static_range(0, 16):
                if (dirty_bits & (1 << i)) != 0:
                    t = tile_t * 16 + i
                    row_valid = t < seqlen
                    base = tile_base + i * dim_i64

                    bit = 1 << i
                    clear0 = (clear0_bits & bit) != 0
                    clear1 = (clear1_bits & bit) != 0
                    clear2 = (clear2_bits & bit) != 0

                    x0 = tl.load(
                        x_ptr + base,
                        mask=d_mask & row_valid,
                        other=0.0,
                    ).to(tl.float32)
                    src1_x = tl.load(
                        x_ptr + base - dim_i64,
                        mask=d_mask & row_valid & clear0,
                        other=0.0,
                    ).to(tl.float32)
                    src2_x = tl.load(
                        x_ptr + base - 2 * dim_i64,
                        mask=d_mask & row_valid & clear1,
                        other=0.0,
                    ).to(tl.float32)
                    src3_x = tl.load(
                        x_ptr + base - 3 * dim_i64,
                        mask=d_mask & row_valid & clear2,
                        other=0.0,
                    ).to(tl.float32)

                    z = x0 * w3 + src1_x * w2
                    z += src2_x * w1
                    z += src3_x * w0 + bias
                    dout = tl.load(
                        dout_ptr + base,
                        mask=d_mask & row_valid,
                        other=0.0,
                    ).to(tl.float32)
                    g = dout * _silu_backward_factor(z)
                    tl.store(g_ptr + base, g, mask=d_mask & row_valid)

        else:
            init_base = b * dim_i64 * 3 + d.to(tl.int64) * 3
            for i in tl.static_range(0, 16):
                if (dirty_bits & (1 << i)) != 0:
                    t = tile_t * 16 + i
                    row_valid = t < seqlen
                    base = tile_base + i * dim_i64

                    bit = 1 << i
                    clear0 = (clear0_bits & bit) != 0
                    clear1 = (clear1_bits & bit) != 0
                    clear2 = (clear2_bits & bit) != 0

                    x0 = tl.load(
                        x_ptr + base,
                        mask=d_mask & row_valid,
                        other=0.0,
                    ).to(tl.float32)
                    src1_x = tl.load(
                        x_ptr + base - dim_i64,
                        mask=d_mask & row_valid & (t >= 1) & clear0,
                        other=0.0,
                    ).to(tl.float32)
                    src1_i = tl.load(
                        initial_ptr + init_base + 2,
                        mask=d_mask & row_valid & (t < 1) & clear0,
                        other=0.0,
                    ).to(tl.float32)

                    src2_x = tl.load(
                        x_ptr + base - 2 * dim_i64,
                        mask=d_mask & row_valid & (t >= 2) & clear1,
                        other=0.0,
                    ).to(tl.float32)
                    src2_i = tl.load(
                        initial_ptr + init_base + (t + 1),
                        mask=d_mask & row_valid & (t < 2) & clear1,
                        other=0.0,
                    ).to(tl.float32)

                    src3_x = tl.load(
                        x_ptr + base - 3 * dim_i64,
                        mask=d_mask & row_valid & (t >= 3) & clear2,
                        other=0.0,
                    ).to(tl.float32)
                    src3_i = tl.load(
                        initial_ptr + init_base + t,
                        mask=d_mask & row_valid & (t < 3) & clear2,
                        other=0.0,
                    ).to(tl.float32)

                    z = x0 * w3 + (src1_x + src1_i) * w2
                    z += (src2_x + src2_i) * w1
                    z += (src3_x + src3_i) * w0 + bias
                    dout = tl.load(
                        dout_ptr + base,
                        mask=d_mask & row_valid,
                        other=0.0,
                    ).to(tl.float32)
                    g = dout * _silu_backward_factor(z)
                    tl.store(g_ptr + base, g, mask=d_mask & row_valid)

        repair_bits = (
            dirty_bits
            | (dirty_bits >> 1)
            | (dirty_bits >> 2)
            | (dirty_bits >> 3)
        ) & 0x1FFF
        valid_base = b * seqlen_i64
        valid_tile_base = valid_base + tile_start

        for i in tl.static_range(0, 13):
            if (repair_bits & (1 << i)) != 0:
                t = tile_t * 16 + i
                row_valid = t < seqlen
                base = tile_base + i * dim_i64
                valid_code = tl.load(
                    dx_valid_ptr + valid_tile_base + i,
                    mask=row_valid,
                    other=0,
                ).to(tl.int32)
                valid1 = (valid_code & 1) != 0
                valid2 = (valid_code & 2) != 0
                valid3 = (valid_code & 4) != 0

                g0 = tl.load(g_ptr + base, mask=d_mask & row_valid, other=0.0)
                g1 = tl.load(g_ptr + base + dim_i64, mask=d_mask & valid1, other=0.0)
                g2 = tl.load(
                    g_ptr + base + 2 * dim_i64,
                    mask=d_mask & valid2,
                    other=0.0,
                )
                g3 = tl.load(
                    g_ptr + base + 3 * dim_i64,
                    mask=d_mask & valid3,
                    other=0.0,
                )
                dx = g0 * w3 + g1 * w2 + g2 * w1 + g3 * w0

                tl.store(dx_ptr + base, dx, mask=d_mask & row_valid)


@triton.jit
def _compute_g_prefix_repair_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    initial_ptr,
    bos_ptr,
    dout_ptr,
    g_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    block_d: tl.constexpr,
):
    pid_d = tl.program_id(0)
    b = tl.program_id(1)

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim
    w0, w1, w2, w3 = _load_weight4_bf16(weight_ptr, d, d_mask)
    bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    b_i64 = b.to(tl.int64)
    dim_i64 = tl.full((), dim, tl.int64)
    seqlen_i64 = tl.full((), seqlen, tl.int64)
    row0 = b_i64 * seqlen_i64 * dim_i64 + d.to(tl.int64)
    init_base = b_i64 * dim_i64 * 3 + d.to(tl.int64) * 3
    bos_base = b_i64 * seqlen_i64

    for i in tl.static_range(0, 3):
        row_valid = i < seqlen
        base = row0 + i * dim_i64
        bos_t = tl.load(bos_ptr + bos_base + i, mask=row_valid, other=True)
        clear0 = row_valid & (~bos_t)

        x0 = tl.load(x_ptr + base, mask=d_mask & row_valid, other=0.0).to(tl.float32)
        z = x0 * w3 + bias
        if i == 0:
            src1_i = tl.load(
                initial_ptr + init_base + 2,
                mask=d_mask & clear0,
                other=0.0,
            ).to(tl.float32)
            src2_i = tl.load(
                initial_ptr + init_base + 1,
                mask=d_mask & clear0,
                other=0.0,
            ).to(tl.float32)
            src3_i = tl.load(
                initial_ptr + init_base + 0,
                mask=d_mask & clear0,
                other=0.0,
            ).to(tl.float32)
            z += src1_i * w2
            z += src2_i * w1
            z += src3_i * w0
        elif i == 1:
            bos_tm1 = tl.load(
                bos_ptr + bos_base + 0,
                mask=row_valid,
                other=False,
            )
            clear1 = clear0 & (~bos_tm1)
            src1_x = tl.load(
                x_ptr + base - dim_i64,
                mask=d_mask & clear0,
                other=0.0,
            ).to(tl.float32)
            src2_i = tl.load(
                initial_ptr + init_base + 2,
                mask=d_mask & clear1,
                other=0.0,
            ).to(tl.float32)
            src3_i = tl.load(
                initial_ptr + init_base + 1,
                mask=d_mask & clear1,
                other=0.0,
            ).to(tl.float32)
            z += src1_x * w2
            z += src2_i * w1
            z += src3_i * w0
        else:
            bos_tm1 = tl.load(
                bos_ptr + bos_base + 1,
                mask=row_valid,
                other=False,
            )
            bos_tm2 = tl.load(
                bos_ptr + bos_base + 0,
                mask=row_valid,
                other=False,
            )
            clear1 = clear0 & (~bos_tm1)
            clear2 = clear1 & (~bos_tm2)
            src1_x = tl.load(
                x_ptr + base - dim_i64,
                mask=d_mask & clear0,
                other=0.0,
            ).to(tl.float32)
            src2_x = tl.load(
                x_ptr + base - 2 * dim_i64,
                mask=d_mask & clear1,
                other=0.0,
            ).to(tl.float32)
            src3_i = tl.load(
                initial_ptr + init_base + 2,
                mask=d_mask & clear2,
                other=0.0,
            ).to(tl.float32)
            z += src1_x * w2
            z += src2_x * w1
            z += src3_i * w0
        dout = tl.load(dout_ptr + base, mask=d_mask & row_valid, other=0.0).to(
            tl.float32
        )
        g = dout * _silu_backward_factor(z)
        tl.store(g_ptr + base, g, mask=d_mask & row_valid)


@triton.jit
def _compute_g_bos_offsets_repair_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    bos_ptr,
    dout_ptr,
    bos_offsets_ptr,
    g_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    block_d: tl.constexpr,
    num_offsets: tl.constexpr,
    num_d_blocks: tl.constexpr,
    repair_group_offsets: tl.constexpr,
):
    pid = tl.program_id(0)
    if repair_group_offsets:
        repair_group_size: tl.constexpr = num_d_blocks * repair_group_offsets
        group = pid // repair_group_size
        within_group = pid - group * repair_group_size
        pid_d = within_group // repair_group_offsets
        offset_id = group * repair_group_offsets + (
            within_group - pid_d * repair_group_offsets
        )
    else:
        pid_d = pid % num_d_blocks
        offset_id = pid // num_d_blocks

    b = tl.load(bos_offsets_ptr + offset_id * 2 + 0).to(tl.int64)
    bos_t0 = tl.load(bos_offsets_ptr + offset_id * 2 + 1).to(tl.int64)
    has_next = (offset_id + 1) < num_offsets
    next_b = tl.load(
        bos_offsets_ptr + (offset_id + 1) * 2 + 0,
        mask=has_next,
        other=-1,
    ).to(tl.int64)
    next_bos_t = tl.load(
        bos_offsets_ptr + (offset_id + 1) * 2 + 1,
        mask=has_next,
        other=-1,
    ).to(tl.int64)
    next_same_batch = has_next & (next_b == b)

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim
    w0, w1, w2, w3 = _load_weight4_bf16(weight_ptr, d, d_mask)
    bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    dim_i64 = tl.full((), dim, tl.int64)
    seqlen_i64 = tl.full((), seqlen, tl.int64)

    for j in tl.static_range(0, 3):
        t = bos_t0 + j
        row_valid = t < seqlen
        if j == 0:
            owner = True
        elif j == 1:
            owner = ~(next_same_batch & (next_bos_t == (bos_t0 + 1)))
        else:
            owner = ~(next_same_batch & (next_bos_t <= (bos_t0 + 2)))

        repair_row = row_valid & owner & (t >= 3)
        base = (b * seqlen_i64 + t) * dim_i64 + d.to(tl.int64)

        x0 = tl.load(x_ptr + base, mask=d_mask & repair_row, other=0.0).to(
            tl.float32
        )
        z = x0 * w3 + bias
        if j >= 1:
            src1 = tl.load(
                x_ptr + base - dim_i64,
                mask=d_mask & repair_row,
                other=0.0,
            ).to(tl.float32)
            z += src1 * w2
        if j >= 2:
            src2 = tl.load(
                x_ptr + base - 2 * dim_i64,
                mask=d_mask & repair_row,
                other=0.0,
            ).to(tl.float32)
            z += src2 * w1
        dout = tl.load(dout_ptr + base, mask=d_mask & repair_row, other=0.0).to(
            tl.float32
        )
        g = dout * _silu_backward_factor(z)
        tl.store(g_ptr + base, g, mask=d_mask & repair_row)


@triton.jit
def _compute_dx_prefix_repair_kernel(
    g_ptr,
    weight_ptr,
    dx_valid_ptr,
    dx_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    block_d: tl.constexpr,
):
    pid_d = tl.program_id(0)
    b = tl.program_id(1)

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim
    w0, w1, w2, w3 = _load_weight4_bf16(weight_ptr, d, d_mask)

    b_i64 = b.to(tl.int64)
    dim_i64 = tl.full((), dim, tl.int64)
    seqlen_i64 = tl.full((), seqlen, tl.int64)
    valid_base = b_i64 * seqlen_i64
    row0 = b_i64 * seqlen_i64 * dim_i64 + d.to(tl.int64)

    for i in tl.static_range(0, 3):
        row_valid = i < seqlen
        base = row0 + i * dim_i64
        valid_code = tl.load(dx_valid_ptr + valid_base + i, mask=row_valid, other=0).to(
            tl.int32
        )
        valid1 = (valid_code & 1) != 0
        valid2 = (valid_code & 2) != 0
        valid3 = (valid_code & 4) != 0
        g0 = tl.load(g_ptr + base, mask=d_mask & row_valid, other=0.0)
        g1 = tl.load(g_ptr + base + dim_i64, mask=d_mask & valid1, other=0.0)
        g2 = tl.load(g_ptr + base + 2 * dim_i64, mask=d_mask & valid2, other=0.0)
        g3 = tl.load(g_ptr + base + 3 * dim_i64, mask=d_mask & valid3, other=0.0)
        dx = g0 * w3 + g1 * w2 + g2 * w1 + g3 * w0
        tl.store(dx_ptr + base, dx, mask=d_mask & row_valid)


@triton.jit
def _compute_dx_bos_offsets_repair_kernel(
    g_ptr,
    weight_ptr,
    dx_valid_ptr,
    bos_offsets_ptr,
    dx_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    block_d: tl.constexpr,
    num_d_blocks: tl.constexpr,
    repair_group_offsets: tl.constexpr,
):
    pid = tl.program_id(0)
    if repair_group_offsets:
        repair_group_size: tl.constexpr = num_d_blocks * repair_group_offsets
        group = pid // repair_group_size
        within_group = pid - group * repair_group_size
        pid_d = within_group // repair_group_offsets
        offset_id = group * repair_group_offsets + (
            within_group - pid_d * repair_group_offsets
        )
    else:
        pid_d = pid % num_d_blocks
        offset_id = pid // num_d_blocks

    b = tl.load(bos_offsets_ptr + offset_id * 2 + 0).to(tl.int64)
    bos_t0 = tl.load(bos_offsets_ptr + offset_id * 2 + 1).to(tl.int64)

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim
    w0, w1, w2, w3 = _load_weight4_bf16(weight_ptr, d, d_mask)

    dim_i64 = tl.full((), dim, tl.int64)
    seqlen_i64 = tl.full((), seqlen, tl.int64)
    valid_base = b * seqlen_i64

    for k in tl.static_range(0, 6):
        t = bos_t0 + k - 3
        row_valid = (t >= 0) & (t < seqlen)
        base = (b * seqlen_i64 + t) * dim_i64 + d.to(tl.int64)
        valid_code = tl.load(
            dx_valid_ptr + valid_base + t,
            mask=row_valid,
            other=0,
        ).to(tl.int32)
        valid1 = (valid_code & 1) != 0
        valid2 = (valid_code & 2) != 0
        valid3 = (valid_code & 4) != 0
        g0 = tl.load(g_ptr + base, mask=d_mask & row_valid, other=0.0)
        g1 = tl.load(g_ptr + base + dim_i64, mask=d_mask & valid1, other=0.0)
        g2 = tl.load(g_ptr + base + 2 * dim_i64, mask=d_mask & valid2, other=0.0)
        g3 = tl.load(g_ptr + base + 3 * dim_i64, mask=d_mask & valid3, other=0.0)
        dx = g0 * w3 + g1 * w2 + g2 * w1 + g3 * w0
        tl.store(dx_ptr + base, dx, mask=d_mask & row_valid)


@triton.jit
def _compute_dx_t8_kernel(
    g_ptr,
    weight_ptr,
    dx_valid_ptr,
    dx_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    block_d: tl.constexpr,
):
    pid_d = tl.program_id(0)
    b = tl.program_id(1)
    tile_t = tl.program_id(2)

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim
    w0, w1, w2, w3 = _load_weight4_bf16(weight_ptr, d, d_mask)

    b_i64 = b.to(tl.int64)
    dim_i64 = tl.full((), dim, tl.int64)
    seqlen_i64 = tl.full((), seqlen, tl.int64)
    valid_base = b_i64 * seqlen_i64
    tile_start = tile_t * 16
    tile_base = (b_i64 * seqlen_i64 + tile_start) * dim_i64 + d.to(tl.int64)
    valid_tile_base = valid_base + tile_start

    for i in tl.static_range(0, 16):
        t = tile_t * 16 + i
        row_valid = t < seqlen
        base = tile_base + i * dim_i64

        valid_code = tl.load(
            dx_valid_ptr + valid_tile_base + i,
            mask=row_valid,
            other=0,
        ).to(tl.int32)
        valid1 = (valid_code & 1) != 0
        valid2 = (valid_code & 2) != 0
        valid3 = (valid_code & 4) != 0

        g0 = tl.load(g_ptr + base, mask=d_mask & row_valid, other=0.0)
        g1 = tl.load(g_ptr + base + dim_i64, mask=d_mask & valid1, other=0.0)
        g2 = tl.load(g_ptr + base + 2 * dim_i64, mask=d_mask & valid2, other=0.0)
        g3 = tl.load(g_ptr + base + 3 * dim_i64, mask=d_mask & valid3, other=0.0)
        dx = g0 * w3 + g1 * w2 + g2 * w1 + g3 * w0

        tl.store(dx_ptr + base, dx, mask=d_mask & row_valid)


@triton.jit
def _compute_dx_boundary_t3_kernel(
    g_ptr,
    weight_ptr,
    dx_valid_ptr,
    dx_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    block_d: tl.constexpr,
):
    pid_d = tl.program_id(0)
    b = tl.program_id(1)
    tile_t = tl.program_id(2)

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim
    w0, w1, w2, w3 = _load_weight4_bf16(weight_ptr, d, d_mask)

    b_i64 = b.to(tl.int64)
    dim_i64 = tl.full((), dim, tl.int64)
    seqlen_i64 = tl.full((), seqlen, tl.int64)
    valid_base = b_i64 * seqlen_i64
    tile_start = tile_t * 16
    tile_base = (b_i64 * seqlen_i64 + tile_start) * dim_i64 + d.to(tl.int64)
    valid_tile_base = valid_base + tile_start

    for i in tl.static_range(13, 16):
        t = tile_t * 16 + i
        row_valid = t < seqlen
        base = tile_base + i * dim_i64
        valid_code = tl.load(
            dx_valid_ptr + valid_tile_base + i,
            mask=row_valid,
            other=0,
        ).to(tl.int32)
        valid1 = (valid_code & 1) != 0
        valid2 = (valid_code & 2) != 0
        valid3 = (valid_code & 4) != 0

        g0 = tl.load(g_ptr + base, mask=d_mask & row_valid, other=0.0)
        g1 = tl.load(g_ptr + base + dim_i64, mask=d_mask & valid1, other=0.0)
        g2 = tl.load(g_ptr + base + 2 * dim_i64, mask=d_mask & valid2, other=0.0)
        g3 = tl.load(g_ptr + base + 3 * dim_i64, mask=d_mask & valid3, other=0.0)
        dx = g0 * w3 + g1 * w2 + g2 * w1 + g3 * w0

        tl.store(dx_ptr + base, dx, mask=d_mask & row_valid)


@triton.jit
def _compute_dx_dirty_boundary_repair_t8_kernel(
    g_ptr,
    weight_ptr,
    dx_valid_ptr,
    dirty_idx_ptr,
    dirty_meta_ptr,
    dx_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    block_d: tl.constexpr,
    tiles_t: tl.constexpr,
):
    pid_d = tl.program_id(0)
    repair_id = tl.program_id(1)

    b = tl.load(dirty_idx_ptr + repair_id * 2 + 0).to(tl.int64)
    tile_t = tl.load(dirty_idx_ptr + repair_id * 2 + 1).to(tl.int64)
    meta = tl.load(dirty_meta_ptr + b * tiles_t + tile_t).to(tl.int64)
    dirty_bits = (meta & 0xFFFF).to(tl.int32)

    prev_meta = tl.load(
        dirty_meta_ptr + b * tiles_t + tile_t - 1,
        mask=tile_t > 0,
        other=0,
    ).to(tl.int64)
    prev_dirty_bits = (prev_meta & 0xFFFF).to(tl.int32)

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim
    w0, w1, w2, w3 = _load_weight4_bf16(weight_ptr, d, d_mask)

    dim_i64 = tl.full((), dim, tl.int64)
    seqlen_i64 = tl.full((), seqlen, tl.int64)
    valid_base = b * seqlen_i64
    tile_start = tile_t * 16
    tile_base = (b * seqlen_i64 + tile_start) * dim_i64 + d.to(tl.int64)
    valid_tile_base = valid_base + tile_start

    for i in tl.static_range(13, 16):
        if i == 13:
            need = (dirty_bits & 0xE000) != 0
        elif i == 14:
            need = (dirty_bits & 0xC000) != 0
        else:
            need = (dirty_bits & 0x8000) != 0

        if need:
            t = tile_t * 16 + i
            row_valid = t < seqlen
            base = tile_base + i * dim_i64
            valid_code = tl.load(
                dx_valid_ptr + valid_tile_base + i,
                mask=row_valid,
                other=0,
            ).to(tl.int32)
            valid1 = (valid_code & 1) != 0
            valid2 = (valid_code & 2) != 0
            valid3 = (valid_code & 4) != 0

            g0 = tl.load(g_ptr + base, mask=d_mask & row_valid, other=0.0)
            g1 = tl.load(g_ptr + base + dim_i64, mask=d_mask & valid1, other=0.0)
            g2 = tl.load(g_ptr + base + 2 * dim_i64, mask=d_mask & valid2, other=0.0)
            g3 = tl.load(g_ptr + base + 3 * dim_i64, mask=d_mask & valid3, other=0.0)
            dx = g0 * w3 + g1 * w2 + g2 * w1 + g3 * w0

            tl.store(dx_ptr + base, dx, mask=d_mask & row_valid)

    if tile_t > 0:
        prev_tile_base = tile_base - 16 * dim_i64
        prev_valid_tile_base = valid_tile_base - 16

        for i in tl.static_range(13, 16):
            if i == 13:
                need = ((dirty_bits & 0x1) != 0) & ((prev_dirty_bits & 0xE000) == 0)
            elif i == 14:
                need = ((dirty_bits & 0x3) != 0) & ((prev_dirty_bits & 0xC000) == 0)
            else:
                need = ((dirty_bits & 0x7) != 0) & ((prev_dirty_bits & 0x8000) == 0)

            if need:
                t = (tile_t - 1) * 16 + i
                row_valid = t < seqlen
                base = prev_tile_base + i * dim_i64
                valid_code = tl.load(
                    dx_valid_ptr + prev_valid_tile_base + i,
                    mask=row_valid,
                    other=0,
                ).to(tl.int32)
                valid1 = (valid_code & 1) != 0
                valid2 = (valid_code & 2) != 0
                valid3 = (valid_code & 4) != 0

                g0 = tl.load(g_ptr + base, mask=d_mask & row_valid, other=0.0)
                g1 = tl.load(g_ptr + base + dim_i64, mask=d_mask & valid1, other=0.0)
                g2 = tl.load(
                    g_ptr + base + 2 * dim_i64,
                    mask=d_mask & valid2,
                    other=0.0,
                )
                g3 = tl.load(
                    g_ptr + base + 3 * dim_i64,
                    mask=d_mask & valid3,
                    other=0.0,
                )
                dx = g0 * w3 + g1 * w2 + g2 * w1 + g3 * w0

                tl.store(dx_ptr + base, dx, mask=d_mask & row_valid)


@triton.jit
def _compute_dx_dirty_tile_repair_t8_kernel(
    g_ptr,
    weight_ptr,
    dx_valid_ptr,
    dirty_idx_ptr,
    dirty_meta_ptr,
    dx_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    block_d: tl.constexpr,
    tiles_t: tl.constexpr,
):
    pid_d = tl.program_id(0)
    repair_id = tl.program_id(1)

    b = tl.load(dirty_idx_ptr + repair_id * 2 + 0).to(tl.int64)
    tile_t = tl.load(dirty_idx_ptr + repair_id * 2 + 1).to(tl.int64)
    meta = tl.load(dirty_meta_ptr + b * tiles_t + tile_t).to(tl.int64)
    dirty_bits = (meta & 0xFFFF).to(tl.int32)
    repair_bits = (
        dirty_bits
        | (dirty_bits >> 1)
        | (dirty_bits >> 2)
        | (dirty_bits >> 3)
    ) & 0x1FFF

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim
    w0, w1, w2, w3 = _load_weight4_bf16(weight_ptr, d, d_mask)

    dim_i64 = tl.full((), dim, tl.int64)
    seqlen_i64 = tl.full((), seqlen, tl.int64)
    valid_base = b * seqlen_i64
    tile_start = tile_t * 16
    tile_base = (b * seqlen_i64 + tile_start) * dim_i64 + d.to(tl.int64)
    valid_tile_base = valid_base + tile_start

    for i in tl.static_range(0, 13):
        if (repair_bits & (1 << i)) != 0:
            t = tile_t * 16 + i
            row_valid = t < seqlen
            base = tile_base + i * dim_i64
            valid_code = tl.load(
                dx_valid_ptr + valid_tile_base + i,
                mask=row_valid,
                other=0,
            ).to(tl.int32)
            valid1 = (valid_code & 1) != 0
            valid2 = (valid_code & 2) != 0
            valid3 = (valid_code & 4) != 0

            g0 = tl.load(g_ptr + base, mask=d_mask & row_valid, other=0.0)
            g1 = tl.load(g_ptr + base + dim_i64, mask=d_mask & valid1, other=0.0)
            g2 = tl.load(g_ptr + base + 2 * dim_i64, mask=d_mask & valid2, other=0.0)
            g3 = tl.load(g_ptr + base + 3 * dim_i64, mask=d_mask & valid3, other=0.0)
            dx = g0 * w3 + g1 * w2 + g2 * w1 + g3 * w0

            tl.store(dx_ptr + base, dx, mask=d_mask & row_valid)


@triton.jit
def _dinitial_kernel(
    g_ptr,
    weight_ptr,
    bos_ptr,
    dx_valid_ptr,
    dfinal_ptr,
    dx_ptr,
    dinitial_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    block_d: tl.constexpr,
):
    pid_d = tl.program_id(0)
    b = tl.program_id(1)

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim
    b_i64 = b.to(tl.int64)
    dim_i64 = tl.full((), dim, tl.int64)
    seqlen_i64 = tl.full((), seqlen, tl.int64)
    row0 = b_i64 * seqlen_i64 * dim_i64 + d.to(tl.int64)
    row1 = row0 + dim_i64
    row2 = row1 + dim_i64
    bos_base = b_i64 * seqlen_i64

    bos0 = tl.load(bos_ptr + bos_base + 0)
    bos1 = tl.load(bos_ptr + bos_base + 1)
    bos2 = tl.load(bos_ptr + bos_base + 2)
    valid0 = ~bos0
    valid1 = valid0 & (~bos1)
    valid2 = valid1 & (~bos2)

    w_base = d * 4
    w0 = tl.load(weight_ptr + w_base + 0, mask=d_mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight_ptr + w_base + 1, mask=d_mask, other=0.0).to(tl.float32)
    w2 = tl.load(weight_ptr + w_base + 2, mask=d_mask, other=0.0).to(tl.float32)
    w3 = tl.load(weight_ptr + w_base + 3, mask=d_mask, other=0.0).to(tl.float32)

    g0 = tl.load(g_ptr + row0, mask=d_mask, other=0.0)
    g1 = tl.load(g_ptr + row1, mask=d_mask, other=0.0)
    g2 = tl.load(g_ptr + row2, mask=d_mask, other=0.0)

    di0 = tl.where(valid0, g0 * w0, 0.0)
    di1 = tl.where(valid0, g0 * w1, 0.0) + tl.where(valid1, g1 * w0, 0.0)
    di2 = (
        tl.where(valid0, g0 * w2, 0.0)
        + tl.where(valid1, g1 * w1, 0.0)
        + tl.where(valid2, g2 * w0, 0.0)
    )

    out_base = b_i64 * dim_i64 * 3 + d.to(tl.int64) * 3
    tl.store(dinitial_ptr + out_base + 0, di0, mask=d_mask)
    tl.store(dinitial_ptr + out_base + 1, di1, mask=d_mask)
    tl.store(dinitial_ptr + out_base + 2, di2, mask=d_mask)

    valid_base = b_i64 * seqlen_i64
    dfinal_base = b_i64 * dim_i64 * 3 + d.to(tl.int64) * 3

    t0 = seqlen - 3
    tail0 = (b_i64 * seqlen_i64 + t0) * dim_i64 + d.to(tl.int64)
    valid_code0 = tl.load(dx_valid_ptr + valid_base + t0).to(tl.int32)
    valid1_0 = (valid_code0 & 1) != 0
    valid2_0 = (valid_code0 & 2) != 0
    tg0_0 = tl.load(g_ptr + tail0, mask=d_mask, other=0.0)
    tg1_0 = tl.load(g_ptr + tail0 + dim_i64, mask=d_mask & valid1_0, other=0.0)
    tg2_0 = tl.load(g_ptr + tail0 + 2 * dim_i64, mask=d_mask & valid2_0, other=0.0)
    df0 = tl.load(dfinal_ptr + dfinal_base + 0, mask=d_mask, other=0.0).to(
        tl.float32
    )
    dx0 = tg0_0 * w3 + tg1_0 * w2 + tg2_0 * w1 + tl.where(valid2_0, df0, 0.0)
    tl.store(dx_ptr + tail0, dx0, mask=d_mask)

    t1 = seqlen - 2
    tail1 = tail0 + dim_i64
    valid_code1 = tl.load(dx_valid_ptr + valid_base + t1).to(tl.int32)
    valid1_1 = (valid_code1 & 1) != 0
    tg0_1 = tl.load(g_ptr + tail1, mask=d_mask, other=0.0)
    tg1_1 = tl.load(g_ptr + tail1 + dim_i64, mask=d_mask & valid1_1, other=0.0)
    df1 = tl.load(dfinal_ptr + dfinal_base + 1, mask=d_mask, other=0.0).to(
        tl.float32
    )
    dx1 = tg0_1 * w3 + tg1_1 * w2 + tl.where(valid1_1, df1, 0.0)
    tl.store(dx_ptr + tail1, dx1, mask=d_mask)

    tail2 = tail1 + dim_i64
    tg0_2 = tl.load(g_ptr + tail2, mask=d_mask, other=0.0)
    df2 = tl.load(dfinal_ptr + dfinal_base + 2, mask=d_mask, other=0.0).to(
        tl.float32
    )
    dx2 = tg0_2 * w3 + df2
    tl.store(dx_ptr + tail2, dx2, mask=d_mask)


@triton.jit
def _dw_dbias_stage1_kernel(
    g_ptr,
    x_ptr,
    stage1_clear_ptr,
    partial_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    chunks_per_batch: tl.constexpr,
    total_chunks: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    full_tile: tl.constexpr,
):
    pid_d = tl.program_id(0)
    chunk_id = tl.program_id(1)
    b = chunk_id // chunks_per_batch
    chunk = chunk_id - b * chunks_per_batch

    t = chunk * block_n + tl.arange(0, block_n)
    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    t_mat = t[None, :]
    d_mat = d[:, None]
    base = (b * seqlen + t_mat) * dim + d_mat
    partial_base = chunk_id * dim + d
    plane_stride = total_chunks * dim

    clear_base = b * seqlen
    if full_tile:
        clear_code = tl.load(stage1_clear_ptr + clear_base + t).to(tl.int32)
    else:
        clear_code = tl.load(
            stage1_clear_ptr + clear_base + t,
            mask=t < seqlen,
            other=0,
        ).to(tl.int32)
    valid1 = (t >= 1) & ((clear_code & 1) != 0)
    valid2 = (t >= 2) & ((clear_code & 2) != 0)
    valid3 = (t >= 3) & ((clear_code & 4) != 0)

    if full_tile:
        g = tl.load(g_ptr + base)
        x0 = tl.load(x_ptr + base, eviction_policy="evict_last").to(tl.float32)
    else:
        mask = (t_mat < seqlen) & (d_mat < dim)
        g = tl.load(g_ptr + base, mask=mask, other=0.0)
        x0 = tl.load(
            x_ptr + base,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(
            tl.float32
        )
    sum_bias = tl.sum(g, axis=1)
    sum_w3 = tl.sum(g * x0, axis=1)
    if full_tile:
        tl.store(partial_ptr + partial_base + 0 * plane_stride, sum_bias)
        tl.store(partial_ptr + partial_base + 4 * plane_stride, sum_w3)
    else:
        tl.store(partial_ptr + partial_base + 0 * plane_stride, sum_bias, mask=d_mask)
        tl.store(partial_ptr + partial_base + 4 * plane_stride, sum_w3, mask=d_mask)

    if full_tile:
        src1_x = tl.load(
            x_ptr + base - dim,
            mask=valid1[None, :],
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
    else:
        src1_x = tl.load(
            x_ptr + base - dim,
            mask=mask & valid1[None, :],
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
    sum_w2 = tl.sum(g * src1_x, axis=1)
    if full_tile:
        tl.store(partial_ptr + partial_base + 3 * plane_stride, sum_w2)
    else:
        tl.store(partial_ptr + partial_base + 3 * plane_stride, sum_w2, mask=d_mask)
    if full_tile:
        src2_x = tl.load(
            x_ptr + base - 2 * dim,
            mask=valid2[None, :],
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
    else:
        src2_x = tl.load(
            x_ptr + base - 2 * dim,
            mask=mask & valid2[None, :],
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
    sum_w1 = tl.sum(g * src2_x, axis=1)
    if full_tile:
        tl.store(partial_ptr + partial_base + 2 * plane_stride, sum_w1)
    else:
        tl.store(partial_ptr + partial_base + 2 * plane_stride, sum_w1, mask=d_mask)
    if full_tile:
        src3_x = tl.load(
            x_ptr + base - 3 * dim,
            mask=valid3[None, :],
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
    else:
        src3_x = tl.load(
            x_ptr + base - 3 * dim,
            mask=mask & valid3[None, :],
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
    sum_w0 = tl.sum(g * src3_x, axis=1)
    if full_tile:
        tl.store(partial_ptr + partial_base + 1 * plane_stride, sum_w0)
    else:
        tl.store(partial_ptr + partial_base + 1 * plane_stride, sum_w0, mask=d_mask)


@triton.jit
def _dw_dbias_stage1_rolling_kernel(
    g_ptr,
    x_ptr,
    stage1_clear_ptr,
    partial_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    chunks_per_batch: tl.constexpr,
    total_chunks: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    full_tile: tl.constexpr,
):
    pid_d = tl.program_id(0)
    chunk_id = tl.program_id(1)
    b = chunk_id // chunks_per_batch
    chunk = chunk_id - b * chunks_per_batch

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim
    sum_bias = tl.zeros((block_d,), tl.float32)
    sum_w0 = tl.zeros((block_d,), tl.float32)
    sum_w1 = tl.zeros((block_d,), tl.float32)
    sum_w2 = tl.zeros((block_d,), tl.float32)
    sum_w3 = tl.zeros((block_d,), tl.float32)

    dim_i64 = tl.full((), dim, tl.int64)
    seqlen_i64 = tl.full((), seqlen, tl.int64)
    b_i64 = b.to(tl.int64)
    chunk_start = chunk * block_n
    chunk_start_i64 = chunk_start.to(tl.int64)
    row_base = (b_i64 * seqlen_i64 + chunk_start_i64) * dim_i64 + d.to(tl.int64)
    clear_base = b * seqlen

    if full_tile:
        prev1 = tl.load(
            x_ptr + row_base - dim_i64,
            mask=chunk_start >= 1,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        prev2 = tl.load(
            x_ptr + row_base - 2 * dim_i64,
            mask=chunk_start >= 2,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        prev3 = tl.load(
            x_ptr + row_base - 3 * dim_i64,
            mask=chunk_start >= 3,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        first_clear_code = tl.load(stage1_clear_ptr + clear_base + chunk_start).to(
            tl.int32
        )
        prev1 = tl.where(
            (chunk_start >= 1) & ((first_clear_code & 1) != 0), prev1, 0.0
        )
        prev2 = tl.where(
            (chunk_start >= 2) & ((first_clear_code & 2) != 0), prev2, 0.0
        )
        prev3 = tl.where(
            (chunk_start >= 3) & ((first_clear_code & 4) != 0), prev3, 0.0
        )

        for i in tl.range(0, block_n, 1, loop_unroll_factor=1):
            t = chunk_start + i
            base = row_base + i * dim_i64
            clear_code = tl.load(stage1_clear_ptr + clear_base + t).to(tl.int32)
            not_bos = (clear_code & 1) != 0

            g = tl.load(g_ptr + base).to(tl.float32)
            x0 = tl.load(x_ptr + base, eviction_policy="evict_last").to(tl.float32)
            src1 = tl.where(not_bos, prev1, 0.0)
            src2 = tl.where(not_bos, prev2, 0.0)
            src3 = tl.where(not_bos, prev3, 0.0)
            sum_bias += g
            sum_w3 += g * x0
            sum_w2 += g * src1
            sum_w1 += g * src2
            sum_w0 += g * src3
            prev3 = src2
            prev2 = src1
            prev1 = x0
    else:
        prev1 = tl.load(
            x_ptr + row_base - dim_i64,
            mask=d_mask & (chunk_start >= 1),
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        prev2 = tl.load(
            x_ptr + row_base - 2 * dim_i64,
            mask=d_mask & (chunk_start >= 2),
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        prev3 = tl.load(
            x_ptr + row_base - 3 * dim_i64,
            mask=d_mask & (chunk_start >= 3),
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        first_clear_code = tl.load(
            stage1_clear_ptr + clear_base + chunk_start,
            mask=chunk_start < seqlen,
            other=0,
        ).to(tl.int32)
        prev1 = tl.where(
            (chunk_start >= 1) & ((first_clear_code & 1) != 0), prev1, 0.0
        )
        prev2 = tl.where(
            (chunk_start >= 2) & ((first_clear_code & 2) != 0), prev2, 0.0
        )
        prev3 = tl.where(
            (chunk_start >= 3) & ((first_clear_code & 4) != 0), prev3, 0.0
        )

        for i in tl.range(0, block_n, 1, loop_unroll_factor=1):
            t = chunk_start + i
            row_valid = t < seqlen
            base = row_base + i * dim_i64
            mask = d_mask & row_valid
            clear_code = tl.load(
                stage1_clear_ptr + clear_base + t,
                mask=row_valid,
                other=0,
            ).to(tl.int32)
            not_bos = (clear_code & 1) != 0

            g = tl.load(g_ptr + base, mask=mask, other=0.0).to(tl.float32)
            x0 = tl.load(
                x_ptr + base,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            src1 = tl.where(row_valid & not_bos, prev1, 0.0)
            src2 = tl.where(row_valid & not_bos, prev2, 0.0)
            src3 = tl.where(row_valid & not_bos, prev3, 0.0)
            sum_bias += g
            sum_w3 += g * x0
            sum_w2 += g * src1
            sum_w1 += g * src2
            sum_w0 += g * src3
            prev3 = src2
            prev2 = src1
            prev1 = x0

    partial_base = chunk_id * dim + d
    plane_stride = total_chunks * dim
    if full_tile:
        tl.store(partial_ptr + partial_base + 0 * plane_stride, sum_bias)
        tl.store(partial_ptr + partial_base + 1 * plane_stride, sum_w0)
        tl.store(partial_ptr + partial_base + 2 * plane_stride, sum_w1)
        tl.store(partial_ptr + partial_base + 3 * plane_stride, sum_w2)
        tl.store(partial_ptr + partial_base + 4 * plane_stride, sum_w3)
    else:
        tl.store(partial_ptr + partial_base + 0 * plane_stride, sum_bias, mask=d_mask)
        tl.store(partial_ptr + partial_base + 1 * plane_stride, sum_w0, mask=d_mask)
        tl.store(partial_ptr + partial_base + 2 * plane_stride, sum_w1, mask=d_mask)
        tl.store(partial_ptr + partial_base + 3 * plane_stride, sum_w2, mask=d_mask)
        tl.store(partial_ptr + partial_base + 4 * plane_stride, sum_w3, mask=d_mask)


@triton.jit
def _dw_initial_correction_kernel(
    g_ptr,
    initial_ptr,
    bos_ptr,
    partial_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    chunks_per_batch: tl.constexpr,
    total_chunks: tl.constexpr,
    block_d: tl.constexpr,
):
    pid_d = tl.program_id(0)
    b = tl.program_id(1)

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    init_base = b * dim * 3 + d * 3
    init0 = tl.load(initial_ptr + init_base + 0, mask=d_mask, other=0.0).to(
        tl.float32
    )
    init1 = tl.load(initial_ptr + init_base + 1, mask=d_mask, other=0.0).to(
        tl.float32
    )
    init2 = tl.load(initial_ptr + init_base + 2, mask=d_mask, other=0.0).to(
        tl.float32
    )

    row0 = (b * seqlen + 0) * dim + d
    row1 = row0 + dim
    row2 = row1 + dim
    g0 = tl.load(g_ptr + row0, mask=d_mask & (seqlen >= 1), other=0.0).to(
        tl.float32
    )
    g1 = tl.load(g_ptr + row1, mask=d_mask & (seqlen >= 2), other=0.0).to(
        tl.float32
    )
    g2 = tl.load(g_ptr + row2, mask=d_mask & (seqlen >= 3), other=0.0).to(
        tl.float32
    )

    bos_base = b * seqlen
    bos0 = tl.load(bos_ptr + bos_base + 0, mask=seqlen >= 1, other=True)
    bos1 = tl.load(bos_ptr + bos_base + 1, mask=seqlen >= 2, other=True)
    bos2 = tl.load(bos_ptr + bos_base + 2, mask=seqlen >= 3, other=True)
    clear0 = (seqlen >= 1) & (~bos0)
    clear1 = (seqlen >= 2) & clear0 & (~bos1)
    clear2 = (seqlen >= 3) & clear1 & (~bos2)

    corr_w0 = tl.where(clear0, g0 * init0, 0.0)
    corr_w0 += tl.where(clear1, g1 * init1, 0.0)
    corr_w0 += tl.where(clear2, g2 * init2, 0.0)
    corr_w1 = tl.where(clear0, g0 * init1, 0.0)
    corr_w1 += tl.where(clear1, g1 * init2, 0.0)
    corr_w2 = tl.where(clear0, g0 * init2, 0.0)

    chunk_id = b * chunks_per_batch
    partial_base = chunk_id * dim + d
    plane_stride = total_chunks * dim
    w0_ptr = partial_ptr + partial_base + 1 * plane_stride
    w1_ptr = partial_ptr + partial_base + 2 * plane_stride
    w2_ptr = partial_ptr + partial_base + 3 * plane_stride
    tl.store(w0_ptr, tl.load(w0_ptr, mask=d_mask, other=0.0) + corr_w0, mask=d_mask)
    tl.store(w1_ptr, tl.load(w1_ptr, mask=d_mask, other=0.0) + corr_w1, mask=d_mask)
    tl.store(w2_ptr, tl.load(w2_ptr, mask=d_mask, other=0.0) + corr_w2, mask=d_mask)


@triton.jit
def _dw_dbias_final_kernel(
    partial_ptr,
    dweight_ptr,
    dbias_ptr,
    dim: tl.constexpr,
    total_chunks: tl.constexpr,
    block_chunks: tl.constexpr,
    block_d: tl.constexpr,
    full_d_tile: tl.constexpr,
):
    pid_d = tl.program_id(0)
    chunks = tl.arange(0, block_chunks)
    d = pid_d * block_d + tl.arange(0, block_d)
    c_mat = chunks[:, None]
    d_mat = d[None, :]
    if full_d_tile:
        mask = c_mat < total_chunks
    else:
        mask = (c_mat < total_chunks) & (d_mat < dim)
    base = c_mat * dim + d_mat
    plane_stride = total_chunks * dim
    d_mask = d < dim

    db = tl.sum(
        tl.load(partial_ptr + base + 0 * plane_stride, mask=mask, other=0.0),
        axis=0,
    )
    if full_d_tile:
        tl.store(dbias_ptr + d, db)
    else:
        tl.store(dbias_ptr + d, db, mask=d_mask)

    dw0 = tl.sum(
        tl.load(partial_ptr + base + 1 * plane_stride, mask=mask, other=0.0),
        axis=0,
    )
    if full_d_tile:
        tl.store(dweight_ptr + d * 4 + 0, dw0)
    else:
        tl.store(dweight_ptr + d * 4 + 0, dw0, mask=d_mask)

    dw1 = tl.sum(
        tl.load(partial_ptr + base + 2 * plane_stride, mask=mask, other=0.0),
        axis=0,
    )
    if full_d_tile:
        tl.store(dweight_ptr + d * 4 + 1, dw1)
    else:
        tl.store(dweight_ptr + d * 4 + 1, dw1, mask=d_mask)

    dw2 = tl.sum(
        tl.load(partial_ptr + base + 3 * plane_stride, mask=mask, other=0.0),
        axis=0,
    )
    if full_d_tile:
        tl.store(dweight_ptr + d * 4 + 2, dw2)
    else:
        tl.store(dweight_ptr + d * 4 + 2, dw2, mask=d_mask)

    dw3 = tl.sum(
        tl.load(partial_ptr + base + 4 * plane_stride, mask=mask, other=0.0),
        axis=0,
    )

    if full_d_tile:
        tl.store(dweight_ptr + d * 4 + 3, dw3)
    else:
        tl.store(dweight_ptr + d * 4 + 3, dw3, mask=d_mask)


@triton.jit
def _dw_dbias_final_plane_kernel(
    partial_ptr,
    dweight_ptr,
    dbias_ptr,
    dim: tl.constexpr,
    total_chunks: tl.constexpr,
    block_chunks: tl.constexpr,
    block_d: tl.constexpr,
    full_d_tile: tl.constexpr,
):
    pid_d = tl.program_id(0)
    plane = tl.program_id(1)
    chunks = tl.arange(0, block_chunks)
    d = pid_d * block_d + tl.arange(0, block_d)
    c_mat = chunks[:, None]
    d_mat = d[None, :]
    if full_d_tile:
        mask = c_mat < total_chunks
    else:
        mask = (c_mat < total_chunks) & (d_mat < dim)
    plane_stride = total_chunks * dim
    vals = tl.load(
        partial_ptr + c_mat * dim + d_mat + plane * plane_stride,
        mask=mask,
        other=0.0,
    )
    acc = tl.sum(vals, axis=0)

    d_mask = d < dim
    if full_d_tile:
        tl.store(dbias_ptr + d, acc, mask=plane == 0)
        w_offset = plane - 1
        tl.store(dweight_ptr + d * 4 + w_offset, acc, mask=plane != 0)
    else:
        tl.store(dbias_ptr + d, acc, mask=d_mask & (plane == 0))
        w_offset = plane - 1
        tl.store(dweight_ptr + d * 4 + w_offset, acc, mask=d_mask & (plane != 0))


@triton.jit
def _fused_chunk_bwd_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    initial_ptr,
    bos_ptr,
    dout_ptr,
    dfinal_ptr,
    dx_ptr,
    dinitial_ptr,
    partial_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    chunks_per_batch: tl.constexpr,
    total_chunks: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    packed_xdout: tl.constexpr,
    zero_state: tl.constexpr,
):
    pid_d = tl.program_id(0)
    chunk_id = tl.program_id(1)

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim
    w0, w1, w2, w3 = _load_weight4_bf16(weight_ptr, d, d_mask)
    bias = _load_bias_bf16_u16(bias_ptr, d, d_mask)

    dim_i64 = tl.full((), dim, tl.int64)
    seqlen_i64 = tl.full((), seqlen, tl.int64)
    if zero_state:
        chunk = chunk_id % chunks_per_batch
        chunk_start = chunk * block_n
        chunk_start_i64 = chunk_start.to(tl.int64)
        token_start_i64 = chunk_id.to(tl.int64) * block_n
        row_base = token_start_i64 * dim_i64 + d.to(tl.int64)
        bos_base = token_start_i64 - chunk_start_i64
        init_base = d.to(tl.int64) * 3
    else:
        b = chunk_id // chunks_per_batch
        chunk = chunk_id - b * chunks_per_batch
        b_i64 = b.to(tl.int64)
        chunk_start = chunk * block_n
        chunk_start_i64 = chunk_start.to(tl.int64)
        row_base = (b_i64 * seqlen_i64 + chunk_start_i64) * dim_i64 + d.to(tl.int64)
        bos_base = b_i64 * seqlen_i64
        init_base = b_i64 * dim_i64 * 3 + d.to(tl.int64) * 3

    is_first_chunk = chunk_start == 0
    if packed_xdout and zero_state:
        bos_chunk_ptr = bos_ptr + bos_base + chunk_start_i64
        bos_shared = _stage_bos_chunk_shared_scalar(
            bos_chunk_ptr,
            is_first_chunk.to(tl.int32),
        )

    if zero_state:
        init0 = tl.zeros((block_d,), tl.float32)
        init1 = tl.zeros((block_d,), tl.float32)
        init2 = tl.zeros((block_d,), tl.float32)
    else:
        init_mask = d_mask & is_first_chunk
        init0 = tl.load(initial_ptr + init_base + 0, mask=init_mask, other=0.0).to(
            tl.float32
        )
        init1 = tl.load(initial_ptr + init_base + 1, mask=init_mask, other=0.0).to(
            tl.float32
        )
        init2 = tl.load(initial_ptr + init_base + 2, mask=init_mask, other=0.0).to(
            tl.float32
        )

    if packed_xdout:
        prev1_packed = tl.load(
            x_ptr.to(tl.pointer_type(tl.uint32)) + row_base - dim_i64,
            mask=d_mask & (~is_first_chunk),
            other=0,
            eviction_policy="evict_last",
        )
        prev2_packed = tl.load(
            x_ptr.to(tl.pointer_type(tl.uint32)) + row_base - 2 * dim_i64,
            mask=d_mask & (~is_first_chunk),
            other=0,
            eviction_policy="evict_last",
        )
        prev3_packed = tl.load(
            x_ptr.to(tl.pointer_type(tl.uint32)) + row_base - 3 * dim_i64,
            mask=d_mask & (~is_first_chunk),
            other=0,
            eviction_policy="evict_last",
        )
        prev1_x = _unpack_x_u32(prev1_packed)
        prev2_x = _unpack_x_u32(prev2_packed)
        prev3_x = _unpack_x_u32(prev3_packed)
    else:
        prev1_x = tl.load(
            x_ptr + row_base - dim_i64,
            mask=d_mask & (~is_first_chunk),
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        prev2_x = tl.load(
            x_ptr + row_base - 2 * dim_i64,
            mask=d_mask & (~is_first_chunk),
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        prev3_x = tl.load(
            x_ptr + row_base - 3 * dim_i64,
            mask=d_mask & (~is_first_chunk),
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
    if packed_xdout and zero_state:
        prev_bos_pair = _load_bos_pair_shared_absolute(bos_shared + 14)
        prev_bos2 = (prev_bos_pair & 0x1) != 0
        prev_bos1 = (prev_bos_pair & 0x100) != 0
    else:
        prev_bos1 = tl.load(
            bos_ptr + bos_base + chunk_start_i64 - 1,
            mask=~is_first_chunk,
            other=False,
        )
        prev_bos2 = tl.load(
            bos_ptr + bos_base + chunk_start_i64 - 2,
            mask=~is_first_chunk,
            other=False,
        )

    prev1 = tl.where(is_first_chunk, init2, prev1_x)
    prev2 = tl.where(
        is_first_chunk,
        init1,
        tl.where(~prev_bos1, prev2_x, 0.0),
    )
    prev3 = tl.where(
        is_first_chunk,
        init0,
        tl.where((~prev_bos1) & (~prev_bos2), prev3_x, 0.0),
    )

    keep_tm1 = ~prev_bos1
    keep_tm2 = ~prev_bos2
    g_m1 = tl.zeros((block_d,), tl.float32)
    g_m2 = tl.zeros((block_d,), tl.float32)
    g_m3 = tl.zeros((block_d,), tl.float32)

    first_g0 = tl.zeros((block_d,), tl.float32)
    first_g1 = tl.zeros((block_d,), tl.float32)
    first_g2 = tl.zeros((block_d,), tl.float32)
    sum_bias = tl.zeros((block_d,), tl.float32)
    sum_w0 = tl.zeros((block_d,), tl.float32)
    sum_w1 = tl.zeros((block_d,), tl.float32)
    sum_w2 = tl.zeros((block_d,), tl.float32)
    sum_w3 = tl.zeros((block_d,), tl.float32)

    if zero_state:
        df0 = tl.zeros((block_d,), tl.float32)
        df1 = tl.zeros((block_d,), tl.float32)
        df2 = tl.zeros((block_d,), tl.float32)
    else:
        is_last_chunk = chunk == (chunks_per_batch - 1)
        df_base = b_i64 * dim_i64 * 3 + d.to(tl.int64) * 3
        df_mask = d_mask & is_last_chunk
        df0 = tl.load(dfinal_ptr + df_base + 0, mask=df_mask, other=0.0).to(tl.float32)
        df1 = tl.load(dfinal_ptr + df_base + 1, mask=df_mask, other=0.0).to(tl.float32)
        df2 = tl.load(dfinal_ptr + df_base + 2, mask=df_mask, other=0.0).to(tl.float32)

    for i in tl.static_range(0, 3):
        t = chunk_start + i
        base = row_base + i * dim_i64

        if packed_xdout:
            packed = tl.load(
                x_ptr.to(tl.pointer_type(tl.uint32)) + base,
                mask=d_mask,
                other=0,
                eviction_policy="evict_last",
            )
            x0, dout = _unpack_xdout_u32(packed)
        else:
            x0 = tl.load(
                x_ptr + base,
                mask=d_mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            dout = tl.load(dout_ptr + base, mask=d_mask, other=0.0).to(
                tl.float32
            )
        bos_t = tl.load(bos_ptr + bos_base + t)
        keep_prev = ~bos_t
        src1 = tl.where(keep_prev, prev1, 0.0)
        src2 = tl.where(keep_prev, prev2, 0.0)
        src3 = tl.where(keep_prev, prev3, 0.0)

        z = _conv4_ffma_z(x0, src1, src2, src3, w0, w1, w2, w3, bias)
        g = _silu_backward_apply(z, dout)

        sum_bias = _ffma_one_acc(sum_bias, g)
        sum_w0 = _ffma_acc(sum_w0, g, src3)
        sum_w1 = _ffma_acc(sum_w1, g, src2)
        sum_w2 = _ffma_acc(sum_w2, g, src1)
        sum_w3 = _ffma_acc(sum_w3, g, x0)

        if i == 0:
            first_g0 = g
        elif i == 1:
            first_g1 = g
        else:
            first_g2 = g

        g_m3 = g_m2
        g_m2 = g_m1
        g_m1 = g
        prev3 = src2
        prev2 = src1
        prev1 = x0
        keep_tm2 = keep_tm1
        keep_tm1 = keep_prev

    if not zero_state:
        bos0 = tl.load(bos_ptr + bos_base + 0)
        bos1 = tl.load(bos_ptr + bos_base + 1)
        bos2 = tl.load(bos_ptr + bos_base + 2)
        valid0 = ~bos0
        valid1_init = valid0 & (~bos1)
        valid2_init = valid1_init & (~bos2)
        di0 = tl.where(valid0, first_g0 * w0, 0.0)
        di1 = tl.where(valid0, first_g0 * w1, 0.0) + tl.where(
            valid1_init, first_g1 * w0, 0.0
        )
        di2 = (
            tl.where(valid0, first_g0 * w2, 0.0)
            + tl.where(valid1_init, first_g1 * w1, 0.0)
            + tl.where(valid2_init, first_g2 * w0, 0.0)
        )
        out_base = b_i64 * dim_i64 * 3 + d.to(tl.int64) * 3
        tl.store(dinitial_ptr + out_base + 0, di0, mask=d_mask & is_first_chunk)
        tl.store(dinitial_ptr + out_base + 1, di1, mask=d_mask & is_first_chunk)
        tl.store(dinitial_ptr + out_base + 2, di2, mask=d_mask & is_first_chunk)

    if packed_xdout:
        main_packed = tl.load(
            x_ptr.to(tl.pointer_type(tl.uint32)) + row_base + 3 * dim_i64,
            mask=d_mask,
            other=0,
            volatile=True,
        )
        next_packed = tl.load(
            x_ptr.to(tl.pointer_type(tl.uint32)) + row_base + 4 * dim_i64,
            mask=d_mask,
            other=0,
            volatile=True,
        )
        main_x0, main_dout = _unpack_xdout_u32(main_packed)
        next_x0, next_dout = _unpack_xdout_u32(next_packed)
    else:
        main_x0 = tl.load(
            x_ptr + row_base + 3 * dim_i64,
            mask=d_mask,
            other=0.0,
            volatile=True,
        ).to(tl.float32)
        main_dout = tl.load(
            dout_ptr + row_base + 3 * dim_i64,
            mask=d_mask,
            other=0.0,
            volatile=True,
        ).to(tl.float32)
        next_x0 = tl.load(
            x_ptr + row_base + 4 * dim_i64,
            mask=d_mask,
            other=0.0,
            volatile=True,
        ).to(tl.float32)
        next_dout = tl.load(
            dout_ptr + row_base + 4 * dim_i64,
            mask=d_mask,
            other=0.0,
            volatile=True,
        ).to(tl.float32)

    i = 3
    t = chunk_start + i
    x0 = main_x0
    dout = main_dout
    bos_t = tl.load(bos_ptr + bos_base + t)
    keep_prev = ~bos_t

    src1 = tl.where(keep_prev, prev1, 0.0)
    src2 = tl.where(keep_prev, prev2, 0.0)
    src3 = tl.where(keep_prev, prev3, 0.0)

    z = _conv4_ffma_z(x0, src1, src2, src3, w0, w1, w2, w3, bias)
    g = _silu_backward_apply(z, dout)

    dx_t = t - 3
    valid1 = keep_tm2
    valid2 = valid1 & keep_tm1
    valid3 = valid2 & keep_prev
    dx = g_m3 * w3
    dx = _masked_ffma3_acc(
        dx,
        g_m2,
        w2,
        valid1,
        g_m1,
        w1,
        valid2,
        g,
        w0,
        valid3,
    )
    _store_bf16_trunc(
        dx_ptr,
        (bos_base + dx_t.to(tl.int64)) * dim_i64 + d.to(tl.int64),
        dx,
        d_mask,
    )

    sum_bias = _ffma_one_acc(sum_bias, g)
    sum_w0 = _ffma_acc(sum_w0, g, src3)
    sum_w1 = _ffma_acc(sum_w1, g, src2)
    sum_w2 = _ffma_acc(sum_w2, g, src1)
    sum_w3 = _ffma_acc(sum_w3, g, x0)

    g_m3 = g_m2
    g_m2 = g_m1
    g_m1 = g
    prev3 = src2
    prev2 = src1
    prev1 = x0
    keep_tm2 = keep_tm1
    keep_tm1 = keep_prev

    main_x0 = next_x0
    main_dout = next_dout

    if packed_xdout and zero_state:
        packed_lane = tl.arange(0, 32)
        packed_lane_d = pid_d * block_d + 2 * packed_lane
        packed_iter_addr = _opaque_packed_lane_address(
            x_ptr.to(tl.pointer_type(tl.uint32))
            + (token_start_i64 + 4) * dim_i64
            + packed_lane_d.to(tl.int64)
        )
        packed_row_bytes = tl.full((32,), dim * 4, tl.uint64)
        packed_quartet_bytes = tl.full((32,), 4 * dim * 4, tl.uint64)

    for i in tl.range(4, block_n - 4, 4, num_stages=1, loop_unroll_factor=1):
        t0 = chunk_start + i
        base0 = row_base + i * dim_i64

        x0 = main_x0
        dout0 = main_dout
        base1 = base0 + dim_i64
        base2 = base1 + dim_i64
        base3 = base2 + dim_i64
        if packed_xdout and zero_state:
            bos_addr = bos_shared + 16 + i
            bos_lo = _load_bos_pair_shared_absolute(bos_addr)
            bos_hi = _load_bos_pair_shared_absolute(bos_addr + 2)
        else:
            bos_lo = tl.load((bos_ptr + bos_base + t0).to(tl.pointer_type(tl.uint16))).to(tl.int32)
            bos_hi = tl.load((bos_ptr + bos_base + t0 + 2).to(tl.pointer_type(tl.uint16))).to(tl.int32)
        if packed_xdout and zero_state:
            packed1 = _load_packed_lane_at_byte_offset(
                packed_iter_addr,
                packed_row_bytes,
            )
            packed2 = _load_packed_lane_at_byte_offset(
                packed_iter_addr,
                2 * packed_row_bytes,
            )
            packed3 = _load_packed_lane_at_byte_offset(
                packed_iter_addr,
                3 * packed_row_bytes,
            )
        elif packed_xdout:
            packed1 = tl.load(
                x_ptr.to(tl.pointer_type(tl.uint32)) + base1,
                volatile=True,
            )
            packed2 = tl.load(
                x_ptr.to(tl.pointer_type(tl.uint32)) + base2,
                volatile=True,
            )
            packed3 = tl.load(
                x_ptr.to(tl.pointer_type(tl.uint32)) + base3,
                volatile=True,
            )
        else:
            x1 = tl.load(
                x_ptr + base1,
                mask=d_mask,
                other=0.0,
                volatile=True,
            ).to(tl.float32)
            dout1 = tl.load(
                dout_ptr + base1,
                mask=d_mask,
                other=0.0,
                volatile=True,
            ).to(tl.float32)
            x2 = tl.load(
                x_ptr + base2,
                mask=d_mask,
                other=0.0,
                volatile=True,
            ).to(tl.float32)
            dout2 = tl.load(
                dout_ptr + base2,
                mask=d_mask,
                other=0.0,
                volatile=True,
            ).to(tl.float32)
            x3 = tl.load(
                x_ptr + base3,
                mask=d_mask,
                other=0.0,
                volatile=True,
            ).to(tl.float32)
            dout3 = tl.load(
                dout_ptr + base3,
                mask=d_mask,
                other=0.0,
                volatile=True,
            ).to(tl.float32)
        next_base = base0 + 4 * dim_i64
        if packed_xdout and zero_state:
            _prefetch_l2_lane_at_byte_offset(
                packed_iter_addr,
                5 * packed_row_bytes,
                (packed_lane & 15) == 0,
            )
            main_packed = _load_packed_lane_at_byte_offset(
                packed_iter_addr,
                4 * packed_row_bytes,
            )
        elif packed_xdout:
            _prefetch_l2_if(
                x_ptr.to(tl.pointer_type(tl.uint32)) + base0 + 5 * dim_i64,
                (d & 7) == 0,
            )
            main_packed = tl.load(
                x_ptr.to(tl.pointer_type(tl.uint32)) + next_base,
                volatile=True,
            )
        else:
            main_x0 = tl.load(
                x_ptr + next_base,
                mask=d_mask,
                other=0.0,
                volatile=True,
            ).to(tl.float32)
            main_dout = tl.load(
                dout_ptr + next_base,
                mask=d_mask,
                other=0.0,
                volatile=True,
            ).to(tl.float32)

        if packed_xdout:
            x1, dout1 = _unpack_xdout_u32(packed1)
            x2, dout2 = _unpack_xdout_u32(packed2)
            x3, dout3 = _unpack_xdout_u32(packed3)
            main_x0, main_dout = _unpack_xdout_u32(main_packed)
        bos_t0 = (bos_lo & 0x1) != 0
        bos_t1 = (bos_lo & 0x100) != 0
        bos_t2 = (bos_hi & 0x1) != 0
        bos_t3 = (bos_hi & 0x100) != 0
        keep_prev0 = ~bos_t0

        src1 = tl.where(keep_prev0, prev1, 0.0)
        src2 = tl.where(keep_prev0, prev2, 0.0)
        src3 = tl.where(keep_prev0, prev3, 0.0)

        z = _conv4_ffma_z(x0, src1, src2, src3, w0, w1, w2, w3, bias)
        g = _silu_backward_apply(z, dout0)

        dx_t = t0 - 3
        valid1 = keep_tm2
        valid2 = valid1 & keep_tm1
        valid3 = valid2 & keep_prev0
        dx = g_m3 * w3
        dx = _masked_ffma3_acc(
            dx,
            g_m2,
            w2,
            valid1,
            g_m1,
            w1,
            valid2,
            g,
            w0,
            valid3,
        )
        _store_bf16_trunc(
            dx_ptr,
            (bos_base + dx_t.to(tl.int64)) * dim_i64 + d.to(tl.int64),
            dx,
            d_mask,
        )

        sum_bias = _ffma_one_acc(sum_bias, g)
        sum_w0 = _ffma_acc(sum_w0, g, src3)
        sum_w1 = _ffma_acc(sum_w1, g, src2)
        sum_w2 = _ffma_acc(sum_w2, g, src1)
        sum_w3 = _ffma_acc(sum_w3, g, x0)

        g_m3 = g_m2
        g_m2 = g_m1
        g_m1 = g
        prev3 = src2
        prev2 = src1
        prev1 = x0
        keep_tm2 = keep_tm1
        keep_tm1 = keep_prev0

        keep_prev1 = ~bos_t1
        src1 = tl.where(keep_prev1, prev1, 0.0)
        src2 = tl.where(keep_prev1, prev2, 0.0)
        src3 = tl.where(keep_prev1, prev3, 0.0)

        z = _conv4_ffma_z(x1, src1, src2, src3, w0, w1, w2, w3, bias)
        g = _silu_backward_apply(z, dout1)

        valid1 = keep_tm2
        valid2 = valid1 & keep_tm1
        valid3 = valid2 & keep_prev1
        dx = g_m3 * w3
        dx = _masked_ffma3_acc(
            dx,
            g_m2,
            w2,
            valid1,
            g_m1,
            w1,
            valid2,
            g,
            w0,
            valid3,
        )
        _store_bf16_trunc(
            dx_ptr,
            (bos_base + (dx_t + 1).to(tl.int64)) * dim_i64 + d.to(tl.int64),
            dx,
            d_mask,
        )

        sum_bias = _ffma_one_acc(sum_bias, g)
        sum_w0 = _ffma_acc(sum_w0, g, src3)
        sum_w1 = _ffma_acc(sum_w1, g, src2)
        sum_w2 = _ffma_acc(sum_w2, g, src1)
        sum_w3 = _ffma_acc(sum_w3, g, x1)

        g_m3 = g_m2
        g_m2 = g_m1
        g_m1 = g
        prev3 = src2
        prev2 = src1
        prev1 = x1
        keep_tm2 = keep_tm1
        keep_tm1 = keep_prev1

        keep_prev2 = ~bos_t2
        src1 = tl.where(keep_prev2, prev1, 0.0)
        src2 = tl.where(keep_prev2, prev2, 0.0)
        src3 = tl.where(keep_prev2, prev3, 0.0)

        z = _conv4_ffma_z(x2, src1, src2, src3, w0, w1, w2, w3, bias)
        g = _silu_backward_apply(z, dout2)

        valid1 = keep_tm2
        valid2 = valid1 & keep_tm1
        valid3 = valid2 & keep_prev2
        dx = g_m3 * w3
        dx = _masked_ffma3_acc(
            dx,
            g_m2,
            w2,
            valid1,
            g_m1,
            w1,
            valid2,
            g,
            w0,
            valid3,
        )
        _store_bf16_trunc(
            dx_ptr,
            (bos_base + (dx_t + 2).to(tl.int64)) * dim_i64 + d.to(tl.int64),
            dx,
            d_mask,
        )

        sum_bias = _ffma_one_acc(sum_bias, g)
        sum_w0 = _ffma_acc(sum_w0, g, src3)
        sum_w1 = _ffma_acc(sum_w1, g, src2)
        sum_w2 = _ffma_acc(sum_w2, g, src1)
        sum_w3 = _ffma_acc(sum_w3, g, x2)

        g_m3 = g_m2
        g_m2 = g_m1
        g_m1 = g
        prev3 = src2
        prev2 = src1
        prev1 = x2
        keep_tm2 = keep_tm1
        keep_tm1 = keep_prev2

        keep_prev3 = ~bos_t3
        src1 = tl.where(keep_prev3, prev1, 0.0)
        src2 = tl.where(keep_prev3, prev2, 0.0)
        src3 = tl.where(keep_prev3, prev3, 0.0)

        z = _conv4_ffma_z(x3, src1, src2, src3, w0, w1, w2, w3, bias)
        g = _silu_backward_apply(z, dout3)

        valid1 = keep_tm2
        valid2 = valid1 & keep_tm1
        valid3 = valid2 & keep_prev3
        dx = g_m3 * w3
        dx = _masked_ffma3_acc(
            dx,
            g_m2,
            w2,
            valid1,
            g_m1,
            w1,
            valid2,
            g,
            w0,
            valid3,
        )
        _store_bf16_trunc(
            dx_ptr,
            (bos_base + (dx_t + 3).to(tl.int64)) * dim_i64 + d.to(tl.int64),
            dx,
            d_mask,
        )

        sum_bias = _ffma_one_acc(sum_bias, g)
        sum_w0 = _ffma_acc(sum_w0, g, src3)
        sum_w1 = _ffma_acc(sum_w1, g, src2)
        sum_w2 = _ffma_acc(sum_w2, g, src1)
        sum_w3 = _ffma_acc(sum_w3, g, x3)

        g_m3 = g_m2
        g_m2 = g_m1
        g_m1 = g
        prev3 = src2
        prev2 = src1
        prev1 = x3
        keep_tm2 = keep_tm1
        keep_tm1 = keep_prev3
        if packed_xdout and zero_state:
            packed_iter_addr = _advance_packed_lane_address(
                packed_iter_addr,
                packed_quartet_bytes,
            )

    i = block_n - 4
    t0 = chunk_start + i
    base0 = row_base + i * dim_i64
    x0 = main_x0
    dout0 = main_dout
    base1 = base0 + dim_i64
    base2 = base1 + dim_i64
    base3 = base2 + dim_i64
    if packed_xdout:
        packed1 = tl.load(
            x_ptr.to(tl.pointer_type(tl.uint32)) + base1,
            mask=d_mask,
            other=0,
            volatile=True,
        )
        packed2 = tl.load(
            x_ptr.to(tl.pointer_type(tl.uint32)) + base2,
            mask=d_mask,
            other=0,
            volatile=True,
        )
        packed3 = tl.load(
            x_ptr.to(tl.pointer_type(tl.uint32)) + base3,
            mask=d_mask,
            other=0,
            volatile=True,
        )
    else:
        x1 = tl.load(
            x_ptr + base1,
            mask=d_mask,
            other=0.0,
            volatile=True,
        ).to(tl.float32)
        dout1 = tl.load(
            dout_ptr + base1,
            mask=d_mask,
            other=0.0,
            volatile=True,
        ).to(tl.float32)
        x2 = tl.load(
            x_ptr + base2,
            mask=d_mask,
            other=0.0,
            volatile=True,
        ).to(tl.float32)
        dout2 = tl.load(
            dout_ptr + base2,
            mask=d_mask,
            other=0.0,
            volatile=True,
        ).to(tl.float32)
        x3 = tl.load(
            x_ptr + base3,
            mask=d_mask,
            other=0.0,
            volatile=True,
        ).to(tl.float32)
        dout3 = tl.load(
            dout_ptr + base3,
            mask=d_mask,
            other=0.0,
            volatile=True,
        ).to(tl.float32)

    bos_word = tl.load((bos_ptr + bos_base + t0).to(tl.pointer_type(tl.uint32))).to(tl.int32)
    if packed_xdout:
        x1, dout1 = _unpack_xdout_u32(packed1)
        x2, dout2 = _unpack_xdout_u32(packed2)
        x3, dout3 = _unpack_xdout_u32(packed3)
    bos_t0 = (bos_word & 0x1) != 0
    bos_t1 = (bos_word & 0x100) != 0
    bos_t2 = (bos_word & 0x10000) != 0
    bos_t3 = (bos_word & 0x1000000) != 0
    keep_prev0 = ~bos_t0

    src1 = tl.where(keep_prev0, prev1, 0.0)
    src2 = tl.where(keep_prev0, prev2, 0.0)
    src3 = tl.where(keep_prev0, prev3, 0.0)

    z = _conv4_ffma_z(x0, src1, src2, src3, w0, w1, w2, w3, bias)
    g = _silu_backward_apply(z, dout0)

    dx_t = t0 - 3
    valid1 = keep_tm2
    valid2 = valid1 & keep_tm1
    valid3 = valid2 & keep_prev0
    dx = g_m3 * w3
    dx = _masked_ffma3_acc(
        dx,
        g_m2,
        w2,
        valid1,
        g_m1,
        w1,
        valid2,
        g,
        w0,
        valid3,
    )
    _store_bf16_trunc(
        dx_ptr,
        (bos_base + dx_t.to(tl.int64)) * dim_i64 + d.to(tl.int64),
        dx,
        d_mask,
    )

    sum_bias = _ffma_one_acc(sum_bias, g)
    sum_w0 = _ffma_acc(sum_w0, g, src3)
    sum_w1 = _ffma_acc(sum_w1, g, src2)
    sum_w2 = _ffma_acc(sum_w2, g, src1)
    sum_w3 = _ffma_acc(sum_w3, g, x0)

    g_m3 = g_m2
    g_m2 = g_m1
    g_m1 = g
    prev3 = src2
    prev2 = src1
    prev1 = x0
    keep_tm2 = keep_tm1
    keep_tm1 = keep_prev0

    keep_prev1 = ~bos_t1
    src1 = tl.where(keep_prev1, prev1, 0.0)
    src2 = tl.where(keep_prev1, prev2, 0.0)
    src3 = tl.where(keep_prev1, prev3, 0.0)

    z = _conv4_ffma_z(x1, src1, src2, src3, w0, w1, w2, w3, bias)
    g = _silu_backward_apply(z, dout1)

    valid1 = keep_tm2
    valid2 = valid1 & keep_tm1
    valid3 = valid2 & keep_prev1
    dx = g_m3 * w3
    dx = _masked_ffma3_acc(
        dx,
        g_m2,
        w2,
        valid1,
        g_m1,
        w1,
        valid2,
        g,
        w0,
        valid3,
    )
    _store_bf16_trunc(
        dx_ptr,
        (bos_base + (dx_t + 1).to(tl.int64)) * dim_i64 + d.to(tl.int64),
        dx,
        d_mask,
    )

    sum_bias = _ffma_one_acc(sum_bias, g)
    sum_w0 = _ffma_acc(sum_w0, g, src3)
    sum_w1 = _ffma_acc(sum_w1, g, src2)
    sum_w2 = _ffma_acc(sum_w2, g, src1)
    sum_w3 = _ffma_acc(sum_w3, g, x1)

    g_m3 = g_m2
    g_m2 = g_m1
    g_m1 = g
    prev3 = src2
    prev2 = src1
    prev1 = x1
    keep_tm2 = keep_tm1
    keep_tm1 = keep_prev1

    keep_prev2 = ~bos_t2
    src1 = tl.where(keep_prev2, prev1, 0.0)
    src2 = tl.where(keep_prev2, prev2, 0.0)
    src3 = tl.where(keep_prev2, prev3, 0.0)

    z = _conv4_ffma_z(x2, src1, src2, src3, w0, w1, w2, w3, bias)
    g = _silu_backward_apply(z, dout2)

    valid1 = keep_tm2
    valid2 = valid1 & keep_tm1
    valid3 = valid2 & keep_prev2
    dx = g_m3 * w3
    dx = _masked_ffma3_acc(
        dx,
        g_m2,
        w2,
        valid1,
        g_m1,
        w1,
        valid2,
        g,
        w0,
        valid3,
    )
    _store_bf16_trunc(
        dx_ptr,
        (bos_base + (dx_t + 2).to(tl.int64)) * dim_i64 + d.to(tl.int64),
        dx,
        d_mask,
    )

    sum_bias = _ffma_one_acc(sum_bias, g)
    sum_w0 = _ffma_acc(sum_w0, g, src3)
    sum_w1 = _ffma_acc(sum_w1, g, src2)
    sum_w2 = _ffma_acc(sum_w2, g, src1)
    sum_w3 = _ffma_acc(sum_w3, g, x2)

    g_m3 = g_m2
    g_m2 = g_m1
    g_m1 = g
    prev3 = src2
    prev2 = src1
    prev1 = x2
    keep_tm2 = keep_tm1
    keep_tm1 = keep_prev2

    keep_prev3 = ~bos_t3
    src1 = tl.where(keep_prev3, prev1, 0.0)
    src2 = tl.where(keep_prev3, prev2, 0.0)
    src3 = tl.where(keep_prev3, prev3, 0.0)

    z = _conv4_ffma_z(x3, src1, src2, src3, w0, w1, w2, w3, bias)
    g = _silu_backward_apply(z, dout3)

    valid1 = keep_tm2
    valid2 = valid1 & keep_tm1
    valid3 = valid2 & keep_prev3
    dx = g_m3 * w3
    dx = _masked_ffma3_acc(
        dx,
        g_m2,
        w2,
        valid1,
        g_m1,
        w1,
        valid2,
        g,
        w0,
        valid3,
    )
    _store_bf16_trunc(
        dx_ptr,
        (bos_base + (dx_t + 3).to(tl.int64)) * dim_i64 + d.to(tl.int64),
        dx,
        d_mask,
    )

    sum_bias = _ffma_one_acc(sum_bias, g)
    sum_w0 = _ffma_acc(sum_w0, g, src3)
    sum_w1 = _ffma_acc(sum_w1, g, src2)
    sum_w2 = _ffma_acc(sum_w2, g, src1)
    sum_w3 = _ffma_acc(sum_w3, g, x3)

    tl.atomic_add(partial_ptr + d + 0 * dim, sum_bias, sem="relaxed")
    tl.atomic_add(partial_ptr + d + 1 * dim, sum_w0, sem="relaxed")
    tl.atomic_add(partial_ptr + d + 2 * dim, sum_w1, sem="relaxed")
    tl.atomic_add(partial_ptr + d + 3 * dim, sum_w2, sem="relaxed")
    tl.atomic_add(partial_ptr + d + 4 * dim, sum_w3, sem="relaxed")

    g_m3 = g_m2
    g_m2 = g_m1
    g_m1 = g
    prev3 = src2
    prev2 = src1
    prev1 = x3
    keep_tm2 = keep_tm1
    keep_tm1 = keep_prev3

    tail_t0 = chunk_start + block_n
    tail_row0_valid = tail_t0 < seqlen
    tail_row1_valid = (tail_t0 + 1) < seqlen
    tail_bos_pair = tl.load(
        (bos_ptr + bos_base + tail_t0).to(tl.pointer_type(tl.uint16)),
        mask=tail_row1_valid,
        other=0,
    ).to(tl.int32)
    tail_bos0_pair = (tail_bos_pair & 0x1) != 0
    tail_bos1 = (tail_bos_pair & 0x100) != 0
    tail_bos0_scalar = tl.load(
        bos_ptr + bos_base + tail_t0,
        mask=tail_row0_valid & (~tail_row1_valid),
        other=True,
    )
    tail_bos0 = tl.where(
        tail_row0_valid,
        tl.where(tail_row1_valid, tail_bos0_pair, tail_bos0_scalar),
        True,
    )
    tail_bos1 = tl.where(tail_row1_valid, tail_bos1, True)

    for j in tl.static_range(0, 3):
        i = block_n + j
        t = chunk_start + i
        row_valid = t < seqlen
        base = row_base + i * dim_i64

        if packed_xdout:
            packed = tl.load(
                x_ptr.to(tl.pointer_type(tl.uint32)) + base,
                mask=d_mask & row_valid,
                other=0,
                volatile=True,
            )
            x0, dout = _unpack_xdout_u32(packed)
        else:
            x0 = tl.load(
                x_ptr + base,
                mask=d_mask & row_valid,
                other=0.0,
                volatile=True,
            ).to(tl.float32)
            dout = tl.load(
                dout_ptr + base,
                mask=d_mask & row_valid,
                other=0.0,
                volatile=True,
            ).to(tl.float32)
        if j == 0:
            bos_t = tail_bos0
        elif j == 1:
            bos_t = tail_bos1
        else:
            tail_bos2_bits = tl.load(
                bos_ptr + bos_base + t,
                mask=row_valid,
                other=1,
            ).to(tl.int32)
            bos_t = (tail_bos2_bits & 0x1) != 0
        keep_prev = row_valid & (~bos_t)
        src1 = tl.where(keep_prev, prev1, 0.0)
        src2 = tl.where(keep_prev, prev2, 0.0)
        src3 = tl.where(keep_prev, prev3, 0.0)

        z = _conv4_ffma_z(x0, src1, src2, src3, w0, w1, w2, w3, bias)
        g = tl.where(row_valid, _silu_backward_apply(z, dout), 0.0)

        dx_t = t - 3
        valid1 = keep_tm2
        valid2 = valid1 & keep_tm1
        valid3 = valid2 & keep_prev
        dx = g_m3 * w3
        dx = _masked_ffma3_acc(
            dx,
            g_m2,
            w2,
            valid1,
            g_m1,
            w1,
            valid2,
            g,
            w0,
            valid3,
        )
        if not zero_state:
            dx += tl.where((dx_t == (seqlen - 3)) & valid2, df0, 0.0)
            dx += tl.where((dx_t == (seqlen - 2)) & valid1, df1, 0.0)
            dx += tl.where(dx_t == (seqlen - 1), df2, 0.0)
        _store_bf16_trunc(
            dx_ptr,
            (bos_base + dx_t.to(tl.int64)) * dim_i64 + d.to(tl.int64),
            dx,
            d_mask & (dx_t < seqlen),
        )

        if j < 2:
            g_m3 = g_m2
            g_m2 = g_m1
            g_m1 = g
            prev3 = src2
            prev2 = src1
            prev1 = x0
            keep_tm2 = keep_tm1
            keep_tm1 = keep_prev


@triton.jit
def _initial_state_correction_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    initial_ptr,
    bos_ptr,
    dfinal_ptr,
    dx_ptr,
    dinitial_ptr,
    partial_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    block_d: tl.constexpr,
):
    pid_d = tl.program_id(0)
    b = tl.program_id(1)
    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim
    w0, w1, w2, w3 = _load_weight4_bf16(weight_ptr, d, d_mask)
    bias = _load_bias_bf16_u16(bias_ptr, d, d_mask)

    dim_i64 = tl.full((), dim, tl.int64)
    seqlen_i64 = tl.full((), seqlen, tl.int64)
    b_i64 = b.to(tl.int64)
    row_base = b_i64 * seqlen_i64 * dim_i64 + d.to(tl.int64)
    bos_base = b_i64 * seqlen_i64
    init_base = b_i64 * dim_i64 * 3 + d.to(tl.int64) * 3

    init0 = tl.load(initial_ptr + init_base + 0, mask=d_mask, other=0.0).to(tl.float32)
    init1 = tl.load(initial_ptr + init_base + 1, mask=d_mask, other=0.0).to(tl.float32)
    init2 = tl.load(initial_ptr + init_base + 2, mask=d_mask, other=0.0).to(tl.float32)

    true_prev1 = init2
    true_prev2 = init1
    true_prev3 = init0
    base_prev1 = tl.zeros((block_d,), tl.float32)
    base_prev2 = tl.zeros((block_d,), tl.float32)
    base_prev3 = tl.zeros((block_d,), tl.float32)

    g0 = tl.zeros((block_d,), tl.float32)
    g1 = tl.zeros((block_d,), tl.float32)
    g2 = tl.zeros((block_d,), tl.float32)
    g3 = tl.zeros((block_d,), tl.float32)
    g4 = tl.zeros((block_d,), tl.float32)
    g5 = tl.zeros((block_d,), tl.float32)

    corr_bias = tl.zeros((block_d,), tl.float32)
    corr_w0 = tl.zeros((block_d,), tl.float32)
    corr_w1 = tl.zeros((block_d,), tl.float32)
    corr_w2 = tl.zeros((block_d,), tl.float32)
    corr_w3 = tl.zeros((block_d,), tl.float32)

    for j in tl.static_range(0, 6):
        base = row_base + j * dim_i64
        packed = tl.load(
            x_ptr.to(tl.pointer_type(tl.uint32)) + base,
            mask=d_mask,
            other=0,
            eviction_policy="evict_last",
        )
        x0, dout = _unpack_xdout_u32(packed)
        bos_t = tl.load(bos_ptr + bos_base + j)
        keep_prev = ~bos_t

        true_src1 = tl.where(keep_prev, true_prev1, 0.0)
        true_src2 = tl.where(keep_prev, true_prev2, 0.0)
        true_src3 = tl.where(keep_prev, true_prev3, 0.0)
        z_true = _conv4_ffma_z(x0, true_src1, true_src2, true_src3, w0, w1, w2, w3, bias)
        g_true = _silu_backward_apply(z_true, dout)

        if j < 3:
            base_src1 = tl.where(keep_prev, base_prev1, 0.0)
            base_src2 = tl.where(keep_prev, base_prev2, 0.0)
            base_src3 = tl.where(keep_prev, base_prev3, 0.0)
            z_base = _conv4_ffma_z(x0, base_src1, base_src2, base_src3, w0, w1, w2, w3, bias)
            g_base = _silu_backward_apply(z_base, dout)
            dg = g_true - g_base
            corr_bias += dg
            corr_w0 += g_true * true_src3 - g_base * base_src3
            corr_w1 += g_true * true_src2 - g_base * base_src2
            corr_w2 += g_true * true_src1 - g_base * base_src1
            corr_w3 += dg * x0
            base_prev3 = base_src2
            base_prev2 = base_src1
            base_prev1 = x0

        if j == 0:
            g0 = g_true
        elif j == 1:
            g1 = g_true
        elif j == 2:
            g2 = g_true
        elif j == 3:
            g3 = g_true
        elif j == 4:
            g4 = g_true
        else:
            g5 = g_true

        true_prev3 = true_src2
        true_prev2 = true_src1
        true_prev1 = x0

    bos0 = tl.load(bos_ptr + bos_base + 0)
    bos1 = tl.load(bos_ptr + bos_base + 1)
    bos2 = tl.load(bos_ptr + bos_base + 2)
    bos3 = tl.load(bos_ptr + bos_base + 3)
    bos4 = tl.load(bos_ptr + bos_base + 4)
    bos5 = tl.load(bos_ptr + bos_base + 5)

    valid0 = ~bos0
    valid1_init = valid0 & (~bos1)
    valid2_init = valid1_init & (~bos2)
    di0 = tl.where(valid0, g0 * w0, 0.0)
    di1 = tl.where(valid0, g0 * w1, 0.0) + tl.where(valid1_init, g1 * w0, 0.0)
    di2 = (
        tl.where(valid0, g0 * w2, 0.0)
        + tl.where(valid1_init, g1 * w1, 0.0)
        + tl.where(valid2_init, g2 * w0, 0.0)
    )
    tl.store(dinitial_ptr + init_base + 0, di0, mask=d_mask)
    tl.store(dinitial_ptr + init_base + 1, di1, mask=d_mask)
    tl.store(dinitial_ptr + init_base + 2, di2, mask=d_mask)

    dx0_valid1 = ~bos1
    dx0_valid2 = dx0_valid1 & (~bos2)
    dx0_valid3 = dx0_valid2 & (~bos3)
    dx0 = (
        g0 * w3
        + tl.where(dx0_valid1, g1 * w2, 0.0)
        + tl.where(dx0_valid2, g2 * w1, 0.0)
        + tl.where(dx0_valid3, g3 * w0, 0.0)
    )
    dx1_valid1 = ~bos2
    dx1_valid2 = dx1_valid1 & (~bos3)
    dx1_valid3 = dx1_valid2 & (~bos4)
    dx1 = (
        g1 * w3
        + tl.where(dx1_valid1, g2 * w2, 0.0)
        + tl.where(dx1_valid2, g3 * w1, 0.0)
        + tl.where(dx1_valid3, g4 * w0, 0.0)
    )
    dx2_valid1 = ~bos3
    dx2_valid2 = dx2_valid1 & (~bos4)
    dx2_valid3 = dx2_valid2 & (~bos5)
    dx2 = (
        g2 * w3
        + tl.where(dx2_valid1, g3 * w2, 0.0)
        + tl.where(dx2_valid2, g4 * w1, 0.0)
        + tl.where(dx2_valid3, g5 * w0, 0.0)
    )
    tl.store(dx_ptr + row_base + 0 * dim_i64, dx0.to(tl.bfloat16), mask=d_mask)
    tl.store(dx_ptr + row_base + 1 * dim_i64, dx1.to(tl.bfloat16), mask=d_mask)
    tl.store(dx_ptr + row_base + 2 * dim_i64, dx2.to(tl.bfloat16), mask=d_mask)

    tl.atomic_add(partial_ptr + d + 0 * dim, corr_bias, sem="relaxed")
    tl.atomic_add(partial_ptr + d + 1 * dim, corr_w0, sem="relaxed")
    tl.atomic_add(partial_ptr + d + 2 * dim, corr_w1, sem="relaxed")
    tl.atomic_add(partial_ptr + d + 3 * dim, corr_w2, sem="relaxed")
    tl.atomic_add(partial_ptr + d + 4 * dim, corr_w3, sem="relaxed")

    last_row0_t = seqlen - 3
    last_row_base = (b_i64 * seqlen_i64 + last_row0_t) * dim_i64 + d.to(tl.int64)

    last_bos1 = tl.load(bos_ptr + bos_base + seqlen_i64 - 2)
    last_bos2 = tl.load(bos_ptr + bos_base + seqlen_i64 - 1)
    df0 = tl.load(dfinal_ptr + init_base + 0, mask=d_mask, other=0.0).to(tl.float32)
    df1 = tl.load(dfinal_ptr + init_base + 1, mask=d_mask, other=0.0).to(tl.float32)
    df2 = tl.load(dfinal_ptr + init_base + 2, mask=d_mask, other=0.0).to(tl.float32)

    last_valid1 = ~last_bos1
    last_valid2 = last_valid1 & (~last_bos2)
    last_dx0 = tl.load(
        dx_ptr + last_row_base + 0 * dim_i64,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    last_dx1 = tl.load(
        dx_ptr + last_row_base + 1 * dim_i64,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    last_dx2 = tl.load(
        dx_ptr + last_row_base + 2 * dim_i64,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    last_dx0 += tl.where(last_valid2, df0, 0.0)
    last_dx1 += tl.where(~last_bos2, df1, 0.0)
    last_dx2 += df2
    tl.store(
        dx_ptr + last_row_base + 0 * dim_i64,
        last_dx0.to(tl.bfloat16),
        mask=d_mask,
    )
    tl.store(
        dx_ptr + last_row_base + 1 * dim_i64,
        last_dx1.to(tl.bfloat16),
        mask=d_mask,
    )
    tl.store(
        dx_ptr + last_row_base + 2 * dim_i64,
        last_dx2.to(tl.bfloat16),
        mask=d_mask,
    )


@triton.jit
def _final_state_correction_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    bos_ptr,
    dfinal_ptr,
    dx_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    block_d: tl.constexpr,
):
    pid_d = tl.program_id(0)
    b = tl.program_id(1)
    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim
    w0, w1, w2, w3 = _load_weight4_bf16(weight_ptr, d, d_mask)
    bias = _load_bias_bf16_u16(bias_ptr, d, d_mask)

    dim_i64 = tl.full((), dim, tl.int64)
    seqlen_i64 = tl.full((), seqlen, tl.int64)
    b_i64 = b.to(tl.int64)
    row0_t = seqlen - 3
    row_base = (b_i64 * seqlen_i64 + row0_t) * dim_i64 + d.to(tl.int64)
    bos_base = b_i64 * seqlen_i64

    g0 = tl.zeros((block_d,), tl.float32)
    g1 = tl.zeros((block_d,), tl.float32)
    g2 = tl.zeros((block_d,), tl.float32)

    for j in tl.static_range(0, 3):
        t = row0_t + j
        base = row_base + j * dim_i64
        packed = tl.load(
            x_ptr.to(tl.pointer_type(tl.uint32)) + base,
            mask=d_mask,
            other=0,
            eviction_policy="evict_last",
        )
        x0, dout = _unpack_xdout_u32(packed)
        prev1 = _unpack_x_u32(
            tl.load(
                x_ptr.to(tl.pointer_type(tl.uint32)) + base - dim_i64,
                mask=d_mask,
                other=0,
                eviction_policy="evict_last",
            )
        )
        prev2 = _unpack_x_u32(
            tl.load(
                x_ptr.to(tl.pointer_type(tl.uint32)) + base - 2 * dim_i64,
                mask=d_mask,
                other=0,
                eviction_policy="evict_last",
            )
        )
        prev3 = _unpack_x_u32(
            tl.load(
                x_ptr.to(tl.pointer_type(tl.uint32)) + base - 3 * dim_i64,
                mask=d_mask,
                other=0,
                eviction_policy="evict_last",
            )
        )
        bos_t = tl.load(bos_ptr + bos_base + t)
        bos_tm1 = tl.load(bos_ptr + bos_base + t - 1)
        bos_tm2 = tl.load(bos_ptr + bos_base + t - 2)
        valid1 = ~bos_t
        valid2 = valid1 & (~bos_tm1)
        valid3 = valid2 & (~bos_tm2)
        src1 = tl.where(valid1, prev1, 0.0)
        src2 = tl.where(valid2, prev2, 0.0)
        src3 = tl.where(valid3, prev3, 0.0)
        z = _conv4_ffma_z(x0, src1, src2, src3, w0, w1, w2, w3, bias)
        g = _silu_backward_apply(z, dout)
        if j == 0:
            g0 = g
        elif j == 1:
            g1 = g
        else:
            g2 = g

    bos1 = tl.load(bos_ptr + bos_base + seqlen_i64 - 2)
    bos2 = tl.load(bos_ptr + bos_base + seqlen_i64 - 1)
    df_base = b_i64 * dim_i64 * 3 + d.to(tl.int64) * 3
    df0 = tl.load(dfinal_ptr + df_base + 0, mask=d_mask, other=0.0).to(tl.float32)
    df1 = tl.load(dfinal_ptr + df_base + 1, mask=d_mask, other=0.0).to(tl.float32)
    df2 = tl.load(dfinal_ptr + df_base + 2, mask=d_mask, other=0.0).to(tl.float32)

    valid0_1 = ~bos1
    valid0_2 = valid0_1 & (~bos2)
    dx0 = g0 * w3 + tl.where(valid0_1, g1 * w2, 0.0) + tl.where(valid0_2, g2 * w1 + df0, 0.0)
    dx1 = g1 * w3 + tl.where(~bos2, g2 * w2 + df1, 0.0)
    dx2 = g2 * w3 + df2
    tl.store(dx_ptr + row_base + 0 * dim_i64, dx0.to(tl.bfloat16), mask=d_mask)
    tl.store(dx_ptr + row_base + 1 * dim_i64, dx1.to(tl.bfloat16), mask=d_mask)
    tl.store(dx_ptr + row_base + 2 * dim_i64, dx2.to(tl.bfloat16), mask=d_mask)


@triton.jit
def _zero_dwdbias_accum_kernel(
    accum_ptr,
    dim: tl.constexpr,
    block_d: tl.constexpr,
):
    pid_d = tl.program_id(0)
    d = pid_d * block_d + tl.arange(0, block_d)
    mask = d < dim
    z = tl.zeros((block_d,), tl.float32)
    tl.store(accum_ptr + d + 0 * dim, z, mask=mask)
    tl.store(accum_ptr + d + 1 * dim, z, mask=mask)
    tl.store(accum_ptr + d + 2 * dim, z, mask=mask)
    tl.store(accum_ptr + d + 3 * dim, z, mask=mask)
    tl.store(accum_ptr + d + 4 * dim, z, mask=mask)


@triton.jit
def _atomic_dwdbias_final_kernel(
    accum_ptr,
    dweight_ptr,
    dbias_ptr,
    dim: tl.constexpr,
    block_d: tl.constexpr,
):
    pid_d = tl.program_id(0)
    d = pid_d * block_d + tl.arange(0, block_d)
    sum_bias = tl.load(accum_ptr + d + 0 * dim)
    sum_w0 = tl.load(accum_ptr + d + 1 * dim)
    sum_w1 = tl.load(accum_ptr + d + 2 * dim)
    sum_w2 = tl.load(accum_ptr + d + 3 * dim)
    sum_w3 = tl.load(accum_ptr + d + 4 * dim)
    tl.store(dbias_ptr + d, sum_bias)
    w_base = d * 4
    tl.store(dweight_ptr + w_base + 0, sum_w0)
    tl.store(dweight_ptr + w_base + 1, sum_w1)
    tl.store(dweight_ptr + w_base + 2, sum_w2)
    tl.store(dweight_ptr + w_base + 3, sum_w3)
    z = tl.zeros((block_d,), tl.float32)
    tl.store(accum_ptr + d + 0 * dim, z)
    tl.store(accum_ptr + d + 1 * dim, z)
    tl.store(accum_ptr + d + 2 * dim, z)
    tl.store(accum_ptr + d + 3 * dim, z)
    tl.store(accum_ptr + d + 4 * dim, z)


def _get_zeroed_partial(
    x: torch.Tensor,
    dim: int,
    final_block_d: int,
) -> torch.Tensor:
    device_idx = x.device.index
    if device_idx is None:
        device_idx = torch.cuda.current_device()
    stream_id = torch.cuda.current_stream(x.device).cuda_stream
    key = (int(device_idx), int(dim), int(stream_id))
    partial = _ZEROED_PARTIAL_CACHE.get(key)
    if partial is None or partial.device != x.device or partial.shape != (5, dim):
        partial = torch.empty((5, dim), device=x.device, dtype=torch.float32)
        _zero_dwdbias_accum_kernel[(triton.cdiv(dim, final_block_d),)](
            partial,
            dim,
            final_block_d,
            num_warps=1,
            num_stages=4,
        )
        _ZEROED_PARTIAL_CACHE[key] = partial
    return partial


def _fast_path_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    bos_mask: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    activation: str | None,
    dout: torch.Tensor | None,
    dfinal_states: torch.Tensor | None,
) -> bool:
    if activation != "silu" or dout is None or dfinal_states is None:
        return False
    if bias is None or initial_states is None or bos_mask is None:
        return False
    if not x.is_cuda or x.dtype != torch.bfloat16:
        return False
    if weight.dtype != torch.bfloat16 or dout.dtype != torch.bfloat16:
        return False
    if bias.dtype != torch.bfloat16 or initial_states.dtype != torch.bfloat16:
        return False
    if dfinal_states.dtype != torch.bfloat16:
        return False
    if x.ndim != 3 or weight.ndim != 2:
        return False
    batch, seqlen, dim = x.shape
    if not isinstance(bos_mask, tuple) or len(bos_mask) != 2:
        return False
    dense_bos_mask, bos_offsets = bos_mask
    if not torch.is_tensor(dense_bos_mask) or not torch.is_tensor(bos_offsets):
        return False
    if bos_offsets.ndim != 2 or bos_offsets.shape[1] != 2:
        return False
    if bos_offsets.dtype != torch.int64 or not bos_offsets.is_cuda:
        return False
    if weight.shape != (dim, 4) or seqlen < 3:
        return False
    if initial_states.shape != (batch, dim, 3):
        return False
    if dout.shape != x.shape or dfinal_states.shape != (batch, dim, 3):
        return False
    if bias.shape != (dim,) or dense_bos_mask.shape != (batch, seqlen):
        return False
    if dense_bos_mask.dtype != torch.bool:
        return False
    return (
        x.is_contiguous()
        and weight.is_contiguous()
        and bias.is_contiguous()
        and initial_states.is_contiguous()
        and dense_bos_mask.is_contiguous()
        and bos_offsets.is_contiguous()
        and dout.is_contiguous()
        and dfinal_states.is_contiguous()
    )


def _packed_xdout_fast_path_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    bos_mask: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    activation: str | None,
    dout: torch.Tensor | None,
    dfinal_states: torch.Tensor | None,
) -> bool:
    if activation != "silu" or dout is not None or dfinal_states is None:
        return False
    if bias is None or initial_states is None or bos_mask is None:
        return False
    if not x.is_cuda or x.dtype != torch.bfloat16:
        return False
    if weight.dtype != torch.bfloat16:
        return False
    if bias.dtype != torch.bfloat16 or initial_states.dtype != torch.bfloat16:
        return False
    if dfinal_states.dtype != torch.bfloat16:
        return False
    if x.ndim != 4 or x.shape[-1] != 2 or weight.ndim != 2:
        return False
    batch, seqlen, dim, _ = x.shape
    if not isinstance(bos_mask, tuple) or len(bos_mask) != 2:
        return False
    dense_bos_mask, bos_offsets = bos_mask
    if not torch.is_tensor(dense_bos_mask) or not torch.is_tensor(bos_offsets):
        return False
    if bos_offsets.ndim != 2 or bos_offsets.shape[1] != 2:
        return False
    if bos_offsets.dtype != torch.int64 or not bos_offsets.is_cuda:
        return False
    if weight.shape != (dim, 4) or seqlen < 3:
        return False
    if initial_states.shape != (batch, dim, 3):
        return False
    if dfinal_states.shape != (batch, dim, 3):
        return False
    if bias.shape != (dim,) or dense_bos_mask.shape != (batch, seqlen):
        return False
    if dense_bos_mask.dtype != torch.bool:
        return False
    return (
        x.is_contiguous()
        and weight.is_contiguous()
        and bias.is_contiguous()
        and initial_states.is_contiguous()
        and dense_bos_mask.is_contiguous()
        and bos_offsets.is_contiguous()
        and dfinal_states.is_contiguous()
    )


def _fast_backward_width4_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    initial_states: torch.Tensor,
    bos_mask: tuple[torch.Tensor, torch.Tensor],
    dout: torch.Tensor | None,
    dfinal_states: torch.Tensor,
    packed_xdout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if packed_xdout:
        batch, seqlen, dim, _ = x.shape
    else:
        batch, seqlen, dim = x.shape
    dense_bos_mask, bos_offsets = bos_mask
    dx = torch.empty((batch, seqlen, dim), device=x.device, dtype=torch.bfloat16)
    dinitial = torch.empty_like(initial_states)

    fused_block_n = 256
    fused_block_d = 64
    fused_chunks_per_batch = triton.cdiv(seqlen, fused_block_n)
    fused_total_chunks = batch * fused_chunks_per_batch
    fused_block_chunks = triton.next_power_of_2(fused_total_chunks)
    use_fused_chunk = (
        seqlen >= fused_block_n
        and (seqlen % fused_block_n) == 0
        and (dim % fused_block_d) == 0
        and fused_block_chunks <= 2048
    )
    if use_fused_chunk:
        final_block_d = 16
        final_d_blocks = triton.cdiv(dim, final_block_d)
        partial = _get_zeroed_partial(x, dim, final_block_d)
        dweight = torch.empty_like(weight)
        dbias = torch.empty_like(bias)
        zero_state_corrections = packed_xdout
        _fused_chunk_bwd_kernel[(triton.cdiv(dim, fused_block_d), fused_total_chunks)](
            x,
            weight,
            bias,
            initial_states,
            dense_bos_mask,
            dout,
            dfinal_states,
            dx,
            dinitial,
            partial,
            seqlen,
            dim,
            fused_chunks_per_batch,
            fused_total_chunks,
            fused_block_n,
            fused_block_d,
            packed_xdout,
            zero_state_corrections,
            num_warps=1,
            num_stages=2,
            enable_fp_fusion=True,
            maxnreg=64,
        )
        if zero_state_corrections:
            repair_block_d = 32
            repair_grid = (triton.cdiv(dim, repair_block_d), batch)
            _initial_state_correction_kernel[repair_grid](
                x,
                weight,
                bias,
                initial_states,
                dense_bos_mask,
                dfinal_states,
                dx,
                dinitial,
                partial,
                seqlen,
                dim,
                repair_block_d,
                num_warps=1,
                num_stages=4,
                enable_fp_fusion=True,
                enable_reflect_ftz=True,
                maxnreg=72,
            )
        _atomic_dwdbias_final_kernel[(final_d_blocks,)](
            partial,
            dweight,
            dbias,
            dim,
            final_block_d,
            num_warps=1,
            num_stages=4,
            enable_reflect_ftz=True,
        )
        return dx, dweight, dbias, dinitial

    if packed_xdout:
        return reference_kernel_fn(
            x,
            weight,
            bias,
            initial_states,
            bos_mask,
            "silu",
            None,
            dfinal_states,
        )

    g = torch.empty((batch, seqlen, dim), device=x.device, dtype=torch.float32)
    block_d = 512
    num_d_blocks = triton.cdiv(dim, block_d)
    tiles_t = triton.cdiv(seqlen, 16)
    tile_rows = batch * tiles_t
    swizzle_group_rows = 16 if (tile_rows % 16) == 0 else 0
    grid_t8 = (num_d_blocks * tile_rows,)
    stage1_clear = torch.empty((batch, seqlen), device=x.device, dtype=torch.uint8)
    dx_valid = torch.empty((batch, seqlen), device=x.device, dtype=torch.uint8)

    _compute_g_t8_kernel[grid_t8](
        x,
        weight,
        bias,
        initial_states,
        dense_bos_mask,
        dout,
        g,
        stage1_clear,
        dx_valid,
        dx,
        seqlen,
        dim,
        block_d,
        tiles_t,
        num_d_blocks,
        swizzle_group_rows,
        num_warps=4,
        num_stages=4,
    )
    _compute_g_prefix_repair_kernel[(num_d_blocks, batch)](
        x,
        weight,
        bias,
        initial_states,
        dense_bos_mask,
        dout,
        g,
        seqlen,
        dim,
        block_d,
        num_warps=4,
        num_stages=4,
    )
    num_bos_offsets = bos_offsets.shape[0]
    repair_group_offsets = 16 if (num_bos_offsets % 16) == 0 else 0
    if num_bos_offsets > 0:
        repair_grid = (num_d_blocks * num_bos_offsets,)
        _compute_g_bos_offsets_repair_kernel[repair_grid](
            x,
            weight,
            bias,
            dense_bos_mask,
            dout,
            bos_offsets,
            g,
            seqlen,
            dim,
            block_d,
            num_bos_offsets,
            num_d_blocks,
            repair_group_offsets,
            num_warps=4,
            num_stages=4,
        )
    _compute_dx_prefix_repair_kernel[(num_d_blocks, batch)](
        g,
        weight,
        dx_valid,
        dx,
        seqlen,
        dim,
        block_d,
        num_warps=4,
        num_stages=4,
    )
    if num_bos_offsets > 0:
        _compute_dx_bos_offsets_repair_kernel[repair_grid](
            g,
            weight,
            dx_valid,
            bos_offsets,
            dx,
            seqlen,
            dim,
            block_d,
            num_d_blocks,
            repair_group_offsets,
            num_warps=4,
            num_stages=4,
        )
    _dinitial_kernel[(num_d_blocks, batch)](
        g,
        weight,
        dense_bos_mask,
        dx_valid,
        dfinal_states,
        dx,
        dinitial,
        seqlen,
        dim,
        block_d,
        num_warps=8,
        num_stages=4,
    )

    block_n = 2048
    reduce_block_d = 32
    stage1_reduce_block_d = 64
    chunks_per_batch = triton.cdiv(seqlen, block_n)
    total_chunks = batch * chunks_per_batch
    block_chunks = triton.next_power_of_2(total_chunks)
    full_stage1_tiles = ((seqlen % block_n) == 0) and (
        (dim % stage1_reduce_block_d) == 0
    )
    if block_chunks > 2048:
        return reference_kernel_fn(
            x,
            weight,
            bias,
            initial_states,
            bos_mask,
            "silu",
            dout,
            dfinal_states,
        )
    reduce_d_blocks = triton.cdiv(dim, reduce_block_d)
    stage1_d_blocks = triton.cdiv(dim, stage1_reduce_block_d)
    partial = torch.empty(
        (5, total_chunks, dim),
        device=x.device,
        dtype=torch.float32,
    )
    dweight = torch.empty_like(weight)
    dbias = torch.empty_like(bias)
    _dw_dbias_stage1_rolling_kernel[(stage1_d_blocks, total_chunks)](
        g,
        x,
        stage1_clear,
        partial,
        seqlen,
        dim,
        chunks_per_batch,
        total_chunks,
        block_n,
        stage1_reduce_block_d,
        full_stage1_tiles,
        num_warps=1,
        num_stages=4,
        enable_fp_fusion=False,
    )
    _dw_initial_correction_kernel[(reduce_d_blocks, batch)](
        g,
        initial_states,
        dense_bos_mask,
        partial,
        seqlen,
        dim,
        chunks_per_batch,
        total_chunks,
        reduce_block_d,
        num_warps=1,
        num_stages=4,
    )
    _dw_dbias_final_kernel[(reduce_d_blocks,)](
        partial,
        dweight,
        dbias,
        dim,
        total_chunks,
        block_chunks,
        reduce_block_d,
        (dim % reduce_block_d) == 0,
        num_warps=8,
        num_stages=4,
    )
    return dx, dweight, dbias, dinitial


def kernel_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    bos_mask: torch.Tensor | None = None,
    activation: str | None = None,
    dout: torch.Tensor | None = None,
    dfinal_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if _fast_path_supported(
        x,
        weight,
        bias,
        initial_states,
        bos_mask,
        activation,
        dout,
        dfinal_states,
    ):
        return _fast_backward_width4_bf16(
            x,
            weight,
            bias,
            initial_states,
            bos_mask,
            dout,
            dfinal_states,
        )
    if _packed_xdout_fast_path_supported(
        x,
        weight,
        bias,
        initial_states,
        bos_mask,
        activation,
        dout,
        dfinal_states,
    ):
        return _fast_backward_width4_bf16(
            x,
            weight,
            bias,
            initial_states,
            bos_mask,
            None,
            dfinal_states,
            packed_xdout=True,
        )

    return reference_kernel_fn(
        x,
        weight,
        bias,
        initial_states,
        bos_mask,
        activation,
        dout,
        dfinal_states,
    )


# The exact a0/452 fused cubin snapshots a noncontiguous R48:R47 dx store
# pointer into R52:R53 on every recurrent trip.  R49 holds tid.x, but its only
# loop use is the lane predicate at PC 0x1740.  R14's low six bits are exactly
# 2*tid.x, so `(R14 & 0x1e) != 0` is equivalent to `(R49 & 0xf) != 0` and frees
# adjacent in-range R49 before the pointer definition.  This fail-closed SM90
# transform carries R48:R49, hoists the two already-ready late stores into the
# old advance sites, advances in their vacated mid-tail slots, and compacts the
# two predicate holes to shorten the loop by two instructions.
_INRANGE_DX_PARENT_TEXT_SHA256 = (
    "880498858cb4c0a2505844d6f3bb71b1693312eeaa9d60ec43c562cb59283c1e"
)
_INRANGE_DX_OUTPUT_TEXT_SHA256 = (
    "9a0f99f7a8bd1eb110038c750b4cadf0c47f589759bf471883e44c0de82c56b1"
)
_INRANGE_DX_POINTER_HIGH_R49 = _sm90_instruction(
    0x0000000FFF317C10,
    0x000FE4000B7EA41F,
)
_INRANGE_DX_LANE_PREDICATE_R14 = _sm90_instruction(
    0x0000001E0EFF7812,
    0x000FC6000788C0FF,
)
_INRANGE_DX_STORE_M2000_R48 = _sm90_instruction(
    0xFFE000373000A986,
    0x000FE8000C101910,
)
_INRANGE_DX_STORE_0_R48 = _sm90_instruction(
    0x000000233000A986,
    0x000FE2000C101910,
)
_INRANGE_DX_STORE_M6000_R48 = _sm90_instruction(
    0xFFA000163000A986,
    0x0003E2000C101910,
)
_INRANGE_DX_STORE_M4000_R48 = _sm90_instruction(
    0xFFC000263000A986,
    0x0005E2000C101910,
)
_INRANGE_DX_TAIL_ADVANCE_LOW = _sm90_instruction(
    0x0000800030307810,
    0x000FC60007F9E0FF,
)
_INRANGE_DX_TAIL_ADVANCE_HIGH = _sm90_instruction(
    0x00000031FF317210,
    0x000FE400027FE4FF,
)
_INRANGE_DX_BRANCH_2570 = _sm90_instruction(
    0xFFFFFFEC00F8B947,
    0x000FF0000383FFFF,
)


def _apply_inrange_dx_peephole(cubin):
    sections = _elf64_sections(cubin)
    text_entry = sections.get(_NARROW_LOOP_TEXT_SECTION)
    if text_entry is None:
        return cubin
    text_offset, text_size = text_entry
    if text_size != 0x4A00 or text_offset + text_size > len(cubin):
        return cubin
    parent_text = cubin[text_offset : text_offset + text_size]
    if (
        hashlib.sha256(parent_text).hexdigest()
        != _INRANGE_DX_PARENT_TEXT_SHA256
    ):
        return cubin

    def read_pc(pc):
        return cubin[text_offset + pc : text_offset + pc + 16]

    expected = {
        0x11C0: _sm90_instruction(0x0000000FFF2F7C10, 0x000FE4000B7EA41F),
        0x1740: _sm90_instruction(0x0000000F31FF7812, 0x000FC6000788C0FF),
        0x2360: _sm90_instruction(0x0000003000347202, 0x000FE20000000F00),
        0x23A0: _sm90_instruction(0x0000002F00357202, 0x000FE20000000F00),
        0x23C0: _sm90_instruction(0x0000800034307810, 0x000FC60007F7E0FF),
        0x23D0: _sm90_instruction(0xFFE000373400A986, 0x000FE8000C101910),
        0x23E0: _sm90_instruction(0x000000233400A986, 0x000FE2000C101910),
        0x23F0: _sm90_instruction(0x00000035FF2F7210, 0x000FE40001FFE4FF),
        0x2490: _sm90_instruction(0xFFA000163400A986, 0x0003E2000C101910),
        0x24D0: _sm90_instruction(0xFFC000263400A986, 0x0005E2000C101910),
        0x2520: _sm90_instruction(0x000000000000781C, 0x000FE20000F2E170),
        0x2540: _sm90_instruction(0x000000000000781C, 0x000FE2000070E170),
        0x2590: _NARROW_LOOP_COMPACT_BRANCH,
    }
    if any(read_pc(pc) != encoded for pc, encoded in expected.items()):
        return cubin

    output = bytearray(cubin)

    def write_pc(pc, encoded):
        output[text_offset + pc : text_offset + pc + 16] = encoded

    write_pc(0x11C0, _INRANGE_DX_POINTER_HIGH_R49)
    write_pc(0x1740, _INRANGE_DX_LANE_PREDICATE_R14)
    write_pc(0x2360, read_pc(0x2520))
    write_pc(0x23A0, read_pc(0x2540))
    write_pc(0x23C0, _INRANGE_DX_STORE_M6000_R48)
    write_pc(0x23D0, _INRANGE_DX_STORE_M2000_R48)
    write_pc(0x23E0, _INRANGE_DX_STORE_0_R48)
    write_pc(0x23F0, _INRANGE_DX_STORE_M4000_R48)
    write_pc(0x2490, _INRANGE_DX_TAIL_ADVANCE_LOW)
    write_pc(0x24D0, _INRANGE_DX_TAIL_ADVANCE_HIGH)

    for target_pc, source_pc in zip(
        (0x2520, 0x2530, 0x2540, 0x2550, 0x2560),
        (0x2530, 0x2550, 0x2560, 0x2570, 0x2580),
        strict=True,
    ):
        write_pc(target_pc, read_pc(source_pc))
    write_pc(0x2570, _INRANGE_DX_BRANCH_2570)
    for pc in (0x2580, 0x2590, 0x25A0, 0x25B0):
        write_pc(pc, _NARROW_LOOP_NOP)

    transformed = bytes(output)
    transformed_text = transformed[text_offset : text_offset + text_size]
    if (
        hashlib.sha256(transformed_text).hexdigest()
        != _INRANGE_DX_OUTPUT_TEXT_SHA256
    ):
        return cubin
    return transformed


# The exact a3/461 recurrence copies unchanged uniform predicate UP0 to P3 at
# PC 0x1a20 solely for the PC 0x1a50 valid3 PLOP3.  SM90 can consume UP0 in
# that same source position directly.  Rotate only dependency-independent
# operations into the freed slot, move the terminal R23 update after all its
# inputs are final, and shorten the backedge by one instruction.  This is
# fail-closed on the complete parent/output fused text, recurrent text, and
# every moved instruction; all nonmatching specializations remain unchanged.
_DIRECT_VALID3_PARENT_TEXT_SHA256 = (
    "9a0f99f7a8bd1eb110038c750b4cadf0c47f589759bf471883e44c0de82c56b1"
)
_DIRECT_VALID3_PARENT_LOOP_SHA256 = (
    "4626701ab9fa326a8d20705d6f268e80425b91ee9813c2cf3d7c9e0a5618fbf7"
)
_DIRECT_VALID3_OUTPUT_TEXT_SHA256 = (
    "efc3080d91faaa3221a336e34feec09fa4933eb7a4f17160f013aded89d0ad25"
)
_DIRECT_VALID3_OUTPUT_LOOP_SHA256 = (
    "415f7a2556230293e36ddd61944e643a875f4695eac10d45421c1cc5989dcca0"
)
_DIRECT_VALID3_COPY_UP0_TO_P3 = _sm90_instruction(
    0x000000000000781C,
    0x000FC60003F6F008,
)
_DIRECT_VALID3_PARENT_COMBINE = _sm90_instruction(
    0x000000000000781C,
    0x000FE40000F60830,
)
_DIRECT_VALID3_UNIFORM_COMBINE = _sm90_instruction(
    0x000000000000781C,
    0x000FE40000F60808,
)
_DIRECT_VALID3_MASK_R30 = _sm90_instruction(
    0xFFFF00001E1E7812,
    0x000FD000078EC0FF,
)
_DIRECT_VALID3_MASK_R31 = _sm90_instruction(
    0xFFFF00001F1F7812,
    0x000FC400078EC0FF,
)
_DIRECT_VALID3_BOS3_EXTRACT = _sm90_instruction(
    0x00000100120F7892,
    0x000FE2000F8EC03F,
)
_DIRECT_VALID3_P1_INVERT = _sm90_instruction(
    0x000000000000781C,
    0x000FE20000F2E170,
)
_DIRECT_VALID3_R23_TERMINAL = _sm90_instruction(
    0x000000112C177223,
    0x000FE20000010017,
)
_DIRECT_VALID3_R38_TERMINAL = _sm90_instruction(
    0x0000002E12267223,
    0x004FE2000001003B,
)
_DIRECT_VALID3_R25_TERMINAL = _sm90_instruction(
    0x0000002D11197223,
    0x000FE20000010020,
)
_DIRECT_VALID3_PARENT_BRANCH = _sm90_instruction(
    0xFFFFFFEC00F8B947,
    0x000FF0000383FFFF,
)
_DIRECT_VALID3_COMPACT_BRANCH = _sm90_instruction(
    0xFFFFFFEC00FCB947,
    0x000FF0000383FFFF,
)


def _apply_direct_valid3_peephole(cubin):
    sections = _elf64_sections(cubin)
    text_entry = sections.get(_NARROW_LOOP_TEXT_SECTION)
    if text_entry is None:
        return cubin
    text_offset, text_size = text_entry
    if text_size != 0x4A00 or text_offset + text_size > len(cubin):
        return cubin
    parent_text = cubin[text_offset : text_offset + text_size]
    if (
        hashlib.sha256(parent_text).hexdigest()
        != _DIRECT_VALID3_PARENT_TEXT_SHA256
    ):
        return cubin
    parent_loop = parent_text[0x1560:0x2580]
    if (
        hashlib.sha256(parent_loop).hexdigest()
        != _DIRECT_VALID3_PARENT_LOOP_SHA256
    ):
        return cubin

    def read_pc(pc):
        return cubin[text_offset + pc : text_offset + pc + 16]

    expected = {
        0x1A20: _DIRECT_VALID3_COPY_UP0_TO_P3,
        0x1A50: _DIRECT_VALID3_PARENT_COMBINE,
        0x1A90: _DIRECT_VALID3_MASK_R30,
        0x1B30: _DIRECT_VALID3_MASK_R31,
        0x1E70: _DIRECT_VALID3_BOS3_EXTRACT,
        0x2360: _DIRECT_VALID3_P1_INVERT,
        0x2540: _DIRECT_VALID3_R23_TERMINAL,
        0x2550: _DIRECT_VALID3_R38_TERMINAL,
        0x2560: _DIRECT_VALID3_R25_TERMINAL,
        0x2570: _DIRECT_VALID3_PARENT_BRANCH,
        0x2580: _NARROW_LOOP_NOP,
    }
    if any(read_pc(pc) != encoded for pc, encoded in expected.items()):
        return cubin

    output = bytearray(cubin)

    def write_pc(pc, encoded):
        output[text_offset + pc : text_offset + pc + 16] = encoded

    write_pc(0x1A20, read_pc(0x1A90))
    write_pc(0x1A50, _DIRECT_VALID3_UNIFORM_COMBINE)
    write_pc(0x1A90, read_pc(0x1B30))
    write_pc(0x1B30, read_pc(0x1E70))
    write_pc(0x1E70, read_pc(0x2360))
    write_pc(0x2360, read_pc(0x2540))
    write_pc(0x2540, read_pc(0x2550))
    write_pc(0x2550, read_pc(0x2560))
    write_pc(0x2560, _DIRECT_VALID3_COMPACT_BRANCH)
    write_pc(0x2570, _NARROW_LOOP_NOP)

    transformed = bytes(output)
    transformed_text = transformed[text_offset : text_offset + text_size]
    transformed_loop = transformed_text[0x1560:0x2570]
    if (
        hashlib.sha256(transformed_text).hexdigest()
        != _DIRECT_VALID3_OUTPUT_TEXT_SHA256
        or hashlib.sha256(transformed_loop).hexdigest()
        != _DIRECT_VALID3_OUTPUT_LOOP_SHA256
    ):
        return cubin
    changed_pcs = [
        pc
        for pc in range(0, text_size, 0x10)
        if parent_text[pc : pc + 16]
        != transformed_text[pc : pc + 16]
    ]
    if changed_pcs != [
        0x1A20,
        0x1A50,
        0x1A90,
        0x1B30,
        0x1E70,
        0x2360,
        0x2540,
        0x2550,
        0x2560,
        0x2570,
    ]:
        return cubin
    return transformed


# The finalized a6/283 recurrence retains the hot row-0 predicate conversion
# at PC 0x15d0.  Test only the alternate SM90 scheduler-yield preference while
# keeping the exact four-cycle dependency gap, predicate value, and following
# FSEL chain.  Fail closed on the complete post-a6/283 fused text and loop so
# every other specialization remains byte-for-byte unchanged.
_PREDICATE_YIELD_PARENT_TEXT_SHA256 = (
    "efc3080d91faaa3221a336e34feec09fa4933eb7a4f17160f013aded89d0ad25"
)
_PREDICATE_YIELD_PARENT_LOOP_SHA256 = (
    "415f7a2556230293e36ddd61944e643a875f4695eac10d45421c1cc5989dcca0"
)
_PREDICATE_YIELD_OUTPUT_TEXT_SHA256 = (
    "d2ba5c85bc0c499dd9fc55d2d4801c072b29fbf469e62b70b09078d280bbf617"
)
_PREDICATE_YIELD_OUTPUT_LOOP_SHA256 = (
    "b0a3204c96e513dc55d285b20d939c430494339fe611c548e6a9225c324fda06"
)
_PREDICATE_YIELD_PC = 0x15D0
_PREDICATE_YIELD_PARENT = _sm90_instruction(
    0x000000000000781C,
    0x000FC80003F6F008,
)
_PREDICATE_YIELD_ALTERNATE = _sm90_instruction(
    0x000000000000781C,
    0x000FE80003F6F008,
)


def _apply_predicate_yield_peephole(cubin):
    sections = _elf64_sections(cubin)
    text_entry = sections.get(_NARROW_LOOP_TEXT_SECTION)
    if text_entry is None:
        return cubin
    text_offset, text_size = text_entry
    if text_size != 0x4A00 or text_offset + text_size > len(cubin):
        return cubin
    parent_text = cubin[text_offset : text_offset + text_size]
    if hashlib.sha256(parent_text).hexdigest() != (
        _PREDICATE_YIELD_PARENT_TEXT_SHA256
    ):
        return cubin
    parent_loop = parent_text[0x1560:0x2570]
    if hashlib.sha256(parent_loop).hexdigest() != (
        _PREDICATE_YIELD_PARENT_LOOP_SHA256
    ):
        return cubin

    instruction_offset = text_offset + _PREDICATE_YIELD_PC
    if cubin[instruction_offset : instruction_offset + 16] != (
        _PREDICATE_YIELD_PARENT
    ):
        return cubin

    output = bytearray(cubin)
    output[instruction_offset : instruction_offset + 16] = (
        _PREDICATE_YIELD_ALTERNATE
    )
    transformed = bytes(output)
    transformed_text = transformed[text_offset : text_offset + text_size]
    transformed_loop = transformed_text[0x1560:0x2570]
    if (
        hashlib.sha256(transformed_text).hexdigest()
        != _PREDICATE_YIELD_OUTPUT_TEXT_SHA256
        or hashlib.sha256(transformed_loop).hexdigest()
        != _PREDICATE_YIELD_OUTPUT_LOOP_SHA256
    ):
        return cubin
    return transformed


# The exact a1/473 recurrence keeps one packed scalar in R55 (bank 3) from
# its PRMT definition at PC 0x1950 through three consumers.  R1 is already
# declared by the cubin but its entry LDC value is never read.  Remap only
# that complete live range to R1 (bank 1), removing the R55/R59 and R3/R55
# operand-bank coincidences while preserving REG63 and every instruction's
# opcode, control word, PC, and dependency distance.  Fail closed on exact
# executable text, recurrence, metadata, and all four parent encodings.
_R1_BANK_PARENT_TEXT_SHA256 = (
    "d2ba5c85bc0c499dd9fc55d2d4801c072b29fbf469e62b70b09078d280bbf617"
)
_R1_BANK_PARENT_LOOP_SHA256 = (
    "b0a3204c96e513dc55d285b20d939c430494339fe611c548e6a9225c324fda06"
)
_R1_BANK_OUTPUT_TEXT_SHA256 = (
    "77ca6abbdb026e30f7e6964a2132b20c4a1536e5a724da55cb36d5928c0ba1f6"
)
_R1_BANK_OUTPUT_LOOP_SHA256 = (
    "5bc66cbcc1384c060e3e2f3e1177c307ce8be51a21eb777ec3bf2978bf318611"
)
_R1_BANK_NVINFO_SHA256 = (
    "cb7dad07866f1dbeed3ea1ff4dece472eb858b8958cd3f064ada1d7cfcd82552"
)
_R1_BANK_KERNEL_NVINFO_SHA256 = (
    "13c7d6092b8f78555e5d2982286153299ecb94bbd746ab9b052b052469241f37"
)
_R1_BANK_PATCHES = {
    0x1950: (
        _sm90_instruction(0x000010441E377816, 0x004FCA00000000FF),
        _sm90_instruction(0x000010441E017816, 0x004FCA00000000FF),
    ),
    0x1960: (
        _sm90_instruction(0x0000003708007223, 0x000FC8000001003B),
        _sm90_instruction(0x0000000108007223, 0x000FC8000001003B),
    ),
    0x1BC0: (
        _sm90_instruction(0x000000FF372B7208, 0x000FC40004800000),
        _sm90_instruction(0x000000FF012B7208, 0x000FC40004800000),
    ),
    0x1C20: (
        _sm90_instruction(0x0000003703327223, 0x000FE40000010032),
        _sm90_instruction(0x0000000103327223, 0x000FE40000010032),
    ),
}


def _apply_r1_bank_remap_peephole(cubin):
    sections = _elf64_sections(cubin)
    text_entry = sections.get(_NARROW_LOOP_TEXT_SECTION)
    nvinfo_entry = sections.get(".nv.info")
    kernel_nvinfo_entry = sections.get(
        ".nv.info._fused_chunk_bwd_kernel"
    )
    if (
        text_entry is None
        or nvinfo_entry is None
        or kernel_nvinfo_entry is None
    ):
        return cubin
    text_offset, text_size = text_entry
    nvinfo_offset, nvinfo_size = nvinfo_entry
    kernel_nvinfo_offset, kernel_nvinfo_size = kernel_nvinfo_entry
    if (
        text_size != 0x4A00
        or text_offset + text_size > len(cubin)
        or nvinfo_offset + nvinfo_size > len(cubin)
        or kernel_nvinfo_offset + kernel_nvinfo_size > len(cubin)
    ):
        return cubin

    parent_text = cubin[text_offset : text_offset + text_size]
    parent_loop = parent_text[0x1560:0x2570]
    parent_nvinfo = cubin[nvinfo_offset : nvinfo_offset + nvinfo_size]
    parent_kernel_nvinfo = cubin[
        kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
    ]
    if (
        hashlib.sha256(parent_text).hexdigest()
        != _R1_BANK_PARENT_TEXT_SHA256
        or hashlib.sha256(parent_loop).hexdigest()
        != _R1_BANK_PARENT_LOOP_SHA256
        or hashlib.sha256(parent_nvinfo).hexdigest()
        != _R1_BANK_NVINFO_SHA256
        or hashlib.sha256(parent_kernel_nvinfo).hexdigest()
        != _R1_BANK_KERNEL_NVINFO_SHA256
        or parent_nvinfo[8:12] != (63).to_bytes(4, "little")
    ):
        return cubin

    def read_pc(pc):
        return cubin[text_offset + pc : text_offset + pc + 16]

    if any(
        read_pc(pc) != before
        for pc, (before, _after) in _R1_BANK_PATCHES.items()
    ):
        return cubin

    output = bytearray(cubin)
    for pc, (before, after) in _R1_BANK_PATCHES.items():
        if before[8:] != after[8:]:
            return cubin
        output[text_offset + pc : text_offset + pc + 16] = after
    transformed = bytes(output)
    transformed_text = transformed[text_offset : text_offset + text_size]
    transformed_loop = transformed_text[0x1560:0x2570]
    if (
        hashlib.sha256(transformed_text).hexdigest()
        != _R1_BANK_OUTPUT_TEXT_SHA256
        or hashlib.sha256(transformed_loop).hexdigest()
        != _R1_BANK_OUTPUT_LOOP_SHA256
        or transformed[nvinfo_offset : nvinfo_offset + nvinfo_size]
        != parent_nvinfo
        or transformed[
            kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
        ]
        != parent_kernel_nvinfo
    ):
        return cubin
    changed_pcs = [
        pc
        for pc in range(0, text_size, 0x10)
        if parent_text[pc : pc + 16]
        != transformed_text[pc : pc + 16]
    ]
    if changed_pcs != [0x1950, 0x1960, 0x1BC0, 0x1C20]:
        return cubin
    return transformed


# The packed stress specialization benefits from staging the two BOS flags
# preceding each 256-token chunk in the CTA's existing shared-memory transfer.
# That source change perturbs ptxas scheduling and register allocation, so the
# exact recurrence optimizations above intentionally fail closed.  Recompose
# their semantic transforms for this one exact raw SM90 text: narrow the dead
# high loop induction, carry the dx pointer directly in R48:R49, consume UP0
# directly for valid3, retain the measured row-predicate yield preference, and
# remap the short packed R55 live range to otherwise-dead declared R1.  The
# complete input/intermediate/output executable hashes and unchanged metadata
# make every mismatch fall back to ptxas output.
_HALO_PARENT_TEXT_SHA256 = (
    "52492a94e1ded1a5149fd9e20f15cf8c01858877060e040609cefe7af8067ed5"
)
_HALO_NARROW_TEXT_SHA256 = (
    "955d67ac7e8efa94184c9b4bb4d021e5295ed62e03963f60570c12eff7917834"
)
_HALO_INRANGE_TEXT_SHA256 = (
    "2640b3a83cd567ca2e7f463cfd0cc05f0ddbf98167adf19c9852aa34238af591"
)
_HALO_DIRECT_TEXT_SHA256 = (
    "af41a696a2e6ca0ffb49d42c20b48432c68796502eb4bf933135c395e8b12e57"
)
_HALO_OUTPUT_TEXT_SHA256 = (
    "ad7a23739921c1560e6fc7049ff1ce147c6fb0f96968b469424447ea18d527cc"
)
_HALO_NVINFO_SHA256 = (
    "cb7dad07866f1dbeed3ea1ff4dece472eb858b8958cd3f064ada1d7cfcd82552"
)
_HALO_KERNEL_NVINFO_SHA256 = (
    "e4dbc0dab44838061d445808b157f92d6197d3d6988b064a92b1d9a2798d1773"
)
_HALO_BRANCH_RAW = _sm90_instruction(
    0xFFFFFFEC00E8B947,
    0x000FF6000383FFFF,
)
_HALO_BRANCH_NARROW = _sm90_instruction(
    0xFFFFFFEC00F0B947,
    0x000FF6000383FFFF,
)
_HALO_BRANCH_INRANGE = _sm90_instruction(
    0xFFFFFFEC00F8B947,
    0x000FF6000383FFFF,
)
_HALO_BRANCH_DIRECT = _sm90_instruction(
    0xFFFFFFEC00FCB947,
    0x000FF6000383FFFF,
)


def _apply_halo_composition_peephole(cubin):
    sections = _elf64_sections(cubin)
    text_entry = sections.get(_NARROW_LOOP_TEXT_SECTION)
    nvinfo_entry = sections.get(".nv.info")
    kernel_nvinfo_entry = sections.get(
        ".nv.info._fused_chunk_bwd_kernel"
    )
    if (
        text_entry is None
        or nvinfo_entry is None
        or kernel_nvinfo_entry is None
    ):
        return cubin
    text_offset, text_size = text_entry
    nvinfo_offset, nvinfo_size = nvinfo_entry
    kernel_nvinfo_offset, kernel_nvinfo_size = kernel_nvinfo_entry
    if (
        text_size != 0x4A00
        or text_offset + text_size > len(cubin)
        or nvinfo_offset + nvinfo_size > len(cubin)
        or kernel_nvinfo_offset + kernel_nvinfo_size > len(cubin)
    ):
        return cubin

    parent_text = cubin[text_offset : text_offset + text_size]
    parent_nvinfo = cubin[nvinfo_offset : nvinfo_offset + nvinfo_size]
    parent_kernel_nvinfo = cubin[
        kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
    ]
    if (
        hashlib.sha256(parent_text).hexdigest()
        != _HALO_PARENT_TEXT_SHA256
        or hashlib.sha256(parent_nvinfo).hexdigest()
        != _HALO_NVINFO_SHA256
        or hashlib.sha256(parent_kernel_nvinfo).hexdigest()
        != _HALO_KERNEL_NVINFO_SHA256
        or parent_nvinfo[8:12] != (63).to_bytes(4, "little")
    ):
        return cubin

    output = bytearray(cubin)

    def read_pc(pc):
        start = text_offset + pc
        return bytes(output[start : start + 16])

    def write_pc(pc, encoded):
        if len(encoded) != 16:
            return False
        start = text_offset + pc
        output[start : start + 16] = encoded
        return True

    def text_hash():
        return hashlib.sha256(
            output[text_offset : text_offset + text_size]
        ).hexdigest()

    def compact_loop(start, branch, removed):
        sequence = [
            read_pc(pc)
            for pc in range(start, branch + 0x10, 0x10)
            if pc not in removed
        ]
        for index, encoded in enumerate(sequence):
            write_pc(start + index * 0x10, encoded)
        new_branch = start + (len(sequence) - 1) * 0x10
        for pc in range(new_branch + 0x10, branch + 0x10, 0x10):
            write_pc(pc, _NARROW_LOOP_NOP)
        return new_branch

    # Delete the dead high-half uniform add/compare, filling their issue slots
    # with two already-ready terminal reductions.
    if (
        read_pc(0x2360) != _NARROW_LOOP_HIGH_ADD
        or read_pc(0x23A0) != _NARROW_LOOP_HIGH_COMPARE
        or read_pc(0x25A0) != _HALO_BRANCH_RAW
    ):
        return cubin
    write_pc(0x2360, read_pc(0x24B0))
    write_pc(0x23A0, read_pc(0x2530))
    if compact_loop(0x1550, 0x25A0, {0x24B0, 0x2530}) != 0x2580:
        return cubin
    write_pc(0x2580, _HALO_BRANCH_NARROW)
    if text_hash() != _HALO_NARROW_TEXT_SHA256:
        return cubin

    # R49's only recurrent tid.x use is equivalent to the low five bits of
    # R14.  Reuse R49 as the adjacent high dx pointer, gather the four stores,
    # and move pointer advances into their vacated late slots.
    expected_inrange = {
        0x11A0: _sm90_instruction(
            0x0000000FFF2F7C10,
            0x000FE4000B7EA41D,
        ),
        0x1700: _sm90_instruction(
            0x0000000F31FF7812,
            0x000FE2000788C0FF,
        ),
        0x2340: _sm90_instruction(
            0x0000002F00237202,
            0x000FE40000000F00,
        ),
        0x2390: _sm90_instruction(
            0x0000003000227202,
            0x000FC60000000F00,
        ),
    }
    if any(
        read_pc(pc) != encoded
        for pc, encoded in expected_inrange.items()
    ):
        return cubin
    p1_invert = read_pc(0x2540)
    p0_invert = read_pc(0x2560)
    write_pc(
        0x11A0,
        _sm90_instruction(0x0000000FFF317C10, 0x000FE4000B7EA41D),
    )
    write_pc(
        0x1700,
        _sm90_instruction(0x0000001E0EFF7812, 0x000FE2000788C0FF),
    )
    write_pc(0x2340, p1_invert)
    write_pc(0x2390, p0_invert)
    write_pc(
        0x23B0,
        _sm90_instruction(0xFFA000183000A986, 0x0003E2000C101910),
    )
    write_pc(
        0x23C0,
        _sm90_instruction(0xFFE0001F3000A986, 0x000FE8000C101910),
    )
    write_pc(
        0x23D0,
        _sm90_instruction(0x000000353000A986, 0x000FE2000C101910),
    )
    write_pc(
        0x23E0,
        _sm90_instruction(0xFFC000263000A986, 0x0005E2000C101910),
    )
    write_pc(
        0x2450,
        _sm90_instruction(0x0000800030307810, 0x000FC60007F9E0FF),
    )
    write_pc(
        0x2490,
        _sm90_instruction(0x00000031FF317210, 0x000FE400027FE4FF),
    )
    if compact_loop(0x1550, 0x2580, {0x2540, 0x2560}) != 0x2560:
        return cubin
    write_pc(0x2560, _HALO_BRANCH_INRANGE)
    if text_hash() != _HALO_INRANGE_TEXT_SHA256:
        return cubin

    # Replace the redundant UP0-to-P3 copy with direct uniform consumption and
    # rotate only dependency-independent masks/extracts into the freed slots.
    if (
        read_pc(0x1A10) != _DIRECT_VALID3_COPY_UP0_TO_P3
        or read_pc(0x1A40) != _DIRECT_VALID3_PARENT_COMBINE
        or read_pc(0x2560) != _HALO_BRANCH_INRANGE
    ):
        return cubin
    mask_r30 = read_pc(0x1A90)
    mask_r31 = read_pc(0x1B50)
    bos3_extract = read_pc(0x1E20)
    p1_invert = read_pc(0x2340)
    terminal_r22 = read_pc(0x2540)
    terminal_r21 = read_pc(0x2550)
    write_pc(0x1A10, mask_r30)
    write_pc(0x1A40, _DIRECT_VALID3_UNIFORM_COMBINE)
    write_pc(0x1A90, mask_r31)
    write_pc(0x1B50, bos3_extract)
    write_pc(0x1E20, p1_invert)
    write_pc(0x2340, terminal_r22)
    write_pc(0x2540, terminal_r21)
    write_pc(0x2550, _HALO_BRANCH_DIRECT)
    write_pc(0x2560, _NARROW_LOOP_NOP)
    if text_hash() != _HALO_DIRECT_TEXT_SHA256:
        return cubin

    # Preserve the finalized scheduler preference and move the complete short
    # packed live range from bank-3 R55 to declared, dead-on-entry bank-1 R1.
    if read_pc(0x15C0) != _PREDICATE_YIELD_PARENT:
        return cubin
    write_pc(0x15C0, _PREDICATE_YIELD_ALTERNATE)
    r1_patches = {
        0x1930: (
            _sm90_instruction(0x000010441E377816, 0x004FCA00000000FF),
            _sm90_instruction(0x000010441E017816, 0x004FCA00000000FF),
        ),
        0x1940: (
            _sm90_instruction(0x0000003708007223, 0x000FC8000001003B),
            _sm90_instruction(0x0000000108007223, 0x000FC8000001003B),
        ),
        0x1BC0: (
            _sm90_instruction(0x000000FF371F7208, 0x000FE20004800000),
            _sm90_instruction(0x000000FF011F7208, 0x000FE20004800000),
        ),
        0x1C10: (
            _sm90_instruction(0x0000003703327223, 0x000FE20000010032),
            _sm90_instruction(0x0000000103327223, 0x000FE20000010032),
        ),
    }
    for pc, (before, after) in r1_patches.items():
        if read_pc(pc) != before:
            return cubin
        write_pc(pc, after)

    transformed = bytes(output)
    transformed_text = transformed[text_offset : text_offset + text_size]
    if (
        hashlib.sha256(transformed_text).hexdigest()
        != _HALO_OUTPUT_TEXT_SHA256
        or transformed[nvinfo_offset : nvinfo_offset + nvinfo_size]
        != parent_nvinfo
        or transformed[
            kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
        ]
        != parent_kernel_nvinfo
    ):
        return cubin
    return transformed


# The halo transfer already stages the current chunk's BOS bytes at shared
# offsets 16..271.  For the exact finalized halo executable, redirect only the
# four once-per-CTA prologue byte loads to offsets 16..19.  Keep their original
# destinations and scheduler/barrier controls independent; the recurrent loop
# continues to use its finalized pairs of shared halfword loads.
_PROLOGUE_BOS_SHARED_PARENT_TEXT_SHA256 = (
    "ad7a23739921c1560e6fc7049ff1ce147c6fb0f96968b469424447ea18d527cc"
)
_PROLOGUE_BOS_SHARED_OUTPUT_TEXT_SHA256 = (
    "8b31e01ccc950a32c69484a4d41d3d5fc270d9c4c79f8e5b85f01bdeb9adea01"
)


def _apply_prologue_bos_shared_peephole(cubin):
    sections = _elf64_sections(cubin)
    text_entry = sections.get(_NARROW_LOOP_TEXT_SECTION)
    nvinfo_entry = sections.get(".nv.info")
    kernel_nvinfo_entry = sections.get(
        ".nv.info._fused_chunk_bwd_kernel"
    )
    if (
        text_entry is None
        or nvinfo_entry is None
        or kernel_nvinfo_entry is None
    ):
        return cubin
    text_offset, text_size = text_entry
    nvinfo_offset, nvinfo_size = nvinfo_entry
    kernel_nvinfo_offset, kernel_nvinfo_size = kernel_nvinfo_entry
    if (
        text_size != 0x4A00
        or text_offset + text_size > len(cubin)
        or nvinfo_offset + nvinfo_size > len(cubin)
        or kernel_nvinfo_offset + kernel_nvinfo_size > len(cubin)
    ):
        return cubin

    parent_text = cubin[text_offset : text_offset + text_size]
    parent_nvinfo = cubin[nvinfo_offset : nvinfo_offset + nvinfo_size]
    parent_kernel_nvinfo = cubin[
        kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
    ]
    if (
        hashlib.sha256(parent_text).hexdigest()
        != _PROLOGUE_BOS_SHARED_PARENT_TEXT_SHA256
        or hashlib.sha256(parent_nvinfo).hexdigest()
        != _HALO_NVINFO_SHA256
        or hashlib.sha256(parent_kernel_nvinfo).hexdigest()
        != _HALO_KERNEL_NVINFO_SHA256
        or parent_nvinfo[8:12] != (63).to_bytes(4, "little")
    ):
        return cubin

    patches = {
        0x0490: (
            _sm90_instruction(
                0x000000100A207981,
                0x000F68000C1E1100,
            ),
            _sm90_instruction(
                0x00001008FF207984,
                0x000F680008000000,
            ),
        ),
        0x0540: (
            _sm90_instruction(
                0x000002100A2A7981,
                0x000F62000C1E1100,
            ),
            _sm90_instruction(
                0x00001208FF2A7984,
                0x000F620008000000,
            ),
        ),
        0x0580: (
            _sm90_instruction(
                0x00000110082B7981,
                0x000362000C1E1100,
            ),
            _sm90_instruction(
                0x00001108FF2B7984,
                0x0003620008000000,
            ),
        ),
        0x0610: (
            _sm90_instruction(
                0x0000031024297981,
                0x000368000C1E1100,
            ),
            _sm90_instruction(
                0x00001308FF297984,
                0x0003680008000000,
            ),
        ),
    }
    output = bytearray(cubin)
    for pc, (before, after) in patches.items():
        start = text_offset + pc
        if bytes(output[start : start + 16]) != before:
            return cubin
        output[start : start + 16] = after

    transformed = bytes(output)
    transformed_text = transformed[text_offset : text_offset + text_size]
    changed_pcs = [
        pc
        for pc in range(0, text_size, 0x10)
        if parent_text[pc : pc + 16]
        != transformed_text[pc : pc + 16]
    ]
    if (
        changed_pcs != sorted(patches)
        or hashlib.sha256(transformed_text).hexdigest()
        != _PROLOGUE_BOS_SHARED_OUTPUT_TEXT_SHA256
        or transformed[nvinfo_offset : nvinfo_offset + nvinfo_size]
        != parent_nvinfo
        or transformed[
            kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
        ]
        != parent_kernel_nvinfo
    ):
        return cubin
    return transformed


# In the exact finalized REG63 recurrence, the last packed row materializes
# src3 = keep ? R0 : 0 in R41 and consumes it in only two FMAs.  Reuse the
# already-live row predicate directly, redirect the last-row gradient producer
# into loop-carried R25, and remove the now-dead FSEL.  Compact complete
# instruction/control words through the backedge; every other specialization
# fails closed.
_DELETE_LAST_SRC3_FSEL_PARENT_TEXT_SHA256 = (
    "8b31e01ccc950a32c69484a4d41d3d5fc270d9c4c79f8e5b85f01bdeb9adea01"
)
_DELETE_LAST_SRC3_FSEL_OUTPUT_TEXT_SHA256 = (
    "9850d602b289d658efe0fa83123a61f8b6c93ff7aafa9674a61c782a22be4df8"
)
_DELETE_LAST_SRC3_FSEL_NOP = _sm90_instruction(
    0x0000000000007918,
    0x000FC00000000000,
)
_DELETE_LAST_SRC3_FSEL_PARENT = _sm90_instruction(
    0x000000FF00297208,
    0x000FE20004000000,
)
_DELETE_LAST_SRC3_FSEL_Z_PARENT = _sm90_instruction(
    0x0000002905207223,
    0x000FE20000010020,
)
_DELETE_LAST_SRC3_FSEL_Z_PREDICATED = _sm90_instruction(
    0x0000000005208223,
    0x000FE20000010020,
)
_DELETE_LAST_SRC3_FSEL_G_PARENT = _sm90_instruction(
    0x0000003A101E7223,
    0x000FE20000010037,
)
_DELETE_LAST_SRC3_FSEL_G_TO_R25 = _sm90_instruction(
    0x0000003A10197223,
    0x000FE20000010037,
)
_DELETE_LAST_SRC3_FSEL_DW_PARENT = _sm90_instruction(
    0x0000001229197223,
    0x000FE2000001001E,
)
_DELETE_LAST_SRC3_FSEL_DW_PREDICATED = _sm90_instruction(
    0x0000001200190223,
    0x000FE20000010019,
)
_DELETE_LAST_SRC3_FSEL_DX_PARENT = _sm90_instruction(
    0x000000190F207223,
    0x040FE20000010032,
)
_DELETE_LAST_SRC3_FSEL_BRANCH_PARENT = _sm90_instruction(
    0xFFFFFFEC00FCB947,
    0x000FF6000383FFFF,
)
_DELETE_LAST_SRC3_FSEL_BRANCH_COMPACT = _sm90_instruction(
    0xFFFFFFF00000B947,
    0x000FF6000383FFFF,
)


def _apply_delete_last_src3_fsel_peephole(cubin):
    sections = _elf64_sections(cubin)
    text_entry = sections.get(_NARROW_LOOP_TEXT_SECTION)
    nvinfo_entry = sections.get(".nv.info")
    kernel_nvinfo_entry = sections.get(
        ".nv.info._fused_chunk_bwd_kernel"
    )
    if (
        text_entry is None
        or nvinfo_entry is None
        or kernel_nvinfo_entry is None
    ):
        return cubin
    text_offset, text_size = text_entry
    nvinfo_offset, nvinfo_size = nvinfo_entry
    kernel_nvinfo_offset, kernel_nvinfo_size = kernel_nvinfo_entry
    if (
        text_size != 0x4A00
        or text_offset + text_size > len(cubin)
        or nvinfo_offset + nvinfo_size > len(cubin)
        or kernel_nvinfo_offset + kernel_nvinfo_size > len(cubin)
    ):
        return cubin

    parent_text = cubin[text_offset : text_offset + text_size]
    parent_nvinfo = cubin[nvinfo_offset : nvinfo_offset + nvinfo_size]
    parent_kernel_nvinfo = cubin[
        kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
    ]
    if (
        hashlib.sha256(parent_text).hexdigest()
        != _DELETE_LAST_SRC3_FSEL_PARENT_TEXT_SHA256
        or hashlib.sha256(parent_nvinfo).hexdigest()
        != _HALO_NVINFO_SHA256
        or hashlib.sha256(parent_kernel_nvinfo).hexdigest()
        != _HALO_KERNEL_NVINFO_SHA256
        or parent_nvinfo[8:12] != (63).to_bytes(4, "little")
    ):
        return cubin

    def read_pc(pc):
        return parent_text[pc : pc + 16]

    expected = {
        0x1F40: _DELETE_LAST_SRC3_FSEL_PARENT,
        0x1F90: _DELETE_LAST_SRC3_FSEL_Z_PARENT,
        0x20D0: _DELETE_LAST_SRC3_FSEL_G_PARENT,
        0x23A0: _DELETE_LAST_SRC3_FSEL_DW_PARENT,
        0x2430: _DELETE_LAST_SRC3_FSEL_DX_PARENT,
        0x2550: _DELETE_LAST_SRC3_FSEL_BRANCH_PARENT,
    }
    if any(read_pc(pc) != encoded for pc, encoded in expected.items()):
        return cubin

    replacements = {
        0x1F90: _DELETE_LAST_SRC3_FSEL_Z_PREDICATED,
        0x20D0: _DELETE_LAST_SRC3_FSEL_G_TO_R25,
        0x23A0: _DELETE_LAST_SRC3_FSEL_DW_PREDICATED,
    }
    compacted = [
        replacements.get(pc, read_pc(pc))
        for pc in range(0x1F50, 0x2560, 0x10)
    ]
    if (
        len(compacted) != 97
        or compacted[-1] != _DELETE_LAST_SRC3_FSEL_BRANCH_PARENT
    ):
        return cubin
    compacted[-1] = _DELETE_LAST_SRC3_FSEL_BRANCH_COMPACT

    output = bytearray(cubin)
    start = text_offset + 0x1F40
    output[start : start + len(compacted) * 16] = b"".join(compacted)
    output[text_offset + 0x2550 : text_offset + 0x2560] = (
        _DELETE_LAST_SRC3_FSEL_NOP
    )
    transformed = bytes(output)
    transformed_text = transformed[text_offset : text_offset + text_size]
    if (
        hashlib.sha256(transformed_text).hexdigest()
        != _DELETE_LAST_SRC3_FSEL_OUTPUT_TEXT_SHA256
        or parent_text[:0x1F40] != transformed_text[:0x1F40]
        or parent_text[0x2560:] != transformed_text[0x2560:]
        or transformed[nvinfo_offset : nvinfo_offset + nvinfo_size]
        != parent_nvinfo
        or transformed[
            kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
        ]
        != parent_kernel_nvinfo
    ):
        return cubin
    return transformed


# After the exact src3 deletion above, the adjacent last-row src2 value still
# materializes as R54 = keep ? R30 : 0 for two FFMA consumers.  Predicate both
# consumers on the already-live original-polarity P0 and keep raw R30 alive by
# moving its intervening SiLU temporary chain into now-dead R54.  Compact one
# complete instruction/control word through the backedge.  Exact text,
# metadata, resources, encodings, and output text all fail closed.
_DELETE_LAST_SRC2_FSEL_PARENT_TEXT_SHA256 = (
    "9850d602b289d658efe0fa83123a61f8b6c93ff7aafa9674a61c782a22be4df8"
)
_DELETE_LAST_SRC2_FSEL_OUTPUT_TEXT_SHA256 = (
    "ce0f9326c9a80069df0defe10c1881f3c71293a06042d23b34f2b375347e03e6"
)
_DELETE_LAST_SRC2_FSEL_PARENT = _sm90_instruction(
    0x000000FF1E367208,
    0x000FE20004000000,
)
_DELETE_LAST_SRC2_FSEL_FORWARD_PARENT = _sm90_instruction(
    0x000000360A1F7223,
    0x000FE20000010007,
)
_DELETE_LAST_SRC2_FSEL_FORWARD_PREDICATED = _sm90_instruction(
    0x0000001E0A1F8223,
    0x000FE20000010007,
)
_DELETE_LAST_SRC2_FSEL_SILU_0_PARENT = _sm90_instruction(
    0x3F000000201E7823,
    0x000FC600000100FF,
)
_DELETE_LAST_SRC2_FSEL_SILU_0_TO_R54 = _sm90_instruction(
    0x3F00000020367823,
    0x000FC600000100FF,
)
_DELETE_LAST_SRC2_FSEL_SILU_1_PARENT = _sm90_instruction(
    0x0000001E001E7308,
    0x000E620000002400,
)
_DELETE_LAST_SRC2_FSEL_SILU_1_TO_R54 = _sm90_instruction(
    0x0000003600367308,
    0x000E620000002400,
)
_DELETE_LAST_SRC2_FSEL_SILU_2_PARENT = _sm90_instruction(
    0x3F0000001E397423,
    0x002FC60000010827,
)
_DELETE_LAST_SRC2_FSEL_SILU_2_FROM_R54 = _sm90_instruction(
    0x3F00000036397423,
    0x002FC60000010827,
)
_DELETE_LAST_SRC2_FSEL_SILU_3_PARENT = _sm90_instruction(
    0x3F0000001E207423,
    0x080FE20000010027,
)
_DELETE_LAST_SRC2_FSEL_SILU_3_FROM_R54 = _sm90_instruction(
    0x3F00000036207423,
    0x080FE20000010027,
)
_DELETE_LAST_SRC2_FSEL_DW_PARENT = _sm90_instruction(
    0x0000003611287223,
    0x000FE20000010028,
)
_DELETE_LAST_SRC2_FSEL_DW_PREDICATED = _sm90_instruction(
    0x0000001E11288223,
    0x000FE20000010028,
)
_DELETE_LAST_SRC2_FSEL_BRANCH_PARENT = _sm90_instruction(
    0xFFFFFFF00000B947,
    0x000FF6000383FFFF,
)
_DELETE_LAST_SRC2_FSEL_BRANCH_COMPACT = _sm90_instruction(
    0xFFFFFFF00004B947,
    0x000FF6000383FFFF,
)


def _apply_delete_last_src2_fsel_peephole(cubin):
    sections = _elf64_sections(cubin)
    text_entry = sections.get(_NARROW_LOOP_TEXT_SECTION)
    nvinfo_entry = sections.get(".nv.info")
    kernel_nvinfo_entry = sections.get(
        ".nv.info._fused_chunk_bwd_kernel"
    )
    if (
        text_entry is None
        or nvinfo_entry is None
        or kernel_nvinfo_entry is None
    ):
        return cubin
    text_offset, text_size = text_entry
    nvinfo_offset, nvinfo_size = nvinfo_entry
    kernel_nvinfo_offset, kernel_nvinfo_size = kernel_nvinfo_entry
    if (
        text_size != 0x4A00
        or text_offset + text_size > len(cubin)
        or nvinfo_offset + nvinfo_size > len(cubin)
        or kernel_nvinfo_offset + kernel_nvinfo_size > len(cubin)
    ):
        return cubin

    parent_text = cubin[text_offset : text_offset + text_size]
    parent_nvinfo = cubin[nvinfo_offset : nvinfo_offset + nvinfo_size]
    parent_kernel_nvinfo = cubin[
        kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
    ]
    if (
        hashlib.sha256(parent_text).hexdigest()
        != _DELETE_LAST_SRC2_FSEL_PARENT_TEXT_SHA256
        or hashlib.sha256(parent_nvinfo).hexdigest()
        != _HALO_NVINFO_SHA256
        or hashlib.sha256(parent_kernel_nvinfo).hexdigest()
        != _HALO_KERNEL_NVINFO_SHA256
        or parent_nvinfo[8:12] != (63).to_bytes(4, "little")
    ):
        return cubin

    def read_pc(pc):
        return parent_text[pc : pc + 16]

    expected = {
        0x1F20: _DELETE_LAST_SRC2_FSEL_PARENT,
        0x1F60: _DELETE_LAST_SRC2_FSEL_FORWARD_PARENT,
        0x1FF0: _DELETE_LAST_SRC2_FSEL_SILU_0_PARENT,
        0x2010: _DELETE_LAST_SRC2_FSEL_SILU_1_PARENT,
        0x2080: _DELETE_LAST_SRC2_FSEL_SILU_2_PARENT,
        0x20B0: _DELETE_LAST_SRC2_FSEL_SILU_3_PARENT,
        0x2350: _DELETE_LAST_SRC2_FSEL_DW_PARENT,
        0x2540: _DELETE_LAST_SRC2_FSEL_BRANCH_PARENT,
    }
    if any(read_pc(pc) != encoded for pc, encoded in expected.items()):
        return cubin

    replacements = {
        0x1F60: _DELETE_LAST_SRC2_FSEL_FORWARD_PREDICATED,
        0x1FF0: _DELETE_LAST_SRC2_FSEL_SILU_0_TO_R54,
        0x2010: _DELETE_LAST_SRC2_FSEL_SILU_1_TO_R54,
        0x2080: _DELETE_LAST_SRC2_FSEL_SILU_2_FROM_R54,
        0x20B0: _DELETE_LAST_SRC2_FSEL_SILU_3_FROM_R54,
        0x2350: _DELETE_LAST_SRC2_FSEL_DW_PREDICATED,
    }
    compacted = [
        replacements.get(pc, read_pc(pc))
        for pc in range(0x1F30, 0x2550, 0x10)
    ]
    if (
        len(compacted) != 98
        or compacted[-1] != _DELETE_LAST_SRC2_FSEL_BRANCH_PARENT
    ):
        return cubin
    compacted[-1] = _DELETE_LAST_SRC2_FSEL_BRANCH_COMPACT

    output = bytearray(cubin)
    start = text_offset + 0x1F20
    output[start : start + len(compacted) * 16] = b"".join(compacted)
    output[text_offset + 0x2540 : text_offset + 0x2550] = (
        _DELETE_LAST_SRC3_FSEL_NOP
    )
    transformed = bytes(output)
    transformed_text = transformed[text_offset : text_offset + text_size]
    if (
        hashlib.sha256(transformed_text).hexdigest()
        != _DELETE_LAST_SRC2_FSEL_OUTPUT_TEXT_SHA256
        or parent_text[:0x1F20] != transformed_text[:0x1F20]
        or parent_text[0x2550:] != transformed_text[0x2550:]
        or transformed[nvinfo_offset : nvinfo_offset + nvinfo_size]
        != parent_nvinfo
        or transformed[
            kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
        ]
        != parent_kernel_nvinfo
    ):
        return cubin
    return transformed


# The second cyclic source still materializes R44 = keep ? R42 : 0 near the
# front of the recurrence.  Keep raw R42 live and predicate its two consumers,
# while moving the intervening newly-computed row-2 SiLU chain into dead R44.
# Delay the unchanged last-row R42 select until both raw-R42 consumers finish.
# Compact the deleted FSEL through the backedge.  The first FP32 convolution
# chain is reassociated, so correctness is established by the external suite;
# exact input/output text, instruction words, metadata, and resources all fail
# closed here.
_DELETE_ROW2_R44_FSEL_PARENT_TEXT_SHA256 = (
    "ce0f9326c9a80069df0defe10c1881f3c71293a06042d23b34f2b375347e03e6"
)
_DELETE_ROW2_R44_FSEL_OUTPUT_TEXT_SHA256 = (
    "a3d631eecc5e01b859b4b730b82d49b15bc3ff632f0092c026528a4f5aac475a"
)
_DELETE_ROW2_R44_FSEL_PARENT = _sm90_instruction(
    0x000000FF2A2C7208,
    0x000FC40004800000,
)
_DELETE_ROW2_R44_FSEL_MOVED = _sm90_instruction(
    0x000000FF3A2A7208,
    0x000FE20004000000,
)
_DELETE_ROW2_R44_FSEL_BRANCH_PARENT = _sm90_instruction(
    0xFFFFFFF00004B947,
    0x000FF6000383FFFF,
)
_DELETE_ROW2_R44_FSEL_BRANCH_COMPACT = _sm90_instruction(
    0xFFFFFFF00008B947,
    0x000FF6000383FFFF,
)


def _apply_delete_row2_r44_fsel_peephole(cubin):
    sections = _elf64_sections(cubin)
    text_entry = sections.get(_NARROW_LOOP_TEXT_SECTION)
    nvinfo_entry = sections.get(".nv.info")
    kernel_nvinfo_entry = sections.get(
        ".nv.info._fused_chunk_bwd_kernel"
    )
    if (
        text_entry is None
        or nvinfo_entry is None
        or kernel_nvinfo_entry is None
    ):
        return cubin
    text_offset, text_size = text_entry
    nvinfo_offset, nvinfo_size = nvinfo_entry
    kernel_nvinfo_offset, kernel_nvinfo_size = kernel_nvinfo_entry
    if (
        text_size != 0x4A00
        or text_offset + text_size > len(cubin)
        or nvinfo_offset + nvinfo_size > len(cubin)
        or kernel_nvinfo_offset + kernel_nvinfo_size > len(cubin)
    ):
        return cubin

    parent_text = cubin[text_offset : text_offset + text_size]
    parent_nvinfo = cubin[nvinfo_offset : nvinfo_offset + nvinfo_size]
    parent_kernel_nvinfo = cubin[
        kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
    ]
    if (
        hashlib.sha256(parent_text).hexdigest()
        != _DELETE_ROW2_R44_FSEL_PARENT_TEXT_SHA256
        or hashlib.sha256(parent_nvinfo).hexdigest()
        != _HALO_NVINFO_SHA256
        or hashlib.sha256(parent_kernel_nvinfo).hexdigest()
        != _HALO_KERNEL_NVINFO_SHA256
        or parent_nvinfo[8:12] != (63).to_bytes(4, "little")
    ):
        return cubin

    def read_pc(pc):
        return parent_text[pc : pc + 16]

    expected = {
        0x1B20: _DELETE_ROW2_R44_FSEL_PARENT,
        0x1B80: _sm90_instruction(
            0x0000002C0A197223, 0x000FE40000010007
        ),
        0x1BD0: _sm90_instruction(
            0x0000001E0C137223, 0x000FE20000010019
        ),
        0x1BF0: _sm90_instruction(
            0x0000001F042B7223, 0x000FE20000010013
        ),
        0x1C70: _sm90_instruction(
            0x000000140D2A7223, 0x000FE20000010026
        ),
        0x1CC0: _sm90_instruction(
            0x0000003A052A7223, 0x000FE2000001002A
        ),
        0x1CF0: _sm90_instruction(
            0x00000000092A7223, 0x040FE2000001002A
        ),
        0x1D40: _sm90_instruction(
            0x3F0000002A107823, 0x000FE200000100FF
        ),
        0x1E40: _sm90_instruction(
            0x3F8000002A0F7423, 0x000FE2000001000F
        ),
        0x1ED0: _DELETE_ROW2_R44_FSEL_MOVED,
        0x1F20: _sm90_instruction(
            0x0000002A0D207223, 0x000FE20000010020
        ),
        0x1F80: _sm90_instruction(
            0x0000002C0F287223, 0x000FE20000010028
        ),
        0x2530: _DELETE_ROW2_R44_FSEL_BRANCH_PARENT,
    }
    if any(read_pc(pc) != encoded for pc, encoded in expected.items()):
        return cubin

    replacements = {
        0x1B80: _sm90_instruction(
            0x0000001E0C197223, 0x000FE40000010007
        ),
        0x1BD0: _sm90_instruction(
            0x0000002A0A199223, 0x000FE20000010019
        ),
        0x1BF0: _sm90_instruction(
            0x0000001F042B7223, 0x000FE20000010019
        ),
        0x1C70: _sm90_instruction(
            0x000000140D2C7223, 0x000FE20000010026
        ),
        0x1CC0: _sm90_instruction(
            0x0000003A052C7223, 0x000FE2000001002C
        ),
        0x1CF0: _sm90_instruction(
            0x00000000092C7223, 0x040FE2000001002C
        ),
        0x1D40: _sm90_instruction(
            0x3F0000002C107823, 0x000FE200000100FF
        ),
        0x1E40: _sm90_instruction(
            0x3F8000002C0F7423, 0x000FE2000001000F
        ),
        0x1F20: _sm90_instruction(
            0x0000003A0D208223, 0x000FE20000010020
        ),
        0x1F80: _sm90_instruction(
            0x0000002A0F281223, 0x000FE20000010028
        ),
    }
    compacted = []
    for pc in range(0x1B20, 0x2540, 0x10):
        if pc in (0x1B20, 0x1ED0):
            continue
        compacted.append(replacements.get(pc, read_pc(pc)))
        if pc == 0x1F80:
            compacted.append(_DELETE_ROW2_R44_FSEL_MOVED)
    if (
        len(compacted) != 161
        or compacted[-1] != _DELETE_ROW2_R44_FSEL_BRANCH_PARENT
    ):
        return cubin
    compacted[-1] = _DELETE_ROW2_R44_FSEL_BRANCH_COMPACT

    output = bytearray(cubin)
    start = text_offset + 0x1B20
    output[start : start + len(compacted) * 16] = b"".join(compacted)
    output[text_offset + 0x2530 : text_offset + 0x2540] = (
        _DELETE_LAST_SRC3_FSEL_NOP
    )
    transformed = bytes(output)
    transformed_text = transformed[text_offset : text_offset + text_size]
    if (
        hashlib.sha256(transformed_text).hexdigest()
        != _DELETE_ROW2_R44_FSEL_OUTPUT_TEXT_SHA256
        or parent_text[:0x1B20] != transformed_text[:0x1B20]
        or parent_text[0x2540:] != transformed_text[0x2540:]
        or transformed[nvinfo_offset : nvinfo_offset + nvinfo_size]
        != parent_nvinfo
        or transformed[
            kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
        ]
        != parent_kernel_nvinfo
    ):
        return cubin
    return transformed


# Delete the entry-time R57 = entry_valid ? R42 : 0 selector.  Keep entry R42
# raw through both consumers, move its temporary SiLU chain into dead R0,
# preserve the carried R0 addend in R57 across the backedge, and delay the
# later row-valid R42 select until the raw value dies.  The two convolution
# sums are reassociated, so numerical acceptance belongs to validation; exact
# parent/output text and metadata make every other specialization fail closed.
_DELETE_ENTRY_R57_FSEL_PARENT_TEXT_SHA256 = (
    "a3d631eecc5e01b859b4b730b82d49b15bc3ff632f0092c026528a4f5aac475a"
)
_DELETE_ENTRY_R57_FSEL_OUTPUT_TEXT_SHA256 = (
    "4d8fe823a8b3547e96b52446b7ab73a5e53d07e1ab8dc402185e25eb0807c9a0"
)
_DELETE_ENTRY_R57_FSEL_DELETE = _sm90_instruction(
    0x000000FF2A397208,
    0x000FE40005800000,
)
_DELETE_ENTRY_R57_FSEL_PROLOGUE = _sm90_instruction(
    0x0000001235397223,
    0x080FE20000010022,
)
_DELETE_ENTRY_R57_FSEL_REPLACEMENTS = {
    0x1620: _sm90_instruction(
        0x000000290D347223, 0x000FE20000010006
    ),
    0x1660: _sm90_instruction(
        0x0000002A0B34B223, 0x000FC40000010034
    ),
    0x1690: _sm90_instruction(
        0x0000003208007223, 0x000FE40000010027
    ),
    0x16B0: _sm90_instruction(
        0x3F00000000357823, 0x000FE400000100FF
    ),
    0x1720: _sm90_instruction(
        0x3F800000003B7423, 0x000FE2000001003B
    ),
    0x1730: _sm90_instruction(
        0x3F00000034007423, 0x0A0FE20000010827
    ),
    0x1750: _sm90_instruction(
        0x3F8000002B2B7423, 0x000FE20000010000
    ),
    0x1760: _sm90_instruction(
        0x3F00000034007423, 0x000FE20000010027
    ),
    0x1780: _sm90_instruction(
        0x0000002B00007220, 0x000FE20000410000
    ),
    0x17B0: _sm90_instruction(
        0x0000002500347220, 0x000FE40000410000
    ),
    0x17E0: _sm90_instruction(
        0x0000000DFF007C0C, 0x000FC8000BFA5070
    ),
    0x17F0: _sm90_instruction(
        0x000000FF2C367208, 0x000FE40006800000
    ),
    0x18C0: _sm90_instruction(
        0x000000FF29297208, 0x000FCA0006800000
    ),
    0x1990: _sm90_instruction(
        0x000000FF2E2E7208, 0x000FE40006800000
    ),
    0x19A0: _sm90_instruction(
        0x000000FF33337208, 0x000FC60006800000
    ),
    0x2450: _sm90_instruction(
        0x0000003312397223, 0x000FE20000010021
    ),
}
_DELETE_ENTRY_R57_FSEL_SPECIAL_BLOCK = (
    _sm90_instruction(0x0000003537287223, 0x000FC60000010028),
    _sm90_instruction(0x000000360A377223, 0x000FE20000010007),
    _sm90_instruction(0x0000002D0C37D223, 0x000FE20000010037),
    _sm90_instruction(0x000000FF322B7208, 0x000FCA0006800000),
    _sm90_instruction(0x0000002B043B7223, 0x000FE20000010037),
    _sm90_instruction(0x000000342A39B223, 0x000FE20000010039),
    _sm90_instruction(0x000000FF2D2A7208, 0x000FC60006800000),
)
_DELETE_ENTRY_R57_FSEL_BRANCH_COMPACT = _sm90_instruction(
    0xFFFFFFF0000CB947,
    0x000FF6000383FFFF,
)


def _apply_delete_entry_r57_fsel_peephole(cubin):
    sections = _elf64_sections(cubin)
    text_entry = sections.get(_NARROW_LOOP_TEXT_SECTION)
    nvinfo_entry = sections.get(".nv.info")
    kernel_nvinfo_entry = sections.get(
        ".nv.info._fused_chunk_bwd_kernel"
    )
    if (
        text_entry is None
        or nvinfo_entry is None
        or kernel_nvinfo_entry is None
    ):
        return cubin
    text_offset, text_size = text_entry
    nvinfo_offset, nvinfo_size = nvinfo_entry
    kernel_nvinfo_offset, kernel_nvinfo_size = kernel_nvinfo_entry
    if (
        text_size != 0x4A00
        or text_offset + text_size > len(cubin)
        or nvinfo_offset + nvinfo_size > len(cubin)
        or kernel_nvinfo_offset + kernel_nvinfo_size > len(cubin)
    ):
        return cubin

    parent_text = cubin[text_offset : text_offset + text_size]
    parent_nvinfo = cubin[nvinfo_offset : nvinfo_offset + nvinfo_size]
    parent_kernel_nvinfo = cubin[
        kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
    ]
    if (
        hashlib.sha256(parent_text).hexdigest()
        != _DELETE_ENTRY_R57_FSEL_PARENT_TEXT_SHA256
        or hashlib.sha256(parent_nvinfo).hexdigest()
        != _HALO_NVINFO_SHA256
        or hashlib.sha256(parent_kernel_nvinfo).hexdigest()
        != _HALO_KERNEL_NVINFO_SHA256
        or parent_nvinfo[8:12] != (63).to_bytes(4, "little")
    ):
        return cubin

    def read_pc(pc):
        return parent_text[pc : pc + 16]

    expected = {
        0x1480: _sm90_instruction(
            0x0000001235007223, 0x080FE20000010022
        ),
        0x15E0: _DELETE_ENTRY_R57_FSEL_DELETE,
        0x1620: _sm90_instruction(
            0x000000390B347223, 0x000FE20000010006
        ),
        0x1660: _sm90_instruction(
            0x000000290D347223, 0x000FC40000010034
        ),
        0x1800: _sm90_instruction(
            0x000000FF2D2A7208, 0x000FC60005800000
        ),
        0x1810: _sm90_instruction(
            0x000000360A2B7223, 0x000FE20000010007
        ),
        0x1820: _sm90_instruction(
            0x0000003537287223, 0x000FC60000010028
        ),
        0x1830: _sm90_instruction(
            0x0000002A0C377223, 0x000FE2000001002B
        ),
        0x1860: _sm90_instruction(
            0x0000003439397223, 0x000FE20000010000
        ),
        0x2450: _sm90_instruction(
            0x0000003312007223, 0x000FE20000010021
        ),
        0x2520: _DELETE_ROW2_R44_FSEL_BRANCH_COMPACT,
    }
    if any(read_pc(pc) != encoded for pc, encoded in expected.items()):
        return cubin

    compacted = []
    pc = 0x15F0
    while pc <= 0x2520:
        if pc == 0x1800:
            compacted.extend(_DELETE_ENTRY_R57_FSEL_SPECIAL_BLOCK)
            pc = 0x1870
            continue
        compacted.append(
            _DELETE_ENTRY_R57_FSEL_REPLACEMENTS.get(pc, read_pc(pc))
        )
        pc += 0x10
    if (
        len(compacted) != 244
        or compacted[-1] != _DELETE_ROW2_R44_FSEL_BRANCH_COMPACT
    ):
        return cubin
    compacted[-1] = _DELETE_ENTRY_R57_FSEL_BRANCH_COMPACT

    output = bytearray(cubin)
    output[text_offset + 0x1480 : text_offset + 0x1490] = (
        _DELETE_ENTRY_R57_FSEL_PROLOGUE
    )
    start = text_offset + 0x15E0
    output[start : start + len(compacted) * 16] = b"".join(compacted)
    output[text_offset + 0x2520 : text_offset + 0x2530] = (
        _DELETE_LAST_SRC3_FSEL_NOP
    )
    transformed = bytes(output)
    transformed_text = transformed[text_offset : text_offset + text_size]
    if (
        hashlib.sha256(transformed_text).hexdigest()
        != _DELETE_ENTRY_R57_FSEL_OUTPUT_TEXT_SHA256
        or parent_text[:0x1480] != transformed_text[:0x1480]
        or parent_text[0x2530:] != transformed_text[0x2530:]
        or transformed[nvinfo_offset : nvinfo_offset + nvinfo_size]
        != parent_nvinfo
        or transformed[
            kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
        ]
        != parent_kernel_nvinfo
    ):
        return cubin
    return transformed


# Delete the P1-stage R31 = valid ? R1 : 0 selector. Repurpose the later R19
# selector to carry the same selected R1 through all three consumers, and
# consume its displaced selected-R46 value from raw R46 under P1. The unrelated
# R46 overwrite interval rotates through dead R60. R57/R33/R19 are restored at
# the backedge, while two former NOPs restore the distinct exit-only R33/R57
# values. Exact text, words, metadata, resources, and control flow fail closed.
_DELETE_P1_R31_FSEL_PARENT_TEXT_SHA256 = (
    "4d8fe823a8b3547e96b52446b7ab73a5e53d07e1ab8dc402185e25eb0807c9a0"
)
_DELETE_P1_R31_FSEL_OUTPUT_TEXT_SHA256 = (
    "2c2a46ddb6936e501db7b8c622981bc6271cee4a975ef05ddac2f9179ccfee6a"
)
_DELETE_P1_R31_FSEL_EXPECTED = {
    0x1BA0: _sm90_instruction(
        0x000000FF011F7208, 0x000FE20004800000
    ),
    0x1BD0: _sm90_instruction(
        0x0000001F042B7223, 0x000FE20000010019
    ),
    0x1BE0: _sm90_instruction(
        0x000000FF2E137208, 0x000FE20004800000
    ),
    0x1C20: _sm90_instruction(
        0x000000130B267223, 0x000FE20000010006
    ),
    0x1C50: _sm90_instruction(
        0x000000140D2C7223, 0x000FE20000010026
    ),
    0x1D00: _sm90_instruction(
        0x00000010092E7220, 0x000FE20000400000
    ),
    0x1D40: _sm90_instruction(
        0x00000012052E0223, 0x000FC8000001002E
    ),
    0x1D80: _sm90_instruction(
        0x000000020B2E4223, 0x000FE2000001002E
    ),
    0x1D90: _sm90_instruction(
        0x3F000000263C7423, 0x002FE40000010827
    ),
    0x1DB0: _sm90_instruction(
        0x000000340D2E3223, 0x000FE2000001002E
    ),
    0x1DC0: _sm90_instruction(
        0x3F800000293C7423, 0x000FE2000001003C
    ),
    0x1DD0: _sm90_instruction(
        0x3F00000026297423, 0x000FC60000010027
    ),
    0x1DE0: _sm90_instruction(
        0x000076320F267816, 0x000FE2000000002E
    ),
    0x1DF0: _sm90_instruction(
        0x3F000000100F7423, 0x0C4FE20000010827
    ),
    0x1E00: _sm90_instruction(
        0x000000000000781C, 0x000FE20000F2E170
    ),
    0x1E10: _sm90_instruction(
        0x3F00000010107423, 0x000FE40000010027
    ),
    0x1EB0: _sm90_instruction(
        0x0000001310217223, 0x040FE20000010039
    ),
    0x1ED0: _sm90_instruction(
        0x0000001410137223, 0x000FE20000010015
    ),
    0x1F00: _sm90_instruction(
        0x0000001F0F157223, 0x000FE20000010038
    ),
    0x1F10: _sm90_instruction(
        0x000000FF1F2B7208, 0x000FE20004000000
    ),
    0x2050: _sm90_instruction(
        0x3F00000036397423, 0x002FC60000010827
    ),
    0x2070: _sm90_instruction(
        0x3F80000020397423, 0x000FE20000010039
    ),
    0x2220: _sm90_instruction(
        0x0000003920207220, 0x000FE20000410000
    ),
    0x2440: _sm90_instruction(
        0x0000003312397223, 0x000FE20000010021
    ),
    0x24A0: _sm90_instruction(
        0x000000122A187223, 0x082FE20000010013
    ),
    0x2510: _sm90_instruction(
        0xFFFFFFF0000CB947, 0x000FF6000383FFFF
    ),
    0x2520: _DELETE_LAST_SRC3_FSEL_NOP,
}
_DELETE_P1_R31_FSEL_REPLACEMENTS = {
    0x1BD0: _sm90_instruction(
        0x000000FF01137208, 0x000FE20004800000
    ),
    0x1BE0: _sm90_instruction(
        0x00000013042B7223, 0x000FE20000010019
    ),
    0x1C20: _sm90_instruction(
        0x000000140D2C7223, 0x000FE20000010006
    ),
    0x1C50: _sm90_instruction(
        0x0000002E0B2C9223, 0x000FE2000001002C
    ),
    0x1D00: _sm90_instruction(
        0x00000010093C7220, 0x000FE20000400000
    ),
    0x1D40: _sm90_instruction(
        0x00000012053C0223, 0x000FC8000001003C
    ),
    0x1D80: _sm90_instruction(
        0x000000020B3C4223, 0x000FE2000001003C
    ),
    0x1D90: _DELETE_P1_R31_FSEL_EXPECTED[0x1DF0],
    0x1DB0: _sm90_instruction(
        0x000000340D3C3223, 0x000FE2000001003C
    ),
    0x1DC0: _DELETE_P1_R31_FSEL_EXPECTED[0x1E00],
    0x1DD0: _DELETE_P1_R31_FSEL_EXPECTED[0x1E10],
    0x1DE0: _sm90_instruction(
        0x000076320F267816, 0x000FE2000000003C
    ),
    0x1DF0: _DELETE_P1_R31_FSEL_EXPECTED[0x1D90],
    0x1E00: _DELETE_P1_R31_FSEL_EXPECTED[0x1DC0],
    0x1E10: _DELETE_P1_R31_FSEL_EXPECTED[0x1DD0],
    0x1EB0: _sm90_instruction(
        0x0000002E10391223, 0x040FE20000010039
    ),
    0x1ED0: _sm90_instruction(
        0x0000001410217223, 0x000FE20000010015
    ),
    0x1F00: _sm90_instruction(
        0x000000130F157223, 0x000FE20000010038
    ),
    0x1F10: _sm90_instruction(
        0x000000FF132B7208, 0x000FE20004000000
    ),
    0x2050: _sm90_instruction(
        0x3F00000036137423, 0x002FC60000010827
    ),
    0x2070: _sm90_instruction(
        0x3F80000020137423, 0x000FE20000010013
    ),
    0x2220: _sm90_instruction(
        0x0000001320207220, 0x000FE20000410000
    ),
    0x2440: _sm90_instruction(
        0x000000331239B223, 0x000FE20000010039
    ),
    0x24A0: _sm90_instruction(
        0x000000122A187223, 0x082FE20000010021
    ),
    0x2510: _sm90_instruction(
        0xFFFFFFF00010B947, 0x000FF6000383FFFF
    ),
}
_DELETE_P1_R31_FSEL_EXIT_MOV = _sm90_instruction(
    0x0000003900217202, 0x000FE40000000F00
)
_DELETE_P1_R31_FSEL_EXIT_FINAL = _sm90_instruction(
    0x0000003312397223, 0x000FE20000010021
)


def _apply_delete_p1_r31_fsel_peephole(cubin):
    sections = _elf64_sections(cubin)
    text_entry = sections.get(_NARROW_LOOP_TEXT_SECTION)
    nvinfo_entry = sections.get(".nv.info")
    kernel_nvinfo_entry = sections.get(
        ".nv.info._fused_chunk_bwd_kernel"
    )
    if (
        text_entry is None
        or nvinfo_entry is None
        or kernel_nvinfo_entry is None
    ):
        return cubin
    text_offset, text_size = text_entry
    nvinfo_offset, nvinfo_size = nvinfo_entry
    kernel_nvinfo_offset, kernel_nvinfo_size = kernel_nvinfo_entry
    if (
        text_size != 0x4A00
        or text_offset + text_size > len(cubin)
        or nvinfo_offset + nvinfo_size > len(cubin)
        or kernel_nvinfo_offset + kernel_nvinfo_size > len(cubin)
    ):
        return cubin

    parent_text = cubin[text_offset : text_offset + text_size]
    parent_nvinfo = cubin[nvinfo_offset : nvinfo_offset + nvinfo_size]
    parent_kernel_nvinfo = cubin[
        kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
    ]
    if (
        hashlib.sha256(parent_text).hexdigest()
        != _DELETE_P1_R31_FSEL_PARENT_TEXT_SHA256
        or hashlib.sha256(parent_nvinfo).hexdigest()
        != _HALO_NVINFO_SHA256
        or hashlib.sha256(parent_kernel_nvinfo).hexdigest()
        != _HALO_KERNEL_NVINFO_SHA256
        or parent_nvinfo[8:12] != (63).to_bytes(4, "little")
    ):
        return cubin

    def read_pc(pc):
        return parent_text[pc : pc + 16]

    if any(
        read_pc(pc) != encoded
        for pc, encoded in _DELETE_P1_R31_FSEL_EXPECTED.items()
    ):
        return cubin

    compacted = []
    for pc in range(0x1BA0, 0x2520, 0x10):
        if pc == 0x1BA0:
            continue
        compacted.append(
            _DELETE_P1_R31_FSEL_REPLACEMENTS.get(pc, read_pc(pc))
        )
    if (
        len(compacted) != 151
        or compacted[-1]
        != _DELETE_P1_R31_FSEL_REPLACEMENTS[0x2510]
    ):
        return cubin

    output = bytearray(cubin)
    start = text_offset + 0x1BA0
    output[start : start + len(compacted) * 16] = b"".join(compacted)
    output[text_offset + 0x2510 : text_offset + 0x2520] = (
        _DELETE_P1_R31_FSEL_EXIT_MOV
    )
    output[text_offset + 0x2520 : text_offset + 0x2530] = (
        _DELETE_P1_R31_FSEL_EXIT_FINAL
    )
    transformed = bytes(output)
    transformed_text = transformed[text_offset : text_offset + text_size]
    if (
        hashlib.sha256(transformed_text).hexdigest()
        != _DELETE_P1_R31_FSEL_OUTPUT_TEXT_SHA256
        or parent_text[:0x1BA0] != transformed_text[:0x1BA0]
        or parent_text[0x2530:] != transformed_text[0x2530:]
        or transformed[nvinfo_offset : nvinfo_offset + nvinfo_size]
        != parent_nvinfo
        or transformed[
            kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
        ]
        != parent_kernel_nvinfo
    ):
        return cubin
    return transformed


# Compose the independently validated entry-R45 source-select deletion onto
# the exact a3/482 executable. The a3/482 transform begins after this edit's
# predicate lifetime, so its complete suffix (including its two exit restores)
# is shifted byte-for-byte. Exact text, words, metadata, resources, and branch
# displacement fail closed.
_COMPOSE_ENTRY_R45_A3_482_PARENT_TEXT_SHA256 = (
    "2c2a46ddb6936e501db7b8c622981bc6271cee4a975ef05ddac2f9179ccfee6a"
)
_COMPOSE_ENTRY_R45_A3_482_OUTPUT_TEXT_SHA256 = (
    "7675e6551b62b1c37b73c2501f77c321d53dd301a23e87748c970dff971bec81"
)
_COMPOSE_ENTRY_R45_A3_482_DELETE = _sm90_instruction(
    0x000000FF2D2D7208, 0x000FE20005800000
)
_COMPOSE_ENTRY_R45_A3_482_PARENT_BRANCH = _sm90_instruction(
    0xFFFFFFF00010B947, 0x000FF6000383FFFF
)
_COMPOSE_ENTRY_R45_A3_482_COMPACT_BRANCH = _sm90_instruction(
    0xFFFFFFF00014B947, 0x000FF6000383FFFF
)
_COMPOSE_ENTRY_R45_A3_482_EXIT_MOV = _sm90_instruction(
    0x0000003900217202, 0x000FE40000000F00
)
_COMPOSE_ENTRY_R45_A3_482_EXIT_FINAL = _sm90_instruction(
    0x0000003312397223, 0x000FE20000010021
)
_COMPOSE_ENTRY_R45_A3_482_REPLACEMENTS = {
    # Preserve the additive base when entry R45 is invalid.
    0x1660: _sm90_instruction(
        0x0000002D0427B223, 0x000FE40000010027
    ),
    # Produce the later invalid bit as a uniform predicate.
    0x17C0: _sm90_instruction(
        0x00000100090D7892, 0x000FCC000F82C03F
    ),
    # P5=UP1; P6=!entry_P3&&!UP1.
    0x17D0: _sm90_instruction(
        0x000000000088781C, 0x000FC80001DCE01C
    ),
    # Add raw R45 only on the original valid paths.
    0x1810: _sm90_instruction(
        0x0000002D0C376223, 0x000FE20000010037
    ),
    0x1850: _sm90_instruction(
        0x000000FF2D2A7208, 0x000FC60003000000
    ),
    0x1870: _sm90_instruction(
        0x000000352D14B223, 0x000FE40000010014
    ),
    0x2500: _COMPOSE_ENTRY_R45_A3_482_COMPACT_BRANCH,
}


def _apply_compose_entry_r45_a3_482_peephole(cubin):
    sections = _elf64_sections(cubin)
    text_entry = sections.get(_NARROW_LOOP_TEXT_SECTION)
    nvinfo_entry = sections.get(".nv.info")
    kernel_nvinfo_entry = sections.get(
        ".nv.info._fused_chunk_bwd_kernel"
    )
    if (
        text_entry is None
        or nvinfo_entry is None
        or kernel_nvinfo_entry is None
    ):
        return cubin
    text_offset, text_size = text_entry
    nvinfo_offset, nvinfo_size = nvinfo_entry
    kernel_nvinfo_offset, kernel_nvinfo_size = kernel_nvinfo_entry
    if (
        text_size != 0x4A00
        or text_offset + text_size > len(cubin)
        or nvinfo_offset + nvinfo_size > len(cubin)
        or kernel_nvinfo_offset + kernel_nvinfo_size > len(cubin)
    ):
        return cubin

    parent_text = cubin[text_offset : text_offset + text_size]
    parent_nvinfo = cubin[nvinfo_offset : nvinfo_offset + nvinfo_size]
    parent_kernel_nvinfo = cubin[
        kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
    ]
    if (
        hashlib.sha256(parent_text).hexdigest()
        != _COMPOSE_ENTRY_R45_A3_482_PARENT_TEXT_SHA256
        or hashlib.sha256(parent_nvinfo).hexdigest()
        != _HALO_NVINFO_SHA256
        or hashlib.sha256(parent_kernel_nvinfo).hexdigest()
        != _HALO_KERNEL_NVINFO_SHA256
        or parent_nvinfo[8:12] != (63).to_bytes(4, "little")
    ):
        return cubin

    def read_pc(pc):
        return parent_text[pc : pc + 16]

    expected = {
        0x1620: _COMPOSE_ENTRY_R45_A3_482_DELETE,
        0x1660: _sm90_instruction(
            0x0000002D04277223, 0x000FE40000010027
        ),
        0x17C0: _sm90_instruction(
            0x00000100090D7892, 0x000FCC000F8EC03F
        ),
        0x17D0: _sm90_instruction(
            0x0000000DFF007C0C, 0x000FC8000BFA5070
        ),
        0x1810: _sm90_instruction(
            0x0000002D0C37D223, 0x000FE20000010037
        ),
        0x1850: _sm90_instruction(
            0x000000FF2D2A7208, 0x000FC60006800000
        ),
        0x1870: _sm90_instruction(
            0x000000352D147223, 0x000FE40000010014
        ),
        0x1BA0: _sm90_instruction(
            0x0000002A0A199223, 0x000FE20000010019
        ),
        0x2500: _COMPOSE_ENTRY_R45_A3_482_PARENT_BRANCH,
        0x2510: _COMPOSE_ENTRY_R45_A3_482_EXIT_MOV,
        0x2520: _COMPOSE_ENTRY_R45_A3_482_EXIT_FINAL,
        0x2530: _DELETE_LAST_SRC3_FSEL_NOP,
    }
    if any(read_pc(pc) != encoded for pc, encoded in expected.items()):
        return cubin

    compacted = [
        _COMPOSE_ENTRY_R45_A3_482_REPLACEMENTS.get(pc, read_pc(pc))
        for pc in range(0x1630, 0x2530, 0x10)
    ]
    if (
        len(compacted) != 240
        or compacted[-3]
        != _COMPOSE_ENTRY_R45_A3_482_COMPACT_BRANCH
        or compacted[-2:]
        != [
            _COMPOSE_ENTRY_R45_A3_482_EXIT_MOV,
            _COMPOSE_ENTRY_R45_A3_482_EXIT_FINAL,
        ]
    ):
        return cubin

    output = bytearray(cubin)
    start = text_offset + 0x1620
    output[start : start + len(compacted) * 16] = b"".join(compacted)
    output[text_offset + 0x2520 : text_offset + 0x2530] = (
        _DELETE_LAST_SRC3_FSEL_NOP
    )
    transformed = bytes(output)
    transformed_text = transformed[text_offset : text_offset + text_size]
    if (
        hashlib.sha256(transformed_text).hexdigest()
        != _COMPOSE_ENTRY_R45_A3_482_OUTPUT_TEXT_SHA256
        or parent_text[:0x1620] != transformed_text[:0x1620]
        or parent_text[0x2530:] != transformed_text[0x2530:]
        or transformed[nvinfo_offset : nvinfo_offset + nvinfo_size]
        != parent_nvinfo
        or transformed[
            kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
        ]
        != parent_kernel_nvinfo
    ):
        return cubin
    return transformed


# a5/440 exposes P6 as the exact conjunction of entry validity and later-row
# validity. Reuse it to keep entry R44 raw, predicate its two additive
# consumers with the existing entry P3, and replace the later nested selection
# with one P6 selection. This removes the entry R44 FSEL without the temporary
# accumulator/register bridge required by the earlier a5/438 realization.
_COMPOSE_ENTRY_R44_A5_440_PARENT_TEXT_SHA256 = (
    "7675e6551b62b1c37b73c2501f77c321d53dd301a23e87748c970dff971bec81"
)
_COMPOSE_ENTRY_R44_A5_440_OUTPUT_TEXT_SHA256 = (
    "a69d2a6d3a19b2e356133ffacfaa5bbd0fce0584c5be752ec942405e206aa1ea"
)
_COMPOSE_ENTRY_R44_A5_440_DELETE = _sm90_instruction(
    0x000000FF2C2C7208, 0x000FE20005800000
)
_COMPOSE_ENTRY_R44_A5_440_PARENT_BRANCH = _sm90_instruction(
    0xFFFFFFF00014B947, 0x000FF6000383FFFF
)
_COMPOSE_ENTRY_R44_A5_440_COMPACT_BRANCH = _sm90_instruction(
    0xFFFFFFF00018B947, 0x000FF6000383FFFF
)
_COMPOSE_ENTRY_R44_A5_440_EXIT_MOV = _sm90_instruction(
    0x0000003900217202, 0x000FE40000000F00
)
_COMPOSE_ENTRY_R44_A5_440_EXIT_FINAL = _sm90_instruction(
    0x0000003312397223, 0x000FE20000010021
)
_COMPOSE_ENTRY_R44_A5_440_REPLACEMENTS = {
    # Raw R44 contributes only on entry-valid lanes.
    0x1620: _sm90_instruction(
        0x0000002C0C27B223, 0x000FE20000010027
    ),
    # P6 is exactly entry-valid && later-valid.
    0x17D0: _sm90_instruction(
        0x000000FF2C367208, 0x000FE40003000000
    ),
    # Preserve the existing R19 accumulator on entry-invalid lanes.
    0x1850: _sm90_instruction(
        0x000000352C13B223, 0x080FE20000010013
    ),
    0x24F0: _COMPOSE_ENTRY_R44_A5_440_COMPACT_BRANCH,
}


def _apply_compose_entry_r44_a5_440_peephole(cubin):
    sections = _elf64_sections(cubin)
    text_entry = sections.get(_NARROW_LOOP_TEXT_SECTION)
    nvinfo_entry = sections.get(".nv.info")
    kernel_nvinfo_entry = sections.get(
        ".nv.info._fused_chunk_bwd_kernel"
    )
    if (
        text_entry is None
        or nvinfo_entry is None
        or kernel_nvinfo_entry is None
    ):
        return cubin
    text_offset, text_size = text_entry
    nvinfo_offset, nvinfo_size = nvinfo_entry
    kernel_nvinfo_offset, kernel_nvinfo_size = kernel_nvinfo_entry
    if (
        text_size != 0x4A00
        or text_offset + text_size > len(cubin)
        or nvinfo_offset + nvinfo_size > len(cubin)
        or kernel_nvinfo_offset + kernel_nvinfo_size > len(cubin)
    ):
        return cubin

    parent_text = cubin[text_offset : text_offset + text_size]
    parent_nvinfo = cubin[nvinfo_offset : nvinfo_offset + nvinfo_size]
    parent_kernel_nvinfo = cubin[
        kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
    ]
    if (
        hashlib.sha256(parent_text).hexdigest()
        != _COMPOSE_ENTRY_R44_A5_440_PARENT_TEXT_SHA256
        or hashlib.sha256(parent_nvinfo).hexdigest()
        != _HALO_NVINFO_SHA256
        or hashlib.sha256(parent_kernel_nvinfo).hexdigest()
        != _HALO_KERNEL_NVINFO_SHA256
        or parent_nvinfo[8:12] != (63).to_bytes(4, "little")
    ):
        return cubin

    def read_pc(pc):
        return parent_text[pc : pc + 16]

    expected = {
        0x15E0: _COMPOSE_ENTRY_R44_A5_440_DELETE,
        0x1620: _sm90_instruction(
            0x0000002C0C277223, 0x000FE20000010027
        ),
        0x17C0: _sm90_instruction(
            0x000000000088781C, 0x000FC80001DCE01C
        ),
        0x17D0: _sm90_instruction(
            0x000000FF2C367208, 0x000FE40006800000
        ),
        0x1850: _sm90_instruction(
            0x000000352C137223, 0x080FE20000010013
        ),
        0x24F0: _COMPOSE_ENTRY_R44_A5_440_PARENT_BRANCH,
        0x2500: _COMPOSE_ENTRY_R44_A5_440_EXIT_MOV,
        0x2510: _COMPOSE_ENTRY_R44_A5_440_EXIT_FINAL,
        0x2520: _DELETE_LAST_SRC3_FSEL_NOP,
    }
    if any(read_pc(pc) != encoded for pc, encoded in expected.items()):
        return cubin

    compacted = [
        _COMPOSE_ENTRY_R44_A5_440_REPLACEMENTS.get(pc, read_pc(pc))
        for pc in range(0x15F0, 0x2520, 0x10)
    ]
    if (
        len(compacted) != 243
        or compacted[-3]
        != _COMPOSE_ENTRY_R44_A5_440_COMPACT_BRANCH
        or compacted[-2:]
        != [
            _COMPOSE_ENTRY_R44_A5_440_EXIT_MOV,
            _COMPOSE_ENTRY_R44_A5_440_EXIT_FINAL,
        ]
    ):
        return cubin

    output = bytearray(cubin)
    start = text_offset + 0x15E0
    output[start : start + len(compacted) * 16] = b"".join(compacted)
    output[text_offset + 0x2510 : text_offset + 0x2520] = (
        _DELETE_LAST_SRC3_FSEL_NOP
    )
    transformed = bytes(output)
    transformed_text = transformed[text_offset : text_offset + text_size]
    if (
        hashlib.sha256(transformed_text).hexdigest()
        != _COMPOSE_ENTRY_R44_A5_440_OUTPUT_TEXT_SHA256
        or parent_text[:0x15E0] != transformed_text[:0x15E0]
        or parent_text[0x2520:] != transformed_text[0x2520:]
        or transformed[nvinfo_offset : nvinfo_offset + nvinfo_size]
        != parent_nvinfo
        or transformed[
            kernel_nvinfo_offset : kernel_nvinfo_offset + kernel_nvinfo_size
        ]
        != parent_kernel_nvinfo
    ):
        return cubin
    return transformed
