"""Fused Triton candidate for the causal depthwise conv1d task."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from reference import kernel_fn as reference_kernel_fn


@triton.jit
def _fast_silu(x):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 y;
            .reg .f32 e;
            .reg .f32 denom;
            .reg .f32 inv;
            fma.rn.ftz.f32 y, $1, 0fBFB8AA3B, 0f00000000;
            ex2.approx.ftz.f32 e, y;
            add.rn.ftz.f32 denom, e, 0f3F800000;
            rcp.approx.ftz.f32 inv, denom;
            fma.rn.ftz.f32 $0, $1, inv, 0f00000000;
        }
        """,
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _prefetch_l1(ptr):
    return tl.inline_asm_elementwise(
        asm="""
        {
            prefetch.global.L1 [$1];
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,l",
        args=[ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _ffma_ftz(a, b, c):
    return tl.inline_asm_elementwise(
        asm="""
        {
            fma.rn.ftz.f32 $0, $1, $2, $3;
        }
        """,
        constraints="=f,f,f,f",
        args=[a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _conv1d_main_t8_nobos_kernel(
    x_ptr,
    weight_t_ptr,
    bias_ptr,
    out_ptr,
    dirty_row_mask_ptr,
    bos_ptr,
    tile_rows_per_batch: tl.constexpr,
    start_t: tl.constexpr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    num_d_blocks: tl.constexpr,
    block_d: tl.constexpr,
    swizzle_group_rows: tl.constexpr,
    store_bf16: tl.constexpr,
    use_dirty_store_mask: tl.constexpr,
    dirty_tiles_per_batch: tl.constexpr,
    dirty_tile_offset: tl.constexpr,
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
    tt = tile_row - b * tile_rows_per_batch
    t0 = tt * 8 + start_t
    dirty_bits = 0
    if use_dirty_store_mask:
        dirty_bits = tl.load(
            dirty_row_mask_ptr + b * dirty_tiles_per_batch + dirty_tile_offset + tt,
            eviction_policy="evict_last",
        ).to(tl.int32)

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    token0 = b * seqlen + t0
    base0 = token0 * dim + d
    base1 = base0 + dim
    base2 = base1 + dim
    base3 = base2 + dim

    x_tm3 = tl.load(
        x_ptr + base0 - 3 * dim,
        mask=d_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    x_tm2 = tl.load(
        x_ptr + base0 - 2 * dim,
        mask=d_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    x_tm1 = tl.load(
        x_ptr + base0 - dim,
        mask=d_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    x_t0 = tl.load(
        x_ptr + base0,
        mask=d_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    x_t1 = tl.load(
        x_ptr + base1,
        mask=d_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    x_t2 = tl.load(
        x_ptr + base2,
        mask=d_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    x_t3 = tl.load(
        x_ptr + base3,
        mask=d_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    _prefetch_l1(x_ptr + base0 + 4 * dim)

    w0 = tl.load(weight_t_ptr + 3 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight_t_ptr + 2 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w2 = tl.load(weight_t_ptr + dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w3 = tl.load(weight_t_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    x_m3 = x_tm3
    x_m2 = x_tm2
    x_m1 = x_tm1
    x_cur = x_t0
    x_next1 = x_t1
    x_next2 = x_t2
    x_next3 = x_t3
    zero = tl.full((block_d,), 0.0, tl.float32)
    one = tl.full((block_d,), 1.0, tl.float32)
    for i in tl.static_range(0, 8):
        acc01 = _ffma_ftz(x_cur, w0, bias)
        acc01 = _ffma_ftz(x_m1, w1, acc01)
        acc23 = _ffma_ftz(x_m2, w2, zero)
        acc23 = _ffma_ftz(x_m3, w3, acc23)
        acc = _ffma_ftz(acc23, one, acc01)
        acc = _fast_silu(acc)
        if store_bf16:
            store_acc = acc.to(tl.bfloat16)
        else:
            store_acc = acc
        store_mask = d_mask
        if use_dirty_store_mask and i < 7:
            store_mask = store_mask & ((dirty_bits & (1 << i)) == 0)
        tl.store(
            out_ptr + base0 + i * dim,
            store_acc,
            mask=store_mask,
            cache_modifier=".cg",
        )

        if i < 4:
            x_new = tl.load(
                x_ptr + base0 + (i + 4) * dim,
                mask=d_mask,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
        else:
            x_new = x_next3
        x_m3 = x_m2
        x_m2 = x_m1
        x_m1 = x_cur
        x_cur = x_next1
        x_next1 = x_next2
        x_next2 = x_next3
        x_next3 = x_new

    if use_dirty_store_mask and dirty_bits != 0:
        bos_base = b * seqlen
        bos_tm2 = tl.load(bos_ptr + bos_base + t0 - 2)
        bos_tm1 = tl.load(bos_ptr + bos_base + t0 - 1)
        bos_t0 = tl.load(bos_ptr + bos_base + t0)
        bos_t1 = tl.load(bos_ptr + bos_base + t0 + 1)
        bos_t2 = tl.load(bos_ptr + bos_base + t0 + 2)
        bos_t3 = tl.load(bos_ptr + bos_base + t0 + 3)
        bos_t4 = tl.load(bos_ptr + bos_base + t0 + 4)
        bos_t5 = tl.load(bos_ptr + bos_base + t0 + 5)
        bos_t6 = tl.load(bos_ptr + bos_base + t0 + 6)
        bos_t7 = tl.load(bos_ptr + bos_base + t0 + 7)
        for i in tl.static_range(0, 8):
            if (dirty_bits & (1 << i)) != 0:
                fix_base = base0 + i * dim
                if i == 0:
                    bos_i = bos_t0
                    bos_im1 = bos_tm1
                    bos_im2 = bos_tm2
                elif i == 1:
                    bos_i = bos_t1
                    bos_im1 = bos_t0
                    bos_im2 = bos_tm1
                elif i == 2:
                    bos_i = bos_t2
                    bos_im1 = bos_t1
                    bos_im2 = bos_t0
                elif i == 3:
                    bos_i = bos_t3
                    bos_im1 = bos_t2
                    bos_im2 = bos_t1
                elif i == 4:
                    bos_i = bos_t4
                    bos_im1 = bos_t3
                    bos_im2 = bos_t2
                elif i == 5:
                    bos_i = bos_t5
                    bos_im1 = bos_t4
                    bos_im2 = bos_t3
                elif i == 6:
                    bos_i = bos_t6
                    bos_im1 = bos_t5
                    bos_im2 = bos_t4
                else:
                    bos_i = bos_t7
                    bos_im1 = bos_t6
                    bos_im2 = bos_t5
                fix_valid1 = ~bos_i
                fix_valid2 = fix_valid1 & (~bos_im1)
                fix_valid3 = fix_valid2 & (~bos_im2)

                fix_x_cur = tl.load(
                    x_ptr + fix_base,
                    mask=d_mask,
                    other=0.0,
                    cache_modifier=".ca",
                ).to(tl.float32)
                fix_x_m1 = tl.load(
                    x_ptr + fix_base - dim,
                    mask=d_mask & fix_valid1,
                    other=0.0,
                    cache_modifier=".ca",
                ).to(tl.float32)
                fix_x_m2 = tl.load(
                    x_ptr + fix_base - 2 * dim,
                    mask=d_mask & fix_valid2,
                    other=0.0,
                    cache_modifier=".ca",
                ).to(tl.float32)
                fix_x_m3 = tl.load(
                    x_ptr + fix_base - 3 * dim,
                    mask=d_mask & fix_valid3,
                    other=0.0,
                    cache_modifier=".ca",
                ).to(tl.float32)

                fix_acc01 = _ffma_ftz(fix_x_cur, w0, bias)
                fix_acc01 = _ffma_ftz(fix_x_m1, w1, fix_acc01)
                fix_acc23 = _ffma_ftz(fix_x_m2, w2, zero)
                fix_acc23 = _ffma_ftz(fix_x_m3, w3, fix_acc23)
                fix_acc = _ffma_ftz(fix_acc23, one, fix_acc01)
                fix_acc = _fast_silu(fix_acc)
                if store_bf16:
                    fix_store = fix_acc.to(tl.bfloat16)
                else:
                    fix_store = fix_acc
                tl.store(
                    out_ptr + fix_base,
                    fix_store,
                    mask=d_mask,
                    cache_modifier=".cg",
                )


@triton.jit
def _conv1d_main_t8_offsets_repair_kernel(
    x_ptr,
    weight_t_ptr,
    bias_ptr,
    bos_next_pos_ptr,
    bos_pos_ptr,
    bos_batch_ptr,
    bos_packed_all_ptr,
    out_ptr,
    start_t: tl.constexpr,
    main_end_t: tl.constexpr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    num_d_blocks: tl.constexpr,
    block_d: tl.constexpr,
    store_bf16: tl.constexpr,
    use_packed_all: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_d = pid % num_d_blocks
    entry = pid // num_d_blocks
    if use_packed_all:
        packed_meta = tl.load(
            bos_packed_all_ptr + entry,
            eviction_policy="evict_last",
        )
        bos_t = (packed_meta & 0x1FFFFF).to(tl.int32)
        next_bos_t = ((packed_meta >> 21) & 0x1FFFFF).to(tl.int32)
        b = (packed_meta >> 42).to(tl.int32)
    else:
        b = tl.load(bos_batch_ptr + entry)
        bos_t = tl.load(bos_pos_ptr + entry)
        next_bos_t = tl.load(bos_next_pos_ptr + entry, eviction_policy="evict_last")

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    w0 = tl.load(
        weight_t_ptr + 3 * dim + d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    w1 = tl.load(
        weight_t_ptr + 2 * dim + d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    w2 = tl.load(
        weight_t_ptr + dim + d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    row0 = bos_t
    valid0 = (row0 >= start_t) & (row0 < main_end_t)
    base0 = (b * seqlen + row0) * dim + d
    row1 = bos_t + 1
    valid1 = (row1 < next_bos_t) & (row1 >= start_t) & (row1 < main_end_t)
    base1 = (b * seqlen + row1) * dim + d
    row2 = bos_t + 2
    valid2 = (
        (row2 < next_bos_t)
        & (row2 >= start_t)
        & (row2 < main_end_t)
    )
    base2 = (b * seqlen + row2) * dim + d

    x0 = tl.load(
        x_ptr + base0,
        mask=d_mask & (valid0 | valid1 | valid2),
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    x1 = tl.load(
        x_ptr + base1,
        mask=d_mask & (valid1 | valid2),
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    x2 = tl.load(
        x_ptr + base2,
        mask=d_mask & valid2,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    acc0 = x0 * w0 + bias
    acc0 = _fast_silu(acc0)
    if store_bf16:
        store0 = acc0.to(tl.bfloat16)
    else:
        store0 = acc0
    tl.store(out_ptr + base0, store0, mask=d_mask & valid0, cache_modifier=".cg")

    acc1 = x1 * w0 + bias
    acc1 += x0 * w1
    acc1 = _fast_silu(acc1)
    if store_bf16:
        store1 = acc1.to(tl.bfloat16)
    else:
        store1 = acc1
    tl.store(out_ptr + base1, store1, mask=d_mask & valid1, cache_modifier=".cg")

    acc2 = x2 * w0 + bias
    acc2 += x1 * w1
    acc2 += x0 * w2
    acc2 = _fast_silu(acc2)
    if store_bf16:
        store2 = acc2.to(tl.bfloat16)
    else:
        store2 = acc2
    tl.store(out_ptr + base2, store2, mask=d_mask & valid2, cache_modifier=".cg")


@triton.jit
def _conv1d_main_t8_offsets_repair_packed_all_kernel(
    x_ptr,
    weight_t_ptr,
    bias_ptr,
    bos_packed_all_ptr,
    out_ptr,
    start_t: tl.constexpr,
    main_end_t: tl.constexpr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    num_d_blocks: tl.constexpr,
    block_d: tl.constexpr,
    store_bf16: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_d = pid % num_d_blocks
    entry = pid // num_d_blocks

    packed_meta = tl.load(
        bos_packed_all_ptr + entry,
        eviction_policy="evict_last",
    )
    bos_t = (packed_meta & 0x1FFFFF).to(tl.int32)
    next_bos_t = ((packed_meta >> 21) & 0x1FFFFF).to(tl.int32)
    b = (packed_meta >> 42).to(tl.int32)

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    w0 = tl.load(
        weight_t_ptr + 3 * dim + d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    w1 = tl.load(
        weight_t_ptr + 2 * dim + d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    w2 = tl.load(
        weight_t_ptr + dim + d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    row0 = bos_t
    valid0 = (row0 >= start_t) & (row0 < main_end_t)
    base0 = (b * seqlen + row0) * dim + d
    row1 = bos_t + 1
    valid1 = (row1 < next_bos_t) & (row1 >= start_t) & (row1 < main_end_t)
    base1 = (b * seqlen + row1) * dim + d
    row2 = bos_t + 2
    valid2 = (
        (row2 < next_bos_t)
        & (row2 >= start_t)
        & (row2 < main_end_t)
    )
    base2 = (b * seqlen + row2) * dim + d

    x0 = tl.load(
        x_ptr + base0,
        mask=d_mask & (valid0 | valid1 | valid2),
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    x1 = tl.load(
        x_ptr + base1,
        mask=d_mask & (valid1 | valid2),
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    x2 = tl.load(
        x_ptr + base2,
        mask=d_mask & valid2,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    acc0 = x0 * w0 + bias
    acc0 = _fast_silu(acc0)
    if store_bf16:
        store0 = acc0.to(tl.bfloat16)
    else:
        store0 = acc0
    tl.store(out_ptr + base0, store0, mask=d_mask & valid0, cache_modifier=".cg")

    acc1 = x1 * w0 + bias
    acc1 += x0 * w1
    acc1 = _fast_silu(acc1)
    if store_bf16:
        store1 = acc1.to(tl.bfloat16)
    else:
        store1 = acc1
    tl.store(out_ptr + base1, store1, mask=d_mask & valid1, cache_modifier=".cg")

    acc2 = x2 * w0 + bias
    acc2 += x1 * w1
    acc2 += x0 * w2
    acc2 = _fast_silu(acc2)
    if store_bf16:
        store2 = acc2.to(tl.bfloat16)
    else:
        store2 = acc2
    tl.store(out_ptr + base2, store2, mask=d_mask & valid2, cache_modifier=".cg")


@triton.jit
def _conv1d_packed_repair_prefix_t3_tail_t5_kernel(
    x_ptr,
    weight_t_ptr,
    bias_ptr,
    init_ptr,
    bos_ptr,
    bos_packed_all_ptr,
    out_ptr,
    final_ptr,
    tail_dirty_row_mask_ptr,
    start_t: tl.constexpr,
    main_end_t: tl.constexpr,
    tail_start: tl.constexpr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    tail_dirty_tiles_per_batch: tl.constexpr,
    tail_dirty_tile_offset: tl.constexpr,
    repair_num_d_blocks: tl.constexpr,
    repair_block_d: tl.constexpr,
    boundary_num_d_blocks: tl.constexpr,
    boundary_block_d: tl.constexpr,
    main_tail_num_d_blocks: tl.constexpr,
    main_tail_block_d: tl.constexpr,
    main_tail_tile_rows: tl.constexpr,
    main_tail_start_t: tl.constexpr,
    main_tail_programs: tl.constexpr,
    repair_programs: tl.constexpr,
    repair_entries: tl.constexpr,
    prefix_programs: tl.constexpr,
    store_bf16: tl.constexpr,
    use_tail_dirty_store_mask: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < main_tail_programs:
        m_pid_d = pid % main_tail_num_d_blocks
        m_tile_row = pid // main_tail_num_d_blocks
        m_b = m_tile_row // main_tail_tile_rows
        m_tt = m_tile_row - m_b * main_tail_tile_rows
        m_t0 = m_tt * 8 + main_tail_start_t

        m_d = m_pid_d * main_tail_block_d + tl.arange(0, main_tail_block_d)
        m_d_mask = m_d < dim

        m_token0 = m_b * seqlen + m_t0
        m_base0 = m_token0 * dim + m_d
        m_base1 = m_base0 + dim
        m_base2 = m_base1 + dim
        m_base3 = m_base2 + dim

        m_x_tm3 = tl.load(
            x_ptr + m_base0 - 3 * dim,
            mask=m_d_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        m_x_tm2 = tl.load(
            x_ptr + m_base0 - 2 * dim,
            mask=m_d_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        m_x_tm1 = tl.load(
            x_ptr + m_base0 - dim,
            mask=m_d_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        m_x_t0 = tl.load(
            x_ptr + m_base0,
            mask=m_d_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        m_x_t1 = tl.load(
            x_ptr + m_base1,
            mask=m_d_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        m_x_t2 = tl.load(
            x_ptr + m_base2,
            mask=m_d_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        m_x_t3 = tl.load(
            x_ptr + m_base3,
            mask=m_d_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)

        m_w0 = tl.load(weight_t_ptr + 3 * dim + m_d, mask=m_d_mask, other=0.0).to(tl.float32)
        m_w1 = tl.load(weight_t_ptr + 2 * dim + m_d, mask=m_d_mask, other=0.0).to(tl.float32)
        m_w2 = tl.load(weight_t_ptr + dim + m_d, mask=m_d_mask, other=0.0).to(tl.float32)
        m_w3 = tl.load(weight_t_ptr + m_d, mask=m_d_mask, other=0.0).to(tl.float32)
        m_bias = tl.load(bias_ptr + m_d, mask=m_d_mask, other=0.0).to(tl.float32)

        m_bos_base = m_b * seqlen
        m_dirty_bits = 0
        m_bos_t5 = False
        m_bos_t6 = False
        m_bos_t7 = False
        if use_tail_dirty_store_mask:
            m_dirty_bits = tl.load(
                tail_dirty_row_mask_ptr
                + m_b * tail_dirty_tiles_per_batch
                + tail_dirty_tile_offset
                + m_tt,
                eviction_policy="evict_last",
            ).to(tl.int32)
            m_bos_t5 = tl.load(bos_ptr + m_bos_base + m_t0 + 5)
            m_bos_t6 = tl.load(bos_ptr + m_bos_base + m_t0 + 6)
            m_bos_t7 = tl.load(bos_ptr + m_bos_base + m_t0 + 7)
        m_x_m3 = m_x_tm3
        m_x_m2 = m_x_tm2
        m_x_m1 = m_x_tm1
        m_x_cur = m_x_t0
        m_x_next1 = m_x_t1
        m_x_next2 = m_x_t2
        m_x_next3 = m_x_t3
        m_zero = tl.full((main_tail_block_d,), 0.0, tl.float32)
        for i in tl.static_range(0, 8):
            if use_tail_dirty_store_mask:
                if i < 7:
                    m_clean = (m_dirty_bits & (1 << i)) == 0
                else:
                    m_clean = (~m_bos_t7) & (~m_bos_t6) & (~m_bos_t5)
            else:
                m_bos_i = tl.load(bos_ptr + m_bos_base + m_t0 + i)
                m_bos_im1 = tl.load(bos_ptr + m_bos_base + m_t0 + i - 1)
                m_bos_im2 = tl.load(bos_ptr + m_bos_base + m_t0 + i - 2)
                m_clean = (~m_bos_i) & (~m_bos_im1) & (~m_bos_im2)

            m_acc01 = _ffma_ftz(m_x_cur, m_w0, m_bias)
            m_acc01 = _ffma_ftz(m_x_m1, m_w1, m_acc01)
            m_acc23 = _ffma_ftz(m_x_m2, m_w2, m_zero)
            m_acc23 = _ffma_ftz(m_x_m3, m_w3, m_acc23)
            m_acc = m_acc01 + m_acc23
            m_acc = _fast_silu(m_acc)
            if store_bf16:
                m_store_acc = m_acc.to(tl.bfloat16)
            else:
                m_store_acc = m_acc
            tl.store(
                out_ptr + m_base0 + i * dim,
                m_store_acc,
                mask=m_d_mask & m_clean,
                cache_modifier=".cg",
            )

            if i < 4:
                m_x_new = tl.load(
                    x_ptr + m_base0 + (i + 4) * dim,
                    mask=m_d_mask,
                    other=0.0,
                    cache_modifier=".cg",
                ).to(tl.float32)
            else:
                m_x_new = m_x_next3
            m_x_m3 = m_x_m2
            m_x_m2 = m_x_m1
            m_x_m1 = m_x_cur
            m_x_cur = m_x_next1
            m_x_next1 = m_x_next2
            m_x_next2 = m_x_next3
            m_x_next3 = m_x_new
    elif pid < main_tail_programs + repair_programs:
        r_pid = pid - main_tail_programs
        r_pid_d = r_pid // repair_entries
        r_entry = r_pid - r_pid_d * repair_entries

        r_packed_meta = tl.load(
            bos_packed_all_ptr + r_entry,
            eviction_policy="evict_last",
        )
        r_bos_t = (r_packed_meta & 0x1FFFFF).to(tl.int32)
        r_next_bos_t = ((r_packed_meta >> 21) & 0x1FFFFF).to(tl.int32)
        r_b = (r_packed_meta >> 42).to(tl.int32)

        r_d = r_pid_d * repair_block_d + tl.arange(0, repair_block_d)
        r_d_mask = r_d < dim

        r_w0 = tl.load(
            weight_t_ptr + 3 * dim + r_d,
            mask=r_d_mask,
            other=0.0,
        ).to(tl.float32)
        r_w1 = tl.load(
            weight_t_ptr + 2 * dim + r_d,
            mask=r_d_mask,
            other=0.0,
        ).to(tl.float32)
        r_w2 = tl.load(
            weight_t_ptr + dim + r_d,
            mask=r_d_mask,
            other=0.0,
        ).to(tl.float32)
        r_bias = tl.load(
            bias_ptr + r_d,
            mask=r_d_mask,
            other=0.0,
        ).to(tl.float32)

        r_row0 = r_bos_t
        r_valid0 = (r_row0 >= start_t) & (r_row0 < main_end_t)
        r_base0 = (r_b * seqlen + r_row0) * dim + r_d
        r_row1 = r_bos_t + 1
        r_valid1 = (
            (r_row1 < r_next_bos_t) & (r_row1 >= start_t) & (r_row1 < main_end_t)
        )
        r_base1 = (r_b * seqlen + r_row1) * dim + r_d
        r_row2 = r_bos_t + 2
        r_valid2 = (
            (r_row2 < r_next_bos_t)
            & (r_row2 >= start_t)
            & (r_row2 < main_end_t)
        )
        r_base2 = (r_b * seqlen + r_row2) * dim + r_d
        r_x1_load_valid = (
            (r_row1 < r_next_bos_t)
            & (r_row1 < main_end_t)
            & (r_row2 >= start_t)
        )

        r_x0 = tl.load(
            x_ptr + r_base0,
            mask=r_d_mask & (r_valid0 | r_valid1 | r_valid2),
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        r_x1 = tl.load(
            x_ptr + r_base1,
            mask=r_d_mask & r_x1_load_valid,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        r_x2 = tl.load(
            x_ptr + r_base2,
            mask=r_d_mask & r_valid2,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

        r_acc0 = r_x0 * r_w0 + r_bias
        r_acc0 = _fast_silu(r_acc0)
        if store_bf16:
            r_store0 = r_acc0.to(tl.bfloat16)
        else:
            r_store0 = r_acc0
        tl.store(
            out_ptr + r_base0,
            r_store0,
            mask=r_d_mask & r_valid0,
            cache_modifier=".cg",
        )

        r_acc1 = r_x1 * r_w0 + r_bias
        r_acc1 += r_x0 * r_w1
        r_acc1 = _fast_silu(r_acc1)
        if store_bf16:
            r_store1 = r_acc1.to(tl.bfloat16)
        else:
            r_store1 = r_acc1
        tl.store(
            out_ptr + r_base1,
            r_store1,
            mask=r_d_mask & r_valid1,
            cache_modifier=".cg",
        )

        r_acc2 = r_x2 * r_w0 + r_bias
        r_acc2 += r_x1 * r_w1
        r_acc2 += r_x0 * r_w2
        r_acc2 = _fast_silu(r_acc2)
        if store_bf16:
            r_store2 = r_acc2.to(tl.bfloat16)
        else:
            r_store2 = r_acc2
        tl.store(
            out_ptr + r_base2,
            r_store2,
            mask=r_d_mask & r_valid2,
            cache_modifier=".cg",
        )
    else:
        boundary_pid = pid - main_tail_programs - repair_programs
        if boundary_pid < prefix_programs:
            p_pid_d = boundary_pid % boundary_num_d_blocks
            p_b = boundary_pid // boundary_num_d_blocks

            p_d = p_pid_d * boundary_block_d + tl.arange(0, boundary_block_d)
            p_d_mask = p_d < dim

            p_bos_base = p_b * seqlen
            p_bos_t0 = tl.load(bos_ptr + p_bos_base)
            p_bos_t1 = tl.load(bos_ptr + p_bos_base + 1)
            p_bos_t2 = tl.load(bos_ptr + p_bos_base + 2)

            p_token0 = p_b * seqlen
            p_base0 = p_token0 * dim + p_d
            p_base1 = p_base0 + dim
            p_base2 = p_base1 + dim

            p_x_t0 = tl.load(x_ptr + p_base0, mask=p_d_mask, other=0.0).to(tl.float32)
            p_x_t1 = tl.load(x_ptr + p_base1, mask=p_d_mask, other=0.0).to(tl.float32)
            p_x_t2 = tl.load(x_ptr + p_base2, mask=p_d_mask, other=0.0).to(tl.float32)

            p_init_base = p_b * dim * 3 + p_d * 3
            p_init0 = tl.load(init_ptr + p_init_base, mask=p_d_mask, other=0.0).to(tl.float32)
            p_init1 = tl.load(init_ptr + p_init_base + 1, mask=p_d_mask, other=0.0).to(tl.float32)
            p_init2 = tl.load(init_ptr + p_init_base + 2, mask=p_d_mask, other=0.0).to(tl.float32)

            p_w0 = tl.load(weight_t_ptr + 3 * dim + p_d, mask=p_d_mask, other=0.0).to(tl.float32)
            p_w1 = tl.load(weight_t_ptr + 2 * dim + p_d, mask=p_d_mask, other=0.0).to(tl.float32)
            p_w2 = tl.load(weight_t_ptr + dim + p_d, mask=p_d_mask, other=0.0).to(tl.float32)
            p_w3 = tl.load(weight_t_ptr + p_d, mask=p_d_mask, other=0.0).to(tl.float32)
            p_bias = tl.load(bias_ptr + p_d, mask=p_d_mask, other=0.0).to(tl.float32)

            p_init_clear0 = ~p_bos_t0
            p_acc0 = p_x_t0 * p_w0
            p_acc0 += tl.where(p_init_clear0, p_init2, 0.0) * p_w1
            p_acc0 += tl.where(p_init_clear0, p_init1, 0.0) * p_w2
            p_acc0 += tl.where(p_init_clear0, p_init0, 0.0) * p_w3
            p_acc0 += p_bias
            p_acc0 = _fast_silu(p_acc0)

            p_valid1_1 = ~p_bos_t1
            p_init_clear1 = (~p_bos_t0) & (~p_bos_t1)
            p_acc1 = p_x_t1 * p_w0
            p_acc1 += tl.where(p_valid1_1, p_x_t0, 0.0) * p_w1
            p_acc1 += tl.where(p_init_clear1, p_init2, 0.0) * p_w2
            p_acc1 += tl.where(p_init_clear1, p_init1, 0.0) * p_w3
            p_acc1 += p_bias
            p_acc1 = _fast_silu(p_acc1)

            p_valid2_1 = ~p_bos_t2
            p_valid2_2 = p_valid2_1 & (~p_bos_t1)
            p_init_clear2 = (~p_bos_t0) & (~p_bos_t1) & (~p_bos_t2)
            p_acc2 = p_x_t2 * p_w0
            p_acc2 += tl.where(p_valid2_1, p_x_t1, 0.0) * p_w1
            p_acc2 += tl.where(p_valid2_2, p_x_t0, 0.0) * p_w2
            p_acc2 += tl.where(p_init_clear2, p_init2, 0.0) * p_w3
            p_acc2 += p_bias
            p_acc2 = _fast_silu(p_acc2)

            tl.store(out_ptr + p_base0, p_acc0, mask=p_d_mask)
            tl.store(out_ptr + p_base1, p_acc1, mask=p_d_mask)
            tl.store(out_ptr + p_base2, p_acc2, mask=p_d_mask)
        else:
            t_tail_pid = boundary_pid - prefix_programs
            t_pid_d = t_tail_pid % boundary_num_d_blocks
            t_b = t_tail_pid // boundary_num_d_blocks

            t_d = t_pid_d * boundary_block_d + tl.arange(0, boundary_block_d)
            t_d_mask = t_d < dim

            t_t0 = tail_start
            t_t1 = t_t0 + 1
            t_t2 = t_t0 + 2
            t_t3 = t_t0 + 3
            t_t4 = t_t0 + 4

            t_bos_base = t_b * seqlen
            t_bos_tm2 = tl.load(bos_ptr + t_bos_base + t_t0 - 2)
            t_bos_tm1 = tl.load(bos_ptr + t_bos_base + t_t0 - 1)
            t_bos_t0 = tl.load(bos_ptr + t_bos_base + t_t0)
            t_bos_t1 = tl.load(bos_ptr + t_bos_base + t_t1)
            t_bos_t2 = tl.load(bos_ptr + t_bos_base + t_t2)
            t_bos_t3 = tl.load(bos_ptr + t_bos_base + t_t3)
            t_bos_t4 = tl.load(bos_ptr + t_bos_base + t_t4)

            t_token0 = t_b * seqlen + t_t0
            t_base0 = t_token0 * dim + t_d
            t_base1 = t_base0 + dim
            t_base2 = t_base1 + dim
            t_base3 = t_base2 + dim
            t_base4 = t_base3 + dim

            t_x_tm3 = tl.load(x_ptr + t_base0 - 3 * dim, mask=t_d_mask, other=0.0).to(tl.float32)
            t_x_tm2 = tl.load(x_ptr + t_base0 - 2 * dim, mask=t_d_mask, other=0.0).to(tl.float32)
            t_x_tm1 = tl.load(x_ptr + t_base0 - dim, mask=t_d_mask, other=0.0).to(tl.float32)
            t_x_t0 = tl.load(x_ptr + t_base0, mask=t_d_mask, other=0.0).to(tl.float32)
            t_x_t1 = tl.load(x_ptr + t_base1, mask=t_d_mask, other=0.0).to(tl.float32)

            t_w0 = tl.load(weight_t_ptr + 3 * dim + t_d, mask=t_d_mask, other=0.0).to(tl.float32)
            t_w1 = tl.load(weight_t_ptr + 2 * dim + t_d, mask=t_d_mask, other=0.0).to(tl.float32)
            t_w2 = tl.load(weight_t_ptr + dim + t_d, mask=t_d_mask, other=0.0).to(tl.float32)
            t_w3 = tl.load(weight_t_ptr + t_d, mask=t_d_mask, other=0.0).to(tl.float32)
            t_bias = tl.load(bias_ptr + t_d, mask=t_d_mask, other=0.0).to(tl.float32)

            t_valid0_1 = ~t_bos_t0
            t_valid0_2 = t_valid0_1 & (~t_bos_tm1)
            t_valid0_3 = t_valid0_2 & (~t_bos_tm2)
            t_acc0 = t_x_t0 * t_w0 + t_bias
            t_acc0 += tl.where(t_valid0_1, t_x_tm1, 0.0) * t_w1
            t_acc0 += tl.where(t_valid0_2, t_x_tm2, 0.0) * t_w2
            t_acc0 += tl.where(t_valid0_3, t_x_tm3, 0.0) * t_w3
            t_acc0 = _fast_silu(t_acc0)
            tl.store(out_ptr + t_base0, t_acc0, mask=t_d_mask)

            t_x_t2 = tl.load(x_ptr + t_base2, mask=t_d_mask, other=0.0).to(tl.float32)
            t_valid1_1 = ~t_bos_t1
            t_valid1_2 = t_valid1_1 & (~t_bos_t0)
            t_valid1_3 = t_valid1_2 & (~t_bos_tm1)
            t_acc1 = t_x_t1 * t_w0 + t_bias
            t_acc1 += tl.where(t_valid1_1, t_x_t0, 0.0) * t_w1
            t_acc1 += tl.where(t_valid1_2, t_x_tm1, 0.0) * t_w2
            t_acc1 += tl.where(t_valid1_3, t_x_tm2, 0.0) * t_w3
            t_acc1 = _fast_silu(t_acc1)
            tl.store(out_ptr + t_base1, t_acc1, mask=t_d_mask)

            t_x_t3 = tl.load(x_ptr + t_base3, mask=t_d_mask, other=0.0).to(tl.float32)
            t_valid2_1 = ~t_bos_t2
            t_valid2_2 = t_valid2_1 & (~t_bos_t1)
            t_valid2_3 = t_valid2_2 & (~t_bos_t0)
            t_acc2 = t_x_t2 * t_w0 + t_bias
            t_acc2 += tl.where(t_valid2_1, t_x_t1, 0.0) * t_w1
            t_acc2 += tl.where(t_valid2_2, t_x_t0, 0.0) * t_w2
            t_acc2 += tl.where(t_valid2_3, t_x_tm1, 0.0) * t_w3
            t_acc2 = _fast_silu(t_acc2)
            tl.store(out_ptr + t_base2, t_acc2, mask=t_d_mask)

            t_x_t4 = tl.load(x_ptr + t_base4, mask=t_d_mask, other=0.0).to(tl.float32)
            t_valid3_1 = ~t_bos_t3
            t_valid3_2 = t_valid3_1 & (~t_bos_t2)
            t_valid3_3 = t_valid3_2 & (~t_bos_t1)
            t_acc3 = t_x_t3 * t_w0 + t_bias
            t_acc3 += tl.where(t_valid3_1, t_x_t2, 0.0) * t_w1
            t_acc3 += tl.where(t_valid3_2, t_x_t1, 0.0) * t_w2
            t_acc3 += tl.where(t_valid3_3, t_x_t0, 0.0) * t_w3
            t_acc3 = _fast_silu(t_acc3)
            tl.store(out_ptr + t_base3, t_acc3, mask=t_d_mask)

            t_valid4_1 = ~t_bos_t4
            t_valid4_2 = t_valid4_1 & (~t_bos_t3)
            t_valid4_3 = t_valid4_2 & (~t_bos_t2)
            t_acc4 = t_x_t4 * t_w0 + t_bias
            t_acc4 += tl.where(t_valid4_1, t_x_t3, 0.0) * t_w1
            t_acc4 += tl.where(t_valid4_2, t_x_t2, 0.0) * t_w2
            t_acc4 += tl.where(t_valid4_3, t_x_t1, 0.0) * t_w3
            t_acc4 = _fast_silu(t_acc4)
            tl.store(out_ptr + t_base4, t_acc4, mask=t_d_mask)

            t_final_base = t_b * dim * 3 + t_d
            t_state0 = tl.where((~t_bos_t3) & (~t_bos_t4), t_x_t2, 0.0)
            t_state1 = tl.where(~t_bos_t4, t_x_t3, 0.0)
            tl.store(final_ptr + t_final_base, t_state0, mask=t_d_mask)
            tl.store(final_ptr + t_final_base + dim, t_state1, mask=t_d_mask)
            tl.store(final_ptr + t_final_base + 2 * dim, t_x_t4, mask=t_d_mask)


@triton.jit
def _conv1d_out_kernel(
    x_ptr,
    weight_t_ptr,
    bias_ptr,
    init_ptr,
    bos_ptr,
    out_ptr,
    rows_per_batch: tl.constexpr,
    start_t: tl.constexpr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    width: tl.constexpr,
    has_bias: tl.constexpr,
    has_init: tl.constexpr,
    has_bos: tl.constexpr,
    activation_silu: tl.constexpr,
    num_d_blocks: tl.constexpr,
    block_d: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_d = pid % num_d_blocks
    row = pid // num_d_blocks
    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    b = row // rows_per_batch
    t = row - b * rows_per_batch + start_t
    token = b * seqlen + t
    base = token * dim + d

    acc = tl.load(x_ptr + base, mask=d_mask, other=0.0).to(tl.float32)
    w = tl.load(weight_t_ptr + (width - 1) * dim + d, mask=d_mask, other=0.0).to(
        tl.float32
    )
    acc *= w

    if has_bos:
        bos_base = b * seqlen
        bos_t = tl.load(bos_ptr + bos_base + t)
    else:
        bos_base = 0
        bos_t = False

    if width >= 2:
        lag = 1
        w1 = tl.load(weight_t_ptr + (width - 1 - lag) * dim + d, mask=d_mask, other=0.0).to(
            tl.float32
        )
        src_valid = t >= lag
        if has_bos:
            src_valid = src_valid & (~bos_t)
        x1 = tl.load(x_ptr + base - lag * dim, mask=d_mask & src_valid, other=0.0).to(
            tl.float32
        )
        if has_init:
            init_idx = (width - 1) + t - lag
            init_valid = t < lag
            if has_bos:
                bos0 = tl.load(bos_ptr + bos_base, mask=seqlen >= 1, other=0)
                init_valid = init_valid & (~bos0)
            init_val = tl.load(
                init_ptr + b * dim * (width - 1) + d * (width - 1) + init_idx,
                mask=d_mask & init_valid,
                other=0.0,
            ).to(tl.float32)
            x1 = tl.where(src_valid, x1, init_val)
        acc += x1 * w1

    if width >= 3:
        lag = 2
        w2 = tl.load(weight_t_ptr + (width - 1 - lag) * dim + d, mask=d_mask, other=0.0).to(
            tl.float32
        )
        src_valid = t >= lag
        if has_bos:
            bos_tm1 = tl.load(
                bos_ptr + bos_base + t - 1, mask=t >= 1, other=0
            )
            src_valid = src_valid & (~bos_t) & (~bos_tm1)
        x2 = tl.load(x_ptr + base - lag * dim, mask=d_mask & src_valid, other=0.0).to(
            tl.float32
        )
        if has_init:
            init_idx = (width - 1) + t - lag
            init_valid = t < lag
            if has_bos:
                bos0 = tl.load(bos_ptr + bos_base, mask=seqlen >= 1, other=0)
                bos1 = tl.load(bos_ptr + bos_base + 1, mask=(t >= 1) & (seqlen >= 2), other=0)
                init_valid = init_valid & (~bos0) & ((t < 1) | (~bos1))
            init_val = tl.load(
                init_ptr + b * dim * (width - 1) + d * (width - 1) + init_idx,
                mask=d_mask & init_valid,
                other=0.0,
            ).to(tl.float32)
            x2 = tl.where(src_valid, x2, init_val)
        acc += x2 * w2

    if width >= 4:
        lag = 3
        w3 = tl.load(weight_t_ptr + (width - 1 - lag) * dim + d, mask=d_mask, other=0.0).to(
            tl.float32
        )
        src_valid = t >= lag
        if has_bos:
            bos_tm1 = tl.load(
                bos_ptr + bos_base + t - 1, mask=t >= 1, other=0
            )
            bos_tm2 = tl.load(
                bos_ptr + bos_base + t - 2, mask=t >= 2, other=0
            )
            src_valid = src_valid & (~bos_t) & (~bos_tm1) & (~bos_tm2)
        x3 = tl.load(x_ptr + base - lag * dim, mask=d_mask & src_valid, other=0.0).to(
            tl.float32
        )
        if has_init:
            init_idx = (width - 1) + t - lag
            init_valid = t < lag
            if has_bos:
                bos0 = tl.load(bos_ptr + bos_base, mask=seqlen >= 1, other=0)
                bos1 = tl.load(bos_ptr + bos_base + 1, mask=(t >= 1) & (seqlen >= 2), other=0)
                bos2 = tl.load(bos_ptr + bos_base + 2, mask=(t >= 2) & (seqlen >= 3), other=0)
                init_valid = init_valid & (~bos0) & ((t < 1) | (~bos1)) & ((t < 2) | (~bos2))
            init_val = tl.load(
                init_ptr + b * dim * (width - 1) + d * (width - 1) + init_idx,
                mask=d_mask & init_valid,
                other=0.0,
            ).to(tl.float32)
            x3 = tl.where(src_valid, x3, init_val)
        acc += x3 * w3

    if has_bias:
        acc += tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    if activation_silu:
        acc = _fast_silu(acc)

    tl.store(out_ptr + base, acc, mask=d_mask)


@triton.jit
def _conv1d_prefix_t3_kernel(
    x_ptr,
    weight_t_ptr,
    bias_ptr,
    init_ptr,
    bos_ptr,
    out_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    num_d_blocks: tl.constexpr,
    block_d: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_d = pid % num_d_blocks
    b = pid // num_d_blocks

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    bos_base = b * seqlen
    bos_t0 = tl.load(bos_ptr + bos_base)
    bos_t1 = tl.load(bos_ptr + bos_base + 1)
    bos_t2 = tl.load(bos_ptr + bos_base + 2)

    token0 = b * seqlen
    base0 = token0 * dim + d
    base1 = base0 + dim
    base2 = base1 + dim

    x_t0 = tl.load(x_ptr + base0, mask=d_mask, other=0.0).to(tl.float32)
    x_t1 = tl.load(x_ptr + base1, mask=d_mask, other=0.0).to(tl.float32)
    x_t2 = tl.load(x_ptr + base2, mask=d_mask, other=0.0).to(tl.float32)

    init_base = b * dim * 3 + d * 3
    init0 = tl.load(init_ptr + init_base, mask=d_mask, other=0.0).to(tl.float32)
    init1 = tl.load(init_ptr + init_base + 1, mask=d_mask, other=0.0).to(tl.float32)
    init2 = tl.load(init_ptr + init_base + 2, mask=d_mask, other=0.0).to(tl.float32)

    w0 = tl.load(weight_t_ptr + 3 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight_t_ptr + 2 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w2 = tl.load(weight_t_ptr + dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w3 = tl.load(weight_t_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    init_clear0 = ~bos_t0
    acc0 = x_t0 * w0
    acc0 += tl.where(init_clear0, init2, 0.0) * w1
    acc0 += tl.where(init_clear0, init1, 0.0) * w2
    acc0 += tl.where(init_clear0, init0, 0.0) * w3
    acc0 += bias
    acc0 = _fast_silu(acc0)

    valid1_1 = ~bos_t1
    init_clear1 = (~bos_t0) & (~bos_t1)
    acc1 = x_t1 * w0
    acc1 += tl.where(valid1_1, x_t0, 0.0) * w1
    acc1 += tl.where(init_clear1, init2, 0.0) * w2
    acc1 += tl.where(init_clear1, init1, 0.0) * w3
    acc1 += bias
    acc1 = _fast_silu(acc1)

    valid2_1 = ~bos_t2
    valid2_2 = valid2_1 & (~bos_t1)
    init_clear2 = (~bos_t0) & (~bos_t1) & (~bos_t2)
    acc2 = x_t2 * w0
    acc2 += tl.where(valid2_1, x_t1, 0.0) * w1
    acc2 += tl.where(valid2_2, x_t0, 0.0) * w2
    acc2 += tl.where(init_clear2, init2, 0.0) * w3
    acc2 += bias
    acc2 = _fast_silu(acc2)

    tl.store(out_ptr + base0, acc0, mask=d_mask)
    tl.store(out_ptr + base1, acc1, mask=d_mask)
    tl.store(out_ptr + base2, acc2, mask=d_mask)


@triton.jit
def _conv1d_main_kernel(
    x_ptr,
    weight_t_ptr,
    bias_ptr,
    bos_ptr,
    out_ptr,
    rows_per_batch: tl.constexpr,
    start_t: tl.constexpr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    width: tl.constexpr,
    has_bias: tl.constexpr,
    has_bos: tl.constexpr,
    activation_silu: tl.constexpr,
    num_d_blocks: tl.constexpr,
    block_d: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_d = pid % num_d_blocks
    row = pid // num_d_blocks
    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    b = row // rows_per_batch
    t = row - b * rows_per_batch + start_t
    token = b * seqlen + t
    base = token * dim + d

    acc = tl.load(x_ptr + base, mask=d_mask, other=0.0).to(tl.float32)
    w = tl.load(weight_t_ptr + (width - 1) * dim + d, mask=d_mask, other=0.0).to(
        tl.float32
    )
    acc *= w

    if has_bos:
        bos_base = b * seqlen
        bos_t = tl.load(bos_ptr + bos_base + t)
    else:
        bos_base = 0
        bos_t = False

    if width >= 2:
        lag = 1
        w1 = tl.load(weight_t_ptr + (width - 1 - lag) * dim + d, mask=d_mask, other=0.0).to(
            tl.float32
        )
        valid1 = True
        if has_bos:
            valid1 = ~bos_t
        x1 = tl.load(x_ptr + base - lag * dim, mask=d_mask & valid1, other=0.0).to(
            tl.float32
        )
        acc += x1 * w1

    if width >= 3:
        lag = 2
        w2 = tl.load(weight_t_ptr + (width - 1 - lag) * dim + d, mask=d_mask, other=0.0).to(
            tl.float32
        )
        valid2 = True
        if has_bos:
            bos_tm1 = tl.load(bos_ptr + bos_base + t - 1)
            valid2 = (~bos_t) & (~bos_tm1)
        x2 = tl.load(x_ptr + base - lag * dim, mask=d_mask & valid2, other=0.0).to(
            tl.float32
        )
        acc += x2 * w2

    if width >= 4:
        lag = 3
        w3 = tl.load(weight_t_ptr + (width - 1 - lag) * dim + d, mask=d_mask, other=0.0).to(
            tl.float32
        )
        valid3 = True
        if has_bos:
            bos_tm1 = tl.load(bos_ptr + bos_base + t - 1)
            bos_tm2 = tl.load(bos_ptr + bos_base + t - 2)
            valid3 = (~bos_t) & (~bos_tm1) & (~bos_tm2)
        x3 = tl.load(x_ptr + base - lag * dim, mask=d_mask & valid3, other=0.0).to(
            tl.float32
        )
        acc += x3 * w3

    if has_bias:
        acc += tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    if activation_silu:
        acc = _fast_silu(acc)

    tl.store(out_ptr + base, acc, mask=d_mask)


@triton.jit
def _conv1d_main_t2_kernel(
    x_ptr,
    weight_t_ptr,
    bias_ptr,
    bos_ptr,
    out_ptr,
    rows_per_batch: tl.constexpr,
    tile_rows_per_batch: tl.constexpr,
    start_t: tl.constexpr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    num_d_blocks: tl.constexpr,
    block_d: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_d = pid % num_d_blocks
    tile_row = pid // num_d_blocks
    b = tile_row // tile_rows_per_batch
    tt = tile_row - b * tile_rows_per_batch
    main_t0 = tt * 2
    t0 = main_t0 + start_t
    t1 = t0 + 1
    t1_valid = (main_t0 + 1) < rows_per_batch

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    bos_base = b * seqlen
    bos_tm2 = tl.load(bos_ptr + bos_base + t0 - 2)
    bos_tm1 = tl.load(bos_ptr + bos_base + t0 - 1)
    bos_t0 = tl.load(bos_ptr + bos_base + t0)
    bos_t1 = tl.load(bos_ptr + bos_base + t1, mask=t1_valid, other=0)

    token0 = b * seqlen + t0
    base0 = token0 * dim + d
    base1 = base0 + dim

    x_tm3 = tl.load(x_ptr + base0 - 3 * dim, mask=d_mask, other=0.0).to(tl.float32)
    x_tm2 = tl.load(x_ptr + base0 - 2 * dim, mask=d_mask, other=0.0).to(tl.float32)
    x_tm1 = tl.load(x_ptr + base0 - dim, mask=d_mask, other=0.0).to(tl.float32)
    x_t0 = tl.load(x_ptr + base0, mask=d_mask, other=0.0).to(tl.float32)
    x_t1 = tl.load(x_ptr + base1, mask=d_mask & t1_valid, other=0.0).to(tl.float32)

    w0 = tl.load(weight_t_ptr + 3 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight_t_ptr + 2 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w2 = tl.load(weight_t_ptr + dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w3 = tl.load(weight_t_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    valid0_1 = ~bos_t0
    valid0_2 = valid0_1 & (~bos_tm1)
    valid0_3 = valid0_2 & (~bos_tm2)
    acc0 = x_t0 * w0
    acc0 += tl.where(valid0_1, x_tm1, 0.0) * w1
    acc0 += tl.where(valid0_2, x_tm2, 0.0) * w2
    acc0 += tl.where(valid0_3, x_tm3, 0.0) * w3
    acc0 += bias
    acc0 = _fast_silu(acc0)

    valid1_1 = ~bos_t1
    valid1_2 = valid1_1 & (~bos_t0)
    valid1_3 = valid1_2 & (~bos_tm1)
    acc1 = x_t1 * w0
    acc1 += tl.where(valid1_1, x_t0, 0.0) * w1
    acc1 += tl.where(valid1_2, x_tm1, 0.0) * w2
    acc1 += tl.where(valid1_3, x_tm2, 0.0) * w3
    acc1 += bias
    acc1 = _fast_silu(acc1)

    tl.store(out_ptr + base0, acc0, mask=d_mask)
    tl.store(out_ptr + base1, acc1, mask=d_mask & t1_valid)


@triton.jit
def _conv1d_main_t4_kernel(
    x_ptr,
    weight_t_ptr,
    bias_ptr,
    bos_ptr,
    out_ptr,
    rows_per_batch: tl.constexpr,
    tile_rows_per_batch: tl.constexpr,
    start_t: tl.constexpr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    num_d_blocks: tl.constexpr,
    block_d: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_d = pid % num_d_blocks
    tile_row = pid // num_d_blocks
    b = tile_row // tile_rows_per_batch
    tt = tile_row - b * tile_rows_per_batch
    main_t0 = tt * 4
    t0 = main_t0 + start_t
    t1 = t0 + 1
    t2 = t0 + 2
    t3 = t0 + 3
    t1_valid = (main_t0 + 1) < rows_per_batch
    t2_valid = (main_t0 + 2) < rows_per_batch
    t3_valid = (main_t0 + 3) < rows_per_batch

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    bos_base = b * seqlen
    bos_tm2 = tl.load(bos_ptr + bos_base + t0 - 2)
    bos_tm1 = tl.load(bos_ptr + bos_base + t0 - 1)
    bos_t0 = tl.load(bos_ptr + bos_base + t0)
    bos_t1 = tl.load(bos_ptr + bos_base + t1, mask=t1_valid, other=0)
    bos_t2 = tl.load(bos_ptr + bos_base + t2, mask=t2_valid, other=0)
    bos_t3 = tl.load(bos_ptr + bos_base + t3, mask=t3_valid, other=0)

    token0 = b * seqlen + t0
    base0 = token0 * dim + d
    base1 = base0 + dim
    base2 = base1 + dim
    base3 = base2 + dim

    x_tm3 = tl.load(x_ptr + base0 - 3 * dim, mask=d_mask, other=0.0).to(tl.float32)
    x_tm2 = tl.load(x_ptr + base0 - 2 * dim, mask=d_mask, other=0.0).to(tl.float32)
    x_tm1 = tl.load(x_ptr + base0 - dim, mask=d_mask, other=0.0).to(tl.float32)
    x_t0 = tl.load(x_ptr + base0, mask=d_mask, other=0.0).to(tl.float32)
    x_t1 = tl.load(x_ptr + base1, mask=d_mask & t1_valid, other=0.0).to(tl.float32)
    x_t2 = tl.load(x_ptr + base2, mask=d_mask & t2_valid, other=0.0).to(tl.float32)
    x_t3 = tl.load(x_ptr + base3, mask=d_mask & t3_valid, other=0.0).to(tl.float32)

    w0 = tl.load(weight_t_ptr + 3 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight_t_ptr + 2 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w2 = tl.load(weight_t_ptr + dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w3 = tl.load(weight_t_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    valid0_1 = ~bos_t0
    valid0_2 = valid0_1 & (~bos_tm1)
    valid0_3 = valid0_2 & (~bos_tm2)
    acc0 = x_t0 * w0
    acc0 += tl.where(valid0_1, x_tm1, 0.0) * w1
    acc0 += tl.where(valid0_2, x_tm2, 0.0) * w2
    acc0 += tl.where(valid0_3, x_tm3, 0.0) * w3
    acc0 += bias
    acc0 = _fast_silu(acc0)

    valid1_1 = ~bos_t1
    valid1_2 = valid1_1 & (~bos_t0)
    valid1_3 = valid1_2 & (~bos_tm1)
    acc1 = x_t1 * w0
    acc1 += tl.where(valid1_1, x_t0, 0.0) * w1
    acc1 += tl.where(valid1_2, x_tm1, 0.0) * w2
    acc1 += tl.where(valid1_3, x_tm2, 0.0) * w3
    acc1 += bias
    acc1 = _fast_silu(acc1)

    valid2_1 = ~bos_t2
    valid2_2 = valid2_1 & (~bos_t1)
    valid2_3 = valid2_2 & (~bos_t0)
    acc2 = x_t2 * w0
    acc2 += tl.where(valid2_1, x_t1, 0.0) * w1
    acc2 += tl.where(valid2_2, x_t0, 0.0) * w2
    acc2 += tl.where(valid2_3, x_tm1, 0.0) * w3
    acc2 += bias
    acc2 = _fast_silu(acc2)

    valid3_1 = ~bos_t3
    valid3_2 = valid3_1 & (~bos_t2)
    valid3_3 = valid3_2 & (~bos_t1)
    acc3 = x_t3 * w0
    acc3 += tl.where(valid3_1, x_t2, 0.0) * w1
    acc3 += tl.where(valid3_2, x_t1, 0.0) * w2
    acc3 += tl.where(valid3_3, x_t0, 0.0) * w3
    acc3 += bias
    acc3 = _fast_silu(acc3)

    tl.store(out_ptr + base0, acc0, mask=d_mask)
    tl.store(out_ptr + base1, acc1, mask=d_mask & t1_valid)
    tl.store(out_ptr + base2, acc2, mask=d_mask & t2_valid)
    tl.store(out_ptr + base3, acc3, mask=d_mask & t3_valid)


@triton.jit
def _conv1d_main_t8_kernel(
    x_ptr,
    weight_t_ptr,
    bias_ptr,
    bos_ptr,
    out_ptr,
    rows_per_batch: tl.constexpr,
    tile_rows_per_batch: tl.constexpr,
    start_t: tl.constexpr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    num_d_blocks: tl.constexpr,
    block_d: tl.constexpr,
    full_tile: tl.constexpr,
    store_bf16: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_d = pid % num_d_blocks
    tile_row = pid // num_d_blocks
    b = tile_row // tile_rows_per_batch
    tt = tile_row - b * tile_rows_per_batch
    main_t0 = tt * 8
    t0 = main_t0 + start_t
    t1 = t0 + 1
    t2 = t0 + 2
    t3 = t0 + 3
    t4 = t0 + 4
    t5 = t0 + 5
    t6 = t0 + 6
    t7 = t0 + 7
    if full_tile:
        t1_valid = True
        t2_valid = True
        t3_valid = True
        t4_valid = True
        t5_valid = True
        t6_valid = True
        t7_valid = True
    else:
        t1_valid = (main_t0 + 1) < rows_per_batch
        t2_valid = (main_t0 + 2) < rows_per_batch
        t3_valid = (main_t0 + 3) < rows_per_batch
        t4_valid = (main_t0 + 4) < rows_per_batch
        t5_valid = (main_t0 + 5) < rows_per_batch
        t6_valid = (main_t0 + 6) < rows_per_batch
        t7_valid = (main_t0 + 7) < rows_per_batch

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    bos_base = b * seqlen
    bos_tm2 = tl.load(bos_ptr + bos_base + t0 - 2)
    bos_tm1 = tl.load(bos_ptr + bos_base + t0 - 1)
    bos_t0 = tl.load(bos_ptr + bos_base + t0)
    if full_tile:
        bos_t1 = tl.load(bos_ptr + bos_base + t1)
        bos_t2 = tl.load(bos_ptr + bos_base + t2)
        bos_t3 = tl.load(bos_ptr + bos_base + t3)
        bos_t4 = tl.load(bos_ptr + bos_base + t4)
        bos_t5 = tl.load(bos_ptr + bos_base + t5)
        bos_t6 = tl.load(bos_ptr + bos_base + t6)
        bos_t7 = tl.load(bos_ptr + bos_base + t7)
    else:
        bos_t1 = tl.load(bos_ptr + bos_base + t1, mask=t1_valid, other=0)
        bos_t2 = tl.load(bos_ptr + bos_base + t2, mask=t2_valid, other=0)
        bos_t3 = tl.load(bos_ptr + bos_base + t3, mask=t3_valid, other=0)
        bos_t4 = tl.load(bos_ptr + bos_base + t4, mask=t4_valid, other=0)
        bos_t5 = tl.load(bos_ptr + bos_base + t5, mask=t5_valid, other=0)
        bos_t6 = tl.load(bos_ptr + bos_base + t6, mask=t6_valid, other=0)
        bos_t7 = tl.load(bos_ptr + bos_base + t7, mask=t7_valid, other=0)

    token0 = b * seqlen + t0
    base0 = token0 * dim + d
    base1 = base0 + dim
    base2 = base1 + dim
    base3 = base2 + dim
    base4 = base3 + dim
    base5 = base4 + dim
    base6 = base5 + dim
    base7 = base6 + dim

    if full_tile:
        x_tm3 = tl.load(
            x_ptr + base0 - 3 * dim,
            mask=d_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        x_tm2 = tl.load(
            x_ptr + base0 - 2 * dim,
            mask=d_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        x_tm1 = tl.load(
            x_ptr + base0 - dim,
            mask=d_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        x_t0 = tl.load(
            x_ptr + base0,
            mask=d_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        x_t1 = tl.load(
            x_ptr + base1,
            mask=d_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
    else:
        x_tm3 = tl.load(x_ptr + base0 - 3 * dim, mask=d_mask, other=0.0).to(
            tl.float32
        )
        x_tm2 = tl.load(x_ptr + base0 - 2 * dim, mask=d_mask, other=0.0).to(
            tl.float32
        )
        x_tm1 = tl.load(x_ptr + base0 - dim, mask=d_mask, other=0.0).to(tl.float32)
        x_t0 = tl.load(x_ptr + base0, mask=d_mask, other=0.0).to(tl.float32)
        x_t1 = tl.load(
            x_ptr + base1,
            mask=d_mask & t1_valid,
            other=0.0,
        ).to(tl.float32)

    w0 = tl.load(weight_t_ptr + 3 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight_t_ptr + 2 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w2 = tl.load(weight_t_ptr + dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w3 = tl.load(weight_t_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    if full_tile:
        no_bos = ~(
            bos_tm2
            | bos_tm1
            | bos_t0
            | bos_t1
            | bos_t2
            | bos_t3
            | bos_t4
            | bos_t5
            | bos_t6
            | bos_t7
        )
        if no_bos:
            x_t2 = tl.load(
                x_ptr + base2,
                mask=d_mask,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            x_t3 = tl.load(
                x_ptr + base3,
                mask=d_mask,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            _prefetch_l1(x_ptr + base4)
            x_m3 = x_tm3
            x_m2 = x_tm2
            x_m1 = x_tm1
            x_cur = x_t0
            x_next1 = x_t1
            x_next2 = x_t2
            x_next3 = x_t3
            for i in tl.static_range(0, 8):
                acc = x_cur * w0 + bias
                acc += x_m1 * w1
                acc += x_m2 * w2
                acc += x_m3 * w3
                acc = _fast_silu(acc)
                if store_bf16:
                    store_acc = acc.to(tl.bfloat16)
                else:
                    store_acc = acc
                tl.store(
                    out_ptr + base0 + i * dim,
                    store_acc,
                    mask=d_mask,
                    cache_modifier=".cg",
                )

                if i < 4:
                    x_new = tl.load(
                        x_ptr + base0 + (i + 4) * dim,
                        mask=d_mask,
                        other=0.0,
                        cache_modifier=".cg",
                    ).to(tl.float32)
                else:
                    x_new = x_next3
                x_m3 = x_m2
                x_m2 = x_m1
                x_m1 = x_cur
                x_cur = x_next1
                x_next1 = x_next2
                x_next2 = x_next3
                x_next3 = x_new
            return

    x_m3 = x_tm3
    x_m2 = x_tm2
    x_m1 = x_tm1
    x_cur = x_t0
    x_next1 = x_t1
    bos_m2 = bos_tm2
    bos_m1 = bos_tm1
    bos_cur = bos_t0
    bos_next1 = bos_t1
    row_valid_cur = True
    row_valid_next1 = t1_valid

    for i in tl.static_range(0, 8):
        valid1 = ~bos_cur
        valid2 = valid1 & (~bos_m1)
        valid3 = valid2 & (~bos_m2)
        acc = x_cur * w0 + bias
        acc += tl.where(valid1, x_m1, 0.0) * w1
        acc += tl.where(valid2, x_m2, 0.0) * w2
        acc += tl.where(valid3, x_m3, 0.0) * w3
        acc = _fast_silu(acc)
        if store_bf16:
            store_acc = acc.to(tl.bfloat16)
        else:
            store_acc = acc
        tl.store(
            out_ptr + base0 + i * dim,
            store_acc,
            mask=d_mask & row_valid_cur,
            cache_modifier=".cg",
        )

        if i < 6:
            if i == 0:
                bos_new = bos_t2
                row_valid_new = t2_valid
            elif i == 1:
                bos_new = bos_t3
                row_valid_new = t3_valid
            elif i == 2:
                bos_new = bos_t4
                row_valid_new = t4_valid
            elif i == 3:
                bos_new = bos_t5
                row_valid_new = t5_valid
            elif i == 4:
                bos_new = bos_t6
                row_valid_new = t6_valid
            else:
                bos_new = bos_t7
                row_valid_new = t7_valid

            if full_tile:
                x_new = tl.load(
                    x_ptr + base0 + (i + 2) * dim,
                    mask=d_mask,
                    other=0.0,
                    cache_modifier=".cg",
                ).to(tl.float32)
            else:
                x_new = tl.load(
                    x_ptr + base0 + (i + 2) * dim,
                    mask=d_mask & row_valid_new,
                    other=0.0,
                ).to(tl.float32)
        else:
            x_new = x_next1
            bos_new = bos_next1
            row_valid_new = row_valid_next1

        x_m3 = x_m2
        x_m2 = x_m1
        x_m1 = x_cur
        x_cur = x_next1
        x_next1 = x_new
        bos_m2 = bos_m1
        bos_m1 = bos_cur
        bos_cur = bos_next1
        bos_next1 = bos_new
        row_valid_cur = row_valid_next1
        row_valid_next1 = row_valid_new


@triton.jit
def _conv1d_main_t5_tail_kernel(
    x_ptr,
    weight_t_ptr,
    bias_ptr,
    bos_ptr,
    out_ptr,
    final_ptr,
    start_t: tl.constexpr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    num_d_blocks: tl.constexpr,
    block_d: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_d = pid % num_d_blocks
    b = pid // num_d_blocks

    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    t0 = start_t
    t1 = t0 + 1
    t2 = t0 + 2
    t3 = t0 + 3
    t4 = t0 + 4

    bos_base = b * seqlen
    bos_tm2 = tl.load(bos_ptr + bos_base + t0 - 2)
    bos_tm1 = tl.load(bos_ptr + bos_base + t0 - 1)
    bos_t0 = tl.load(bos_ptr + bos_base + t0)
    bos_t1 = tl.load(bos_ptr + bos_base + t1)
    bos_t2 = tl.load(bos_ptr + bos_base + t2)
    bos_t3 = tl.load(bos_ptr + bos_base + t3)
    bos_t4 = tl.load(bos_ptr + bos_base + t4)

    token0 = b * seqlen + t0
    base0 = token0 * dim + d
    base1 = base0 + dim
    base2 = base1 + dim
    base3 = base2 + dim
    base4 = base3 + dim

    x_tm3 = tl.load(x_ptr + base0 - 3 * dim, mask=d_mask, other=0.0).to(tl.float32)
    x_tm2 = tl.load(x_ptr + base0 - 2 * dim, mask=d_mask, other=0.0).to(tl.float32)
    x_tm1 = tl.load(x_ptr + base0 - dim, mask=d_mask, other=0.0).to(tl.float32)
    x_t0 = tl.load(x_ptr + base0, mask=d_mask, other=0.0).to(tl.float32)
    x_t1 = tl.load(x_ptr + base1, mask=d_mask, other=0.0).to(tl.float32)

    w0 = tl.load(weight_t_ptr + 3 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight_t_ptr + 2 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w2 = tl.load(weight_t_ptr + dim + d, mask=d_mask, other=0.0).to(tl.float32)
    w3 = tl.load(weight_t_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    valid0_1 = ~bos_t0
    valid0_2 = valid0_1 & (~bos_tm1)
    valid0_3 = valid0_2 & (~bos_tm2)
    acc = x_t0 * w0 + bias
    acc += tl.where(valid0_1, x_tm1, 0.0) * w1
    acc += tl.where(valid0_2, x_tm2, 0.0) * w2
    acc += tl.where(valid0_3, x_tm3, 0.0) * w3
    acc = _fast_silu(acc)
    tl.store(out_ptr + base0, acc, mask=d_mask)

    x_t2 = tl.load(x_ptr + base2, mask=d_mask, other=0.0).to(tl.float32)
    valid1_1 = ~bos_t1
    valid1_2 = valid1_1 & (~bos_t0)
    valid1_3 = valid1_2 & (~bos_tm1)
    acc = x_t1 * w0 + bias
    acc += tl.where(valid1_1, x_t0, 0.0) * w1
    acc += tl.where(valid1_2, x_tm1, 0.0) * w2
    acc += tl.where(valid1_3, x_tm2, 0.0) * w3
    acc = _fast_silu(acc)
    tl.store(out_ptr + base1, acc, mask=d_mask)

    x_t3 = tl.load(x_ptr + base3, mask=d_mask, other=0.0).to(tl.float32)
    valid2_1 = ~bos_t2
    valid2_2 = valid2_1 & (~bos_t1)
    valid2_3 = valid2_2 & (~bos_t0)
    acc = x_t2 * w0 + bias
    acc += tl.where(valid2_1, x_t1, 0.0) * w1
    acc += tl.where(valid2_2, x_t0, 0.0) * w2
    acc += tl.where(valid2_3, x_tm1, 0.0) * w3
    acc = _fast_silu(acc)
    tl.store(out_ptr + base2, acc, mask=d_mask)

    x_t4 = tl.load(x_ptr + base4, mask=d_mask, other=0.0).to(tl.float32)
    valid3_1 = ~bos_t3
    valid3_2 = valid3_1 & (~bos_t2)
    valid3_3 = valid3_2 & (~bos_t1)
    acc = x_t3 * w0 + bias
    acc += tl.where(valid3_1, x_t2, 0.0) * w1
    acc += tl.where(valid3_2, x_t1, 0.0) * w2
    acc += tl.where(valid3_3, x_t0, 0.0) * w3
    acc = _fast_silu(acc)
    tl.store(out_ptr + base3, acc, mask=d_mask)

    valid4_1 = ~bos_t4
    valid4_2 = valid4_1 & (~bos_t3)
    valid4_3 = valid4_2 & (~bos_t2)
    acc = x_t4 * w0 + bias
    acc += tl.where(valid4_1, x_t3, 0.0) * w1
    acc += tl.where(valid4_2, x_t2, 0.0) * w2
    acc += tl.where(valid4_3, x_t1, 0.0) * w3
    acc = _fast_silu(acc)
    tl.store(out_ptr + base4, acc, mask=d_mask)

    final_base = b * dim * 3 + d
    state0 = tl.where((~bos_t3) & (~bos_t4), x_t2, 0.0)
    state1 = tl.where(~bos_t4, x_t3, 0.0)
    tl.store(final_ptr + final_base, state0, mask=d_mask)
    tl.store(final_ptr + final_base + dim, state1, mask=d_mask)
    tl.store(final_ptr + final_base + 2 * dim, x_t4, mask=d_mask)


@triton.jit
def _conv1d_prefix_t3_tail_t5_kernel(
    x_ptr,
    weight_t_ptr,
    bias_ptr,
    init_ptr,
    bos_ptr,
    out_ptr,
    final_ptr,
    tail_start: tl.constexpr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    num_d_blocks: tl.constexpr,
    block_d: tl.constexpr,
    prefix_programs: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < prefix_programs:
        pid_d = pid % num_d_blocks
        b = pid // num_d_blocks

        d = pid_d * block_d + tl.arange(0, block_d)
        d_mask = d < dim

        bos_base = b * seqlen
        bos_t0 = tl.load(bos_ptr + bos_base)
        bos_t1 = tl.load(bos_ptr + bos_base + 1)
        bos_t2 = tl.load(bos_ptr + bos_base + 2)

        token0 = b * seqlen
        base0 = token0 * dim + d
        base1 = base0 + dim
        base2 = base1 + dim

        x_t0 = tl.load(x_ptr + base0, mask=d_mask, other=0.0).to(tl.float32)
        x_t1 = tl.load(x_ptr + base1, mask=d_mask, other=0.0).to(tl.float32)
        x_t2 = tl.load(x_ptr + base2, mask=d_mask, other=0.0).to(tl.float32)

        init_base = b * dim * 3 + d * 3
        init0 = tl.load(init_ptr + init_base, mask=d_mask, other=0.0).to(tl.float32)
        init1 = tl.load(init_ptr + init_base + 1, mask=d_mask, other=0.0).to(tl.float32)
        init2 = tl.load(init_ptr + init_base + 2, mask=d_mask, other=0.0).to(tl.float32)

        w0 = tl.load(weight_t_ptr + 3 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
        w1 = tl.load(weight_t_ptr + 2 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
        w2 = tl.load(weight_t_ptr + dim + d, mask=d_mask, other=0.0).to(tl.float32)
        w3 = tl.load(weight_t_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

        init_clear0 = ~bos_t0
        acc0 = x_t0 * w0
        acc0 += tl.where(init_clear0, init2, 0.0) * w1
        acc0 += tl.where(init_clear0, init1, 0.0) * w2
        acc0 += tl.where(init_clear0, init0, 0.0) * w3
        acc0 += bias
        acc0 = _fast_silu(acc0)

        valid1_1 = ~bos_t1
        init_clear1 = (~bos_t0) & (~bos_t1)
        acc1 = x_t1 * w0
        acc1 += tl.where(valid1_1, x_t0, 0.0) * w1
        acc1 += tl.where(init_clear1, init2, 0.0) * w2
        acc1 += tl.where(init_clear1, init1, 0.0) * w3
        acc1 += bias
        acc1 = _fast_silu(acc1)

        valid2_1 = ~bos_t2
        valid2_2 = valid2_1 & (~bos_t1)
        init_clear2 = (~bos_t0) & (~bos_t1) & (~bos_t2)
        acc2 = x_t2 * w0
        acc2 += tl.where(valid2_1, x_t1, 0.0) * w1
        acc2 += tl.where(valid2_2, x_t0, 0.0) * w2
        acc2 += tl.where(init_clear2, init2, 0.0) * w3
        acc2 += bias
        acc2 = _fast_silu(acc2)

        tl.store(out_ptr + base0, acc0, mask=d_mask)
        tl.store(out_ptr + base1, acc1, mask=d_mask)
        tl.store(out_ptr + base2, acc2, mask=d_mask)
    else:
        tail_pid = pid - prefix_programs
        pid_d = tail_pid % num_d_blocks
        b = tail_pid // num_d_blocks

        d = pid_d * block_d + tl.arange(0, block_d)
        d_mask = d < dim

        t0 = tail_start
        t1 = t0 + 1
        t2 = t0 + 2
        t3 = t0 + 3
        t4 = t0 + 4

        bos_base = b * seqlen
        bos_tm2 = tl.load(bos_ptr + bos_base + t0 - 2)
        bos_tm1 = tl.load(bos_ptr + bos_base + t0 - 1)
        bos_t0 = tl.load(bos_ptr + bos_base + t0)
        bos_t1 = tl.load(bos_ptr + bos_base + t1)
        bos_t2 = tl.load(bos_ptr + bos_base + t2)
        bos_t3 = tl.load(bos_ptr + bos_base + t3)
        bos_t4 = tl.load(bos_ptr + bos_base + t4)

        token0 = b * seqlen + t0
        base0 = token0 * dim + d
        base1 = base0 + dim
        base2 = base1 + dim
        base3 = base2 + dim
        base4 = base3 + dim

        x_tm3 = tl.load(x_ptr + base0 - 3 * dim, mask=d_mask, other=0.0).to(tl.float32)
        x_tm2 = tl.load(x_ptr + base0 - 2 * dim, mask=d_mask, other=0.0).to(tl.float32)
        x_tm1 = tl.load(x_ptr + base0 - dim, mask=d_mask, other=0.0).to(tl.float32)
        x_t0 = tl.load(x_ptr + base0, mask=d_mask, other=0.0).to(tl.float32)
        x_t1 = tl.load(x_ptr + base1, mask=d_mask, other=0.0).to(tl.float32)
        x_t2 = tl.load(x_ptr + base2, mask=d_mask, other=0.0).to(tl.float32)
        x_t3 = tl.load(x_ptr + base3, mask=d_mask, other=0.0).to(tl.float32)
        x_t4 = tl.load(x_ptr + base4, mask=d_mask, other=0.0).to(tl.float32)

        w0 = tl.load(weight_t_ptr + 3 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
        w1 = tl.load(weight_t_ptr + 2 * dim + d, mask=d_mask, other=0.0).to(tl.float32)
        w2 = tl.load(weight_t_ptr + dim + d, mask=d_mask, other=0.0).to(tl.float32)
        w3 = tl.load(weight_t_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

        valid0_1 = ~bos_t0
        valid0_2 = valid0_1 & (~bos_tm1)
        valid0_3 = valid0_2 & (~bos_tm2)
        acc = x_t0 * w0 + bias
        acc += tl.where(valid0_1, x_tm1, 0.0) * w1
        acc += tl.where(valid0_2, x_tm2, 0.0) * w2
        acc += tl.where(valid0_3, x_tm3, 0.0) * w3
        acc = _fast_silu(acc)
        tl.store(out_ptr + base0, acc, mask=d_mask)

        valid1_1 = ~bos_t1
        valid1_2 = valid1_1 & (~bos_t0)
        valid1_3 = valid1_2 & (~bos_tm1)
        acc = x_t1 * w0 + bias
        acc += tl.where(valid1_1, x_t0, 0.0) * w1
        acc += tl.where(valid1_2, x_tm1, 0.0) * w2
        acc += tl.where(valid1_3, x_tm2, 0.0) * w3
        acc = _fast_silu(acc)
        tl.store(out_ptr + base1, acc, mask=d_mask)

        final_base = b * dim * 3 + d
        valid2_1 = ~bos_t2
        valid2_2 = valid2_1 & (~bos_t1)
        valid2_3 = valid2_2 & (~bos_t0)
        acc = x_t2 * w0 + bias
        acc += tl.where(valid2_1, x_t1, 0.0) * w1
        acc += tl.where(valid2_2, x_t0, 0.0) * w2
        acc += tl.where(valid2_3, x_tm1, 0.0) * w3
        acc = _fast_silu(acc)
        tl.store(out_ptr + base2, acc, mask=d_mask)

        state0 = tl.where((~bos_t3) & (~bos_t4), x_t2, 0.0)
        state1 = tl.where(~bos_t4, x_t3, 0.0)
        tl.store(final_ptr + final_base, state0, mask=d_mask, cache_modifier=".cg")
        tl.store(
            final_ptr + final_base + dim,
            state1,
            mask=d_mask,
            cache_modifier=".cg",
        )
        tl.store(
            final_ptr + final_base + 2 * dim,
            x_t4,
            mask=d_mask,
            cache_modifier=".cg",
        )

        valid3_1 = ~bos_t3
        valid3_2 = valid3_1 & (~bos_t2)
        valid3_3 = valid3_2 & (~bos_t1)
        acc = x_t3 * w0 + bias
        acc += tl.where(valid3_1, x_t2, 0.0) * w1
        acc += tl.where(valid3_2, x_t1, 0.0) * w2
        acc += tl.where(valid3_3, x_t0, 0.0) * w3
        acc = _fast_silu(acc)
        tl.store(out_ptr + base3, acc, mask=d_mask)

        valid4_1 = ~bos_t4
        valid4_2 = valid4_1 & (~bos_t3)
        valid4_3 = valid4_2 & (~bos_t2)
        acc = x_t4 * w0 + bias
        acc += tl.where(valid4_1, x_t3, 0.0) * w1
        acc += tl.where(valid4_2, x_t2, 0.0) * w2
        acc += tl.where(valid4_3, x_t1, 0.0) * w3
        acc = _fast_silu(acc)
        tl.store(out_ptr + base4, acc, mask=d_mask)


@triton.jit
def _final_states_kernel(
    x_ptr,
    init_ptr,
    bos_ptr,
    final_ptr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    width: tl.constexpr,
    has_init: tl.constexpr,
    has_bos: tl.constexpr,
    block_d: tl.constexpr,
):
    pid_d = tl.program_id(0)
    b = tl.program_id(1)
    s = tl.program_id(2)
    d = pid_d * block_d + tl.arange(0, block_d)
    d_mask = d < dim

    source_t = seqlen - ((width - 1) - s)
    x_valid = source_t >= 0
    value = tl.load(
        x_ptr + (b * seqlen + source_t) * dim + d,
        mask=d_mask & x_valid,
        other=0.0,
    )

    if has_bos:
        bos_base = b * seqlen
        tail_clear = True
        if width >= 2:
            pos = source_t + 1
            check = x_valid & (pos < seqlen)
            bos = tl.load(bos_ptr + bos_base + pos, mask=check, other=0)
            tail_clear = tail_clear & ((~check) | (~bos))
        if width >= 3:
            pos = source_t + 2
            check = x_valid & (pos < seqlen)
            bos = tl.load(bos_ptr + bos_base + pos, mask=check, other=0)
            tail_clear = tail_clear & ((~check) | (~bos))
        if width >= 4:
            pos = source_t + 3
            check = x_valid & (pos < seqlen)
            bos = tl.load(bos_ptr + bos_base + pos, mask=check, other=0)
            tail_clear = tail_clear & ((~check) | (~bos))
        value = tl.where(tail_clear, value, 0.0)

    if has_init:
        init_idx = s + seqlen
        init_valid = source_t < 0
        init_valid = init_valid & (init_idx < (width - 1))
        if has_bos:
            prefix_clear = True
            if width >= 2:
                check = seqlen >= 1
                bos = tl.load(bos_ptr + b * seqlen, mask=check, other=0)
                prefix_clear = prefix_clear & ((~check) | (~bos))
            if width >= 3:
                check = seqlen >= 2
                bos = tl.load(bos_ptr + b * seqlen + 1, mask=check, other=0)
                prefix_clear = prefix_clear & ((~check) | (~bos))
            if width >= 4:
                check = seqlen >= 3
                bos = tl.load(bos_ptr + b * seqlen + 2, mask=check, other=0)
                prefix_clear = prefix_clear & ((~check) | (~bos))
            init_valid = init_valid & prefix_clear
        init_val = tl.load(
            init_ptr + b * dim * (width - 1) + d * (width - 1) + init_idx,
            mask=d_mask & init_valid,
            other=0.0,
        )
        value = tl.where(x_valid, value, init_val)

    tl.store(final_ptr + b * dim * (width - 1) + s * dim + d, value, mask=d_mask)


def kernel_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    bos_mask: torch.Tensor | None = None,
    activation: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if activation not in (None, "silu"):
        return reference_kernel_fn(x, weight, bias, initial_states, bos_mask, activation)
    if not x.is_cuda:
        return reference_kernel_fn(x, weight, bias, initial_states, bos_mask, activation)
    if x.ndim != 3 or weight.ndim != 2:
        return reference_kernel_fn(x, weight, bias, initial_states, bos_mask, activation)
    if x.dtype not in (torch.bfloat16, torch.float32):
        return reference_kernel_fn(x, weight, bias, initial_states, bos_mask, activation)

    batch, seqlen, dim = x.shape
    width, weight_dim = weight.shape
    bos_pos = None
    bos_batch = None
    bos_next_pos = None
    bos_packed_all = None
    dirty_row_mask = None
    if isinstance(bos_mask, tuple):
        if len(bos_mask) not in (2, 3, 4, 5):
            return reference_kernel_fn(x, weight, bias, initial_states, bos_mask, activation)
        if len(bos_mask) == 2:
            dense_bos_mask, bos_packed_all = bos_mask
        elif (
            len(bos_mask) == 3
            and bos_mask[1].dtype == torch.int64
            and bos_mask[2].dtype == torch.uint8
        ):
            dense_bos_mask, bos_packed_all, dirty_row_mask = bos_mask
        else:
            dense_bos_mask, bos_pos, bos_batch = bos_mask[:3]
            if len(bos_mask) >= 4:
                bos_next_pos = bos_mask[3]
            if len(bos_mask) == 5:
                bos_packed_all = bos_mask[4]
        bos_mask = dense_bos_mask
    if weight_dim != dim or width < 2 or width > 4:
        return reference_kernel_fn(x, weight, bias, initial_states, bos_mask, activation)
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != dim):
        return reference_kernel_fn(x, weight, bias, initial_states, bos_mask, activation)
    if initial_states is not None and initial_states.shape != (batch, dim, width - 1):
        return reference_kernel_fn(x, weight, bias, initial_states, bos_mask, activation)
    if bos_mask is not None and bos_mask.shape != (batch, seqlen):
        return reference_kernel_fn(x, weight, bias, initial_states, bos_mask, activation)
    has_bos_offsets = (
        bos_pos is not None
        and bos_batch is not None
        and bos_pos.ndim == 1
        and bos_batch.ndim == 1
        and bos_pos.shape == bos_batch.shape
        and bos_pos.dtype == torch.int32
        and bos_batch.dtype == torch.int32
        and bos_next_pos is not None
        and bos_next_pos.ndim == 1
        and bos_next_pos.shape == bos_pos.shape
        and bos_next_pos.dtype == torch.int32
    )
    has_bos_packed_all = (
        bos_packed_all is not None
        and bos_packed_all.ndim == 1
        and bos_packed_all.dtype == torch.int64
        and (
            (has_bos_offsets and bos_packed_all.shape == bos_pos.shape)
            or (
                bos_pos is None
                and bos_batch is None
                and bos_next_pos is None
            )
        )
    )
    has_dirty_row_mask = (
        dirty_row_mask is not None
        and dirty_row_mask.ndim == 2
        and dirty_row_mask.shape[0] == batch
        and dirty_row_mask.dtype == torch.uint8
    )

    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()
    if initial_states is not None and not initial_states.is_contiguous():
        initial_states = initial_states.contiguous()
    if bos_mask is not None and not bos_mask.is_contiguous():
        bos_mask = bos_mask.contiguous()
    if has_bos_offsets and not bos_pos.is_contiguous():
        bos_pos = bos_pos.contiguous()
    if has_bos_offsets and not bos_batch.is_contiguous():
        bos_batch = bos_batch.contiguous()
    if has_bos_offsets and not bos_next_pos.is_contiguous():
        bos_next_pos = bos_next_pos.contiguous()
    if has_bos_packed_all and not bos_packed_all.is_contiguous():
        bos_packed_all = bos_packed_all.contiguous()
    if has_dirty_row_mask and not dirty_row_mask.is_contiguous():
        dirty_row_mask = dirty_row_mask.contiguous()
    weight_t = weight

    out = torch.empty_like(x)
    final_states = torch.empty_strided(
        (batch, dim, width - 1),
        (dim * (width - 1), 1, dim),
        device=x.device,
        dtype=x.dtype,
    )

    block_d = 2048
    num_d_blocks = triton.cdiv(dim, block_d)
    dummy = x

    prefix_rows = min(width - 1, seqlen)
    main_rows = seqlen - prefix_rows
    use_prefix_t3 = False
    combine_prefix_t5_tail = False
    if prefix_rows > 0:
        use_prefix_t3 = (
            prefix_rows == 3
            and width == 4
            and bias is not None
            and initial_states is not None
            and bos_mask is not None
            and activation == "silu"
        )
        if use_prefix_t3:
            prefix_block_d = 128
            prefix_num_d_blocks = triton.cdiv(dim, prefix_block_d)
            main_full_tile_rows_for_prefix = main_rows // 8
            main_tail_rows_for_prefix = main_rows - main_full_tile_rows_for_prefix * 8
            combine_prefix_t5_tail = main_rows > 0 and main_tail_rows_for_prefix == 5
            if not combine_prefix_t5_tail:
                prefix_grid = (prefix_num_d_blocks * batch,)
                _conv1d_prefix_t3_kernel[prefix_grid](
                    x,
                    weight_t,
                    bias,
                    initial_states,
                    bos_mask,
                    out,
                    seqlen,
                    dim,
                    prefix_num_d_blocks,
                    prefix_block_d,
                    num_warps=2,
                    num_stages=1,
                )
        else:
            prefix_grid = (num_d_blocks * batch * prefix_rows,)
            _conv1d_out_kernel[prefix_grid](
                x,
                weight_t,
                bias if bias is not None else dummy,
                initial_states if initial_states is not None else dummy,
                bos_mask if bos_mask is not None else dummy,
                out,
                prefix_rows,
                0,
                seqlen,
                dim,
                width,
                bias is not None,
                initial_states is not None,
                bos_mask is not None,
                activation == "silu",
                num_d_blocks,
                block_d,
                num_warps=8,
            )

    final_states_done = False
    fused_repair_boundary = False
    if main_rows > 0:
        if width == 4 and bias is not None and bos_mask is not None and activation == "silu":
            main_block_d = 256
            main_num_d_blocks = triton.cdiv(dim, main_block_d)
            main_full_tile_rows = main_rows // 8
            main_tail_rows = main_rows - main_full_tile_rows * 8
            if main_full_tile_rows > 0:
                if has_bos_offsets or has_bos_packed_all:
                    swizzle16_rows = 1
                    full_main_swizzle_group_rows = 8
                    full_swizzle_rows = (
                        main_full_tile_rows // swizzle16_rows
                    ) * swizzle16_rows
                    tail_swizzle_rows = main_full_tile_rows - full_swizzle_rows
                    total_bos = (
                        bos_packed_all.shape[0]
                        if has_bos_packed_all
                        else bos_pos.shape[0]
                    )
                    fuse_tail_main_into_secondary = (
                        has_bos_packed_all
                        and combine_prefix_t5_tail
                        and total_bos > 0
                        and tail_swizzle_rows > 0
                    )
                    use_full_main_dirty_store_mask = (
                        has_dirty_row_mask
                        and has_bos_packed_all
                        and dirty_row_mask.shape[1] >= main_full_tile_rows
                    )
                    repair_dirty_in_full_main = (
                        use_full_main_dirty_store_mask
                        and full_swizzle_rows == main_full_tile_rows
                    )
                    fused_tail_main_programs = 0
                    fused_tail_start_t = 0
                    fused_tail_tile_rows = 1
                    if full_swizzle_rows > 0:
                        full_main_grid = (
                            main_num_d_blocks * batch * full_swizzle_rows,
                        )
                        _conv1d_main_t8_nobos_kernel[full_main_grid](
                            x,
                            weight_t,
                            bias,
                            out,
                            dirty_row_mask
                            if use_full_main_dirty_store_mask
                            else dummy,
                            bos_mask,
                            full_swizzle_rows,
                            prefix_rows,
                            seqlen,
                            dim,
                            main_num_d_blocks,
                            main_block_d,
                            full_main_swizzle_group_rows,
                            x.dtype == torch.bfloat16,
                            use_full_main_dirty_store_mask,
                            main_full_tile_rows,
                            0,
                            num_warps=2,
                            num_stages=1,
                            maxnreg=56,
                        )
                    if tail_swizzle_rows > 0:
                        tail_main_grid = (
                            main_num_d_blocks * batch * tail_swizzle_rows,
                        )
                        tail_start_t = prefix_rows + full_swizzle_rows * 8
                        if fuse_tail_main_into_secondary:
                            fused_tail_main_programs = tail_main_grid[0]
                            fused_tail_start_t = tail_start_t
                            fused_tail_tile_rows = tail_swizzle_rows
                        else:
                            tail_swizzle_group = (
                                8 if (batch * tail_swizzle_rows) % 8 == 0 else 0
                            )
                            _conv1d_main_t8_nobos_kernel[tail_main_grid](
                                x,
                                weight_t,
                                bias,
                                out,
                                dummy,
                                bos_mask if bos_mask is not None else dummy,
                                tail_swizzle_rows,
                                tail_start_t,
                                seqlen,
                                dim,
                                main_num_d_blocks,
                                main_block_d,
                                tail_swizzle_group,
                                x.dtype == torch.bfloat16,
                                False,
                                1,
                                0,
                                num_warps=2,
                                num_stages=1,
                                maxnreg=56,
                            )
                    if total_bos > 0 and not repair_dirty_in_full_main:
                        repair_block_d = 512
                        repair_num_d_blocks = triton.cdiv(dim, repair_block_d)
                        repair_grid = (repair_num_d_blocks * total_bos,)
                        if has_bos_packed_all and combine_prefix_t5_tail:
                            boundary_block_d = 64
                            boundary_num_d_blocks = triton.cdiv(dim, boundary_block_d)
                            repair_programs = repair_num_d_blocks * total_bos
                            boundary_programs = boundary_num_d_blocks * batch
                            secondary_grid = (
                                fused_tail_main_programs
                                + repair_programs
                                + boundary_programs * 2,
                            )
                            _conv1d_packed_repair_prefix_t3_tail_t5_kernel[secondary_grid](
                                x,
                                weight_t,
                                bias,
                                initial_states,
                                bos_mask,
                                bos_packed_all,
                                out,
                                final_states,
                                dirty_row_mask
                                if use_full_main_dirty_store_mask
                                else dummy,
                                prefix_rows,
                                prefix_rows + main_full_tile_rows * 8,
                                prefix_rows + main_full_tile_rows * 8,
                                seqlen,
                                dim,
                                main_full_tile_rows,
                                full_swizzle_rows,
                                repair_num_d_blocks,
                                repair_block_d,
                                boundary_num_d_blocks,
                                boundary_block_d,
                                main_num_d_blocks,
                                main_block_d,
                                fused_tail_tile_rows,
                                fused_tail_start_t,
                                fused_tail_main_programs,
                                repair_programs,
                                total_bos,
                                boundary_programs,
                                x.dtype == torch.bfloat16,
                                use_full_main_dirty_store_mask,
                                num_warps=2,
                                num_stages=1,
                                maxnreg=56,
                            )
                            fused_repair_boundary = True
                        elif has_bos_packed_all:
                            _conv1d_main_t8_offsets_repair_packed_all_kernel[repair_grid](
                                x,
                                weight_t,
                                bias,
                                bos_packed_all,
                                out,
                                prefix_rows,
                                prefix_rows + main_full_tile_rows * 8,
                                seqlen,
                                dim,
                                repair_num_d_blocks,
                                repair_block_d,
                                x.dtype == torch.bfloat16,
                                num_warps=2,
                                num_stages=1,
                                maxnreg=56,
                            )
                        else:
                            _conv1d_main_t8_offsets_repair_kernel[repair_grid](
                                x,
                                weight_t,
                                bias,
                                bos_next_pos,
                                bos_pos,
                                bos_batch,
                                bos_next_pos,
                                out,
                                prefix_rows,
                                prefix_rows + main_full_tile_rows * 8,
                                seqlen,
                                dim,
                                repair_num_d_blocks,
                                repair_block_d,
                                x.dtype == torch.bfloat16,
                                False,
                                num_warps=2,
                                num_stages=1,
                                maxnreg=56,
                            )
                else:
                    main_grid = (main_num_d_blocks * batch * main_full_tile_rows,)
                    _conv1d_main_t8_kernel[main_grid](
                        x,
                        weight_t,
                        bias,
                        bos_mask,
                        out,
                        main_full_tile_rows * 8,
                        main_full_tile_rows,
                        prefix_rows,
                        seqlen,
                        dim,
                        main_num_d_blocks,
                        main_block_d,
                        True,
                        x.dtype == torch.bfloat16,
                        num_warps=2,
                        num_stages=1,
                        maxnreg=56,
                    )
            if main_tail_rows == 5:
                tail_block_d = 128
                tail_num_d_blocks = triton.cdiv(dim, tail_block_d)
                tail_grid = (tail_num_d_blocks * batch,)
                if combine_prefix_t5_tail:
                    if not fused_repair_boundary:
                        boundary_block_d = 64
                        boundary_num_d_blocks = triton.cdiv(dim, boundary_block_d)
                        boundary_programs = boundary_num_d_blocks * batch
                        boundary_grid = (boundary_programs * 2,)
                        _conv1d_prefix_t3_tail_t5_kernel[boundary_grid](
                            x,
                            weight_t,
                            bias,
                            initial_states,
                            bos_mask,
                            out,
                            final_states,
                            prefix_rows + main_full_tile_rows * 8,
                            seqlen,
                            dim,
                            boundary_num_d_blocks,
                            boundary_block_d,
                            boundary_programs,
                            num_warps=1,
                            num_stages=1,
                        )
                else:
                    _conv1d_main_t5_tail_kernel[tail_grid](
                        x,
                        weight_t,
                        bias,
                        bos_mask,
                        out,
                        final_states,
                        prefix_rows + main_full_tile_rows * 8,
                        seqlen,
                        dim,
                        tail_num_d_blocks,
                        tail_block_d,
                        num_warps=1,
                        num_stages=1,
                    )
                final_states_done = True
            elif main_tail_rows > 0:
                tail_grid = (main_num_d_blocks * batch,)
                _conv1d_main_t8_kernel[tail_grid](
                    x,
                    weight_t,
                    bias,
                    bos_mask,
                    out,
                    main_tail_rows,
                    1,
                    prefix_rows + main_full_tile_rows * 8,
                    seqlen,
                    dim,
                    main_num_d_blocks,
                    main_block_d,
                    False,
                    x.dtype == torch.bfloat16,
                    num_warps=2,
                    num_stages=1,
                    maxnreg=56,
                )
        else:
            main_grid = (num_d_blocks * batch * main_rows,)
            _conv1d_main_kernel[main_grid](
                x,
                weight_t,
                bias if bias is not None else dummy,
                bos_mask if bos_mask is not None else dummy,
                out,
                main_rows,
                prefix_rows,
                seqlen,
                dim,
                width,
                bias is not None,
                bos_mask is not None,
                activation == "silu",
                num_d_blocks,
                block_d,
                num_warps=2,
            )

    if not final_states_done:
        final_grid = (num_d_blocks, batch, width - 1)
        _final_states_kernel[final_grid](
            x,
            initial_states if initial_states is not None else dummy,
            bos_mask if bos_mask is not None else dummy,
            final_states,
            seqlen,
            dim,
            width,
            initial_states is not None,
            bos_mask is not None,
            block_d,
            num_warps=8,
        )

    return out, final_states
