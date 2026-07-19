from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline


_MODULE = None


def _load_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    cpp_src = r"""
    #include <torch/extension.h>

    void final_reduce(torch::Tensor partial, torch::Tensor dweight, torch::Tensor dbias,
                      int64_t dim, int64_t total_chunks);
    """

    cuda_src = r"""
    #include <torch/extension.h>
    #include <ATen/cuda/CUDAContext.h>
    #include <c10/cuda/CUDAException.h>
    #include <cuda_bf16.h>
    #include <cuda_runtime.h>

    __global__ void final_reduce_kernel(
        const float* __restrict__ partial,
        __nv_bfloat16* __restrict__ dweight,
        __nv_bfloat16* __restrict__ dbias,
        int dim,
        int total_chunks) {
        const int lane_d = threadIdx.x & 31;
        const int r = threadIdx.x >> 5;
        const int d = blockIdx.x * 32 + lane_d;
        const int plane = blockIdx.y;

        float acc = 0.0f;
        if (d < dim) {
            const int plane_base = plane * total_chunks * dim;
            for (int c = r; c < total_chunks; c += 8) {
                acc += partial[plane_base + c * dim + d];
            }
        }

        __shared__ float warp_sums[8][32];
        warp_sums[r][lane_d] = acc;
        __syncthreads();

        if (r == 0 && d < dim) {
            float sum = warp_sums[0][lane_d];
            #pragma unroll
            for (int rr = 1; rr < 8; ++rr) {
                sum += warp_sums[rr][lane_d];
            }
            if (plane == 0) {
                dbias[d] = __float2bfloat16(sum);
            } else {
                dweight[d * 4 + (plane - 1)] = __float2bfloat16(sum);
            }
        }
    }

    void final_reduce(torch::Tensor partial, torch::Tensor dweight, torch::Tensor dbias,
                      int64_t dim64, int64_t total_chunks64) {
        const int dim = static_cast<int>(dim64);
        const int total_chunks = static_cast<int>(total_chunks64);
        dim3 grid((dim + 31) / 32, 5, 1);
        dim3 block(256, 1, 1);
        auto stream = at::cuda::getCurrentCUDAStream();
        final_reduce_kernel<<<grid, block, 0, stream>>>(
            partial.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(dweight.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(dbias.data_ptr<at::BFloat16>()),
            dim,
            total_chunks);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    """

    _MODULE = load_inline(
        name="dw_bwd_final_reduce_a0_150",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["final_reduce"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return _MODULE


def final_reduce(partial: torch.Tensor, dweight: torch.Tensor, dbias: torch.Tensor) -> None:
    module = _load_module()
    _, total_chunks, dim = partial.shape
    module.final_reduce(partial, dweight, dbias, dim, total_chunks)
