#pragma once
// Copyright (c) 2025 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cutlass/float8.h>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace dosa_score_emit {
template <uint32_t M, uint32_t N, typename Fp8>
__device__ __forceinline__ void issue_h32_mma(Fp8 *keys, Fp8 *queries,
                                              uint32_t key_row, uint32_t tmem,
                                              uint32_t completion_barrier) {
  auto instruction =
      cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                  float, M, N, cute::UMMA::Major::K,
                                  cute::UMMA::Major::K>();
  auto runtime = cute::UMMA::make_runtime_instr_desc(instruction);
#pragma unroll
  for (uint32_t k = 0; k < 128 / 32; ++k) {
    auto ad =
        deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, 128,
                                              128>(keys, key_row, k * 32);
    auto bd =
        deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, 128,
                                              128>(queries, 0, k * 32);
    deep_gemm::ptx::SM100_MMA_F8F6F4_SS::fma(ad, bd, tmem, k, runtime);
  }
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one."
               "shared::cluster.b64 [%0];" ::"r"(completion_barrier));
}
} // namespace dosa_score_emit
