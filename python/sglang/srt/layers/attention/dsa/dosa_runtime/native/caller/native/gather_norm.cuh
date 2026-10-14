#pragma once
// Included inside pair_swap_gather after the original gather/host helpers.
// Raw FP8 bytes are retained from the original vector copy. The added field
// exactly matches the original metadata integer norm and outward rounding.
__device__ __forceinline__ double maximum_with_nan(double a, double b) {
  return isnan(a) || isnan(b) ? __longlong_as_double(0x7ff8000000000000LL) : fmax(a, b);
}
__global__ void paged_gather_norm_kernel(
    const char *__restrict__ kv_cache, char *__restrict__ dst_k,
    char *__restrict__ dst_scale, const int *__restrict__ block_table,
    const int *__restrict__ permutation, int64_t token_stride, int64_t head_dim,
    int64_t block_stride, int64_t cache_block_size, int num_tokens,
    int quant_block_size, double *partial_maximum, int *partial_bad) {
  const int token = blockIdx.x * blockDim.y + threadIdx.y;
  const int x = threadIdx.x, linear = threadIdx.y * blockDim.x + x;
  const int lane = linear & 31, warp = linear >> 5;
  const bool valid = token < num_tokens;
  int source = x == 0 && valid ? permutation[token] : -1;
  source = __shfl_sync(0xffffffffu, source, 0, kGatherThreadsX);
  unsigned long long square = 0;
  uint32_t bad = 0;
  float scale = 0.0f;
  if (valid) {
    if (source < 0 || source >= num_tokens) { asm volatile("trap;"); return; }
    const int block = block_table[source / cache_block_size];
    const int64_t base = int64_t(block) * block_stride;
    const int64_t row = (source % cache_block_size) * head_dim;
    const int64_t dst = int64_t(token) * token_stride;
    const uint4 words = *reinterpret_cast<const uint4 *>(kv_cache + base + row + x * 16);
    *reinterpret_cast<uint4 *>(dst_k + dst + x * 16) = words;
    const uint32_t values[4] = {words.x, words.y, words.z, words.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t raw = (values[i] >> (j * 8)) & 127u;
        const uint32_t exponent = raw >> 3, mantissa = raw & 7;
        const uint32_t unit = exponent == 0 ? mantissa : (8u + mantissa) << (exponent - 1);
        square += static_cast<unsigned long long>(unit) * unit;
        bad |= raw == 127;
      }
    }
    if (x == 0) {
      const int64_t scale_offset = base + cache_block_size * head_dim + row * 4 / quant_block_size;
      scale = *reinterpret_cast<const float *>(kv_cache + scale_offset);
      *reinterpret_cast<float *>(dst_scale + dst * 4 / quant_block_size) = scale;
    }
  }
#pragma unroll
  for (int offset = 4; offset; offset >>= 1) {
    square += __shfl_down_sync(0xffffffffu, square, offset, 8);
    bad |= __shfl_down_sync(0xffffffffu, bad, offset, 8);
  }
  double maximum = 0.0;
  uint32_t row_bad = 0;
  if (x == 0 && valid) {
    const uint32_t magnitude = __float_as_uint(scale) & 0x7fffffffu;
    row_bad = bad | (magnitude >= 0x7f800000u) |
              (magnitude != 0 && magnitude < 0x00800000u);
    const double norm = __dsqrt_ru(static_cast<double>(square)) * 0.001953125;
    maximum = __dmul_ru(norm, fabs(static_cast<double>(scale)));
  }
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    maximum = maximum_with_nan(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
  row_bad = __reduce_or_sync(0xffffffffu, row_bad);
  __shared__ double warp_maximum[8];
  __shared__ uint32_t warp_bad[8];
  if (lane == 0) { warp_maximum[warp] = maximum; warp_bad[warp] = row_bad; }
  __syncthreads();
  if (linear == 0) {
    double m = warp_maximum[0]; uint32_t b = warp_bad[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) { m = maximum_with_nan(m, warp_maximum[i]); b |= warp_bad[i]; }
    partial_maximum[blockIdx.x] = m;
    partial_bad[blockIdx.x] = static_cast<int>(b);
  }
}

void check_norm_outputs(const torch::Tensor &dst_k, const torch::Tensor &maximum,
                        const torch::Tensor &bad) {
  const int64_t parts = (dst_k.size(0) + kGatherBlockY - 1) / kGatherBlockY;
  TORCH_CHECK(dst_k.size(1) == 128, "norm gather is original FP8 D128 only");
  TORCH_CHECK(maximum.is_cuda() && bad.is_cuda() && maximum.device() == dst_k.device() &&
      bad.device() == dst_k.device() && maximum.scalar_type() == torch::kFloat64 &&
      bad.scalar_type() == torch::kInt && maximum.is_contiguous() && bad.is_contiguous() &&
      maximum.numel() >= parts && bad.numel() >= parts, "F64 maximum/I32 bad partial buffers");
}
void launch_gather_norm(const torch::Tensor &cache, const torch::Tensor &dst_k,
    const torch::Tensor &dst_scale, const torch::Tensor &table,
    const torch::Tensor &permutation, const torch::Tensor &maximum,
    const torch::Tensor &bad, cudaStream_t stream) {
  const int n = dst_k.size(0), h = dst_k.size(1);
  const dim3 grid((n + kGatherBlockY - 1) / kGatherBlockY);
  const dim3 block(kGatherThreadsX, kGatherBlockY);
  paged_gather_norm_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const char *>(cache.data_ptr<uint8_t>()),
      reinterpret_cast<char *>(dst_k.data_ptr<uint8_t>()),
      reinterpret_cast<char *>(dst_scale.data_ptr<uint8_t>()), table.data_ptr<int>(),
      permutation.data_ptr<int>(), dst_k.stride(0), h, cache.stride(0), cache.size(1), n,
      h * 4 / dst_scale.size(1), maximum.data_ptr<double>(), bad.data_ptr<int>());
}


inline constexpr auto plan_and_permuted_paged_gather_norm_out =
    plan_and_gather<check_norm_outputs, launch_gather_norm,
                    torch::Tensor, torch::Tensor>;
