// Hardware-derived shared high family:4*(paid+2) stays within128 TMEM columns.
constexpr int kHighPaid = dosa_geometry::Heads == 64 ? 28 : 20;
constexpr int kScanHeads = kHighPaid + 2;
constexpr int kWeightPairs = kHighPaid / 2;
constexpr int kWeightWords = 4 * kWeightPairs;
constexpr int kRecordWords = kWeightWords + 24;
constexpr int kTileColumns = 4 * kScanHeads;
// Upward-BF16 revision. Include as bf16_pack_ru.cuh, in its own module
// namespace. The qualified209 FP16 source remains unchanged.
// REQUIRED fast-reader addition: force VISIBLE key keep when scale is not in
// (0, 0x1p64f]. Original RN chains/DD/guard/raw/out/visibility stay unchanged.
template<bool LiveNK=false>__global__ void prepare_kernel(
    const uint8_t* fine, const float* meta, const float* dd_scale,
    const float* raw, const float* out, const int* hint,
    const int* starts, const int* ends, const int* budget,
    const float* tau, int* error, int nq, int nk,
    uint32_t* record, int* fast_budget, int* slow_budget, int seed, const int32_t* live_nk) {
  (void)fine; // No fine scan: nonfinite raw-TC propagation is in the contract.
  const int group = blockIdx.x, t = threadIdx.x;
  const int qr = t >> 5, lane = t & 31, row = group * 4 + qr;
  if (group >= nq / 4) return;
  uint32_t* dst = record + uint64_t(group) * kRecordWords;
  const int old_budget = budget[group];
  if (t == 0) {
    dst[kWeightWords + 16] = 0;
    fast_budget[group] = 14;
    slow_budget[group] = old_budget;
  }
  if constexpr(LiveNK){
    const int logical_nk=__ldg(live_nk);
    if(logical_nk<seed||logical_nk>nk){
      if(t<4)atomicOr(error+group*4+t,1);
      return; // Initial flags/routes are already safe; no original plan read.
    }
    nk=logical_nk; // The original ends<=NK domain, not the physical capacity.
  }
  if (old_budget != kHighPaid) return; // Uniform, before any inactive plan read.
  __shared__ int good_q[4], start_q[4], end_q[4];
  __shared__ uint32_t raw_q[4], out_q[4], dd_exp[8];
  if (t < 7) dst[kWeightWords + 17 + t] = 0;
  const float* mw = meta + uint64_t(row) * 96;
  bool good = true;
  if (lane < kWeightPairs) {
    const float a0 = mw[2 + 2 * lane], a1 = mw[3 + 2 * lane];
    const uint32_t b0 = __float_as_uint(a0), b1 = __float_as_uint(a1);
    // Positive low discarded bits need one upward BF16 ULP. Truncating a
    // negative finite payload already rounds toward +infinity. Preserve +/-0.
    const uint32_t h0 = (b0 >> 16) + (((b0 & 0x80000000u) == 0u && (b0 & 65535u) != 0u) ? 1u : 0u);
    const uint32_t h1 = (b1 >> 16) + (((b1 & 0x80000000u) == 0u && (b1 & 65535u) != 0u) ? 1u : 0u);
    const float f0 = __uint_as_float(h0 << 16), f1 = __uint_as_float(h1 << 16);
    good = isfinite(a0) && isfinite(a1) && fabsf(a0) <= 65504.f &&
        fabsf(a1) <= 65504.f && isfinite(f0) && isfinite(f1) &&
        f0 >= a0 && f1 >= a1;
    dst[qr * kWeightPairs + lane] = h0 | (h1 << 16);
  }
  const bool all_good = __all_sync(0xffffffffu, good);
  if (lane == 0) {
    const float c = mw[kScanHeads], rv = raw[row], ov = out[row], tv = tau[row];
    const int sv = starts[row], ev = ends[row];
    bool valid = all_good && hint[row] == 2 && error[row] == 0 &&
        isfinite(c) && c >= 0.f && c <= 0x1p16f &&
        isfinite(rv) && rv >= 0.f && rv <= 0x1p16f &&
        isfinite(ov) && ov >= 0.f && ov <= 0x1p16f && isfinite(tv) &&
        sv >= 0 && sv <= ev && ev <= nk;
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float value = dd_scale[uint64_t(row) * kScanHeads + j];
      const uint32_t bits = __float_as_uint(value);
      const uint32_t exponent = (bits >> 23) & 255u;
      valid &= (bits & 0x807fffffu) == 0u && exponent > 0u &&
          exponent < 255u && value <= 0x1p16f;
      dd_exp[2 * qr + j] = exponent;
    }
    dst[kWeightWords + 8 + qr] = __float_as_uint(tv);
    dst[kWeightWords + 12 + qr] = __float_as_uint(c); // Exact original Cold, no added guard.
    raw_q[qr] = __float_as_uint(rv); out_q[qr] = __float_as_uint(ov);
    start_q[qr] = sv; end_q[qr] = ev; good_q[qr] = valid ? 1 : 0;
  }
  __syncthreads(); // Invalid rows participate; flags publish only after all data.
  if (t == 0) {
    int base = end_q[0];
    #pragma unroll
    for (int q = 1; q < 4; ++q) base = min(base, end_q[q]);
    bool valid = true;
    uint32_t offsets[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int64_t difference = int64_t(end_q[q]) - int64_t(base);
      valid &= good_q[q] != 0 && raw_q[q] == raw_q[0] && out_q[q] == out_q[0] &&
          start_q[q] == start_q[0] && difference >= 0 && difference <= 65535;
      offsets[q] = uint32_t(difference) & 65535u;
    }
    dst[kWeightWords + 0] = dd_exp[0] | (dd_exp[1] << 8) | (dd_exp[2] << 16) | (dd_exp[3] << 24);
    dst[kWeightWords + 1] = dd_exp[4] | (dd_exp[5] << 8) | (dd_exp[6] << 16) | (dd_exp[7] << 24);
    dst[kWeightWords + 2] = raw_q[0]; dst[kWeightWords + 3] = out_q[0];
    dst[kWeightWords + 4] = uint32_t(start_q[0]); dst[kWeightWords + 5] = uint32_t(base);
    dst[kWeightWords + 6] = offsets[0] | (offsets[1] << 16);
    dst[kWeightWords + 7] = offsets[2] | (offsets[3] << 16);
    fast_budget[group] = valid ? kHighPaid : 14;
    slow_budget[group] = valid ? 14 : old_budget;
    dst[kWeightWords + 16] = valid ? 1u : 0u;
  }
}
