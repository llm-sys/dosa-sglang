template <bool ResetVotes = true>
__global__ void carry_votes_emit_warp_kernel(
    int32_t *__restrict__ votes, int count, int min_index, int max_vote,
    int64_t *__restrict__ out_idx, const int32_t *__restrict__ state) {
  constexpr int kSteps = kCarryTileItems / kCarryThreads;
  constexpr int kWarpItems = 32 * kSteps;
  constexpr unsigned kFullMask = 0xffffffffu;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int block = blockIdx.x;
  const int begin = block * kCarryTileItems + warp * kWarpItems;
  const int threshold = state[kCarryThreshold];
  const int tie_block = state[kCarryTieBlock];
  const int tie_take = state[kCarryTieTake];
  const int output_base = state[kCarryBlockOffsets + block];
  const unsigned lane_bit = 1u << lane;
  const unsigned lane_mask = lane_bit - 1u;

  __shared__ unsigned s_greater[kCarryWarps][kSteps];
  __shared__ unsigned s_equal[kCarryWarps][kSteps];
  __shared__ int s_greater_count[kCarryWarps];
  __shared__ int s_equal_count[kCarryWarps];
  __shared__ int s_equal_quota[kCarryWarps];
  __shared__ int s_output_base[kCarryWarps];

  int greater_count = 0;
  int equal_count = 0;
#pragma unroll 1
  for (int step = 0; step < kSteps; ++step) {
    const int index = begin + step * 32 + lane;
    const bool valid = index < count;
    const int raw = valid ? votes[index] : 0;
    const int value = raw < 0 ? 0 : (raw > max_vote ? max_vote : raw);
    if constexpr (ResetVotes) {
      if (valid) votes[index] = 0;
    }
    const bool eligible = valid && index >= min_index;
    const unsigned greater = __ballot_sync(kFullMask, eligible && value > threshold);
    const unsigned equal = __ballot_sync(kFullMask, eligible && value == threshold);
    greater_count += __popc(greater);
    equal_count += __popc(equal);
    if (lane == 0) {
      s_greater[warp][step] = greater;
      s_equal[warp][step] = equal;
    }
  }
  if (lane == 0) {
    s_greater_count[warp] = greater_count;
    s_equal_count[warp] = equal_count;
  }
  __syncthreads();

  if (tid == 0) {
    int equals_before = 0;
    int selected_before = 0;
#pragma unroll
    for (int source_warp = 0; source_warp < kCarryWarps; ++source_warp) {
      const int equal = s_equal_count[source_warp];
      int quota = 0;
      if (block < tie_block) {
        quota = equal;
      } else if (block == tie_block) {
        const int remaining = tie_take - equals_before;
        quota = remaining <= 0 ? 0 : (remaining < equal ? remaining : equal);
      }
      s_equal_quota[source_warp] = quota;
      s_output_base[source_warp] = selected_before;
      equals_before += equal;
      selected_before += s_greater_count[source_warp] + quota;
    }
  }
  __syncthreads();

  const int quota = s_equal_quota[warp];
  const int warp_output_base = output_base + s_output_base[warp];
  int equals_seen = 0;
  int selected_seen = 0;
#pragma unroll 1
  for (int step = 0; step < kSteps; ++step) {
    const unsigned greater = s_greater[warp][step];
    const unsigned equal = s_equal[warp][step];
    const int equal_rank = equals_seen + __popc(equal & lane_mask);
    const bool take_equal = (equal & lane_bit) != 0 && equal_rank < quota;
    const bool selected = (greater & lane_bit) != 0 || take_equal;
    const unsigned mask = __ballot_sync(kFullMask, selected);
    if (selected) {
      const int rank = selected_seen + __popc(mask & lane_mask);
      out_idx[warp_output_base + rank] =
          static_cast<int64_t>(begin + step * 32 + lane);
    }
    equals_seen += __popc(equal);
    selected_seen += __popc(mask);
  }
}
