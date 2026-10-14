// Shared static/device-extent implementation. Included inside the selected
// namespace policy; keep the distinct live metadata pipeline and guards.

namespace DOSA_PAGED_NAMESPACE {

// These are candidate-slot coordinates. They are deliberately distinct from
// the prefix-permuted K IDs stored in the pool and from Full's affine origin.
struct SparseWorkItem {
  int stream_row;
  int first_slot;
  int end_slot;
  uint32_t failed_queries;
};

// M1 retains the qualified Few producer schedule: 128 candidates, two raw-K
// slots and one TMEM accumulator. Both BQ1 and BQ4 use this access object.
// It contains no head-reduction, score predicate, record reservation or writer
// loop. run_score_emit owns those operations and calls these bounded hooks.
template <int BQ, bool Global20 = true>
class PagedScoreAccess {
public:
  static_assert(BQ == 1 || BQ == 4);
  static constexpr int kQueries = BQ;
  static constexpr int kMmaQueries = BQ*kHeads > 128 ? 128/kHeads : BQ;
  static constexpr int kConsumerThreads = 128;
  static constexpr int kMathWarps = 4;
  static constexpr int kTileKeys = PacketIds;
  static constexpr bool kCtaReservation = BQ == 4 && Global20;
  static constexpr bool kRawReturnedBeforeEmit = false;

  __device__ __forceinline__ PagedScoreAccess(
      Smem<BQ> &shared, const Args &args, const cute::TmaDescriptor *key_map,
      SparseWorkItem work, uint32_t tmem, uint32_t &raw_phases,
      uint32_t &mma_sequence, int thread)
      : s_(shared), a_(args), map_(key_map), work_(work), tm_(tmem),
        raw_phases_(raw_phases), mma_sequence_(mma_sequence), thread_(thread),
        packets_((work.end_slot - work.first_slot + PacketIds - 1) / PacketIds) {}

  __device__ __forceinline__ void initialize_work() {
    const int row = work_.stream_row;
    if constexpr (BQ == 4 && Global20) {
      visibility_ = group_visibility<BQ>(a_, row);
      #pragma unroll
      for (int query = 0; query < BQ; ++query)
        if (row * BQ + query < a_.qrows)
          group_tau_[query] = a_.tau[row * BQ + query];
    } else if constexpr (BQ == 1 && Global20) {
      // Same immutable-per-work tau contract as BQ4; errors remain fresh.
      if (row < a_.qrows) group_tau_[0] = a_.tau[row];
    }
    for (int i = thread_; i < BQ * kHeads * 8; i += kConsumerThreads)
      reinterpret_cast<uint4 *>(s_.q)[swizzle(i / 8, i % 8)] =
          row * BQ + i / (kHeads * 8) < a_.qrows
          ? reinterpret_cast<const uint4 *>(a_.q + uint64_t(row) * BQ * kHeads * 128)[i]
          : make_uint4(0, 0, 0, 0);
    for (int head=thread_; head<BQ*kHeads; head+=kConsumerThreads)
      s_.weights[head] = row*BQ + head/kHeads < a_.qrows
          ? a_.w[row*BQ*kHeads+head] : 0.0f;
    prepare_metadata<BQ, BQ == 4 && Global20>(
        s_, a_, row, work_.first_slot, work_.end_slot, 0, thread_, visibility_);
    if (packets_ > 1)
      prepare_metadata<BQ, BQ == 4 && Global20>(
          s_, a_, row, work_.first_slot + PacketIds, work_.end_slot,
          1, thread_, visibility_);
    if (thread_ == 0) s_.gathered[0].arrive_and_expect_tx(128 * 128);
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    gather(s_, map_, 0, thread_);
    if constexpr (BQ == 4 && Global20)
      if (packets_ > 2)
        next_ = load_future<BQ, true>(a_, row, work_.first_slot + 2 * PacketIds,
                                      work_.end_slot, thread_, visibility_);
  }

  __device__ __forceinline__ int tile_count() const { return packets_; }

  __device__ __forceinline__ void before_tile(int tile) {
    slot_ = tile & 1;
    active_keys_ = min(PacketIds, work_.end_slot - work_.first_slot - tile * PacketIds);
    if (thread_ == 0) {
      s_.gathered[slot_].wait((raw_phases_ >> slot_) & 1u);
      raw_phases_ ^= 1u << slot_;
      deep_gemm::ptx::tcgen05_after_thread_sync();
      issue_h32_mma<128, kMmaQueries * kHeads>(
          reinterpret_cast<cutlass::float_e4m3_t *>(s_.keys[slot_]),
          reinterpret_cast<cutlass::float_e4m3_t *>(s_.q), 0, tm_,
          cute::cast_smem_ptr_to_uint(&s_.done));
      if (tile + 1 < packets_)
        s_.gathered[slot_ ^ 1].arrive_and_expect_tx(128 * 128);
    }
    // All other gather issuers observe the complete alternate-slot credit.
    __syncthreads();
    if (tile + 1 < packets_) gather(s_, map_, slot_ ^ 1, thread_);
    future_ = {};
    if constexpr (BQ == 4 && Global20) {
      future_ = next_;
#if DOSA_PAGED_LIVE_EXTENT
      pending_scale_ = tile + 3 < packets_;
      if (pending_scale_)
        id_stage_ = load_future_id(
#else
      if (tile + 3 < packets_)
        next_ = load_future<BQ, true>(
#endif
            a_, work_.stream_row, work_.first_slot + (tile + 3) * PacketIds,
#if DOSA_PAGED_LIVE_EXTENT
            work_.end_slot, thread_);
#else
            work_.end_slot, thread_, visibility_);
#endif
    } else if (tile + 2 < packets_) {
      future_ = load_future<BQ>(
          a_, work_.stream_row, work_.first_slot + (tile + 2) * PacketIds,
          work_.end_slot, thread_);
    }
    s_.done.wait(mma_sequence_ & 1u);
    deep_gemm::ptx::tcgen05_after_thread_sync();
#if DOSA_PAGED_LIVE_EXTENT
  }

  __device__ __forceinline__ void finish_future_scale(int query) {
    if constexpr (BQ == 4 && Global20) {
      if(query==0&&pending_scale_)
        next_=DOSA_PAGED_NAMESPACE::finish_future_scale(a_,id_stage_,visibility_);
    }
#else
#endif
  }

  // Bound TMEM to128columns while retaining the same candidate tile and
  // exact per-head arithmetic. Drain each query half before accumulator reuse.
  __device__ __forceinline__ void advance_query_stage(int query) {
    if constexpr (kMmaQueries < BQ) {
      if (query + 1 < BQ && (query + 1) % kMmaQueries == 0) {
        deep_gemm::ptx::tcgen05_before_thread_sync();
        __syncthreads();
        ++mma_sequence_;
        if (thread_ == 0) {
          deep_gemm::ptx::tcgen05_after_thread_sync();
          issue_h32_mma<128, kMmaQueries * kHeads>(
              reinterpret_cast<cutlass::float_e4m3_t *>(s_.keys[slot_]),
              reinterpret_cast<cutlass::float_e4m3_t *>(s_.q + (query+1)*kHeads*128),
              0, tm_, cute::cast_smem_ptr_to_uint(&s_.done));
        }
        s_.done.wait(mma_sequence_ & 1u);
        deep_gemm::ptx::tcgen05_after_thread_sync();
      }
    }
  }

  __device__ __forceinline__ uint32_t tmem_base(int query_first) const {
    return tm_ + (query_first % kMmaQueries) * kHeads;
  }
  __device__ __forceinline__ void after_tmem_read(bool) const {
    // Original Few has no early accumulator credit. The common read geometry
    // still waits after both x16 reads; reuse follows after_emit's CTA fence.
  }
  __device__ __forceinline__ void after_emit(int tile) {
    if (tile + 2 < packets_)
      publish_future(s_, a_, work_.stream_row, slot_, thread_, future_);
    deep_gemm::ptx::tcgen05_before_thread_sync();
    __syncthreads();
    ++mma_sequence_;
  }

  __device__ __forceinline__ int query_index(int query) const {
    return work_.stream_row * BQ + query;
  }
  __device__ __forceinline__ int row_base() const {
    return work_.stream_row * BQ;
  }
  __device__ __forceinline__ bool query_active(int query) const {
    return query_index(query) < a_.qrows &&
        (work_.failed_queries & (1u << query)) == 0;
  }
  __device__ __forceinline__ bool lane_active() const { return thread_ < active_keys_; }
  __device__ __forceinline__ bool candidate_for_query(int) const {
    // The old common U pool is shared across the real queries. It has no
    // per-query membership mask; original H32 and cutoff decide each keep.
    return true;
  }
  __device__ __forceinline__ bool visible(int) const {
    // prepare_metadata/load_future already enforce the original all-real-query
    // predicate before any scale access, and publish the original error.
    return true;
  }
  __device__ __forceinline__ int key_index() const { return s_.ids[slot_][thread_]; }
  __device__ __forceinline__ uint32_t key_label() const {
    // The public wrapper proves origin+nk <=2^20. This is an integer K-ID
    // offset, never the Full surrogate's floating-point affine origin.
    return uint32_t(int64_t(key_index()) + a_.origin);
  }
  __device__ __forceinline__ int64_t candidate_slot(int tile) const {
    return int64_t(work_.first_slot) + tile * PacketIds + thread_;
  }
  __device__ __forceinline__ float lane_scale() const { return s_.scale[slot_][thread_]; }
  __device__ __forceinline__ float weight(int query, int head) const {
    return s_.weights[query * kHeads + head];
  }
  __device__ __forceinline__ float cutoff(int query) const {
    if constexpr (Global20) return group_tau_[query];
    return a_.tau[query_index(query)];
  }
  __device__ __forceinline__ int error(int query) const {
    return a_.error[query_index(query)];
  }
  __device__ __forceinline__ int32_t *error_ptr(int query) const {
    return a_.error + query_index(query);
  }
  __device__ __forceinline__ void flag_error(int query, int bit) const {
    atomicOr(error_ptr(query), bit);
  }
  __device__ __forceinline__ int query_start(int query) const {
    return a_.starts[query_index(query)];
  }
  __device__ __forceinline__ int query_end(int query) const {
    return a_.ends[query_index(query)];
  }
  __device__ __forceinline__ dosa_score_emit::RecordOutput output_view() const {
    dosa_score_emit::RecordOutput output{};
    output.low = static_cast<uint8_t *>(a_.scores);
    output.words = a_.outids;
    output.counts = a_.outcounts;
    output.error = a_.error;
    output.rows = a_.qrows;
    output.capacity = a_.output_cap;
    output.low_stride = a_.score_stride;
    output.word_stride = a_.output_id_stride;
    return output;
  }
  __device__ __forceinline__ dosa_score_emit::EmitStorage emit_storage() {
    dosa_score_emit::EmitStorage storage{};
    storage.cta_counts = &s_.emit_counts[0][0];
    storage.cta_starts = &s_.emit_starts[0][0];
    return storage;
  }
  __device__ __forceinline__ uint32_t *cta_counts(int query) { return s_.emit_counts[query]; }
  __device__ __forceinline__ uint64_t *cta_starts(int query) { return s_.emit_starts[query]; }
  __device__ __forceinline__ int thread() const { return thread_; }
  __device__ __forceinline__ int warp() const { return thread_ / 32; }
  __device__ __forceinline__ int lane() const { return thread_ % 32; }

private:
  Smem<BQ> &s_;
  const Args &a_;
  const cute::TmaDescriptor *map_;
  SparseWorkItem work_;
  uint32_t tm_;
  uint32_t &raw_phases_;
  uint32_t &mma_sequence_;
  int thread_, packets_;
  int slot_ = 0, active_keys_ = 0;
  Visibility visibility_{};
  float group_tau_[BQ]{};
  FutureMetadata next_{}, future_{};
#if DOSA_PAGED_LIVE_EXTENT
  FutureId id_stage_{};
  bool pending_scale_=false;
#else
#endif
};

} // namespace DOSA_PAGED_NAMESPACE
