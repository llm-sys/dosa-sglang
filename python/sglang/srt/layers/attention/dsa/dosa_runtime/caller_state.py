"""Private single-request FP8 caller state shared by Full and Few.

Extracted from the qualified caller. Each instance owns its queues, carry
histograms, side stream and grow-only geometry. No environment-controlled JIT.
"""

import torch


class CallerState:
    HOT_PREFIX = 12288
    NB = 256
    HEADROOM = 0.0
    _PAGED_CANDIDATE_INLINE = 8192
    _PAGED_CANDIDATE_PAGE = 4096
    _PAGED_POOL_PAGES_PER_ROW = 12
    PRODUCTION_MIN_S = 196608
    PRODUCTION_MAX_S = 1048576
    CARRY_RECENT_ROWS = 1536
    OVF_WATERMARK = 65536
    _CARRY_TOPK_MAX_BLOCKS = 128
    _CARRY_TOPK_STATE_INTS = 136
    TIERED_SEED_12K = False
    ENABLED = True
    PAGED_CANDIDATES = True
    CANDIDATE_STORAGE_DTYPE = torch.uint8

    def __init__(self, native, *, gather=None):
        self.native = native
        self.gather = native if gather is None else gather
        self._AUX_CACHE = {}
        self._CARRY_TOPK_WORKSPACE = {}
        self._CARRY_VOTE_BUFS = {}
        self._HOT_CARRY = {}
        self._STOCK_PENDING = {}
        self._HOT_STREAM = {}
        self._PAGED_CAND_BUFS = {}
        self._PAIR_PLAN_BUFS = {}
        self._PREP_BUFS = {}
        self._request_slots = {}
        self._reserved_carry = {}

    def _ks0_keh(self, Q, head, dev):
        """Cached zero-starts and sample-end tensors: torch.zeros/torch.full are a
        kernel launch each and measurably cost ~0.1-0.2ms/chunk on the hot path."""
        key = (str(dev), head)
        entry = self._AUX_CACHE.get(key)
        if entry is None or entry[0].shape[0] < Q:
            qmax = max(Q, 1024)
            entry = (
                torch.zeros(qmax, dtype=torch.int32, device=dev),
                torch.full((qmax,), head, dtype=torch.int32, device=dev),
            )
            self._AUX_CACHE[key] = entry
        return (entry[0][:Q], entry[1][:Q])

    def _stream_id(self, dev):
        return (
            int(torch.cuda.current_stream(dev).cuda_stream)
            if getattr(dev, "type", None) == "cuda"
            else 0
        )

    def _pair_plan_bufs(self, sequence_length, dev):
        """Persistent pair-swap planner workspace with geometric growth."""
        key = (str(dev), self._stream_id(dev))
        state = self._PAIR_PLAN_BUFS.get(key)
        if (state is None or state["cap"] < sequence_length
                or state["epoch"] + 1 >= 1 << 30):
            cap = max(16384, 1 << (sequence_length - 1).bit_length())
            state = {
                "cap": cap,
                "epoch": 0,
                "hot_epoch": torch.zeros(cap, dtype=torch.int32, device=dev),
                "permutation": torch.arange(cap, dtype=torch.int32, device=dev),
                "swap_a": torch.empty(self.HOT_PREFIX, dtype=torch.int32, device=dev),
                "swap_b": torch.empty(self.HOT_PREFIX, dtype=torch.int32, device=dev),
                "counts": torch.zeros(4, dtype=torch.int32, device=dev),
            }
            self._PAIR_PLAN_BUFS[key] = state
        state["epoch"] += 1
        return state, state["epoch"]

    def _paged_cand_bufs(self, Q, logical_cap, dev):
        """Acquire one reusable inline tier and shared overflow-page arena.

        The shared pool has no per-row cap: all rows contend for one arena. The
        page table uses flat maximum-width backing storage so its active view stays
        contiguous. Stale payload pages are unreachable and need no clearing.
        """
        Q = int(Q)
        logical_cap = int(logical_cap)
        if (
            Q <= 0
            or not self._PAGED_CANDIDATE_INLINE < logical_cap <= self.PRODUCTION_MAX_S
        ):
            raise ValueError(
                f"invalid paged candidate shape Q={Q}, logical_cap={logical_cap}"
            )
        table_pages = (
            logical_cap - self._PAGED_CANDIDATE_INLINE + self._PAGED_CANDIDATE_PAGE - 1
        ) // self._PAGED_CANDIDATE_PAGE
        pool_pages = self._PAGED_POOL_PAGES_PER_ROW * Q
        key = (str(dev), self.CANDIDATE_STORAGE_DTYPE)
        b = self._PAGED_CAND_BUFS.get(key)
        if b is None or b["q"] < Q:
            self._PAGED_CAND_BUFS[key] = None
            del b
            qm = max(int(Q), 4096)
            max_table_pages = (
                self.PRODUCTION_MAX_S
                - self._PAGED_CANDIDATE_INLINE
                + self._PAGED_CANDIDATE_PAGE
                - 1
            ) // self._PAGED_CANDIDATE_PAGE
            b = {
                "q": qm,
                "iv": torch.empty(
                    qm,
                    self._PAGED_CANDIDATE_INLINE,
                    dtype=self.CANDIDATE_STORAGE_DTYPE,
                    device=dev,
                ),
                "ii": torch.empty(
                    qm, self._PAGED_CANDIDATE_INLINE, dtype=torch.int32, device=dev
                ),
                "ov": torch.empty(
                    self._PAGED_POOL_PAGES_PER_ROW * qm,
                    self._PAGED_CANDIDATE_PAGE,
                    dtype=self.CANDIDATE_STORAGE_DTYPE,
                    device=dev,
                ),
                "oi": torch.empty(
                    self._PAGED_POOL_PAGES_PER_ROW * qm,
                    self._PAGED_CANDIDATE_PAGE,
                    dtype=torch.int32,
                    device=dev,
                ),
                "pt": torch.empty(qm * max_table_pages, dtype=torch.int32, device=dev),
                "head": torch.empty(1, dtype=torch.int32, device=dev),
                "status": torch.empty(1, dtype=torch.int32, device=dev),
            }
            self._PAGED_CAND_BUFS[key] = b
        table_words = Q * table_pages
        return {
            "inline_val": b["iv"][:Q],
            "inline_idx": b["ii"][:Q],
            "overflow_val": b["ov"][:pool_pages],
            "overflow_idx": b["oi"][:pool_pages],
            "page_table": b["pt"][:table_words].view(Q, table_pages),
            "pool_head": b["head"],
            "pool_status": b["status"],
        }

    def _carry_vote_hist(self, nv, dev, hot_key, waited_event=None):
        """Acquire a per-layer histogram for selector-fused carry votes."""
        key = (str(dev), hot_key)
        entry = self._CARRY_VOTE_BUFS.get(key)
        if entry is not None and entry["free_event"] is not None:
            free_event = entry["free_event"]
            if free_event is not waited_event:
                torch.cuda.current_stream(dev).wait_event(free_event)
        if entry is None or entry["buf"].numel() < nv:
            cap = max(1024, 1 << (nv - 1).bit_length())
            hot = None if entry is None else entry.get("hot")
            ready_event = None if entry is None else entry.get("ready_event")
            if ready_event is None:
                ready_event = torch.cuda.Event()
            entry = {
                "buf": torch.zeros(cap, dtype=torch.int32, device=dev),
                "free_event": None,
                "ready_event": ready_event,
                "hot": (
                    hot
                    if hot is not None
                    else torch.empty(self.HOT_PREFIX, dtype=torch.int64, device=dev)
                ),
                "needs_reset": False,
                "dirty_extent": 0,
            }
            self._CARRY_VOTE_BUFS[key] = entry
        votes = entry["buf"][:nv]
        if entry.get("needs_reset", False):
            reset_extent = max(nv, entry.get("dirty_extent", 0))
            entry["buf"][:reset_extent].zero_()
            entry["needs_reset"] = False
            entry["dirty_extent"] = 0
        entry["needs_reset"] = True
        entry["dirty_extent"] = max(entry.get("dirty_extent", 0), nv)
        return votes

    def _carry_topk_workspace(self, max_vote, dev):
        """Caller-owned workspace serialized by the one side stream per device."""
        key = str(dev)
        bins = max_vote + 1
        entry = self._CARRY_TOPK_WORKSPACE.get(key)
        if entry is None or entry["partial"].shape[1] < bins:
            entry = {
                "partial": torch.empty(
                    (self._CARRY_TOPK_MAX_BLOCKS, bins), dtype=torch.int16, device=dev
                ),
                "state": torch.zeros(
                    self._CARRY_TOPK_STATE_INTS, dtype=torch.int32, device=dev
                ),
            }
            self._CARRY_TOPK_WORKSPACE[key] = entry
        return entry

    def _publish_carry(self, hot_key, votes, nv, min_index, max_vote):
        """Publish the voted HOT set on the per-device side stream."""
        if nv - min_index < self.HOT_PREFIX:
            return
        dev = votes.device
        key = (str(dev), hot_key)
        entry = self._CARRY_VOTE_BUFS[key]
        side = self._HOT_STREAM.get(str(dev))
        if side is None:
            side = torch.cuda.Stream(device=dev)
            self._HOT_STREAM[str(dev)] = side
        carry_ext = self.native
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            votes.record_stream(side)
            hot_n = self.HOT_PREFIX
            use_custom = (
                carry_ext is not None
                and hasattr(carry_ext, "carry_votes_topk_reset_")
                and (nv <= 1048576)
                and (0 < max_vote <= 8192)
            )
            if use_custom:
                hot = entry["hot"][:hot_n]
                workspace = self._carry_topk_workspace(max_vote, dev)
                carry_ext.carry_votes_topk_reset_(
                    votes,
                    hot,
                    workspace["partial"],
                    workspace["state"],
                    hot_n,
                    max_vote,
                    min_index,
                )
                entry["needs_reset"] = False
                entry["dirty_extent"] = 0
            else:
                if min_index > 0:
                    votes[:min_index].fill_(torch.iinfo(torch.int32).min)
                hot = votes.topk(hot_n).indices
            ready = entry["ready_event"]
            ready.record(side)
        entry["free_event"] = ready
        self._HOT_CARRY[key] = (hot, nv, ready, min_index)

    def _prep_bufs(self, Q, nb, dev):
        """Caller-owned seed_prep outputs. Reusing these kills the 0.1-0.5GB
        per-call alloc churn that forced the CUDA allocator into pathological
        behavior at small Q (256K: 4.5ms without an event-paced probe)."""
        key = (str(dev), nb)
        b = self._PREP_BUFS.get(key)
        if b is None or b["q"] < Q:
            qm = max(Q, 1024)
            b = {
                "q": qm,
                "o": torch.empty(qm, device=dev, dtype=torch.float32),
                "inv": torch.empty(qm, device=dev, dtype=torch.float32),
                "th": torch.empty(qm, dtype=torch.int32, device=dev),
                "bc": torch.empty(qm, nb, dtype=torch.int32, device=dev),
                "cc": torch.empty(qm, dtype=torch.int32, device=dev),
                "status": torch.empty(qm, dtype=torch.int32, device=dev),
            }
            self._PREP_BUFS[key] = b
        return b

    def prepare_permuted_gather(
        self,
        kv_cache,
        dst_k,
        dst_scale,
        block_table,
        *,
        sequence_length,
        query_length,
        num_reqs,
        common_end,
        window_start,
        hot_key,
    ):
        """Pair-swap HOT12288 into the paged-gather prefix on the main stream."""
        try:
            S = int(sequence_length)
            Q = int(query_length)
            ks = int(window_start)
            common_end = int(common_end)
            if (
                not self.ENABLED
                or int(num_reqs) != 1
                or (
                    not self.PRODUCTION_MIN_S
                    <= S
                    <= self.PRODUCTION_MAX_S
                )
                or Q <= 0
                or (hot_key is None)
                or (ks < 0)
                or (ks % 4 != 0)
                or (ks + self.HOT_PREFIX > common_end)
                or (common_end > S)
                or (dst_scale.shape != (S, 4))
                or (dst_scale.dtype != torch.uint8)
                or (
                    (tuple(dst_k.shape), dst_k.dtype)
                    not in (((S, 128), torch.float8_e4m3fn), ((S, 64), torch.uint8))
                )
                or (dst_k.device.type != "cuda")
                or (torch.cuda.get_device_capability(dst_k.device)[0] != 10)
            ):
                return None
            carry = self._HOT_CARRY.get((str(dst_k.device), hot_key))
            expected_carry_len = 4
            carry_valid = not (
                carry is None
                or len(carry) != expected_carry_len
                or carry[1] > common_end
                or (carry[3] < ks)
                or (carry[0].dim() != 1)
                or (carry[0].numel() < self.HOT_PREFIX)
            )
            if not carry_valid:
                return None
            ext = self.gather
            required_planner_ops = ("plan_and_permuted_paged_gather_out",)
            if ext is None or not all(
                (hasattr(ext, name) for name in required_planner_ops)
            ):
                return None
            hot = carry[0][: self.HOT_PREFIX]
            carry_event = carry[2]
            stream = torch.cuda.current_stream(dst_k.device)
            if carry_event is not None:
                carry_event.wait(stream)
            hot.record_stream(stream)
            state, epoch = self._pair_plan_bufs(S, dst_k.device)
            permutation = state["permutation"][:S]
            ext.plan_and_permuted_paged_gather_out(
                hot,
                state["hot_epoch"][:S],
                permutation,
                state["swap_a"],
                state["swap_b"],
                state["counts"],
                ks,
                common_end,
                epoch,
                kv_cache,
                dst_k.view(torch.uint8),
                dst_scale,
                block_table,
            )
            return {
                "permutation": permutation,
                "carry_event": carry_event,
                "sequence_length": S,
                "query_length": Q,
                "window_start": ks,
                "common_end": common_end,
                "qualification_sequence_length": S,
                "planner_state": state,
            }
        except Exception as e:
            print(f"[litetopk] exact-once permuted gather declined: {e}", flush=True)
            return None

    def reserve_layer(self, device, layer_id, max_keys):
        """Allocate one layer's vote/HOT storage without publishing a prior."""
        slot = (str(device), layer_id)
        if slot in self._reserved_carry or slot in self._request_slots:
            return
        key = (layer_id, "__dosa_workspace_only__")
        self._carry_vote_hist(max_keys, device, key)
        entry = self._CARRY_VOTE_BUFS.pop((str(device), key))
        entry["needs_reset"] = False
        entry["dirty_extent"] = 0
        self._reserved_carry[slot] = entry
        self._carry_topk_workspace(self.CARRY_RECENT_ROWS, device)

    def begin_request(self, device, hot_key):
        if not isinstance(hot_key, tuple) or len(hot_key) != 2:
            return
        layer, request = hot_key
        slot = (str(device), layer)
        current = (str(device), hot_key)
        previous = self._request_slots.get(slot)
        if previous is not None and previous != current:
            self._HOT_CARRY.pop(previous, None)
            self._STOCK_PENDING.pop(previous, None)
            entry = self._CARRY_VOTE_BUFS.pop(previous, None)
            if entry is not None:
                self._reserved_carry[slot] = entry
        if current not in self._CARRY_VOTE_BUFS and slot in self._reserved_carry:
            self._CARRY_VOTE_BUFS[current] = self._reserved_carry.pop(slot)
        self._request_slots[slot] = current
