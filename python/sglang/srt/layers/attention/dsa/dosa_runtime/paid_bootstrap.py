"""Paid current-query hints for a cold, fixed-geometry execution slot.

This optional initializer changes only hints. Every query still computes its
original H32 seed scores and safe threshold. The caller chooses which query
sizes use it; installing it is not a default policy for small queries.
"""

import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["NK", "NPAGES"])
def _identity_gather(CACHE, SCALES, TABLE, K, S, NK, NPAGES=0):
    page = tl.program_id(0)
    i = tl.arange(0, 64)
    d = tl.arange(0, 128)
    physical = tl.load(TABLE + page).to(tl.int64)
    valid = page * 64 + i < NK
    if NPAGES > 0:
        # This additional cold-hint read must not go outside the physical pool.
        # Masked pages overwrite the scratch with zeros, never stale contents.
        valid = valid & (physical >= 0) & (physical < NPAGES)
    raw = tl.load(
        CACHE + physical * 8448 + i[:, None] * 128 + d[None, :],
        valid[:, None],
        0,
    )
    scale = tl.load(SCALES + physical * 2112 + 2048 + i, valid, 0.0)
    output_valid = page * 64 + i < NK
    tl.store(K + (page * 64 + i[:, None]) * 128 + d[None, :], raw, output_valid[:, None])
    tl.store(S + page * 64 + i, scale, output_valid)


@triton.jit(do_not_specialize=["NV"])
def _add_votes(IDS, VOTES, N: tl.constexpr, NV, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    key = tl.load(IDS + i, i < N, -1)
    tl.atomic_add(VOTES + key, 1, (i < N) & (key >= 0) & (key < NV), sem="relaxed")


@triton.jit
def _spread_queries(Q, W, OQ, OW, NQ: tl.constexpr, HEADS: tl.constexpr):
    anchor = tl.program_id(0)
    row = anchor * (NQ - 1) // 3
    i = tl.arange(0, HEADS * 128)
    tl.store(OQ + anchor * HEADS * 128 + i, tl.load(Q + row * HEADS * 128 + i))
    weight = tl.load(W + row * HEADS + i, i < HEADS, 0.0)
    tl.store(OW + anchor * HEADS + i, weight, i < HEADS)


class PaidBootstrap:
    """One reserved owner for the shared four-anchor original-H32 cold hints.

    Full and Few use the current query spread, sample scorer and unordered FI.
    Every call still computes the original H32 seed scores and safe threshold.
    """

    B = 4

    def __init__(self, backend, sample_module):
        if backend.lt.TIERED_SEED_12K or backend.lt.HOT_PREFIX != 12288:
            raise ValueError("bootstrap requires the ordinary 12K hint contract")
        if not 12288 <= backend.max_k <= 1048576:
            raise ValueError("bootstrap requires a legal reserved key capacity")
        if sample_module is None or not callable(getattr(sample_module, "seed_score", None)):
            raise ValueError("provide the shared sample.seed_score module")
        self.backend, self.sample_module = backend, sample_module
        self.score_heads = backend.score_heads
        heads = self.score_heads
        self.cache = self.table = self.current_q = self.current_w = None
        self.nk = self.common_end = self._capacity_nk = backend.max_k
        self.current_queries = 0
        self.carry = backend.modules["caller"]
        device, nk = backend.device, backend.max_k
        self.k = torch.empty((nk, 128), device=device, dtype=torch.uint8)
        self.scale = torch.empty(nk, device=device, dtype=torch.float32)
        self.votes = torch.zeros(nk, device=device, dtype=torch.int32)
        self.partial = torch.empty(
            (backend.lt._CARRY_TOPK_MAX_BLOCKS, 5), device=device, dtype=torch.int16
        )
        self.state = torch.zeros(
            backend.lt._CARRY_TOPK_STATE_INTS, device=device, dtype=torch.int32
        )
        self.starts = torch.zeros(4, device=device, dtype=torch.int32)
        self.ends = torch.full((4,), nk, device=device, dtype=torch.int32)
        self.query_pack = torch.empty((4, heads, 128), device=device, dtype=torch.uint8)
        self.weight_pack = torch.empty((4, heads), device=device, dtype=torch.float32)
        self.scores = torch.empty((4, nk), device=device, dtype=torch.float32)
        self.masks = torch.empty((4, nk), device=device,
                                 dtype=torch.int32 if heads == 32 else torch.int64)
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.events = self.query_events = self.last_ids = None
        self.calls = 0
        self._k_storage, self._scale_storage = self.k, self.scale
        self._votes_storage = self.votes
        self._score_storage, self._mask_storage = self.scores, self.masks
        self._page_bound = 0
        self.ready = torch.cuda.Event()
        self.ready.record(torch.cuda.current_stream(device))

    def cold_call(self, cache, table, nk, common_end, queries, hot, strong, start,
                  *, source_q=None, source_w=None):
        """Borrow a request's operands only while its backend slot is held."""
        with torch.cuda.device(self.backend.device):
            torch.cuda.current_stream(self.backend.device).wait_event(self.ready)
            try:
                self.bind_current(cache, table, nk=nk, common_end=common_end,
                                  queries=queries, source_q=source_q, source_w=source_w)
                self.record_current_stream()
                return self(hot, strong, start)
            finally:
                # All borrowed inputs were recorded on the consuming stream.
                self.cache = self.table = self.current_q = self.current_w = None

    def bind_current(self, cache, table64, *, nk, common_end, queries,
                     source_q=None, source_w=None):
        """Bind a real cold call while its execution slot lock is held."""
        if not 12288 <= common_end <= nk <= self._capacity_nk:
            raise ValueError("cold bootstrap exceeds its reserved key capacity")
        query_capacity = self.backend.max_q
        if source_q is not None or source_w is not None:
            if (source_q is None or source_w is None
                    or source_q.device != self.backend.device or source_w.device != self.backend.device
                    or source_q.dtype != torch.float8_e4m3fn or source_w.dtype != torch.float32
                    or tuple(source_q.shape) != (queries, self.score_heads, 128)
                    or tuple(source_w.shape) != (queries, self.score_heads)
                    or not source_q.is_contiguous() or not source_w.is_contiguous()):
                raise ValueError("cold bootstrap requires complete current Q/W")
            query_capacity = queries
        # The original interpolation repeats real rows for fewer than four queries.
        if not 1 <= queries <= query_capacity:
            raise ValueError("cold bootstrap query/common extent exceeds capacity")
        self.current_q, self.current_w = source_q, source_w
        if (cache.dtype != torch.uint8 or table64.dtype != torch.int32
                or cache.device != self.backend.device
                or table64.device != self.backend.device
                or not cache.is_contiguous() or not table64.is_contiguous()
                or table64.numel() < triton.cdiv(nk, 64)):
            raise ValueError("cold bootstrap needs the actual contiguous cache/table")
        if cache.shape[0] == 0 or not ((cache.ndim == 2 and cache.shape[1] == 8448)
                or (cache.ndim == 3 and tuple(cache.shape[1:]) == (64, 132))):
            raise ValueError("cold bootstrap requires original64x132 physical pages")
        if table64.ndim != 2 or table64.shape[0] != 1:
            raise ValueError("cold bootstrap requires one actual page table")
        self.cache, self.table = cache, table64
        self.k = self._k_storage[:nk]
        self.scale = self._scale_storage[:nk]
        self.votes = self._votes_storage[:common_end]
        # The sampler needs dense BxNK views, not row-strided capacity slices.
        self.scores = self._score_storage.view(-1)[:self.B * nk].view(self.B, nk)
        self.masks = self._mask_storage.view(-1)[:self.B * nk].view(self.B, nk)
        if common_end != self.common_end:
            self.ends.fill_(common_end)
        self.nk, self.common_end, self.current_queries = nk, common_end, queries
        self._page_bound = cache.shape[0]

    def record_current_stream(self):
        stream = torch.cuda.current_stream(self.backend.device)
        for value in (self._k_storage, self._scale_storage, self._votes_storage,
                      self._score_storage, self._mask_storage, self.partial,
                      self.state, self.starts, self.ends, self.query_pack,
                      self.weight_pack, self.cache, self.table, self.current_q, self.current_w):
            if value is not None:
                value.record_stream(stream)

    def __call__(self, hot, strong=None, start=0):
        import flashinfer

        if self.cache is None or self.table is None:
            raise RuntimeError("bind a real cache/table before invoking bootstrap")
        if strong is not None or start != 0 or hot.numel() != 12288:
            raise ValueError("bootstrap requires ordinary visible 12K hints")
        if self.current_queries < 1:
            raise ValueError("spread bootstrap needs the current query count")
        self.query_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(2)
        ]
        self.query_events[0].record()
        _spread_queries[(4,)](
            (self.backend.q_input if self.current_q is None else self.current_q).view(torch.uint8),
            self.backend.w_input if self.current_w is None else self.current_w,
            self.query_pack,
            self.weight_pack,
            self.current_queries,
            self.score_heads,
            num_warps=4,
        )
        self.query_events[1].record()
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        self.events[0].record()
        _identity_gather[(triton.cdiv(self.nk, 64),)](
            self.cache,
            self.cache.view(torch.float32),
            self.table,
            self.k,
            self.scale,
            self.nk,
            NPAGES=self._page_bound,
            num_warps=4,
        )
        self.events[1].record()
        query = self.query_pack.view(torch.float8_e4m3fn)
        weights = self.weight_pack
        # Retain and pay for the original sample emitter's sign-mask stores.
        self.sample_module.seed_score(
            query,
            self.k.view(torch.float8_e4m3fn),
            self.scale,
            weights,
            self.scores,
            self.masks,
            self.sms,
        )
        scores = self.scores
        self.events[2].record()
        ids = flashinfer.top_k_ragged_transform(
            scores,
            self.starts,
            self.ends,
            2048,
            deterministic=False,
            dsa_graph_safe=True,
        )
        self.events[3].record()
        self.votes.zero_()
        _add_votes[(triton.cdiv(self.B * 2048, 256),)](
            ids, self.votes, self.B * 2048, self.common_end, 256
        )
        self.carry.carry_votes_topk_reset_(
            self.votes, hot, self.partial, self.state, 12288, self.B, start
        )
        self.events[4].record()
        self.last_ids = ids
        self.calls += 1

    def invoice(self):
        """Read completed event spans; the caller must have synchronized."""
        if self.events is None:
            raise RuntimeError("no bootstrap invocation has completed")
        phases = ("identity_gather_ms", "H32_ms", "FI_ms", "votes_select_reset_ms")
        result = {
            name: self.events[i].elapsed_time(self.events[i + 1])
            for i, name in enumerate(phases)
        }
        if self.query_events is not None:
            result["query_pack_ms"] = self.query_events[0].elapsed_time(
                self.query_events[1]
            )
        return result
