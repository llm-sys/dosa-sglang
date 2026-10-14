"""Global completed-H32 cells in the shared byte-plus-int record layout."""

import torch


class CompletedRecords:
    """Own H32 records and their output pipeline; call inputs stay explicit."""

    def __init__(self, runtime, module, rows, device, low_codes=None):
        self.runtime = runtime
        self.module = module
        self.direct_seed = None
        self.low_codes = low_codes if low_codes is not None else torch.empty(
            (rows, runtime.capacity), device=device, dtype=torch.uint8)

    def prepare_h32(self):
        """Validate the paid seed and bind exact readers to their score storage."""
        r = self.runtime
        low = self.low_codes[:r.rows]
        if r.live_nk is not None:
            if self.direct_seed is None:
                raise RuntimeError("Device-live-NK requires the current direct record5 seed path")
            r.live_exact.validate_seed5_live_nk(
                r.ids, r.counts, r.seed_tau, r.safe_tau, r.starts, r.ends,
                r.error, r.nk, r.live_nk, r.seed)
            readers = (r.live_exact.exact_paged_solo_live_nk,
                       r.live_exact.exact_paged_common_live_nk)
            return readers, low, (r.live_nk, r.seed)
        if self.direct_seed is None:
            self.module.pack_seed(
                r.values, r.ids, r.counts, r.safe_tau, r.starts, r.ends, r.error, low, r.nk
            )
        else:
            self.direct_seed.validate_seed5(
                r.ids, r.counts, r.seed_tau, r.safe_tau,
                r.starts, r.ends, r.error, r.nk,
            )
        return (self.module.exact_paged_solo, self.module.exact_paged_common), low, ()

    def bind_output(self, rows, device, selector, post):
        self.selector = selector
        self.post = post

        def integers(*shape):
            return torch.empty(shape, device=device, dtype=torch.int32)

        self.epsilon = torch.zeros(rows, device=device, dtype=torch.float32)
        self.status = integers(rows)
        self.diag = integers(rows, 5)
        self.boundary_ids = integers(rows, 4096)
        self.boundary_counts = integers(rows)
        self.temporary = torch.empty(
            (rows, 8192), device=device, dtype=torch.float32
        )
        self.temporary_ids = integers(rows, 8192)
        self.temporary_counts = integers(rows)
        self.done = integers(rows)
        self.fi_lengths = integers(rows)
        self.tau = torch.empty(rows, device=device, dtype=torch.float32)

    def select_and_publish(self, runtime, q, k, scale):
        r = runtime
        n = r.rows
        self.selector.select_dense5(
            self.low_codes[:n],
            r.ids, r.counts, r.starts, r.ends, self.epsilon[:n], r.pool_status,
            r.error, r.ordered_output, self.status[:n], self.diag[:n],
            self.boundary_ids[:n], self.boundary_counts[:n], self.temporary[:n],
            self.temporary_ids[:n], self.temporary_counts[:n], self.done[:n], r.nk,
        )
        self.post.complete_h32_fused_large(
            q, k, scale, r.w, self.boundary_ids[:n], self.boundary_counts[:n],
            self.diag[:n], r.starts, r.ends, r.error, r.ordered_output,
            self.temporary[:n], self.temporary_ids[:n], self.temporary_counts[:n],
            self.fi_lengths[:n], self.tau[:n],
        )
        r.output.index_copy_(0, r.order, r.ordered_output)


def install_record5(backend, exact, selector, post, *, seed=None):
    """Install before a fresh backend's first call or private graph capture.

    The original U/guard/seed/H32/leaf remain unchanged. Only completed H32
    scores are mapped to global20bit order cells. Full's original5B surrogate
    has its own mathematical domain; it is not silently reinterpreted here.
    """
    if (
        backend.hot_graph is not None
        or backend.last_done is not None
        or getattr(backend, "startup_ready", None) is not None
    ):
        raise RuntimeError("Install record5 before calls/capture")
    if backend.max_k > (1 << 20):
        raise ValueError("Record5 requires the20bit K-ID extent")
    if (
        getattr(exact, "RECORD_BYTES", None) != 5
        or getattr(exact, "SCORE_BITS", None) != 20
        or not hasattr(selector, "select_dense5")
    ):
        raise ValueError("Record5 requires matching global20bit native modules")
    if seed is not None and not all(hasattr(seed, name) for name in (
        "prepare5", "finish5", "validate_seed5"
    )):
        raise ValueError("Direct record5 seed requires matching native entry points")
    r = backend.rt
    low = None
    if seed is None:
        prefix = torch.empty(
            (backend.max_q, 2048), device=backend.device, dtype=torch.float32
        )
        backend.backing["values"] = prefix
        r.values = prefix if backend.shape is None else prefix[:r.rows]
    elif (r.values.dtype == torch.uint8
          and tuple(r.values.shape) == (backend.max_q, r.capacity)):
        low = r.values
    else:
        # Legacy explicit factories may already own a dense F32 table. Release
        # that table before allocating its replacement; direct live factories
        # allocate the typed low-byte storage from construction instead.
        backend.backing["values"] = None
        r.values = None
    owner = CompletedRecords(r, exact, backend.max_q, backend.device, low_codes=low)
    if seed is not None:
        # No selected F32 table remains. The normal raw DG seed logits and
        # original exact seed tau still feed the unchanged metadata producer.
        backend.backing["values"] = owner.low_codes
        r.values = owner.low_codes[:r.rows]
        r.seed_init.bind_core(seed, packed=True)
        owner.direct_seed = seed
    owner.bind_output(backend.max_q, backend.device, selector, post)
    r.record_output = owner
    backend.record_format = "ordered20-after-H32-5B"
    backend.record5_owner = owner
    backend.record5_direct_seed = seed is not None
    return backend
