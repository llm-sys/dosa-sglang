"""One current full execution slot for ordinary projected-Q/K prefill calls."""

from __future__ import annotations
from .paged_backend import PagedBackend
from pathlib import Path
import torch
from .post_refiner import ConditionalRecoveryRefiner
from .private_capture import DirectCaptureFactory
from .gather_norm_adapter import GatherNormAdapter
from .full_core import run_full_core
from .native_binding import load_modules
from .caller_state import CallerState
from .stock_carry import StockCarryPublisher
from . import surrogate_metadata as metadata


class FullBackend(PagedBackend):
    """Serial per-device slot; private hot/cold graphs are never exported.

    Capacities allocate scratch once. Every call supplies actual Q/K, original
    cache/tables and visibility. Existing original-H32 recovery remains active.
    """

    _full_paged = True

    def __init__(self, device, *, max_queries, max_keys, modules=None):
        if not (
            max_queries > 0 and max_queries % 8 == 0 and 196608 <= max_keys <= 1048576
        ):
            raise ValueError(
                "full capacity requires Q multiple of8 and K in196608..1048576"
            )
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("current full backend requires CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.capture_factory = DirectCaptureFactory()
        self.bootstrap = None
        self.max_q, self.max_k = int(max_queries), int(max_keys)
        self.modules = load_modules("full") if modules is None else modules
        m = self.modules
        self.refiner = ConditionalRecoveryRefiner(
            m["post"],
            self.max_q,
            self.max_k,
            self.device,
        )
        r = self.refiner
        r.use_fused_large = True
        r.conditional_enabled = True
        r.metadata = metadata
        r.metadata_workspace = metadata.allocate_workspace(self.max_k, self.device)
        self.native = GatherNormAdapter(
            m["caller"],
            r.metadata_workspace,
            self.max_k,
        )
        r.native = self.native
        self.backing = self.tensor_buffers(r)
        self.meta_backing = self.tensor_buffers(r.metadata_workspace)
        self.gather_backing = (self.native.maximum, self.native.bad)
        self._allocate_query_inputs(self.max_q)
        self._allocate_key_storage(self.max_k)
        self.output_storage = torch.empty(
            (self.max_q, 2048), device=self.device, dtype=torch.int32
        )
        self.run_max = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.over_events = torch.zeros_like(self.run_max)
        self.cold_hot = torch.empty(12288, device=self.device, dtype=torch.int64)
        self.lt = CallerState(m["caller"], gather=self.native)
        self.lt.CANDIDATE_STORAGE_DTYPE = torch.uint8
        self.stock_carry = StockCarryPublisher(self.lt, m["caller"], self.device, self.max_q, self.max_k)
        # Existing helpers already expose flat actual-width views from grow-only backing.
        self.lt._prep_bufs(self.max_q, self.lt.NB, self.device)
        self.lt._paged_cand_bufs(self.max_q, self.max_k, self.device)
        self.lt._ks0_keh(self.max_q, self.lt._PAGED_CANDIDATE_INLINE, self.device)
        self._initialize_slot()
        self.last_plan = None
        self.seed_logits = None
        self.fixed_recovery_capacity = hasattr(self.refiner.core, "replay_capacity")
        self._recovery_extent_host = None
        self._recovery_extent_binding = None
        if self.fixed_recovery_capacity:
            self.refiner.recovery_k_capacity = self.pk_storage
            self.refiner.recovery_scale_capacity = self.ps_storage.view(torch.float32).view(-1)[:self.max_k]
            self.refiner.recovery_logical_nk = torch.empty(
                1, device=self.device, dtype=torch.int32
            )

    def _capture_bytes_per_row(self):
        return (8192 + 256) * 4

    def _ensure_model_buffers(self):
        if self.seed_logits is None:
            self.seed_logits = torch.empty(
                (self.max_q, self.lt._PAGED_CANDIDATE_INLINE + 256),
                device=self.device, dtype=torch.float32,
            )

    def _wait_for_rebind(self, *, keep_recovery=False):
        if self.last_done is not None:
            self.last_done.synchronize()
        self.hot_graph = None
        self.graph_kind = None
        r = self.refiner
        if not keep_recovery:
            r.active_recovery_entry = None
            r.recovery_entries = []
        r.last_recovery_args = None
        r.last_recovery_dispatch = None

    def bind_shape(self, rows, nk):
        extent_binding = (
            (nk, self.refiner.recovery_logical_nk.data_ptr())
            if self.fixed_recovery_capacity else None
        )
        if self.shape == (rows, nk) and self._recovery_extent_binding == extent_binding:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("external capture cannot rebind")
        if not (
            0 < rows <= self.max_q and rows % 8 == 0 and 196608 <= nk <= self.max_k
        ):
            raise ValueError("actual shape exceeds full allocation capacity")
        keep_recovery = (
            self.fixed_recovery_capacity
            and self.shape is not None
            and self.shape[0] == rows
        )
        self._wait_for_rebind(keep_recovery=keep_recovery)
        r = self.refiner
        if self.fixed_recovery_capacity:
            r.recovery_k_capacity = self.pk_storage
            r.recovery_scale_capacity = self.ps_storage.view(torch.float32).view(-1)[:self.max_k]
        b = self.backing
        for name in [
            "boundary_ids",
            "boundary_counts",
            "values",
            "ids",
            "counts",
            "error",
            "epsilon",
            "offsets",
            "tau",
            "final_tau",
            "map_status",
            "fi_lengths",
        ]:
            setattr(r, name, b[name][:rows])
        r.tasks = b["tasks"][: rows * 32]
        r.rows, r.nk, r.replay_rows = rows, nk, min(rows, 512)
        replay_pitch = self.max_k if self.fixed_recovery_capacity else nk
        r.replay_scores = self.flat_view(b["replay_scores"], (r.replay_rows, replay_pitch))
        r.replay_counts = b["replay_counts"][: r.replay_rows]
        r.replay_selected = b["replay_selected"][: r.replay_rows]
        w = r.metadata_workspace
        w.nk = nk
        w.parts = (nk + w.block - 1) // w.block
        w.partial = self.meta_backing["partial"][: w.parts]
        w.partial_bad = self.meta_backing["partial_bad"][: w.parts]
        self.native.parts = (nk + 31) // 32
        # Q16K serving grows K through several powers of two. Keep one reducer
        # specialization for that path; the actual NP still masks every load.
        self.native.fixed_finish = rows == 16384
        self.native.maximum = self.gather_backing[0][: self.native.parts]
        self.native.bad = self.gather_backing[1][: self.native.parts]
        self._bind_gather_views(nk)
        self.output = self.output_storage[:rows]
        self.prep = self.lt._prep_bufs(rows, self.lt.NB, self.device)
        self.paged = self.lt._paged_cand_bufs(rows, nk, self.device)
        if self.fixed_recovery_capacity and self._recovery_extent_binding != extent_binding:
            # All views are bound before publishing a new logical extent.
            # Startup allocation dependencies precede a first metadata write
            # from a different caller stream; last_done was completed above.
            stream = torch.cuda.current_stream(self.device)
            if self.startup_ready is not None:
                stream.wait_event(self.startup_ready)
            r.recovery_logical_nk.fill_(nk)
            self._recovery_extent_host = nk
            self._recovery_extent_binding = extent_binding
        self.shape = (rows, nk)
        self.generation += 1

    def _runtime_call(self):
        rows, _ = self.shape
        return run_full_core(
            self.lt,
            self.modules["primary"],
            self.refiner,
            self.q_input[:rows],
            self.pk,
            self.ps,
            self.w_input[:rows],
            self.start_input[:rows],
            self.end_input[:rows],
            self.output,
            seed_workspace=self.seed_logits,
        )

    def _capture_after_real_call(self):
        with self.capture_factory.session() as factory:
            with torch.cuda.device(self.device):
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(self.device))
                ready.synchronize()
                self.refiner.prepare_recovery_graph()
                stream = factory.new_stream(self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                graph = factory.new_graph()
                # Gather/seed/recovery already own their outputs. Remaining
                # scratch is sized by Q and the scale-reduction specialization;
                # changing K alone does not require a new allocation shape.
                allocation_key = ((self.shape[0], self.native.reduction_width)
                                  if self.native.fused else None)
                with factory.capture(graph, stream, allocation_key=allocation_key):
                    self._runtime_call()
            self.refiner.retain_on(
                graph,
                self.lt,
                self.q_input,
                self.w_input,
                self.start_input,
                self.end_input,
                self.pk,
                self.ps,
                self.output,
                self.prep,
                self.paged,
                self.refiner.metadata_workspace,
                self.native.maximum,
                self.native.bad,
                self.seed_logits,
            )
            self.hot_graph = graph
            self.graph_kind = self.native.fused
            self.setup_calls += 1
            self._reserve_uncaptured_capacity()

    def binding_manifest(self):
        return dict(
            variant="record5-global-scale-conditional-carry-warp-full",
            native={k: str(v.__file__) for k, v in self.modules.items()},
            max_queries=self.max_q,
            max_keys=self.max_k,
            actual_shape=self.shape,
            fixed_recovery_capacity=self.fixed_recovery_capacity,
            recovery_storage_pitch=(self.max_k if self.fixed_recovery_capacity else None),
            recovery_logical_extent_host=self._recovery_extent_host,
            seed=self.lt._PAGED_CANDIDATE_INLINE,
            stock_prior=self.stock_carry.manifest(),
            hot_prefix=self.lt.HOT_PREFIX,
            candidate_value_dtype=str(self.lt.CANDIDATE_STORAGE_DTYPE),
            pool_pages_per_actual_query=self.lt._PAGED_POOL_PAGES_PER_ROW,
            carry_planner="qualified suffix-prefix plan042641Z",
            carry_emitter="qualified warp-contiguous emitter061236Z",
            tiered_seed_12k=self.lt.TIERED_SEED_12K,
            boundary_capacity=4096,
            exact_values_capacity=8192,
            recovery="current fused-large then original8batch body behind any(error!=0)",
            mapping="qualified inplace map/votes/PAGED then caller output copy",
            model_load_workspace_reserved=self.startup_ready is not None,
            capture_reserved_bytes=self.capture_reserved_bytes,
            seed_workspace_bytes=(0 if self.seed_logits is None else self.seed_logits.numel() * self.seed_logits.element_size()),
            capture_reserve_released=self.capture_reserve_released,
            reserve_remaining_bytes=(
                0
                if self.capture_memory_reserve is None
                else self.capture_memory_reserve.numel()
            ),
            reserved_layer_count=len(self.lt._reserved_carry)
            + len(self.lt._request_slots),
        )




def create_current_full_backend(device, *, max_queries, max_keys, modules=None):
    return FullBackend(
        device, max_queries=max_queries, max_keys=max_keys, modules=modules
    )
