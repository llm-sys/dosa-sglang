"""Few prefill backend around the shared FusedRuntime.

Private hot/cold graphs own their workspaces. Calls arrive through the existing
ordinary prefill split-op; this object is not captured by an external graph.
No model projections, K quantization, fixture loading or installed JIT rebuild.
"""

from __future__ import annotations
from collections import OrderedDict
from .paged_backend import PagedBackend
from pathlib import Path
import torch
from .few_core import FusedRuntime
from .native_binding import load_modules
from .caller_state import CallerState
from .native_binding import FEW_SERVICE_LIMITS, FEW_PREFILL_MIN_KEYS
from .stock_carry import StockCarryPublisher
from . import transform_index as mapper
from .indexed import CertificateWriter, H22Record
from . import few_metadata as metadata
from .post_refiner import tensor_binding
from .private_capture import DirectCaptureFactory
from .query_slots import FEW_RECORD_CAPACITY, few_id_pool_pages

CURRENT_COMBINATION = "majority-identity-H16K256-fourmath-exact-schedule-H18-H22-query-pair-minority-fill-bf16-coeff-v1-hybrid12K-seedview-carry061236"


class GatherRho:
    def __init__(self, module, rt):
        self.module, self.rt = module, rt

    def plan_and_permuted_paged_gather_out(self, *args):
        # Gather remains the original actual-NK outer call. Capacity-backed
        # rho shares the same base address, but its gather view has actual N.
        rho = self.rt.rho[:args[-3].shape[0]] if self.rt.live_nk is not None else self.rt.rho
        self.module.plan_and_permuted_paged_gather_rho_out(*args, rho)
        self.rt.rho_from_gather = True
        self.rt.rho_source_signature = (
            args[-3].data_ptr(),
            args[-2].data_ptr(),
            self.rt.rho.data_ptr(),
        )


class FewBackend(PagedBackend):
    """One serial per-device execution slot, shared across Indexer layers.

    max_queries/max_keys are allocation capacities. bind_shape builds real,
    contiguous views, not [:Q,:NK] slices retaining a stale wider row stride.
    Huge tensors allocate once here; shape changes destroy private graphs only
    after the last complete caller/side-stream consumer. A real first call warms
    each new geometry before capture; no dummy or fabricated token warmup.
    """

    _full_paged = False

    def __init__(
        self, device, *, max_queries, max_keys,
        record_capacity=FEW_RECORD_CAPACITY, modules=None,
        record5_direct=False, score_heads=32, id_pool_pages=None,
    ):
        assert max_queries > 0 and max_queries % 8 == 0 and 12288 <= max_keys <= 1048576
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.capture_factory = DirectCaptureFactory()
        self.bootstrap = None
        self.score_heads = score_heads
        self.max_q = max_queries
        self.max_k = max_keys
        self.record_capacity = record_capacity
        self._record5_direct_storage = bool(record5_direct)
        self.fixed_id_pool = id_pool_pages is not None
        self.max_pages = (few_id_pool_pages(max_queries, max_keys)
                          if id_pool_pages is None else int(id_pool_pages))
        if self.max_pages < 256 * max_queries:
            raise ValueError("ID pool cannot be smaller than the original row reservation")
        self._initialize_slot()
        self.shape_graphs = OrderedDict()
        self.graph_cache_hits = self.graph_cache_evictions = 0
        self.live_nk_enabled = False
        self.logical_nk = None
        self.native_loads = 0
        self.modules = load_modules("few", score_heads=score_heads) if modules is None else modules
        self.native_loads = len(self.modules)
        m = self.modules
        self.rt = FusedRuntime(
            max_queries, max_keys, self.device, modules=m, metadata=metadata,
            capacity=record_capacity, pool_pages=self.max_pages, score_heads=score_heads,
            values_dtype=torch.uint8 if record5_direct else torch.float32,
        )
        self.fp16_emitter = H22Record(m["audit"], max_queries, self.device)
        writer = CertificateWriter(
            self.fp16_emitter, m, max_queries, max_keys, self.device, seed=12288
        )
        self.rt.fused = writer
        self.backing = self.tensor_buffers(self.rt)
        self.plan_backing = {k: list(v) for k, v in self.rt.plans.items()}
        self.meta_backing = self.tensor_buffers(self.rt.metadata_workspace)
        self.seed_backing = self.tensor_buffers(self.rt.seed_init)
        self.leaf_base = next(iter(self.rt.leaf_selectors.values()))
        self.leaf_backing = self.tensor_buffers(self.leaf_base)
        self.h16_base = self.tensor_buffers(writer.work)
        self.h18_base = self.tensor_buffers(self.rt.fused.work_h18)
        self._allocate_key_storage(max_keys)
        self._allocate_query_inputs(max_queries)
        self.map_status_storage = torch.empty(
            max_queries, device=self.device, dtype=torch.int32
        )
        self.run_max = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.over_events = torch.zeros_like(self.run_max)
        self.cold_hot = torch.empty(12288, device=self.device, dtype=torch.int64)
        self.gather_adapter = GatherRho(m["caller"], self.rt)
        self.lt = CallerState(m["caller"], gather=self.gather_adapter)
        self.lt.PRODUCTION_MIN_S = FEW_SERVICE_LIMITS["min_keys"]
        self.stock_carry = StockCarryPublisher(
            self.lt, m["caller"], self.device, self.max_q, self.max_k,
            min_keys=FEW_PREFILL_MIN_KEYS,
        )
        assert self.lt.PRODUCTION_MIN_S == self.rt.seed == self.lt.HOT_PREFIX
        self.mapper = mapper
        # The planner remains outside the hot graph, so its ordinary Python
        # epoch advances on every call. Do not freeze it by wrapping the whole
        # Python carry pipeline in a CUDA graph.
        # The legacy module keeps actual-K views and the original rebuild path.
        self.fixed_recovery_capacity = hasattr(m["leaf"], "replay_and_leaf_capacity")
        self._recovery_extent_binding = None
        if self.fixed_recovery_capacity:
            self.rt.recovery_k_capacity = self.pk_storage
            self.rt.recovery_scale_capacity = self.ps_storage.view(torch.float32).view(-1)[:self.max_k]
            self.rt.recovery_logical_nk = torch.empty(1, device=self.device, dtype=torch.int32)
        # Install after all original backing exists, before this slot can run.
        # This also covers the standalone create_backend/create_live_backend.
        from .key_rho_runtime import install_if_required as install_key_rho

        install_key_rho(self)
        from .direct_abs_runtime import install_if_required as install_direct_abs

        install_direct_abs(self)

    def _capture_bytes_per_row(self):
        return ((self.rt.seed + 256) * 4 + 2 * 2048 * 4 + 32 * 4)

    def _ensure_model_buffers(self):
        # These are the exact buffers later read/written by DG and original FI.
        # A generic uint8 allocation cannot reserve another CUDA graph pool.
        if self.rt.seed_logits is None:
            self.rt.seed_logits = torch.empty(
                (self.max_q, self.rt.seed + 256), device=self.device, dtype=torch.float32
            )
        if self.rt.seed_init.selected is None:
            self.rt.seed_init.selected = torch.empty(
                (self.max_q, 2048), device=self.device, dtype=torch.int32
            )

    def binding_manifest(self):
        """Return actual loaded dependencies and geometry, without GPU reads."""
        r = self.rt
        loaded = {key: str(module.__file__) for key, module in self.modules.items()}
        loaded.update(
            planner=str(r.planner128.__file__),
            exact=str((r.exact if r.record_output is None
                       else r.record_output.module).__file__),
            seed=str(r.seed_init.core.__file__),
            publisher=str(r.publisher.__file__),
            conditional=str(r.conditional.__file__),
            leaf_replay=str(r.leaf_producer.__file__),
        )
        return dict(
            combination=CURRENT_COMBINATION,
            fp16_coarse_record=self.fp16_emitter.workspace_invoice(),
            native=loaded,
            seed_python=str(Path(__file__).with_name("seed_initializer.py")),
            seed=12288,
            h16_tile=256,
            h18_tile=128,
            max_queries=self.max_q,
            logical_query_capacity=getattr(self, "logical_query_capacity", self.max_q),
            scratch_query_capacity=self.max_q,
            slab_call=getattr(self, "slab_last_call", None),
            max_keys=self.max_k,
            record_capacity=self.record_capacity,
            pages_per_actual_query=r.pool_pages / r.rows,
            reserved_id_pool_pages=self.max_pages,
            actual_id_pool_pages=r.pool_pages,
            fixed_id_pool=self.fixed_id_pool,
            actual_shape=self.shape,
            model_load_workspace_reserved=self.startup_ready is not None,
            capture_reserved_bytes=self.capture_reserved_bytes,
            seed_workspace_bytes=(0 if r.seed_logits is None else r.seed_logits.numel() * 4),
            seed_fi_workspace_bytes=(0 if r.seed_init.selected is None else r.seed_init.selected.numel() * 4),
            capture_reserve_released=self.capture_reserve_released,
            reserve_remaining_bytes=(
                0
                if self.capture_memory_reserve is None
                else self.capture_memory_reserve.numel()
            ),
            field_preparation=(r.abs_planner.invoice()
                               if r.abs_planner is not None else None),
            fixed_recovery_capacity=self.fixed_recovery_capacity,
            recovery_child_builds=r.recovery_child_builds,
        )

    _GRAPH_BINDINGS = (
        "cold_entry", "retained_cold_entries", "cold_args", "call_signature",
        "seed_score_layout",
    )

    def _graph_key(self):
        return self.shape, self._recovery_extent_binding

    def _restore_shape_graph(self):
        # bind_shape has reconstructed the same static addresses and strides.
        # Only private graphs/leases are cached, never caller data or telemetry.
        if not (self.live_nk_enabled and self.fixed_recovery_capacity):
            return
        key = self._graph_key()
        entry = self.shape_graphs.pop(key, None)
        if entry is None:
            # Retire after _wait_for_rebind has fenced every previous consumer.
            # Make room before capture, so at most two parent graphs remain live.
            if len(self.shape_graphs) >= 2:
                self.shape_graphs.popitem(last=False)
                self.graph_cache_evictions += 1
            return
        self.shape_graphs[key] = entry
        self.hot_graph, self.graph_kind, bindings = entry
        for name, value in bindings.items():
            setattr(self.rt, name, value)
        self.graph_cache_hits += 1

    def _remember_shape_graph(self):
        if self.live_nk_enabled and self.fixed_recovery_capacity:
            self.shape_graphs[self._graph_key()] = (
                self.hot_graph, self.graph_kind,
                {name: getattr(self.rt, name) for name in self._GRAPH_BINDINGS},
            )
            assert len(self.shape_graphs) <= 2

    def _wait_for_rebind(self, *, keep_recovery=False):
        if self.last_done is not None:
            self.last_done.synchronize()
        # These are private graphs, never exported/captured by a caller.
        self.hot_graph = None
        self.graph_kind = None
        r = self.rt
        # Each parent leases its child; release the dispatch before the body.
        if not keep_recovery:
            r.cold_entry = None
            r.retained_cold_entries = []

    def bind_shape(self, rows, nk):
        if self.live_nk_enabled:
            if not 12288 <= nk <= self.max_k:
                raise ValueError("Actual NK outside live capacity")
            self.logical_nk = nk
            nk = self.max_k
        extent_binding = (
            (nk, tensor_binding(self.rt.recovery_logical_nk))
            if self.fixed_recovery_capacity else None
        )
        if self.shape == (rows, nk) and self._recovery_extent_binding == extent_binding:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("shape rebind during external capture")
        if not (0 < rows <= self.max_q and rows % 8 == 0 and 12288 <= nk <= self.max_k):
            raise ValueError("actual shape outside allocated native geometry")
        keep_recovery = (self.fixed_recovery_capacity and self.shape is not None
                         and self.shape[0] == rows)
        self._wait_for_rebind(keep_recovery=keep_recovery)
        r = self.rt
        if self.fixed_recovery_capacity:
            r.recovery_k_capacity = self.pk_storage
            r.recovery_scale_capacity = self.ps_storage.view(torch.float32).view(-1)[:self.max_k]
        b = self.backing
        pages = (self.max_pages if self.fixed_id_pool
                 else few_id_pool_pages(rows, nk))
        for name in [
            "q",
            "w",
            "starts",
            "ends",
            "capture",
            "safe_tau",
            "error",
            "zero",
            "seed_end",
            "seed_tau",
            "sample_scores",
            "sample_masks",
            "values",
            "ids",
            "counts",
            "ordered_output",
            "output",
            "counts_original",
            "error_original",
            "replay_work",
            "order",
            "error_before_completion",
        ]:
            setattr(r, name, b[name][:rows])
        for name in ["budget4", "common_counts"]:
            setattr(r, name, b[name][: rows // 4])
        r.budget8 = b["budget8"][: rows // 8]
        r.solo_counts = b["solo_counts"][:rows]
        r.rho = b["rho"][:nk]
        r.keep_words = self.flat_view(
            b["keep_words"], (rows // 4, (nk - r.seed + 31) // 32, 4)
        )
        r.page_table = self.flat_view(
            b["page_table"], (rows + rows // 4, (nk + 127) // 128)
        )
        r.id_pool = b["id_pool"][:pages]
        r.solo_tasks = b["solo_tasks"][:pages]
        r.common_tasks = b["common_tasks"][:pages]
        r.rows, r.nk, r.pool_pages = rows, nk, pages
        r.replay_batch = min(rows, 512)
        recovery_pitch = self.max_k if self.fixed_recovery_capacity else nk
        r.replay_scores = self.flat_view(b["replay_scores"], (r.replay_batch, recovery_pitch))
        r.replay_counts = b["replay_counts"][: r.replay_batch]
        r.plans = {
            paid: [x[:rows] for x in values]
            for paid, values in self.plan_backing.items()
        }
        w = r.metadata_workspace
        w.rows = rows
        w.nk = nk
        w.parts = (nk + 1023) // 1024
        w.ps = self.meta_backing["ps"][: w.parts]
        w.pb = self.meta_backing["pb"][: w.parts]
        w.seed_bad = self.meta_backing["seed_bad"][:rows]
        for name in ["zero", "lengths", "diag"]:
            setattr(r.seed_init, name, self.seed_backing[name][:rows])
        if not self.live_nk_enabled:
            for work, base, tile, grid_factor in [
                (r.fused.work, self.h16_base, 256, 2),
                (r.fused.work_h18, self.h18_base, 128, 1),
            ]:
                work.rows, work.groups = rows, rows // 8
                work.tiles = (nk - r.seed + tile - 1) // tile
                work.grid = grid_factor * max(work.groups, 8 * work.sms)
                work.ids = base["ids"][: work.groups]
                work.inverse = base["inverse"][: work.groups]
        chunks = (recovery_pitch + 32767) // 32768
        r.leaf_selectors = {}
        sizes = {
            min(r.replay_batch, rows - begin)
            for begin in range(0, rows, r.replay_batch)
        }
        for size in sizes:
            leaf = object.__new__(type(self.leaf_base))
            leaf.__dict__.update(self.leaf_base.__dict__)
            leaf.nq = size
            leaf.nk = recovery_pitch
            leaf.chunks = chunks
            leaf.leaf = self.leaf_backing["leaf"][: 8 * chunks]
            leaf.leaf_lengths = self.leaf_backing["leaf_lengths"][: 8 * chunks]
            leaf.leaf_zero = self.leaf_backing["leaf_zero"][: 8 * chunks]
            leaf.merge = self.flat_view(self.leaf_backing["merge"], (8, chunks * 2048))
            leaf.ids = self.flat_view(self.leaf_backing["ids"], (8, chunks * 2048))
            leaf.fallback_lengths = self.leaf_backing["fallback_lengths"][:size]
            leaf.full_zero = self.leaf_backing["full_zero"][:size]
            r.leaf_selectors[size] = leaf
        self._bind_gather_views(nk)
        self.map_status = self.map_status_storage[:rows]
        if self.fixed_recovery_capacity and self._recovery_extent_binding != extent_binding:
            stream = torch.cuda.current_stream(self.device)
            if self.startup_ready is not None:
                stream.wait_event(self.startup_ready)
            r.recovery_logical_nk.fill_(nk)
            self._recovery_extent_binding = extent_binding
        self.shape = (rows, nk)
        self.generation += 1
        if self.live_nk_enabled:
            from .live_nk_runtime import bind_components
            bind_components(self, rows)
        self._restore_shape_graph()

    def _runtime_call(self):
        rows, _ = self.shape
        return self.rt(
            self.q_input[:rows],
            self.pk,
            self.ps,
            self.w_input[:rows],
            self.start_input[:rows],
            self.end_input[:rows],
        )

    def _capture_after_real_call(self):
        with self.capture_factory.session() as factory:
            # First use runs the actual request through the original eager recovery
            # body, warming all needed native/FI variants with real fields.
            with torch.cuda.device(self.device):
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(self.device))
                ready.synchronize()
                # A changed gather policy may replace this shape's parent. Release
                # its cache lease only after all preceding work has completed.
                self.shape_graphs.pop(self._graph_key(), None)
                self.hot_graph = None
                self.graph_kind = None
                self.rt.prepare_cold_graph()
                if self.live_nk_enabled:
                    self.rt.fused.prepare_h22_graphs()
                stream = factory.new_stream(self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                graph = factory.new_graph()
                with factory.capture(graph, stream):
                    self._runtime_call()
            graph._dosa_recovery_lease = (
                self.rt.cold_entry,
                self.rt.seed_logits,
                self.rt.seed_init.selected,
            )
            if self.live_nk_enabled:
                graph._dosa_h22_children = self.rt.fused.h22_graphs
            self.hot_graph = graph
            self.graph_kind = self.rt.rho_from_gather
            self.setup_calls += 1
            self._remember_shape_graph()
            self._reserve_uncaptured_capacity()



def create_current_backend(
    device, *, max_queries, max_keys, record_capacity=FEW_RECORD_CAPACITY, modules=None
):
    """Create one reusable execution slot with the pinned, accepted few combination.

    Call once per serial device slot and reuse run_paged across layers/requests.
    Capacities control allocation; actual Q/K remain per-call run_paged inputs.
    """
    return FewBackend(
        device,
        max_queries=max_queries,
        max_keys=max_keys,
        record_capacity=record_capacity,
        modules=modules,
    )
