"""Complete paged-ID Few execution over already physically gathered K.
The caller charges actual gather/permutation and final map/vote/carry/PAGED.
All query ordering, exact seed/FI, sample128 plan, rho, direct-tau metadata, paged-ID certificate scan + original H32 completion,
final FI and real recovery execute on every call; nothing numerical is cached.
"""

from dataclasses import dataclass
import torch, flashinfer, deep_gemm, triton
import triton.language as tl
from .leaf_runtime import FusedLeafReplayFI
from .seed_initializer import ExactSeedInitializer, seed_scores_view
from .seed_output import seed_logits
from .post_refiner import tensor_binding
from .private_capture import capture_child
from .query_slots import FEW_RECORD_CAPACITY, few_id_pool_pages


@dataclass(frozen=True)
class RecoveryGraphs:
    """Publish a complete graph pair and its operand leases together."""

    dispatch: object
    body: object
    signature: tuple
    inputs: tuple
    outputs: tuple


class FusedRuntime:
    def __init__(
        self, rows, nk, device, *, modules, metadata,
        capacity=FEW_RECORD_CAPACITY, pool_pages=None,
        values_dtype=torch.float32, score_heads=32,
    ):
        assert rows % 8 == 0 and nk >= 12288
        self.score_heads = score_heads
        mask_dtype = torch.int32 if score_heads == 32 else torch.int64
        self.core = modules["sample"]
        self.metadata = metadata
        self.fused = None  # The owning backend binds its certificate writer.
        self.pool_pages = (few_id_pool_pages(rows, nk)
                           if pool_pages is None else int(pool_pages))
        self.seed = 12288
        self.exact = modules["exact"]
        self.compactor = modules["common"]
        self.publisher = modules["publisher"]
        self.record_output = None
        self.rows, self.nk, self.capacity = rows, nk, capacity
        self.live_nk = None
        self.live_common = None
        self.live_exact = None
        h16_contract = getattr(modules.get("h16"), "field_contract", None)
        self._requires_abs_shift = (
            callable(h16_contract) and h16_contract() == "abs-shift-dd2-v1"
        )
        self.abs_planner = None
        key_contract = getattr(modules.get("h16"), "required_key_rho_contract", None)
        self._required_key_rho_contract = key_contract() if callable(key_contract) else None
        self.key_rho_contract = None
        self._qualified_rms = None
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count

        def empty(*shape, dtype=torch.float32):
            return torch.empty(shape, device=device, dtype=dtype)

        self.q = empty(rows, score_heads, 128, dtype=torch.uint8)
        self.w = empty(rows, score_heads)
        self.starts = empty(rows, dtype=torch.int32)
        self.ends = empty(rows, dtype=torch.int32)
        self.budget8 = empty(rows // 8, dtype=torch.int32)
        self.budget4 = empty(rows // 4, dtype=torch.int32)
        self.capture = empty(rows, 129, dtype=mask_dtype)
        self.planner128 = modules["planner"]
        self.sample_ids = 32 + torch.arange(128, device=device, dtype=torch.int64) * 64
        self.sample_k = empty(128, 128, dtype=torch.uint8)
        self.sample_scale = empty(128)
        self.sample_scores = empty(rows, 128)
        self.sample_masks = empty(rows, 128, dtype=mask_dtype)
        self.safe_tau = empty(rows)
        self.rho = empty(nk, dtype=torch.int32)
        self.error = empty(rows, dtype=torch.int32)
        self.metadata_workspace = metadata.Workspace(nk, rows, device)
        self.zero = torch.zeros(rows, device=device, dtype=torch.int32)
        self.seed_end = torch.full((rows,), 12288, device=device, dtype=torch.int32)
        self.seed_init = ExactSeedInitializer(rows, device, modules["seed"])
        self.seed_logits = None
        self.seed_tau = empty(rows)
        self.high_paid = 28 if score_heads == 64 else 20
        self.plans = {}
        for paid in [14, 16, self.high_paid]:
            H = paid + 2
            words = 64 if paid <= 16 else 96
            self.plans[paid] = [
                empty(rows, H, 128, dtype=torch.uint8),
                empty(rows, H),
                empty(rows, words),
                empty(rows),
                empty(rows),
                empty(rows, dtype=torch.int32),
                empty(rows, paid, dtype=torch.int32),
                empty(rows, 2, dtype=torch.int64),
                empty(rows, 4, dtype=torch.int32),
            ]
        self.keep_words = empty(
            rows // 4, triton.cdiv(nk - 12288, 32), 4, dtype=torch.int32
        )
        self.id_pool = empty(self.pool_pages, 128, dtype=torch.int32)
        self.page_table = empty(
            rows + rows // 4, triton.cdiv(nk, 128), dtype=torch.int32
        )
        self.pool_head = empty(1, dtype=torch.int32)
        self.pool_status = empty(1, dtype=torch.int32)
        self.solo_counts = empty(rows, dtype=torch.int32)
        self.common_counts = empty(rows // 4, dtype=torch.int32)
        self.solo_tasks = empty(self.pool_pages, 2, dtype=torch.int32)
        self.common_tasks = empty(self.pool_pages, 2, dtype=torch.int32)
        self.solo_n = empty(1, dtype=torch.int32)
        self.common_n = empty(1, dtype=torch.int32)
        self.error_before_completion = empty(rows, dtype=torch.int32)
        self.values = empty(rows, capacity, dtype=values_dtype)
        self.ids = empty(rows, capacity, dtype=torch.int32)
        self.counts = empty(rows, dtype=torch.int32)
        self.empty_byte = empty(0, dtype=torch.uint8)
        self.empty_float = empty(0)
        self.ordered_output = empty(rows, 2048, dtype=torch.int32)
        self.output = empty(rows, 2048, dtype=torch.int32)
        self.counts_original = empty(rows, dtype=torch.int32)
        self.error_original = empty(rows, dtype=torch.int32)
        self.replay_batch = min(rows, 512)
        self.replay_scores = empty(self.replay_batch, nk)
        self.replay_counts = empty(self.replay_batch, dtype=torch.int32)
        self.replay_work = empty(rows, dtype=torch.int32)

        self.leaf_producer = modules["leaf"]
        self.leaf_selectors = {
            size: FusedLeafReplayFI(size, nk, device, modules["hierarchy"])
            for size in {
                min(self.replay_batch, rows - begin)
                for begin in range(0, rows, self.replay_batch)
            }
        }
        self.order = empty(rows, dtype=torch.int64)
        self.conditional = modules["post"]
        self.cold_entry = None
        self.retained_cold_entries = []
        self.cold_runs = torch.zeros(1, device=device, dtype=torch.int32)
        self.forced_error = None
        self.recovery_k_capacity = None
        self.recovery_scale_capacity = None
        self.recovery_logical_nk = None
        self.recovery_child_builds = 0

    def _recovery_binding(self, args):
        qs, k, scale = args
        if self.recovery_logical_nk is None:
            return (k.data_ptr(), scale.data_ptr(), tuple(k.shape), tuple(k.stride()), str(k.device))
        operands = (*args, self.w, self.starts, self.ends, self.error, self.order,
                    self.output, self.replay_scores, self.replay_counts,
                    self.replay_work, self.cold_runs, self.recovery_logical_nk)
        leaves = tuple(
            (size, leaf.nq, leaf.nk, leaf.chunks, id(leaf.core),
             tuple((name, tensor_binding(value)) for name, value in sorted(vars(leaf).items())
                   if isinstance(value, torch.Tensor)))
            for size, leaf in sorted(self.leaf_selectors.items())
        )
        return (tuple(tensor_binding(value) for value in operands), self.rows,
                self.replay_batch, self.replay_scores.shape[1], leaves,
                id(self.leaf_producer), id(self.publisher), id(self.conditional))

    def recovery_body(self):
        qs, k, scale = self.cold_args
        outputs = []
        self.cold_runs.add_(1)
        for begin in range(0, self.rows, self.replay_batch):
            end = min(self.rows, begin + self.replay_batch)
            n = end - begin
            status = self.error[begin:end]
            rv = self.replay_scores[:n]
            rc = self.replay_counts[:n]
            recovered = self.leaf_selectors[n].replay_and_select(
                self.leaf_producer,
                qs[begin:end],
                k,
                scale,
                self.w[begin:end],
                self.starts[begin:end],
                self.ends[begin:end],
                status,
                rv,
                rc,
                logical_nk=self.recovery_logical_nk,
            )
            self.publisher.recovered(
                recovered, self.order[begin:end], status, self.output
            )
            self.replay_work[begin:end].copy_(rc)
            outputs.append(recovered)
        return outputs

    def prepare_cold_graph(self):
        # All cold native/FI variants have already executed in the untimed warm call.
        # Captured Q/W/order/status are owned static buffers; K/scale must retain this
        # ordinary cache identity. The graphs and allocated outputs own their lifetime.
        if self.recovery_logical_nk is not None:
            if (self.cold_entry is not None
                    and self.cold_entry.signature == self.call_signature):
                return
            # Retained parents already lease their old child and all operands.
            # Only one compatible body/dispatch pair is current in this slot.
            self.cold_entry = None
            self.retained_cold_entries = []
        inputs = self.cold_args
        with torch.cuda.device(self.error.device):
            with capture_child(self.error.device) as body:
                outputs = self.recovery_body()
            # Keep every recovery operand alive if a parent outlives this slot.
            body._dosa_recovery_inputs = (
                inputs, outputs, self.w, self.starts, self.ends,
                self.error, self.order, self.output, self.replay_scores,
                self.replay_counts, self.replay_work, self.cold_runs,
                self.leaf_selectors,
                self.recovery_logical_nk,
            )
            with capture_child(self.error.device, keep_graph=False) as dispatch:
                self.conditional.attach_if_any(self.error, False, body.raw_cuda_graph())
            dispatch._dosa_recovery_lease = (body, outputs, inputs)
        # A failed build leaves no half-published body/dispatch pair. Legacy
        # actual-K parents retain their previous entry only after success.
        if self.recovery_logical_nk is None and self.cold_entry is not None:
            self.retained_cold_entries.append(self.cold_entry)
        self.cold_entry = RecoveryGraphs(
            dispatch, body, self.call_signature, inputs, tuple(outputs)
        )
        self.recovery_child_builds += 1
        self.cold_runs.zero_()

    def complete_h32(self, q, k, scale):
        """Complete solo then common work using explicitly bound score storage."""
        if self.record_output is None:
            readers = (self.exact.exact_paged_solo, self.exact.exact_paged_common)
            values, extent = self.values, ()
        else:
            readers, values, extent = self.record_output.prepare_h32()
        streams = (
            (self.solo_counts, self.solo_tasks, self.solo_n,
             self.page_table[:self.rows]),
            (self.common_counts, self.common_tasks, self.common_n,
             self.page_table[self.rows:]),
        )
        for read, (counts, tasks, task_count, pages) in zip(readers, streams):
            read(
                q, k, scale, self.w, self.safe_tau, self.id_pool,
                counts, tasks, task_count, values, self.ids, self.counts,
                0, self.sms, pages, self.pool_pages, self.starts, self.ends,
                self.error, *extent,
            )

    def select(self, values, counts):
        return flashinfer.top_k_ragged_transform(
            values,
            self.zero[: len(values)],
            counts,
            2048,
            deterministic=False,
            dsa_graph_safe=True,
        )

    def __call__(self, q, k, scale, weights, starts, ends):
        if self._requires_abs_shift and self.planner128 is not self.abs_planner:
            raise RuntimeError("ABS H16 requires its owner-local direct field planner")
        if self._required_key_rho_contract is not None and (
            self.key_rho_contract != self._required_key_rho_contract
            or self._qualified_rms is None
        ):
            raise RuntimeError("H16 requires matching current-scale rho producers")
        if self.live_nk is None:
            self.page_table.fill_(-1)
        self.pool_head.zero_()
        self.pool_status.zero_()
        self.solo_counts.zero_()
        self.common_counts.zero_()
        self.solo_n.zero_()
        self.common_n.zero_()
        self.error.zero_()
        self.replay_work.zero_()
        if self.live_nk is not None:
            from .live_nk_ops import initialize_table
            initialize_table(self)
        np = (weights > 0).sum(1)
        tags = (np > 14).long() + (np > 16).long() + (np > self.high_paid).long()
        self.order.copy_(tags.argsort(stable=True))
        order = self.order
        torch.index_select(q.view(torch.uint8), 0, order, out=self.q)
        torch.index_select(weights, 0, order, out=self.w)
        torch.index_select(starts, 0, order, out=self.starts)
        torch.index_select(ends, 0, order, out=self.ends)
        mx = np.index_select(0, order).reshape(-1, 8).max(1).values
        self.budget8.copy_(torch.where(mx <= 14, 14, torch.where(mx <= 16, 16, self.high_paid)))
        self.budget4.copy_(self.budget8.repeat_interleave(2))
        qs = self.q.view(torch.float8_e4m3fn)
        seed_scores = seed_scores_view(
            seed_logits(qs, k[:12288], scale[:12288], self.w, self.zero,
                        self.seed_end, workspace=self.seed_logits)
        )
        self.seed_score_layout = dict(
            shape=list(seed_scores.shape),
            stride=list(seed_scores.stride()),
            storage_offset=seed_scores.storage_offset(),
            storage_bytes=seed_scores.untyped_storage().nbytes(),
            contiguous=seed_scores.is_contiguous(),
        )
        self.seed_init(
            seed_scores, self.values, self.ids, self.counts, self.seed_tau, self.error
        )
        tau = self.seed_tau
        torch.index_select(k.view(torch.uint8), 0, self.sample_ids, out=self.sample_k)
        torch.index_select(scale, 0, self.sample_ids, out=self.sample_scale)
        self.core.seed_score(
            qs,
            self.sample_k.view(torch.float8_e4m3fn),
            self.sample_scale,
            self.w,
            self.sample_scores,
            self.sample_masks,
            self.sms,
        )
        self.capture[:, :128].copy_(self.sample_masks)
        self.capture[:, 128].copy_(self.error)
        for paid in [14, 16, self.high_paid]:
            self.planner128.plan128(
                paid, qs, self.w, self.capture, self.plans[paid], self.budget8
            )
        if not (
            self.rho_from_gather
            and self.rho_source_signature
            == (k.data_ptr(), scale.data_ptr(), self.rho.data_ptr())
        ):
            if self._qualified_rms is not None:
                self._qualified_rms(
                    k.view(torch.uint8), scale, self.rho, self.live_nk, self.sms, self.seed
                )
            elif self.live_nk is None:
                self.core.rms(k.view(torch.uint8), self.rho)
            else:
                self.live_exact.rms_live_nk(k.view(torch.uint8), self.rho, self.live_nk, self.sms, self.seed)
        self.metadata.prepare(
            self.w,
            self.plans,
            self.budget8,
            tau,
            seed_scores,
            scale,
            self.rho,
            self.error,
            self.safe_tau,
            self.metadata_workspace,
            seed_diag=(self.seed_init.diag if getattr(
                self.seed_init.core, "seed_finite_contract", None
            ) == "all-raw-seed-finite-diag-mode3-v1" else None),
        )
        for paid in [14, 16, self.high_paid]:
            v = self.plans[paid]
            x = [
                self.q,
                k.view(torch.uint8),
                scale,
                self.rho,
                self.w,
                v[0],
                v[1],
                v[2],
                v[3],
                v[4],
                v[5],
                self.starts,
                self.ends,
                self.budget4,
                self.safe_tau,
            ]
            streams = [
                self.solo_counts,
                self.solo_tasks,
                self.solo_n,
                self.common_counts,
                self.common_tasks,
                self.common_n,
            ]
            self.fused.emit_trusted_candidates(
                paid,
                x,
                streams,
                self.id_pool,
                self.page_table,
                self.pool_head,
                self.pool_status,
                self.error,
                12288,
                self.keep_words,
            )
        if self.live_nk is None:
            self.compactor.compact_candidates(
                streams,
                self.id_pool,
                self.page_table,
                self.pool_head,
                self.pool_status,
                self.error,
                self.keep_words,
                self.budget4,
                12288,
                self.nk,
                self.starts,
                self.ends,
            )
        else:
            self.live_common.compact_candidates_live_nk(
                streams, self.id_pool, self.page_table, self.pool_head,
                self.pool_status, self.error, self.keep_words, self.budget4,
                self.seed, self.nk, self.starts, self.ends, self.live_nk)
        self.error_before_completion.copy_(self.error)
        # Raw counts and raw task counters are retained. The exact reader validates/clamps
        # task counts against actual allocations and flags any invalid count/page/ID.
        self.complete_h32(qs, k, scale)
        if self.record_output is None:
            pick = self.select(self.values, self.counts.clamp(0, self.capacity))
            self.publisher.normal(self.ids, pick, order, self.error, self.output)
        else:
            self.record_output.select_and_publish(self, qs, k, scale)
        recovery_k = k if self.recovery_logical_nk is None else self.recovery_k_capacity
        recovery_scale = scale if self.recovery_logical_nk is None else self.recovery_scale_capacity
        self.cold_args = (qs, recovery_k, recovery_scale)
        self.call_signature = self._recovery_binding(self.cold_args)
        if self.forced_error is not None:
            self.error.bitwise_or_(self.forced_error)
        entry = self.cold_entry
        if entry is not None and self.call_signature == entry.signature:
            if torch.cuda.is_current_stream_capturing():
                self.conditional.attach_if_any(
                    self.error, False, entry.body.raw_cuda_graph()
                )
            else:
                entry.dispatch.replay()
        else:
            self.recovery_body()
        self.counts_original.index_copy_(0, order, self.counts)
        self.error_original.index_copy_(0, order, self.error)
        return self.output
