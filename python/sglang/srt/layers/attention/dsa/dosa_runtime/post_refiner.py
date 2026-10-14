"""Capture-only IF around the current fused refiner's original recovery body."""

from dataclasses import dataclass
import torch
from .fi_output import topk_into
from .private_capture import capture_child


@dataclass(frozen=True)
class RecoveryEntry:
    signature: tuple
    graph: object
    inputs: tuple
    views: tuple
    outputs: tuple
    modules: tuple
    audit: object


def tensor_binding(t):
    return (
        t.data_ptr(),
        tuple(t.shape),
        tuple(t.stride()),
        str(t.dtype),
        str(t.device),
        t.storage_offset(),
    )


class ConditionalRecoveryRefiner:
    """Current fused-large completion followed by exact conditional recovery.

    The native post module owns completion, replay and conditional dispatch.
    Eager warmup and captured recovery execute the same recovery body.
    """

    def __init__(
        self,
        post,
        rows,
        nk,
        device,
    ):
        self.core = self.post = self.conditional = post
        self.rows, self.nk = rows, nk
        self.replay_rows = min(rows, 512)

        def ints(*shape):
            return torch.empty(shape, dtype=torch.int32, device=device)

        def floats(*shape):
            return torch.empty(shape, dtype=torch.float32, device=device)

        self.boundary_ids = ints(rows, 4096)
        self.boundary_counts = ints(rows)
        self.tasks = ints(rows * 32, 2)
        self.task_count = ints(1)
        self.values = floats(rows, 8192)
        self.ids = ints(rows, 8192)
        self.counts = ints(rows)
        self.error = ints(rows)
        self.epsilon = floats(rows)
        self.metadata = None
        self.metadata_workspace = None
        self.offsets = torch.zeros(rows, dtype=torch.int32, device=device)
        self.tau = torch.full((rows,), -torch.inf, dtype=torch.float32, device=device)
        self.final_tau = floats(rows)
        self.fi_lengths = ints(rows)
        self.replay_scores = floats(self.replay_rows, nk)
        self.replay_counts = ints(self.replay_rows)
        self.replay_selected = ints(self.replay_rows, 2048)
        self.map_status = torch.zeros(rows, dtype=torch.int32, device=device)
        self.use_fused_large = True
        self.conditional_enabled = False
        self.recovery_audit = None
        self.recovery_entries = []
        self.active_recovery_entry = None
        self.last_recovery_args = None
        self.last_recovery_dispatch = None
        self.recovery_k_capacity = None
        self.recovery_scale_capacity = None
        self.recovery_logical_nk = None

    def select(self, scores, counts):
        if len(scores) > len(self.replay_selected):
            raise ValueError("recovery FI batch exceeds its selected output capacity")
        return topk_into(
            scores, self.offsets[:len(scores)], counts,
            self.replay_selected[:len(scores)],
        )

    def __call__(self, q, k, scale, weights, starts, ends, paged, prep, output):
        if not self.use_fused_large:
            raise RuntimeError(
                "This published refiner supports the frozen fused-large path"
            )
        self.post.collect_band(
            paged["inline_val"],
            paged["inline_idx"],
            paged["overflow_val"],
            paged["overflow_idx"],
            paged["page_table"],
            prep["cc"][: self.rows],
            paged["pool_status"],
            prep["status"][: self.rows],
            prep["bc"][: self.rows],
            starts,
            ends,
            self.boundary_ids,
            self.boundary_counts,
            self.tasks,
            self.task_count,
            self.values,
            self.ids,
            self.counts,
            self.error,
            self.nk,
            self.nk,
            self.epsilon,
            output,
        )
        self.post.complete_h32_fused_large(
            q,
            k,
            scale,
            weights,
            self.boundary_ids,
            self.boundary_counts,
            prep["bc"][: self.rows],
            starts,
            ends,
            self.error,
            output,
            self.values,
            self.ids,
            self.counts,
            self.fi_lengths,
            self.final_tau,
        )
        self.recover(q, k, scale, weights, starts, ends, output)
        return output

    def _replay(self, *args):
        if self.recovery_logical_nk is None:
            return self.core.replay(*args)
        return self.core.replay_capacity(*args, self.recovery_logical_nk)

    def _recover_direct(self, q, k, scale, weights, starts, ends, output, *, audit=None):
        views, outputs = [], []
        if audit is not None:
            audit[0].add_(1)
        for begin in range(0, self.rows, self.replay_rows):
            end = min(self.rows, begin + self.replay_rows)
            count = end - begin
            dense = self.replay_scores[:count]
            counts = self.replay_counts[:count]
            status = self.error[begin:end]
            call_views = (
                q[begin:end],
                k,
                scale,
                weights[begin:end],
                starts[begin:end],
                ends[begin:end],
                status,
                dense,
                counts,
                output[begin:end],
            )
            views.append(call_views)
            self._replay(*call_views[:9])
            if audit is not None:
                audit[1][begin:end].copy_(counts)
            recovered = self.select(dense, counts)
            self.post.publish_replay(recovered, call_views[9], status)
            outputs.append(recovered)
        return tuple(views), tuple(outputs)

    def _binding(self, args):
        tensors = (
            *args,
            self.error,
            self.replay_scores,
            self.replay_counts,
            self.replay_selected,
            self.offsets,
        )
        if self.recovery_logical_nk is not None:
            tensors += (self.recovery_logical_nk,)
        audit = self.recovery_audit
        audit_signature = (
            None if audit is None else tuple(tensor_binding(t) for t in audit)
        )
        return (
            tuple(tensor_binding(t) for t in tensors),
            self.rows,
            (self.replay_scores.shape[1] if self.recovery_logical_nk is not None else self.nk),
            self.replay_rows,
            id(self.core),
            id(self.post),
            audit_signature,
        )

    def recover(self, q, k, scale, weights, starts, ends, output):
        if self.recovery_logical_nk is not None:
            k, scale = self.recovery_k_capacity, self.recovery_scale_capacity
        args = (q, k, scale, weights, starts, ends, output)
        self.last_recovery_args = args
        entry = self.active_recovery_entry
        capturing = torch.cuda.is_current_stream_capturing()
        if (
            self.conditional_enabled
            and capturing
            and entry is not None
            and self._binding(args) == entry.signature
        ):
            self.conditional.attach_if_any(
                self.error, False, entry.graph.raw_cuda_graph()
            )
            self.last_recovery_dispatch = "conditional"
        else:
            if not self.conditional_enabled:
                self.last_recovery_dispatch = "direct_disabled"
            elif not capturing:
                self.last_recovery_dispatch = "direct_eager"
            elif entry is None:
                self.last_recovery_dispatch = "direct_unprepared"
            else:
                self.last_recovery_dispatch = "direct_signature_mismatch"
            # Publish consumes the shared FI output before the next chunk.
            self._recover_direct(*args)

    def prepare_recovery_graph(self):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare the cold child outside outer capture")
        if self.last_recovery_args is None:
            raise RuntimeError(
                "Warm the current complete call before preparing recovery"
            )
        args = self.last_recovery_args
        signature = self._binding(args)
        if (
            self.active_recovery_entry is not None
            and self.active_recovery_entry.signature == signature
        ):
            return self.active_recovery_entry
        if self.recovery_logical_nk is not None:
            # Keep one current compatible child. Older parents already lease
            # their entry snapshots and preserve any still-needed storage.
            self.active_recovery_entry = None
            self.recovery_entries = []
        q = args[0]
        audit = self.recovery_audit
        with capture_child(q.device) as body:
            views, outputs = self._recover_direct(*args, audit=audit)
        entry = RecoveryEntry(
            signature,
            body,
            (*args, self.error, self.replay_scores, self.replay_counts,
             self.replay_selected, self.offsets, self.recovery_logical_nk),
            views,
            outputs,
            (self.core, self.post, self.conditional),
            audit,
        )
        self.recovery_entries.append(entry)
        self.active_recovery_entry = entry
        return entry

    def retain_on(self, outer_graph, *caller_owners):
        # CUDAGraph is the Python owner passed back to the caller. Keeping the
        # lease there makes child tensors survive even if a refiner is dropped
        # or prepares a different binding for a later outer graph.
        workspaces = tuple(
            value for value in self.__dict__.values() if isinstance(value, torch.Tensor)
        )
        outer_graph._dosa_recovery_lease = (
            self,
            tuple(self.recovery_entries),
            workspaces,
            caller_owners,
        )
        return outer_graph
