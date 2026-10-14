"""One static/live certificate writer and its packed H22 record owner.

Native export names retain fp16 for ABI compatibility; the opaque record payload
is RU-BF16. Every execution rebuilds the record. Slow/fast Q4 groups are disjoint,
and each captured parent retains its original child operands.
"""

import torch

from .schedule import StaticSchedule, LiveSchedule


class H22Record:
    def __init__(self, native, max_queries, device):
        if max_queries % 4:
            raise ValueError('Packed H22 record requires Q4 groups')
        required = ('emit_trusted_candidates', 'prepare_fp16_h22', 'emit_fp16_candidates')
        if not all(callable(getattr(native, name, None)) for name in required):
            raise RuntimeError('Qualified audit module must supply prepare, slow and fast H22')
        self.max_queries = int(max_queries)
        self.paid_budget = getattr(native, "packed_high_budget", lambda: 20)()
        record_words = getattr(native, "packed_record_words", lambda: 64)()
        self.record_backing = torch.empty((max_queries // 4, record_words), device=device, dtype=torch.int32)
        self.fast_backing = torch.empty(max_queries // 4, device=device, dtype=torch.int32)
        self.slow_backing = torch.empty_like(self.fast_backing)
        self.record = self.record_backing
        self.fast_budget = self.fast_backing
        self.slow_budget = self.slow_backing
        self.bound_rows = max_queries

    def bind(self, rows):
        # Backings never grow. Original backend waits its last-done event before
        # shape rebind/graph destruction; all graphs retain these owned storages.
        if rows % 4 or not 0 < rows <= self.max_queries:
            raise ValueError('Record query extent exceeds the reserved Q4 capacity')
        self.record = self.record_backing[: rows // 4]
        self.fast_budget = self.fast_backing[: rows // 4]
        self.slow_budget = self.slow_backing[: rows // 4]
        self.bound_rows = rows

    def workspace_invoice(self):
        return dict(
            record_encoding='RU_BF16',
            record_bytes=self.record_backing.numel() * self.record_backing.element_size(),
            budget_bytes=sum(x.numel() * x.element_size()
                             for x in (self.fast_backing, self.slow_backing)),
            record_shape=list(self.record.shape), bound_rows=self.bound_rows,
            per_call_prepare=True, original_budget_unchanged=True,
            slow_fast_Q4_mutually_exclusive=True,
        )


class CertificateWriter:
    def __init__(self, packed, modules, rows, nk, device, seed, *, owner=None):
        self.packed, self.m, self.owner = packed, modules, owner
        self.live_nk = None if owner is None else owner.rt.live_nk
        if self.live_nk is None:
            self.work = StaticSchedule(rows, nk, device, paid=14, seed=seed)
            self.work_h18 = StaticSchedule(rows, nk, device, paid=16, seed=seed)
            self.emit_h16 = modules['h16'].emit_h16_indexed_trusted_grid
            self.emit_h18 = modules['h18'].emit_h18_indexed_trusted
            self.prepare_h22 = modules['audit'].prepare_fp16_h22
            self.emit_h22 = modules['audit'].emit_trusted_candidates
            self.emit_packed_h22 = modules['audit'].emit_fp16_candidates
            self.h22_conditional = None
        else:
            self.work = LiveSchedule(rows, nk, self.live_nk, paid=14, seed=seed,
                                     backing=owner.h16_base)
            self.work_h18 = LiveSchedule(rows, nk, self.live_nk, paid=16, seed=seed,
                                         backing=owner.h18_base)
            self.emit_h16 = modules['h16'].emit_h16_indexed_trusted_live_k
            self.emit_h18 = modules['h18'].emit_h18_indexed_trusted_live_k
            self.prepare_h22 = modules['audit'].prepare_fp16_h22_live_k
            self.emit_h22 = modules['audit'].emit_trusted_candidates_live_k
            self.emit_packed_h22 = modules['audit'].emit_fp16_candidates_live_k
            self.packed.bind(rows)
            self.h22_conditional = getattr(owner.rt.conditional, 'attach_if_any_not_in', None)
            if self.h22_conditional is not None and not callable(self.h22_conditional):
                raise RuntimeError('Invalid conditional H22 native binding')
        self.h22_args = self.h22_graphs = None

    def prepare_h22_graphs(self):
        if self.h22_conditional is None:
            return
        from .private_capture import capture_child

        if self.h22_args is None:
            raise RuntimeError('Warm the actual H22 dispatch before capture')
        graphs = []
        for method, args in zip((self.emit_h22, self.emit_packed_h22), self.h22_args):
            with capture_child(self.owner.device) as body:
                method(*args)
            body._dosa_h22_inputs = args
            graphs.append(body)
        self.h22_graphs = tuple(graphs)

    def emit_trusted_candidates(self, paid, x, streams, pool, table, head,
                                status, error, seed, words):
        args = (paid, x, streams, pool, table, head, status, error, seed, words)
        live = self.live_nk is not None
        extent = (self.live_nk,) if live else ()
        if paid in (14, 16):
            work = self.work if paid == 14 else self.work_h18
            if not live:
                assert seed == work.seed
            work.build(x[13])
            grid = (work.grid,) if live or paid == 14 else ()
            method = self.emit_h16 if paid == 14 else self.emit_h18
            return method(*args, work.ids, work.inverse, work.config, *grid, *extent)
        if paid != self.packed.paid_budget:
            raise ValueError('Unsupported certificate family')
        p = self.packed
        if not live:
            p.bind(x[0].shape[0])
        preparation_extent = (seed, self.live_nk) if live else ()
        self.prepare_h22(x, error, p.record, p.fast_budget, p.slow_budget,
                         *preparation_extent)
        slow, fast = list(x), list(x)
        slow[13], fast[13] = p.slow_budget, p.fast_budget
        self.h22_args = (
            (paid, slow, streams, pool, table, head, status, error, seed, words, *extent),
            (paid, fast, streams, pool, table, head, status, error, seed, words, p.record, *extent),
        )
        if self.h22_graphs is not None and torch.cuda.is_current_stream_capturing():
            self.h22_conditional(p.slow_budget, 14, 16, self.h22_graphs[0].raw_cuda_graph())
            self.h22_conditional(p.fast_budget, 14, 14, self.h22_graphs[1].raw_cuda_graph())
        else:
            self.emit_h22(*self.h22_args[0])
            self.emit_packed_h22(*self.h22_args[1])
