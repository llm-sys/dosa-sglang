"""Allocator reuse for a serial outer capture; child graph factories stay private."""
from contextlib import contextmanager
import threading
import weakref

import torch
from . import private_capture

_INSTALL_LOCK = threading.Lock()


class CaptureGeneration:
    def __init__(self, device, private):
        with torch.cuda.device(device):
            self.pool = torch.cuda.MemPool()
            self.stream = private.new_capture_stream(device)
            # MemPool owns only the device allocator's reference. A captured
            # empty graph also keeps the host allocator's same-ID pool alive.
            # Its separate stream never owns the real graph's BLAS workspace.
            self.anchor_stream = private.new_capture_stream(device)
            self.anchor = private.new_capture_graph(keep_graph=True)
            self.anchor._dosa_capture_stream = self.anchor_stream
            self.anchor._dosa_pool_lease = self.pool
            torch.cuda.synchronize(device)
            with torch.cuda.stream(self.anchor_stream):
                self.anchor.capture_begin(pool=self.pool.id, capture_error_mode="global")
                self.anchor.capture_end()
        self.state = "retired"
        self.serial = 0
        self.allocation_key = None

    def reset_completed(self, serial):
        # A late destructor of an already-reset graph must not retire a newer
        # capture that uses the same generation's stream and cached storage.
        if serial == self.serial:
            self.state = "retired"


class CaptureWorkspace:
    """Reuse retired graph allocations; retained old graphs get isolated generations.

    Objects/streams/pools are created lazily inside the first actual capture.
    No tensor or numerical input is fabricated. Reuse requires a completed
    native reset; merely dropping a weak reference is not sufficient evidence.
    """

    def __init__(self, backend, private=private_capture):
        self.device, self.private = backend.device, private
        self.current = None
        self.pending = None
        self.generations = []
        self.failed = False
        self.captures = 0

        base = private._PrivateCUDAGraph

        class PoolGraph(base):
            def reset(graph):
                super(PoolGraph, graph).reset()
                generation = getattr(graph, '_dosa_pool_generation', None)
                if generation is not None:
                    generation.reset_completed(graph._dosa_pool_serial)

        self.graph_class = PoolGraph

    @contextmanager
    def session(self):
        self.begin()
        try:
            yield self
        finally:
            self.pending = None

    def new_stream(self, device):
        if torch.device(device) != self.device:
            raise RuntimeError("Capture workspace device changed")
        return self.stream

    def begin(self):
        if self.failed:
            raise RuntimeError("Discard this workspace after capture failure")
        self.pending = None

    def ensure(self):
        if self.failed:
            raise RuntimeError("Discard this workspace after capture failure")
        if self.pending is None:
            if self.current is None or self.current.state != "retired":
                self.current = CaptureGeneration(self.device, self.private)
                self.generations.append(weakref.ref(self.current))
            self.pending = self.current
        return self.pending

    @property
    def stream(self):
        return self.ensure().stream

    def new_graph(self, **kwargs):
        generation = self.ensure()
        if generation.state != "retired":
            raise RuntimeError("A previous native graph has not retired")
        generation.serial += 1
        graph = self.graph_class(**kwargs)
        # The private graph finalizer resets before releasing this strong pool and
        # stream lease. Old captured graphs remain valid in their own generation.
        graph._dosa_pool_generation = generation
        graph._dosa_pool_serial = generation.serial
        generation.state = "pending"
        return graph

    @contextmanager
    def capture(self, graph, stream, *, allocation_key=None):
        generation = graph._dosa_pool_generation
        if stream is not generation.stream or generation is not self.pending:
            raise RuntimeError("Wrong capture generation/stream")
        with torch.cuda.device(self.device):
            try:
                with self.private.capture_preserving_host_cache(
                    graph, stream, pool=generation.pool.id,
                    reclaim_device_cache=(allocation_key is None
                                          or generation.allocation_key != allocation_key),
                ):
                    yield
            except BaseException:
                self.failed = True
                generation.state = "failed"
                raise
            generation.state = "captured"
            generation.allocation_key = allocation_key
            self.captures += 1

    def report(self):
        records = []
        for reference in self.generations:
            generation = reference()
            if generation is None:
                continue
            segments = generation.pool.snapshot()
            records.append({
                "pool_id": list(generation.pool.id),
                "stream": generation.stream.cuda_stream,
                "anchor_stream": generation.anchor_stream.cuda_stream,
                "host_pool_graph_anchor": True,
                "state": generation.state, "serial": generation.serial,
                "allocation_key": generation.allocation_key,
                "pool_reserved_bytes": sum(s['total_size'] for s in segments),
                "pool_allocated_bytes": sum(s['allocated_size'] for s in segments),
                "segment_addresses": [s['address'] for s in segments],
            })
        return {"captures": self.captures, "failed": self.failed,
                "generations_created": len(self.generations), "live_generations": records}


def install_outer_capture_workspace(backend):
    """Give this fresh backend its own pool, streams and capture factory."""
    if torch.cuda.memory.get_allocator_backend() != "native":
        raise RuntimeError("Capture workspace requires the native CUDA allocator")
    with _INSTALL_LOCK:
        if backend.hot_graph is not None or backend.last_done is not None:
            raise RuntimeError("Install only on a fresh backend")
        if hasattr(backend, "_outer_capture_workspace"):
            raise RuntimeError("Capture workspace already installed")
        owner = CaptureWorkspace(backend)
        backend.capture_factory = owner
        backend._outer_capture_workspace = owner
        return owner
