"""Capture private CUDA work while preserving the service's host cache."""

import gc
from contextlib import contextmanager
import weakref

import torch


class _PrivateCUDAGraph(torch.cuda.CUDAGraph):
    def __del__(self):
        # CPython can clear a subclass dict before calling the C++ deallocator.
        # Reset while the stream is still owned; the later C++ reset is a no-op.
        stream = getattr(self, "_dosa_capture_stream", None)
        try:
            super().__del__()  # Preserve PyTorch input-liveness tracker cleanup.
        finally:
            try:
                self.reset()
            except Exception:
                pass  # Cleanup must not mask an existing CUDA/capture failure.
            finally:
                _ = stream


def new_capture_graph(*, keep_graph=False):
    """Create a private graph whose reset precedes its owned stream's release."""
    return _PrivateCUDAGraph(keep_graph=keep_graph)


def _destroy_capture_stream(runtime, raw_stream, device):
    # A completed graph releases its Python stream lease on destruction. CUDA
    # stream destruction is asynchronous with respect to already queued work.
    # Do not replace a caller's original exception if CUDA has already failed.
    try:
        with torch.cuda.device(device):
            runtime.cudaStreamDestroy(raw_stream)
    except Exception:
        pass


def new_capture_stream(device):
    """Own an unpooled stream so graph destruction cannot clear another's BLAS cache.

    torch.cuda.Stream uses a 32-stream round-robin pool. PyTorch graph reset
    clears cuBLAS workspaces by raw capture stream, even when that graph never
    called cuBLAS. An independently owned stream prevents this aliasing.
    """
    from cuda.bindings import runtime

    device = torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    with torch.cuda.device(device):
        status, raw = runtime.cudaStreamCreateWithFlags(runtime.cudaStreamNonBlocking)
        if status != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA capture stream creation failed: {status}")
        try:
            stream = torch.cuda.ExternalStream(int(raw), device=device)
        except BaseException:
            runtime.cudaStreamDestroy(raw)
            raise
    cleanup = weakref.finalize(stream, _destroy_capture_stream, runtime, raw, device)
    cleanup.atexit = False  # Process teardown owns streams still leased by graphs.
    return stream


@contextmanager
def capture_preserving_host_cache(graph, stream, *, pool=None, reclaim_device_cache=True):
    """Retain the normal capture synchronization, stream and capture mode.

    Reclaim device cache when a private pool needs its initial allocation or a
    changed allocation shape. An explicit retired-pool owner can retain matching
    blocks. Do not empty the process-wide host cache while other graphs exist.
    Graph construction, keep_graph, optional pool and input/output ownership stay
    at the caller. Both direct and reusable-pool captures use this transaction.
    """
    graph._dosa_capture_stream = stream
    torch.cuda.synchronize(stream.device)
    # Match the installed context except for its host-cache clearing operation.
    config = getattr(torch.compiler, "config", None)
    if getattr(config, "force_cudagraph_gc", False):
        gc.collect()
    if reclaim_device_cache:
        torch.cuda.empty_cache()
    with torch.cuda.stream(stream):
        options = {} if pool is None else {"pool": pool}
        graph.capture_begin(capture_error_mode="global", **options)
        try:
            yield
        finally:
            graph.capture_end()


@contextmanager
def capture_child(device, *, keep_graph=True):
    """Capture a child with its own stream, retaining the normal transaction."""
    stream = new_capture_stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = new_capture_graph(keep_graph=keep_graph)
    with capture_preserving_host_cache(graph, stream):
        yield graph


class DirectCaptureFactory:
    """Instance-owned outer capture policy without a reusable allocation pool."""

    @contextmanager
    def session(self):
        yield self

    @staticmethod
    def new_stream(device):
        return new_capture_stream(device)

    @staticmethod
    def new_graph(**kwargs):
        return new_capture_graph(**kwargs)

    @staticmethod
    def capture(graph, stream, *, allocation_key=None):
        # No retained pool is available to satisfy an allocation-reuse contract.
        return capture_preserving_host_cache(graph, stream)
