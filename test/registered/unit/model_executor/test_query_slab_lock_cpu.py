"""The actual query dispatcher must serialize its state reset before CUDA work."""
import ast
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest


SOURCE = (Path(__file__).resolve().parents[4] / "python/sglang/srt/layers/attention"
          "/dsa/dosa_runtime/slabbed_few.py")


def overlapping_calls(source):
    # Extract the production closure without constructing CUDA-owned buffers.
    tree = ast.parse(source)
    installer = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == "install_query_slabs")
    wrapper = next(n for n in installer.body if isinstance(n, ast.FunctionDef)
                   and n.name == "run")
    contended, large_started = threading.Event(), threading.Event()
    raw_lock = threading.RLock()

    class ObservedLock:
        def __enter__(self):
            if not raw_lock.acquire(blocking=False):
                contended.set()
                raw_lock.acquire()
            return self

        def __exit__(self, *_):
            raw_lock.release()

    backend = SimpleNamespace(lock=ObservedLock(), max_q=16, slab_last_call=None)
    completed, errors = [], []

    def large(b, original, q, *args, **kwargs):
        with b.lock:
            b.slab_last_call = {"queued_complete": False}
            large_started.set()
            if not contended.wait(timeout=3):
                raise AssertionError("Small call did not reach lock contention")
            b.slab_last_call["queued_complete"] = True
            completed.append("large")
            return True

    def original(q, *args, **kwargs):
        with backend.lock:
            completed.append("small")
            return True

    scope = dict(backend=backend, original=original, run_slabbed_paged=large)
    module = ast.Module(body=[wrapper], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)

    def invoke(rows):
        try:
            scope["run"](range(rows))
        except BaseException as error:
            errors.append(error)

    a, b = threading.Thread(target=invoke, args=(32,)), threading.Thread(target=invoke, args=(4,))
    a.start()
    if not large_started.wait(timeout=3):
        raise AssertionError("Large call did not start")
    b.start()
    a.join(timeout=4)
    b.join(timeout=4)
    if a.is_alive() or b.is_alive():
        raise AssertionError("Query dispatcher deadlocked")
    return completed, errors, backend.slab_last_call


class QuerySlabLockTest(unittest.TestCase):
    def test_small_query_waits_before_resetting_large_query_state(self):
        completed, errors, state = overlapping_calls(SOURCE.read_text())
        self.assertEqual(errors, [])
        self.assertEqual(completed, ["large", "small"])
        self.assertIsNone(state)


if __name__ == "__main__":
    unittest.main()
