"""CPU policy checks; graph retirement and CUDA synchronization stay unchanged."""
import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[3] / 'python/sglang/srt/layers/attention/dsa/dosa_runtime'


def load_class(filename, name, namespace):
    nodes = ast.parse((ROOT / filename).read_text()).body
    cls = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == name)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(ROOT), 'exec'), namespace)
    return namespace[name]


class AllocationOwnerTest(unittest.TestCase):
    def setUp(self):
        self.reclamation = []

        @contextmanager
        def capture(graph, stream, *, pool=None, reclaim_device_cache=True):
            self.reclamation.append((pool, reclaim_device_cache))
            yield

        self.capture = capture
        namespace = dict(contextmanager=contextmanager, private_capture=None,
                         torch=SimpleNamespace(cuda=SimpleNamespace(device=lambda _: nullcontext())))
        cls = load_class('capture_workspace.py', 'CaptureWorkspace', namespace)
        self.owner = cls.__new__(cls)
        self.owner.device = 'cuda:0'
        self.owner.private = SimpleNamespace(capture_preserving_host_cache=capture)
        self.owner.failed = False
        self.owner.captures = 0
        self.new_generation()

    def new_generation(self):
        self.generation = SimpleNamespace(pool=SimpleNamespace(id=object()),
                                          stream=object(), allocation_key=None, state='pending')
        self.graph = SimpleNamespace(_dosa_pool_generation=self.generation)
        self.owner.pending = self.generation

    def run_capture(self, key):
        with self.owner.capture(self.graph, self.generation.stream, allocation_key=key):
            pass

    def test_only_matching_completed_allocation_contract_reuses(self):
        first = (16384, 32768)
        self.run_capture(first)
        self.run_capture(first)
        self.run_capture((8192, 32768))
        self.run_capture((8192, 16384))
        self.assertEqual([reclaim for _, reclaim in self.reclamation], [True, False, True, True])
        self.assertEqual(self.generation.allocation_key, (8192, 16384))

    def test_no_opt_in_and_new_generation_always_reclaim(self):
        self.run_capture(None)
        self.run_capture(None)
        self.run_capture((16384, 32768))
        old_pool = self.generation.pool.id
        self.new_generation()
        self.run_capture((16384, 32768))
        self.assertEqual([reclaim for _, reclaim in self.reclamation], [True] * 4)
        self.assertIsNot(old_pool, self.generation.pool.id)

    def test_failure_does_not_publish_allocation_contract(self):
        with self.assertRaisesRegex(RuntimeError, 'capture failed'):
            with self.owner.capture(self.graph, self.generation.stream, allocation_key=(16384, 32768)):
                raise RuntimeError('capture failed')
        self.assertTrue(self.owner.failed)
        self.assertEqual(self.generation.state, 'failed')
        self.assertIsNone(self.generation.allocation_key)
        self.assertEqual(self.owner.captures, 0)
        with self.assertRaisesRegex(RuntimeError, 'Discard'):
            self.owner.begin()

    def test_wrong_owner_or_stream_cannot_reuse_pool(self):
        with self.assertRaisesRegex(RuntimeError, 'Wrong'):
            with self.owner.capture(self.graph, object(), allocation_key=(16384, 32768)):
                pass
        self.assertEqual(self.reclamation, [])

    def test_direct_capture_has_no_reusable_pool(self):
        namespace = dict(contextmanager=contextmanager,
                         capture_preserving_host_cache=self.capture)
        cls = load_class('private_capture.py', 'DirectCaptureFactory', namespace)
        for _ in range(2):
            with cls.capture(object(), object(), allocation_key=(16384, 32768)):
                pass
        self.assertEqual(self.reclamation, [(None, True), (None, True)])


if __name__ == '__main__':
    unittest.main()
