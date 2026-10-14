"""Logical-call ownership and original row order without a CUDA dependency."""
import ast
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'python/sglang').is_dir())
RUNTIME = ROOT / 'python/sglang/srt/layers/attention/dsa/dosa_runtime'
PACKAGE = '_dosa_query_dispatch_test'


def production_node(file, name, namespace):
    tree = ast.parse((RUNTIME / file).read_text())
    node = next(n for n in tree.body if getattr(n, 'name', None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(RUNTIME / file), 'exec'), namespace)
    return namespace[name]


class View:
    def __init__(self, values, indices=None):
        self.values = values
        self.indices = list(range(len(values))) if indices is None else indices

    def __getitem__(self, index):
        return View(self.values, self.indices[index])

    def copy_(self, other):
        for target, source in zip(self.indices, other.indices):
            self.values[target] = other.values[source]

    def index_copy_(self, dim, order, source):
        assert dim == 0
        for target, index in zip(order, source.indices):
            self.values[self.indices[target]] = source.values[index]


class QueryDispatchTests(unittest.TestCase):
    def setUp(self):
        ns = dict(threading=threading, torch=NS(Tensor=View), __package__=PACKAGE)
        self.backend_type = production_node('paged_backend.py', 'PagedBackend', ns)
        self.backend = self.backend_type()
        self.backend._initialize_slot()
        self.backend.max_q = 16
        self.options = dict(nk=131072, common_end=98305,
                            last_visible=131072, hot_key=(0, 'request'))
        self.module = ModuleType(PACKAGE + '.slabbed_few')
        self.patcher = patch.dict(sys.modules, {self.module.__name__: self.module})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def call(self, rows, **options):
        return self.backend.run_paged(range(rows), *([None] * 7),
                                      **(self.options | options))

    def test_direct_and_slab_routes_keep_the_same_block_callable(self):
        calls = []

        def block(q, *inputs, **options):
            calls.append(('block', len(q), options))
            return True

        def slab(owner, original, q, *inputs, **options):
            self.assertIs(owner, self.backend)
            self.assertIs(original, block)
            calls.append(('slab', len(q), options))
            return True

        self.backend._run_paged_block = block
        self.module.run_slabbed_paged = slab
        self.assertTrue(self.call(8))
        self.backend.logical_query_capacity = 32
        self.backend.lock = threading.RLock()
        self.backend.slab_last_call = {'previous': True}
        self.assertTrue(self.call(8, causal_query_start=98304))
        self.assertIsNone(self.backend.slab_last_call)
        self.assertTrue(self.call(32, causal_query_start=98304))
        self.assertEqual([c[:2] for c in calls], [('block', 8), ('block', 8), ('slab', 32)])
        self.assertNotIn('causal_query_start', calls[1][2])
        self.assertEqual(calls[2][2]['causal_query_start'], 98304)

    def test_small_call_cannot_reset_inflight_large_call(self):
        raw_lock = threading.RLock()
        contended, entered = threading.Event(), threading.Event()
        completed, errors = [], []

        class ObservedLock:
            def __enter__(self):
                if not raw_lock.acquire(blocking=False):
                    contended.set()
                    raw_lock.acquire()

            def __exit__(self, *_):
                raw_lock.release()

        def block(*args, **kwargs):
            completed.append('small')
            return True

        def slab(owner, original, *args, **kwargs):
            with owner.lock:
                owner.slab_last_call = {'queued_complete': False}
                entered.set()
                self.assertTrue(contended.wait(3))
                self.assertIsNotNone(owner.slab_last_call)
                owner.slab_last_call['queued_complete'] = True
                completed.append('large')
                return True

        def invoke(rows):
            try:
                self.call(rows)
            except BaseException as error:
                errors.append(error)

        self.backend.logical_query_capacity = 32
        self.backend.lock = ObservedLock()
        self.backend._run_paged_block = block
        self.module.run_slabbed_paged = slab
        large = threading.Thread(target=invoke, args=(32,))
        small = threading.Thread(target=invoke, args=(8,))
        large.start()
        self.assertTrue(entered.wait(3))
        small.start()
        large.join(4)
        small.join(4)
        self.assertFalse(large.is_alive() or small.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(completed, ['large', 'small'])
        self.assertIsNone(self.backend.slab_last_call)

    def test_sorted_diagnostics_restore_rows_before_padding_is_removed(self):
        copy = production_node('slabbed_few.py', 'copy_slab_results', {})
        ledger = {name: View([-1] * 8) for name in
                  ('counts_original', 'error_before_completion', 'replay_work')}
        owner = NS(slab_ledger=ledger, slab_original_scratch=View([0] * 4),
                   rt=NS(order=[2, 0, 3, 1], counts_original=View([1, 2, 3, 4]),
                         error_before_completion=View([10, 20, 30, 40]),
                         replay_work=View([100, 200, 300, 400])))
        copy(owner, 2, 5, 4)
        self.assertEqual(ledger['counts_original'].values, [-1, -1, 1, 2, 3, -1, -1, -1])
        self.assertEqual(ledger['error_before_completion'].values,
                         [-1, -1, 20, 40, 10, -1, -1, -1])
        self.assertEqual(ledger['replay_work'].values,
                         [-1, -1, 200, 400, 100, -1, -1, -1])


class QueryStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            '_dosa_storage_geometry', RUNTIME / 'query_slots.py')
        cls.geometry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.geometry)
        cls.slabs = production_node('query_slabs.py', 'query_slabs', {})

    def test_h64_batches_fit_the_existing_id_budget_at_one_million_keys(self):
        g = self.geometry
        for logical, expected in ((4096, 512), (8192, 1024),
                                  (16384, 2048), (32768, 2048)):
            rows, pages = g.service_query_storage(logical, 1048576, 64)
            self.assertEqual(rows, expected)
            self.assertEqual(pages, 256 * min(logical, 16384))
            self.assertLessEqual(g.causal_id_page_bound(rows, 1048576), pages)
            self.assertGreater(g.causal_id_page_bound(rows * 2, 1048576), pages)
            self.assertEqual(g.validate_query_capacity(rows, minimum=8), rows)
        self.assertFalse(g.service_rows_eligible(2048, 2048))
        self.assertTrue(g.service_rows_eligible(2048, 16384))

    def test_service_candidate_storage_fits_causal_work_and_model_budget(self):
        # Every ID page owns 128 I32 IDs and two I32[2] task entries.
        # Record5 owns one byte plus one I32 word per candidate. These are
        # resident before model KV sizing, not lazy request allocations.
        page_bytes, record_bytes = (128 + 2 + 2) * 4, 5
        self.assertEqual(self.geometry.FEW_RECORD_CAPACITY, 114688)
        for heads in (32, 64):
            for logical in (4096, 4104, 8192, 12344, 16384, 32760, 32768):
                for keys in (196608, 262144, 524288, 786432, 1048576):
                    with self.subTest(heads=heads, logical=logical, keys=keys):
                        rows, pages = self.geometry.service_query_storage(logical, keys, heads)
                        self.assertIsNotNone(pages)
                        self.assertEqual(rows % 8, 0)
                        self.assertLessEqual(rows, min(logical, 16384))
                        self.assertGreaterEqual(rows, 8)
                        self.assertLessEqual(self.geometry.causal_id_page_bound(rows, keys), pages)
                        original_rows = min(logical, 16384)
                        original_bytes = original_rows * (65536 * record_bytes + 256 * page_bytes)
                        self.assertLessEqual(rows * self.geometry.FEW_RECORD_CAPACITY * record_bytes
                                             + pages * page_bytes,
                                             original_bytes)

    def test_active_id_views_fit_reserved_backing_at_every_smaller_shape(self):
        g = self.geometry
        for capacity in (512, 2048, 8192, 16384):
            reserved = g.few_id_pool_pages(capacity, 1048576)
            for rows in (8, capacity // 2, capacity):
                for keys in (196608, 524288, 1048576):
                    with self.subTest(capacity=capacity, rows=rows, keys=keys):
                        active = g.few_id_pool_pages(rows, keys)
                        self.assertLessEqual(active, reserved)
                        self.assertGreaterEqual(active, 256 * rows)
                        self.assertGreaterEqual(active, g.causal_id_page_bound(rows, keys))

    def test_causal_union_bound_includes_disjoint_tail_copies(self):
        # Adversarially mix distant endpoints after query sorting. A key in
        # all four intervals occupies one ID; tail keys can occupy three solos.
        ends = (1024, 1009, 1023, 1010)
        seed = 8
        ids = sum(1 if all(k < end for end in ends)
                  else sum(k < end for end in ends)
                  for k in range(seed, max(ends)))
        rows = 16
        pages = (ids + 127) // 128 + 4
        self.assertLessEqual(pages * (rows // 4),
                             self.geometry.causal_id_page_bound(rows, 1024, seed))

    def test_bounded_slabs_cover_queries_and_original_carry_window(self):
        # Short tails can split the final carry window across multiple slabs.
        # Cover every tail size around Q8 alignment and that window boundary.
        for capacity in (8, 512, 2048, 8192):
            for rows in (1, 7, 8, capacity - 1, capacity, capacity + 1,
                         capacity + 7, capacity + 1535, capacity + 1536,
                         16384, 32761, 32768):
                with self.subTest(capacity=capacity, rows=rows):
                    slabs = list(type(self).slabs(rows, capacity, 1536))
                    cursor, carry = 0, []
                    for begin, end, physical, overlap in slabs:
                        self.assertEqual(begin, cursor)
                        self.assertGreater(end, begin)
                        self.assertEqual(physical % 8, 0)
                        self.assertLessEqual(end - begin, physical)
                        self.assertLess(physical - (end - begin), 8)
                        self.assertLessEqual(physical, capacity)
                        carry.extend(range(end - overlap, end))
                        cursor = end
                    self.assertEqual(cursor, rows)
                    self.assertEqual(carry, list(range(max(0, rows - 1536), rows)))


if __name__ == '__main__':
    unittest.main()
