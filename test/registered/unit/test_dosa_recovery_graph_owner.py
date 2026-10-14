"""A recovery graph becomes visible only with its complete dispatch and leases."""
import ast
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, FrozenInstanceError
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
import weakref

ROOT = next(path for path in Path(__file__).resolve().parents if (path / 'python/sglang').is_dir())
SOURCE = ROOT / 'python/sglang/srt/layers/attention/dsa/dosa_runtime/few_core.py'


class Graph:
    def raw_cuda_graph(self):
        return self


class RecoveryGraphOwnerTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        entry = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == 'RecoveryGraphs')
        runtime = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                       and node.name == 'FusedRuntime')
        prepare = next(node for node in runtime.body if isinstance(node, ast.FunctionDef)
                       and node.name == 'prepare_cold_graph')
        self.failure = None
        self.graphs, self.log = [], []
        self.before_create = lambda: None

        def graph(**kwargs):
            self.before_create()
            role = 'body' if kwargs.get('keep_graph') else 'dispatch'
            if self.failure == role + '_create':
                raise RuntimeError(self.failure)
            value = Graph()
            value.role = role
            self.graphs.append(weakref.ref(value))
            return value

        @contextmanager
        def capture(value, stream):
            self.log.append(('begin', value.role))
            yield
            if self.failure == value.role + '_end':
                raise RuntimeError(self.failure)
            self.log.append(('end', value.role))

        namespace = dict(__name__=__name__, dataclass=dataclass,
            torch=SimpleNamespace(cuda=SimpleNamespace(device=lambda dev: nullcontext(),
                  current_stream=lambda dev: 'caller-stream')),
            new_capture_stream=lambda dev: SimpleNamespace(wait_stream=lambda other: None),
            new_capture_graph=graph, capture_preserving_host_cache=capture)
        code = ast.Module(body=[entry, prepare], type_ignores=[])
        exec(compile(ast.fix_missing_locations(code), str(SOURCE), 'exec'), namespace)
        self.Entry = namespace['RecoveryGraphs']
        self.owner = SimpleNamespace(recovery_logical_nk=object(), cold_entry=None,
            retained_cold_entries=[], cold_args=(object(), object(), object()),
            call_signature=('new-shape',), error=SimpleNamespace(device='cuda:0'),
            cold_runs=SimpleNamespace(zero_=lambda: self.log.append(('reset-count',))),
            conditional=SimpleNamespace(attach_if_any=lambda *args: self.log.append(('attach',))),
            recovery_child_builds=0)
        for name in ('w', 'starts', 'ends', 'order', 'output', 'replay_scores',
                     'replay_counts', 'replay_work', 'leaf_selectors'):
            setattr(self.owner, name, object())
        self.outputs = [object(), object()]
        self.owner.recovery_body = lambda: self.outputs
        self.owner.prepare = MethodType(namespace['prepare_cold_graph'], self.owner)

    def previous_entry(self, signature=('old-shape',)):
        body, dispatch = Graph(), Graph()
        dispatch._dosa_recovery_lease = (body, (), ())
        return self.Entry(dispatch, body, signature, (), ())

    def test_success_publishes_complete_frozen_pair(self):
        self.owner.prepare()
        entry = self.owner.cold_entry
        self.assertEqual(entry.signature, self.owner.call_signature)
        self.assertIs(entry.inputs, self.owner.cold_args)
        self.assertEqual(entry.outputs, tuple(self.outputs))
        self.assertIs(entry.dispatch._dosa_recovery_lease[0], entry.body)
        self.assertIs(entry.body._dosa_recovery_inputs[0], entry.inputs)
        self.assertEqual(self.owner.recovery_child_builds, 1)
        with self.assertRaises(FrozenInstanceError):
            entry.signature = ('mutated',)

    def test_matching_capacity_entry_skips_all_capture_work(self):
        previous = self.previous_entry(self.owner.call_signature)
        self.owner.cold_entry = previous
        self.owner.prepare()
        self.assertIs(self.owner.cold_entry, previous)
        self.assertEqual(self.log, [])
        self.assertEqual(self.graphs, [])

    def test_capacity_releases_unretained_old_pair_before_build(self):
        previous = self.previous_entry()
        reference = weakref.ref(previous)
        self.owner.cold_entry = previous
        del previous
        self.before_create = lambda: self.assertIsNone(reference())
        self.owner.prepare()

    def test_failed_capacity_build_never_publishes_partial_pair(self):
        for failure in ('body_create', 'body_end', 'dispatch_create', 'dispatch_end'):
            with self.subTest(failure=failure):
                self.failure = failure
                self.owner.cold_entry = self.previous_entry()
                with self.assertRaisesRegex(RuntimeError, failure):
                    self.owner.prepare()
                self.assertIsNone(self.owner.cold_entry)
                self.assertEqual(self.owner.recovery_child_builds, 0)
                self.assertEqual(self.owner.retained_cold_entries, [])

    def test_legacy_failure_preserves_previous_entry_without_duplicate_retention(self):
        self.owner.recovery_logical_nk = None
        previous = self.previous_entry()
        self.owner.cold_entry = previous
        self.failure = 'dispatch_end'
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, self.failure):
                self.owner.prepare()
            self.assertIs(self.owner.cold_entry, previous)
            self.assertEqual(self.owner.retained_cold_entries, [])
        self.failure = None
        self.owner.prepare()
        self.assertEqual(self.owner.retained_cold_entries, [previous])
        self.assertIsNot(self.owner.cold_entry, previous)

    def test_retained_parent_keeps_old_graphs_and_operands(self):
        self.owner.prepare()
        parent = SimpleNamespace(lease=self.owner.cold_entry)
        old_body = parent.lease.body
        self.owner.call_signature = ('another-shape',)
        self.owner.prepare()
        self.assertIs(parent.lease.body, old_body)
        self.assertIs(parent.lease.inputs, self.owner.cold_args)
        self.assertIsNot(self.owner.cold_entry, parent.lease)


if __name__ == '__main__':
    unittest.main(verbosity=2)
