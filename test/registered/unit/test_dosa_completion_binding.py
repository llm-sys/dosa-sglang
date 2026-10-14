"""Check the shared H32 reader ABI without importing CUDA dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'python/sglang').is_dir())
RUNTIME = ROOT / 'python/sglang/srt/layers/attention/dsa/dosa_runtime'


def definition(file, name, owner=None):
    nodes = ast.parse((RUNTIME / file).read_text()).body
    if owner:
        nodes = next(node.body for node in nodes if getattr(node, 'name', None) == owner)
    node = next(node for node in nodes if getattr(node, 'name', None) == name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), file, 'exec'), namespace)
    return namespace[name]


class Storage:
    def __init__(self, name):
        self.name = name

    def __getitem__(self, rows):
        return self.name, rows.start, rows.stop


class Reader:
    def __init__(self, calls):
        self.calls = calls

    def __getattr__(self, name):
        def invoke(*args):
            self.calls.append((name, args))
        return invoke


class CompletionBindingTests(unittest.TestCase):
    def setUp(self):
        self.complete = definition('few_core.py', 'complete_h32', 'FusedRuntime')
        self.adapter = definition('record5.py', 'CompletedRecords')
        self.calls = []
        fields = ('w safe_tau id_pool solo_counts solo_tasks solo_n common_counts '
                  'common_tasks common_n values ids counts starts ends error seed_tau')
        self.runtime = SimpleNamespace(**{name: object() for name in fields.split()})
        self.runtime.rows = 16
        self.runtime.nk = 229376
        self.runtime.seed = 12288
        self.runtime.sms = 148
        self.runtime.pool_pages = 4096
        self.runtime.page_table = Storage('pages')
        self.runtime.live_nk = None
        self.runtime.record_output = None
        self.operands = tuple(object() for _ in range(3))
        self.native = Reader(self.calls)
        self.runtime.exact = self.native

    def check_readers(self, packed, live):
        readers = [(name, args) for name, args in self.calls if 'exact_paged_' in name]
        self.assertEqual(len(readers), 2)
        r = self.runtime
        for index, (name, args) in enumerate(readers):
            stream = 'solo' if index == 0 else 'common'
            suffix = '_live_nk' if live else ''
            self.assertEqual(name, 'exact_paged_' + stream + suffix)
            expected = (
                *self.operands, r.w, r.safe_tau, r.id_pool,
                getattr(r, stream + '_counts'), getattr(r, stream + '_tasks'),
                getattr(r, stream + '_n'), ('low', None, 16) if packed else r.values,
                r.ids, r.counts, 0, r.sms,
                ('pages', None, 16) if index == 0 else ('pages', 16, None),
                r.pool_pages, r.starts, r.ends, r.error,
            )
            if live:
                expected += (r.live_nk, r.seed)
            self.assertEqual(args, expected)

    def test_unpacked_static_readers_share_operands(self):
        self.runtime.exact = self.native
        self.complete(self.runtime, *self.operands)
        self.check_readers(packed=False, live=False)

    def test_record_seed_precedes_both_static_and_live_readers(self):
        for live in (False, True):
            with self.subTest(live=live):
                self.calls.clear()
                r = self.runtime
                r.live_nk = object() if live else None
                r.live_exact = self.native
                r.record_output = self.adapter(r, self.native, r.rows, None, Storage('low'))
                r.record_output.direct_seed = self.native
                self.complete(r, *self.operands)
                expected = 'validate_seed5_live_nk' if live else 'validate_seed5'
                self.assertEqual(self.calls[0][0], expected)
                self.check_readers(packed=True, live=live)

    def test_failed_solo_binding_never_dispatches_common(self):
        r = self.runtime
        r.record_output = self.adapter(r, self.native, r.rows, None, Storage('low'))
        r.record_output.direct_seed = SimpleNamespace(validate_seed5=lambda *args: 1 / 0)
        with self.assertRaises(ZeroDivisionError):
            self.complete(r, *self.operands)
        self.assertEqual(self.calls, [])

    def test_indirect_record_seed_is_packed_before_readers(self):
        r = self.runtime
        r.record_output = self.adapter(r, self.native, r.rows, None, Storage('low'))
        self.complete(r, *self.operands)
        self.assertEqual(self.calls[0][0], 'pack_seed')
        self.check_readers(packed=True, live=False)

    def test_live_record_requires_direct_seed_before_any_reader(self):
        r = self.runtime
        r.live_nk = object()
        r.live_exact = self.native
        r.record_output = self.adapter(r, self.native, r.rows, None, Storage('low'))
        with self.assertRaisesRegex(RuntimeError, 'direct record5 seed'):
            self.complete(r, *self.operands)
        self.assertEqual(self.calls, [])

    def test_publication_uses_explicit_operands_and_owned_records(self):
        r = self.runtime
        owner = self.adapter(r, self.native, r.rows, None, Storage('low'))
        fields = ('epsilon status diag boundary_ids boundary_counts temporary '
                  'temporary_ids temporary_counts done fi_lengths tau')
        for name in fields.split():
            setattr(owner, name, Storage(name))
        owner.selector = owner.post = self.native
        r.pool_status, r.ordered_output, r.order = object(), object(), object()
        r.output = SimpleNamespace(index_copy_=self.native.index_copy_)
        pairs = [tuple(object() for _ in range(3)), self.operands]
        for operands in pairs:
            owner.select_and_publish(r, *operands)
        self.assertEqual([name for name, _ in self.calls],
                         ['select_dense5', 'complete_h32_fused_large', 'index_copy_'] * 2)
        for offset, operands in zip((0, 3), pairs):
            self.assertEqual(self.calls[offset][1][0], ('low', None, r.rows))
            self.assertEqual(self.calls[offset + 1][1][:3], operands)
            self.assertEqual(self.calls[offset + 2][1],
                             (0, r.order, r.ordered_output))
        self.assertNotIn('current_inputs', vars(owner))


if __name__ == '__main__':
    unittest.main()
