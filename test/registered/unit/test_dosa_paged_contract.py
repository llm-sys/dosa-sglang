"""Shared public tensor contract and unchanged exact-recovery FI call ABI."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'python/sglang').is_dir())
RUNTIME = ROOT / 'python/sglang/srt/layers/attention/dsa/dosa_runtime'
TORCH = NS(float8_e4m3fn='fp8', float32='f32', uint8='u8', int32='i32',
           cuda=NS(is_current_stream_capturing=lambda: False))


class Tensor:
    def __init__(self, shape, dtype='i32', device='cuda', storage=None, contiguous=True):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.ndim, self.storage, self.contiguous = len(shape), storage or object(), contiguous

    def __len__(self):
        return self.shape[0]

    def numel(self):
        total = 1
        for size in self.shape:
            total *= size
        return total

    def is_contiguous(self):
        return self.contiguous

    def __getitem__(self, index):
        assert isinstance(index, slice) and index.start is None and index.step is None
        return Tensor((min(index.stop, len(self)), *self.shape[1:]), self.dtype,
                      self.device, self.storage, self.contiguous)


def function(source, name, namespace):
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual production function>', 'exec'), namespace)
    return namespace[name]


class PagedContractTests(unittest.TestCase):
    def setUp(self):
        self.ns = {'torch': TORCH}
        self.valid = function((RUNTIME / 'paged_backend.py').read_text(), 'valid_paged_inputs', self.ns)

    def inputs(self, rows=32768):
        return [Tensor((rows, 32, 128), 'fp8'), Tensor((rows, 32), 'f32'),
                Tensor((2048, 64, 132), 'u8'), Tensor((1, 2048)), Tensor((1, 131072)),
                Tensor((rows,)), Tensor((rows,)), Tensor((rows, 2048))]

    def test_supported_geometry_and_tensor_contract(self):
        for rows in (8, 988, 4096, 16384, 32768):
            self.assertTrue(self.valid('cuda', *self.inputs(rows)))
        for index, replacement in [(1, Tensor((32768, 32), 'fp8')),
                                   (2, Tensor((2048, 64, 132), 'f32')),
                                   (6, Tensor((32767,))), (7, Tensor((32768, 2047)))]:
            values = self.inputs()
            values[index] = replacement
            self.assertFalse(self.valid('cuda', *values))

    def test_h64_requires_matching_q_weight_and_slot_geometry(self):
        values = self.inputs()
        values[0] = Tensor((32768, 64, 128), 'fp8')
        values[1] = Tensor((32768, 64), 'f32')
        self.assertFalse(self.valid('cuda', *values))
        self.assertTrue(self.valid('cuda', *values, score_heads=64))
        values[1] = Tensor((32768, 32), 'f32')
        self.assertFalse(self.valid('cuda', *values, score_heads=64))

    def test_scalar_and_non_matrix_tables_decline_before_either_dispatch(self):
        slabbed = function((RUNTIME / 'slabbed_few.py').read_text(), '_eligible_slab_call', self.ns)
        paged = function((RUNTIME / 'paged_backend.py').read_text(), '_run_paged_block', self.ns)
        self.ns['execution_rows'] = lambda backend, rows: rows
        backend = NS(device='cuda', score_heads=32, max_q=16384, max_k=1048576, logical_query_capacity=32768,
                     _full_paged=False, _record5_direct_storage=False,
                     lt=NS(PRODUCTION_MIN_S=12288))
        options = dict(nk=131072, common_end=98305, last_visible=131072, hot_key=(0, 'one'))
        for slot in (3, 4):
            for shape in ((), (2048,), (1, 1, 2048), (2, 2048)):
                values = self.inputs()
                values[slot] = Tensor(shape)
                self.assertFalse(self.valid('cuda', *values))
                self.assertFalse(slabbed(backend, *values, **options))
                self.assertFalse(paged(backend, *values, **options))

    def test_full_recovery_calls_the_same_fi_abi_and_owned_output(self):
        calls, workspace = [], object()

        def buffer(*args, **kwargs):
            calls.append(('cache', args, kwargs))
            return workspace

        def descriptor(value):
            return (value.shape, value.dtype, value.device, value.storage) if isinstance(value, Tensor) else value

        def radix(*args, **kwargs):
            calls.append(('radix', tuple(map(descriptor, args)), kwargs))

        ns = dict(torch=TORCH, _get_cache_buf=buffer,
                  get_topk_module=lambda: NS(radix_topk_ragged_transform=radix))
        helper = function((RUNTIME / 'fi_output.py').read_text(), 'topk_into', ns)
        ns['topk_into'] = helper
        current = function((RUNTIME / 'post_refiner.py').read_text(), 'select', ns)
        owner = NS(offsets=Tensor((16384,)), replay_selected=Tensor((512, 2048)))
        scores, counts = Tensor((3, 131072), 'f32'), Tensor((3,))
        after = current(owner, scores, counts)
        self.assertEqual(calls, [
            ('cache', ('radix_topk_row_states_cuda', 1024 * 1024, 'cuda'), {'zero_init': True}),
            ('radix', (descriptor(scores), descriptor(owner.replay_selected[:3]),
                       descriptor(owner.offsets[:3]), descriptor(counts), workspace,
                       2048, False, 0, True), {'row_starts': None}),
        ])
        self.assertIs(after.storage, owner.replay_selected.storage)


if __name__ == '__main__':
    unittest.main()
