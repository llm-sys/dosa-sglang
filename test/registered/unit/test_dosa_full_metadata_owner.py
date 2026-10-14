"""CPU routing checks for original and gather-produced Full key metadata."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[3] / 'python/sglang/srt/layers/attention/dsa/dosa_runtime'


class Kernel:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((self.name, grid, args, kwargs))
        return launch


class MetadataOwnerTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.byte = object()
        self.key_bytes = object()
        self.keys = SimpleNamespace(view=lambda dtype: self.key_bytes)
        tree = ast.parse((ROOT / 'surrogate_metadata.py').read_text())
        owner = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == 'SurrogateMetadata')
        namespace = dict(
            torch=SimpleNamespace(uint8=self.byte),
            triton=SimpleNamespace(next_power_of_2=lambda n: 1 << (n-1).bit_length()),
            _key_parts=Kernel('parts', self.calls),
            _key_finish=Kernel('finish', self.calls),
        )
        exec(compile(ast.Module(body=[owner], type_ignores=[]), str(ROOT), 'exec'), namespace)
        self.owner = namespace['SurrogateMetadata'].__new__(namespace['SurrogateMetadata'])
        self.owner.__dict__.update(nk=2176, block=128, parts=17, partial=object(),
                                   partial_bad=object(), maximum=object(), key_bad=object(),
                                   gather=None)

    def check_unfused(self):
        owner = self.owner
        scale = object()
        owner.keys(self.keys, scale)
        self.assertEqual(self.calls, [
            ('parts', (17,), (self.key_bytes, scale, owner.partial, owner.partial_bad,
                              2176, 128), dict(num_warps=8)),
            ('finish', (1,), (owner.partial, owner.partial_bad, owner.maximum,
                              owner.key_bad, 17, 32), dict(num_warps=8)),
        ])

    def test_original_path_without_gather(self):
        self.check_unfused()

    def test_unfused_gather_keeps_original_path(self):
        self.owner.gather = SimpleNamespace(fused=False)
        self.check_unfused()

    def test_gather_reduction_uses_actual_extent_and_fixed_policy(self):
        owner = self.owner
        gather = SimpleNamespace(fused=True, parts=68, finish_capacity=256,
                                 maximum=object(), bad=object(), fixed_finish=False)
        owner.gather = gather
        for fixed, width in ((False, 128), (True, 256)):
            with self.subTest(fixed=fixed):
                self.calls.clear()
                gather.fixed_finish = fixed
                gather.reduction_width = width
                owner.keys(self.keys, object())
                self.assertEqual(self.calls, [
                    ('finish', (1,), (gather.maximum, gather.bad, owner.maximum,
                                      owner.key_bad, 68, width), dict(num_warps=8))])
        self.assertNotIn('keys', vars(owner))

    def test_adapter_preserves_method_and_owns_only_gather_output(self):
        tree = ast.parse((ROOT / 'gather_norm_adapter.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        allocated = []

        def empty(size, *, device, dtype):
            value = SimpleNamespace(size=size, device=device, dtype=dtype)
            allocated.append(value)
            return value

        namespace = dict(
            torch=SimpleNamespace(empty=empty, float64='f64', int32='i32'),
            triton=SimpleNamespace(next_power_of_2=lambda n: 1 << (n-1).bit_length()),
        )
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(ROOT), 'exec'), namespace)
        owner = self.owner
        owner.maximum = SimpleNamespace(device='cuda:0')
        original_keys = owner.keys
        calls = []
        core = SimpleNamespace(plan_and_permuted_paged_gather_norm_out=lambda *args: calls.append(args))
        adapter = namespace['GatherNormAdapter'](core, owner, 2176)
        self.assertEqual(owner.keys, original_keys)
        self.assertIs(owner.gather, adapter)
        self.assertFalse(adapter.fused)
        args = (object(), object())
        adapter.plan_and_permuted_paged_gather_out(*args)
        self.assertEqual(calls, [(*args, *allocated)])
        self.assertTrue(adapter.fused)
        self.assertEqual(adapter.parts, 68)
        self.assertEqual(adapter.reduction_width, 128)
        adapter.fixed_finish = True
        adapter.finish_capacity = 256
        self.assertEqual(adapter.reduction_width, 256)


if __name__ == '__main__':
    unittest.main()
