"""Schedule construction must preserve owned/borrowed storage and graph identity."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

REPO = next(p for p in Path(__file__).resolve().parents if (p / "python/sglang").is_dir())
ROOT = REPO / "python/sglang/srt/layers/attention/dsa/dosa_runtime"
PREFIX = "isolated_dosa_schedule_storage"


class Tensor:
    def __init__(self, size, *, device="cuda:0", dtype="int32", base=None):
        self.shape, self.device, self.dtype = (size,), device, dtype
        self.is_cuda, self.base = True, base

    def data_ptr(self):
        return self.base.data_ptr() if self.base is not None else id(self)

    def stride(self):
        return (1,)

    def is_contiguous(self):
        return True

    def __getitem__(self, item):
        size = len(range(*item.indices(self.shape[0])))
        return Tensor(size, device=self.device, dtype=self.dtype, base=self)


class Kernel:
    def __init__(self, function):
        self.function, self.calls = function, []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
        return launch


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.allocations = []
        def empty(size, *, device, dtype):
            value = Tensor(size, device=device, dtype=dtype)
            self.allocations.append(value)
            return value
        torch = types.SimpleNamespace(
            int32="int32", empty=empty,
            empty_like=lambda value: empty(value.shape[0], device=value.device,
                                           dtype=value.dtype),
            cuda=types.SimpleNamespace(get_device_properties=lambda device:
                                      types.SimpleNamespace(multi_processor_count=148)),
        )
        def jit(function=None, **kwargs):
            return Kernel(function) if function is not None else Kernel
        triton = types.ModuleType("triton")
        triton.jit = jit
        triton.cdiv = lambda a, b: (a + b - 1) // b
        triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
        language = types.ModuleType("triton.language")
        language.constexpr = object
        triton.language = language
        package = types.ModuleType(PREFIX)
        package.__path__ = [str(ROOT)]
        self.modules = patch.dict(sys.modules, {
            "torch": torch, "triton": triton, "triton.language": language,
            PREFIX: package,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        for name in ("query_slots", "schedule"):
            spec = importlib.util.spec_from_file_location(PREFIX + "." + name, ROOT / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        self.schedule = sys.modules[PREFIX + ".schedule"]
        self.extent = Tensor(1)

    def create(self, family, rows=64, **kwargs):
        return self.schedule.LiveSchedule(
            rows, 1048576, self.extent, paid=14 if family == 16 else 16, **kwargs)

    def test_owned_storage_still_allocates_once_per_output(self):
        for family in (16, 18):
            with self.subTest(family=family):
                self.allocations.clear()
                work = self.create(family)
                self.assertEqual([x.shape for x in self.allocations], [(8,), (8,), (2,)])
                self.assertEqual(work._output_leases, (work.ids, work.inverse, work.config))

    def test_borrowed_storage_has_no_allocations_and_keeps_old_leases(self):
        backing = dict(ids=Tensor(32), inverse=Tensor(32), config=Tensor(2))
        for family in (16, 18):
            with self.subTest(family=family):
                old = self.create(family, rows=128, backing=backing)
                old_signature = old._outputs
                current = self.create(family, rows=64, backing=backing)
                self.assertFalse(self.allocations)
                self.assertEqual(current.ids.shape, (8,))
                self.assertEqual(current.ids.data_ptr(), backing["ids"].data_ptr())
                self.assertIs(current.config, backing["config"])
                self.assertEqual(old.ids.shape, (16,))
                self.assertEqual(old._outputs, old_signature)
                current.build(Tensor(16))
                self.assertIs(current._prepare.calls[-1][1][2], current.ids)
                current.ids = Tensor(8)
                with self.assertRaisesRegex(RuntimeError, "output storage changes"):
                    current.build(Tensor(16))

    def test_bad_borrowed_capacity_rejected_before_a_kernel(self):
        for bad in (Tensor(7), Tensor(8, dtype="float32"), Tensor(8, device="cuda:1")):
            with self.subTest(shape=bad.shape, dtype=bad.dtype, device=bad.device):
                backing = dict(ids=bad, inverse=Tensor(8), config=Tensor(2))
                with self.assertRaisesRegex(ValueError, "Schedule backing"):
                    self.create(16, backing=backing)
                self.assertFalse(self.allocations)

    def test_live_geometry_and_operand_replacement_require_new_graph(self):
        for family in (16, 18):
            for field in ("geometry", "extent", "budget"):
                with self.subTest(family=family, field=field):
                    work = self.create(family)
                    budget = Tensor(16)
                    work.build(budget)
                    before = len(work._prepare.calls)
                    if field == "geometry":
                        work.grid += 1
                    elif field == "extent":
                        work.live_nk = Tensor(1)
                    else:
                        budget = Tensor(16)
                    with self.assertRaises(RuntimeError):
                        work.build(budget)
                    self.assertEqual(len(work._prepare.calls), before)

    def test_static_family_keeps_original_tile_and_launch_grid(self):
        for paid, tiles, grid in ((14, 4048, 2368), (16, 8096, 1184)):
            with self.subTest(paid=paid):
                work = self.schedule.StaticSchedule(64, 1048576, "cuda:0", paid=paid)
                self.assertEqual((work.tiles, work.grid), (tiles, grid))
                work.build(Tensor(16))
                _, args, options = self.schedule.prepare_static.calls[-1]
                self.assertEqual(args[4:], (8, tiles, 148, grid, 8, paid))
                self.assertEqual(options, {"num_warps": 4})


if __name__ == "__main__":
    unittest.main(verbosity=2)
