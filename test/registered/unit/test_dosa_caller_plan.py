"""Planner storage generations and explicit-device prior dependencies."""
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'python/sglang').is_dir())
SOURCE = ROOT / 'python/sglang/srt/layers/attention/dsa/dosa_runtime/caller_state.py'


class Device:
    type = 'cuda'
    def __init__(self, index):
        self.index = index
    def __str__(self):
        return f'cuda:{self.index}'


class Tensor:
    def __init__(self, shape, device, dtype='int32', log=None):
        self.shape, self.device, self.dtype = tuple(shape), device, dtype
        self.log = log
    def __getitem__(self, item):
        size = len(range(*item.indices(self.shape[0])))
        return Tensor((size, *self.shape[1:]), self.device, self.dtype, self.log)
    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result
    def dim(self):
        return len(self.shape)
    def view(self, dtype):
        return Tensor(self.shape, self.device, dtype, self.log)
    def record_stream(self, stream):
        self.log.append(('record_stream', stream))


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.log, self.allocations = [], []
        self.device = Device(0)
        self.stream = types.SimpleNamespace(cuda_stream=17)
        self.foreign = types.SimpleNamespace(cuda_stream=29)
        def current_stream(device=None):
            return self.stream if device is self.device else self.foreign
        def allocate(*shape, device, dtype):
            value = Tensor(shape, device, dtype, self.log)
            self.allocations.append(value)
            return value
        torch = types.SimpleNamespace(
            uint8='uint8', int16='int16', int32='int32', int64='int64',
            float32='float32', float8_e4m3fn='fp8',
            empty=allocate, zeros=allocate, arange=allocate,
            cuda=types.SimpleNamespace(current_stream=current_stream,
                                      get_device_capability=lambda dev: (10, 0)),
        )
        spec = importlib.util.spec_from_file_location('caller_plan_under_test', SOURCE)
        module = importlib.util.module_from_spec(spec)
        self.patch = patch.dict('sys.modules', {'torch': torch})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        spec.loader.exec_module(module)
        self.native = types.SimpleNamespace(plan_and_permuted_paged_gather_out=
                                           lambda *args: self.log.append(('native', args)))
        self.caller = module.CallerState(self.native)

    def test_epoch_is_owned_by_the_reused_or_grown_storage(self):
        first, epoch = self.caller._pair_plan_bufs(196608, self.device)
        self.assertEqual(epoch, 1)
        same, epoch = self.caller._pair_plan_bufs(131072, self.device)
        self.assertIs(first, same)
        self.assertEqual(epoch, 2)
        grown, epoch = self.caller._pair_plan_bufs(524289, self.device)
        self.assertIsNot(first, grown)
        self.assertEqual((grown['cap'], epoch), (1048576, 1))
        self.assertEqual(first['epoch'], 2)
        self.assertFalse(hasattr(self.caller, '_PAIR_PLAN_EPOCH'))

    def test_epoch_wrap_replaces_storage_once_and_restarts_at_one(self):
        first, _ = self.caller._pair_plan_bufs(196608, self.device)
        first['epoch'] = (1 << 30) - 2
        same, epoch = self.caller._pair_plan_bufs(196608, self.device)
        self.assertIs(first, same)
        self.assertEqual(epoch, (1 << 30) - 1)
        self.allocations.clear()
        fresh, epoch = self.caller._pair_plan_bufs(196608, self.device)
        self.assertIsNot(first, fresh)
        self.assertEqual(epoch, 1)
        self.assertEqual(len(self.allocations), 5)

    def test_wait_and_lifetime_use_the_destination_stream(self):
        n = 196608
        hot = Tensor((12288,), self.device, 'int64', self.log)
        ready = types.SimpleNamespace(wait=lambda stream=None:
                                      self.log.append(('wait', stream)))
        key = (6, 'independent-request')
        self.caller._HOT_CARRY[(str(self.device), key)] = (hot, n - 4096, ready, 0)
        plan = self.caller.prepare_permuted_gather(
            object(), Tensor((n, 128), self.device, 'fp8', self.log),
            Tensor((n, 4), self.device, 'uint8', self.log), object(),
            sequence_length=n, query_length=4096, num_reqs=1,
            common_end=n - 4096, window_start=0, hot_key=key,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(self.log[:2], [('wait', self.stream), ('record_stream', self.stream)])
        self.assertEqual(self.log[2][0], 'native')
        self.assertIs(plan['carry_event'], ready)
        self.assertEqual(plan['planner_state']['epoch'], 1)

    def test_missing_prior_still_declines_without_work(self):
        n = 196608
        result = self.caller.prepare_permuted_gather(
            object(), Tensor((n, 128), self.device, 'fp8', self.log),
            Tensor((n, 4), self.device, 'uint8', self.log), object(),
            sequence_length=n, query_length=4096, num_reqs=1,
            common_end=n - 4096, window_start=0, hot_key=(6, 'new-request'),
        )
        self.assertIsNone(result)
        self.assertFalse(self.log)
        self.assertFalse(self.allocations)

    def test_query_admission_is_independent_of_previous_call_shapes(self):
        n = 229376
        key = (6, 'independent-request')
        hot = Tensor((12288,), self.device, 'int64', self.log)
        self.caller._HOT_CARRY[(str(self.device), key)] = (hot, 196608, None, 0)
        for rows in (32768, 8, 16384, 8192, 992, 4096, 16392, 32768):
            with self.subTest(rows=rows):
                plan = self.caller.prepare_permuted_gather(
                    object(), Tensor((n, 128), self.device, 'fp8', self.log),
                    Tensor((n, 4), self.device, 'uint8', self.log), object(),
                    sequence_length=n, query_length=rows, num_reqs=1,
                    common_end=196608, window_start=0, hot_key=key,
                )
                self.assertIsNotNone(plan)
                self.assertEqual(plan['query_length'], rows)
        self.assertNotIn('FP8_FUSED_QUERY_LENS', vars(self.caller))

    def test_nonpositive_queries_decline_before_planner_work(self):
        n = 229376
        for rows in (0, -1):
            result = self.caller.prepare_permuted_gather(
                object(), Tensor((n, 128), self.device, 'fp8', self.log),
                Tensor((n, 4), self.device, 'uint8', self.log), object(),
                sequence_length=n, query_length=rows, num_reqs=1,
                common_end=196608, window_start=0, hot_key=(6, 'one'),
            )
            self.assertIsNone(result)
        self.assertFalse(self.log)
        self.assertFalse(self.allocations)


if __name__ == '__main__':
    unittest.main(verbosity=2)
