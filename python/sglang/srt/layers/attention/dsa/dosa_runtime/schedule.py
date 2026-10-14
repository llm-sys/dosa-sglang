"""Stable certificate compaction, partition policies and schedule storage.

H16 keeps the exact partition policy and twice the base grid. H18 keeps the
original eight-wave policy and base grid. Static and device-live K share these
policies; live schedules additionally bind all graph-owned storage identities.
"""

import torch
import triton
import triton.language as tl

from .query_slots import MAX_FEW_QUERY_CAPACITY


@triton.jit
def compact_groups(BUDGET, IDS, INV, N, PAID: tl.constexpr, B: tl.constexpr):
    g = tl.arange(0, B)
    ok = g < N
    b0 = tl.load(BUDGET + 2 * g, ok, 0)
    b1 = tl.load(BUDGET + 2 * g + 1, ok, 0)
    active = ok & ((b0 == PAID) | (b1 == PAID))
    pos = tl.cumsum(active.to(tl.int32), 0) - 1
    tl.store(IDS + pos, g, active)
    tl.store(INV + g, tl.where(active, pos, -1), ok)
    return tl.sum(active.to(tl.int32), 0)


@triton.jit
def exact_parts(count, tiles, sms: tl.constexpr, grid):
    # The callers bound tiles <= 8192 before using the exact256 breakpoints.
    groups = tl.maximum(count, 1)
    t = tl.maximum(tiles, 1)
    cap = tl.maximum(1, tl.minimum(t, grid // groups))
    chosen = tl.full((), 1, tl.int32)
    if (count > 0) & (tiles > 0):
        j = tl.arange(0, 256)
        d = j % 128 + 1
        p = tl.where(j < 128, d, (t - 1) // d + 1)
        valid = p <= cap
        pe = tl.minimum(p, cap)
        cost = ((groups * pe + sms - 1) // sms) * ((t + pe - 1) // pe)
        best = tl.min(tl.where(valid, cost, 2147483647), 0)
        chosen = tl.min(tl.where(valid & (cost == best), p, 2147483647), 0)
    return chosen


@triton.jit
def eight_wave_parts(count, tiles, sms: tl.constexpr, grid):
    groups = tl.maximum(count, 1)
    minwaves = (groups + sms - 1) // sms
    maxparts = tl.maximum(1, tl.minimum(tl.maximum(tiles, 1), grid // groups))
    chosen = tl.full((), 1, tl.int32)
    best = tl.full((), 2147483647, tl.int32)
    for delta in tl.static_range(8):
        wave = minwaves + delta
        p = tl.maximum(1, tl.minimum(maxparts, (wave * sms) // groups))
        cost = ((groups * p + sms - 1) // sms) * ((tiles + p - 1) // p)
        better = (cost < best) | ((cost == best) & (p < chosen))
        chosen = tl.where(better, p, chosen)
        best = tl.minimum(best, cost)
    return chosen


@triton.jit(do_not_specialize=["N", "TILES", "GRID"])
def prepare_static(BUDGET, IDS, INV, CONFIG, N, TILES, SMS: tl.constexpr,
                   GRID, B: tl.constexpr, PAID: tl.constexpr):
    count = compact_groups(BUDGET, IDS, INV, N, PAID, B)
    if PAID == 14:
        tl.static_assert(SMS > 0)
        # Keep this expression context: extracting it changes integer-division
        # lowering and regresses static H16 despite equivalent arithmetic.
        groups = tl.maximum(count, 1)
        t = tl.maximum(TILES, 1)
        maxparts = tl.maximum(1, tl.minimum(t, GRID // groups))
        chosen = tl.full((), 1, tl.int32)

        if (count > 0) & (TILES > 0) & (TILES <= 8192):
            j = tl.arange(0, 256)
            d = (j % 128) + 1
            p = tl.where(j < 128, d, (t - 1) // d + 1)
            valid = p <= maxparts
            # Invalid lanes retain their original validity, but use harmless
            # in-cap operands so every evaluated product stays in the same range.
            pe = tl.minimum(p, maxparts)
            costs = ((groups * pe + SMS - 1) // SMS) * ((t + pe - 1) // pe)
            best = tl.min(tl.where(valid, costs, 2147483647), 0)
            chosen = tl.min(tl.where(valid & (costs == best), p, 2147483647), 0)
        elif (count > 0) & (TILES > 8192):
            # Preserve a legal original policy outside the proved exact domain.
            # The admitted service's H16 maximum is4048, so it never takes this.
            minwaves = (groups + SMS - 1) // SMS
            best = tl.full((), 2147483647, tl.int32)
            for delta in tl.static_range(8):
                wave = minwaves + delta
                p = tl.maximum(1, tl.minimum(maxparts, (wave * SMS) // groups))
                costs = ((groups * p + SMS - 1) // SMS) * ((TILES + p - 1) // p)
                better = (costs < best) | ((costs == best) & (p < chosen))
                chosen = tl.where(better, p, chosen)
                best = tl.minimum(best, costs)

        chosen = tl.where((count == 0) | (TILES <= 0), 1, chosen)
    else:
        chosen = eight_wave_parts(count, TILES, SMS, GRID)
    tl.store(CONFIG, count)
    tl.store(CONFIG + 1, chosen)


@triton.jit(do_not_specialize=["N", "GRID"])
def prepare_live(BUDGET, LIVE_NK, IDS, INV, CONFIG, N, GRID,
                 SMS: tl.constexpr, MAX_K: tl.constexpr, SEED: tl.constexpr,
                 TILE: tl.constexpr, PAID: tl.constexpr, B: tl.constexpr):
    tl.static_assert(SMS > 0)
    tl.static_assert((PAID == 14 and TILE == 256) or (PAID == 16 and TILE == 128))
    tl.static_assert(SEED >= 0)
    tl.static_assert(SEED <= MAX_K)
    tl.static_assert(MAX_K <= 1048576)
    tl.static_assert(SEED % 256 == 0)
    tl.static_assert((MAX_K - SEED + TILE - 1) // TILE <= 8192)
    count = compact_groups(BUDGET, IDS, INV, N, PAID, B)
    nk = tl.load(LIVE_NK)
    valid = (nk >= SEED) & (nk <= MAX_K)
    # Clip before subtraction to avoid integer overflow. Invalid input still
    # publishes count=-1; clipping does not admit an invalid logical extent.
    safe_nk = tl.minimum(tl.maximum(nk, SEED), MAX_K)
    tiles = (safe_nk - SEED + TILE - 1) // TILE
    chosen = tl.full((), 1, tl.int32)
    if valid:
        if PAID == 14:
            chosen = exact_parts(count, tiles, SMS, GRID)
        else:
            chosen = eight_wave_parts(count, tiles, SMS, GRID)
    tl.store(CONFIG, tl.where(valid, count, -1))
    tl.store(CONFIG + 1, chosen)


def _binding(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.dtype, tensor.device)


class StaticSchedule:
    def __init__(self, rows, nk, device, *, paid, seed=12288, backing=None):
        assert rows > 0 and rows % 8 == 0 and 0 <= seed <= nk and seed % 256 == 0
        if paid not in (14, 16):
            raise ValueError("Indexed schedules require the H16 or H18 family")
        self.rows, self.groups = rows, rows // 8
        self.max_k, self.seed, self.paid = nk, seed, paid
        self.tile = 256 if paid == 14 else 128
        self.device = device
        self.tiles = self.capacity_tiles = triton.cdiv(nk - seed, self.tile)
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.launch_grid_multiplier = 2 if paid == 14 else 1
        self.grid = self.launch_grid_multiplier * max(self.groups, 8 * self.sms)
        if backing is None:
            self.ids = torch.empty(self.groups, device=device, dtype=torch.int32)
            self.inverse = torch.empty_like(self.ids)
            self.config = torch.empty(2, device=device, dtype=torch.int32)
        else:
            self.ids = backing["ids"][:self.groups]
            self.inverse = backing["inverse"][:self.groups]
            self.config = backing["config"]
            for tensor, size in ((self.ids, self.groups),
                                 (self.inverse, self.groups), (self.config, 2)):
                if (tensor.device != device or tensor.dtype != torch.int32
                        or tuple(tensor.shape) != (size,) or not tensor.is_contiguous()):
                    raise ValueError("Schedule backing requires contiguous CUDA int32 capacity")

    def build(self, budget):
        prepare_static[(1,)](
            budget, self.ids, self.inverse, self.config, self.groups, self.tiles,
            self.sms, self.grid, triton.next_power_of_2(self.groups), self.paid,
            num_warps=4,
        )


class LiveSchedule(StaticSchedule):
    _prepare = staticmethod(prepare_live)

    def __init__(self, rows, max_k, live_nk, *, paid, seed=12288, backing=None):
        if not (0 < rows <= MAX_FEW_QUERY_CAPACITY and rows % 8 == 0):
            raise ValueError("fixed execution Q must be8-aligned and within the supported capacity")
        if not (0 <= seed <= max_k <= 1048576 and seed % 256 == 0):
            raise ValueError("seed/capacity outside original geometry")
        if not (live_nk.is_cuda and live_nk.dtype == torch.int32
                and tuple(live_nk.shape) == (1,) and live_nk.is_contiguous()):
            raise ValueError("live_nk must be one contiguous CUDA int32")
        super().__init__(rows, max_k, live_nk.device, paid=paid, seed=seed, backing=backing)
        if not 0 < self.sms < 65536:
            raise ValueError("physical SM count outside original supported domain")
        self.live_nk = self._extent_lease = live_nk
        self._extent_binding = _binding(live_nk)
        self._budget_binding = self._budget_lease = None
        self._outputs = tuple(_binding(x) for x in (self.ids, self.inverse, self.config))
        self._output_leases = (self.ids, self.inverse, self.config)
        self._geometry = self._geometry_binding()

    def _geometry_binding(self):
        return (self.rows, self.groups, self.max_k, self.seed, self.paid, self.tile,
                self.sms, self.launch_grid_multiplier, self.grid, self.capacity_tiles)

    def build(self, budget4):
        if self._geometry_binding() != self._geometry:
            raise RuntimeError("new Q/capacity/policy requires a new workspace and graph")
        if not (budget4.is_cuda and budget4.device == self.device
                and budget4.dtype == torch.int32 and budget4.is_contiguous()
                and tuple(budget4.shape) == (self.rows // 4,)):
            raise ValueError("budget4 must be the fixed Q/4 CUDA int32 buffer")
        if _binding(self.live_nk) != self._extent_binding:
            raise RuntimeError("replace the workspace/graph when live_nk storage changes")
        if tuple(_binding(x) for x in (self.ids, self.inverse, self.config)) != self._outputs:
            raise RuntimeError("replace the workspace/graph when output storage changes")
        current = _binding(budget4)
        if self._budget_binding is None:
            self._budget_binding, self._budget_lease = current, budget4
        elif current != self._budget_binding:
            raise RuntimeError("replace the workspace/graph when budget storage changes")
        self._prepare[(1,)](
            budget4, self.live_nk, self.ids, self.inverse, self.config,
            self.groups, self.grid, self.sms, self.max_k, self.seed,
            self.tile, self.paid, triton.next_power_of_2(self.groups), num_warps=4,
        )

    def binding_manifest(self):
        return dict(rows=self.rows, groups=self.groups, max_k=self.max_k,
                    seed=self.seed, paid=self.paid, tile=self.tile,
                    capacity_tiles=self.capacity_tiles, physical_sms=self.sms,
                    launch_grid_multiplier=self.launch_grid_multiplier,
                    grid=self.grid, live_nk_pointer=self.live_nk.data_ptr(),
                    logical_tiles="device-derived; no host tiles argument")
