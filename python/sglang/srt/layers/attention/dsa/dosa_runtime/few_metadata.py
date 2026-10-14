"""Original U against already-paid literal seed tau; no affine/full-surrogate fields."""

import math
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as ld


@triton.jit
def add_up(x, y):
    return ld.add_ru(x, y)


@triton.jit(do_not_specialize=["N"])
def key_parts(KS, RHO, PS, PB, N, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < N
    s = tl.load(KS + i, m, 1.0)
    bits = s.to(tl.int32, bitcast=True)
    r = tl.load(RHO + i, m, 0)
    ok = (bits >= 0x00800000) & (bits < 0x7F800000) & (r >= 0) & (r < 127)
    tl.store(PS + tl.program_id(0), tl.max(tl.where(m & ok, s, 0.0), 0))
    tl.store(PB + tl.program_id(0), tl.max((m & ~ok).to(tl.int32), 0))


@triton.jit(do_not_specialize=["N"])
def key_finish(PS, PB, S, BAD, N, B: tl.constexpr):
    i = tl.arange(0, B)
    tl.store(S, tl.max(tl.load(PS + i, i < N, 0.0), 0))
    tl.store(BAD, tl.max(tl.load(PB + i, i < N, 0), 0))


@triton.jit
def seed_finite(
    SS, SB, K: tl.constexpr, STRIDE: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    x = tl.load(SS + row * STRIDE + i, i < K, 0.0)
    mag = x.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    tl.store(SB + row, tl.max((mag >= 0x7F800000).to(tl.int32), 0))


@triton.jit
def query_check(
    W,
    H14,
    H16,
    H20,
    BUDGET,
    S,
    KBAD,
    SB,
    TAU,
    ERR,
    SAFE,
    GROUP: tl.constexpr,
    DOT: tl.constexpr,
    G: tl.constexpr,
    SEED_DIAG: tl.constexpr,
    HEADS: tl.constexpr,
    HIGH: tl.constexpr,
):
    row = tl.program_id(0)
    b = tl.load(BUDGET + row // GROUP)
    valid_family = (b == 14) | (b == 16) | (b == HIGH)
    hp = tl.where(b == 14, H14, tl.where(b == 16, H16, H20))
    hint = tl.load(hp + row)
    h = tl.arange(0, HEADS)
    w = tl.load(W + row * HEADS + h)
    mag = w.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    domain = (mag == 0) | ((mag >= 0x15800000) & (mag <= 0x6E800000))
    sw = tl.reduce(tl.abs(w).to(tl.float64), 0, add_up)
    one = tl.full((), 1.0, tl.float64)
    u = tl.full((), 2.0**-24, tl.float64)
    d = ld.mul_ru(
        ld.add_ru(one, tl.full((), DOT, tl.float64)),
        tl.full((), 25690112.0, tl.float64),
    )
    pre = ld.mul_ru(ld.mul_ru(d, sw), ld.add_ru(one, tl.full((), G, tl.float64)))
    double_pre = ld.mul_ru(tl.full((), 2.0, tl.float64), pre)
    smax = ld.add_ru(
        ld.mul_ru(ld.mul_ru(ld.add_ru(one, u), tl.load(S).to(tl.float64)), pre),
        tl.full((), 2.0**-126, tl.float64),
    )
    limit = tl.full((), 3.4028234663852886e38, tl.float64)
    tau = tl.load(TAU + row)
    bad = (
        ~valid_family
        | (hint != 2)
        | (tl.load(KBAD) != 0)
        | (tl.load(SB + row * 4 + 3) == 3 if SEED_DIAG else tl.load(SB + row) != 0)
        | (tl.sum((~domain).to(tl.int32)) != 0)
        | (sw > tl.full((), 2.0**100, tl.float64))
        | ~(double_pre < limit)
        | ~(smax < limit)
        | ~(tl.abs(tau) < float("inf"))
    )
    if bad:
        tl.atomic_or(ERR + row, 1024)
    tl.store(SAFE + row, tl.where(bad | (tl.load(ERR + row) != 0), -float("inf"), tau))


class Workspace:
    def __init__(self, nk, rows, device):
        self.nk = nk
        self.live_nk = None
        self.live_workers = None
        self.live_seed = 12288
        self.rows = rows
        self.parts = triton.cdiv(nk, 1024)
        self.reduction_width = triton.next_power_of_2(self.parts)
        self.ps = torch.empty(self.parts, device=device, dtype=torch.float32)
        self.pb = torch.empty(self.parts, device=device, dtype=torch.int32)
        self.smax = torch.empty(1, device=device, dtype=torch.float32)
        self.key_bad = torch.empty(1, device=device, dtype=torch.int32)
        self.seed_bad = torch.empty(rows, device=device, dtype=torch.int32)

    def keys(self, scale, rho):
        if self.live_nk is not None:
            from .live_nk_ops import prepare_live_keys
            return prepare_live_keys(self, scale, rho, key_finish)
        key_parts[(self.parts,)](
            scale, rho, self.ps, self.pb, self.nk, 1024, num_warps=4
        )
        key_finish[(1,)](
            self.ps,
            self.pb,
            self.smax,
            self.key_bad,
            self.parts,
            self.reduction_width,
            num_warps=4,
        )

    def seed(self, seed_scores):
        seed_finite[(self.rows,)](
            seed_scores,
            self.seed_bad,
            seed_scores.shape[1],
            seed_scores.stride(0),
            triton.next_power_of_2(seed_scores.shape[1]),
            num_warps=8,
        )


def queries(weights, plans, budget8, tau, error, safe_tau, workspace, group_size=8, seed_diag=None):
    dot = math.nextafter(128 * 2.0**-23 / (1 - 128 * 2.0**-23), math.inf)
    steps = weights.shape[1] // 4 + 3
    g = math.nextafter(steps * 2.0**-24 / (1 - steps * 2.0**-24), math.inf)
    query_check[(len(weights),)](
        weights,
        plans[14][5],
        plans[16][5],
        plans[max(plans)][5],
        budget8,
        workspace.smax,
        workspace.key_bad,
        workspace.seed_bad if seed_diag is None else seed_diag,
        tau,
        error,
        safe_tau,
        group_size,
        dot,
        g,
        seed_diag is not None,
        weights.shape[1],
        max(plans),
        num_warps=1,
        enable_fp_fusion=False,
    )


def prepare(
    weights,
    plans,
    budget8,
    tau,
    seed_scores,
    scale,
    rho,
    error,
    safe_tau,
    workspace,
    group_size=8,
    seed_diag=None,
):
    workspace.keys(scale, rho)
    if seed_diag is None:
        workspace.seed(seed_scores)
    queries(weights, plans, budget8, tau, error, safe_tau, workspace, group_size, seed_diag)
