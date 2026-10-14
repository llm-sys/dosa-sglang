"""GPU error-envelope metadata; all large-domain work is explicit and charged."""

import math
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as ld


@triton.jit
def _sum_up(a, b):
    return ld.add_ru(a, b)


@triton.jit
def _key_parts(K, S, PM, PB, N: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0) * B + tl.arange(0, B)
    d = tl.arange(0, 128)
    raw = (
        tl.load(K + row[:, None] * 128 + d[None, :], row[:, None] < N, 0).to(tl.int32)
        & 127
    )
    ex = raw >> 3
    ma = raw & 7
    unit = tl.where(ex == 0, ma, (8 + ma) << tl.maximum(ex - 1, 0)).to(tl.int64)
    sq = tl.sum(unit * unit, 1)
    norm = ld.sqrt_ru(sq.to(tl.float64)) * 0.001953125
    scale = tl.load(S + row, row < N, 0)
    mag = scale.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    badscale = (mag >= 0x7F800000) | ((mag != 0) & (mag < 0x00800000))
    bad = (
        tl.sum(
            ((tl.sum((raw == 127).to(tl.int32), 1) > 0) | badscale).to(tl.int32)
            * (row < N).to(tl.int32),
            0,
        )
        > 0
    )
    m = tl.max(ld.mul_ru(norm, tl.abs(scale).to(tl.float64)), 0)
    tl.store(PM + tl.program_id(0), m)
    tl.store(PB + tl.program_id(0), bad.to(tl.int32))


@triton.jit(do_not_specialize=["NP"])
def _key_finish(PM, PB, M, BAD, NP, B: tl.constexpr):
    x = tl.arange(0, B)
    mm = tl.max(tl.load(PM + x, x < NP, 0), 0)
    bad = tl.max(tl.load(PB + x, x < NP, 0), 0)
    tl.store(M, mm)
    tl.store(BAD, bad)


@triton.jit
def _query_eps(
    Q,
    W,
    O,
    I,
    M,
    KBAD,
    E,
    ERROR,
    N: tl.constexpr,
    DOT: tl.constexpr,
    GAMMA: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, 32)
    d = tl.arange(0, 128)
    raw = tl.load(Q + row * 4096 + h[:, None] * 128 + d[None, :]).to(tl.int32) & 127
    ex = raw >> 3
    ma = raw & 7
    unit = tl.where(ex == 0, ma, (8 + ma) << tl.maximum(ex - 1, 0)).to(tl.int64)
    qnorm = ld.sqrt_ru(tl.sum(unit * unit, 1).to(tl.float64)) * 0.001953125
    w = tl.load(W + row * 32 + h)
    iv = tl.load(I + row)
    origin = tl.load(O + row)
    off = (-origin) * iv
    fold = w * (-iv)
    wm = w.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    fm = fold.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    ok_w = (wm == 0) | ((wm >= 0x15800000) & (wm <= 0x6E800000))
    ok_f = (fm == 0) | ((fm >= 0x15800000) & (fm <= 0x6E800000))
    ivm = iv.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    finite_o = (origin.to(tl.int32, bitcast=True) & 0x7FFFFFFF) < 0x7F800000
    finite_off = (off.to(tl.int32, bitcast=True) & 0x7FFFFFFF) < 0x7F800000
    bad = (
        (tl.sum((raw == 127).to(tl.int32)) > 0)
        | (tl.sum((~ok_w | ~ok_f).to(tl.int32)) > 0)
        | (~finite_o)
        | (~finite_off)
        | (iv <= 0)
        | (ivm < 0x00800000)
        | (ivm >= 0x7F800000)
        | (tl.load(KBAD) != 0)
    )
    wd = w.to(tl.float64)
    fd = fold.to(tl.float64)
    v = iv.to(tl.float64)
    oo = tl.abs(off).to(tl.float64)
    # Product of two F32s is exact in F64; their cancellation is exact here.
    discrepancy = tl.abs(fd + v * wd)
    su = tl.reduce(ld.mul_ru(tl.abs(wd), qnorm), 0, _sum_up)
    sf = tl.reduce(ld.mul_ru(tl.abs(fd), qnorm), 0, _sum_up)
    sd = tl.reduce(ld.mul_ru(discrepancy, qnorm), 0, _sum_up)
    one = tl.full((), 1, tl.float64)
    u = tl.full((), 2.0**-24, tl.float64)
    tiny = tl.full((), 2.0**-126, tl.float64)
    g = tl.full((), GAMMA, tl.float64)
    dg = tl.full((), DOT, tl.float64)
    og = ld.add_ru(one, g)
    ou = ld.add_ru(one, u)
    factor = ld.mul_ru(
        ld.mul_ru(ld.add_ru(one, dg), tl.load(M)), tl.full((), 1 + 2.0**-40, tl.float64)
    )
    A = ld.mul_ru(factor, su)
    F = ld.mul_ru(factor, sf)
    delta = ld.mul_ru(factor, sd)
    true_scale = ld.add_ru(ld.mul_ru(ld.add_ru(g, ld.mul_ru(u, og)), A), tiny)
    smax = ld.add_ru(ld.mul_ru(ld.mul_ru(ou, og), A), tiny)
    tmax = ld.add_ru(ld.mul_ru(v, smax), oo)
    fmax = ld.add_ru(ld.mul_ru(og, F), oo)
    e_true = ld.add_ru(ld.add_ru(ld.mul_ru(v, true_scale), ld.mul_ru(u, tmax)), tiny)
    e_fast = ld.add_ru(
        ld.add_ru(delta, ld.mul_ru(g, F)), ld.add_ru(ld.mul_ru(u, fmax), tiny)
    )
    ep64 = ld.mul_ru(ld.add_ru(e_true, e_fast), tl.full((), 1 + 2.0**-40, tl.float64))
    ep = ld.nextafter(ld.double2float_ru(ep64), tl.full((), float("inf"), tl.float32))
    maximum = tl.full((), 3.4028234663852886e38, tl.float64)
    bad = (
        bad
        | (smax >= maximum)
        | (tmax >= maximum)
        | (fmax >= maximum)
        | ((ep.to(tl.int32, bitcast=True) & 0x7FFFFFFF) >= 0x7F800000)
    )
    tl.store(E + row, tl.where(bad, 0.0, ep))
    # Sticky metadata status is outside scan scratch and is never cleared here.
    if bad:
        tl.atomic_or(ERROR + row, 16)


class SurrogateMetadata:
    def __init__(self, nk, device):
        self.nk = nk
        self.block = 128
        self.parts = triton.cdiv(nk, self.block)
        self.partial = torch.empty(self.parts, dtype=torch.float64, device=device)
        self.partial_bad = torch.empty(self.parts, dtype=torch.int32, device=device)
        self.maximum = torch.empty(1, dtype=torch.float64, device=device)
        self.key_bad = torch.empty(1, dtype=torch.int32, device=device)
        self.gather = None

    def keys(self, k, scale):
        gather = self.gather
        if gather is not None and gather.fused:
            partial, bad, parts = gather.maximum, gather.bad, gather.parts
            width = gather.reduction_width
        else:
            partial, bad, parts = self.partial, self.partial_bad, self.parts
            width = triton.next_power_of_2(parts)
            _key_parts[(parts,)](
                k.view(torch.uint8), scale, partial, bad, self.nk, self.block,
                num_warps=8,
            )
        _key_finish[(1,)](
            partial, bad, self.maximum, self.key_bad, parts, width,
            num_warps=8,
        )

    def queries(self, q, w, origin, inv, epsilon, error):
        dot = math.nextafter(128 * 2.0**-23 / (1 - 128 * 2.0**-23), math.inf)
        gamma = math.nextafter(11 * 2.0**-24 / (1 - 11 * 2.0**-24), math.inf)
        _query_eps[(len(q),)](
            q.view(torch.uint8),
            w,
            origin,
            inv,
            self.maximum,
            self.key_bad,
            epsilon,
            error,
            len(q),
            dot,
            gamma,
            num_warps=4,
        )


def allocate_workspace(nk, device):
    return SurrogateMetadata(nk, device)


def prepare(q, k, scale, w, origin, inv, epsilon, error, workspace):
    """Call after seed_prep; OR failure bits into the existing seed status."""
    workspace.keys(k, scale)
    workspace.queries(q, w, origin, inv, epsilon, error)
