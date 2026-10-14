"""Device-live-NK initialization/key metadata; no host readback or cached numerics."""
import triton
import triton.language as tl


@triton.jit
def clear_live_table(TABLE, ERROR, STATUS, LIVE, Q: tl.constexpr,
                     STRIDE: tl.constexpr, CAP: tl.constexpr,
                     SEED: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    nk = tl.load(LIVE)
    valid = (nk >= SEED) & (nk <= CAP)
    if valid:
        columns = (nk - SEED + 127) // 128
        for begin in range(0, columns, BLOCK):
            c = begin + tl.arange(0, BLOCK)
            tl.store(TABLE + row * STRIDE + c, -1, c < columns)
    else:
        if row < Q:
            tl.atomic_or(ERROR + row, 1)
        if row == 0:
            tl.atomic_or(STATUS, 2)


def initialize_table(runtime):
    # All original row/scalar zeroing precedes this call. A grow clears every
    # currently live column, including columns that belonged to an older epoch.
    clear_live_table[(runtime.rows + runtime.rows // 4,)](
        runtime.page_table, runtime.error, runtime.pool_status, runtime.live_nk,
        runtime.rows, runtime.page_table.stride(0), runtime.nk, runtime.seed,
        256, num_warps=4)


@triton.jit
def key_parts_live(KS, RHO, PS, PB, LIVE, CAP: tl.constexpr,
                   SEED: tl.constexpr, WORKERS: tl.constexpr, BLOCK: tl.constexpr):
    worker = tl.program_id(0)
    nk = tl.load(LIVE)
    valid = (nk >= SEED) & (nk <= CAP)
    maximum = tl.full((BLOCK,), 0.0, tl.float32)
    bad = tl.full((BLOCK,), 0, tl.int32)
    if valid:
        parts = (nk + BLOCK - 1) // BLOCK
        for part in range(worker, parts, WORKERS):
            i = part * BLOCK + tl.arange(0, BLOCK)
            m = i < nk
            scale = tl.load(KS + i, m, 1.0)
            bits = scale.to(tl.int32, bitcast=True)
            rho = tl.load(RHO + i, m, 0)
            ok = (bits >= 0x00800000) & (bits < 0x7F800000) & (rho >= 0) & (rho < 127)
            maximum = tl.maximum(maximum, tl.where(m & ok, scale, 0.0))
            bad = bad | (m & ~ok).to(tl.int32)
    else:
        bad = tl.full((BLOCK,), 1, tl.int32)
    tl.store(PS + worker, tl.max(maximum, 0))
    tl.store(PB + worker, tl.max(bad, 0))


def prepare_live_keys(workspace, scale, rho, key_finish):
    workers = workspace.live_workers
    key_parts_live[(workers,)](
        scale, rho, workspace.ps, workspace.pb, workspace.live_nk,
        workspace.nk, workspace.live_seed, workers, 1024, num_warps=4)
    # Every worker writes even when inactive; this fixed small reduction never
    # observes old per-worker values after shrink. Key loads are live-bounded.
    key_finish[(1,)](workspace.ps, workspace.pb, workspace.smax,
                    workspace.key_bad, workers, triton.next_power_of_2(workers),
                    num_warps=4)
