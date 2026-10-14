"""Frozen record5 numerical chain; caller owns gather and final map/carry."""

from .seed_output import seed_logits


def run_full_core(lt, native, refiner, q, k, scale, weights, starts, ends, output, *, seed_workspace=None):
    rows, nk = len(q), len(k)
    seed = lt._PAGED_CANDIDATE_INLINE
    prep = lt._prep_bufs(rows, lt.NB, q.device)
    paged = lt._paged_cand_bufs(rows, nk, q.device)
    seed_start, seed_end = lt._ks0_keh(rows, seed, q.device)
    refiner.error.zero_()
    logits = seed_logits(
        q,
        k[:seed],
        scale[:seed],
        weights,
        seed_start,
        seed_end,
        workspace=seed_workspace,
    )
    native.seed_prep_checked(
        logits,
        lt.NB,
        2048,
        seed,
        seed,
        lt.HEADROOM,
        0,
        1,
        prep["o"][:rows],
        prep["inv"][:rows],
        prep["th"][:rows],
        prep["bc"][:rows],
        paged["inline_val"],
        paged["inline_idx"],
        prep["cc"][:rows],
        refiner.error,
    )
    refiner.metadata.prepare(
        q,
        k,
        scale,
        weights,
        prep["o"][:rows],
        prep["inv"][:rows],
        refiner.epsilon,
        refiner.error,
        refiner.metadata_workspace,
    )
    # Native validates a four-scale-padded allocation; the caller owns this tail.
    aligned = (nk + 3) // 4 * 4
    scale_native = scale.as_strided((aligned,), (1,))
    native.surrogate_scan(
        q,
        k,
        scale_native,
        weights,
        seed_end,
        ends,
        prep["o"][:rows],
        prep["inv"][:rows],
        prep["th"][:rows],
        paged["inline_val"],
        paged["inline_idx"],
        paged["overflow_val"],
        paged["overflow_idx"],
        paged["page_table"],
        paged["pool_head"],
        paged["pool_status"],
        prep["cc"][:rows],
        prep["bc"][:rows],
        nk,
        lt.NB,
        2048,
        refiner.epsilon,
    )
    native.h2048_coarse_band_paged_(
        paged["inline_val"],
        paged["inline_idx"],
        paged["overflow_val"],
        paged["overflow_idx"],
        paged["page_table"],
        prep["cc"][:rows],
        paged["pool_status"],
        output,
        prep["status"][:rows],
        prep["bc"][:rows],
        nk,
        nk,
        [
            starts,
            ends,
            refiner.epsilon,
            refiner.error,
            refiner.boundary_ids,
            refiner.boundary_counts,
            refiner.values,
            refiner.ids,
            refiner.counts,
            refiner.tasks.view(-1)[:rows],
        ],
    )
    refiner(q, k, scale, weights, starts, ends, paged, prep, output)
    return output
