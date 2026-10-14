#pragma once
#include "paged_score_types.cuh"
#include "../common/score_emit_loop.cuh"
#define DOSA_PAGED_NAMESPACE paged_bitphase_completion
#define DOSA_PAGED_LIVE_EXTENT 0
#include "../common/paged_score_access.inl"
#undef DOSA_PAGED_LIVE_EXTENT
#undef DOSA_PAGED_NAMESPACE
