#pragma once
#include "paged_score_types.cuh"
#include "paged_score_access.cuh"
#define DOSA_PAGED_NAMESPACE paged_bitphase_completion
#define DOSA_PAGED_LIVE_EXTENT 0
#include "../common/exact_paged.inl"
#undef DOSA_PAGED_LIVE_EXTENT
#undef DOSA_PAGED_NAMESPACE
