#pragma once
#include "paged_score_types_live.cuh"
#include "../common/score_emit_loop.cuh"
#define DOSA_PAGED_NAMESPACE live_record5_completion
#define DOSA_PAGED_LIVE_EXTENT 1
#include "../common/paged_score_access.inl"
#undef DOSA_PAGED_LIVE_EXTENT
#undef DOSA_PAGED_NAMESPACE
