#pragma once
#define DOSA_EMIT_TENSOR_TYPES certicascade_streaming_exact
#define DOSA_EMIT_MAKE_MAP make_map
#define DOSA_EMIT_SCALE_STRIDE scale_row_stride
#define DOSA_EMIT_DEFAULT_NAMESPACE fused_few
#include "../common/emit_types.inl"
#undef DOSA_EMIT_DEFAULT_NAMESPACE
#undef DOSA_EMIT_SCALE_STRIDE
#undef DOSA_EMIT_MAKE_MAP
#undef DOSA_EMIT_TENSOR_TYPES
