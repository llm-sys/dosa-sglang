#pragma once
#define DOSA_EMIT_TENSOR_TYPES f2_live_h22_tensor_types
#define DOSA_EMIT_MAKE_MAP f2_h22_late_make_map
#define DOSA_EMIT_SCALE_STRIDE f2_h22_late_scale_row_stride
#define DOSA_EMIT_DEFAULT_NAMESPACE f2_live_h22_fallback
#include "../common/emit_types.inl"
#undef DOSA_EMIT_DEFAULT_NAMESPACE
#undef DOSA_EMIT_SCALE_STRIDE
#undef DOSA_EMIT_MAKE_MAP
#undef DOSA_EMIT_TENSOR_TYPES
