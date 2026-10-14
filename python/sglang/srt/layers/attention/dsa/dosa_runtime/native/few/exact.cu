#include "h32_common.cuh"
#include "exact_paged.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("exact_paged_solo", &exact_paged_completion<1>);
    m.def("exact_paged_common", &exact_paged_completion<4>);
}
