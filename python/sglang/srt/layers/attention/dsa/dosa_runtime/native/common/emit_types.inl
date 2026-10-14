// Shared layout, TMA host helpers and strict finite pruning predicate.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda_fp8.h>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/impls/sm100_mqa_logits.cuh>
namespace DOSA_EMIT_TENSOR_TYPES { using torch::Tensor; }
inline CUtensorMap DOSA_EMIT_MAKE_MAP(void *pointer, CUtensorMapDataType dtype,
                            std::initializer_list<cuuint64_t> dimensions,
                            std::initializer_list<cuuint64_t> strides,
                            std::initializer_list<cuuint32_t> box,
                            CUtensorMapSwizzle swizzle) {
  CUtensorMap map;
  const cuuint32_t element_strides[3] = {1, 1, 1};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, dtype, dimensions.size(), pointer, dimensions.begin(),
      strides.begin(), box.begin(), element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS,
              "cuTensorMapEncodeTiled failed: ", result);
  return map;
}
inline cuuint64_t DOSA_EMIT_SCALE_STRIDE(int64_t nk){
  return (static_cast<cuuint64_t>(nk)*sizeof(float)+15)&~cuuint64_t(15);
}
#ifdef DOSA_FEW_NAMESPACE
namespace DOSA_FEW_NAMESPACE {
#else
namespace DOSA_EMIT_DEFAULT_NAMESPACE {
#endif
using namespace DOSA_EMIT_TENSOR_TYPES;
using Barrier=cutlass::arch::ClusterTransactionBarrier;
struct PagedArgs {
 int *pool,*table,*head,*status,*error,*solo_counts,*common_counts,*solo_n,*common_n;
 int2 *solo_tasks,*common_tasks;
 unsigned*words;int word_count;
 int nq,nk,max_records,columns,pages,solo_task_cap,common_task_cap;
};
struct Param {float guard,lo0,lo1,raw,out,tau;int start,end,hint;};
struct Args {
 const uint4 *q,*k,*fine;const float *ks,*w,*fine_scale,*meta,*raw,*out,*tau;
 const uint32_t*rho;const int*hint,*starts,*ends,*budget;
 float*values;int*ids,*counts,*error,*packets,*members,*debug;
 unsigned long long*waits;const uint8_t*keep_ref;const float*score_ref;
 int nq,nk,cap,seed;
};
__device__ __forceinline__ uint32_t trusted_keep_mask(float value,float tau,
    uint32_t eligible,uint32_t bit){
 uint32_t mask;
 asm volatile("{.reg .pred finite,below,active,keep; .reg .f32 magnitude; "
   "abs.f32 magnitude,%1; setp.lt.f32 finite,magnitude,0f7f800000; "
   "setp.lt.f32 below,%1,%2; and.pred keep,finite,below; not.pred keep,keep; "
   "setp.ne.u32 active,%3,0; and.pred keep,keep,active; selp.u32 %0,%4,0,keep;}"
   :"=r"(mask):"f"(value),"f"(tau),"r"(eligible),"r"(bit));
 return mask;
}
template<int Count>__device__ __forceinline__ void load_tmem(const uint32_t address,float*out){
 using Loader=std::conditional_t<Count==64,cute::SM100_TMEM_LOAD_32dp32b64x,
              std::conditional_t<Count==32,cute::SM100_TMEM_LOAD_32dp32b32x,
              std::conditional_t<Count==16,cute::SM100_TMEM_LOAD_32dp32b16x,
              std::conditional_t<Count==8,cute::SM100_TMEM_LOAD_32dp32b8x,
              std::conditional_t<Count==4,cute::SM100_TMEM_LOAD_32dp32b4x,cute::SM100_TMEM_LOAD_32dp32b2x>>>>>;
 [&]<size_t...I>(cute::index_sequence<I...>){Loader::copy(address,reinterpret_cast<uint32_t*>(out)[I]...);}(cute::make_index_sequence<Count>{});
}
}
