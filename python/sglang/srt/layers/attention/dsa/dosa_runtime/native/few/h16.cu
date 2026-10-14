#include "../common/emit_host.cuh"
#include "../common/query_capacity.cuh"
#include <math_constants.h>
#include <limits>
#define DOSA_FEW_NAMESPACE f1_abs_qfma
#include "emit_types.cuh"
#undef DOSA_FEW_NAMESPACE
namespace f1_abs_qfma {
#include "h16_kernel.cuh"
}
template<bool LiveNK>
void f1_abs_qfma_launch(int paid,std::vector<torch::Tensor>x,std::vector<torch::Tensor>streams,
 torch::Tensor pool,torch::Tensor table,torch::Tensor head,torch::Tensor status,torch::Tensor error,int64_t seed,torch::Tensor words,torch::Tensor work_ids,torch::Tensor work_inverse,torch::Tensor work_config,int64_t launch_grid,torch::Tensor live_nk){
 TORCH_CHECK(x.size()==15&&streams.size()==6,"paged ID stream ABI");
 TORCH_CHECK(paid==14,"indexed entry is H16 only");
 if constexpr(LiveNK){
  TORCH_CHECK(live_nk.defined()&&live_nk.is_cuda()&&live_nk.device()==x[0].device()&&live_nk.scalar_type()==torch::kInt32&&live_nk.is_contiguous()&&live_nk.dim()==1&&live_nk.numel()==1,"stable CUDAint32[1] liveNK");
  TORCH_CHECK(x[0].size(0)>0&&x[0].size(0)<=dosa_capacity::kMaxQueryRows&&x[1].size(0)>0&&x[1].size(0)<=1048576,"liveQ/Kcapacity domain");
 }

 for(const auto&t:{work_ids,work_inverse,work_config})TORCH_CHECK(t.is_cuda()&&t.device()==x[0].device()&&t.is_contiguous()&&t.scalar_type()==torch::kInt32,"indexed int32 workspace");
 TORCH_CHECK(x[0].size(0)%8==0&&work_ids.numel()==x[0].size(0)/8&&work_inverse.numel()==work_ids.numel()&&work_config.numel()==2,"indexed descriptor capacities");
 using namespace f1_abs_qfma;
 auto [a, pg] = dosa_emit_host::bind<Args, PagedArgs>(
     paid, x, streams, pool, table, head, status, error, seed, words);
 int nq = a.nq, nk = a.nk;
 c10::cuda::CUDAGuard guard(x[0].device());auto stream=at::cuda::getCurrentCUDAStream();
 a.ids=work_ids.data_ptr<int>();a.members=work_inverse.data_ptr<int>();a.counts=work_config.data_ptr<int>();
 auto mq=make_map(x[5].data_ptr(),CU_TENSOR_MAP_DATA_TYPE_UINT8,{128,(cuuint64_t)nq*16},{128},{128,128},CU_TENSOR_MAP_SWIZZLE_128B);
 auto mw=make_map(x[7].data_ptr(),CU_TENSOR_MAP_DATA_TYPE_FLOAT32,{64,(cuuint64_t)nq},{256},{16,8},CU_TENSOR_MAP_SWIZZLE_NONE);
 auto mk=make_map(x[1].data_ptr(),CU_TENSOR_MAP_DATA_TYPE_UINT8,{128,(cuuint64_t)nk},{128},{128,256},CU_TENSOR_MAP_SWIZZLE_128B);
 auto ms=make_map(x[2].data_ptr(),CU_TENSOR_MAP_DATA_TYPE_FLOAT32,{(cuuint64_t)nk,1},{scale_row_stride(nk)},{256,1},CU_TENSOR_MAP_SWIZZLE_NONE);
 auto mr=make_map(x[3].data_ptr(),CU_TENSOR_MAP_DATA_TYPE_UINT32,{(cuuint64_t)nk,1},{scale_row_stride(nk)},{256,1},CU_TENSOR_MAP_SWIZZLE_NONE);
 TORCH_CHECK(launch_grid>0&&launch_grid<=std::numeric_limits<int>::max(),"explicit launch_grid must bepositive int32");
 int grid=static_cast<int>(launch_grid);
 C10_CUDA_CHECK(cudaFuncSetAttribute(k256_four_math_kernel<true,true,true,LiveNK>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(K256FourWGStorage)));
 C10_CUDA_CHECK(cudaFuncSetAttribute(k256_four_math_kernel<true,true,true,LiveNK>,cudaFuncAttributePreferredSharedMemoryCarveout,cudaSharedmemCarveoutMaxShared));
 k256_four_math_kernel<true,true,true,LiveNK><<<grid,640,sizeof(K256FourWGStorage),stream>>>(a,mq,mw,mk,ms,mr,pg,LiveNK?live_nk.data_ptr<int32_t>():nullptr);C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void f1_abs_qfma_legacy_grid(int paid,std::vector<torch::Tensor>x,std::vector<torch::Tensor>streams,
 torch::Tensor pool,torch::Tensor table,torch::Tensor head,torch::Tensor status,torch::Tensor error,int64_t seed,torch::Tensor words,torch::Tensor work_ids,torch::Tensor work_inverse,torch::Tensor work_config,int64_t launch_grid){
 f1_abs_qfma_launch<false>(paid,x,streams,pool,table,head,status,error,seed,words,work_ids,work_inverse,work_config,launch_grid,torch::Tensor());
}
void f1_abs_qfma_legacy(int paid,std::vector<torch::Tensor>x,std::vector<torch::Tensor>streams,
 torch::Tensor pool,torch::Tensor table,torch::Tensor head,torch::Tensor status,torch::Tensor error,int64_t seed,torch::Tensor words,torch::Tensor work_ids,torch::Tensor work_inverse,torch::Tensor work_config){
 TORCH_CHECK(x.size()==15,"originalx15 ABI");c10::cuda::CUDAGuard guard(x[0].device());
 int grid=std::max(int(x[0].size(0)/8),8*at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
 f1_abs_qfma_legacy_grid(paid,x,streams,pool,table,head,status,error,seed,words,work_ids,work_inverse,work_config,grid);
}
void f1_abs_qfma_trusted_live_k(int paid,std::vector<torch::Tensor>x,std::vector<torch::Tensor>streams,
 torch::Tensor pool,torch::Tensor table,torch::Tensor head,torch::Tensor status,torch::Tensor error,int64_t seed,torch::Tensor words,torch::Tensor work_ids,torch::Tensor work_inverse,torch::Tensor work_config,int64_t launch_grid,torch::Tensor live_nk){
 f1_abs_qfma_launch<true>(paid,x,streams,pool,table,head,status,error,seed,words,work_ids,work_inverse,work_config,launch_grid,live_nk);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("max_query_rows",[](){return dosa_capacity::kMaxQueryRows;});
 m.def("required_key_rho_contract",[](){return "rms128-ks-positive-le2p64-v1";});
 m.def("field_contract",[](){return "abs-shift-dd2-v1";});
 m.def("emit_h16_indexed_trusted",&f1_abs_qfma_legacy);
 m.def("emit_h16_indexed_trusted_grid",&f1_abs_qfma_legacy_grid);
 m.def("emit_h16_indexed_trusted_live_k",&f1_abs_qfma_trusted_live_k);
 m.def("packed_shared_bytes",[](){return sizeof(f1_abs_qfma::K256FourWGStorage);});
}
