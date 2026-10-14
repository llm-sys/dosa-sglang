#include "../common/emit_host.cuh"
#include "../common/query_capacity.cuh"
#include <cuda_fp16.h>
#define DOSA_FEW_NAMESPACE f2_h22_late
#include "emit_types.cuh"
#undef DOSA_FEW_NAMESPACE
namespace f2_h22_late {
#include "../common/h22_pack.cuh"
#include "../common/h22_packed_scan.cuh"
}
namespace f2_h22_late_original {
using namespace f2_h22_late;
#include "../common/h22_scan.cuh"
}
namespace f2_h22_late_host {
using torch::Tensor;
std::pair<f2_h22_late::Args,f2_h22_late::PagedArgs> bind(int paid,const std::vector<Tensor>&x,const std::vector<Tensor>&streams,
 Tensor pool,Tensor table,Tensor head,Tensor status,Tensor error,int64_t seed,Tensor words){
 TORCH_CHECK(x.size()==15&&streams.size()==6,"paged ID stream ABI");
 return dosa_emit_host::bind<f2_h22_late::Args, f2_h22_late::PagedArgs>(
     paid, x, streams, pool, table, head, status, error, seed, words);
}
void record_contract(const Tensor& record,int nq,const Tensor& reference){
 TORCH_CHECK(record.is_cuda()&&record.device()==reference.device()&&record.is_contiguous()&&record.scalar_type()==torch::kInt32&&record.dim()==2&&record.size(0)==nq/4&&record.size(1)==f2_h22_late::kRecordWords,"FP16 V1 I32[Q4,64] record");
}
template<bool LiveNK>void prepare_live(std::vector<Tensor>x,Tensor error,Tensor record,Tensor fast_budget,Tensor slow_budget,int64_t seed,Tensor live_nk){
 if constexpr(LiveNK){
  TORCH_CHECK(live_nk.defined()&&live_nk.is_cuda()&&live_nk.device()==x[0].device()
   &&live_nk.scalar_type()==torch::kInt32&&live_nk.is_contiguous()
   &&live_nk.dim()==1&&live_nk.numel()==1,"stable device int32[1] liveNK");
  TORCH_CHECK(x[0].size(0)>0&&x[0].size(0)<=dosa_capacity::kMaxQueryRows&&x[1].size(0)>0
   &&x[1].size(0)<=1048576,"live Q/K capacity domain");
 }
 if constexpr(LiveNK)TORCH_CHECK(seed>=0&&seed<=x[1].size(0)&&seed%256==0,"aligned capacity seed");
 TORCH_CHECK(x.size()==15,"original x15");int nq=x[0].size(0),nk=x[1].size(0);
 TORCH_CHECK(nq>0&&nq%4==0&&nk>0,"Q4 andK extent");
 for(const auto&t:x)TORCH_CHECK(t.is_cuda()&&t.device()==x[0].device()&&t.is_contiguous(),"original contiguous x15");
 TORCH_CHECK(x[5].scalar_type()==torch::kUInt8&&x[5].numel()==int64_t(nq)*f2_h22_late::kScanHeads*128&&x[6].numel()==int64_t(nq)*f2_h22_late::kScanHeads&&x[7].numel()==int64_t(nq)*96,"H22 planner fields");
 for(int i:{8,9,10,11,12,14})TORCH_CHECK(x[i].numel()==nq,"query fields");
 TORCH_CHECK(x[13].numel()==nq/4,"original Q4budget");record_contract(record,nq,x[0]);
 for(const auto&t:{error,fast_budget,slow_budget})TORCH_CHECK(t.is_cuda()&&t.device()==x[0].device()&&t.is_contiguous()&&t.scalar_type()==torch::kInt32,"int32workspace");
 TORCH_CHECK(error.numel()==nq&&fast_budget.numel()==nq/4&&slow_budget.numel()==nq/4,"workspace size");
 const void* rb=record.storage().data_ptr().get();const void* fb=fast_budget.storage().data_ptr().get();const void* sb=slow_budget.storage().data_ptr().get();
 TORCH_CHECK(rb!=fb&&rb!=sb&&fb!=sb,"independent packedworkspace storage");
 for(const auto&t:x){const void* original=t.storage().data_ptr().get();TORCH_CHECK(rb!=original&&fb!=original&&sb!=original,"workspace must not alias any original input storage");}
 TORCH_CHECK(rb!=error.storage().data_ptr().get()&&fb!=error.storage().data_ptr().get()&&sb!=error.storage().data_ptr().get(),"original error is read-only");
 c10::cuda::CUDAGuard guard(x[0].device());auto stream=at::cuda::getCurrentCUDAStream();
 f2_h22_late::prepare_kernel<LiveNK><<<nq/4,128,0,stream>>>(x[5].data_ptr<uint8_t>(),x[7].data_ptr<float>(),x[6].data_ptr<float>(),x[8].data_ptr<float>(),x[9].data_ptr<float>(),x[10].data_ptr<int>(),x[11].data_ptr<int>(),x[12].data_ptr<int>(),x[13].data_ptr<int>(),x[14].data_ptr<float>(),error.data_ptr<int>(),nq,nk,reinterpret_cast<uint32_t*>(record.data_ptr<int>()),fast_budget.data_ptr<int>(),slow_budget.data_ptr<int>(),int(seed),LiveNK?live_nk.data_ptr<int32_t>():nullptr);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<int H,bool LiveNK>void emit_slow_family(int paid,std::vector<Tensor>x,std::vector<Tensor>streams,
 Tensor pool,Tensor table,Tensor head,Tensor status,Tensor error,int64_t seed,Tensor words,Tensor live_nk){
 if constexpr(LiveNK){
  TORCH_CHECK(live_nk.defined()&&live_nk.is_cuda()&&live_nk.device()==x[0].device()
   &&live_nk.scalar_type()==torch::kInt32&&live_nk.is_contiguous()
   &&live_nk.dim()==1&&live_nk.numel()==1,"stable device int32[1] liveNK");
  TORCH_CHECK(x[0].size(0)>0&&x[0].size(0)<=dosa_capacity::kMaxQueryRows&&x[1].size(0)>0
   &&x[1].size(0)<=1048576,"live Q/K capacity domain");
 }

 TORCH_CHECK(paid==H-2,"original H22 fallback");auto pair=bind(paid,x,streams,pool,table,head,status,error,seed,words);auto a=pair.first;auto pg=pair.second;
 c10::cuda::CUDAGuard guard(x[0].device());auto stream=at::cuda::getCurrentCUDAStream();
 auto mk=f2_h22_late_make_map(x[1].data_ptr(),CU_TENSOR_MAP_DATA_TYPE_UINT8,{128,(cuuint64_t)a.nk},{128},{128,256},CU_TENSOR_MAP_SWIZZLE_128B);
 auto ms=f2_h22_late_make_map(x[2].data_ptr(),CU_TENSOR_MAP_DATA_TYPE_FLOAT32,{(cuuint64_t)a.nk,1},{f2_h22_late_scale_row_stride(a.nk)},{256,1},CU_TENSOR_MAP_SWIZZLE_NONE);
 auto mr=f2_h22_late_make_map(x[3].data_ptr(),CU_TENSOR_MAP_DATA_TYPE_UINT32,{(cuuint64_t)a.nk,1},{f2_h22_late_scale_row_stride(a.nk)},{256,1},CU_TENSOR_MAP_SWIZZLE_NONE);
 auto mq=f2_h22_late_make_map(x[5].data_ptr(),CU_TENSOR_MAP_DATA_TYPE_UINT8,{128,(cuuint64_t)a.nq*H},{128},{128,4*H},CU_TENSOR_MAP_SWIZZLE_128B);
 auto mw=f2_h22_late_make_map(x[7].data_ptr(),CU_TENSOR_MAP_DATA_TYPE_FLOAT32,{96,(cuuint64_t)a.nq},{384},{(H+3)/4*4,4},CU_TENSOR_MAP_SWIZZLE_NONE);
 using namespace f2_h22_late_original;
 C10_CUDA_CHECK(cudaFuncSetAttribute(kernel<H,4,true,true,LiveNK>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Storage<H,4>)));
 kernel<H,4,true,true,LiveNK><<<dim3(a.nq/4,8),384,sizeof(Storage<H,4>),stream>>>(a,mq,mw,mk,ms,mr,pg,LiveNK?live_nk.data_ptr<int32_t>():nullptr);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<bool LiveNK>void emit_slow_live(int paid,std::vector<Tensor>x,std::vector<Tensor>streams,
 Tensor pool,Tensor table,Tensor head,Tensor status,Tensor error,int64_t seed,Tensor words,Tensor live_nk){
 if constexpr(dosa_geometry::Heads==64){
 if(paid==28){
  return emit_slow_family<30,LiveNK>(paid,std::move(x),std::move(streams),pool,table,head,status,error,seed,words,live_nk);
 }
 }
 emit_slow_family<22,LiveNK>(paid,std::move(x),std::move(streams),pool,table,head,status,error,seed,words,live_nk);
}
template<bool LiveNK>void emit_fast_live(int paid,std::vector<Tensor>x,std::vector<Tensor>streams,
 Tensor pool,Tensor table,Tensor head,Tensor status,Tensor error,int64_t seed,Tensor words,Tensor record,Tensor live_nk){
 if constexpr(LiveNK){
  TORCH_CHECK(live_nk.defined()&&live_nk.is_cuda()&&live_nk.device()==x[0].device()
   &&live_nk.scalar_type()==torch::kInt32&&live_nk.is_contiguous()
   &&live_nk.dim()==1&&live_nk.numel()==1,"stable device int32[1] liveNK");
  TORCH_CHECK(x[0].size(0)>0&&x[0].size(0)<=dosa_capacity::kMaxQueryRows&&x[1].size(0)>0
   &&x[1].size(0)<=1048576,"live Q/K capacity domain");
 }

 TORCH_CHECK(paid==f2_h22_late::kHighPaid,"FP16 V1 onlyH22");auto pair=bind(paid,x,streams,pool,table,head,status,error,seed,words);auto a=pair.first;auto pg=pair.second;record_contract(record,a.nq,x[0]);
 TORCH_CHECK(record.storage().data_ptr().get()!=words.storage().data_ptr().get(),"packed record/output storage must not overlap");
 c10::cuda::CUDAGuard guard(x[0].device());auto stream=at::cuda::getCurrentCUDAStream();
 auto mk=f2_h22_late_make_map(x[1].data_ptr(),CU_TENSOR_MAP_DATA_TYPE_UINT8,{128,(cuuint64_t)a.nk},{128},{128,256},CU_TENSOR_MAP_SWIZZLE_128B);
 auto ms=f2_h22_late_make_map(x[2].data_ptr(),CU_TENSOR_MAP_DATA_TYPE_FLOAT32,{(cuuint64_t)a.nk,1},{f2_h22_late_scale_row_stride(a.nk)},{256,1},CU_TENSOR_MAP_SWIZZLE_NONE);
 auto mr=f2_h22_late_make_map(x[3].data_ptr(),CU_TENSOR_MAP_DATA_TYPE_UINT32,{(cuuint64_t)a.nk,1},{f2_h22_late_scale_row_stride(a.nk)},{256,1},CU_TENSOR_MAP_SWIZZLE_NONE);
 auto mq=f2_h22_late_make_map(x[5].data_ptr(),CU_TENSOR_MAP_DATA_TYPE_UINT8,{128,(cuuint64_t)a.nq*f2_h22_late::kScanHeads},{128},{128,f2_h22_late::kTileColumns},CU_TENSOR_MAP_SWIZZLE_128B);
 auto mp=f2_h22_late_make_map(record.data_ptr(),CU_TENSOR_MAP_DATA_TYPE_UINT32,{f2_h22_late::kRecordWords,(cuuint64_t)a.nq/4},{f2_h22_late::kRecordWords*4},{f2_h22_late::kRecordWords,1},CU_TENSOR_MAP_SWIZZLE_NONE);
 C10_CUDA_CHECK(cudaFuncSetAttribute(f2_h22_late::packed_kernel<LiveNK>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(f2_h22_late::PackedStorage)));
 C10_CUDA_CHECK(cudaFuncSetAttribute(f2_h22_late::packed_kernel<LiveNK>,cudaFuncAttributePreferredSharedMemoryCarveout,cudaSharedmemCarveoutMaxShared));
 f2_h22_late::packed_kernel<LiveNK><<<dim3(a.nq/4,8),384,sizeof(f2_h22_late::PackedStorage),stream>>>(a,mq,mp,mk,ms,mr,pg,reinterpret_cast<uint32_t*>(record.data_ptr<int>()),LiveNK?live_nk.data_ptr<int32_t>():nullptr);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
namespace f2_h22_late_host {
// Transfer arguments into the shared static specialization without copying
// Tensor handles or the input vectors. The public signatures stay unchanged.
void prepare_fp16_h22(std::vector<Tensor> x, Tensor error, Tensor record,
                      Tensor fast_budget, Tensor slow_budget) {
  prepare_live<false>(std::move(x), std::move(error), std::move(record),
                      std::move(fast_budget), std::move(slow_budget), 0, Tensor());
}

void emit_trusted_candidates(
    int paid, std::vector<Tensor> x, std::vector<Tensor> streams,
    Tensor pool, Tensor table, Tensor head, Tensor status, Tensor error,
    int64_t seed, Tensor words) {
  emit_slow_live<false>(paid, std::move(x), std::move(streams), std::move(pool),
                        std::move(table), std::move(head), std::move(status),
                        std::move(error), seed, std::move(words), Tensor());
}

void emit_fp16_candidates(
    int paid, std::vector<Tensor> x, std::vector<Tensor> streams,
    Tensor pool, Tensor table, Tensor head, Tensor status, Tensor error,
    int64_t seed, Tensor words, Tensor record) {
  emit_fast_live<false>(paid, std::move(x), std::move(streams), std::move(pool),
                        std::move(table), std::move(head), std::move(status),
                        std::move(error), seed, std::move(words),
                        std::move(record), Tensor());
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("packed_record_words",[](){return f2_h22_late::kRecordWords;});
 m.def("packed_high_budget",[](){return f2_h22_late::kHighPaid;});
 m.def("max_query_rows",[](){return dosa_capacity::kMaxQueryRows;});
 m.def("emit_trusted_candidates",&f2_h22_late_host::emit_trusted_candidates);
 m.def("prepare_fp16_h22",&f2_h22_late_host::prepare_fp16_h22);
 m.def("emit_fp16_candidates",&f2_h22_late_host::emit_fp16_candidates);
 m.def("prepare_fp16_h22_live_k",&f2_h22_late_host::prepare_live<true>);
 m.def("emit_trusted_candidates_live_k",&f2_h22_late_host::emit_slow_live<true>);
 m.def("emit_fp16_candidates_live_k",&f2_h22_late_host::emit_fast_live<true>);
 m.def("packed_shared_bytes",[](){return sizeof(f2_h22_late::PackedStorage);});
}
