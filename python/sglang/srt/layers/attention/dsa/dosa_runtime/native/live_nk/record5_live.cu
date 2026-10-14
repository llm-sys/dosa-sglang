#include "../common/h32_common.cuh"
#include "exact_paged_live.cuh"
using torch::Tensor;
namespace live_record5_seed {
constexpr int Top=2048;
__global__ void validate5(const int* words,int64_t stride,int* counts,
    const float* seed_tau,const float* safe_tau,const int* starts,
    const int* ends,int* error,const int* live_nk,int nk_capacity,int seed) {
 const int nk=*live_nk;
 if(nk<seed||nk>nk_capacity){if(threadIdx.x==0){atomicOr(error+blockIdx.x,1);counts[blockIdx.x]=0;}return;}
 const int row=blockIdx.x,t=threadIdx.x;
 // The original initializer publishes the exact minimum selected seed score.
 // This witness proves the former per-record score >= safe_tau check without
 // retaining or rereading an intermediate selected F32 table.
 if(error[row]!=0||counts[row]!=Top||!isfinite(seed_tau[row])||
     !isfinite(safe_tau[row])||safe_tau[row]>seed_tau[row]||starts[row]<0||
     ends[row]<starts[row]||ends[row]>nk) {
  if(t==0){atomicOr(error+row,1);counts[row]=0;}
  return;
 }
 bool bad=false;
 for(int j=t;j<Top;j+=blockDim.x) {
  const uint32_t word=uint32_t(words[int64_t(row)*stride+j]);
  const uint32_t high=word>>20;
  const int id=int(word&0xFFFFFu);
  // The finite20bit interval has complete low8 bins at both endpoints, so
  // checking its high12 bits is equivalent and does not read the byte plane.
  bad|=high<(h32_record5::Min>>8)||high>(h32_record5::Max>>8)||
      id<starts[row]||id>=ends[row]||id>=nk;
 }
 if(__syncthreads_or(bad)&&t==0){atomicOr(error+row,1);counts[row]=0;}
}
}
namespace live_record5_host {
void validate_seed5_live_nk(Tensor words,Tensor counts,Tensor seed_tau,Tensor safe_tau,
    Tensor starts,Tensor ends,Tensor error,int64_t nk,Tensor live_nk,int64_t seed) {
 c10::cuda::CUDAGuard guard(words.device());
 const std::vector<Tensor> buffers={words,counts,seed_tau,safe_tau,starts,ends,error,live_nk};
 TORCH_CHECK(live_nk.scalar_type()==torch::kInt32&&live_nk.dim()==1&&live_nk.numel()==1&&seed>=0&&seed<=nk,"fixed live NK scalar/seed");
 for(const auto& x:buffers)
  TORCH_CHECK(x.is_cuda()&&x.device()==words.device()&&x.is_contiguous(),
      "record5 seed validation needs same-device contiguous buffers");
 TORCH_CHECK(words.scalar_type()==torch::kInt32&&words.dim()==2&&words.size(0)>0&&
     words.size(1)>=2048&&words.size(1)<=(1<<20)&&nk>0&&nk<=(1<<20),
     "record5 seed integer capacity and key extent");
 const int64_t rows=words.size(0);
 for(const auto& x:{counts,starts,ends,error})
  TORCH_CHECK(x.scalar_type()==torch::kInt32&&x.numel()==rows,"record5 I32 rows");
 for(const auto& x:{seed_tau,safe_tau})
  TORCH_CHECK(x.scalar_type()==torch::kFloat32&&x.numel()==rows,"record5 F32 witness rows");
 for(size_t i=0;i<buffers.size();++i)for(size_t j=i+1;j<buffers.size();++j) {
  const uintptr_t a=reinterpret_cast<uintptr_t>(buffers[i].data_ptr());
  const uintptr_t b=reinterpret_cast<uintptr_t>(buffers[j].data_ptr());
  TORCH_CHECK(!(a<b+buffers[j].numel()*buffers[j].element_size()&&
      b<a+buffers[i].numel()*buffers[i].element_size()),"record5 seed validation operand alias");
 }
 live_record5_seed::validate5<<<rows,256,0,at::cuda::getCurrentCUDAStream()>>>(
     words.data_ptr<int>(),words.stride(0),counts.data_ptr<int>(),seed_tau.data_ptr<float>(),
     safe_tau.data_ptr<float>(),starts.data_ptr<int>(),ends.data_ptr<int>(),error.data_ptr<int>(),live_nk.data_ptr<int>(),int(nk),int(seed));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // live_record5_host

#include "rms_live.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("rms_live_nk",&live_record5_host::rms_live_nk);
 m.attr("RECORD_BYTES")=5;m.attr("SCORE_BITS")=20;
 m.def("validate_seed5_live_nk",&live_record5_host::validate_seed5_live_nk);
 m.def("exact_paged_solo_live_nk",&live_record5_host::exact_paged_completion<1,false,true>);
 m.def("exact_paged_common_live_nk",&live_record5_host::exact_paged_completion<4,false,true>);
}
