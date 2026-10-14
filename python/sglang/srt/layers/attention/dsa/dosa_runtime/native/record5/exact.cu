#include "../common/h32_common.cuh"
#include "../few/exact_paged.cuh"

namespace seed_record5 {
__global__ void pack(const float* values,int64_t stride,int* words,int64_t word_stride,
    uint8_t* low,int64_t low_stride,int* counts,const float* tau,
    const int* starts,const int* ends,int* error,int nk) {
  const int row=blockIdx.x,t=threadIdx.x,n=counts[row];
  const float threshold=tau[row];
  if(error[row]!=0||n!=2048||!isfinite(threshold)||starts[row]<0||
      ends[row]<starts[row]||ends[row]>nk) {
    if(t==0){atomicOr(error+row,1);counts[row]=0;}
    return;
  }
  bool bad=false;
  for(int j=t;j<n;j+=blockDim.x) {
    const float score=values[int64_t(row)*stride+j];
    const int id=words[int64_t(row)*word_stride+j];
    const uint32_t cp=h32_record5::encode(score);
    const bool valid=isfinite(score)&&score>=threshold&&__float_as_uint(score)!=0x80000000u&&
        id>=starts[row]&&id<ends[row]&&id>=0&&id<nk&&uint32_t(id)<0x100000u&&
        cp>=h32_record5::Min&&cp<=h32_record5::Max;
    const uint32_t stored=valid?cp:h32_record5::Invalid;
    words[int64_t(row)*word_stride+j]=int32_t(h32_record5::pack_word(stored,uint32_t(id)));
    low[int64_t(row)*low_stride+j]=uint8_t(stored);
    bad|=!valid;
  }
  if(__syncthreads_or(bad)&&t==0){atomicOr(error+row,1);counts[row]=0;}
}
}

void pack_seed5(torch::Tensor values,torch::Tensor words,torch::Tensor counts,
    torch::Tensor tau,torch::Tensor starts,torch::Tensor ends,torch::Tensor error,
    torch::Tensor low,int64_t nk) {
  c10::cuda::CUDAGuard guard(values.device());
  const std::vector<torch::Tensor> buffers={values,words,counts,tau,starts,ends,error,low};
  for(const auto&x:buffers)
    TORCH_CHECK(x.is_cuda()&&x.device()==values.device()&&x.is_contiguous(),"record5 seed same-device contiguous buffers");
  TORCH_CHECK(values.scalar_type()==torch::kFloat&&values.dim()==2&&values.size(0)>0&&values.size(1)>=2048,
      "original F32 selected seed prefix");
  const int64_t rows=values.size(0);
  TORCH_CHECK(words.scalar_type()==torch::kInt&&words.dim()==2&&words.size(0)==rows&&words.size(1)>=2048&&
      words.size(1)<=(1<<20),"record5 integer row capacity");
  TORCH_CHECK(low.scalar_type()==torch::kByte&&low.sizes()==words.sizes(),"record5 byte plane matches integer plane");
  for(const auto&x:{counts,starts,ends,error})TORCH_CHECK(x.scalar_type()==torch::kInt&&x.numel()==rows,"record5 I32 row metadata");
  TORCH_CHECK(tau.scalar_type()==torch::kFloat&&tau.numel()==rows&&nk>0&&nk<=(1<<20),"record5 F32 thresholds and index extent");
  for(size_t i=0;i<buffers.size();++i)for(size_t j=i+1;j<buffers.size();++j) {
    const uintptr_t a=reinterpret_cast<uintptr_t>(buffers[i].data_ptr());
    const uintptr_t b=reinterpret_cast<uintptr_t>(buffers[j].data_ptr());
    TORCH_CHECK(!(a<b+buffers[j].numel()*buffers[j].element_size()&&
                  b<a+buffers[i].numel()*buffers[i].element_size()),"record5 seed operand alias");
  }
  seed_record5::pack<<<rows,256,0,at::cuda::getCurrentCUDAStream()>>>(values.data_ptr<float>(),values.stride(0),
      words.data_ptr<int>(),words.stride(0),low.data_ptr<uint8_t>(),low.stride(0),counts.data_ptr<int>(),
      tau.data_ptr<float>(),starts.data_ptr<int>(),ends.data_ptr<int>(),error.data_ptr<int>(),int(nk));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.attr("RECORD_BYTES")=5;
  m.attr("SCORE_BITS")=20;
  m.def("pack_seed",&pack_seed5);
  m.def("exact_paged_solo",&exact_paged_completion<1,false,true>);
  m.def("exact_paged_common",&exact_paged_completion<4,false,true>);
}
