#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
using torch::Tensor;
constexpr int Limit=8, Segment=32768, Top=2048;
__global__ void gather_kernel(const float*leaf,const int*selected,const int*lengths,int chunks,const int*rows,const int*state,int*finite,float*merge,int*ids){
 if(*state>Limit)return;
 int slot=blockIdx.x/chunks,seg=blockIdx.x%chunks;
 if(rows[slot]<0)return;
 // The finite mask is published by a prior kernel. A later selected-ID fault
 // only rejects this row; all remaining writes are bounded temporary data.
 if(!finite[slot])return;
 int length=lengths[blockIdx.x];
 for(int j=threadIdx.x;j<Top;j+=blockDim.x){
  int i=selected[int64_t(blockIdx.x)*Top+j];
  int64_t out=int64_t(slot)*(chunks*Top)+seg*Top+j;
  if(i>=0&&i<length){merge[out]=leaf[int64_t(blockIdx.x)*Segment+i];ids[out]=seg*Segment+i;}
  else {merge[out]=-INFINITY;ids[out]=-1;if(i!=-1||length>=Top)atomicExch(finite+slot,0);}
 }
}
__global__ void final_meta_kernel(const int*counts,int nq,int chunks,const int*rows,const int*state,const int*finite,int*merge_lengths,int*fallback_lengths){
 int r=blockIdx.x*blockDim.x+threadIdx.x;
 if(r<Limit)merge_lengths[r]=(*state<=Limit&&rows[r]>=0&&finite[r])?chunks*Top:0;
 if(r>=nq)return;
 bool fast=false;
 if(*state<=Limit)for(int i=0;i<Limit;++i)fast|=rows[i]==r&&finite[i];
 fallback_lengths[r]=fast?0:counts[r];
}
__global__ void publish_kernel(const int*ids,const int*selected,int width,const int*rows,const int*state,const int*finite,int*out){
 int slot=blockIdx.x;
 if(*state>Limit||rows[slot]<0||!finite[slot])return;
 for(int j=threadIdx.x;j<Top;j+=blockDim.x){
  int i=selected[slot*Top+j];
  // Original FI on >=2048 finite candidates guarantees this range and a
  // nonnegative selected ID. Retain a bounds guard before any indexed load.
  out[int64_t(rows[slot])*Top+j]=(i>=0&&i<width)?ids[int64_t(slot)*width+i]:-1;
 }
}
void tensors(std::initializer_list<Tensor> xs,torch::ScalarType type,const Tensor&anchor){
 for(const auto&x:xs)TORCH_CHECK(x.is_cuda()&&x.device()==anchor.device()&&x.scalar_type()==type&&x.is_contiguous(),"same-device contiguous tensor contract");
}
void gather(Tensor leaf,Tensor selected,Tensor lengths,Tensor rows,Tensor state,Tensor finite,Tensor merge,Tensor ids){
 c10::cuda::CUDAGuard guard(leaf.device());tensors({leaf,merge},torch::kFloat32,leaf);tensors({selected,lengths,rows,state,finite,ids},torch::kInt32,leaf);
 int chunks=leaf.size(0)/Limit;TORCH_CHECK(leaf.dim()==2&&leaf.size(1)==Segment&&chunks>0&&chunks<=32&&selected.sizes()==torch::IntArrayRef({Limit*chunks,Top})&&merge.sizes()==torch::IntArrayRef({Limit,chunks*Top})&&ids.sizes()==merge.sizes(),"merge shapes");
 gather_kernel<<<Limit*chunks,256,0,at::cuda::getCurrentCUDAStream()>>>(leaf.data_ptr<float>(),selected.data_ptr<int>(),lengths.data_ptr<int>(),chunks,rows.data_ptr<int>(),state.data_ptr<int>(),finite.data_ptr<int>(),merge.data_ptr<float>(),ids.data_ptr<int>());C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void final_meta(Tensor counts,Tensor rows,Tensor state,Tensor finite,Tensor merge_lengths,Tensor fallback_lengths,int64_t chunks){
 c10::cuda::CUDAGuard guard(counts.device());tensors({counts,rows,state,finite,merge_lengths,fallback_lengths},torch::kInt32,counts);
 TORCH_CHECK(merge_lengths.numel()==Limit&&fallback_lengths.numel()==counts.numel(),"final lengths");
 final_meta_kernel<<<(std::max<int64_t>(counts.numel(),Limit)+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(counts.data_ptr<int>(),counts.numel(),chunks,rows.data_ptr<int>(),state.data_ptr<int>(),finite.data_ptr<int>(),merge_lengths.data_ptr<int>(),fallback_lengths.data_ptr<int>());C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void publish(Tensor ids,Tensor selected,Tensor rows,Tensor state,Tensor finite,Tensor output){
 c10::cuda::CUDAGuard guard(ids.device());tensors({ids,selected,rows,state,finite,output},torch::kInt32,ids);
 TORCH_CHECK(ids.dim()==2&&ids.size(0)==Limit&&selected.sizes()==torch::IntArrayRef({Limit,Top})&&output.dim()==2&&output.size(1)==Top,"publish shapes");
 publish_kernel<<<Limit,256,0,at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(),selected.data_ptr<int>(),ids.size(1),rows.data_ptr<int>(),state.data_ptr<int>(),finite.data_ptr<int>(),output.data_ptr<int>());C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather", &gather);
    m.def("final_meta", &final_meta);
    m.def("publish", &publish);
}
