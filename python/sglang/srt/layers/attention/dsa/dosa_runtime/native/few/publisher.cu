#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <climits>
__global__ void final_publish(const int*ids,const int*selected,const int64_t*order,const int*error,int*out,int cap,int64_t id_stride,int64_t out_stride){
 int row=blockIdx.x;
 if(error[row]!=0)return;
 int64_t destination=order[row];
 for(int j=threadIdx.x;j<2048;j+=blockDim.x){
  int col=max(0,min(cap-1,selected[int64_t(row)*2048+j]));
  out[destination*out_stride+j]=ids[int64_t(row)*id_stride+col];
 }
}
__global__ void replay_publish(const int*selected,const int64_t*order,const int*error,int*out,int64_t out_stride){
 int row=blockIdx.x;
 if(error[row]==0)return;
 int64_t destination=order[row];
 for(int j=threadIdx.x;j<2048;j+=blockDim.x)out[destination*out_stride+j]=selected[int64_t(row)*2048+j];
}
void contract(torch::Tensor selected,torch::Tensor order,torch::Tensor error,torch::Tensor output){
 TORCH_CHECK(selected.is_cuda()&&selected.scalar_type()==torch::kInt32&&selected.is_contiguous()&&selected.dim()==2&&selected.size(1)==2048,"selected contiguous CUDA I32 Qx2048");
 TORCH_CHECK(order.device()==selected.device()&&order.scalar_type()==torch::kInt64&&order.is_contiguous()&&order.numel()==selected.size(0),"original-query permutation slice");
 TORCH_CHECK(error.device()==selected.device()&&error.scalar_type()==torch::kInt32&&error.is_contiguous()&&error.numel()==selected.size(0),"query status slice");
 TORCH_CHECK(output.device()==selected.device()&&output.scalar_type()==torch::kInt32&&output.dim()==2&&output.size(1)==2048&&output.stride(1)==1&&output.stride(0)>=2048,"original-query output");
 // order is the caller's existing argsort permutation or its contiguous slice;
 // its values name distinct valid rows of output. This is not a new sort.
 TORCH_CHECK(selected.size(0)>0&&selected.size(0)<=INT_MAX,"supported query batch");
}
void normal(torch::Tensor ids,torch::Tensor selected,torch::Tensor order,torch::Tensor error,torch::Tensor output){
 c10::cuda::CUDAGuard guard(selected.device());contract(selected,order,error,output);
 TORCH_CHECK(ids.device()==selected.device()&&ids.scalar_type()==torch::kInt32&&ids.dim()==2&&ids.size(0)==selected.size(0)&&ids.size(1)>0&&ids.size(1)<=INT_MAX&&ids.stride(1)==1&&ids.stride(0)>=0,"candidate logical-ID rows");
 final_publish<<<selected.size(0),256,0,at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(),selected.data_ptr<int>(),order.data_ptr<int64_t>(),error.data_ptr<int>(),output.data_ptr<int>(),ids.size(1),ids.stride(0),output.stride(0));C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void recovered(torch::Tensor selected,torch::Tensor order,torch::Tensor error,torch::Tensor output){
 c10::cuda::CUDAGuard guard(selected.device());contract(selected,order,error,output);
 replay_publish<<<selected.size(0),256,0,at::cuda::getCurrentCUDAStream()>>>(selected.data_ptr<int>(),order.data_ptr<int64_t>(),error.data_ptr<int>(),output.data_ptr<int>(),output.stride(0));C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("normal",&normal);m.def("recovered",&recovered);}
