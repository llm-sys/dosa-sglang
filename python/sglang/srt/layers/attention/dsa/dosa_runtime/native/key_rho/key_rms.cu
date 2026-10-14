#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <climits>
namespace key_rms {
#include "../common/rms_integer.inl"

__global__ void rms_qualified_kernel(const uint32_t* __restrict__ codes,const uint32_t* __restrict__ scale_bits,uint32_t* __restrict__ sidecar,int64_t n) {
  const int64_t row=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)>>5;
  const int lane=threadIdx.x&31;
  if(row>=n)return;
  const uint32_t packed=codes[row*32+lane];
  uint32_t invalid=0,block_code=max4_magnitude(packed,invalid);
  unsigned long long sum_square=sum4_square_units(packed);
  #pragma unroll
  for(int delta=16;delta;delta>>=1){
    block_code=max(block_code,__shfl_down_sync(0xffffffffu,block_code,delta,32));
    invalid|=__shfl_down_sync(0xffffffffu,invalid,delta,32);
    sum_square+=__shfl_down_sync(0xffffffffu,sum_square,delta,32);
  }
  if(lane==0){invalid|=!((scale_bits[row]-1u)<0x5f800000u);sidecar[row]=invalid?0x7fu:exact_rms_code(sum_square,block_code,128);}
}
__global__ void rms_live_qualified_kernel(const uint32_t*codes,const uint32_t*scale_bits,uint32_t*sidecar,int nk_capacity,const int*live_nk,int seed){
 const int nk=*live_nk;if(nk<seed||nk>nk_capacity)return;
 const int lane=threadIdx.x&31;
 for(int64_t row=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)>>5;row<nk;row+=(int64_t(gridDim.x)*blockDim.x)>>5){
  const uint32_t packed=codes[row*32+lane];
  uint32_t invalid=0,block_code=max4_magnitude(packed,invalid);
  unsigned long long sum_square=sum4_square_units(packed);
  #pragma unroll
  for(int delta=16;delta;delta>>=1){
    block_code=max(block_code,__shfl_down_sync(0xffffffffu,block_code,delta,32));
    invalid|=__shfl_down_sync(0xffffffffu,invalid,delta,32);
    sum_square+=__shfl_down_sync(0xffffffffu,sum_square,delta,32);
  }
  if(lane==0){invalid|=!((scale_bits[row]-1u)<0x5f800000u);sidecar[row]=invalid?0x7fu:exact_rms_code(sum_square,block_code,128);}
 }
}
void rms_qualified(torch::Tensor codes,torch::Tensor scale,torch::Tensor sidecar){
 TORCH_CHECK(codes.is_cuda()&&sidecar.is_cuda()&&scale.is_cuda()&&codes.device()==sidecar.device()&&codes.device()==scale.device(),"same CUDA device required");
 TORCH_CHECK(codes.is_contiguous()&&codes.scalar_type()==torch::kUInt8&&codes.dim()==2&&codes.size(1)==128,"codes: contiguous uint8[N,128]");
 TORCH_CHECK(scale.is_contiguous()&&scale.scalar_type()==torch::kFloat32&&scale.dim()==1&&scale.numel()==codes.size(0),"scale: contiguous F32[N]");
 TORCH_CHECK(sidecar.is_contiguous()&&sidecar.scalar_type()==torch::kInt32&&sidecar.dim()==1&&sidecar.numel()==codes.size(0),"rho: contiguous I32[N]");
 TORCH_CHECK(reinterpret_cast<uintptr_t>(codes.data_ptr())%4==0,"packed E4 alignment");
 int64_t n=codes.size(0);TORCH_CHECK(n<=INT32_MAX,"too many keys");if(!n)return;
 c10::cuda::CUDAGuard guard(codes.device());
 rms_qualified_kernel<<<(n+3)/4,128,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint32_t*>(codes.data_ptr<uint8_t>()),reinterpret_cast<const uint32_t*>(scale.data_ptr<float>()),reinterpret_cast<uint32_t*>(sidecar.data_ptr<int32_t>()),n);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void rms_live_nk_qualified(torch::Tensor codes,torch::Tensor scale,torch::Tensor sidecar,torch::Tensor live_nk,int64_t sms,int64_t seed){
 TORCH_CHECK(codes.is_cuda()&&sidecar.is_cuda()&&live_nk.is_cuda()&&codes.device()==sidecar.device()&&codes.device()==live_nk.device(),"live RMS device");
 TORCH_CHECK(codes.is_contiguous()&&codes.scalar_type()==torch::kUInt8&&codes.dim()==2&&codes.size(1)==128&&codes.size(0)>0&&codes.size(0)<=(1<<20),"physical E4 K capacity");
 TORCH_CHECK(sidecar.is_contiguous()&&sidecar.scalar_type()==torch::kInt32&&sidecar.dim()==1&&sidecar.numel()==codes.size(0),"physical rho capacity");
 TORCH_CHECK(live_nk.is_contiguous()&&live_nk.scalar_type()==torch::kInt32&&live_nk.dim()==1&&live_nk.numel()==1&&sms>0&&sms<65536&&seed>=0&&seed<=codes.size(0),"live RMS scalar/grid");
 TORCH_CHECK(reinterpret_cast<uintptr_t>(codes.data_ptr())%4==0,"E4 packed alignment");
 TORCH_CHECK(scale.is_cuda()&&scale.device()==codes.device()&&scale.is_contiguous()&&scale.scalar_type()==torch::kFloat32&&scale.dim()==1&&scale.numel()==codes.size(0),"scale: contiguous F32 capacity");
 c10::cuda::CUDAGuard guard(codes.device());
 rms_live_qualified_kernel<<<4*sms,128,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint32_t*>(codes.data_ptr<uint8_t>()),reinterpret_cast<const uint32_t*>(scale.data_ptr<float>()),reinterpret_cast<uint32_t*>(sidecar.data_ptr<int32_t>()),int(codes.size(0)),live_nk.data_ptr<int>(),int(seed));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("rms_qualified",&key_rms::rms_qualified);
 m.def("rms_live_nk_qualified",&key_rms::rms_live_nk_qualified);
 m.def("key_rho_contract",[](){return "rms128-ks-positive-le2p64-v1";});
}
