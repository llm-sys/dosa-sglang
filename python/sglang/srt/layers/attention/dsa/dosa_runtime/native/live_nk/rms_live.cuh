#pragma once
#include <cuda_fp8.h>
namespace live_record5_rms {
// Exact integer arithmetic copied from certicascade-rms4-append-c170.
// Finite E4 magnitudes are magnitude_unit(code)/512; sign is irrelevant.
#include "../common/rms_integer.inl"

__global__ void rms_kernel(const uint32_t*codes,uint32_t*sidecar,int nk_capacity,const int*live_nk,int seed){
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
  if(lane==0)sidecar[row]=invalid?0x7fu:exact_rms_code(sum_square,block_code,128);
 }
}
}
namespace live_record5_host {
void rms_live_nk(Tensor codes,Tensor sidecar,Tensor live_nk,int64_t sms,int64_t seed){
 TORCH_CHECK(codes.is_cuda()&&sidecar.is_cuda()&&live_nk.is_cuda()&&codes.device()==sidecar.device()&&codes.device()==live_nk.device(),"live RMS device");
 TORCH_CHECK(codes.is_contiguous()&&codes.scalar_type()==torch::kUInt8&&codes.dim()==2&&codes.size(1)==128&&codes.size(0)>0&&codes.size(0)<=(1<<20),"physical E4 K capacity");
 TORCH_CHECK(sidecar.is_contiguous()&&sidecar.scalar_type()==torch::kInt32&&sidecar.dim()==1&&sidecar.numel()==codes.size(0),"physical rho capacity");
 TORCH_CHECK(live_nk.is_contiguous()&&live_nk.scalar_type()==torch::kInt32&&live_nk.dim()==1&&live_nk.numel()==1&&sms>0&&sms<65536&&seed>=0&&seed<=codes.size(0),"live RMS scalar/grid");
 TORCH_CHECK(reinterpret_cast<uintptr_t>(codes.data_ptr())%4==0,"E4 packed alignment");
 c10::cuda::CUDAGuard guard(codes.device());
 live_record5_rms::rms_kernel<<<4*sms,128,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint32_t*>(codes.data_ptr<uint8_t>()),reinterpret_cast<uint32_t*>(sidecar.data_ptr<int32_t>()),int(codes.size(0)),live_nk.data_ptr<int>(),int(seed));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
