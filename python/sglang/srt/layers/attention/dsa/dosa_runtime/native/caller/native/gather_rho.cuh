#pragma once
// Same raw-byte RMS; no scale is applied to its value. Invalid scale joins
// the current-key bad flag, so a finite rho also certifies 0<KS<=2^64.
#include "../../common/rms_integer.inl"

__global__ void paged_gather_rho_kernel(
    const char* __restrict__ cache, char* __restrict__ dst_k,
    char* __restrict__ dst_scale, const int* __restrict__ table,
    const int* __restrict__ permutation, int64_t block_stride,
    int page_size, int n, int* __restrict__ rho) {
  const int row=blockIdx.x*blockDim.y+threadIdx.y, x=threadIdx.x;
  const bool valid=row<n;
  int src=x==0 && valid?permutation[row]:-1;
  src=__shfl_sync(0xffffffffu,src,0,8);
  unsigned long long square=0;
  uint32_t bad=0,kappa=0;
  if(valid){
    if(src<0 || src>=n){asm volatile("trap;");return;}
    const int block=table[src/page_size];
    const int64_t base=int64_t(block)*block_stride;
    const int within=src%page_size;
    const uint4 words=*reinterpret_cast<const uint4*>(cache+base+within*128+x*16);
    *reinterpret_cast<uint4*>(dst_k+int64_t(row)*128+x*16)=words;
    const uint32_t v[4]={words.x,words.y,words.z,words.w};
#pragma unroll
    for(int i=0;i<4;++i){
      kappa=max(kappa,max4_magnitude(v[i],bad));
      square+=sum4_square_units(v[i]);
    }
    if(x==0){
      const uint32_t scale_bits=*reinterpret_cast<const uint32_t*>(cache+base+page_size*128+within*4);
      *reinterpret_cast<uint32_t*>(dst_scale+int64_t(row)*4)=scale_bits;
      // Positive F32 bit order: accepts subnormal through2^64 inclusive.
      bad|=!((scale_bits-1u)<0x5f800000u);
    }
  }
#pragma unroll
  for(int delta=4;delta;delta>>=1){
    square+=__shfl_down_sync(0xffffffffu,square,delta,8);
    kappa=max(kappa,__shfl_down_sync(0xffffffffu,kappa,delta,8));
    bad|=__shfl_down_sync(0xffffffffu,bad,delta,8);
  }
  if(x==0 && valid)rho[row]=bad?127:exact_rms_code(square,kappa,128);
}
void check_rho_output(const torch::Tensor& k,const torch::Tensor& rho){
  TORCH_CHECK(k.size(1)==128,"rho gather supports original E4M3 D128 only");
  TORCH_CHECK(rho.is_cuda() && rho.device()==k.device() &&
    rho.scalar_type()==torch::kInt32 && rho.is_contiguous() &&
    rho.dim()==1 && rho.numel()==k.size(0),"rho: contiguous I32[N] same device");
}
void launch_gather_rho(const torch::Tensor& cache,const torch::Tensor& k,
    const torch::Tensor& scale,const torch::Tensor& table,
    const torch::Tensor& permutation,const torch::Tensor& rho,cudaStream_t stream){
  const int n=k.size(0);if(!n)return;
  paged_gather_rho_kernel<<<(n+31)/32,dim3(8,32),0,stream>>>(
    reinterpret_cast<const char*>(cache.data_ptr<uint8_t>()),
    reinterpret_cast<char*>(k.data_ptr<uint8_t>()),
    reinterpret_cast<char*>(scale.data_ptr<uint8_t>()),table.data_ptr<int>(),
    permutation.data_ptr<int>(),cache.stride(0),cache.size(1),n,rho.data_ptr<int>());
}

inline constexpr auto plan_and_permuted_paged_gather_rho_out =
    plan_and_gather<check_rho_output, launch_gather_rho, torch::Tensor>;
