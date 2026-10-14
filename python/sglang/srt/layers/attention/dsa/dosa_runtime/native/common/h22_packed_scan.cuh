// Fixed V1: Q4 K amortization, two raw stages, one UMMA slot perK128mathWG.
// record: 40half2 /2DDexp /raw,out,start,endbase /2endoffs /4tau /4guard /flags.
struct alignas(1024) PackedStorage {
 alignas(1024)uint8_t fine[4*kScanHeads*128];alignas(16)uint32_t record[kRecordWords];
 alignas(16)uint2 decoded_weights[kWeightWords];
 alignas(1024)uint8_t raw_k[2][256*128];float raw_scale[2][256];uint32_t raw_rho[2][256];
 Barrier full_q,full_coeff,full_kv[2],empty_kv[2],full_umma[2],empty_umma[2];uint32_t tmem;
};
__device__ __forceinline__ float2 decode_bf16_pair(uint32_t bits){
 float2 result;
 // Exact BF16 payload reconstruction, including BF16 subnormals. These
 // volatile bit operations retain40 packed words across the K loop; no
 // F16 conversion/FMA is needed to restore the two independent F32 values.
 uint32_t lo,hi;
 asm volatile("{.reg .b32 packed; mov.b32 packed,%2; shl.b32 %0,packed,16; and.b32 %1,packed,0xffff0000;}"
   :"=r"(lo),"=r"(hi):"r"(bits));
 result.x=__uint_as_float(lo);result.y=__uint_as_float(hi);
 return result;
}
// Two exact F32 bit patterns loaded together from the CTA coefficient cache.
__device__ __forceinline__ float2 load_decoded_pair(const uint2* value){
 uint32_t lo,hi;uint32_t address=uint32_t(__cvta_generic_to_shared(value));
 asm volatile("ld.shared.v2.b32 {%0,%1},[%2];" : "=r"(lo),"=r"(hi) : "r"(address) : "memory");
 return make_float2(__uint_as_float(lo),__uint_as_float(hi));
}
template<int Q>__device__ __forceinline__ uint32_t consume_packed_query(
 const uint32_t start,int buf,PackedStorage&s,
 const uint32_t(&dd)[2],const uint32_t(&end_delta)[2],const float(&tau)[4],const float(&guard)[4],
 float raw,float out,int visible_start,int end_base,float scale,float rho,int key,int nk){
 // Current query only: issue coefficient loads before the three TMEM waits.
 float2 current_weights[kWeightPairs];
 #pragma unroll
 for(int h=0;h<kWeightPairs;++h)current_weights[h]=load_decoded_pair(s.decoded_weights+Q*kWeightPairs+h);
 float acc[kScanHeads];uint32_t address=start+Q*kScanHeads;
 load_tmem<16>(address,acc);cutlass::arch::fence_view_async_tmem_load();
 if constexpr(kScanHeads==30){
  load_tmem<8>(address+16,acc+16);cutlass::arch::fence_view_async_tmem_load();
  load_tmem<4>(address+24,acc+24);cutlass::arch::fence_view_async_tmem_load();
  load_tmem<2>(address+28,acc+28);cutlass::arch::fence_view_async_tmem_load();
 }else{
  load_tmem<4>(address+16,acc+16);cutlass::arch::fence_view_async_tmem_load();
  load_tmem<2>(address+20,acc+20);cutlass::arch::fence_view_async_tmem_load();
 }
 if constexpr(Q==3){deep_gemm::ptx::tcgen05_before_thread_sync();s.empty_umma[buf].arrive();}
 float2 s0=make_float2(0,0),s1=make_float2(0,0);
 #pragma unroll
 for(int h=0;h<kHighPaid;h+=4){
  float2 w0=current_weights[h/2];
  s0=__ffma2_rn(__fadd2_rn(make_float2(acc[h+2],acc[h+3]),make_float2(fabsf(acc[h+2]),fabsf(acc[h+3]))),w0,s0);
  float2 w1=current_weights[h/2+1];
  s1=__ffma2_rn(__fadd2_rn(make_float2(acc[h+4],acc[h+5]),make_float2(fabsf(acc[h+4]),fabsf(acc[h+5]))),w1,s1);
 }
 float2 pair=__fadd2_rn(s0,s1);float paid=__fmul_rn(__fadd_rn(pair.x,pair.y),0.5f);
 uint32_t ex=dd[Q/2]>>((Q%2)*16);
 float lo_scale0=__uint_as_float((ex&255u)<<23),lo_scale1=__uint_as_float(((ex>>8)&255u)<<23);
 float lo0=__fmul_rn(acc[0],lo_scale0);float neg=__fadd_rn(0.0f,__fmaf_rn(acc[1],lo_scale1,lo0));
 float signed_score=__fadd_rn(paid,-fmaxf(neg,0.0f));float guard_term=__fmul_rn(rho,guard[Q]);
 float u=__fadd_rn(__fmul_rn(scale,__fadd_rn(__fadd_rn(signed_score,raw),guard_term)),out);
 int visible_end=end_base+int((end_delta[Q/2]>>((Q%2)*16))&65535u);
 bool visible=key<nk&&key>=visible_start&&key<visible_end;
 // RU-BF16 monotonicity uses this finite keyscale domain. Domain failure
 // keeps the visible pair; all TMEM/slot lifecycle remains unconditional.
 bool scale_admitted=isfinite(scale)&&scale>0.f&&scale<=0x1p64f;
 return !scale_admitted ? (visible ? (1u<<Q) : 0u) : trusted_keep_mask(u,tau[Q],uint32_t(visible),1u<<Q);
}
template<bool LiveNK=false>__global__ __launch_bounds__(384,2) void packed_kernel(
 const __grid_constant__ Args a,const __grid_constant__ cute::TmaDescriptor mq,
 const __grid_constant__ cute::TmaDescriptor mp,const __grid_constant__ cute::TmaDescriptor mk,
 const __grid_constant__ cute::TmaDescriptor ms,const __grid_constant__ cute::TmaDescriptor mr,
 const __grid_constant__ PagedArgs page,const uint32_t* records,const int32_t* live_nk){
 int t=threadIdx.x,g=blockIdx.x,warp=t/32,lane=t&31;
 int logical_nk=a.nk,active_word_count=page.word_count;
 if constexpr(LiveNK){
  logical_nk=__ldg(live_nk);
  if(logical_nk<a.seed||logical_nk>a.nk){
   if(blockIdx.x==0&&blockIdx.y==0)for(int r=t;r<a.nq;r+=blockDim.x)atomicOr(a.error+r,1);
   return;
  }
  active_word_count=(logical_nk-a.seed+31)/32;
 }

 if(a.budget[g]!=kHighPaid)return;
 if(records[uint64_t(g)*kRecordWords+kWeightWords+16]!=1u){if(t<4)atomicOr(a.error+g*4+t,1);return;}
 if constexpr(LiveNK){if(logical_nk==a.seed)return;}
 extern __shared__ __align__(1024) uint8_t memory[];auto&s=*reinterpret_cast<PackedStorage*>(memory);
 if(warp==8){cute::prefetch_tma_descriptor(&mq);cute::prefetch_tma_descriptor(&mp);cute::prefetch_tma_descriptor(&mk);cute::prefetch_tma_descriptor(&ms);cute::prefetch_tma_descriptor(&mr);}
 if(warp==8&&cute::elect_one_sync()){
  s.full_q.init(1);s.full_coeff.init(1);
  #pragma unroll
  for(int i=0;i<2;++i){s.full_kv[i].init(1);s.empty_kv[i].init(256);}
  cutlass::arch::fence_barrier_init();
 }
 if(warp==9){
  if(cute::elect_one_sync()){
   #pragma unroll
   for(int i=0;i<2;++i){s.full_umma[i].init(1);s.empty_umma[i].init(128);}
   cutlass::arch::fence_barrier_init();
  }
  cute::TMEM::Allocator1Sm().allocate(256,&s.tmem);cute::TMEM::Allocator1Sm().release_allocation_lock();
 }
 asm volatile("fence.proxy.async.shared::cta;":::"memory");__syncthreads();
 int total=(logical_nk-a.seed+255)/256,first=total*int(blockIdx.y)/8,stop=total*int(blockIdx.y+1)/8,blocks=stop-first;
 if(warp>=8)cutlass::arch::warpgroup_reg_dealloc<40>();
 if(warp==8){
  if(cute::elect_one_sync()){
   deep_gemm::tma::copy<128,kTileColumns,128>(&mq,&s.full_q,s.fine,0,g*kTileColumns);
   deep_gemm::tma::copy<kRecordWords,1,0>(&mp,&s.full_q,s.record,0,g);
   s.full_q.arrive_and_expect_tx(kTileColumns*128+kRecordWords*4);
   for(int block=0;block<blocks;++block){int stage=block%2,phase=(block/2)&1;
    s.empty_kv[stage].wait(phase^1);int offset=a.seed+(first+block)*256;
    deep_gemm::tma::copy<128,256,128>(&mk,&s.full_kv[stage],s.raw_k[stage],0,offset);
    deep_gemm::tma::copy<256,1,0>(&ms,&s.full_kv[stage],s.raw_scale[stage],offset,0);
    deep_gemm::tma::copy<256,1,0>(&mr,&s.full_kv[stage],s.raw_rho[stage],offset,0);
    s.full_kv[stage].arrive_and_expect_tx(34816);
   }
  }
 }else if(warp==9||warp==10){
  int wg=warp-9;auto instruction=cute::UMMA::make_instr_desc<cutlass::float_e4m3_t,cutlass::float_e4m3_t,float,128,kTileColumns,cute::UMMA::Major::K,cute::UMMA::Major::K>();auto runtime=cute::UMMA::make_runtime_instr_desc(instruction);
  s.full_q.wait(0);
  for(int block=0;block<blocks;++block){int stage=block%2,phase=(block/2)&1,up=block&1;
   s.full_kv[stage].wait(phase);s.empty_umma[wg].wait(up^1);deep_gemm::ptx::tcgen05_after_thread_sync();
   #pragma unroll
   for(int k=0;k<4;++k){
    auto ad=deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K,0,128,128>(reinterpret_cast<cutlass::float_e4m3_t*>(s.raw_k[stage]),wg*128,k*32);
    auto bd=deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K,0,128,128>(reinterpret_cast<cutlass::float_e4m3_t*>(s.fine),0,k*32);
    cute::SM100_MMA_F8F6F4_SS::fma(ad,bd,s.tmem+wg*kTileColumns,k,runtime);
   }
   cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&s.full_umma[wg]));
  }
 }else if(warp==11){
  s.full_q.wait(0);
  for(int i=lane;i<kWeightWords;i+=32){
   uint32_t bits=deep_gemm::ptx::ld_shared(s.record+i);
   s.decoded_weights[i]=make_uint2(bits<<16,bits&0xffff0000u);
  }
  __syncwarp();if(cute::elect_one_sync())s.full_coeff.arrive();
 }else if(warp<8){
  cutlass::arch::warpgroup_reg_alloc<96>();
  // Same stable scalar epoch, reloaded once AFTER the dynamic register
  // allocation. Do not carry startup NK across the donation boundary.
  int math_nk=a.nk;
  if constexpr(LiveNK){
   asm volatile("ld.global.nc.u32 %0,[%1];":"=r"(math_nk):"l"(live_nk):"memory");
  }
  const int math_word_count=LiveNK?(math_nk-a.seed+31)/32:page.word_count;

  int wg=t/128;s.full_coeff.wait(0);
  const uint32_t math_tmem_base=s.tmem+wg*kTileColumns;
  uint32_t dd[2],end_delta[2];float tau[4],guard[4];
  #pragma unroll
  for(int i=0;i<2;++i){dd[i]=deep_gemm::ptx::ld_shared(s.record+kWeightWords+0+i);end_delta[i]=deep_gemm::ptx::ld_shared(s.record+kWeightWords+6+i);}
  #pragma unroll
  for(int i=0;i<4;++i){tau[i]=__uint_as_float(deep_gemm::ptx::ld_shared(s.record+kWeightWords+8+i));guard[i]=__uint_as_float(deep_gemm::ptx::ld_shared(s.record+kWeightWords+12+i));}
  float raw=__uint_as_float(deep_gemm::ptx::ld_shared(s.record+kWeightWords+2)),out=__uint_as_float(deep_gemm::ptx::ld_shared(s.record+kWeightWords+3));
  int start=int(deep_gemm::ptx::ld_shared(s.record+kWeightWords+4)),end_base=int(deep_gemm::ptx::ld_shared(s.record+kWeightWords+5));
  for(int block=0;block<blocks;++block){int stage=block%2,phase=(block/2)&1,up=block&1;
   s.full_kv[stage].wait(phase);float scale=deep_gemm::ptx::ld_shared(&s.raw_scale[stage][t]);uint32_t code=deep_gemm::ptx::ld_shared(&s.raw_rho[stage][t])&255u;
   s.full_umma[wg].wait(up);deep_gemm::ptx::tcgen05_after_thread_sync();s.empty_kv[stage].arrive();
   __half_raw half=__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(code),__NV_E4M3);float rho=__half2float(static_cast<__half>(half));
   int key=a.seed+(first+block)*256+t;uint32_t address=math_tmem_base,m=0;
   m|=consume_packed_query<0>(address,wg,s,dd,end_delta,tau,guard,raw,out,start,end_base,scale,rho,key,math_nk);
   asm volatile("" : "+r"(m) :: "memory");
   m|=consume_packed_query<1>(address,wg,s,dd,end_delta,tau,guard,raw,out,start,end_base,scale,rho,key,math_nk);
   asm volatile("" : "+r"(m) :: "memory");
   m|=consume_packed_query<2>(address,wg,s,dd,end_delta,tau,guard,raw,out,start,end_base,scale,rho,key,math_nk);
   asm volatile("" : "+r"(m) :: "memory");
   m|=consume_packed_query<3>(address,wg,s,dd,end_delta,tau,guard,raw,out,start,end_base,scale,rho,key,math_nk);
   int word=(first+block)*8+warp;unsigned value=0;
   #pragma unroll
   for(int qr=0;qr<4;++qr){unsigned mask=__ballot_sync(0xffffffffu,(m&(1u<<qr))!=0);if(lane==qr)value=mask;}
   if(lane<4&&word<math_word_count)page.words[(int64_t(g)*page.word_count+word)*4+lane]=value;
  }
 }
 __syncthreads();if(warp==9)cute::TMEM::Allocator1Sm().free(s.tmem,256);
}
