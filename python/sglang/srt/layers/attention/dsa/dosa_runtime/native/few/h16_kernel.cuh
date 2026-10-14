// All32 lanes reach both query ballots, including inactive/invisible keys.
// This is the same qualified_keep_mask predicate, directly consumed by vote.
__device__ __forceinline__ uint32_t qualified_keep_ballot(float value,float tau,
    uint32_t eligible,uint32_t qualified){
 uint32_t ballot;
 asm volatile("{.reg .pred finite,below,active,valid,keep; .reg .f32 magnitude; "
   "abs.f32 magnitude,%1; setp.lt.f32 finite,magnitude,0f7f800000; "
   "setp.lt.f32 below,%1,%2; and.pred keep,finite,below; "
   "setp.ne.u32 valid,%4,0; and.pred keep,keep,valid; not.pred keep,keep; "
   "setp.ne.u32 active,%3,0; and.pred keep,keep,active; "
   "vote.ballot.sync.b32 %0,keep,0xffffffff;}"
   :"=r"(ballot):"f"(value),"f"(tau),"r"(eligible),"r"(qualified));
 return ballot;
}
// Requires freshly prepared abs-shift-dd2-v1 fields; original H16 geometry.
// Four math WGs own2queries each. OriginalK256/raw4/twoUMA256; creditafterkh1read32.
struct alignas(1024) K256FourWGStorage {
 alignas(1024) uint8_t fine[8*16*128];alignas(16) float fw[8][16];
 alignas(1024) uint8_t raw_k[4][256*128];float raw_scale[4][256];uint32_t raw_rho[4][256];
 Param row[8];Barrier full_q,full_kv[4],empty_kv[4],full_umma[2],empty_umma[2];uint32_t tmem;
};
__device__ __forceinline__ void h16_load_tmem32(uint32_t address,float*out){
 [&]<size_t...I>(cute::index_sequence<I...>){cute::SM100_TMEM_LOAD_32dp32b32x::copy(address,reinterpret_cast<uint32_t*>(out)[I]...);}(cute::make_index_sequence<32>{});
}
// Equivalent to replacing a failed-qualified value with +Inf before the
// original keep mask; finite(value) is evaluated once. NaN/tie/NF keep.
__device__ __forceinline__ uint32_t qualified_keep_mask(float value,float tau,
    uint32_t eligible,uint32_t qualified,uint32_t bit){
 uint32_t mask;
 asm volatile("{.reg .pred finite,below,active,valid,keep; .reg .f32 magnitude; "
   "abs.f32 magnitude,%1; setp.lt.f32 finite,magnitude,0f7f800000; "
   "setp.lt.f32 below,%1,%2; and.pred keep,finite,below; "
   "setp.ne.u32 valid,%4,0; and.pred keep,keep,valid; not.pred keep,keep; "
   "setp.ne.u32 active,%3,0; and.pred keep,keep,active; selp.u32 %0,%5,0,keep;}"
   :"=r"(mask):"f"(value),"f"(tau),"r"(eligible),"r"(qualified),"r"(bit));
 return mask;
}
// Same represented ABS/DD core, reduced in four RN chains. The retained
// direct planner SCREEN (80-round envelope) covers these six-round chains.
// Mixed weights are prepared once outside the K loop; no new query fields.
__device__ __forceinline__ void h16_abs_core(const float*acc,const float weights[2][16],const Param resident[2],float core[2],uint32_t&neg_valid){
 float2 s0[2]={make_float2(0,0),make_float2(0,0)},s1[2]={make_float2(0,0),make_float2(0,0)};
 #pragma unroll
 for(int h=0;h<16;h+=4){
  #pragma unroll
  for(int qr=0;qr<2;++qr){
   float2 a0=h==0?make_float2(acc[qr*16],acc[qr*16+1]):make_float2(fabsf(acc[qr*16+h]),fabsf(acc[qr*16+h+1]));
   float2 a1=make_float2(fabsf(acc[qr*16+h+2]),fabsf(acc[qr*16+h+3]));
   s0[qr]=__ffma2_rn(a0,make_float2(weights[qr][h],weights[qr][h+1]),s0[qr]);
   s1[qr]=__ffma2_rn(a1,make_float2(weights[qr][h+2],weights[qr][h+3]),s1[qr]);
  }
 }
 #pragma unroll
 for(int qr=0;qr<2;++qr){
  float2 pair=__fadd2_rn(s0[qr],s1[qr]);core[qr]=__fadd_rn(pair.x,pair.y);
 }
 // Valid field scales and finite E4 raw dots imply finite DD. Bad/nonfinite
 // arithmetic still fails the unchanged final finite-value keep predicate.
 neg_valid=3u;
}
template<bool Emit=false,bool Trusted=false,bool Indexed=false,bool LiveNK=false>__global__ __launch_bounds__(640,1) void k256_four_math_kernel(
 const __grid_constant__ Args a,const __grid_constant__ cute::TmaDescriptor mq,
 const __grid_constant__ cute::TmaDescriptor mw,const __grid_constant__ cute::TmaDescriptor mk,
 const __grid_constant__ cute::TmaDescriptor ms,const __grid_constant__ cute::TmaDescriptor mr,const __grid_constant__ PagedArgs page,const int32_t* live_nk){
 static_assert(Emit,"Four-WG prototype exposes production words only");constexpr int H=16;int t=threadIdx.x,g=blockIdx.x,warp=t/32,lane=t&31;
 static_assert(!LiveNK || (Emit && Trusted && Indexed),"LiveNK onlysupports trustedIndexedwords");
 int logical_nk=a.nk,active_word_count=page.word_count;
 if constexpr(LiveNK){
  logical_nk=__ldg(live_nk);
  if(logical_nk<a.seed||logical_nk>a.nk){
   if(blockIdx.x==0)for(int r=t;r<a.nq;r+=blockDim.x)atomicOr(a.error+r,1);
   return;
  }
  active_word_count=(logical_nk-a.seed+31)/32;
 }
 int first_block=0,total_blocks=(logical_nk-a.seed+255)/256,blocks=total_blocks;
 if constexpr(Indexed){
  int count=__ldg(a.counts),parts=__ldg(a.counts+1),groups=a.nq/8;
  bool malformed=count<0||count>groups||parts<1||parts>max(total_blocks,1)||int64_t(count)*parts>gridDim.x;
  if(malformed){if(blockIdx.x==0)for(int r=t;r<a.nq;r+=blockDim.x)atomicOr(a.error+r,1);return;}
  if(blockIdx.x==0){
   bool bad=false;
   for(int qg=t;qg<groups;qg+=blockDim.x){
    int b0=__ldg(a.budget+qg*2),b1=__ldg(a.budget+qg*2+1),pos=__ldg(a.members+qg);
    if(b0!=12&&!dosa_geometry::valid_certificate_budget(b0))for(int j=0;j<4;++j)atomicOr(a.error+qg*8+j,1);
    if(b1!=12&&!dosa_geometry::valid_certificate_budget(b1))for(int j=4;j<8;++j)atomicOr(a.error+qg*8+j,1);
    bool active=b0==14||b1==14;
    if(active){if(pos<0||pos>=count)bad=true;else if(__ldg(a.ids+pos)!=qg)bad=true;}
    else if(pos!=-1)bad=true;
   }
   if(__syncthreads_or(bad))for(int r=t;r<a.nq;r+=blockDim.x)atomicOr(a.error+r,1);
  }
  int work=blockIdx.x;if(work>=count*parts)return;int pos=work/parts,part=work%parts;g=__ldg(a.ids+pos);
  if(g<0||g>=groups){for(int r=t;r<a.nq;r+=blockDim.x)atomicOr(a.error+r,1);return;}
  if(__ldg(a.members+g)!=pos){for(int r=t;r<a.nq;r+=blockDim.x)atomicOr(a.error+r,1);return;}
  first_block=total_blocks*part/parts;blocks=total_blocks*(part+1)/parts-first_block;
 }
 bool eligible=false;
 #pragma unroll
 for(int j=0;j<2;++j){int b=a.budget[g*2+j];if(!dosa_geometry::valid_certificate_budget(b)&&(!Indexed||b!=12)){if(t<4)atomicOr(a.error+(g*2+j)*4+t,1);}eligible|=b==14;}
 if(!eligible)return;
 if constexpr(LiveNK){if(total_blocks==0)return;}
 extern __shared__ __align__(1024) uint8_t memory[];auto&s=*reinterpret_cast<K256FourWGStorage*>(memory);
 if(t<8){int r=g*8+t;Param p;p.guard=a.meta[r*64+16];p.lo0=a.fine_scale[r*16];p.lo1=a.fine_scale[r*16+1];p.raw=a.raw[r];p.out=a.out[r];p.tau=a.tau[r];p.start=a.starts[r];p.end=a.ends[r];p.hint=a.hint[r];s.row[t]=p;}
 if(warp==16){cute::prefetch_tma_descriptor(&mq);cute::prefetch_tma_descriptor(&mw);cute::prefetch_tma_descriptor(&mk);cute::prefetch_tma_descriptor(&ms);cute::prefetch_tma_descriptor(&mr);}
 if(warp==16&&cute::elect_one_sync()){
  s.full_q.init(1);
  #pragma unroll
  for(int i=0;i<4;++i){s.full_kv[i].init(1);s.empty_kv[i].init(512);}
  cutlass::arch::fence_barrier_init();
 }
 if(warp==17){
  if(cute::elect_one_sync()){
   #pragma unroll
   for(int i=0;i<2;++i){s.full_umma[i].init(1);s.empty_umma[i].init(512);}
   cutlass::arch::fence_barrier_init();
  }
  cute::TMEM::Allocator1Sm().allocate(512,&s.tmem);cute::TMEM::Allocator1Sm().release_allocation_lock();
 }
 asm volatile("fence.proxy.async.shared::cta;":::"memory");__syncthreads();
 if(warp>=16)cutlass::arch::warpgroup_reg_dealloc<32>();
 if(warp==16){
  if(cute::elect_one_sync()){
   deep_gemm::tma::copy<128,128,128>(&mq,&s.full_q,s.fine,0,g*128);
   deep_gemm::tma::copy<16,8,0>(&mw,&s.full_q,&s.fw[0][0],0,g*8);
   s.full_q.arrive_and_expect_tx(8*16*128+8*16*4);
   for(int block=0;block<blocks;++block){int stage=block%4,phase=(block/4)&1;
    s.empty_kv[stage].wait(phase^1);
    deep_gemm::tma::copy<128,256,128>(&mk,&s.full_kv[stage],s.raw_k[stage],0,a.seed+(first_block+block)*256);
    deep_gemm::tma::copy<256,1,0>(&ms,&s.full_kv[stage],s.raw_scale[stage],a.seed+(first_block+block)*256,0);
    deep_gemm::tma::copy<256,1,0>(&mr,&s.full_kv[stage],s.raw_rho[stage],a.seed+(first_block+block)*256,0);
    s.full_kv[stage].arrive_and_expect_tx(256*128+2*256*4);
   }
  }
 }else if(warp==17){
  auto instruction=cute::UMMA::make_instr_desc<cutlass::float_e4m3_t,cutlass::float_e4m3_t,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::K>();
  auto runtime=cute::UMMA::make_runtime_instr_desc(instruction);s.full_q.wait(0);
  for(int block=0;block<blocks;++block){int stage=block%4,phase=(block/4)&1,u=block%2,up=(block/2)&1;
   s.full_kv[stage].wait(phase);s.empty_umma[u].wait(up^1);deep_gemm::ptx::tcgen05_after_thread_sync();
   #pragma unroll
   for(int kh=0;kh<2;++kh){
    #pragma unroll
    for(int k=0;k<4;++k){
     auto ad=deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K,0,128,128>(reinterpret_cast<cutlass::float_e4m3_t*>(s.raw_k[stage]),kh*128,k*32);
     auto bd=deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K,0,128,128>(reinterpret_cast<cutlass::float_e4m3_t*>(s.fine),0,k*32);
     cute::SM100_MMA_F8F6F4_SS::fma(ad,bd,s.tmem+u*256+kh*128,k,runtime);
    }
   }
   cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&s.full_umma[u]));
  }
 }else if(warp>=18){}
 else{
  cutlass::arch::warpgroup_reg_alloc<112>();
  // Same-epoch NK is immutable. Reload after register donation so it does not
  // spill across the startup / specialization boundary or inside the K loop.
  int math_nk=a.nk;
  if constexpr(LiveNK)asm volatile("ld.global.nc.u32 %0,[%1];":"=r"(math_nk):"l"(live_nk):"memory");
  const int math_word_count=LiveNK?(math_nk-a.seed+31)/32:page.word_count;
  int wg=t/128,kt=t%128,group=g*2+(wg/2),channel=(wg&1)*2;float weights[2][16];Param resident[2];bool eligible_q[2];
  s.full_q.wait(0);
  #pragma unroll
  for(int qr=0;qr<2;++qr){resident[qr]=s.row[wg*2+qr];eligible_q[qr]=a.budget[group]==14;
   #pragma unroll
   for(int h=0;h<16;++h)weights[qr][h]=h==0?-resident[qr].lo0:(h==1?-resident[qr].lo1:__fmul_rn(0.5f,deep_gemm::ptx::ld_shared(&s.fw[wg*2+qr][h])));
  }
  // Original Q4/channel address, advanced eight physical words per K256.
  // Non-writing lanes form integers only; the original lane/bounds gate
  // dominates every conversion back to a dereferenced pointer.
  uint64_t word_address=reinterpret_cast<uint64_t>(page.words)
    +(uint64_t(group)*uint64_t(page.word_count)+uint64_t(first_block)*8u+uint64_t(warp&3))*16u
    +uint64_t(channel+lane)*4u;
  // Rebuild the immutable loop extent after register redistribution, like
  // math_nk above. Do not carry the producer's blocks value across setmaxnreg.
  int math_blocks=(math_nk-a.seed+255)/256;
  if constexpr(Indexed){
   int math_parts=__ldg(a.counts+1),math_part=blockIdx.x%math_parts;
   math_blocks=math_blocks*(math_part+1)/math_parts-math_blocks*math_part/math_parts;
  }
  #pragma unroll 2
  for(int block=0;block<math_blocks;++block){int stage=block%4,phase=(block/4)&1,u=block%2,up=(block/2)&1;
   s.full_kv[stage].wait(phase);
   float scales[2]={deep_gemm::ptx::ld_shared(&s.raw_scale[stage][kt]),deep_gemm::ptx::ld_shared(&s.raw_scale[stage][128+kt])};
   uint32_t codes[2]={deep_gemm::ptx::ld_shared(&s.raw_rho[stage][kt]),deep_gemm::ptx::ld_shared(&s.raw_rho[stage][128+kt])};
   s.full_umma[u].wait(up);deep_gemm::ptx::tcgen05_after_thread_sync();s.empty_kv[stage].arrive();
   float first_core[2];uint32_t first_neg_valid;
   {
    float first_acc[32];h16_load_tmem32(s.tmem+u*256+wg*32,first_acc);cutlass::arch::fence_view_async_tmem_load();
    h16_abs_core(first_acc,weights,resident,first_core,first_neg_valid);
   }
   // Only two scores plus the original finite-DD bits remain from kh0.
   // Both TMEM halves are retired before granting the original 512 credits.
   float second_acc[32];h16_load_tmem32(s.tmem+u*256+128+wg*32,second_acc);cutlass::arch::fence_view_async_tmem_load();
   deep_gemm::ptx::tcgen05_before_thread_sync();s.empty_umma[u].arrive();
   #pragma unroll
   for(int kh=0;kh<2;++kh){float scale=scales[kh];uint32_t code=codes[kh];
    float core[2];uint32_t neg_valid;
    if(kh==0){core[0]=first_core[0];core[1]=first_core[1];neg_valid=first_neg_valid;}
    else h16_abs_core(second_acc,weights,resident,core,neg_valid);
   __half_raw half=__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(code),__NV_E4M3);float rho=__half2float(static_cast<__half>(half));
   int key=a.seed+(first_block+block)*256+kh*128+kt;bool keyvalid=key<math_nk&&code<127u;uint32_t ballots[2];
   #pragma unroll
   for(int qr=0;qr<2;++qr){
    const Param&p=resident[qr];float signed_score=core[qr];bool finite_neg=(neg_valid>>qr)&1u;
    float tail=__fmaf_ru(rho,p.guard,__fadd_ru(signed_score,p.raw));
    float value=__fmaf_ru(scale,tail,p.out);
    bool visible=key<math_nk&&key>=p.start&&key<p.end;
    if constexpr(Trusted){
     uint32_t qualified=uint32_t(keyvalid&&p.hint==2&&finite_neg);
     ballots[qr]=qualified_keep_ballot(value,p.tau,uint32_t(eligible_q[qr])&uint32_t(visible),qualified);
    }else{
     if(!(keyvalid&&p.hint==2&&finite_neg&&isfinite(value)))value=CUDART_INF_F;
     ballots[qr]=__ballot_sync(0xffffffffu,eligible_q[qr]&&visible&&(!keyvalid||p.hint!=2||!isfinite(value)||!(value<p.tau)));
    }
   }
   int word=(first_block+block)*8+kh*4+(warp&3);unsigned value=lane==0?ballots[0]:ballots[1];
   // Two WGs share a Q4 word: this WG owns only its two channels.
   if(lane<2&&eligible_q[0]&&word<math_word_count)*reinterpret_cast<uint32_t*>(word_address+uint64_t(kh)*64u)=value;
   }
   word_address+=128u;
  }
 }
 __syncthreads();if(warp==17)cute::TMEM::Allocator1Sm().free(s.tmem,512);
}
