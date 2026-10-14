// Exact original Trusted keep predicate consumed directly by the warp vote.
__device__ __forceinline__ uint32_t h18_keep_ballot(float value,float tau,uint32_t eligible){
 uint32_t mask;
 asm volatile("{.reg .pred finite,below,active,keep; .reg .f32 magnitude; "
  "abs.f32 magnitude,%1; setp.lt.f32 finite,magnitude,0f7f800000; "
  "setp.lt.f32 below,%1,%2; and.pred keep,finite,below; not.pred keep,keep; "
  "setp.ne.u32 active,%3,0; and.pred keep,keep,active; "
  "vote.ballot.sync.b32 %0,keep,0xffffffff;}"
  :"=r"(mask):"f"(value),"f"(tau),"r"(eligible));
 return mask;
}
// Two math WGs split eight queries across the SAME 128-key MMA tile.
// The full 64-column fragment is read before either WG starts scalar reduction.
struct alignas(1024) F2LiveH18Storage {
 static constexpr int H=18;
 static constexpr int W=(H+3)/4*4;
 alignas(1024) uint8_t fine[8*H*128];alignas(16) float fw[8][W];
 alignas(1024) uint8_t raw_k[8][128*128];float raw_scale[8][128];uint32_t raw_rho[8][128];
 Param row[8];Barrier full_q,full_kv[8],empty_kv[8],full_umma[4],empty_umma[4];uint32_t tmem;
};
template<bool Emit=false,bool Trusted=false,bool Indexed=false,bool LiveNK=false>__global__ __launch_bounds__(384,1) void f2_live_h18_kernel(
 const __grid_constant__ Args a,const __grid_constant__ cute::TmaDescriptor mq,
 const __grid_constant__ cute::TmaDescriptor mw,const __grid_constant__ cute::TmaDescriptor mk,
 const __grid_constant__ cute::TmaDescriptor ms,const __grid_constant__ cute::TmaDescriptor mr,const __grid_constant__ PagedArgs page,const int32_t* live_nk){
 constexpr int H=18,N=8*H,W=(H+3)/4*4,Stages=3;int t=threadIdx.x,g=blockIdx.x,warp=t/32,lane=t&31;
 int logical_nk=a.nk,active_word_count=page.word_count;
 if constexpr(LiveNK){
  logical_nk=__ldg(live_nk);
  if(logical_nk<a.seed||logical_nk>a.nk){
   if(blockIdx.x==0)for(int r=t;r<a.nq;r+=blockDim.x)atomicOr(a.error+r,1);
   return;
  }
  active_word_count=(logical_nk-a.seed+31)/32;
 }

 int first_block=0,total_blocks=(logical_nk-a.seed+127)/128,blocks=total_blocks;
 if constexpr(Indexed){
  int count=__ldg(a.counts),parts=__ldg(a.counts+1),groups=a.nq/8;
  bool malformed=count<0||count>groups||parts<1||parts>max(total_blocks,1)||int64_t(count)*parts>gridDim.x;
  if(malformed){if(blockIdx.x==0)for(int r=t;r<a.nq;r+=blockDim.x)atomicOr(a.error+r,1);return;}
  if(blockIdx.x==0){
   bool bad=false;
   for(int qg=t;qg<groups;qg+=blockDim.x){
    int b0=__ldg(a.budget+qg*2),b1=__ldg(a.budget+qg*2+1),pos=__ldg(a.members+qg);
    if(!dosa_geometry::valid_certificate_budget(b0))for(int j=0;j<4;++j)atomicOr(a.error+qg*8+j,1);
    if(!dosa_geometry::valid_certificate_budget(b1))for(int j=4;j<8;++j)atomicOr(a.error+qg*8+j,1);
    bool active=b0==16||b1==16;
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
 else {first_block=total_blocks*int(blockIdx.y)/int(gridDim.y);blocks=total_blocks*int(blockIdx.y+1)/int(gridDim.y)-first_block;}
 bool eligible=false;
 #pragma unroll
 for(int j=0;j<2;++j){int b=a.budget[g*2+j];if(!dosa_geometry::valid_certificate_budget(b)){if(t<4)atomicOr(a.error+(g*2+j)*4+t,1);}eligible|=b==H-2;}
 if(!eligible)return;
 if constexpr(LiveNK){if(total_blocks==0)return;}
 extern __shared__ __align__(1024) uint8_t memory[];auto&s=*reinterpret_cast<F2LiveH18Storage*>(memory);
 if(t<8){int r=g*8+t;Param p;p.guard=a.meta[r*64+H];p.lo0=a.fine_scale[r*H];p.lo1=a.fine_scale[r*H+1];p.raw=a.raw[r];p.out=a.out[r];p.tau=a.hint[r]==2?a.tau[r]:__uint_as_float(0xff800000u);p.start=a.starts[r];p.end=a.ends[r];p.hint=Trusted?2:a.hint[r];s.row[t]=p;}
 if(warp==8){cute::prefetch_tma_descriptor(&mq);cute::prefetch_tma_descriptor(&mw);cute::prefetch_tma_descriptor(&mk);cute::prefetch_tma_descriptor(&ms);cute::prefetch_tma_descriptor(&mr);}
 if(warp==8&&cute::elect_one_sync()){
  s.full_q.init(1);
  #pragma unroll
  for(int i=0;i<8;++i){s.full_kv[i].init(1);s.empty_kv[i].init(256);}
  cutlass::arch::fence_barrier_init();
 }
 if(warp==9){
  if(cute::elect_one_sync()){
   #pragma unroll
   for(int i=0;i<Stages;++i){s.full_umma[i].init(1);s.empty_umma[i].init(256);}
   cutlass::arch::fence_barrier_init();
  }
  cute::TMEM::Allocator1Sm().allocate(512,&s.tmem);cute::TMEM::Allocator1Sm().release_allocation_lock();
 }
 asm volatile("fence.proxy.async.shared::cta;":::"memory");__syncthreads();
 if(warp==8){
  cutlass::arch::warpgroup_reg_dealloc<40>();
  if(cute::elect_one_sync()){
   deep_gemm::tma::copy<128,N,128>(&mq,&s.full_q,s.fine,0,g*N);
   deep_gemm::tma::copy<W,8,0>(&mw,&s.full_q,&s.fw[0][0],0,g*8);
   s.full_q.arrive_and_expect_tx(8*H*128+8*W*4);
   for(int block=0;block<blocks;++block){int stage=block%8,phase=(block/8)&1;
    s.empty_kv[stage].wait(phase^1);
    deep_gemm::tma::copy<128,128,128>(&mk,&s.full_kv[stage],s.raw_k[stage],0,a.seed+(first_block+block)*128);
    deep_gemm::tma::copy<128,1,0>(&ms,&s.full_kv[stage],s.raw_scale[stage],a.seed+(first_block+block)*128,0);
    deep_gemm::tma::copy<128,1,0>(&mr,&s.full_kv[stage],s.raw_rho[stage],a.seed+(first_block+block)*128,0);
    s.full_kv[stage].arrive_and_expect_tx(128*128+2*128*4);
   }
  }
 }else if(warp==9){
  cutlass::arch::warpgroup_reg_dealloc<40>();
  auto instruction=cute::UMMA::make_instr_desc<cutlass::float_e4m3_t,cutlass::float_e4m3_t,float,128,N,cute::UMMA::Major::K,cute::UMMA::Major::K>();
  auto runtime=cute::UMMA::make_runtime_instr_desc(instruction);s.full_q.wait(0);
  for(int block=0;block<blocks;++block){int stage=block%8,phase=(block/8)&1,u=block%Stages,up=(block/Stages)&1;
   s.full_kv[stage].wait(phase);s.empty_umma[u].wait(up^1);deep_gemm::ptx::tcgen05_after_thread_sync();
   #pragma unroll
   for(int k=0;k<4;++k){
    auto ad=deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K,0,128,128>(reinterpret_cast<cutlass::float_e4m3_t*>(s.raw_k[stage]),0,k*32);
    auto bd=deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K,0,128,128>(reinterpret_cast<cutlass::float_e4m3_t*>(s.fine),0,k*32);
    cute::SM100_MMA_F8F6F4_SS::fma(ad,bd,s.tmem+u*N,k,runtime);
   }
   cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&s.full_umma[u]));
  }
 }else if(warp>=10){cutlass::arch::warpgroup_reg_dealloc<40>();}
 else{
  cutlass::arch::warpgroup_reg_alloc<232>();
  int wg=t/128,kt=t%128,group=g*2+wg;int kept[5]={};float weights[4][H];Param resident[4];bool eligible_q[4];
  s.full_q.wait(0);
  #pragma unroll
  for(int qr=0;qr<4;++qr){resident[qr]=s.row[wg*4+qr];eligible_q[qr]=a.budget[group]==H-2;
   #pragma unroll
   for(int h=0;h<H;++h)weights[qr][h]=deep_gemm::ptx::ld_shared(&s.fw[wg*4+qr][h]);
  }
  const uint32_t query_tmem_base=s.tmem+wg*(4*H);
  // Preserve Q4/lane ownership; form a byte address once and advance64B/K128.
  uint64_t word_address=reinterpret_cast<uint64_t>(page.words)
   +(uint64_t(group)*uint64_t(page.word_count)+uint64_t(first_block)*4u+uint64_t(warp&3))*16u+uint64_t(lane)*4u;
  for(int block=0;block<blocks;++block){int stage=block%8,phase=(block/8)&1,u=block%Stages,up=(block/Stages)&1;
   s.full_kv[stage].wait(phase);
   float scale=deep_gemm::ptx::ld_shared(&s.raw_scale[stage][kt]);uint32_t code=deep_gemm::ptx::ld_shared(&s.raw_rho[stage][kt]);
   s.full_umma[u].wait(up);deep_gemm::ptx::tcgen05_after_thread_sync();s.empty_kv[stage].arrive();
   float acc[4*H];uint32_t address=query_tmem_base+u*N;load_tmem<64>(address,acc);cutlass::arch::fence_view_async_tmem_load();
   if constexpr(H==18){load_tmem<8>(address+64,acc+64);cutlass::arch::fence_view_async_tmem_load();}
   deep_gemm::ptx::tcgen05_before_thread_sync();s.empty_umma[u].arrive();
   __half_raw half=__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(code),__NV_E4M3);float rho=__half2float(static_cast<__half>(half));
   int key=a.seed+(first_block+block)*128+kt;bool keyvalid=true;if constexpr(!Trusted)keyvalid=key<logical_nk&&code<127&&isfinite(scale)&&scale>0;uint32_t m=0,word_value=0;
   #pragma unroll
   for(int i=0;i<4;++i){int qr=i;
    float2 s0=make_float2(0,0),s1=make_float2(0,0);
    #pragma unroll
    for(int h=0;h<H-2;h+=4){
     s0=__ffma2_rn(make_float2(fabsf(acc[i*H+h+2]),fabsf(acc[i*H+h+3])),make_float2(weights[qr][h+2],weights[qr][h+3]),s0);
     if(h+2<H-2)s1=__ffma2_rn(make_float2(fabsf(acc[i*H+h+4]),fabsf(acc[i*H+h+5])),make_float2(weights[qr][h+4],weights[qr][h+5]),s1);
    }
    float2 pair=__fadd2_rn(s0,s1);float paid_sum=__fadd_rn(pair.x,pair.y);const Param&p=resident[qr];
    float lo0=__fmul_rn(acc[i*H],p.lo0);float neg=__fadd_rn(0.0f,__fmaf_rn(acc[i*H+1],p.lo1,lo0));
    float signed_score=__fmaf_rn(paid_sum,0.5f,-neg);
    float tail=__fmaf_ru(rho,p.guard,__fadd_ru(signed_score,p.raw));
    float value=__fmaf_ru(scale,tail,p.out);
    if(!(code<127u&&isfinite(neg)))value=__uint_as_float(0x7f800000u);
    bool visible=key<logical_nk&&key>=p.start&&key<p.end;
    if constexpr(Trusted){
     if constexpr(Emit){
      unsigned ballot=h18_keep_ballot(value,p.tau,uint32_t(eligible_q[qr])&uint32_t(visible));
      if(lane==qr)word_value=ballot;
     }else{
      const uint32_t bit=trusted_keep_mask(value,p.tau,uint32_t(eligible_q[qr])&uint32_t(visible),1u<<qr);
      m|=bit;kept[qr]+=int(bit!=0);
     }
    }else{
     bool keep=eligible_q[qr]&&visible&&(!keyvalid||p.hint!=2||!isfinite(value)||!(value<p.tau));
     m|=keep?1u<<qr:0;if constexpr(!Emit)kept[qr]+=int(keep);
    }
   }
   if constexpr(Emit){
    int word=(first_block+block)*4+(warp&3);
    if constexpr(!Trusted){
     #pragma unroll
     for(int qr=0;qr<4;++qr){unsigned mask=__ballot_sync(0xffffffffu,((m>>qr)&1u)!=0);if(lane==qr)word_value=mask;}
    }
    if(lane<4&&eligible_q[0]&&word<active_word_count)*reinterpret_cast<uint32_t*>(word_address)=word_value;
    word_address+=64u;
   }else kept[4]+=int(m==15);
   if constexpr(!Emit)if(a.keep_ref&&key<logical_nk&&m!=a.keep_ref[uint64_t(group)*a.nk+key])atomicAdd(a.debug,1);
  }
  if constexpr(!Emit){
  #pragma unroll
  for(int qr=0;qr<5;++qr){int total=kept[qr];
   #pragma unroll
   for(int d=16;d;d>>=1)total+=__shfl_down_sync(0xffffffffu,total,d);
   if(lane==0)atomicAdd(a.packets+group*5+qr,total);
  }
  }
 }
 __syncthreads();if(warp==9)cute::TMEM::Allocator1Sm().free(s.tmem,512);
}
