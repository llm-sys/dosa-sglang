template<int H,int BQ=4>struct alignas(1024) Storage {
 static constexpr int W=(H+3)/4*4;
 alignas(1024)uint8_t fine[BQ*H*128];alignas(16)float fw[BQ][W];
 alignas(1024)uint8_t raw_k[4][256*128];float raw_scale[4][256];uint32_t raw_rho[4][256];
 Param row[BQ];Barrier full_q,full_kv[4],empty_kv[4],full_umma[4],empty_umma[4];uint32_t tmem;
};
template<int H,int BQ=4,bool Emit=false,bool Trusted=false,bool LiveNK=false>__global__ __launch_bounds__(384,1) void kernel(
 const __grid_constant__ Args a,const __grid_constant__ cute::TmaDescriptor mq,
 const __grid_constant__ cute::TmaDescriptor mw,const __grid_constant__ cute::TmaDescriptor mk,
 const __grid_constant__ cute::TmaDescriptor ms,const __grid_constant__ cute::TmaDescriptor mr,const __grid_constant__ PagedArgs page,const int32_t* live_nk){
 constexpr int N=BQ*H,W=(H+3)/4*4,TM=N*4<=256?256:512;
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

 int basegroup=g*(BQ/4);bool eligible=false;
 #pragma unroll
 for(int j=0;j<BQ/4;++j){int budget=a.budget[basegroup+j];if(!dosa_geometry::valid_certificate_budget(budget)){if(t<4)atomicOr(a.error+(basegroup+j)*4+t,1);}eligible|=budget==H-2;}
 if(!eligible)return;
 if constexpr(LiveNK){if(logical_nk==a.seed)return;}
 extern __shared__ __align__(1024) uint8_t memory[];auto&s=*reinterpret_cast<Storage<H,BQ>*>(memory);
 constexpr int Words=H>=22?96:64;
 if(t<BQ){int r=g*BQ+t;Param p;
  p.guard=a.meta[r*Words+H];p.lo0=a.fine_scale[r*H];p.lo1=a.fine_scale[r*H+1];p.raw=a.raw[r];p.out=a.out[r];p.tau=a.tau[r];p.start=a.starts[r];p.end=a.ends[r];p.hint=Trusted?2:a.hint[r];s.row[t]=p;
 }
 if(warp==8){cute::prefetch_tma_descriptor(&mq);cute::prefetch_tma_descriptor(&mw);cute::prefetch_tma_descriptor(&mk);cute::prefetch_tma_descriptor(&ms);cute::prefetch_tma_descriptor(&mr);}
 if(warp==8&&cute::elect_one_sync()){
  s.full_q.init(1);
  #pragma unroll
  for(int i=0;i<4;++i){s.full_kv[i].init(1);s.empty_kv[i].init(256);}
  cutlass::arch::fence_barrier_init();
 }
 if(warp==9){
  if(cute::elect_one_sync()){
   #pragma unroll
   for(int i=0;i<4;++i){s.full_umma[i].init(1);s.empty_umma[i].init(128);}
   cutlass::arch::fence_barrier_init();
  }
  cute::TMEM::Allocator1Sm().allocate(TM,&s.tmem);cute::TMEM::Allocator1Sm().release_allocation_lock();
 }
 asm volatile("fence.proxy.async.shared::cta;":::"memory");__syncthreads();
 const int all_blocks=(logical_nk-a.seed+255)/256;
 const int first_block=H>=22?all_blocks*int(blockIdx.y)/8:0;
 const int stop_block=H>=22?all_blocks*int(blockIdx.y+1)/8:all_blocks;
 const int blocks=stop_block-first_block;
 if(warp==8){
  cutlass::arch::warpgroup_reg_dealloc<40>();
  if(cute::elect_one_sync()){
   deep_gemm::tma::copy<128,BQ*H,128>(&mq,&s.full_q,s.fine,0,g*BQ*H);
   deep_gemm::tma::copy<W,BQ,0>(&mw,&s.full_q,&s.fw[0][0],0,g*BQ);
   s.full_q.arrive_and_expect_tx(BQ*H*128+BQ*W*4);
   for(int block=0;block<blocks;++block){int stage=block%4,phase=(block/4)&1;
    s.empty_kv[stage].wait(phase^1);
    deep_gemm::tma::copy<128,256,128>(&mk,&s.full_kv[stage],s.raw_k[stage],0,a.seed+(first_block+block)*256);
    deep_gemm::tma::copy<256,1,0>(&ms,&s.full_kv[stage],s.raw_scale[stage],a.seed+(first_block+block)*256,0);
    deep_gemm::tma::copy<256,1,0>(&mr,&s.full_kv[stage],s.raw_rho[stage],a.seed+(first_block+block)*256,0);
    s.full_kv[stage].arrive_and_expect_tx(256*128+2*256*4);
   }
  }
 }else if(warp==9||warp==10){
  cutlass::arch::warpgroup_reg_dealloc<40>();
  const int wg=warp-9;
  auto instruction=cute::UMMA::make_instr_desc<cutlass::float_e4m3_t,cutlass::float_e4m3_t,float,128,N,cute::UMMA::Major::K,cute::UMMA::Major::K>();
  auto runtime=cute::UMMA::make_runtime_instr_desc(instruction);
  s.full_q.wait(0);
  for(int block=0;block<blocks;++block){int stage=block%4,phase=(block/4)&1;
   s.full_kv[stage].wait(phase);int us=block%2,up=(block/2)&1;
   {int buf=wg*2+us;s.empty_umma[buf].wait(up^1);deep_gemm::ptx::tcgen05_after_thread_sync();
    #pragma unroll
    for(int k=0;k<4;++k){
     auto ad=deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K,0,128,128>(reinterpret_cast<cutlass::float_e4m3_t*>(s.raw_k[stage]),wg*128,k*32);
     auto bd=deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K,0,128,128>(reinterpret_cast<cutlass::float_e4m3_t*>(s.fine),0,k*32);
     cute::SM100_MMA_F8F6F4_SS::fma(ad,bd,s.tmem+buf*N,k,runtime);
    }
    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&s.full_umma[buf]));
   }
  }
 }else if(warp>=11){cutlass::arch::warpgroup_reg_dealloc<40>();}
 else{
  cutlass::arch::warpgroup_reg_alloc<232>();
  int wg=t/128;int kept[BQ+BQ/4]={};float weights[BQ][H];Param resident[BQ];bool eligible_q[BQ];
  s.full_q.wait(0);
  #pragma unroll
  for(int qr=0;qr<BQ;++qr){
   resident[qr]=s.row[qr];eligible_q[qr]=a.budget[basegroup+qr/4]==H-2;
   #pragma unroll
   for(int h=0;h<H;++h)weights[qr][h]=deep_gemm::ptx::ld_shared(&s.fw[qr][h]);
  }
  for(int block=0;block<blocks;++block){int stage=block%4,phase=(block/4)&1;int buf=wg*2+(block%2),up=(block/2)&1;
   s.full_kv[stage].wait(phase);
   float scale=deep_gemm::ptx::ld_shared(&s.raw_scale[stage][t]);uint32_t code=deep_gemm::ptx::ld_shared(&s.raw_rho[stage][t])&255u;
   s.full_umma[buf].wait(up);deep_gemm::ptx::tcgen05_after_thread_sync();
   s.empty_kv[stage].arrive();
   uint32_t start=s.tmem+buf*N;
   __half_raw half=__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(code),__NV_E4M3);float rho=__half2float(static_cast<__half>(half));
   int key=a.seed+(first_block+block)*256+t;bool keyvalid=true;if constexpr(!Trusted)keyvalid=key<logical_nk&&code<127&&isfinite(scale)&&scale>0;uint32_t m=0;
   #pragma unroll
   for(int chunk=0;chunk<BQ/4;++chunk){
    constexpr int Part=4*H;float acc[Part];uint32_t address=start+chunk*Part;
    load_tmem<64>(address,acc);cutlass::arch::fence_view_async_tmem_load();
    if constexpr(Part==72){load_tmem<8>(address+64,acc+64);cutlass::arch::fence_view_async_tmem_load();}
    if constexpr(Part==88){load_tmem<16>(address+64,acc+64);cutlass::arch::fence_view_async_tmem_load();load_tmem<8>(address+80,acc+80);cutlass::arch::fence_view_async_tmem_load();}
    if constexpr(Part==120){load_tmem<32>(address+64,acc+64);cutlass::arch::fence_view_async_tmem_load();load_tmem<16>(address+96,acc+96);cutlass::arch::fence_view_async_tmem_load();load_tmem<8>(address+112,acc+112);cutlass::arch::fence_view_async_tmem_load();}
    if(chunk+1==BQ/4){deep_gemm::ptx::tcgen05_before_thread_sync();s.empty_umma[buf].arrive();}
   #pragma unroll
   for(int pair_query=0;pair_query<4;pair_query+=2){
    // Two independent queries share scheduling opportunities. Each query's
    // original four RN paid chains retain exactly the same update order.
    constexpr int Paid=H-2;
    float2 s0[2]={make_float2(0,0),make_float2(0,0)};
    float2 s1[2]={make_float2(0,0),make_float2(0,0)};
    #pragma unroll
    for(int h=0;h<Paid;h+=4){
     #pragma unroll
     for(int j=0;j<2;++j){int i=pair_query+j,qr=chunk*4+i;
      s0[j]=__ffma2_rn(__fadd2_rn(make_float2(acc[i*H+h+2],acc[i*H+h+3]),make_float2(fabsf(acc[i*H+h+2]),fabsf(acc[i*H+h+3]))),make_float2(weights[qr][h+2],weights[qr][h+3]),s0[j]);
      if(h+2<Paid)s1[j]=__ffma2_rn(__fadd2_rn(make_float2(acc[i*H+h+4],acc[i*H+h+5]),make_float2(fabsf(acc[i*H+h+4]),fabsf(acc[i*H+h+5]))),make_float2(weights[qr][h+4],weights[qr][h+5]),s1[j]);
     }
    }
    #pragma unroll
    for(int j=0;j<2;++j){int i=pair_query+j,qr=chunk*4+i;
    float2 pair=__fadd2_rn(s0[j],s1[j]);float paid=__fmul_rn(__fadd_rn(pair.x,pair.y),0.5f);const Param&p=resident[qr];
    float lo0=__fmul_rn(acc[i*H],p.lo0);float neg=__fadd_rn(0.0f,__fmaf_rn(acc[i*H+1],p.lo1,lo0));
    float signed_score=__fadd_rn(paid,-fmaxf(neg,0.0f));float guard=__fmul_rn(rho,p.guard);
    float u=__fadd_rn(__fmul_rn(scale,__fadd_rn(__fadd_rn(signed_score,p.raw),guard)),p.out);
    bool visible=key<logical_nk&&key>=p.start&&key<p.end;
    if constexpr(Trusted){
     const uint32_t bit=trusted_keep_mask(u,p.tau,uint32_t(eligible_q[qr])&uint32_t(visible),1u<<qr);
     m|=bit;if constexpr(!Emit)kept[qr]+=int(bit!=0);
    }else{
     bool keep=eligible_q[qr]&&visible&&(!keyvalid||p.hint!=2||!isfinite(u)||!(u<p.tau));
     m|=keep?1u<<qr:0;if constexpr(!Emit)kept[qr]+=int(keep);
    }
   }
   }
   }
   if constexpr(Emit){
    int word=(first_block+block)*8+warp;unsigned value=0;bool owner=false;
    #pragma unroll
    for(int qr=0;qr<BQ;++qr){unsigned mask=__ballot_sync(0xffffffffu,((m>>qr)&1u)!=0);
     if(lane==qr){value=mask;owner=eligible_q[qr];}
    }
    // One coalesced warp store for each Q4's four ballot words, including zero.
    // Public transport layout is [Q4group, raw_word, query_within_group].
    if(lane<BQ&&owner&&word<active_word_count){int row=g*BQ+lane;
     page.words[(int64_t(row/4)*page.word_count+word)*4+row%4]=value;
    }
   }
   #pragma unroll
   for(int j=0;j<BQ/4;++j){uint32_t four=(m>>(j*4))&15u;
    if constexpr(!Emit)kept[BQ+j]+=int(four==15);
    if constexpr(!Emit)if(a.keep_ref&&key<logical_nk&&four!=a.keep_ref[uint64_t(basegroup+j)*a.nk+key])atomicAdd(a.debug,1);
   }
  }
  if constexpr(!Emit){
  #pragma unroll
  for(int qr=0;qr<BQ+BQ/4;++qr){int total=kept[qr];
   #pragma unroll
   for(int delta=16;delta;delta>>=1)total+=__shfl_down_sync(0xffffffffu,total,delta);
   if(lane==0){int group=basegroup+(qr<BQ?qr/4:qr-BQ),column=qr<BQ?qr%4:4;atomicAdd(a.packets+group*5+column,total);}
  }
  }
 }
 __syncthreads();if(warp==9)cute::TMEM::Allocator1Sm().free(s.tmem,TM);
}
