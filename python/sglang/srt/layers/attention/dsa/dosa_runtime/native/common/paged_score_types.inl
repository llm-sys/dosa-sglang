// Shared static/device-extent implementation. Included inside the selected
// namespace policy; keep the distinct live metadata pipeline and guards.
#include "../common/h32_common.cuh"
#include <climits>
#include "../common/record4_codec.cuh"
#include "../common/h32_record5_codec.cuh"

namespace DOSA_PAGED_NAMESPACE {
using namespace certicascade_streaming_exact;
using Barrier=cutlass::arch::ClusterTransactionBarrier;
constexpr int GroupIds=4096,PacketIds=128;
template<int BQ> struct alignas(1024) Smem {
  alignas(1024) uint8_t keys[2][128*128],q[BQ*kHeads*128];
  alignas(128) float weights[BQ*kHeads],scale[2][128];
  alignas(128) int ids[2][128],coordinates[2][128];
  Barrier gathered[2],done;
  uint32_t tmem;
  int job;
  int page[2],bad[2],skip;
  uint32_t emit_counts[BQ][4],emit_base[BQ];
  uint64_t emit_starts[BQ][4];
};
struct Args {
  const uint8_t*q;
  const float*w,*scale,*tau;
  const int*ids,*counts,*packs;
  const int2*tasks;
  void*scores;
  int*outids,*outcounts;
  int nk,stride,origin;
  int64_t ids_stride;
  int*cursor;
  const int*page_table,*starts,*ends;
  int*error;
  int qrows,stream_rows,table_width,pool_pages,output_cap;
  int64_t table_stride,pool_stride,score_stride,output_id_stride;
#if DOSA_PAGED_LIVE_EXTENT
  const int* live_nk; int candidate_seed;
#else
#endif
};
__device__ __forceinline__ int swizzle(int row,int vec){return row*8+(vec^(row%8));}
template<int BQ>
__device__ __forceinline__ void flag_group(const Args&a,int row,int bit){
  for(int qb=0;qb<BQ;++qb)if(row*BQ+qb<a.qrows)atomicOr(a.error+row*BQ+qb,bit);
}
struct Visibility { int begin,end; };
template<int BQ>
__device__ __forceinline__ Visibility group_visibility(const Args&a,int row) {
  Visibility v{INT_MIN,INT_MAX};
  #pragma unroll
  for(int qb=0;qb<BQ;++qb) {
    const int qr=row*BQ+qb;
    if(qr<a.qrows) {
      v.begin=max(v.begin,a.starts[qr]);
      v.end=min(v.end,a.ends[qr]);
    }
  }
  return v;
}
template<int BQ,bool CachedVisibility=false>
__device__ __forceinline__ void prepare_metadata(Smem<BQ>&s,const Args&a,int row,int begin,int stop,int slot,int t,
                                                Visibility visibility={}){
  if(t==0){
    int packet=begin/PacketIds;
    int page=(begin>=0&&packet<a.table_width)?a.page_table[int64_t(row)*a.table_stride+packet]:-1;
    s.page[slot]=page;s.bad[slot]=(page<0||page>=a.pool_pages)?1:0;
  }
  __syncthreads();
  bool active=begin+t<stop;
  int page=s.page[slot];
  bool address=active&&page>=0&&page<a.pool_pages;
  int id=address?a.ids[int64_t(page)*a.pool_stride+t]:0;
  bool valid=address&&id>=0&&id<a.nk;
  if(valid){
    int64_t visible_id=int64_t(id)+a.origin;
    if constexpr(CachedVisibility) {
      valid=visible_id>=visibility.begin&&visible_id<visibility.end;
    } else {
    for(int qb=0;qb<BQ;++qb){
      int qr=row*BQ+qb;
      if(qr<a.qrows&&(visible_id<a.starts[qr]||visible_id>=a.ends[qr]))valid=false;
    }
    }
  }
  if(active&&!valid)atomicOr(&s.bad[slot],1);
  // Invalid input never reaches a scale access or TMA coordinate. The original
  // MMA is still drained uniformly; sticky error suppresses partial output.
  s.ids[slot][t]=valid?id:0;s.coordinates[slot][t]=valid?id:0;
  s.scale[slot][t]=valid?a.scale[id]:0;
  __syncthreads();
  if(t==0&&s.bad[slot])flag_group<BQ>(a,row,1);
  __syncthreads();
}
// Future metadata stays private until the current packet has consumed its
// shared ID/scale. Warp leaders read the immutable page entry independently;
// this removes the CTA rendezvous from the dependency chain before the loads.
struct FutureMetadata { int page,id; float scale; bool bad; };
#if DOSA_PAGED_LIVE_EXTENT
// Hold the immutable page/ID load while one independent H32 query is reduced.
// Scale and visibility are still checked before the original publication point.
struct FutureId { int page,id; uint32_t flags; };
__device__ __forceinline__ FutureId load_future_id(
    const Args&a,int row,int begin,int stop,int t) {
  int page=-1;
  if((t&31)==0) {
    const int packet=begin/PacketIds;
    page=(begin>=0&&packet<a.table_width)
        ?a.page_table[int64_t(row)*a.table_stride+packet]:-1;
  }
  page=__shfl_sync(0xffffffffu,page,0);
  const bool active=begin+t<stop;
  const bool address=active&&page>=0&&page<a.pool_pages;
  const int id=address?a.ids[int64_t(page)*a.pool_stride+t]:0;
  return {page,id,uint32_t(active)|(uint32_t(address)<<1)};
}
__device__ __forceinline__ FutureMetadata finish_future_scale(
    const Args&a,const FutureId&f,Visibility visibility) {
  // Keep every ID-dependent operation in this late stage. In the equivalent
  // C++ expression ptxas hoisted id>=0 immediately after the early ID load,
  // defeating its latency overlap even though the scale load stayed late.
  int id,bad;float scale;
  asm volatile(
    "{ .reg .u32 flags; .reg .s64 visible; .reg .u64 addr;\n"
    "  .reg .pred valid,active,badp,p;\n"
    "  mov.u32 %0,0; mov.f32 %1,0f00000000;\n"
    "  and.b32 flags,%5,2; setp.ne.u32 valid,flags,0;\n"
    "  setp.ge.s32 p,%4,0; and.pred valid,valid,p;\n"
    "  setp.lt.s32 p,%4,%6; and.pred valid,valid,p;\n"
    "  cvt.s64.s32 visible,%4; add.s64 visible,visible,%8;\n"
    "  setp.ge.s64 p,visible,%9; and.pred valid,valid,p;\n"
    "  setp.lt.s64 p,visible,%10; and.pred valid,valid,p;\n"
    "  @valid mad.wide.s32 addr,%4,4,%11;\n"
    "  @valid ld.global.f32 %1,[addr]; @valid mov.b32 %0,%4;\n"
    "  setp.lt.s32 badp,%3,0; setp.ge.s32 p,%3,%7; or.pred badp,badp,p;\n"
    "  and.b32 flags,%5,1; setp.ne.u32 active,flags,0;\n"
    "  not.pred p,valid; and.pred p,p,active; or.pred badp,badp,p;\n"
    "  selp.u32 %2,1,0,badp; }\n"
    : "=&r"(id),"=&f"(scale),"=&r"(bad)
    : "r"(f.page),"r"(f.id),"r"(f.flags),"r"(a.nk),"r"(a.pool_pages),
      "l"(int64_t(a.origin)),"l"(int64_t(visibility.begin)),
      "l"(int64_t(visibility.end)),"l"(a.scale)
    : "memory");
  return {f.page,id,scale,bad!=0};
}
#else
#endif
template<int BQ,bool CachedVisibility=false>
__device__ __forceinline__ FutureMetadata load_future(
    const Args&a,int row,int begin,int stop,int t,Visibility visibility={}) {
  int page=-1;
  if((t&31)==0) {
    const int packet=begin/PacketIds;
    page=(begin>=0&&packet<a.table_width)
        ?a.page_table[int64_t(row)*a.table_stride+packet]:-1;
  }
  page=__shfl_sync(0xffffffffu,page,0);
  const bool active=begin+t<stop;
  const bool address=active&&page>=0&&page<a.pool_pages;
  const int id=address?a.ids[int64_t(page)*a.pool_stride+t]:0;
  bool valid=address&&id>=0&&id<a.nk;
  if(valid) {
    const int64_t visible_id=int64_t(id)+a.origin;
    if constexpr(CachedVisibility) {
      valid=visible_id>=visibility.begin&&visible_id<visibility.end;
    } else {
    for(int qb=0;qb<BQ;++qb) {
      const int qr=row*BQ+qb;
      if(qr<a.qrows&&(visible_id<a.starts[qr]||visible_id>=a.ends[qr]))valid=false;
    }
    }
  }
  return {page,valid?id:0,valid?a.scale[id]:0.0f,
          page<0||page>=a.pool_pages||(active&&!valid)};
}
template<int BQ>
__device__ __forceinline__ void publish_future(
    Smem<BQ>&s,const Args&a,int row,int slot,int t,const FutureMetadata&future) {
  s.ids[slot][t]=future.id;s.coordinates[slot][t]=future.id;
  s.scale[slot][t]=future.scale;
  const int bad=__syncthreads_or(future.bad);
  if(t==0) {
    s.page[slot]=future.page;s.bad[slot]=bad;
    if(bad)flag_group<BQ>(a,row,1);
  }
  // The original end-of-packet CTA barrier follows, publishing the group error
  // and all metadata before the next iteration can issue the future gather.
}
template<int BQ>
__device__ __forceinline__ void gather(Smem<BQ>&s,const cute::TmaDescriptor*map,int slot,int t){
  if((t&31)!=0)return;
  const int warp=t/32;
  #pragma unroll
  for(int packet_part=0;packet_part<8;++packet_part){
    const int r=warp*4+packet_part*16;
    int4 coord=*reinterpret_cast<const int4*>(s.coordinates[slot]+r);
    deep_gemm::ptx::tma_gather4(map,s.gathered[slot],s.keys[slot]+r*128,0,coord,static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL));
  }
}
// Dense, GPU-only filtering of the unchanged original 128-boundary tasks.
template<int BQ>
__global__ void filter_groups(const int2*input,const int*pack_count,int2*groups,int*state,int target_groups,
#if DOSA_PAGED_LIVE_EXTENT
                             int task_cap,const int*counts,int table_width,int stream_rows,int qrows,int*error,
                             const int*live_nk,int nk_capacity,int seed){
  const int logical=*live_nk;
  if(logical<seed||logical>nk_capacity){
    for(int qr=blockIdx.x*blockDim.x+threadIdx.x;qr<qrows;qr+=gridDim.x*blockDim.x)atomicOr(error+qr,2);
    return;
  }
  const int max_records=logical-seed;
  table_width=(max_records+PacketIds-1)/PacketIds;
#else
                             int task_cap,const int*counts,int table_width,int stream_rows,int qrows,int*error){
#endif
  int lane=threadIdx.x%32,warp=threadIdx.x/32,raw=*pack_count;
  int count=max(0,min(raw,task_cap));
  for(int r=blockIdx.x*blockDim.x+threadIdx.x;r<stream_rows;r+=gridDim.x*blockDim.x){
    int n=counts[r];
#if DOSA_PAGED_LIVE_EXTENT
    if(n<0||int64_t(n)>int64_t(max_records))
#else
    if(n<0||int64_t(n)>int64_t(table_width)*PacketIds)
#endif
      for(int qb=0;qb<BQ;++qb)if(r*BQ+qb<qrows)atomicOr(error+r*BQ+qb,2);
  }
  if(raw<0||raw>task_cap){
    for(int qr=blockIdx.x*blockDim.x+threadIdx.x;qr<qrows;qr+=gridDim.x*blockDim.x)atomicOr(error+qr,2);
  }
  int ppg=1;
  #pragma unroll
  for(int size=2;size<=32;size*=2)if(count>=target_groups*size)ppg=size;
  int group_ids=PacketIds*ppg;
  if(blockIdx.x==0&&threadIdx.x==0)state[2]=ppg;
  for(int base=blockIdx.x*blockDim.x+warp*32;base<count;base+=gridDim.x*blockDim.x){
    int i=base+lane;bool active=i<count;
    int2 task=active?input[i]:make_int2(0,1);
    bool row_ok=task.x>=0&&task.x<stream_rows;
    int n=active&&row_ok?counts[task.x]:0;
    bool valid=active&&row_ok&&task.y>=0&&(task.y%PacketIds)==0&&task.y<n&&
#if DOSA_PAGED_LIVE_EXTENT
               n>=0&&int64_t(n)<=int64_t(max_records);
#else
               n>=0&&int64_t(n)<=int64_t(table_width)*PacketIds;
#endif
    if(active&&!valid){
      if(row_ok){for(int qb=0;qb<BQ;++qb)if(task.x*BQ+qb<qrows)atomicOr(error+task.x*BQ+qb,2);}
      else{for(int qr=0;qr<qrows;++qr)atomicOr(error+qr,2);}
    }
    bool keep=valid&&(task.y&(group_ids-1))==0;
    uint32_t mask=__ballot_sync(0xffffffffu,keep),nkeep=__popc(mask),start=0;
    if(lane==0&&nkeep)start=atomicAdd(state,nkeep);
    start=__shfl_sync(0xffffffffu,start,0);
    if(keep)groups[start+__popc(lane?mask&((1u<<lane)-1):0)]=task;
  }
}

} // namespace DOSA_PAGED_NAMESPACE
