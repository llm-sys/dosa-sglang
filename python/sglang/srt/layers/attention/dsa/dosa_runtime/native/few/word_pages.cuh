#pragma once
#include "../common/head_geometry.cuh"
#include <cub/block/block_scan.cuh>
// ID-only page128 pool. Each successful new page owns exactly one H32 task.
namespace fused_few {
struct PagedArgs {
 int *pool,*table,*head,*status,*error,*solo_counts,*common_counts,*solo_n,*common_n;
 int2 *solo_tasks,*common_tasks;
 unsigned*words;int word_count;
 int nq,nk,max_records,columns,pages,solo_task_cap,common_task_cap;
};
__device__ __forceinline__ void page_error(const PagedArgs&a,int row,bool common,int status){
 atomicOr(a.status,status);
 if(common){
  #pragma unroll
  for(int j=0;j<4;++j)atomicOr(a.error+row*4+j,1);
 }else atomicOr(a.error+row,1);
}
__device__ __forceinline__ int ensure_id_page(const PagedArgs&a,int row,int packet,bool common){
 if(packet<0||packet>=a.columns){page_error(a,row,common,2);return -1;}
 int table_row=common?a.nq+row:row;int*slot=a.table+int64_t(table_row)*a.columns+packet;
 int old=atomicCAS(slot,-1,-2);
 if(old==-1){
  int page=atomicAdd(a.head,1);
  if(page<0||page>=a.pages){page_error(a,row,common,1);__threadfence();atomicExch(slot,-3);return -1;}
  int task=atomicAdd(common?a.common_n:a.solo_n,1);
  int cap=common?a.common_task_cap:a.solo_task_cap;
  if(task<0||task>=cap){page_error(a,row,common,2);__threadfence();atomicExch(slot,-3);return -1;}
  (common?a.common_tasks:a.solo_tasks)[task]=make_int2(row,packet*128);
  __threadfence();atomicExch(slot,page);return page;
 }
 while(old==-2){__nanosleep(32);old=atomicAdd(slot,0);}
 if(old<0||old>=a.pages){page_error(a,row,common,old==-3?1:2);return -1;}
 return old;
}

}

namespace fused_few {
// One CTA amortizes each of five stream reservations over64K raw keys.
// Each thread owns eight coalesced uint4 word reads. Input classification,
// contiguous per-row reservations and shared-page arbitration are unchanged.
// Union only inside the intersection of all four original visible intervals.
__global__ void compact_words(const __grid_constant__ PagedArgs a,const int*budget,int seed,const int*starts,const int*ends){
 using Scan=cub::BlockScan<int,256>;
 __shared__ typename Scan::TempStorage scan;
 __shared__ int aggregate,base,valid,first_page,page_count,pages[513];
 int t=threadIdx.x,row=blockIdx.y,first_word=blockIdx.x*2048+t;
 int family=budget[row];bool budget_ok=dosa_geometry::valid_certificate_budget(family);
 if(!budget_ok){if(t==0)page_error(a,row,true,2);return;}
 int lo=max(max(starts[row*4],starts[row*4+1]),max(starts[row*4+2],starts[row*4+3]));
 int hi=min(min(ends[row*4],ends[row*4+1]),min(ends[row*4+2],ends[row*4+3]));
 uint4 packed[8];unsigned common[8];
 #pragma unroll
 for(int j=0;j<8;++j){
  int word=first_word+j*256;
  packed[j]=word<a.word_count?reinterpret_cast<const uint4*>(a.words)[int64_t(row)*a.word_count+word]:make_uint4(0,0,0,0);
  int key=seed+word*32; int left=max(lo-key,0),right=hi-key;
  unsigned lower=left>=32?0u:(0xffffffffu<<min(left,31));
  unsigned upper=right>=32?0xffffffffu:(right<=0?0u:((1u<<min(right,31))-1u));
  unsigned a=packed[j].x,b=packed[j].y,c=packed[j].z,d=packed[j].w;
  common[j]=((a&b)|(a&c)|(a&d)|(b&c)|(b&d)|(c&d))&lower&upper;
 }
 #pragma unroll
 for(int stream=0;stream<5;++stream){
  bool is_common=stream==4;int output_row=is_common?row:row*4+stream;
  unsigned mask[8];int n=0,offset,total;
  #pragma unroll
  for(int j=0;j<8;++j){
   unsigned value=stream==0?packed[j].x:stream==1?packed[j].y:stream==2?packed[j].z:packed[j].w;
   mask[j]=is_common?common[j]:value&~common[j];n+=__popc(mask[j]);
  }
  Scan(scan).ExclusiveSum(n,offset,total);
  if(t==0)aggregate=total;
  __syncthreads();
  if(!aggregate)continue;
  if(t==0){
   base=atomicAdd((is_common?a.common_counts:a.solo_counts)+output_row,aggregate);valid=0;page_count=0;
   if(base<0||base>=a.max_records)page_error(a,output_row,is_common,2);
   else{
    valid=min(aggregate,a.max_records-base);if(valid<aggregate)page_error(a,output_row,is_common,2);
    first_page=base/128;page_count=(base+valid-1)/128-first_page+1;
    if(page_count>513){page_error(a,output_row,is_common,2);valid=0;page_count=0;}
   }
  }
  __syncthreads();
  for(int page=t;page<page_count;page+=256)pages[page]=ensure_id_page(a,output_row,first_page+page,is_common);
  __syncthreads();
  int rank=0;
  #pragma unroll
  for(int j=0;j<8;++j){
   while(mask[j]){
    int bit=__ffs(mask[j])-1;int64_t position=int64_t(base)+offset+rank;
    if(offset+rank<valid&&position>=0&&position<a.max_records){
     int pg=pages[position/128-first_page];
     if(pg>=0&&pg<a.pages)a.pool[int64_t(pg)*128+(position&127)]=seed+(first_word+j*256)*32+bit;
    }
    mask[j]&=mask[j]-1;++rank;
   }
  }
  __syncthreads();
 }
}
}
