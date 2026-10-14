#pragma once
// Copyright (c) 2025 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Shared score-to-record consumer. Full and Few retain distinct mathematical
// domains and access schedules, and execute this tile/reduction/writer body.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <deep_gemm/mma/sm100.cuh>
#include <cutlass/arch/barrier.h>
#include <deep_gemm/ptx/tcgen05.cuh>
#include "record5_codec.cuh"
#include "score_emit_mma.cuh"
#include "head_geometry.cuh"
#include "h32_record5_codec.cuh"

namespace dosa_score_emit {

enum class Domain { FullSurrogate24, OriginalH32Global20 };
struct EncodedRecord { uint32_t code; bool valid; };

template<int Queries, int Threads, int ReadQueries, int TmemElements,
         int LaneSlots, int WindowTiles, bool Coalesced, bool RingRefresh,
         int ReservationWarps>
struct Geometry {
  static constexpr int kHeads=dosa_geometry::Heads;
  static constexpr int kQueries=Queries, kThreads=Threads, kWarps=Threads/32;
  static constexpr int kReadQueries=ReadQueries, kTmemElements=TmemElements;
  static constexpr int kLaneSlots=LaneSlots, kWindowTiles=WindowTiles;
  static constexpr bool kCoalesced=Coalesced, kRingRefresh=RingRefresh;
  static constexpr int kReservationWarps=ReservationWarps;
  static_assert(Queries==1||Queries==4);
  static_assert(Queries%ReadQueries==0 && (ReadQueries*kHeads)%TmemElements==0);
  static_assert(LaneSlots<=255 && WindowTiles<=256);
  static_assert(ReservationWarps==1||ReservationWarps==4);
};

// Input pages have128 IDs; these output pages have4096 records. Never exchange
// the units. Dense Few output uses its own explicit row strides and capacity.
struct RecordOutput {
  uint8_t *low, *overflow_low;
  int32_t *words, *overflow_words, *counts, *error;
  int32_t *page_table, *pool_status;
  uint32_t *pool_head;
  int rows, capacity, pool_pages;
  int64_t low_stride, word_stride;
};
struct EmitStorage {
  uint32_t *records;
  uint2 *coalesced;
  uint32_t *ring_sequence;
  uint32_t *cta_counts;
  uint64_t *cta_starts;
};

template<Domain D> struct Math {
  __device__ __forceinline__ static float2 activate(float x,float y) {
    if constexpr(D==Domain::FullSurrogate24)
      return make_float2(fmaxf(x,0.0f),fmaxf(y,0.0f));
    else
      return __fadd2_rn(make_float2(x,y),make_float2(fabsf(x),fabsf(y)));
  }
  template<class Access>
  __device__ __forceinline__ static float finish(float total,int q,const Access&a) {
    if constexpr(D==Domain::FullSurrogate24)
      return __fmaf_rn(a.lane_scale(),total,a.affine_bias(q));
    else
      return __fmul_rn(__fmul_rn(total,0.5f),a.lane_scale());
  }
  template<class Access>
  __device__ __forceinline__ static bool keep(float value,int q,Access&a) {
    if constexpr(D==Domain::FullSurrogate24)
      return __float_as_int(value)<=__float_as_int(a.cutoff(q)) && a.visible(q);
    else {
      const bool active=a.lane_active()&&a.candidate_for_query(q);
      if(active&&!isfinite(value))a.flag_error(q,8);
      return active&&a.error(q)==0&&(!isfinite(a.cutoff(q))||value>=a.cutoff(q));
    }
  }
  template<class Access>
  __device__ __forceinline__ static EncodedRecord encode(float value,int q,const Access&a) {
    if constexpr(D==Domain::FullSurrogate24) {
      const uint32_t bits=__float_as_uint(value==0.0f?0.0f:value);
      return {((bits&0x80000000u)?~bits:(bits^0x80000000u))>>8,true};
    } else {
      const uint32_t cp=h32_record5::encode(value);
      const int64_t key=a.key_label();
      const bool valid=isfinite(value)&&isfinite(a.cutoff(q))&&value>=a.cutoff(q)&&
          __float_as_uint(value)!=0x80000000u&&key>=0&&key<0x100000&&
          cp>=h32_record5::Min&&cp<=h32_record5::Max;
      return {valid?cp:h32_record5::Invalid,valid};
    }
  }
  __device__ __forceinline__ static uint32_t storage_code(uint32_t code) {
    if constexpr(D==Domain::FullSurrogate24)return record5::compact(code);
    else return code;
  }
};

template<int Elements>
__device__ __forceinline__ void load_tmem(uint32_t address,float*values) {
  static_assert(Elements==16||Elements==128);
  if constexpr(Elements==16) {
    [&]<size_t... I>(cute::index_sequence<I...>) {
      cute::SM100_TMEM_LOAD_32dp32b16x::copy(address,reinterpret_cast<uint32_t*>(values)[I]...);
    }(cute::make_index_sequence<16>{});
  } else {
    [&]<size_t... I>(cute::index_sequence<I...>) {
      cute::SM100_TMEM_LOAD_32dp32b128x::copy(address,reinterpret_cast<uint32_t*>(values)[I]...);
    }(cute::make_index_sequence<128>{});
  }
}

__device__ __forceinline__ void pool_error(int32_t*status,uint32_t bit) {
  if(status)atomicOr(reinterpret_cast<uint32_t*>(status),bit);
}
static __device__ __noinline__ int32_t ensure_record_page(
    uint32_t row,uint32_t ordinal,uint32_t cap,int32_t*table,uint32_t*head,
    int32_t*status,uint32_t pool_pages) {
  const uint32_t row_pages=(cap-8192u+4095u)/4096u;
  int32_t*entry=table+row*row_pages+ordinal;
  int32_t page=atomicCAS(entry,-1,-2);
  if(page==-1) {
    const uint32_t allocated=atomicAdd(head,1u);
    if(allocated>=pool_pages) {
      pool_error(status,2u);atomicExch(entry,-3);return -3;
    }
    page=int32_t(allocated);atomicExch(entry,page);return page;
  }
  while(page==-2) {__nanosleep(64);page=atomicCAS(entry,-2,-2);}
  return page;
}
struct PageReservation {int ordinal0,page0,page1;};
static __device__ __noinline__ PageReservation prepare_record_pages(
    uint32_t row,int base,uint32_t count,uint32_t cap,int32_t*table,uint32_t*head,
    int32_t*status,uint32_t pool_pages) {
  PageReservation out{-1,-1,-1};
  const uint32_t end=uint32_t(base)+count;
  if(end>cap)pool_error(status,4u);
  const uint32_t begin=max(uint32_t(base),8192u),stop=min(end,cap);
  if(begin>=stop)return out;
  out.ordinal0=int((begin-8192u)/4096u);
  const int last=int((stop-1u-8192u)/4096u);
  out.page0=ensure_record_page(row,out.ordinal0,cap,table,head,status,pool_pages);
  if(last!=out.ordinal0)
    out.page1=ensure_record_page(row,last,cap,table,head,status,pool_pages);
  return out;
}

template<Domain D,bool Streaming>
__device__ __forceinline__ void store_code(uint8_t*low,int32_t*word,
                                         uint32_t code,uint32_t key) {
  const uint32_t stored=Math<D>::storage_code(code);
  if constexpr(Streaming) {
    __stcs(low,uint8_t(stored));__stcs(word,int32_t(record5::pack_word(stored,key)));
  } else {
    *low=uint8_t(stored);*word=int32_t(record5::pack_word(stored,key));
  }
}

template<Domain D>
static __device__ __noinline__ void copy_record_segment(
    const uint2*records,int total,uint32_t base,uint8_t*row_low,int32_t*row_words,
    uint8_t*overflow_low,int32_t*overflow_words,int ordinal0,int page0,int page1) {
  for(int at=int(threadIdx.x&31u);at<total;at+=32) {
    const uint2 value=records[at];const uint32_t out=base+uint32_t(at);
    if(out<8192u)store_code<D,true>(row_low+out,row_words+out,value.x,value.y);
    else {
      const uint32_t overflow=out-8192u;const int ordinal=int(overflow/4096u);
      const int page=ordinal==ordinal0?page0:ordinal==ordinal0+1?page1:-3;
      if(page>=0) {
        const uint64_t address=uint64_t(page)*4096u+overflow%4096u;
        store_code<D,true>(overflow_low+address,overflow_words+address,value.x,value.y);
      }
    }
  }
}

// A lane exceeding its small ring quota is already a cold path. Keep its
// address translation and allocator outside the consumer instruction stream.
template<Domain D>
static __device__ __noinline__ void store_lane_overflow(
    uint32_t row,int pos,uint32_t cap,uint32_t code,uint32_t key,
    uint8_t*low,int32_t*words,uint8_t*overflow_low,int32_t*overflow_words,
    int32_t*page_table,uint32_t*pool_head,int32_t*pool_status,uint32_t pool_pages) {
  if(pos>=int(cap)) {
    pool_error(pool_status,4u);
    return;
  }
  if(uint32_t(pos)<8192u) {
    const uint64_t at=uint64_t(row)*8192u+uint32_t(pos);
    store_code<D,true>(low+at,words+at,code,key);
    return;
  }
  const uint32_t overflow=uint32_t(pos)-8192u;
  const int page=ensure_record_page(row,overflow/4096u,cap,page_table,
                                  pool_head,pool_status,pool_pages);
  if(page>=0) {
    const uint64_t at=uint64_t(page)*4096u+overflow%4096u;
    store_code<D,true>(overflow_low+at,overflow_words+at,code,key);
  }
}

template<class G,Domain D>
class RecordWriter {
  RecordOutput out;
  EmitStorage shared;
  uint32_t lane_counts=0;
  static constexpr uint32_t FullMask=0xffffffffu;
 public:
  __device__ __forceinline__ RecordWriter(RecordOutput output,EmitStorage storage)
      :out(output),shared(storage){}

  template<class Access>
  __device__ __forceinline__ void write_one(int q,int64_t column,uint32_t code,
      uint32_t key,int ordinal0,int page0,int page1,Access&a,bool record_valid) {
    const int row=a.row_base()+q;
    if constexpr(D==Domain::FullSurrogate24) {
      const uint32_t pos=uint32_t(column);
      if(pos<8192u)
        store_code<D,true>(out.low+uint64_t(row)*8192u+pos,
                          out.words+uint64_t(row)*8192u+pos,code,key);
      else {
        const uint32_t overflow=pos-8192u;const int ordinal=int(overflow/4096u);
        const int page=ordinal==ordinal0?page0:ordinal==ordinal0+1?page1:-3;
        if(page>=0) {
          const uint64_t address=uint64_t(page)*4096u+overflow%4096u;
          store_code<D,true>(out.overflow_low+address,out.overflow_words+address,code,key);
        }
      }
    } else {
      if(column>=0&&column<out.capacity) {
        store_code<D,false>(out.low+int64_t(row)*out.low_stride+column,
                            out.words+int64_t(row)*out.word_stride+column,code,key);
        if(!record_valid)a.flag_error(q,32);
      } else a.flag_error(q,16);
    }
  }

  template<class Access>
  __device__ __forceinline__ void consume(int tile,int tiles,const float*scores,
                                         uint32_t mask,Access&a) {
    const uint32_t thread=threadIdx.x,lane=thread&31u,warp=thread/32u;
    if constexpr(G::kLaneSlots>1) {
      if(mask) {
        #pragma unroll
        for(int q=0;q<G::kQueries;++q)if((mask>>q)&1u) {
          const uint32_t code=Math<D>::encode(scores[q],q,a).code;
          const uint32_t count=(lane_counts>>(q*8))&255u;
          if(count<G::kLaneSlots) {
            shared.records[((warp*G::kQueries+q)*G::kLaneSlots+count)*32u+lane]=
                (code<<8)|uint32_t(tile%G::kWindowTiles);
            lane_counts+=1u<<(q*8);
          } else {
            const int row=a.row_base()+q;
            const int pos=atomicAdd(out.counts+row,1);
            store_lane_overflow<D>(row,pos,out.capacity,code,uint32_t(a.key_label()),
                out.low,out.words,out.overflow_low,out.overflow_words,
                out.page_table,out.pool_head,out.pool_status,out.pool_pages);
          }
        }
      }
    } else {
      lane_counts=0;
      #pragma unroll
      for(int q=0;q<G::kQueries;++q)lane_counts|=((mask>>q)&1u)<<(q*8);
    }
    const bool flush=(tile+1)%G::kWindowTiles==0||tile+1==tiles;
    if(!flush)return;
    if constexpr(G::kCoalesced) {
      deep_gemm::ptx::tcgen05_before_thread_sync();
      cutlass::arch::NamedBarrier(G::kThreads,1).sync();
      deep_gemm::ptx::tcgen05_after_thread_sync();
      asm volatile("fence.proxy.async.shared::cta;":::"memory");
    }
    int row_base=0,ordinal0=-1,page0=-1,page1=-1;
    uint32_t row_total=0;
    uint32_t lane_offsets[G::kQueries];
    #pragma unroll
    for(int q=0;q<G::kQueries;++q) {
      const uint32_t count=(lane_counts>>(q*8))&255u;
      uint32_t total;
      if constexpr(G::kLaneSlots==1) {
        const uint32_t bits=__ballot_sync(FullMask,count!=0);
        total=__popc(bits);
        lane_offsets[q]=__popc(lane?bits&((1u<<lane)-1u):0u);
      } else {
        total=__reduce_add_sync(FullMask,count);
      }
      if(lane==uint32_t(q))row_total=total;
      if constexpr(G::kReservationWarps==4)
        if(lane==0)shared.cta_counts[q*G::kWarps+warp]=total;
    }
    if constexpr(G::kReservationWarps==4) {
      __syncthreads();
      if(thread<G::kQueries) {
        const uint32_t c0=shared.cta_counts[thread*4],c1=shared.cta_counts[thread*4+1];
        const uint32_t c2=shared.cta_counts[thread*4+2],c3=shared.cta_counts[thread*4+3];
        const uint32_t total=c0+c1+c2+c3;
        const uint32_t base=total?uint32_t(atomicAdd(out.counts+a.row_base()+thread,total)):0u;
        shared.cta_starts[thread*4]=uint64_t(base);
        shared.cta_starts[thread*4+1]=uint64_t(base)+c0;
        shared.cta_starts[thread*4+2]=uint64_t(base)+c0+c1;
        shared.cta_starts[thread*4+3]=uint64_t(base)+c0+c1+c2;
      }
      __syncthreads();
    } else if(lane<G::kQueries) {
      const int row=a.row_base()+int(lane);
      if(row<out.rows&&row_total) {
        row_base=atomicAdd(out.counts+row,int(row_total));
        if constexpr(D==Domain::FullSurrogate24) {
          if(uint32_t(row_base)+row_total>8192u) {
            const auto p=prepare_record_pages(row,row_base,row_total,out.capacity,
                out.page_table,out.pool_head,out.pool_status,out.pool_pages);
            ordinal0=p.ordinal0;page0=p.page0;page1=p.page1;
          }
        }
      }
    }
    if constexpr(G::kLaneSlots>1) {
      static_assert(G::kQueries==4);
      uint32_t p01=(lane_counts&255u)|((lane_counts>>8&255u)<<16);
      uint32_t p23=(lane_counts>>16&255u)|((lane_counts>>24)<<16);
      #pragma unroll
      for(int delta=1;delta<32;delta<<=1) {
        const uint32_t a0=__shfl_up_sync(FullMask,p01,delta),a1=__shfl_up_sync(FullMask,p23,delta);
        if(lane>=uint32_t(delta)){p01+=a0;p23+=a1;}
      }
      #pragma unroll
      for(int q=0;q<4;++q) {
        const uint32_t inclusive=((q<2?p01:p23)>>((q&1)*16))&65535u;
        lane_offsets[q]=inclusive-((lane_counts>>(q*8))&255u);
      }
    }
    #pragma unroll
    for(int q=0;q<G::kQueries;++q) {
      const uint32_t count=(lane_counts>>(q*8))&255u;
      int64_t base;
      if constexpr(G::kReservationWarps==4)base=int64_t(shared.cta_starts[q*G::kWarps+warp]);
      else base=int64_t(uint32_t(__shfl_sync(FullMask,row_base,q)));
      const int ord=__shfl_sync(FullMask,ordinal0,q);
      const int p0=__shfl_sync(FullMask,page0,q),p1=__shfl_sync(FullMask,page1,q);
      if constexpr(G::kCoalesced) {
        uint2*scratch=shared.coalesced+warp*G::kLaneSlots*32u;
        for(uint32_t slot=0;slot<count;++slot) {
          const uint32_t word=shared.records[((warp*G::kQueries+q)*G::kLaneSlots+slot)*32u+lane];
          scratch[lane_offsets[q]+slot]=make_uint2(word>>8,a.key_for_record(tile/G::kWindowTiles*G::kWindowTiles,word&255u));
        }
        __syncwarp(FullMask);
        const int total=int(__shfl_sync(FullMask,row_total,q));
        const int copy=a.row_base()+q<out.rows?min(total,max(out.capacity-int(base),0)):0;
        const uint64_t row=uint64_t(a.row_base()+q)*8192u;
        copy_record_segment<D>(scratch,copy,uint32_t(base),out.low+row,out.words+row,
                               out.overflow_low,out.overflow_words,ord,p0,p1);
        __syncwarp(FullMask);
      } else {
        for(uint32_t slot=0;slot<count;++slot) {
          uint32_t code,key;
          bool record_valid=true;
          if constexpr(G::kLaneSlots==1) {
            const auto record=Math<D>::encode(scores[q],q,a);
            code=record.code;record_valid=record.valid;key=uint32_t(a.key_label());
          } else {
            const uint32_t word=shared.records[((warp*G::kQueries+q)*G::kLaneSlots+slot)*32u+lane];
            code=word>>8;key=a.key_for_record(tile/G::kWindowTiles*G::kWindowTiles,word&255u);
            if constexpr(D==Domain::OriginalH32Global20)record_valid=code!=h32_record5::Invalid;
          }
          const int64_t column=base+lane_offsets[q]+slot;
          if constexpr(D==Domain::FullSurrogate24) {
            if(a.row_base()+q<out.rows&&column<out.capacity)write_one(q,column,code,key,ord,p0,p1,a,record_valid);
          } else write_one(q,column,code,key,ord,p0,p1,a,record_valid);
        }
      }
      if constexpr(G::kRingRefresh)
        for(uint32_t slot=0;slot<count;++slot)
          shared.records[((warp*G::kQueries+q)*G::kLaneSlots+slot)*32u+lane]=0xffffffffu;
    }
    if constexpr(G::kRingRefresh) {
      __syncwarp();__threadfence_block();
      if(lane<G::kQueries)atomicAdd(shared.ring_sequence+warp*G::kQueries+lane,1u);
      __syncwarp();__threadfence_block();
    }
    lane_counts=0;
  }
};

template<class G,Domain D,class Access>
__device__ __forceinline__ void run_score_emit(Access&a,RecordOutput output,EmitStorage storage) {
#include "score_emit_body.inl"

}
} // namespace dosa_score_emit
