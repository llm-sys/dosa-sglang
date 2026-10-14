#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include "../common/record5_ops.cuh"

namespace record4_integer {
constexpr int Top=2048,Threads=256,Bins=4096;
constexpr unsigned Full=0xFFFFFFFFu,IdMask=0xFFFFFu;
struct Args {
  const int *words,*counts,*starts,*ends,*pool;
  const float *epsilon;
  int *error,*output,*status,*diag,*boundary_ids,*boundary_counts;
  float *temporary;
  int *temporary_ids,*temporary_counts,*done;
  int rows,cap,nk,boundary_stride,temporary_stride;
  const uint8_t* low;
};

__device__ void fail(const Args& a,int row,unsigned flags,int raw_count) {
  if(threadIdx.x==0) {
    atomicOr(a.error+row,1);
    a.status[row]=int(flags);
    a.done[row]=0;
    a.boundary_counts[row]=0;
    a.temporary_counts[row]=0;
    for(int j=0;j<5;++j)a.diag[int64_t(row)*5+j]=-1;
    a.diag[int64_t(row)*5+2]=raw_count;
  }
  for(int j=threadIdx.x;j<Top;j+=Threads)a.output[int64_t(row)*Top+j]=-1;
}

template<bool Extended>
__global__ __launch_bounds__(Threads) void select(Args a) {
  const int row=a.rows-1-int(blockIdx.x),t=threadIdx.x,warp=t/32,lane=t%32;
  const int count=a.counts[row];
  __shared__ unsigned hist[Bins],warp_count[8],bad;
  __shared__ int winning_warp,group_before,pivot,certain,uncertain,written_b,written_e;
  if(t==0) {
    bad=0;
    if(count<Top||count>a.cap)bad|=1;
    if(a.starts[row]<0||a.ends[row]<a.starts[row]||a.ends[row]>a.nk)bad|=4;
    // This operation is only the exact after-H32 staircase domain. It cannot
    // silently consume the old full surrogate or a nonzero uncertainty radius.
    if(a.error[row]!=0||*a.pool!=0||record5::Policy<record5::Domain::CompletedH32Global20>::radius_invalid(a.epsilon[row]))bad|=16;
    winning_warp=-1;group_before=0;pivot=-1;certain=uncertain=0;
    written_b=written_e=0;
  }
  for(int bin=t;bin<Bins;bin+=Threads)hist[bin]=0;
  __syncthreads();
  if(bad){fail(a,row,bad,count);return;}

  const int64_t base=int64_t(row)*a.cap;
  // These disjoint input tensors are immutable for the operation. Cache the
  // row bounds once instead of reloading both for every validated record.
  const unsigned row_start=Extended?unsigned(a.starts[row]):0u;
  const unsigned row_end=Extended?unsigned(a.ends[row]):0u;
  if constexpr(Extended) {
    DOSA_RECORD5_INDEPENDENT_WORDS(Threads,t,a.words,base,count,word,slot,
      const unsigned cp=word>>20,id=word&IdMask;
      if(cp<0x8u||cp>0xFF7u) {atomicOr(&bad,2u);continue;}
      if(id>=unsigned(a.nk)||id<row_start||id>=row_end) {
        atomicOr(&bad,4u);continue;
      }
      atomicAdd(hist+cp,1u);
    );
  } else {
    for(int slot=t;slot<count;slot+=Threads) {
      const unsigned word=unsigned(a.words[base+slot]);
      const unsigned cp=word>>20,id=word&IdMask;
      if(cp>0xC00u) {atomicOr(&bad,2u);continue;}
      if(id>=unsigned(a.nk)||id<unsigned(a.starts[row])||
          id>=unsigned(a.ends[row])) {
        atomicOr(&bad,4u);continue;
      }
      atomicAdd(hist+cp,1u);
    }
  }
  __syncthreads();
  if(bad){fail(a,row,bad,count);return;}

  // Each warp reads one contiguous512-bin group, coalesced at each step.
  unsigned local=0;
  #pragma unroll
  for(int part=0;part<16;++part)local+=hist[warp*512+part*32+lane];
  const unsigned group_sum=__reduce_add_sync(Full,local);
  if(lane==0)warp_count[warp]=group_sum;
  __syncthreads();
  if(t==0) {
    unsigned before=0;
    for(int w=0;w<8;++w) {
      if(before<Top&&Top<=before+warp_count[w]) {
        winning_warp=w;group_before=int(before);break;
      }
      before+=warp_count[w];
    }
    if(winning_warp<0)bad|=16;
  }
  __syncthreads();
  if(bad){fail(a,row,bad,count);return;}

  // Only the selected full warp searches its16 contiguous chunks. All branches
  // are warp-uniform; no CTA barrier occurs inside this selected-warp region.
  if(warp==winning_warp) {
    unsigned rank=Top-group_before,seen=0;
    for(int part=0;part<16;++part) {
      const unsigned value=hist[warp*512+part*32+lane];
      const unsigned chunk=__reduce_add_sync(Full,value);
      if(rank>chunk){rank-=chunk;seen+=chunk;continue;}
      unsigned inclusive=value;
      #pragma unroll
      for(int offset=1;offset<32;offset<<=1) {
        const unsigned other=__shfl_up_sync(Full,inclusive,offset);
        if(lane>=offset)inclusive+=other;
      }
      const unsigned mask=__ballot_sync(Full,inclusive>=rank);
      const int winner=__ffs(mask)-1;
      const unsigned selected_count=__shfl_sync(Full,value,winner);
      const unsigned selected_before=__shfl_sync(Full,inclusive-value,winner);
      if(lane==0) {
        pivot=warp*512+part*32+winner;
        certain=group_before+int(seen+selected_before);
        uncertain=int(selected_count);
      }
      break;
    }
  }
  __syncthreads();
  if constexpr(Extended) {
    __shared__ unsigned fine_hist[256];
    // The coarse histogram is dead. Reuse its4096 words for the selected cell
    // when it fits; otherwise retain the original full third-pass algorithm.
    if(t==0) {group_before=uncertain;winning_warp=uncertain<=Bins;}
    fine_hist[t]=0;
    __syncthreads();
    const unsigned coarse=unsigned(pivot);
    if(winning_warp) {
      auto collect=[&](unsigned word,int slot,bool live) {
        const unsigned hi=word>>20,id=word&IdMask;
        const bool is_b=live&&hi<coarse,is_e=live&&hi==coarse;
        const unsigned bm=__ballot_sync(Full,is_b),em=__ballot_sync(Full,is_e);
        if(!(bm|em))return;
        DOSA_RECORD5_RESERVE(Full,lane,bm,em,&written_b,&written_e,bbase,ebase,lower);
        if(is_b) {
          const int at=bbase+__popc(bm&lower);
          if(at<Top)a.output[int64_t(row)*Top+at]=int(id);
          else atomicOr(&bad,16u);
        }
        if(is_e) {
          const unsigned low=unsigned(a.low[base+slot]);
          const int at=ebase+__popc(em&lower);
          if(at<Bins)hist[at]=(low<<20)|id;
          else atomicOr(&bad,16u);
          atomicAdd(fine_hist+low,1u);
        }
      };
      DOSA_RECORD5_COLLECTIVE_WORDS(Threads,warp,lane,a.words,base,count,collect);
    } else {
      for(int slot=t;slot<count;slot+=Threads) {
        const unsigned word=unsigned(a.words[base+slot]);
        if((word>>20)==coarse)atomicAdd(fine_hist+unsigned(a.low[base+slot]),1u);
      }
    }
    __syncthreads();
    if(t==0&&winning_warp&&(written_b!=certain||written_e!=group_before))bad|=16;
    __syncthreads();
    if(bad){fail(a,row,bad,count);return;}
    if(warp==0) {
      unsigned rank=Top-certain,seen=0;
      for(int part=0;part<8;++part) {
        const unsigned value=fine_hist[part*32+lane];
        const unsigned chunk=__reduce_add_sync(Full,value);
        if(rank>chunk){rank-=chunk;seen+=chunk;continue;}
        unsigned inclusive=value;
        #pragma unroll
        for(int offset=1;offset<32;offset<<=1) {
          const unsigned other=__shfl_up_sync(Full,inclusive,offset);
          if(lane>=offset)inclusive+=other;
        }
        const unsigned mask=__ballot_sync(Full,inclusive>=rank);
        const int winner=__ffs(mask)-1;
        const unsigned selected_count=__shfl_sync(Full,value,winner);
        const unsigned selected_before=__shfl_sync(Full,inclusive-value,winner);
        if(lane==0) {
          pivot=int((coarse<<8)|unsigned(part*32+winner));
          certain+=int(seen+selected_before);
          uncertain=int(selected_count);
        }
        break;
      }
    }
    __syncthreads();
    if(t==0&&winning_warp)written_e=0;
  }
  if(t==0) {
    const bool invalid_pivot=Extended?(pivot<0x800||pivot>0xFF7FF):(pivot<0||pivot>0xC00);
    if(invalid_pivot||certain<0||certain>=Top||uncertain<=0||certain+uncertain<Top)bad|=16;
    if(uncertain>a.boundary_stride||certain+uncertain>a.temporary_stride)bad|=32;
  }
  __syncthreads();
  if(bad){fail(a,row,bad,count);return;}

  if(!Extended||!winning_warp) {
  for(int begin=warp*32;begin<count;begin+=Threads) {
    const int slot=begin+lane;
    const bool live=slot<count;
    const unsigned word=live?unsigned(a.words[base+slot]):0u;
    const unsigned coarse=word>>20,id=word&IdMask;
    unsigned cp=coarse;
    if constexpr(Extended) {
      cp=coarse<<8;
      if(live&&coarse==(unsigned(pivot)>>8))cp|=unsigned(a.low[base+slot]);
    }
    const bool is_b=live&&cp<unsigned(pivot),is_e=live&&cp==unsigned(pivot);
    const unsigned bm=__ballot_sync(Full,is_b),em=__ballot_sync(Full,is_e);
    int bbase=0,ebase=0;
    if(lane==0) {
      if(bm)bbase=atomicAdd(&written_b,__popc(bm));
      if(em)ebase=atomicAdd(&written_e,__popc(em));
    }
    bbase=__shfl_sync(Full,bbase,0);ebase=__shfl_sync(Full,ebase,0);
    const unsigned lower=lane?((1u<<lane)-1u):0u;
    if(is_b) {
      const int at=bbase+__popc(bm&lower);
      if(at<Top)a.output[int64_t(row)*Top+at]=int(id);
      else atomicOr(&bad,16u);
    }
    if(is_e) {
      const int at=ebase+__popc(em&lower);
      if(at<a.boundary_stride)a.boundary_ids[int64_t(row)*a.boundary_stride+at]=int(id);
      else atomicOr(&bad,16u);
    }
  }
  }
  if constexpr(Extended) {
    if(winning_warp) {
      // All cached words are immutable until this loop has drained. Fine E
      // goes to the separate global boundary tensor, never into this cache.
      for(int begin=warp*32;begin<group_before;begin+=Threads) {
        const int at=begin+lane;const bool live=at<group_before;
        const unsigned word=live?hist[at]:0u;
        const unsigned code=word>>20,id=word&IdMask;
        const bool is_b=live&&code<(unsigned(pivot)&255u);
        const bool is_e=live&&code==(unsigned(pivot)&255u);
        const unsigned bm=__ballot_sync(Full,is_b),em=__ballot_sync(Full,is_e);
        int bbase=0,ebase=0;
        if(lane==0) {
          if(bm)bbase=atomicAdd(&written_b,__popc(bm));
          if(em)ebase=atomicAdd(&written_e,__popc(em));
        }
        bbase=__shfl_sync(Full,bbase,0);ebase=__shfl_sync(Full,ebase,0);
        const unsigned lower=lane?((1u<<lane)-1u):0u;
        if(is_b) {
          const int dst=bbase+__popc(bm&lower);
          if(dst<Top)a.output[int64_t(row)*Top+dst]=int(id);
          else atomicOr(&bad,16u);
        }
        if(is_e) {
          const int dst=ebase+__popc(em&lower);
          if(dst<a.boundary_stride)a.boundary_ids[int64_t(row)*a.boundary_stride+dst]=int(id);
          else atomicOr(&bad,16u);
        }
      }
    }
  }
  __syncthreads();
  if(t==0&&(written_b!=certain||written_e!=uncertain))bad|=16;
  __syncthreads();
  if(bad){fail(a,row,bad,count);return;}
  if(uncertain>128) {
    for(int j=t;j<certain;j+=Threads) {
      a.temporary[int64_t(row)*a.temporary_stride+j]=FLT_MAX;
      a.temporary_ids[int64_t(row)*a.temporary_stride+j]=a.output[int64_t(row)*Top+j];
    }
  }
  __syncthreads();
  if(t==0) {
    a.status[row]=0;
    a.boundary_counts[row]=uncertain;
    a.temporary_counts[row]=uncertain<=128?0:certain;
    int* diag=a.diag+int64_t(row)*5;
    diag[0]=certain;diag[1]=uncertain;diag[2]=count;diag[3]=certain+uncertain;diag[4]=pivot;
    a.done[row]=1;
  }
}
}

static bool overlaps(const torch::Tensor& a,const torch::Tensor& b) {
  if(a.numel()==0||b.numel()==0)return false;
  const uintptr_t x=reinterpret_cast<uintptr_t>(a.data_ptr()),y=reinterpret_cast<uintptr_t>(b.data_ptr());
  return x<y+b.numel()*b.element_size()&&y<x+a.numel()*a.element_size();
}

template<bool Extended>
void select_dense_impl(torch::Tensor low,torch::Tensor packed,torch::Tensor counts,torch::Tensor starts,torch::Tensor ends,
    torch::Tensor epsilon,torch::Tensor pool_status,torch::Tensor error,torch::Tensor output,
    torch::Tensor status,torch::Tensor diag,torch::Tensor boundary_ids,torch::Tensor boundary_counts,
    torch::Tensor temporary,torch::Tensor temporary_ids,torch::Tensor temporary_counts,torch::Tensor done,int64_t nk) {
  std::vector<torch::Tensor> tensors={packed,counts,starts,ends,epsilon,pool_status,error,output,status,
      diag,boundary_ids,boundary_counts,temporary,temporary_ids,temporary_counts,done};
  if constexpr(Extended) {
    TORCH_CHECK(low.scalar_type()==torch::kByte&&low.sizes()==packed.sizes(),"record5 byte plane extent");
    tensors.push_back(low);
  }
  for(const auto& x:tensors)TORCH_CHECK(x.is_cuda()&&x.device()==packed.device()&&x.is_contiguous(),"record4 integer device/layout");
  TORCH_CHECK(packed.dim()==2&&packed.size(0)>0&&packed.size(1)>=2048&&packed.size(1)<=(1<<20),"record4 dense layout");
  const int rows=packed.size(0),cap=packed.size(1);
  for(const auto&x:{packed,counts,starts,ends,pool_status,error,output,status,diag,boundary_ids,boundary_counts,temporary_ids,temporary_counts,done})
    TORCH_CHECK(x.scalar_type()==torch::kInt,"record4 integer dtype");
  for(const auto&x:{counts,starts,ends,epsilon,error,status,boundary_counts,temporary_counts,done})
    TORCH_CHECK(x.numel()==rows,"record4 row metadata");
  TORCH_CHECK(epsilon.scalar_type()==torch::kFloat&&pool_status.numel()==1&&output.sizes()==torch::IntArrayRef({rows,2048})&&
      diag.numel()>=int64_t(rows)*5&&boundary_ids.sizes()==torch::IntArrayRef({rows,4096})&&
      temporary.dim()==2&&temporary.size(0)==rows&&temporary.size(1)>=6144&&temporary.scalar_type()==torch::kFloat&&
      temporary_ids.sizes()==temporary.sizes()&&nk>0&&nk<=(1<<20),"record4 workspace extent");
  for(size_t i=0;i<tensors.size();++i)for(size_t j=i+1;j<tensors.size();++j)
    TORCH_CHECK(!overlaps(tensors[i],tensors[j]),"record4 operand alias");
  record4_integer::Args args{packed.data_ptr<int>(),counts.data_ptr<int>(),starts.data_ptr<int>(),ends.data_ptr<int>(),
      pool_status.data_ptr<int>(),epsilon.data_ptr<float>(),error.data_ptr<int>(),output.data_ptr<int>(),status.data_ptr<int>(),
      diag.data_ptr<int>(),boundary_ids.data_ptr<int>(),boundary_counts.data_ptr<int>(),temporary.data_ptr<float>(),
      temporary_ids.data_ptr<int>(),temporary_counts.data_ptr<int>(),done.data_ptr<int>(),rows,cap,int(nk),4096,int(temporary.size(1)),Extended?low.data_ptr<uint8_t>():nullptr};
  c10::cuda::CUDAGuard guard(packed.device());
  record4_integer::select<Extended><<<rows,256,0,c10::cuda::getCurrentCUDAStream()>>>(args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void select_dense(torch::Tensor packed,torch::Tensor counts,torch::Tensor starts,torch::Tensor ends,
    torch::Tensor epsilon,torch::Tensor pool_status,torch::Tensor error,torch::Tensor output,
    torch::Tensor status,torch::Tensor diag,torch::Tensor boundary_ids,torch::Tensor boundary_counts,
    torch::Tensor temporary,torch::Tensor temporary_ids,torch::Tensor temporary_counts,torch::Tensor done,int64_t nk) {
  select_dense_impl<false>({},packed,counts,starts,ends,epsilon,pool_status,error,output,status,diag,
      boundary_ids,boundary_counts,temporary,temporary_ids,temporary_counts,done,nk);
}

void select_dense5(torch::Tensor low,torch::Tensor packed,torch::Tensor counts,torch::Tensor starts,torch::Tensor ends,
    torch::Tensor epsilon,torch::Tensor pool_status,torch::Tensor error,torch::Tensor output,
    torch::Tensor status,torch::Tensor diag,torch::Tensor boundary_ids,torch::Tensor boundary_counts,
    torch::Tensor temporary,torch::Tensor temporary_ids,torch::Tensor temporary_counts,torch::Tensor done,int64_t nk) {
  select_dense_impl<true>(low,packed,counts,starts,ends,epsilon,pool_status,error,output,status,diag,
      boundary_ids,boundary_counts,temporary,temporary_ids,temporary_counts,done,nk);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("select_dense",&select_dense);
  m.def("select_dense5",&select_dense5);
}
