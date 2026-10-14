#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda_fp8.h>
#include "word_pages.cuh"
#include "word_pages_live_nk.cuh"
#include <limits>
namespace test_common_live_nk_host {
void compact_candidates(std::vector<torch::Tensor>streams,torch::Tensor pool,torch::Tensor table,
 torch::Tensor head,torch::Tensor status,torch::Tensor error,torch::Tensor words,torch::Tensor budget,int64_t seed,int64_t nk,torch::Tensor starts,torch::Tensor ends){
 TORCH_CHECK(streams.size()==6&&words.dim()==3&&words.size(2)==4&&pool.dim()==2&&table.dim()==2,"word/page shapes");
 int nq=words.size(0)*4,nw=words.size(1);
 TORCH_CHECK(starts.numel()==nq&&ends.numel()==nq,"original visible intervals");
 TORCH_CHECK(nq>0&&nq%4==0&&seed>=0&&seed<=nk&&seed%256==0&&nw==(nk-seed+31)/32,"Q4/aligned seed/word extent");
 for(const auto&t:{pool,table,head,status,error,words,budget,starts,ends})TORCH_CHECK(t.is_cuda()&&t.device()==words.device()&&t.is_contiguous()&&t.scalar_type()==torch::kInt32,"same-device int32 compaction");
 for(const auto&t:streams)TORCH_CHECK(t.is_cuda()&&t.device()==words.device()&&t.is_contiguous()&&t.scalar_type()==torch::kInt32,"same-device int32 streams");
 TORCH_CHECK(pool.size(1)==128&&pool.size(0)>0&&table.size(0)==nq+nq/4&&table.size(1)>=(nk-seed+127)/128,"page capacities");
 TORCH_CHECK(error.numel()==nq&&budget.numel()==nq/4&&head.numel()==1&&status.numel()==1,"query/status capacities");
 TORCH_CHECK(streams[0].numel()==nq&&streams[3].numel()==nq/4&&streams[2].numel()==1&&streams[5].numel()==1,"stream counts");
 for(int i:{1,4})TORCH_CHECK(streams[i].dim()==2&&streams[i].size(1)==2&&streams[i].size(0)>0,"task capacities");
 test_common_live_nk_legacy::PagedArgs pg{};
 pg.pool=pool.data_ptr<int>();pg.table=table.data_ptr<int>();pg.head=head.data_ptr<int>();pg.status=status.data_ptr<int>();pg.error=error.data_ptr<int>();
 pg.solo_counts=streams[0].data_ptr<int>();pg.solo_tasks=reinterpret_cast<int2*>(streams[1].data_ptr<int>());pg.solo_n=streams[2].data_ptr<int>();
 pg.common_counts=streams[3].data_ptr<int>();pg.common_tasks=reinterpret_cast<int2*>(streams[4].data_ptr<int>());pg.common_n=streams[5].data_ptr<int>();
 pg.words=reinterpret_cast<unsigned*>(words.data_ptr<int>());pg.word_count=nw;pg.nq=nq;pg.nk=nk;pg.max_records=nk-seed;pg.columns=table.size(1);pg.pages=pool.size(0);pg.solo_task_cap=streams[1].size(0);pg.common_task_cap=streams[4].size(0);
 if(!nw)return;
 c10::cuda::CUDAGuard guard(words.device());auto stream=at::cuda::getCurrentCUDAStream();
 test_common_live_nk_legacy::compact_words<<<dim3((nw+2047)/2048,nq/4),256,0,stream>>>(pg,budget.data_ptr<int>(),seed,starts.data_ptr<int>(),ends.data_ptr<int>());C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Independent opt-in ABI. nk_capacity and all tensor shapes remain fixed;
// logical liveNK is read on device each launch. No existing entry is replaced.
void compact_candidates_live_nk(std::vector<torch::Tensor> streams,torch::Tensor pool,torch::Tensor table,
 torch::Tensor head,torch::Tensor status,torch::Tensor error,torch::Tensor words,torch::Tensor budget,
 int64_t seed,int64_t nk_capacity,torch::Tensor starts,torch::Tensor ends,torch::Tensor live_nk){
 TORCH_CHECK(streams.size()==6&&words.dim()==3&&words.size(2)==4&&pool.dim()==2&&table.dim()==2,"liveNK word/page shapes");
 int64_t nq64=words.size(0)*4,nw64=words.size(1);
 TORCH_CHECK(nq64>0&&nq64+nq64/4<=std::numeric_limits<int>::max()&&nq64%4==0,"liveNK Q4 extent");
 TORCH_CHECK(seed>=0&&seed<=nk_capacity&&seed%256==0&&nk_capacity<=(1<<20),"fixed Few capacity and aligned seed");
 TORCH_CHECK(nw64==(nk_capacity-seed+31)/32,"physical word pitch covers fixed capacity");
 int nq=int(nq64),nw=int(nw64);
 for(const auto&t:{pool,head,status,error,words,budget,starts,ends,live_nk})
  TORCH_CHECK(t.is_cuda()&&t.device()==words.device()&&t.is_contiguous()&&t.scalar_type()==torch::kInt32,"same-device contiguous int32 liveNK compaction");
 for(const auto&t:streams)
  TORCH_CHECK(t.is_cuda()&&t.device()==words.device()&&t.is_contiguous()&&t.scalar_type()==torch::kInt32,"same-device int32 streams");
 TORCH_CHECK(table.is_cuda()&&table.device()==words.device()&&table.scalar_type()==torch::kInt32&&table.stride(1)==1&&table.stride(0)>=table.size(1),"physical table row stride");
 TORCH_CHECK(live_nk.dim()==1&&live_nk.numel()==1&&starts.numel()==nq&&ends.numel()==nq,"fixed liveNK scalar and intervals");
 TORCH_CHECK(pool.size(1)==128&&pool.size(0)>0&&pool.size(0)<=std::numeric_limits<int>::max(),"ID page capacity");
 TORCH_CHECK(table.size(0)==nq+nq/4&&table.size(1)>=(nk_capacity-seed+127)/128&&table.size(1)<=std::numeric_limits<int>::max(),"fixed table column capacity");
 TORCH_CHECK(error.numel()==nq&&budget.numel()==nq/4&&head.numel()==1&&status.numel()==1,"query/status capacities");
 TORCH_CHECK(streams[0].numel()==nq&&streams[3].numel()==nq/4&&streams[2].numel()==1&&streams[5].numel()==1,"stream counts");
 for(int i:{1,4})TORCH_CHECK(streams[i].dim()==2&&streams[i].size(1)==2&&streams[i].size(0)>0&&streams[i].size(0)<=std::numeric_limits<int>::max(),"task capacities");
 TORCH_CHECK(nw==0||reinterpret_cast<uintptr_t>(words.data_ptr<int>())%alignof(uint4)==0,"uint4 word alignment");
 fused_few_live_nk::PagedArgs pg{};
 pg.pool=pool.data_ptr<int>();pg.table=table.data_ptr<int>();pg.head=head.data_ptr<int>();pg.status=status.data_ptr<int>();pg.error=error.data_ptr<int>();
 pg.solo_counts=streams[0].data_ptr<int>();pg.solo_tasks=reinterpret_cast<int2*>(streams[1].data_ptr<int>());pg.solo_n=streams[2].data_ptr<int>();
 pg.common_counts=streams[3].data_ptr<int>();pg.common_tasks=reinterpret_cast<int2*>(streams[4].data_ptr<int>());pg.common_n=streams[5].data_ptr<int>();
 pg.words=reinterpret_cast<unsigned*>(words.data_ptr<int>());pg.word_count=nw;pg.nq=nq;pg.nk=nk_capacity;pg.max_records=nk_capacity-seed;
 pg.columns=int(table.size(1));pg.table_stride=table.stride(0);pg.pages=int(pool.size(0));pg.solo_task_cap=int(streams[1].size(0));pg.common_task_cap=int(streams[4].size(0));
 c10::cuda::CUDAGuard guard(words.device());auto stream=at::cuda::getCurrentCUDAStream();
 // Launch even for zero physical words so invalid device NK is not hidden.
 fused_few_live_nk::compact_words_live_nk<<<dim3(1,nq/4),256,0,stream>>>(pg,budget.data_ptr<int>(),int(seed),starts.data_ptr<int>(),ends.data_ptr<int>(),live_nk.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // unique host namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("compact_candidates",&test_common_live_nk_host::compact_candidates);
 m.def("compact_candidates_live_nk",&test_common_live_nk_host::compact_candidates_live_nk);
}

