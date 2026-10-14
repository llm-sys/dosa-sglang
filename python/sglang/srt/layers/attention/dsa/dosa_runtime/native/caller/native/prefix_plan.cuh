// Same input scan, publication and state ABI. Only partial representation and
// the final-CTA threshold/offset calculation differ from the original plan.
__global__ void carry_votes_plan_prefix_kernel(
    const int32_t* __restrict__ votes,int count,int min_index,int out_k,
    int max_vote,volatile int16_t* __restrict__ partial,int partial_stride,
    int32_t* __restrict__ state) {
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,block=blockIdx.x;
  const int begin=block*kCarryTileItems,end=min(begin+kCarryTileItems,count);
  const int bins=max_vote+1;
  extern __shared__ uint32_t freq[];
  __shared__ int warp_sum[kCarryWarps],scan_base,last,threshold;
  __shared__ int block_gt[kCarryMaxBlocks],block_eq[kCarryMaxBlocks];
  for(int b=tid;b<bins;b+=kCarryThreads)freq[b]=0;
  __syncthreads();
  int zeros=0;
  for(int index=begin+tid;index<end;index+=kCarryThreads) {
    if(index<min_index)continue;
    int value=votes[index];value=value<0?0:(value>max_vote?max_vote:value);
    if(value==0)++zeros;else atomicAdd(freq+value,1u);
  }
  zeros=carry_warp_sum(zeros);
  if(lane==0)warp_sum[warp]=zeros;
  __syncthreads();
  if(warp==0) {
    int value=lane<kCarryWarps?warp_sum[lane]:0;value=carry_warp_sum(value);
    if(lane==0)freq[0]=static_cast<uint32_t>(value);
  }
  if(tid==0)scan_base=0;
  __syncthreads();
  volatile int16_t* row=partial+static_cast<size_t>(block)*partial_stride;
  // P_b(v) is at most8192, even though it is a cumulative rather than raw bin.
  for(int tile=0;tile<bins;tile+=kCarryThreads) {
    const int bin=max_vote-tile-tid;
    int inclusive=bin>=0?static_cast<int>(freq[bin]):0;
    #pragma unroll
    for(int delta=1;delta<32;delta<<=1) {
      int other=__shfl_up_sync(0xffffffffu,inclusive,delta);
      if(lane>=delta)inclusive+=other;
    }
    if(lane==31)warp_sum[warp]=inclusive;
    __syncthreads();
    int before=0;
    #pragma unroll
    for(int w=0;w<kCarryWarps;++w)if(w<warp)before+=warp_sum[w];
    if(bin>=0)row[bin]=static_cast<int16_t>(scan_base+before+inclusive);
    __syncthreads();
    if(tid==0) {
      int total=0;
      #pragma unroll
      for(int w=0;w<kCarryWarps;++w)total+=warp_sum[w];
      scan_base+=total;
    }
    __syncthreads();
  }
  // Preserve the original every-writer release before the last-block ticket.
  __threadfence();
  __syncthreads();
  if(tid==0)last=atomicAdd(state+kCarryTicket,1)==gridDim.x-1;
  __syncthreads();
  if(!last)return;

  if(warp==0) {
    int lo=0,hi=bins; // S(lo)>=K, S(hi)=0<K (virtual high endpoint).
    while(hi-lo>1) {
      int mid=(lo+hi)/2,total=0;
      for(int b=lane;b<gridDim.x;b+=32)
        total+=static_cast<int>(partial[static_cast<size_t>(b)*partial_stride+mid]);
      total=carry_warp_sum(total);total=__shfl_sync(0xffffffffu,total,0);
      if(total>=out_k)lo=mid;else hi=mid;
    }
    if(lane==0)threshold=lo;
  }
  __syncthreads();
  if(tid<gridDim.x) {
    const size_t base=static_cast<size_t>(tid)*partial_stride;
    int ge=static_cast<int>(partial[base+threshold]);
    int gt=threshold<max_vote?static_cast<int>(partial[base+threshold+1]):0;
    block_gt[tid]=gt;block_eq[tid]=ge-gt;
  }
  __syncthreads();
  if(tid==0) {
    int gt=0;
    for(int b=0;b<gridDim.x;++b)gt+=block_gt[b];
    int need=out_k-gt,remaining=need,offset=0,tie_block=gridDim.x-1,tie_take=0;
    for(int b=0;b<gridDim.x;++b) {
      state[kCarryBlockOffsets+b]=offset;
      if(remaining>0&&remaining<=block_eq[b]) {tie_block=b;tie_take=remaining;}
      const int take=min(block_eq[b],max(remaining,0));
      offset+=block_gt[b]+take;remaining-=block_eq[b];
    }
    state[kCarryBlockOffsets+gridDim.x]=offset;
    state[kCarryThreshold]=threshold;state[kCarryTieBlock]=tie_block;
    state[kCarryTieTake]=tie_take;state[kCarryOutK]=out_k;state[kCarryNumBlocks]=gridDim.x;
    __threadfence();atomicExch(state+kCarryTicket,0);
  }
}
