// Shared execution statements. G, D, a, output and storage are in scope.
// No include guard: every specialization consumes this single body.
  RecordWriter<G,D> writer(output,storage);
  const int tiles=a.tile_count();
  for(int tile=0;tile<tiles;++tile) {
    a.before_tile(tile);
    float scores[G::kQueries]{};
    uint32_t keep_mask=0;
    #pragma unroll
    for(int first=0;first<G::kQueries;first+=G::kReadQueries) {
      if constexpr(G::kReadQueries==1)if(!a.query_active(first)) {
#ifdef DOSA_SCORE_EMIT_AFTER_QUERY
        DOSA_SCORE_EMIT_AFTER_QUERY(a,first);
#endif
        continue;
      }
      float accum[G::kReadQueries*G::kHeads];
      #pragma unroll
      for(int part=0;part<G::kReadQueries*G::kHeads/G::kTmemElements;++part)
        load_tmem<G::kTmemElements>(a.tmem_base(first)+part*G::kTmemElements,
                                  accum+part*G::kTmemElements);
      cutlass::arch::fence_view_async_tmem_load();
      a.after_tmem_read(first+G::kReadQueries==G::kQueries);
      float2 sum0[G::kReadQueries],sum1[G::kReadQueries];
      #pragma unroll
      for(int q=0;q<G::kReadQueries;++q){sum0[q]=make_float2(0,0);sum1[q]=make_float2(0,0);}
      #pragma unroll
      for(int h=0;h<G::kHeads;h+=4) {
        #pragma unroll
        for(int q=0;q<G::kReadQueries;++q) {
          const float*acc=accum+q*G::kHeads;
          const float2 x0=Math<D>::activate(acc[h],acc[h+1]);
          const float2 x1=Math<D>::activate(acc[h+2],acc[h+3]);
          const float2 w0=make_float2(a.weight(first+q,h),a.weight(first+q,h+1));
          const float2 w1=make_float2(a.weight(first+q,h+2),a.weight(first+q,h+3));
          sum0[q]=__ffma2_rn(x0,w0,sum0[q]);sum1[q]=__ffma2_rn(x1,w1,sum1[q]);
        }
      }
      #pragma unroll
      for(int q=0;q<G::kReadQueries;++q) {
        const float2 pair=__fadd2_rn(sum0[q],sum1[q]);
        const int row=first+q;
        const float value=Math<D>::finish(__fadd_rn(pair.x,pair.y),row,a);
        scores[row]=value;
        if(Math<D>::keep(value,row,a))keep_mask|=1u<<row;
      }
#ifdef DOSA_SCORE_EMIT_AFTER_QUERY
      DOSA_SCORE_EMIT_AFTER_QUERY(a,first);
#endif
    }
    writer.consume(tile,tiles,scores,keep_mask,a);
    a.after_emit(tile);
  }
