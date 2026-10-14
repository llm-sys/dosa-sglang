// Included inside each caller's existing namespace to preserve native symbols.
// Finite E4 magnitudes are magnitude_unit(code)/512; sign is irrelevant.
// The floating estimate only proposes a code. Integer checks certify the bound.
__device__ __forceinline__ uint32_t magnitude_unit(uint32_t code) {
  const uint32_t exponent=code>>3, mantissa=code&7u;
  return exponent==0?mantissa:(8u+mantissa)<<(exponent-1);
}

__device__ __forceinline__ uint32_t max4_magnitude(
    uint32_t packed,uint32_t& invalid) {
  uint32_t result=0;
  #pragma unroll
  for(int byte=0;byte<4;++byte){
    const uint32_t code=(packed>>(8*byte))&0x7fu;
    invalid|=code==0x7fu;
    result=max(result,code);
  }
  return result;
}

__device__ __forceinline__ unsigned long long sum4_square_units(uint32_t packed) {
  unsigned long long result=0;
  #pragma unroll
  for(int byte=0;byte<4;++byte){
    const uint32_t code=(packed>>(8*byte))&0x7fu;
    if(code!=0x7fu){
      const unsigned long long unit=magnitude_unit(code);
      result+=unit*unit;
    }
  }
  return result;
}

__device__ __forceinline__ uint32_t exact_rms_code(
    unsigned long long sum_square,uint32_t kappa,int dimensions) {
  // Floating point proposes a code; exact current/previous tests prove minimality.
  const float estimate=sqrtf(float(sum_square)/float(dimensions))*(1.0f/512.0f);
  uint32_t candidate=uint32_t(
      __nv_cvt_float_to_fp8(estimate,__NV_SATFINITE,__NV_E4M3));
  unsigned long long unit=magnitude_unit(candidate);
  if(static_cast<unsigned long long>(dimensions)*unit*unit<sum_square)++candidate;
  unit=magnitude_unit(candidate);
  const unsigned long long previous=magnitude_unit(candidate?candidate-1:0);
  if(candidate<=kappa
      &&static_cast<unsigned long long>(dimensions)*unit*unit>=sum_square
      &&(candidate==0
          ||static_cast<unsigned long long>(dimensions)*previous*previous<sum_square))
    return candidate;
  // An inaccurate proposal retains the original integer search below.
  uint32_t lo=0,hi=kappa;
  #pragma unroll
  for(int iteration=0;iteration<7;++iteration){
    const uint32_t mid=(lo+hi)>>1;
    const unsigned long long unit=magnitude_unit(mid);
    const bool enough=static_cast<unsigned long long>(dimensions)*unit*unit>=sum_square;
    if(lo<hi){
      if(enough)hi=mid;
      else lo=mid+1;
    }
  }
  return lo;
}
