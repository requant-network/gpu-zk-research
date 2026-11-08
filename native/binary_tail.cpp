// Requant Network research: x86 PCLMUL suffix for binary-field sumcheck.
#include <immintrin.h>
#include <cstdint>
using U=std::uint64_t;
struct F{U lo,hi;};
static F add(F a,F b){return {a.lo^b.lo,a.hi^b.hi};}
static F multiply(F a,F b){
    __m128i x=_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a));
    __m128i y=_mm_loadu_si128(reinterpret_cast<const __m128i*>(&b));
    __m128i z0=_mm_clmulepi64_si128(x,y,0x00),z2=_mm_clmulepi64_si128(x,y,0x11);
    __m128i xx=_mm_xor_si128(x,_mm_shuffle_epi32(x,0x4e)),yy=_mm_xor_si128(y,_mm_shuffle_epi32(y,0x4e));
    __m128i z1=_mm_xor_si128(_mm_clmulepi64_si128(xx,yy,0x00),_mm_xor_si128(z0,z2));
    F low,middle,high;
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&low),z0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&middle),z1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&high),z2);
    U l0=low.lo,l1=low.hi^middle.lo,h0=high.lo^middle.hi,h1=high.hi;
    U r0=h0^(h0<<1)^(h0<<2)^(h0<<7);
    U r1=h1^(h1<<1)^(h1<<2)^(h1<<7)^(h0>>63)^(h0>>62)^(h0>>57);
    U overflow=(h1>>63)^(h1>>62)^(h1>>57);
    return {l0^r0^overflow^(overflow<<1)^(overflow<<2)^(overflow<<7),l1^r1};
}
extern "C" int has_pclmul(){return __builtin_cpu_supports("pclmul")?1:0;}
extern "C" void cpu_round(const F* a,const F* b,F* out,int n){
    F c0{0,0},c1{0,0},c2{0,0};
    for(int i=0;i<n/2;i++){
        F da=add(a[2*i],a[2*i+1]),db=add(b[2*i],b[2*i+1]);
        c0=add(c0,multiply(a[2*i],b[2*i]));
        c1=add(c1,add(multiply(a[2*i],db),multiply(b[2*i],da)));
        c2=add(c2,multiply(da,db));
    }
    out[0]=c0;out[1]=c1;out[2]=c2;
}
extern "C" void cpu_fold(F* a,F* b,int n,U rlo,U rhi){
    F r{rlo,rhi};
    for(int i=0;i<n/2;i++){
        a[i]=add(a[2*i],multiply(r,add(a[2*i],a[2*i+1])));
        b[i]=add(b[2*i],multiply(r,add(b[2*i],b[2*i+1])));
    }
}
