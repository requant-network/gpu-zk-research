// Requant Network research. GF(2^128), polynomial x^128+x^7+x^2+x+1.
using U = unsigned long long;
struct F { U lo, hi; };
__device__ F add(F a, F b) { return {a.lo^b.lo, a.hi^b.hi}; }
__device__ F carry(U a, U b) {
    F out;
    asm("clmad.lo.u64 %0, %1, %2, 0;" : "=l"(out.lo) : "l"(a), "l"(b));
    asm("clmad.hi.u64 %0, %1, %2, 0;" : "=l"(out.hi) : "l"(a), "l"(b));
    return out;
}
__device__ F accelerated(F a, F b) {
    F z0=carry(a.lo,b.lo), z2=carry(a.hi,b.hi);
    F z1=add(add(carry(a.lo^a.hi,b.lo^b.hi),z0),z2);
    U l0=z0.lo, l1=z0.hi^z1.lo, h0=z2.lo^z1.hi, h1=z2.hi;
    U r0=h0^(h0<<1)^(h0<<2)^(h0<<7);
    U r1=h1^(h1<<1)^(h1<<2)^(h1<<7)^(h0>>63)^(h0>>62)^(h0>>57);
    U overflow=(h1>>63)^(h1>>62)^(h1>>57);
    return {l0^r0^overflow^(overflow<<1)^(overflow<<2)^(overflow<<7),l1^r1};
}
__device__ F bitserial(F a, F b) {
    F out{0,0};
    for(int i=0;i<128;i++) {
        U mask=U(0)-(b.lo&1);
        out.lo^=a.lo&mask; out.hi^=a.hi&mask;
        b.lo=(b.lo>>1)|(b.hi<<63); b.hi>>=1;
        U high=a.hi>>63;
        a.hi=(a.hi<<1)|(a.lo>>63); a.lo=(a.lo<<1)^(0x87ULL*(high));
    }
    return out;
}
__device__ F mul(F a,F b,int mode) { return mode ? accelerated(a,b) : bitserial(a,b); }
extern "C" __global__ void field_products(const F* a,const F* b,F* out,int n,int mode) {
    int i=blockIdx.x*256+threadIdx.x;
    if(i<n)out[i]=mul(a[i],b[i],mode);
}
extern "C" __global__ void round_coefficients(const F* a,const F* b,F* partial,int pairs,int mode) {
    __shared__ F sums[3][256];
    int t=threadIdx.x,i=blockIdx.x*256+t;
    F c0{0,0},c1{0,0},c2{0,0};
    if(i<pairs) {
        F da=add(a[2*i],a[2*i+1]),db=add(b[2*i],b[2*i+1]);
        c0=mul(a[2*i],b[2*i],mode);
        c1=add(mul(a[2*i],db,mode),mul(b[2*i],da,mode));
        c2=mul(da,db,mode);
    }
    sums[0][t]=c0;sums[1][t]=c1;sums[2][t]=c2;
    __syncthreads();
    for(int k=128;k;k>>=1) {
        if(t<k)for(int c=0;c<3;c++)sums[c][t]=add(sums[c][t],sums[c][t+k]);
        __syncthreads();
    }
    if(t==0)for(int c=0;c<3;c++)partial[blockIdx.x*3+c]=sums[c][0];
}
extern "C" __global__ void field_fold(const F* a,const F* b,F* aa,F* bb,int pairs,U rlo,U rhi,int mode) {
    int i=blockIdx.x*256+threadIdx.x;
    if(i<pairs) {
        F r{rlo,rhi};
        aa[i]=add(a[2*i],mul(r,add(a[2*i],a[2*i+1]),mode));
        bb[i]=add(b[2*i],mul(r,add(b[2*i],b[2*i+1]),mode));
    }
}
