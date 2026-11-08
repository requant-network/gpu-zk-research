// Requant Network research: device transcript for a complete CUDA Graph.
#include "binary_fused.cu"
extern "C" __global__ void finish_round(const F* partial,F* output,unsigned char* digest,F* challenge,int blocks){
    __shared__ F sums[3][256];int t=threadIdx.x;
    for(int c=0;c<3;c++){
        F sum{0,0};for(int i=t;i<blocks;i+=256)sum=add(sum,partial[3*i+c]);sums[c][t]=sum;
    }__syncthreads();
    for(int step=128;step;step>>=1){if(t<step)for(int c=0;c<3;c++)sums[c][t]=add(sums[c][t],sums[c][t+step]);__syncthreads();}
    if(t==0){
        for(int c=0;c<3;c++)output[c]=sums[c][0];
        transcript_hash(digest,sums[0][0],sums[1][0],sums[2][0]);
        F r{0,0};for(int i=0;i<8;i++){r.lo|=U(digest[i])<<(8*i);r.hi|=U(digest[8+i])<<(8*i);}challenge[0]=r;
    }
}
extern "C" __global__ void fold_challenge(const F* a,const F* b,F* aa,F* bb,const F* challenge,int pairs){
    int i=blockIdx.x*256+threadIdx.x;
    if(i<pairs){F r=challenge[0];
        aa[i]=add(a[2*i],accelerated(r,add(a[2*i],a[2*i+1])));
        bb[i]=add(b[2*i],accelerated(r,add(b[2*i],b[2*i+1])));
    }
}
