// Requant Network research. Fixed-message SHA-256 and transcript-dependent GPU tail.
#include "binary_fields.cu"
using V=unsigned int;
__device__ V rotate(V x,int n){return (x>>n)|(x<<(32-n));}
__device__ void transcript_hash(unsigned char* digest,F c0,F c1,F c2){
    const V constants[64]={
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    unsigned char message[128]={0};
    for(int i=0;i<32;i++)message[i]=digest[i];
    F cs[3]={c0,c1,c2};
    for(int c=0;c<3;c++)for(int i=0;i<8;i++){
        message[32+c*16+i]=(cs[c].lo>>(8*i))&255;
        message[40+c*16+i]=(cs[c].hi>>(8*i))&255;
    }
    message[80]=0x80;message[126]=2;message[127]=0x80;
    V state[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    for(int block=0;block<2;block++){
        V w[64];
        for(int i=0;i<16;i++){int j=block*64+4*i;w[i]=(V(message[j])<<24)|(V(message[j+1])<<16)|(V(message[j+2])<<8)|message[j+3];}
        for(int i=16;i<64;i++){
            V a=w[i-15],b=w[i-2];
            w[i]=w[i-16]+(rotate(a,7)^rotate(a,18)^(a>>3))+w[i-7]+(rotate(b,17)^rotate(b,19)^(b>>10));
        }
        V a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
        for(int i=0;i<64;i++){
            V t1=h+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^((~e)&g))+constants[i]+w[i];
            V t2=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));
            h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
    }
    for(int i=0;i<32;i++)digest[i]=(state[i/4]>>(24-8*(i%4)))&255;
}
extern "C" __global__ void check_transcript_hash(const unsigned char* input,const F* coefficients,unsigned char* out,int n){
    int i=blockIdx.x*256+threadIdx.x;
    if(i<n){unsigned char digest[32];for(int j=0;j<32;j++)digest[j]=input[32*i+j];
        transcript_hash(digest,coefficients[3*i],coefficients[3*i+1],coefficients[3*i+2]);
        for(int j=0;j<32;j++)out[32*i+j]=digest[j];}
}
extern "C" __global__ void fused_tail(const F* input_a,const F* input_b,const unsigned char* input_digest,F* output,int n){
    __shared__ F a[1024],b[1024],sums[3][256],r;
    __shared__ unsigned char digest[32];
    int t=threadIdx.x;
    for(int i=t;i<n;i+=256){a[i]=input_a[i];b[i]=input_b[i];}
    if(t<32)digest[t]=input_digest[t];__syncthreads();
    int round=0;
    while(n>1){
        F c0{0,0},c1{0,0},c2{0,0};int pairs=n/2;
        for(int i=t;i<pairs;i+=256){
            F da=add(a[2*i],a[2*i+1]),db=add(b[2*i],b[2*i+1]);
            c0=add(c0,accelerated(a[2*i],b[2*i]));
            c1=add(c1,add(accelerated(a[2*i],db),accelerated(b[2*i],da)));
            c2=add(c2,accelerated(da,db));
        }
        sums[0][t]=c0;sums[1][t]=c1;sums[2][t]=c2;__syncthreads();
        for(int step=128;step;step>>=1){if(t<step)for(int c=0;c<3;c++)sums[c][t]=add(sums[c][t],sums[c][t+step]);__syncthreads();}
        if(t==0){
            for(int c=0;c<3;c++)output[3*round+c]=sums[c][0];
            transcript_hash(digest,sums[0][0],sums[1][0],sums[2][0]);
            r={0,0};for(int i=0;i<8;i++){r.lo|=U(digest[i])<<(8*i);r.hi|=U(digest[8+i])<<(8*i);}
        }__syncthreads();
        F aa[2],bb[2];
        for(int i=t,j=0;i<pairs;i+=256,j++){
            aa[j]=add(a[2*i],accelerated(r,add(a[2*i],a[2*i+1])));
            bb[j]=add(b[2*i],accelerated(r,add(b[2*i],b[2*i+1])));
        }__syncthreads();
        for(int i=t,j=0;i<pairs;i+=256,j++){a[i]=aa[j];b[i]=bb[j];}__syncthreads();
        n=pairs;round++;
    }
}
// Global scratch permits a larger suffix without requiring a shared-memory fit.
extern "C" __global__ void global_tail(const F* input_a,const F* input_b,const unsigned char* input_digest,
 F* output,F* a0,F* b0,F* a1,F* b1,int n){
    __shared__ F sums[3][256],r;__shared__ unsigned char digest[32];
    int t=threadIdx.x;const F* a=input_a;const F* b=input_b;
    if(t<32)digest[t]=input_digest[t];__syncthreads();
    int round=0;
    while(n>1){
        F c0{0,0},c1{0,0},c2{0,0};int pairs=n/2;
        for(int i=t;i<pairs;i+=256){
            F da=add(a[2*i],a[2*i+1]),db=add(b[2*i],b[2*i+1]);
            c0=add(c0,accelerated(a[2*i],b[2*i]));
            c1=add(c1,add(accelerated(a[2*i],db),accelerated(b[2*i],da)));
            c2=add(c2,accelerated(da,db));
        }
        sums[0][t]=c0;sums[1][t]=c1;sums[2][t]=c2;__syncthreads();
        for(int step=128;step;step>>=1){if(t<step)for(int c=0;c<3;c++)sums[c][t]=add(sums[c][t],sums[c][t+step]);__syncthreads();}
        if(t==0){
            for(int c=0;c<3;c++)output[3*round+c]=sums[c][0];
            transcript_hash(digest,sums[0][0],sums[1][0],sums[2][0]);
            r={0,0};for(int i=0;i<8;i++){r.lo|=U(digest[i])<<(8*i);r.hi|=U(digest[8+i])<<(8*i);}
        }__syncthreads();
        F* aa=round%2?a1:a0;F* bb=round%2?b1:b0;
        for(int i=t;i<pairs;i+=256){
            aa[i]=add(a[2*i],accelerated(r,add(a[2*i],a[2*i+1])));
            bb[i]=add(b[2*i],accelerated(r,add(b[2*i],b[2*i+1])));
        }__syncthreads();
        a=aa;b=bb;n=pairs;round++;
    }
}
