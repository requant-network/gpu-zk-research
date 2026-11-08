// Requant Network research: CUDA 13.3 carryless multiply availability probe.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using U=unsigned long long;
struct Pair {U lo,hi;};
#define CUDA(call) do {auto status=(call);if(status!=cudaSuccess){std::cerr<<cudaGetErrorString(status)<<"\n";std::exit(1);}}while(0)
__host__ __device__ Pair scalar(U a,U b){
    Pair result{0,0};
    #pragma unroll
    for(int i=0;i<64;i++){
        U mask=U(0)-((b>>i)&1);
        result.lo^=(a<<i)&mask;
        if(i)result.hi^=(a>>(64-i))&mask;
    }
    return result;
}
__device__ Pair native(U a,U b){
    Pair result;
    asm("clmad.lo.u64 %0, %1, %2, 0;":"=l"(result.lo):"l"(a),"l"(b));
    asm("clmad.hi.u64 %0, %1, %2, 0;":"=l"(result.hi):"l"(a),"l"(b));
    return result;
}
template<bool accelerated> __global__ void products(const U* a,const U* b,Pair* out,int n,int work){
    int i=blockIdx.x*256+threadIdx.x;
    if(i<n){U x=a[i];Pair p{};for(int k=0;k<work;k++){
        p=accelerated ? native(x,b[i]) : scalar(x,b[i]);
        x=p.lo^p.hi^0x9e3779b97f4a7c15ULL;
    }out[i]=p;}
}
static Pair oracle(U a,U b,int work){
    Pair result{};
    for(int k=0;k<work;k++){
        // Independent polynomial accumulation in a 128-bit host integer.
        __uint128_t product=0;
        for(int bit=0;bit<64;bit++)if((b>>bit)&1)product^=__uint128_t(a)<<bit;
        result={U(product),U(product>>64)};
        a=result.lo^result.hi^0x9e3779b97f4a7c15ULL;
    }
    return result;
}
int main(){
    cudaDeviceProp device{};CUDA(cudaGetDeviceProperties(&device,0));
    std::mt19937_64 rng(20261109);
    std::cout<<"{\"device\":\""<<device.name<<"\",\"seed\":20261109,\"scope\":\"Unreduced 64x64 carryless product only; scalar bit-loop baseline, not a bitsliced library or a complete binary-field proof\",\"cases\":[";
    bool first=true;
    for(int n:{4096,262144,1048576})for(int work:{1,4}){
        std::vector<U> a(n),b(n);std::vector<Pair> expected(n),actual(n);
        for(int i=0;i<n;i++){a[i]=rng();b[i]=rng();}
        a[0]=0;b[0]=~U(0);a[1]=~U(0);b[1]=~U(0);a[2]=1;b[2]=U(1)<<63;
        for(int i=0;i<n;i++)expected[i]=oracle(a[i],b[i],work);
        U *da,*db;Pair* out;CUDA(cudaMalloc(&da,n*sizeof(U)));CUDA(cudaMalloc(&db,n*sizeof(U)));CUDA(cudaMalloc(&out,n*sizeof(Pair)));
        CUDA(cudaMemcpy(da,a.data(),n*sizeof(U),cudaMemcpyHostToDevice));CUDA(cudaMemcpy(db,b.data(),n*sizeof(U),cudaMemcpyHostToDevice));
        std::vector<float> timings[2];cudaEvent_t start,end;CUDA(cudaEventCreate(&start));CUDA(cudaEventCreate(&end));
        for(int iteration=0;iteration<35;iteration++)for(int j=0;j<2;j++){
            int mode=iteration%2 ? 1-j : j;
            CUDA(cudaEventRecord(start));
            if(mode)products<true><<<(n+255)/256,256>>>(da,db,out,n,work);
            else products<false><<<(n+255)/256,256>>>(da,db,out,n,work);
            CUDA(cudaGetLastError());CUDA(cudaEventRecord(end));CUDA(cudaEventSynchronize(end));
            float elapsed;CUDA(cudaEventElapsedTime(&elapsed,start,end));
            CUDA(cudaMemcpy(actual.data(),out,n*sizeof(Pair),cudaMemcpyDeviceToHost));
            for(int i=0;i<n;i++)if(actual[i].lo!=expected[i].lo||actual[i].hi!=expected[i].hi){std::cerr<<"Mismatch\n";return 2;}
            if(iteration>=5)timings[mode].push_back(elapsed);
        }
        if(!first)std::cout<<",";first=false;
        std::cout<<"{\"n\":"<<n<<",\"dependent_products\":"<<work<<",\"all_outputs_verified\":true,\"samples_ms\":{";
        for(int mode=0;mode<2;mode++){
            if(mode)std::cout<<",";std::cout<<"\""<<(mode?"clmad":"scalar_bit_loop")<<"\":[";
            for(size_t i=0;i<timings[mode].size();i++){if(i)std::cout<<",";std::cout<<timings[mode][i];}std::cout<<"]";
        }std::cout<<"}}";
        CUDA(cudaEventDestroy(start));CUDA(cudaEventDestroy(end));CUDA(cudaFree(da));CUDA(cudaFree(db));CUDA(cudaFree(out));
    }std::cout<<"]}\n";
}
