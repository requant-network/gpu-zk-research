"""GPU public-input sumcheck: exact transcript equivalence and launch fusion."""
import hashlib
import argparse
import json
import pathlib
import struct
import time
from datetime import datetime, timezone

import cupy as cp
import numpy as np
from experiments.sumcheck import P, challenge, commitment, prove, verify

SOURCE = r'''
typedef unsigned int U;
typedef unsigned long long W;
__device__ U mul(U a,U b){return (W)a*b%2013265921U;}
__device__ U fold(U a,U b,U r){return (a+(W)r*((b+(W)2013265921U-a)%2013265921U))%2013265921U;}
extern "C" __global__ void coefficients(const U* a,const U* b,U* aa,U* bb,U* out,int pairs,U r,int fused){
  __shared__ U s[3][256];
  int t=threadIdx.x, i=blockIdx.x*256+t;
  U a0=0,a1=0,b0=0,b1=0;
  if(i<pairs){
    if(fused){
      a0=fold(a[4*i],a[4*i+1],r); a1=fold(a[4*i+2],a[4*i+3],r);
      b0=fold(b[4*i],b[4*i+1],r); b1=fold(b[4*i+2],b[4*i+3],r);
      aa[2*i]=a0; aa[2*i+1]=a1; bb[2*i]=b0; bb[2*i+1]=b1;
    }else{a0=a[2*i];a1=a[2*i+1];b0=b[2*i];b1=b[2*i+1];}
  }
  U da=(a1+(W)2013265921U-a0)%2013265921U,db=(b1+(W)2013265921U-b0)%2013265921U;
  s[0][t]=mul(a0,b0); s[1][t]=(mul(a0,db)+(W)mul(b0,da))%2013265921U;s[2][t]=mul(da,db);
  __syncthreads();
  for(int k=128;k;k>>=1){if(t<k)for(int c=0;c<3;c++)s[c][t]=(s[c][t]+(W)s[c][t+k])%2013265921U;__syncthreads();}
  if(t==0)for(int c=0;c<3;c++)out[3*blockIdx.x+c]=s[c][0];
}
extern "C" __global__ void folding(const U* a,const U* b,U* aa,U* bb,int n,U r){
  int i=blockIdx.x*256+threadIdx.x;
  if(i<n){aa[i]=fold(a[2*i],a[2*i+1],r);bb[i]=fold(b[2*i],b[2*i+1],r);}
}
extern "C" __global__ void reduce_coefficients(const U* partial,U* out,int blocks){
 __shared__ U sums[3][256];int t=threadIdx.x;
 for(int c=0;c<3;c++){U v=0;for(int i=t;i<blocks;i+=256)v=(v+(W)partial[3*i+c])%2013265921U;sums[c][t]=v;}
 __syncthreads();
 for(int k=128;k;k>>=1){if(t<k)for(int c=0;c<3;c++)sums[c][t]=(sums[c][t]+(W)sums[c][t+k])%2013265921U;__syncthreads();}
 if(t==0)for(int c=0;c<3;c++)out[c]=sums[c][0];
}
'''
module = cp.RawModule(code=SOURCE)
coefficients, folding, reduce_coefficients = (module.get_function(x) for x in ('coefficients', 'folding', 'reduce_coefficients'))


def gpu_prove(a, b, digest, fused, device_reduce=False):
    n = len(a)
    rounds = []
    r = 0
    while n > 1:
        combine = fused and bool(rounds)
        current = n // 2 if combine else n
        aa, bb = cp.empty(current, cp.uint32), cp.empty(current, cp.uint32)
        pairs = current // 2
        partial = cp.empty(((pairs + 255) // 256, 3), cp.uint32)
        coefficients((len(partial),), (256,),
                     (a, b, aa, bb, partial, np.int32(pairs), np.uint32(r), np.int32(combine)))
        if device_reduce:
            reduced=cp.empty(3,cp.uint32)
            reduce_coefficients((1,),(256,),(partial,reduced,np.int32(len(partial))))
            cs=cp.asnumpy(reduced).tolist()
        else:
            cs = [int(x) % P for x in cp.asnumpy(partial).astype(np.uint64).sum(axis=0)]
        rounds.append(cs)
        digest, r = challenge(digest, cs)
        if combine:
            a, b, n = aa, bb, current
        if not fused:
            folding(((n // 2 + 255) // 256,), (256,),
                    (a, b, aa, bb, np.int32(n // 2), np.uint32(r)))
            a, b, n = aa, bb, n // 2
        elif n == 2:
            # The verifier checks the final evaluation; no unused final fold is launched.
            break
    cp.cuda.get_current_stream().synchronize()
    return rounds


def main():
    assert __debug__, 'Run without -O: correctness checks are required'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261109)
    parser.add_argument('--output',default='results/gpu-study.json')
    args=parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cases = []
    for n in (1024, 16384, 262144, 1048576):
        for pattern in ('random', 'max', 'zero'):
            a = rng.integers(0, P, n, dtype=np.uint32)
            b = rng.integers(0, P, n, dtype=np.uint32)
            if pattern != 'random':
                a.fill(P - 1 if pattern == 'max' else 0)
            al, bl = a.tolist(), b.tolist()
            digest = commitment(al, bl)
            def streamed_commitment():
                h=hashlib.sha256(b'requant-sumcheck-v1'+struct.pack('<I',n))
                h.update(a.astype('<u4',copy=False).data)
                h.update(b.astype('<u4',copy=False).data)
                return h.digest()
            assert streamed_commitment()==digest
            start = time.perf_counter()
            oracle = prove(al, bl)
            cpu_ms = (time.perf_counter() - start) * 1000
            total = sum(x * y for x, y in zip(al, bl)) % P
            assert verify(al, bl, total, oracle)
            da, db = cp.asarray(a), cp.asarray(b)
            cp.cuda.get_current_stream().synchronize()
            samples = {'separate': [], 'fused': [], 'fused_gpu_reduce': []}
            full_samples={mode:[] for mode in ('original_commitment','streamed_commitment')}
            for iteration in range(35):
                modes=list(samples)
                if iteration % 2:modes.reverse()
                for mode in modes:
                    start = time.perf_counter()
                    actual = gpu_prove(da, db, digest, mode != 'separate', mode=='fused_gpu_reduce')
                    elapsed = (time.perf_counter() - start) * 1000
                    assert actual == oracle, (n, pattern, mode)
                    if iteration >= 5:
                        samples[mode].append(elapsed)
                for mode in full_samples:
                    start=time.perf_counter()
                    full_digest=commitment(al,bl) if mode=='original_commitment' else streamed_commitment()
                    full_proof=gpu_prove(cp.asarray(a),cp.asarray(b),full_digest,True)
                    full_elapsed=(time.perf_counter()-start)*1000
                    assert full_proof==oracle
                    if iteration>=5:full_samples[mode].append(full_elapsed)
            # Changed transcript must fail with the independent verifier.
            damaged = [x[:] for x in oracle]
            damaged[0][0] = (damaged[0][0] + 1) % P
            assert not verify(al, bl, total, damaged)
            cases.append(dict(n=n, pattern=pattern, cpu_reference_ms=cpu_ms,
                              samples_ms=samples, exact_transcript_match=True,
                              full_path_samples_ms=full_samples,
                              tampered_transcript_rejected=True))
            print(n, pattern, {k: round(float(np.median(v)), 3) for k, v in samples.items()}, flush=True)
    metadata = dict(utc=datetime.now(timezone.utc).isoformat(), seed=args.seed,
                    device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                    numpy=np.__version__, cupy=cp.__version__, runtime=cp.cuda.runtime.runtimeGetVersion(),
                    driver=cp.cuda.runtime.driverGetVersion(),
                    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                   (pathlib.Path('gpu_study.py'),pathlib.Path('experiments/sumcheck.py'))},
                    scope='GPU-resident inputs; wall time includes allocations, coefficient D2H, CPU SHA-256 and synchronization. Initial input commitment and H2D excluded. Public-input algebra only.',
                    warmups=5, samples=30, cases=cases)
    pathlib.Path(args.output).write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
