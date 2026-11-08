"""Reduced GF(2^128) products and complete public-input sumcheck rounds."""
import argparse
import hashlib
import json
import pathlib
import random
import time
from datetime import datetime, timezone

import cupy as cp
import numpy as np
from experiments.binary_field import bind, challenge, multiply, prove, verify, self_check


def unpack(values):
    return [int(lo) | int(hi) << 64 for lo, hi in values]


def gpu_proof(a, b, digest, mode, coefficients, folding):
    rounds = []
    n = len(a)
    while n > 1:
        pairs = n//2
        partial = cp.empty(((pairs+255)//256, 3, 2), cp.uint64)
        coefficients((len(partial),), (256,), (a,b,partial,np.int32(pairs),np.int32(mode)))
        cs = unpack(np.bitwise_xor.reduce(cp.asnumpy(partial), axis=0))
        rounds.append(cs)
        digest, r = challenge(digest, cs)
        if pairs > 1:
            aa, bb = cp.empty((pairs,2),cp.uint64), cp.empty((pairs,2),cp.uint64)
            folding(((pairs+255)//256,), (256,),
                    (a,b,aa,bb,np.int32(pairs),np.uint64(r & ((1<<64)-1)),np.uint64(r>>64),np.int32(mode)))
            a,b = aa,bb
        n = pairs
    cp.cuda.get_current_stream().synchronize()
    return rounds


def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--output',default='results/binary-study.json')
    args=parser.parse_args()
    self_check()
    rng=np.random.default_rng(args.seed)
    module=cp.RawModule(path=str(pathlib.Path('build/binary.cubin').resolve()))
    products,coefficients,folding=(module.get_function(x) for x in
                                 ('field_products','round_coefficients','field_fold'))
    micro=[]
    for n in (4096,262144,1048576):
        a=rng.integers(0,1<<64,(n,2),np.uint64)
        b=rng.integers(0,1<<64,(n,2),np.uint64)
        a[:4]=[[0,0],[(1<<64)-1,(1<<64)-1],[0,1<<63],[1,0]]
        b[:4]=[[(1<<64)-1,(1<<64)-1],[(1<<64)-1,(1<<64)-1],[2,0],[0,1<<63]]
        da,db=cp.asarray(a),cp.asarray(b)
        out=cp.empty_like(da)
        products(((n+255)//256,),(256,),(da,db,out,np.int32(n),np.int32(0)))
        expected=cp.asnumpy(out)
        indices=np.unique(np.concatenate((np.arange(4),rng.integers(0,n,1005))))
        for i in indices:
            x,y=unpack(a[i:i+1])[0],unpack(b[i:i+1])[0]
            assert unpack(expected[i:i+1])[0]==multiply(x,y)
        samples={'bitserial':[],'clmad_reduced':[]}
        for iteration in range(35):
            modes=list(samples)
            if iteration%2:modes.reverse()
            for name in modes:
                start,end=cp.cuda.Event(),cp.cuda.Event()
                start.record()
                products(((n+255)//256,),(256,),
                         (da,db,out,np.int32(n),np.int32(name=='clmad_reduced')))
                end.record();end.synchronize()
                elapsed=cp.cuda.get_elapsed_time(start,end)
                assert np.array_equal(cp.asnumpy(out),expected)
                if iteration>=5:samples[name].append(elapsed)
        micro.append(dict(n=n,cpu_oracle_pairs=len(indices),all_outputs_equal=True,samples_ms=samples))
        print('field products',n,{k:float(np.median(v)) for k,v in samples.items()},flush=True)
    proofs=[]
    for n in (256,4096,16384):
        for pattern in ('random','zero','max'):
            a=rng.integers(0,1<<64,(n,2),np.uint64)
            b=rng.integers(0,1<<64,(n,2),np.uint64)
            if pattern=='zero':a.fill(0)
            if pattern=='max':a.fill((1<<64)-1)
            al,bl=unpack(a),unpack(b)
            start=time.perf_counter();oracle=prove(al,bl);cpu_ms=(time.perf_counter()-start)*1000
            claim=0
            for x,y in zip(al,bl):claim^=multiply(x,y)
            assert verify(al,bl,claim,oracle)
            assert not verify(al,bl,claim^1,oracle)
            assert not verify(al,bl,claim,oracle[:-1])
            damaged=[x[:] for x in oracle];damaged[0][0]^=1
            assert not verify(al,bl,claim,damaged)
            da,db=cp.asarray(a),cp.asarray(b)
            digest=bind(al,bl)
            samples={'bitserial':[],'clmad_reduced':[]}
            for iteration in range(35):
                modes=list(samples)
                if iteration%2:modes.reverse()
                for name in modes:
                    start=time.perf_counter()
                    actual=gpu_proof(da,db,digest,name=='clmad_reduced',coefficients,folding)
                    elapsed=(time.perf_counter()-start)*1000
                    assert actual==oracle
                    if iteration>=5:samples[name].append(elapsed)
            proofs.append(dict(n=n,pattern=pattern,cpu_reference_ms=cpu_ms,exact_transcript=True,
                               tampered_cases_rejected=3,samples_ms=samples))
            print('binary sumcheck',n,pattern,{k:float(np.median(v)) for k,v in samples.items()},flush=True)
    paths=['binary_study.py','experiments/binary_field.py','cuda/binary_fields.cu']
    record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,warmups=5,samples=30,
                polynomial='x^128+x^7+x^2+x+1',coordinates='little-endian polynomial coefficients, not GHASH wire encoding',
                device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths},
                cubin_sha256=hashlib.sha256(pathlib.Path('build/binary.cubin').read_bytes()).hexdigest(),
                micro_scope='CUDA-event kernel interval only; complete multiplication and reduction, input transfers excluded',
                proof_scope='Resident inputs and precomputed initial binding; allocations, coefficient D2H, CPU SHA-256 challenges, launches and final synchronization included. Public-input algebra, no succinct commitment or hiding.',
                multiplication_cases=micro,proof_cases=proofs)
    pathlib.Path(args.output).write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
