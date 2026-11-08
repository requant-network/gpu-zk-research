"""Test whether CPU suffix rounds help an otherwise CLMAD GPU sumcheck."""
import argparse
import ctypes
import hashlib
import json
import pathlib
import time
from datetime import datetime,timezone

import cupy as cp
import numpy as np
from binary_study import unpack,gpu_proof
from experiments.binary_field import bind,challenge,multiply,prove,verify


def cpu_suffix(a,b,digest):
    rounds=[]
    while len(a)>1:
        cs=[0,0,0]
        for a0,a1,b0,b1 in zip(a[::2],a[1::2],b[::2],b[1::2]):
            da,db=a0^a1,b0^b1
            cs[0]^=multiply(a0,b0)
            cs[1]^=multiply(a0,db)^multiply(b0,da)
            cs[2]^=multiply(da,db)
        rounds.append(cs);digest,r=challenge(digest,cs)
        if len(a)==2:break
        a=[x^multiply(r,x^y) for x,y in zip(a[::2],a[1::2])]
        b=[x^multiply(r,x^y) for x,y in zip(b[::2],b[1::2])]
    return rounds


def python_suffix(a,b,digest):
    return cpu_suffix(unpack(a),unpack(b),digest)


def hybrid(a,b,digest,cutoff,coefficients,folding,suffix=python_suffix,suffix_on_device=False):
    n=len(a);rounds=[]
    while n>1:
        if n<=cutoff:
            rounds.extend(suffix(a,b,digest) if suffix_on_device else suffix(cp.asnumpy(a),cp.asnumpy(b),digest))
            break
        pairs=n//2;partial=cp.empty(((pairs+255)//256,3,2),cp.uint64)
        coefficients((len(partial),),(256,),(a,b,partial,np.int32(pairs),np.int32(1)))
        cs=unpack(np.bitwise_xor.reduce(cp.asnumpy(partial),axis=0))
        rounds.append(cs);digest,r=challenge(digest,cs)
        if pairs>1:
            aa,bb=cp.empty((pairs,2),cp.uint64),cp.empty((pairs,2),cp.uint64)
            folding(((pairs+255)//256,),(256,),
                (a,b,aa,bb,np.int32(pairs),np.uint64(r&((1<<64)-1)),np.uint64(r>>64),np.int32(1)))
            a,b=aa,bb
        n=pairs
    cp.cuda.get_current_stream().synchronize()
    return rounds


def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--output',default='results/binary-tail.json')
    args=parser.parse_args();rng=np.random.default_rng(args.seed)
    library=ctypes.CDLL(str(pathlib.Path('build/libbinary_tail.so').resolve()))
    library.has_pclmul.restype=ctypes.c_int
    assert library.has_pclmul(),'This experiment requires x86 PCLMUL support'
    library.cpu_round.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]
    library.cpu_round.restype=None
    library.cpu_fold.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int,ctypes.c_uint64,ctypes.c_uint64]
    library.cpu_fold.restype=None
    def native_suffix(a,b,digest):
        n=len(a);rounds=[];output=np.empty((3,2),np.uint64)
        while n>1:
            library.cpu_round(a.ctypes.data,b.ctypes.data,output.ctypes.data,n)
            cs=unpack(output);rounds.append(cs);digest,r=challenge(digest,cs)
            if n>2:library.cpu_fold(a.ctypes.data,b.ctypes.data,n,r&((1<<64)-1),r>>64)
            n//=2
        return rounds
    module=cp.RawModule(path=str(pathlib.Path('build/binary.cubin').resolve()))
    coefficients,folding=(module.get_function(x) for x in ('round_coefficients','field_fold'))
    cases=[]
    for n in (4096,65536):
        a=rng.integers(0,1<<64,(n,2),np.uint64);b=rng.integers(0,1<<64,(n,2),np.uint64)
        al,bl=unpack(a),unpack(b);digest=bind(al,bl);oracle=prove(al,bl)
        assert verify(al,bl,oracle[0][1]^oracle[0][2],oracle)
        da,db=cp.asarray(a),cp.asarray(b)
        samples={mode:[] for mode in ['gpu']+['python_'+str(n) for n in (2,4,8,16)]+
                 ['native_'+str(c) for c in (2,4,8,16,64,256,512,1024,2048,4096,16384,65536) if c<=n]}
        for iteration in range(35):
            modes=list(samples)
            if iteration%2:modes.reverse()
            for key in modes:
                cutoff=int(key.split('_')[1]) if key!='gpu' else 0;start=time.perf_counter()
                suffix=native_suffix if key.startswith('native') else python_suffix
                actual=(hybrid(da,db,digest,cutoff,coefficients,folding,suffix) if cutoff else
                        gpu_proof(da,db,digest,1,coefficients,folding))
                elapsed=(time.perf_counter()-start)*1000
                assert actual==oracle
                if iteration>=5:samples[key].append(elapsed)
        cases.append(dict(n=n,exact_transcripts=True,samples_ms=samples))
        print(n,{k:float(np.median(v)) for k,v in samples.items()},flush=True)
    record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,warmups=5,samples=30,cases=cases,
                native_library_sha256=hashlib.sha256(pathlib.Path('build/libbinary_tail.so').read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in
                               ('binary_tail_study.py','binary_study.py','experiments/binary_field.py','cuda/binary_fields.cu','native/binary_tail.cpp')},
                scope='Same resident-input CLMAD proof; suffix cutoffs compare scalar Python with x86 PCLMUL. Includes suffix D2H, CPU arithmetic and CPU SHA-256; initial binding, upload and verifier excluded. gpu means all rounds on GPU.')
    pathlib.Path(args.output).write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
