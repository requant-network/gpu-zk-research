"""Keep the sequential SHA-256/challenge/fold suffix in one GPU block."""
import argparse
import ctypes
import hashlib
import json
import pathlib
import time
from datetime import datetime,timezone

import cupy as cp
import numpy as np
from binary_study import gpu_proof,unpack
from binary_tail_study import hybrid
from experiments.binary_field import bind,challenge,prove,verify


def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--output',default='results/binary-fused.json')
    args=parser.parse_args();rng=np.random.default_rng(args.seed)
    module=cp.RawModule(path=str(pathlib.Path('build/binary_fused.cubin').resolve()))
    coefficients,folding,tail,global_tail,hash_kernel=(module.get_function(x) for x in
        ('round_coefficients','field_fold','fused_tail','global_tail','check_transcript_hash'))
    digests=rng.integers(0,256,(4096,32),np.uint8)
    cs=rng.integers(0,1<<64,(4096,3,2),np.uint64)
    digests[0].fill(0);digests[1].fill(255);cs[0].fill(0);cs[1].fill((1<<64)-1)
    dd,dc=cp.asarray(digests),cp.asarray(cs);out=cp.empty_like(dd)
    hash_kernel((16,),(256,),(dd,dc,out,np.int32(len(cs))))
    hashes=cp.asnumpy(out)
    for i in range(len(cs)):
        assert hashes[i].tobytes()==hashlib.sha256(digests[i].tobytes()+cs[i].tobytes()).digest()
    library=ctypes.CDLL(str(pathlib.Path('build/libbinary_tail.so').resolve()))
    library.has_pclmul.restype=ctypes.c_int;assert library.has_pclmul()
    library.cpu_round.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int];library.cpu_round.restype=None
    library.cpu_fold.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int,ctypes.c_uint64,ctypes.c_uint64];library.cpu_fold.restype=None
    def native_suffix(a,b,digest):
        n=len(a);rounds=[];output=np.empty((3,2),np.uint64)
        while n>1:
            library.cpu_round(a.ctypes.data,b.ctypes.data,output.ctypes.data,n)
            cs=unpack(output);rounds.append(cs);digest,r=challenge(digest,cs)
            if n>2:library.cpu_fold(a.ctypes.data,b.ctypes.data,n,r&((1<<64)-1),r>>64)
            n//=2
        return rounds
    def device_suffix(a,b,digest):
        n=len(a);assert 2<=n<=1024 and n&(n-1)==0
        initial=cp.asarray(np.frombuffer(digest,np.uint8))
        output=cp.empty((n.bit_length()-1,3,2),cp.uint64)
        tail((1,),(256,),(a,b,initial,output,np.int32(n)))
        host=cp.asnumpy(output)
        return [unpack(row) for row in host]
    def global_suffix(a,b,digest):
        n=len(a);assert 2<=n<=65536 and n&(n-1)==0
        initial=cp.asarray(np.frombuffer(digest,np.uint8))
        output=cp.empty((n.bit_length()-1,3,2),cp.uint64)
        scratch=[cp.empty((n//2,2),cp.uint64) for _ in range(4)]
        global_tail((1,),(256,),(a,b,initial,output,*scratch,np.int32(n)))
        return [unpack(row) for row in cp.asnumpy(output)]
    cases=[]
    for n,pattern in ((4096,'random'),(4096,'zero'),(4096,'max'),(65536,'random')):
        a=rng.integers(0,1<<64,(n,2),np.uint64);b=rng.integers(0,1<<64,(n,2),np.uint64)
        if pattern=='zero':a.fill(0)
        if pattern=='max':a.fill((1<<64)-1)
        al,bl=unpack(a),unpack(b);digest=bind(al,bl);oracle=prove(al,bl)
        assert verify(al,bl,oracle[0][1]^oracle[0][2],oracle)
        da,db=cp.asarray(a),cp.asarray(b)
        modes=['gpu_rounds','native_1024','fused_16','fused_64','fused_256','fused_1024']
        modes.extend('global_'+str(c) for c in (1024,4096,8192,16384,32768,65536) if c<=n)
        samples={mode:[] for mode in modes}
        for iteration in range(35):
            modes=list(samples)
            if iteration%2:modes.reverse()
            for mode in modes:
                start=time.perf_counter()
                if mode=='gpu_rounds':actual=gpu_proof(da,db,digest,1,coefficients,folding)
                else:
                    cutoff=int(mode.split('_')[1])
                    suffix=native_suffix if mode.startswith('native') else global_suffix if mode.startswith('global') else device_suffix
                    actual=hybrid(da,db,digest,cutoff,coefficients,folding,suffix,suffix_on_device=not mode.startswith('native'))
                elapsed=(time.perf_counter()-start)*1000
                assert actual==oracle,(n,pattern,mode)
                if iteration>=5:samples[mode].append(elapsed)
        cases.append(dict(n=n,pattern=pattern,exact_transcripts=True,samples_ms=samples))
        print(n,pattern,{k:float(np.median(v)) for k,v in samples.items()},flush=True)
    paths=['binary_fused_study.py','cuda/binary_fused.cu','binary_study.py','binary_tail_study.py',
           'cuda/binary_fields.cu','native/binary_tail.cpp','experiments/binary_field.py']
    record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,warmups=5,samples=30,
                sha256_fixed_80_byte_checks=len(cs),
                source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths},cases=cases,
                scope='Resident public-input binary sumcheck. Fused suffix keeps coefficient reduction, fixed-80-byte SHA-256 transcript, challenges and folds in one GPU block. Includes one suffix digest H2D and one proof-coefficient D2H; initial input binding and upload excluded. All coefficients match the complete scalar Python CPU proof.')
    pathlib.Path(args.output).write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
