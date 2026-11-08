"""Large public-input binary sumchecks, each gated by a complete CPU proof."""
import argparse
import hashlib
import json
import pathlib
import time
from datetime import datetime,timezone

import cupy as cp
import numpy as np
from binary_study import gpu_proof,unpack
from experiments.binary_field import bind,multiply,prove,verify


def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--output',default='results/binary-scale.json')
    args=parser.parse_args();rng=np.random.default_rng(args.seed)
    module=cp.RawModule(path=str(pathlib.Path('build/binary.cubin').resolve()))
    coefficients,folding=(module.get_function(x) for x in ('round_coefficients','field_fold'))
    cases=[]
    for n in (262144,1048576):
        a=rng.integers(0,1<<64,(n,2),np.uint64);b=rng.integers(0,1<<64,(n,2),np.uint64)
        al,bl=unpack(a),unpack(b)
        print('CPU oracle started',n,flush=True)
        start=time.perf_counter();oracle=prove(al,bl);cpu_ms=(time.perf_counter()-start)*1000
        claim=0
        for x,y in zip(al,bl):claim^=multiply(x,y)
        assert verify(al,bl,claim,oracle)
        assert not verify(al,bl,claim^1,oracle)
        digest=bind(al,bl);da,db=cp.asarray(a),cp.asarray(b)
        samples={'bitserial':[],'clmad_reduced':[]}
        for iteration in range(35):
            modes=list(samples)
            if iteration%2:modes.reverse()
            for mode in modes:
                start=time.perf_counter()
                actual=gpu_proof(da,db,digest,mode=='clmad_reduced',coefficients,folding)
                elapsed=(time.perf_counter()-start)*1000
                assert actual==oracle
                if iteration>=5:samples[mode].append(elapsed)
        cases.append(dict(n=n,cpu_reference_ms=cpu_ms,complete_cpu_transcript_match=True,samples_ms=samples))
        print(n,{k:float(np.median(v)) for k,v in samples.items()},flush=True)
    paths=['binary_scale.py','binary_study.py','experiments/binary_field.py','cuda/binary_fields.cu']
    record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,warmups=5,samples=30,
                source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths},cases=cases,
                scope='All GF(2^128) sumcheck rounds. Resident inputs and precomputed initial binding; allocation, launches, D2H coefficients and CPU transcript challenges included; CPU proof/oracle and initial H2D excluded. Scalar bitserial reference, not optimized bitslicing.')
    pathlib.Path(args.output).write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
