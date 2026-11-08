"""Regress fused tails against the unchanged, CPU-gated large-fixture baseline."""
import argparse
import hashlib
import json
import pathlib
import time
from datetime import datetime,timezone

import cupy as cp
import numpy as np
from binary_study import gpu_proof,unpack
from binary_tail_study import hybrid
from experiments.binary_field import bind


def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=(20261111,20261112),default=20261111)
    parser.add_argument('--output',default='results/binary-fused-scale.json')
    args=parser.parse_args();rng=np.random.default_rng(args.seed)
    oracle_path=pathlib.Path('results/binary-scale.json' if args.seed==20261111 else 'results/binary-scale-replication.json')
    oracle_record=json.loads(oracle_path.read_text())
    assert oracle_record['seed']==args.seed
    for path,expected in oracle_record['source_sha256'].items():
        assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==expected
    primitive_record=json.loads(pathlib.Path('results/binary-study.json').read_text())
    assert hashlib.sha256(pathlib.Path('build/binary.cubin').read_bytes()).hexdigest()==primitive_record['cubin_sha256']
    baseline=cp.RawModule(path=str(pathlib.Path('build/binary.cubin').resolve()))
    coefficients,folding=(baseline.get_function(x) for x in ('round_coefficients','field_fold'))
    fused=cp.RawModule(path=str(pathlib.Path('build/binary_fused.cubin').resolve())).get_function('global_tail')
    def suffix(a,b,digest):
        n=len(a);initial=cp.asarray(np.frombuffer(digest,np.uint8))
        output=cp.empty((n.bit_length()-1,3,2),cp.uint64)
        scratch=[cp.empty((n//2,2),cp.uint64) for _ in range(4)]
        fused((1,),(256,),(a,b,initial,output,*scratch,np.int32(n)))
        return [unpack(row) for row in cp.asnumpy(output)]
    cases=[]
    for previous in oracle_record['cases']:
        n=previous['n'];assert previous['complete_cpu_transcript_match']
        # Reproduce precisely the RNG consumption of binary_scale.py.
        a=rng.integers(0,1<<64,(n,2),np.uint64);b=rng.integers(0,1<<64,(n,2),np.uint64)
        digest=bind(unpack(a),unpack(b));da,db=cp.asarray(a),cp.asarray(b)
        oracle=gpu_proof(da,db,digest,1,coefficients,folding)
        samples={mode:[] for mode in ('gpu_rounds','global_8192','global_16384','global_32768')}
        for iteration in range(35):
            modes=list(samples)
            if iteration%2:modes.reverse()
            for mode in modes:
                start=time.perf_counter()
                actual=(gpu_proof(da,db,digest,1,coefficients,folding) if mode=='gpu_rounds' else
                        hybrid(da,db,digest,int(mode.split('_')[1]),coefficients,folding,suffix,True))
                elapsed=(time.perf_counter()-start)*1000
                assert actual==oracle
                if iteration>=5:samples[mode].append(elapsed)
        transcript=b''.join(c.to_bytes(16,'little') for row in oracle for c in row)
        cases.append(dict(n=n,unchanged_cpu_gated_fixture=True,exact_transcripts=True,
                          input_sha256=hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest(),
                          transcript_sha256=hashlib.sha256(transcript).hexdigest(),samples_ms=samples))
        print(n,{k:float(np.median(v)) for k,v in samples.items()},flush=True)
    record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,warmups=5,samples=30,cases=cases,
                oracle_record=oracle_path.name,oracle_record_sha256=hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in
                               ('binary_fused_scale.py','binary_study.py','binary_tail_study.py','binary_scale.py',
                                'experiments/binary_field.py','cuda/binary_fields.cu','cuda/binary_fused.cu')},
                scope='Exact regression on the same seeded fixtures, source hashes and baseline cubin previously checked against complete scalar CPU proofs by binary_scale.py. The costly scalar CPU prover is not rerun here. All rounds equal the unchanged CPU-gated GPU baseline. Resident-input wall timing includes suffix digest H2D, proof D2H and prefix CPU challenges; initial input binding/upload excluded.')
    pathlib.Path(args.output).write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
