"""Include canonical input hashing and PCIe transfers in graph proof timing."""
import argparse
import hashlib
import json
import pathlib
import time
from datetime import datetime, timezone

import cupy as cp
import numpy as np
from binary_graph_study import make_graph
from binary_study import unpack
from experiments.binary_field import bind, prove

def packed_binding(a,b):
    h=hashlib.sha256(b'requant-binary-sumcheck-v1'+len(a).to_bytes(4,'little'))
    h.update(memoryview(a).cast('B'));h.update(memoryview(b).cast('B'))
    return h.digest()

def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=(20261111,20261112),default=20261111)
    parser.add_argument('--output',default='results/binary-input.json')
    args=parser.parse_args()
    module=cp.RawModule(path=str(pathlib.Path('build/binary_graph.cubin').resolve()))
    kernels={name:module.get_function(name) for name in ('round_coefficients','finish_round','fold_challenge','global_tail')}
    prior_path=pathlib.Path('results/binary-graph.json' if args.seed==20261111 else 'results/binary-graph-replication.json')
    prior=json.loads(prior_path.read_text());rng=np.random.default_rng(args.seed);cases=[]
    for n in (4096,65536,1048576):
        if n==1048576:
            fixed=np.random.default_rng(args.seed)
            fixed.integers(0,1<<64,(262144,2),np.uint64);fixed.integers(0,1<<64,(262144,2),np.uint64)
            a=fixed.integers(0,1<<64,(n,2),np.uint64);b=fixed.integers(0,1<<64,(n,2),np.uint64)
        else:
            a=rng.integers(0,1<<64,(n,2),np.uint64);b=rng.integers(0,1<<64,(n,2),np.uint64)
        a=a.astype('<u8',copy=False);b=b.astype('<u8',copy=False)
        digest=packed_binding(a,b);assert digest==bind(unpack(a),unpack(b))
        previous=next(c for c in prior['cases'] if c['n']==n)
        assert hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest()==previous['input_sha256']
        da,db=cp.asarray(a),cp.asarray(b)
        graph,output,keep=make_graph(da,db,digest,0,kernels)
        graph.launch();oracle=cp.asnumpy(output)
        if n<1048576:assert [unpack(row) for row in oracle]==prove(unpack(a),unpack(b))
        samples={name:[] for name in ('canonical_input_binding','input_upload','complete_proof_wall')}
        for iteration in range(35):
            start=time.perf_counter();current=packed_binding(a,b);binding_ms=(time.perf_counter()-start)*1000
            da.set(a);db.set(b);cp.cuda.get_current_stream().synchronize()
            upload_ms=(time.perf_counter()-start)*1000-binding_ms
            keep[0].set(np.frombuffer(current,np.uint8))
            graph.launch();actual=cp.asnumpy(output);decoded=[unpack(row) for row in actual]
            complete_ms=(time.perf_counter()-start)*1000
            assert np.array_equal(actual,oracle) and current==digest
            if iteration>=5:
                for name,value in zip(samples,(binding_ms,upload_ms,complete_ms)):samples[name].append(value)
        cases.append(dict(n=n,input_sha256=previous['input_sha256'],binding_matches_scalar=True,exact_transcripts=True,samples_ms=samples))
        print(n,{k:float(np.median(v)) for k,v in samples.items()},flush=True)
    record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,warmups=5,samples=30,cases=cases,
        oracle_record=prior_path.name,oracle_record_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in
            ('binary_input_study.py','binary_graph_study.py','binary_study.py','experiments/binary_field.py','cuda/binary_graph.cu','cuda/binary_fused.cu','cuda/binary_fields.cu')},
        scope='Canonical packed little-endian host inputs. Full wall includes SHA-256 binding over both complete inputs, two H2D uploads into preallocated buffers, digest update, graph launch, proof D2H and Python coefficient decoding. Input generation/packing, plan allocation/capture, initial claim computation and verifier excluded. Million-entry fixed fixture inherits the prior directly CPU-gated graph experiment.')
    pathlib.Path(args.output).write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
