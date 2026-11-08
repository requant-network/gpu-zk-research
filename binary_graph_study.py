"""Full device-resident transcript DAG versus the host-driven sumcheck."""
import argparse
import hashlib
import json
import pathlib
import time
from datetime import datetime,timezone

import cupy as cp
import numpy as np
from binary_study import gpu_proof,unpack
from experiments.binary_field import bind,prove,verify


def make_graph(a,b,digest,cutoff,kernels):
    original=len(a);current=len(a);round_index=0;steps=[];keep=[]
    proof=cp.empty((original.bit_length()-1,3,2),cp.uint64)
    initial=cp.asarray(np.frombuffer(digest,np.uint8));state=cp.empty(32,cp.uint8)
    challenge=cp.empty((1,2),cp.uint64)
    while current>1:
        if cutoff and current<=cutoff:
            scratch=[cp.empty((current//2,2),cp.uint64) for _ in range(4)]
            output=proof[round_index:]
            steps.append(('tail',(a,b,state,output,*scratch,np.int32(current))))
            keep.extend([output,*scratch]);break
        pairs=current//2;blocks=(pairs+255)//256
        partial=cp.empty((blocks,3,2),cp.uint64);output=proof[round_index]
        aa,bb=cp.empty((pairs,2),cp.uint64),cp.empty((pairs,2),cp.uint64)
        steps.append(('round',(a,b,partial,output,aa,bb,np.int32(pairs),np.int32(blocks))))
        keep.extend([a,b,partial,output,aa,bb]);a,b=aa,bb;current=pairs;round_index+=1
    def execute():
        cp.copyto(state,initial)
        for kind,values in steps:
            if kind=='tail':kernels['global_tail']((1,),(256,),values)
            else:
                a,b,partial,output,aa,bb,pairs,blocks=values
                kernels['round_coefficients']((int(blocks),),(256,),(a,b,partial,pairs,np.int32(1)))
                kernels['finish_round']((1,),(256,),(partial,output,state,challenge,blocks))
                if pairs>1:kernels['fold_challenge'](((int(pairs)+255)//256,),(256,),(a,b,aa,bb,challenge,pairs))
    execute();cp.cuda.get_current_stream().synchronize()
    capture=cp.cuda.Stream(non_blocking=True)
    with capture:
        capture.begin_capture();execute();graph=capture.end_capture()
    return graph,proof,[initial,state,challenge,steps,keep]


def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=(20261111,20261112),default=20261111)
    parser.add_argument('--output',default='results/binary-graph.json')
    args=parser.parse_args();rng=np.random.default_rng(args.seed)
    module=cp.RawModule(path=str(pathlib.Path('build/binary_graph.cubin').resolve()))
    kernels={name:module.get_function(name) for name in ('round_coefficients','field_fold','finish_round','fold_challenge','global_tail')}
    baseline_module=cp.RawModule(path=str(pathlib.Path('build/binary.cubin').resolve()))
    baseline_coeff,baseline_fold=(baseline_module.get_function(x) for x in ('round_coefficients','field_fold'))
    primitive=json.loads(pathlib.Path('results/binary-study.json').read_text())
    assert hashlib.sha256(pathlib.Path('build/binary.cubin').read_bytes()).hexdigest()==primitive['cubin_sha256']
    prior=pathlib.Path('results/binary-scale.json' if args.seed==20261111 else 'results/binary-scale-replication.json')
    previous=json.loads(prior.read_text())
    for path,expected in previous['source_sha256'].items():assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==expected
    cases=[]
    for n in (4096,65536,1048576):
        if n==1048576:
            fixed=np.random.default_rng(args.seed)
            fixed.integers(0,1<<64,(262144,2),np.uint64);fixed.integers(0,1<<64,(262144,2),np.uint64)
            a=fixed.integers(0,1<<64,(n,2),np.uint64);b=fixed.integers(0,1<<64,(n,2),np.uint64)
            assert any(c['n']==n and c['complete_cpu_transcript_match'] for c in previous['cases'])
        else:
            a=rng.integers(0,1<<64,(n,2),np.uint64);b=rng.integers(0,1<<64,(n,2),np.uint64)
        al,bl=unpack(a),unpack(b);digest=bind(al,bl);da,db=cp.asarray(a),cp.asarray(b)
        oracle=gpu_proof(da,db,digest,1,baseline_coeff,baseline_fold)
        if n<1048576:
            assert oracle==prove(al,bl)
            assert verify(al,bl,oracle[0][1]^oracle[0][2],oracle)
        graphs={name:make_graph(da,db,digest,cutoff,kernels) for name,cutoff in
                [('graph_all_rounds',0),('graph_tail_64',64),('graph_tail_1024',1024),('graph_tail_16384',16384)] if cutoff<=n}
        for name,(graph,out,keep) in graphs.items():
            graph.launch();actual=[unpack(row) for row in cp.asnumpy(out)]
            assert actual==oracle,(n,name)
        samples={name:{'proof_wall_ms':[],'device_elapsed_ms':[]} for name in ['host_rounds',*graphs]}
        for iteration in range(35):
            names=list(samples)
            if iteration%2:names.reverse()
            for name in names:
                if name=='host_rounds':
                    start=time.perf_counter();actual=gpu_proof(da,db,digest,1,baseline_coeff,baseline_fold)
                    wall=(time.perf_counter()-start)*1000
                else:
                    graph,out,keep=graphs[name]
                    begin,end=cp.cuda.Event(),cp.cuda.Event()
                    begin.record();graph.launch();end.record();end.synchronize()
                    elapsed=cp.cuda.get_elapsed_time(begin,end)
                    start=time.perf_counter();graph.launch();actual=[unpack(row) for row in cp.asnumpy(out)]
                    wall=(time.perf_counter()-start)*1000
                    if iteration>=5:samples[name]['device_elapsed_ms'].append(elapsed)
                assert actual==oracle
                if iteration>=5:samples[name]['proof_wall_ms'].append(wall)
        del samples['host_rounds']['device_elapsed_ms']
        cases.append(dict(n=n,oracle='direct CPU proof' if n<1048576 else 'unchanged CPU-gated fixed fixture',
                          input_sha256=hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest(),exact_transcripts=True,samples=samples))
        print(n,{k:float(np.median(v['proof_wall_ms'])) for k,v in samples.items()},flush=True)
    record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,warmups=5,samples=30,cases=cases,
                oracle_record=prior.name,oracle_record_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in
                    ('binary_graph_study.py','cuda/binary_graph.cu','cuda/binary_fused.cu','cuda/binary_fields.cu',
                     'binary_scale.py','binary_study.py','experiments/binary_field.py')},
                scope='Full GF(2^128) sumcheck transcript in a CUDA Graph. Initial input binding and H2D, planning/allocation/JIT/capture excluded for graphs; graph proof wall includes reset, launch, proof D2H and Python coefficient decoding. Host baseline uses warmed dynamic allocations and CPU per-round challenges. Million-entry fixture uses the unchanged directly CPU-gated baseline, not a new scalar CPU prover run.')
    pathlib.Path(args.output).write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
