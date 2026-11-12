"""Complete canonical binding/upload/proof with safe read-only input overlap."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import cupy as cp
import numpy as np
from binary_graph_study import make_graph
from binary_input_study import packed_binding
from binary_study import unpack
from optimized_oracle import inputs


def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('oracle',type=Path)
    parser.add_argument('--output',default='results/input-overlap-study.json');args=parser.parse_args()
    oracle=json.loads(args.oracle.read_text());build_path=Path('results/optimized-graph-build.json');build=json.loads(build_path.read_text())
    binaries={'clmad':Path('build/binary_graph.cubin'),'ctmul32':Path('build/graph_ctmul32.cubin'),'ctmul64':Path('build/graph_ctmul64.cubin')}
    for name in ('ctmul32','ctmul64'):assert hashlib.sha256(binaries[name].read_bytes()).hexdigest()==build['builds'][name]['binary_sha256']
    original=json.loads(Path('results/stage2-build.json').read_text())['builds']['binary_graph']
    assert hashlib.sha256(binaries['clmad'].read_bytes()).hexdigest()==original['binary_sha256']
    kernels={}
    for name,path in binaries.items():
        module=cp.RawModule(path=str(path.resolve()))
        kernels[name]={key:module.get_function(key) for key in ('round_coefficients','finish_round','fold_challenge','global_tail')}
    stream=cp.cuda.Stream(non_blocking=True);stream.use();upload=cp.cuda.Stream(non_blocking=True);cases=[]
    for fixture in oracle['cases']:
        n=fixture['n'];a,b=inputs(oracle['seed'],n,fixture['pattern'])
        assert hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest()==fixture['input_sha256']
        digest=packed_binding(a,b);expected=fixture['transcript']
        setup=time.perf_counter();pins=[cp.cuda.alloc_pinned_memory(x.nbytes) for x in (a,b)]
        pa,pb=[np.frombuffer(pin,np.uint64).reshape(a.shape) for pin in pins]
        np.copyto(pa,a);np.copyto(pb,b);pin_setup_ms=(time.perf_counter()-setup)*1000
        da,db=cp.asarray(a),cp.asarray(b);graphs={name:make_graph(da,db,digest,0,k) for name,k in kernels.items()}
        for graph,out,keep in graphs.values():graph.launch(stream);stream.synchronize();assert [unpack(row) for row in cp.asnumpy(out)]==expected
        modes={name:(backend,kind) for name,backend,kind in (
            ('clmad_pageable_serial','clmad','pageable'),('clmad_pinned_serial','clmad','pinned'),
            ('clmad_pinned_overlap','clmad','overlap'),('clmad_staged_overlap','clmad','staged'),
            ('ctmul32_pinned_overlap','ctmul32','overlap'),('ctmul64_pinned_overlap','ctmul64','overlap'))}
        samples={name:[] for name in modes};resident={name:[] for name in graphs}
        for iteration in range(35):
            names=list(modes)
            if iteration%2:names.reverse()
            for name in names:
                backend,kind=modes[name];graph,out,keep=graphs[backend]
                stream.synchronize();upload.synchronize();start=time.perf_counter()
                if kind=='staged':np.copyto(pa,a);np.copyto(pb,b)
                aa,bb=(a,b) if kind=='pageable' else (pa,pb)
                if kind in ('overlap','staged'):
                    da.set(aa,stream=upload);db.set(bb,stream=upload)
                    current=packed_binding(aa,bb);upload.synchronize()
                else:
                    current=packed_binding(aa,bb);da.set(aa);db.set(bb);stream.synchronize()
                keep[0].set(np.frombuffer(current,np.uint8));graph.launch(stream)
                actual=[unpack(row) for row in cp.asnumpy(out)]
                elapsed=(time.perf_counter()-start)*1000
                assert current==digest and actual==expected,(name,n)
                if iteration>=5:samples[name].append(elapsed)
            for name,(graph,out,keep) in graphs.items():
                begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record();graph.launch(stream);end.record();end.synchronize()
                elapsed=cp.cuda.get_elapsed_time(begin,end);assert [unpack(row) for row in cp.asnumpy(out)]==expected
                if iteration>=5:resident[name].append(elapsed)
        cases.append(dict(n=n,pattern=fixture['pattern'],input_sha256=fixture['input_sha256'],
                          every_binding_and_transcript_exact=True,pinned_setup_ms=pin_setup_ms,
                          pinned_bytes=a.nbytes+b.nbytes,samples_ms=samples,resident_graph_device_ms=resident))
        print(n,fixture['pattern'],{name:float(np.median(v)) for name,v in samples.items()},flush=True)
        del graphs,da,db,pa,pb,pins
        cp.get_default_memory_pool().free_all_blocks();cp.get_default_pinned_memory_pool().free_all_blocks()
    paths=['input_overlap_study.py','optimized_oracle.py','binary_graph_study.py','binary_input_study.py','binary_study.py','experiments/binary_field.py','build_optimized_graph.py']
    record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=oracle['seed'],warmups=5,samples=30,cases=cases,
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
                result_sha256={args.oracle.name:hashlib.sha256(args.oracle.read_bytes()).hexdigest(),build_path.name:hashlib.sha256(build_path.read_bytes()).hexdigest()},
                scope='Canonical same SHA256 bytes and full exact transcripts. GPU DMA and CPU hash concurrently READ the same immutable pinned arrays; both complete before dependent proof execution. Serial pinned comparator uses the same buffers. Staged overlap includes ordinary-to-pinned copies; pinned-only modes require caller reusable pinned inputs. Setup/pinning/planning outside timers and retained separately. Full wall includes hashing, both H2D, digest update, graph, proof D2H and decoding; no initial claim or application verifier. Resident backend comparison uses identical captured scheduling.')
    Path(args.output).write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
