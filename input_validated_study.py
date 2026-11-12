"""Explicit-stream, output-cleared qualification of canonical input overlap."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import cupy as cp
import numpy as np
from binary_input_study import packed_binding
from binary_graph_study import make_graph as legacy_graph
from binary_study import unpack
from optimized_oracle import inputs


def captured_proof(a,b,digest,kernels):
    """Every captured copy/kernel receives the capture stream explicitly."""
    n=len(a);proof=cp.empty((n.bit_length()-1,3,2),cp.uint64)
    initial=cp.asarray(np.frombuffer(digest,np.uint8));state=cp.empty(32,cp.uint8)
    challenge=cp.empty((1,2),cp.uint64);steps=[];keep=[];current=n
    while current>1:
        pairs=current//2;blocks=(pairs+255)//256
        partial=cp.empty((blocks,3,2),cp.uint64);output=proof[len(steps)]
        aa,bb=cp.empty((pairs,2),cp.uint64),cp.empty((pairs,2),cp.uint64)
        steps.append((a,b,partial,output,aa,bb,np.int32(pairs),np.int32(blocks)))
        keep.extend([a,b,partial,output,aa,bb]);a,b=aa,bb;current=pairs
    def execute(target):
        cp.cuda.runtime.memcpyAsync(state.data.ptr,initial.data.ptr,32,cp.cuda.runtime.memcpyDeviceToDevice,target.ptr)
        for a,b,partial,output,aa,bb,pairs,blocks in steps:
            kernels['round_coefficients']((int(blocks),),(256,),(a,b,partial,pairs,np.int32(1)),stream=target)
            kernels['finish_round']((1,),(256,),(partial,output,state,challenge,blocks),stream=target)
            if pairs>1:kernels['fold_challenge'](((int(pairs)+255)//256,),(256,),(a,b,aa,bb,challenge,pairs),stream=target)
    warm=cp.cuda.get_current_stream();execute(warm);warm.synchronize()
    capture=cp.cuda.Stream(non_blocking=True);capture.begin_capture();execute(capture);graph=capture.end_capture()
    return graph,proof,[initial,state,challenge,steps,keep]


def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('oracle',type=Path)
    parser.add_argument('--output',default='results/input-validated-study.json');args=parser.parse_args()
    oracle=json.loads(args.oracle.read_text());build_path=Path('results/optimized-graph-build.json');build=json.loads(build_path.read_text())
    binaries={'clmad':Path('build/binary_graph.cubin'),'ctmul32':Path('build/graph_ctmul32.cubin'),'ctmul64':Path('build/graph_ctmul64.cubin')}
    for name in ('ctmul32','ctmul64'):assert hashlib.sha256(binaries[name].read_bytes()).hexdigest()==build['builds'][name]['binary_sha256']
    original=json.loads(Path('results/stage2-build.json').read_text())['builds']['binary_graph']
    assert hashlib.sha256(binaries['clmad'].read_bytes()).hexdigest()==original['binary_sha256']
    kernels={}
    for name,path in binaries.items():
        module=cp.RawModule(path=str(path.resolve()))
        kernels[name]={key:module.get_function(key) for key in ('round_coefficients','finish_round','fold_challenge','global_tail')}
    stream=cp.cuda.Stream(non_blocking=True);stream.use();upload=cp.cuda.Stream(non_blocking=True);cases=[];legacy_probe=None
    for fixture in oracle['cases']:
        n=fixture['n'];a,b=inputs(oracle['seed'],n,fixture['pattern'])
        assert hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest()==fixture['input_sha256']
        digest=packed_binding(a,b);expected=fixture['transcript']
        setup=time.perf_counter();pins=[cp.cuda.alloc_pinned_memory(x.nbytes) for x in (a,b)]
        pa,pb=[np.frombuffer(pin,np.uint64).reshape(a.shape) for pin in pins]
        np.copyto(pa,a);np.copyto(pb,b);pin_setup_ms=(time.perf_counter()-setup)*1000
        da,db=cp.asarray(a),cp.asarray(b)
        if legacy_probe is None:
            before=cp.cuda.get_current_stream().ptr
            old_graph,old_out,old_keep=legacy_graph(da,db,digest,0,kernels['clmad'])
            changed=cp.cuda.get_current_stream().ptr!=before
            cp.cuda.runtime.memsetAsync(old_out.data.ptr,0xa5,old_out.nbytes,stream.ptr)
            old_graph.launch(stream);stream.synchronize()
            assert [unpack(row) for row in old_out.get(stream=stream)]==expected
            legacy_probe=dict(caller_current_stream_changed=changed,explicitly_fenced_legacy_graph_replays=True)
            del old_graph,old_out,old_keep
        graphs={name:captured_proof(da,db,digest,k) for name,k in kernels.items()}
        for graph,out,keep in graphs.values():
            cp.cuda.runtime.memsetAsync(out.data.ptr,0xa5,out.nbytes,stream.ptr)
            graph.launch(stream);stream.synchronize();assert [unpack(row) for row in out.get(stream=stream)]==expected
        modes={name:(backend,kind) for name,backend,kind in (
            ('clmad_pageable_serial','clmad','pageable'),('clmad_pinned_serial','clmad','pinned'),
            ('clmad_pinned_overlap','clmad','overlap'),('clmad_staged_overlap','clmad','staged'),
            ('ctmul32_pinned_overlap','ctmul32','overlap'),('ctmul64_pinned_overlap','ctmul64','overlap'))}
        samples={name:[] for name in modes};resident={name:[] for name in graphs};resident_wall={name:[] for name in graphs}
        for iteration in range(35):
            names=list(modes)
            if iteration%2:names.reverse()
            for name in names:
                backend,kind=modes[name];graph,out,keep=graphs[backend]
                stream.synchronize();upload.synchronize()
                cp.cuda.runtime.memsetAsync(out.data.ptr,0xa5,out.nbytes,stream.ptr)
                stream.synchronize();start=time.perf_counter()
                if kind=='staged':np.copyto(pa,a);np.copyto(pb,b)
                aa,bb=(a,b) if kind=='pageable' else (pa,pb)
                if kind in ('overlap','staged'):
                    da.set(aa,stream=upload);db.set(bb,stream=upload)
                    current=packed_binding(aa,bb);upload.synchronize()
                else:
                    current=packed_binding(aa,bb);da.set(aa,stream=stream);db.set(bb,stream=stream);stream.synchronize()
                keep[0].set(np.frombuffer(current,np.uint8),stream=stream);graph.launch(stream)
                stream.synchronize()
                actual=[unpack(row) for row in out.get(stream=stream)]
                elapsed=(time.perf_counter()-start)*1000
                assert current==digest and actual==expected,(name,n)
                if iteration>=5:samples[name].append(elapsed)
            for name,(graph,out,keep) in graphs.items():
                cp.cuda.runtime.memsetAsync(out.data.ptr,0xa5,out.nbytes,stream.ptr)
                begin,end=cp.cuda.Event(),cp.cuda.Event();start=time.perf_counter();begin.record(stream);graph.launch(stream);end.record(stream);end.synchronize()
                wall=(time.perf_counter()-start)*1000
                elapsed=cp.cuda.get_elapsed_time(begin,end);assert [unpack(row) for row in out.get(stream=stream)]==expected
                assert elapsed<=wall+0.1,(name,n,elapsed,wall)
                if iteration>=5:resident[name].append(elapsed);resident_wall[name].append(wall)
        cases.append(dict(n=n,pattern=fixture['pattern'],input_sha256=fixture['input_sha256'],
                          every_binding_and_transcript_exact=True,pinned_setup_ms=pin_setup_ms,
                          pinned_bytes=a.nbytes+b.nbytes,samples_ms=samples,resident_graph_device_ms=resident,resident_graph_wall_ms=resident_wall))
        print(n,fixture['pattern'],{name:float(np.median(v)) for name,v in samples.items()},flush=True)
        del graphs,da,db,pa,pb,pins
        cp.get_default_memory_pool().free_all_blocks();cp.get_default_pinned_memory_pool().free_all_blocks()
    paths=['input_validated_study.py','input_overlap_study.py','optimized_oracle.py','binary_graph_study.py','binary_input_study.py','binary_study.py','experiments/binary_field.py','build_optimized_graph.py']
    record=dict(legacy_stream_probe=legacy_probe,explicit_capture_stream_operations=True,explicit_stream_events=True,output_cleared_before_every_launch=True,host_boundary_waits_for_graph=True,utc=datetime.now(timezone.utc).isoformat(),seed=oracle['seed'],warmups=5,samples=30,cases=cases,
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
                result_sha256={args.oracle.name:hashlib.sha256(args.oracle.read_bytes()).hexdigest(),build_path.name:hashlib.sha256(build_path.read_bytes()).hexdigest()},
                scope='Explicit event/upload/download stream and completed graph boundary; sentinel-clear outside timer before every launch prevents stale-output acceptance. Supersedes input_overlap_study timing qualification. Canonical same SHA256 bytes and full exact transcripts. GPU DMA and CPU hash concurrently READ the same immutable pinned arrays; both complete before dependent proof execution. Serial pinned comparator uses the same buffers. Staged overlap includes ordinary-to-pinned copies; pinned-only modes require caller reusable pinned inputs. Setup/pinning/planning outside timers and retained separately. Full wall includes hashing, both H2D, digest update, graph, proof D2H and decoding; no initial claim or application verifier. Resident backend comparison uses identical captured scheduling.')
    Path(args.output).write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
