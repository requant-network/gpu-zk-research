"""Keep complete transcript scheduling identical for optimized field comparators."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from build_optimized import REVISION


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('upstream',type=Path);parser.add_argument('--nvcc',default='nvcc')
    args=parser.parse_args();assert subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()==REVISION
    prior_path=Path('results/optimized-build.json');prior=json.loads(prior_path.read_text())
    base=Path('build/optimized_fields.cu').read_text();assert hashlib.sha256(base.encode()).hexdigest()==prior['generated_source_sha256']
    fused=Path('cuda/binary_fused.cu').read_text();graph=Path('cuda/binary_graph.cu').read_text()
    assert fused.count('#include "binary_fields.cu"')==1 and graph.count('#include "binary_fused.cu"')==1
    outputs={}
    for name,mode in (('ctmul32',2),('ctmul64',3)):
        source=base.replace('accelerated(', 'requant_clmad(')
        marker='__device__ F mul(F a,F b,int mode) {';assert source.count(marker)==1
        source=source.replace(marker,marker+f'\n mode={mode};')
        source+='\n__device__ F accelerated(F a,F b){return mul(a,b,'+str(mode)+');}\n'
        source+=fused.replace('#include "binary_fields.cu"','')+graph.replace('#include "binary_fused.cu"','')
        generated=Path('build')/f'graph_{name}.cu';binary=generated.with_suffix('.cubin');generated.write_text(source)
        subprocess.run([args.nvcc,'-O3','-std=c++20','-arch=sm_120','-cubin','-I'+str(args.upstream.resolve()/'src'),str(generated),'-o',str(binary)],check=True)
        outputs[name]=dict(generated_source_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    paths=['build_optimized_graph.py','build_optimized.py','cuda/binary_fields.cu','cuda/binary_fused.cu','cuda/binary_graph.cu']
    record=dict(revision=REVISION,builds=outputs,source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
                result_sha256={prior_path.name:hashlib.sha256(prior_path.read_bytes()).hexdigest()},
                scope='Only arithmetic backend changes. Same exact field/basis, coefficient/fold kernels, device SHA256 and captured transcript graph. Generated variants always retain polynomial reduction.')
    Path('results/optimized-graph-build.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
