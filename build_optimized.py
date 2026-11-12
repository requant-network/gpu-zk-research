"""Build matched GF128 kernels with pinned optimized software comparators."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

REVISION = '293cd240f0f73b37fa03f3082d616dc81d8cb3e5'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('--nvcc', default='nvcc')
    args = parser.parse_args()
    assert subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    source = Path('cuda/binary_fields.cu').read_text()
    old = '__device__ F mul(F a,F b,int mode) { return mode ? accelerated(a,b) : bitserial(a,b); }'
    assert source.count(old) == 1
    new = '''__device__ F mul(F a,F b,int mode) {
    if(mode==0)return bitserial(a,b);
    if(mode==1)return accelerated(a,b);
    if(mode==2){
        uint32_t aa[4]={uint32_t(a.lo),uint32_t(a.lo>>32),uint32_t(a.hi),uint32_t(a.hi>>32)};
        uint32_t bb[4]={uint32_t(b.lo),uint32_t(b.lo>>32),uint32_t(b.hi),uint32_t(b.hi>>32)};
        auto c=ghash_ctmul32::Ghash(aa)*ghash_ctmul32::Ghash(bb);
        return {U(c.limbs[0])|(U(c.limbs[1])<<32),U(c.limbs[2])|(U(c.limbs[3])<<32)};
    }
    uint64_t aa[2]={a.lo,a.hi},bb[2]={b.lo,b.hi};
    auto c=ghash_ctmul64::Ghash(aa)*ghash_ctmul64::Ghash(bb);
    return {c.limbs[0],c.limbs[1]};
}'''
    source = '#include "ulvt/finite_fields/ghash_ctmul32.cuh"\n#include "ulvt/finite_fields/ghash_ctmul64.cuh"\n' + source.replace(old, new)
    build = Path('build'); build.mkdir(exist_ok=True)
    generated = build / 'optimized_fields.cu'; generated.write_text(source)
    command = [args.nvcc, '-O3', '-std=c++20', '-arch=sm_120', '-cubin',
               '-I' + str(args.upstream.resolve() / 'src'), str(generated), '-o', 'build/optimized_fields.cubin']
    subprocess.run(command, check=True)
    headers = ['src/ulvt/finite_fields/ghash_ctmul32.cuh', 'src/ulvt/finite_fields/ghash_ctmul64.cuh']
    record = dict(revision=REVISION, flags=command[1:5],
                  source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in ('build_optimized.py', 'cuda/binary_fields.cu')},
                  upstream_source_sha256={p: hashlib.sha256((args.upstream / p).read_bytes()).hexdigest() for p in headers},
                  generated_source_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
                  cubin_sha256=hashlib.sha256(Path('build/optimized_fields.cubin').read_bytes()).hexdigest())
    Path('results/optimized-build.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
