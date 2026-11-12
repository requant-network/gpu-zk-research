"""Matched complete transcripts: scalar, CLMAD, and pinned ctmul32/64."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone
import cupy as cp
import numpy as np
from binary_study import gpu_proof
from experiments.binary_field import bind
from optimized_oracle import inputs


def main():
    assert __debug__
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('oracle', type=Path)
    parser.add_argument('--output', default='results/optimized-study.json')
    args = parser.parse_args()
    oracle = json.loads(args.oracle.read_text())
    for name, expected in oracle['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    build_path = Path('results/optimized-build.json'); build = json.loads(build_path.read_text())
    binary = Path('build/optimized_fields.cubin')
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == build['cubin_sha256']
    module = cp.RawModule(path=str(binary.resolve()))
    coefficients, folding = [module.get_function(name) for name in ('round_coefficients', 'field_fold')]
    cases = []
    for fixture in oracle['cases']:
        n, pattern = fixture['n'], fixture['pattern']
        a, b = inputs(oracle['seed'], n, pattern)
        assert hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest() == fixture['input_sha256']
        al = [int(x) | int(y) << 64 for x, y in a]; bl = [int(x) | int(y) << 64 for x, y in b]
        digest = bind(al, bl); da, db = cp.asarray(a), cp.asarray(b)
        modes = {'clmad': 1, 'pinned_ctmul32': 2, 'pinned_ctmul64': 3}
        if n == 4096:
            modes['scalar'] = 0
        samples = {name: [] for name in modes}
        for iteration in range(35):
            names = list(modes)
            if iteration % 2:
                names.reverse()
            for name in names:
                start = time.perf_counter()
                actual = gpu_proof(da, db, digest, modes[name], coefficients, folding)
                elapsed = (time.perf_counter()-start)*1000
                assert actual == fixture['transcript'], (n, pattern, name)
                if iteration >= 5:
                    samples[name].append(elapsed)
        cases.append(dict(n=n, pattern=pattern, input_sha256=fixture['input_sha256'],
                          every_complete_transcript_exact=True, samples_ms=samples))
        print(n, pattern, {name: float(np.median(values)) for name, values in samples.items()}, flush=True)
    paths = ['optimized_study.py', 'optimized_oracle.py', 'binary_study.py', 'experiments/binary_field.py', 'cuda/binary_fields.cu', 'build_optimized.py']
    record = dict(utc=datetime.now(timezone.utc).isoformat(), seed=oracle['seed'], warmups=5, samples=30,
                  cases=cases, revision=build['revision'], cubin_sha256=build['cubin_sha256'],
                  source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
                  result_sha256={args.oracle.name: hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
                                 build_path.name: hashlib.sha256(build_path.read_bytes()).hexdigest()},
                  scope='Exact same polynomial, basis, inputs, rounds, allocation policy, coefficient transfers and host SHA256 challenges. Resident complete algebra transcript; excludes initial binding/upload and verifier work. Pinned optimized constant-time software multiplication comparators, not an upstream full prover or a tower-basis bitsliced benchmark.')
    Path(args.output).write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
