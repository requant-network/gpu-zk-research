"""Retain complete arbitrary-precision transcripts for matched comparator inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from experiments.binary_field import prove, verify


def inputs(seed, n, pattern):
    rng = np.random.default_rng(seed + n)
    a = rng.integers(0, 1 << 64, (n, 2), np.uint64)
    b = rng.integers(0, 1 << 64, (n, 2), np.uint64)
    if pattern == 'zero':
        a.fill(0)
    if pattern == 'max':
        a.fill((1 << 64) - 1)
    return a, b


def main():
    assert __debug__
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261112)
    parser.add_argument('--output', default='results/optimized-oracle-20261112.json')
    args = parser.parse_args()
    cases = []
    for n, pattern in ((4096, 'random'), (4096, 'zero'), (4096, 'max'),
                       (65536, 'random'), (1048576, 'random')):
        a, b = inputs(args.seed, n, pattern)
        al = [int(x) | int(y) << 64 for x, y in a]
        bl = [int(x) | int(y) << 64 for x, y in b]
        start = time.monotonic(); transcript = prove(al, bl)
        if n == 4096:
            claim = transcript[0][1] ^ transcript[0][2]
            assert verify(al, bl, claim, transcript)
            broken = [row[:] for row in transcript]; broken[0][0] ^= 1
            assert not verify(al, bl, claim, broken)
        cases.append(dict(n=n, pattern=pattern, input_sha256=hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest(),
                          transcript=transcript, cpu_seconds=time.monotonic()-start))
        print(args.seed, n, pattern, 'CPU transcript complete', flush=True)
    record = dict(seed=args.seed, cases=cases,
                  source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in ('optimized_oracle.py', 'experiments/binary_field.py')})
    Path(args.output).write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
