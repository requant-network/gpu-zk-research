# GPU ZK Research

Reproducible research into finite-field arithmetic and GPU proving pipelines.

A research project by [Requant Network](https://requant.network).

**Stage 0 вЂ” CPU arithmetic baseline and CUDA diagnostic preparation.**

## Research question

Can a memory-efficient sumcheck folding pipeline reduce complete proof-generation cost while preserving every algebraic check?

## Current experiment scope

The first experiment implements a degree-two dot-product sumcheck over the BabyBear prime 2013265921. It binds the transcript to the public inputs with SHA-256, checks every round identity, and recomputes the final multilinear evaluation from the public inputs. Verification is linear in the input size. This is an algebraic research baseline, not a zero-knowledge or succinct proof system. Modulo reduction of SHA-256 challenges is a deterministic prototype choice, not a formal Fiat-Shamir security claim.

## Checks

python run.py validates 15 honest proofs and rejects 45 altered coefficients, wrong claims, and truncated transcripts. Input sizes are 2, 16, 256, 4096, and 16384. Patterns include zero, maximum canonical field elements, and seeded random values.

The CUDA diagnostic checks one multilinear fold against 4096 exact Python field computations. It does not yet generate a complete sumcheck proof on GPU.

## Reproduce the CPU experiment

Requires Python 3.12 or newer and its standard library. Run without Python's `-O` option, because validation uses assertions.

```sh
python run.py
```

This writes `results/cpu-baseline.json` and regenerates `results/gpu-fixture.bin`. The JSON includes UTC time, environment information, and SHA-256 hashes of the executed experiment sources. The binary fixture is generated locally and excluded from Git.

## Prepare and run the CUDA diagnostic

CUDA 12.8 or newer is required for the Blackwell target used here. Compilation does not execute a GPU workload.

```sh
mkdir -p build
nvcc -std=c++17 -O3 -arch=sm_120 cuda/probe.cu -o build/gpu-check
```

When a GPU is available, regenerate the fixture using `python run.py`, then run:

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./build/gpu-check results/gpu-fixture.bin
```

The diagnostic uploads the fixture, warms up ten launches, times 100 launches using CUDA events, and compares every result with the exact fixture. Its timing covers the diagnostic kernel only. CPU preparation and transfers are outside that timer.

## Research record

- [Research questions and next experiments](RESEARCH.md)
- [Measurement rules](METHODS.md)
- [Observed results and limitations](RESULTS.md)
- [Raw CPU baseline](results/cpu-baseline.json)

## License and contributions

Original experiment code is MIT licensed. References retain their own licenses. Contributions should include a reproducible command, source revision, environment, correctness evidence, and a description of measurement scope. Small negative results are welcome.

## Primary references

- [ZKProphet](https://arxiv.org/abs/2509.22684)
- [FlashProve research prototype](https://github.com/anthropics/FlashProve)
- [Succinct prover documentation](https://docs.succinct.xyz/docs/provers/introduction)
