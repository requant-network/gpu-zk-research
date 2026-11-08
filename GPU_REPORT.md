# GPU sumcheck and binary-field research

Research by [Requant Network](https://requant.network). Campaign date: 9 November 2026, Asia/Yekaterinburg; raw records use UTC.

This is an executed, reproducible research campaign on a GeForce RTX 5060 Ti 16 GB (36 SMs), paired with an AMD Ryzen 5 5500. The host exposes PCIe Gen 3, with an observed x8 link in the environment snapshot. Do not extrapolate transfer timings to a Gen 5 workstation or a datacenter GPU. The software record reports Python 3.12.3, NumPy 2.2.6, CuPy 13.6.0, NVRTC 12.8, runtime API 12090 and driver 610.43.03. CUDA compilation uses 12.8.61 unless explicitly stated otherwise.

Each variant has five warm-ups and thirty measured repetitions per case. A second campaign changes the input seed from 20261109 to 20261110. Tables show milliseconds. p95 uses the nearest-rank definition. Repetitions reuse each case's inputs; they do not establish population confidence intervals. Default dynamic clocks were retained. The experiments ran sequentially on the released GPU.

[Primary raw results](results/gpu-study.json), [independent-seed replication](results/gpu-replication.json), [complete median/p95 table](results/gpu-summary.csv), [environment](results/gpu-environment.json), [methods](METHODS.md), and [executable study](gpu_study.py) are part of this record. `python summarize_results.py` verifies source hashes and regenerates the CSV. Correctness checks run outside the measured intervals.

## Questions and mathematical contract

We test whether fold/round fusion improves a transcript-dependent dot-product sumcheck, whether coefficient reduction belongs on CPU or GPU, and whether input binding erases a kernel improvement.

The field is BabyBear, p = 2013265921. For each adjacent pair define da = a1-a0 and db = b1-b0. A round sends c0 = sum(a0*b0), c1 = sum(a0*db+b0*da), c2 = sum(da*db), all modulo p. It must satisfy 2*c0+c1+c2 = the current claim. SHA-256 derives the next challenge from the input-bound transcript and these canonical coefficients. The next tables are a0+r*da and b0+r*db. No challenge is known before the preceding coefficients are available.

The verifier recomputes the final multilinear evaluations from all public inputs. This is an exact public-input algebraic test: verification is linear, the 31-bit field is a prototype parameter, and there is no hiding, polynomial commitment opening, or claimed cryptographic security level. Repeated proofs are not independent security trials.

## Implementations and correctness

The separate variant launches a fold followed by coefficient work. The fused variant combines the previous round's fold with the next round's coefficient calculation. The final fold is omitted because its result is checked by the independent verifier and is not transmitted. All variants return the same complete transcript. A third variant reduces block coefficients to three field elements on GPU before copying them to the host.

Both seeds cover N = 1024, 16384, 262144 and 1048576, with zero, maximum-element, and random inputs. Every measured proof equals the arbitrary-precision CPU transcript, and each original transcript passes the independent verifier. Every case also rejects a changed coefficient. The Stage-0 suite separately rejects incorrect claims and truncated transcripts. The study checks 24 distinct input/proof cases and 4200 variant/transcript executions across the two seeds.

## GPU-resident round pipeline

Inputs and the initial commitment are prepared before this timer. Wall time includes device allocations from a warmed pool, kernel enqueue, coefficient D2H, CPU SHA-256 challenge generation, and synchronization. It excludes input upload and verifier work. Therefore these are not complete proof-service latencies.

| N | Separate median | Fused median | GPU reduction median | GPU reduction p95 | Replication median |
| --- | --- | --- | --- | --- | --- |
| 1024 | 0.7310 | 0.6374 | 0.6260 | 0.6378 | 0.6242 |
| 16384 | 1.0151 | 0.8916 | 0.8779 | 0.8973 | 0.8750 |
| 262144 | 1.3305 | 1.1627 | 1.1270 | 1.1453 | 1.1313 |
| 1048576 | 1.5270 | 1.3557 | 1.2423 | 1.2826 | 1.2355 |

Fusion consistently helps, but its benefit is smaller than a headline based on a folding kernel alone would suggest. GPU coefficient reduction helps the largest tables, but an extra reduction launch can cancel that benefit for small tables. The measured resident gains are around the project's provisional 20% gate at the largest size; they should not be presented as a robust protocol-level 20% improvement.

## Input binding and upload included

Both paths compute exactly the same SHA-256 commitment bytes and use the same fused prover. The original path packs Python integer lists into byte strings. The streaming path feeds the contiguous little-endian NumPy arrays directly into the same hash. Both include input H2D and every round; they exclude random input generation, list construction and verification.

| N | Original binding median | Streamed binding median | Streamed p95 | Replication streamed median |
| --- | --- | --- | --- | --- |
| 1024 | 0.7558 | 0.7276 | 0.7445 | 0.7286 |
| 16384 | 1.4058 | 1.0491 | 1.0726 | 1.0453 |
| 262144 | 14.5778 | 2.5879 | 2.6177 | 2.5731 |
| 1048576 | 77.0422 | 7.0340 | 7.0970 | 7.0102 |

At the largest size, removing Python packing and temporary copies matters much more than the incremental round-kernel optimization. This is an implementation finding, not a faster hash function or a new proving protocol. A native prover that already hashes contiguous buffers should not expect this packing-related gain. Two input tables occupy 8 MiB at N = 2^20; this excludes scratch, CUDA context and host copies. Peak process/VRAM use was not profiled.

## CUDA 13.3 carryless multiplication: a tested new lead

[NVIDIA's CUDA 13.3 article](https://developer.nvidia.com/blog/?p=119649) introduces `clmad` for binary-polynomial multiplication. We compiled a separate SM-120 executable with CUDA 13.3 and verified that both the high and low halves execute correctly on this 5060 Ti. Every output after every launch is checked against an independent CPU polynomial oracle using a 128-bit integer. Zero, all-ones and high-bit boundary inputs are included. This separate primitive probe uses seed 20261109 only; the two-seed replication above applies to the BabyBear pipeline.

| Elements | Dependent products | Scalar bit-loop median | CLMAD median | Ratio | CLMAD p95 |
| --- | --- | --- | --- | --- | --- |
| 4096 | 1 | 0.0039 | 0.0037 | 1.05x | 0.0044 |
| 4096 | 4 | 0.0054 | 0.0040 | 1.34x | 0.0044 |
| 262144 | 1 | 0.0327 | 0.0073 | 4.49x | 0.0085 |
| 262144 | 4 | 0.0722 | 0.0163 | 4.43x | 0.0168 |
| 1048576 | 1 | 0.1217 | 0.0209 | 5.83x | 0.0228 |
| 1048576 | 4 | 0.2743 | 0.0550 | 4.99x | 0.0558 |

The comparison baseline is a scalar, fully unrolled bit-loop, not an optimized bitsliced library. The result establishes instruction availability, exact unreduced products, and a useful microbenchmark lead. It does not establish a complete GF(2^128) multiplication, GHASH throughput, or sumcheck acceleration. Polynomial reduction and a binary-field protocol remain necessary. Changing from BabyBear to a binary field changes the protocol and cannot be inserted as an arithmetic shortcut.

[Carryless source](cuda/carryless.cu), [raw output](results/carryless.json), [compiler and source provenance](results/carryless-provenance.json). Reproduce separately with CUDA 13.3:

```sh
mkdir -p build
/path/to/cuda-13.3/bin/nvcc -std=c++17 -O3 -arch=sm_120 cuda/carryless.cu -o build/carryless
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./build/carryless > results/carryless.json
```

## Relation to current work and next decisions

[zkDuel, October 2026](https://arxiv.org/abs/2610.37710) also investigates launch and reduction overhead across field widths. Its H100/A100 comparisons do not predict this consumer-card result. Our fold fusion and reduction are engineering ablations, not a novelty claim over existing provers. NVIDIA's binary-field results motivate the separate carryless experiment; their reported speedups are not our measurements.

1. **Prioritize binary-field exploration:** implement and independently check irreducible-polynomial reduction, then compare complete GF(2^128) rounds with a pinned optimized bitsliced baseline. This is the most current mathematical/systems lead in this campaign.
2. **Retain contiguous input binding and fused BabyBear rounds.** Replace Python dispatch with a native persistent pipeline only after accounting for the sequential challenge dependency.
3. **Before prover-market claims:** integrate an authenticated polynomial commitment and a pinned upstream verifier. Benchmark its complete proof artifact, memory use, fees and verification separately.

No complete ZK system, network submission, energy measurement or mining/prover income was tested. This report closes the bounded Stage-1 arithmetic campaign and states the remaining protocol work explicitly.
