# Binary-field proving pipeline: final research report

A research project by [Requant Network](https://requant.network). Campaign completed on 2026-11-09, Asia/Yekaterinburg; raw timestamps use UTC.

## Finding

On one RTX 5060 Ti 16 GB, moving the complete GF(2^128) sumcheck transcript, including per-round SHA-256 challenges, into a CUDA Graph reduces the measured million-entry resident-input proof wall time from about 1.88 ms to 0.538 ms. This is a comparison with this repository's host-driven CLMAD implementation. It is not a comparison with an optimized upstream prover or a complete zero-knowledge system.

Including canonical host input binding and uploads raises the optimized path to about **21.91 ms** at that size. Hashing alone costs about 16.14 ms and uploads about 5.18 ms. Consequently, input handling is the next practical bottleneck; the 3.5x resident improvement is not an end-to-end application gain.

The experiment also overturns two premature conclusions. A slow Python CPU suffix does not imply a slow CPU suffix: a native PCLMUL implementation improves the host-driven pipeline. A fused GPU suffix improves that pipeline further, but after removing host challenges with a graph, the largest input is fastest without a single-CTA suffix. Optimization decisions change with the surrounding pipeline.

## Research questions and decisions

| Question | Evidence | Decision |
| --- | --- | --- |
| Does native carryless multiplication help exact reduced field arithmetic? | All GPU products match a separate bit-serial GPU implementation; sampled pairs match arbitrary-precision CPU arithmetic. | Yes against the tested scalar baseline; no bitsliced state-of-the-art comparison. |
| Can CPU suffixes reduce small-round overhead? | Python and native PCLMUL suffixes compared against complete CPU-gated transcripts. | Native suffix useful for a host-driven prover; Python-only rejection was insufficient. |
| Can GPU fusion remove transcript synchronization? | On-device SHA-256 checked against hashlib; shared and global-scratch suffixes match full transcripts. | Yes, within a size-dependent crossover. |
| Can the entire fixed-shape transcript be replayed on the device? | All-round CUDA Graphs, two seeds and exact transcript checks. | Yes; strongest resident-input result at one million entries. |
| Does this establish a faster production ZK prover? | No hiding layer, succinct commitment, upstream integration or application verifier. | No. The completed result concerns the defined algebraic prover. |

## Mathematical contract

The field is GF(2)[X]/(X^128 + X^7 + X^2 + X + 1). Bit i is the coefficient of X^i; elements are serialized as 16 little-endian bytes. `experiments/binary_field.py` independently forms an arbitrary-precision carryless product and reduces it. A Rabin irreducibility check and field identities are CPU self-checks.

Each claim is the XOR-sum of pairwise products. For a round polynomial g(T) = c0 + c1 T + c2 T^2, characteristic two gives g(0) + g(1) = c1 + c2. Both tables fold as x0 + r(x0 + x1). The initial SHA-256 binding includes the domain, the u32 little-endian input length, and both complete arrays. Each challenge hashes the previous digest and three canonical coefficients; the first 16 digest bytes interpreted little-endian form the next field element.

The verifier recomputes evaluations from public inputs and checks all round identities and the final product. Its work is linear in the input length. There is no witness hiding, succinct polynomial commitment, or formal Fiat-Shamir security claim. “Proof” in the timing tables means this explicitly defined transcript computation; the initial dot-product claim and verification are separate work.

## Implementations and validation

`cuda/binary_fields.cu` implements a 128-step scalar GPU reference and three 64-bit carryless products using Karatsuba, followed by exact polynomial reduction. The CUDA 13.3 CLMAD feature motivating this experiment is documented by [NVIDIA](https://developer.nvidia.com/blog/?p=119649). That feature is available on several GPU generations; these results do not establish a Blackwell-exclusive advantage.

The primitive campaign checks every GPU output against the separate GPU reference and sampled edge/random pairs against the CPU oracle. Complete CPU proof comparisons cover random, zero and maximum-element inputs at small and medium sizes. `binary_scale.py` runs the complete independent CPU prover at 262,144 and 1,048,576 entries for both seeds. Large fused/graph variants reuse those same CPU-gated fixtures and unchanged baseline sources/binary; their raw records identify the oracle record and input hash. This is transitive fixed-fixture conformance, not another independent CPU prover run for every variant.

`cuda/binary_fused.cu` contains fixed-length SHA-256 plus shared-memory and global-scratch single-CTA suffixes. The SHA routine is compared with 4,096 hashlib messages per seed. `native/binary_tail.cpp` uses x86 PCLMUL with runtime capability detection. `cuda/binary_graph.cu` keeps the running digest, challenge, coefficient reduction and folds on the device. Replays reset the transcript state, retain all graph buffers, and compare the complete output transcript. CPU checks reject changed coefficients, wrong claims and truncated proofs.

## Measured results

All latency cells below are **median / nearest-rank p95, milliseconds**. Seeds are kept separate.

### Reduced field multiplication (resident CUDA events)

| Seed | Elements | Scalar bit-serial | Karatsuba CLMAD + reduction |
| --- | --- | --- | --- |
| 20261111 | 4096 | 0.0122 / 0.0139 | 0.0093 / 0.0111 |
| 20261111 | 262144 | 0.0930 / 0.0945 | 0.0297 / 0.0306 |
| 20261111 | 1048576 | 0.3173 / 0.3281 | 0.1159 / 0.1260 |
| 20261112 | 4096 | 0.0120 / 0.0171 | 0.0092 / 0.0152 |
| 20261112 | 262144 | 0.0939 / 0.0953 | 0.0291 / 0.0299 |
| 20261112 | 1048576 | 0.3146 / 0.3225 | 0.1136 / 0.1202 |

### Complete resident transcript wall time

| Seed | Elements per input | Variant | Proof wall |
| --- | --- | --- | --- |
| 20261111 | 4096 | host_rounds | 0.9162 / 0.9402 |
| 20261111 | 4096 | graph_all_rounds | 0.1489 / 0.1570 |
| 20261111 | 4096 | graph_tail_1024 | 0.1306 / 0.1320 |
| 20261111 | 65536 | host_rounds | 1.2252 / 1.2436 |
| 20261111 | 65536 | graph_all_rounds | 0.2075 / 0.2149 |
| 20261111 | 65536 | graph_tail_1024 | 0.1865 / 0.1881 |
| 20261111 | 1048576 | host_rounds | 1.8698 / 1.9006 |
| 20261111 | 1048576 | graph_all_rounds | 0.5384 / 0.5526 |
| 20261111 | 1048576 | graph_tail_1024 | 0.5636 / 0.5684 |
| 20261112 | 4096 | host_rounds | 0.9281 / 0.9587 |
| 20261112 | 4096 | graph_all_rounds | 0.1485 / 0.1508 |
| 20261112 | 4096 | graph_tail_1024 | 0.1297 / 0.1341 |
| 20261112 | 65536 | host_rounds | 1.2337 / 1.2533 |
| 20261112 | 65536 | graph_all_rounds | 0.2059 / 0.2096 |
| 20261112 | 65536 | graph_tail_1024 | 0.1858 / 0.1928 |
| 20261112 | 1048576 | host_rounds | 1.8915 / 2.0056 |
| 20261112 | 1048576 | graph_all_rounds | 0.5384 / 0.5421 |
| 20261112 | 1048576 | graph_tail_1024 | 0.5627 / 0.5688 |

### Reconsidering a rejected CPU suffix (65,536 entries)

| Seed | Host-driven variant | Proof wall |
| --- | --- | --- |
| 20261111 | gpu | 1.2345 / 1.2574 |
| 20261111 | python_16 | 2.3865 / 2.4283 |
| 20261111 | native_1024 | 0.7517 / 0.7786 |
| 20261112 | gpu | 1.2576 / 1.2746 |
| 20261112 | python_16 | 2.4036 / 2.4960 |
| 20261112 | native_1024 | 0.7614 / 0.7825 |

### Canonical host inputs through returned transcript

| Seed | Elements per input | SHA-256 binding | Both input uploads | Full measured path |
| --- | --- | --- | --- | --- |
| 20261111 | 4096 | 0.0667 / 0.0724 | 0.0415 / 0.0437 | 0.2625 / 0.2684 |
| 20261111 | 65536 | 1.0118 / 1.0135 | 0.3639 / 0.3651 | 1.5911 / 1.6032 |
| 20261111 | 1048576 | 16.1339 / 16.1638 | 5.1780 / 5.1900 | 21.9089 / 21.9400 |
| 20261112 | 4096 | 0.0667 / 0.0723 | 0.0418 / 0.0428 | 0.2628 / 0.2728 |
| 20261112 | 65536 | 1.0123 / 1.0172 | 0.3641 / 0.3670 | 1.5921 / 1.6073 |
| 20261112 | 1048576 | 16.1416 / 16.1588 | 5.1721 / 5.1987 | 21.9110 / 21.9531 |

Raw records: [primitive/proof](results/binary-study.json), [large CPU oracle](results/binary-scale.json), [native suffix](results/binary-tail.json), [fused suffix](results/binary-fused.json), [large fused suffix](results/binary-fused-scale.json), [all-round graph](results/binary-graph.json), [input-inclusive follow-up](results/binary-input.json). Matching `-replication.json` files retain the second seed; primitive replication is named `binary-replication.json`.

The complete median/p95 tables for every tested suffix, pattern, primitive and timing scope are in [stage2-summary.csv](results/stage2-summary.csv). The earlier BabyBear campaign remains in [GPU_REPORT.md](GPU_REPORT.md).

## Measurement boundaries

Both Stage-2 input seeds are 20261111 and 20261112. Each timing variant has five warm-ups and thirty measured repetitions. Repetitions reuse a fixture; they are latency observations, not thirty independent workloads. Alternating order is used where variants share a timing loop. Dynamic clocks and OS activity remain possible confounders; no clock, voltage or power limit was changed, and no competing GPU research job was intentionally run.

Resident proof wall time includes graph launch/reset, proof D2H and Python coefficient decoding. It excludes input binding/upload, input generation, initial claim computation, graph planning/allocation/capture, compilation and verification. The host baseline includes warmed dynamic allocations and per-round CPU SHA-256. This comparison measures the deployed scheduling choices together; it does not isolate graph submission from allocation removal.

The input-inclusive follow-up starts with canonical packed host arrays and includes SHA-256 over both full inputs, both H2D uploads, digest update, graph replay, proof D2H and coefficient decoding. Host input generation/packing, initial claim, planning and verification remain excluded. It quantifies how input handling limits the practical gain; a resident 3.5x improvement must not be advertised as a 3.5x application speedup.

CUDA event intervals and wall measurements are separately labeled in the raw records. No joules-per-proof estimate is made: power was not sampled with sufficient temporal resolution. GPU allocation observations are not a full production prover memory budget.

## Reproduction

Use Linux x86_64, the pinned [CuPy requirements](requirements-gpu.txt), CUDA 12.x headers/runtime for CuPy, and CUDA **13.3** nvcc for CLMAD CUBINs. The measured nvcc is 13.3.73, target sm_120, g++ 13.3.0; compiler flags and binary/source hashes are in [stage2-build.json](results/stage2-build.json). Run without Python `-O`.

```sh
mkdir -p build results
nvcc -O3 -std=c++17 -arch=sm_120 -cubin cuda/binary_fields.cu -o build/binary.cubin
nvcc -O3 -std=c++17 -arch=sm_120 -cubin cuda/binary_fused.cu -o build/binary_fused.cubin
nvcc -O3 -std=c++17 -arch=sm_120 -cubin cuda/binary_graph.cu -o build/binary_graph.cubin
g++ -O3 -std=c++17 -shared -fPIC -mpclmul -msse2 native/binary_tail.cpp -o build/libbinary_tail.so
python collect_binary_build.py
python experiments/binary_field.py
export CUDA_PATH=/path/to/cuda-12.8
export PYTHON=/path/to/cupy-venv/bin/python
STUDY=binary_study.py bash run_gpu.sh
STUDY=binary_study.py bash run_gpu.sh --seed 20261112 --output results/binary-replication.json
STUDY=binary_scale.py bash run_gpu.sh
STUDY=binary_scale.py bash run_gpu.sh --seed 20261112 --output results/binary-scale-replication.json
STUDY=binary_tail_study.py bash run_gpu.sh
STUDY=binary_fused_study.py bash run_gpu.sh
STUDY=binary_fused_scale.py bash run_gpu.sh
STUDY=binary_graph_study.py bash run_gpu.sh
STUDY=binary_input_study.py bash run_gpu.sh
python verify_stage2.py
```

Repeat the last five studies with `--seed 20261112 --output results/<study-name>-replication.json`, preserving their documented default filename stem. The scale and graph studies depend on earlier oracle records, so retain the order above. The million-entry independent Python CPU prover is intentionally slow. For an offline artifact check, run only `python verify_stage2.py`; it verifies sources and derives the public CSV without a GPU. CI performs CPU/reference and artifact checks; it does not rerun the GPU campaign.

## Conclusion and remaining research

The completed campaign supports device-resident transcript scheduling for this sumcheck and rejects a universal suffix cutoff. The next distinct study would integrate these choices into a pinned binary-field proof system with a genuine application verifier, hiding/commitment costs and input-binding pipeline. An optimized bitsliced baseline, larger composition polynomials, multi-GPU behavior and measured energy remain open. No claim of research novelty, production security or mining rewards follows from this campaign.
