# Optimized comparison and input-pipeline closure

Requant Network, 12 November 2026. This supplement resolves the optimized-comparator and canonical input-pipeline questions in the [historical report](FINAL_REPORT.md).

## Comparison contract

The new comparators are the pinned Binius-GPU `ghash_ctmul32` and `ghash_ctmul64` implementations at commit `293cd240f0f73b37fa03f3082d616dc81d8cb3e5`. They implement the **same polynomial and little-endian basis** as this experiment, so comparison does not change the field or reinterpret input bits. They are optimized constant-time software multiplication implementations; they are not the binary-tower bitsliced prover. The latter uses a different basis and cannot be substituted without a validated field isomorphism. The native CLMAD result must therefore be compared against these optimized same-field implementations before using the old scalar headline.

Both independently generated CPU oracle cohorts contain full arbitrary-precision transcripts through 1,048,576 entries. Five cases per cohort cover random/zero/max input at 4,096 entries plus larger random vectors at 65,536 and 1,048,576. All GPU variants must match the complete CPU transcript on every repetition. Host-driven tests use the same allocations, coefficient transfers, challenge hashing and fold schedule; variant order alternates, with five warm-ups and thirty samples.

| Cohort | Entries | CLMAD host-driven median, ms | ctmul32, ms | ctmul64, ms |
| --- | ---: | ---: | ---: | ---: |
| 20261112 | 4096 | 0.9268 | 0.9244 | 0.9230 |
| 20261112 | 65536 | 1.2519 | 1.2523 | 1.2580 |
| 20261112 | 1048576 | 1.9014 | 2.0478 | 2.1147 |
| 20261212 | 4096 | 0.9356 | 0.9354 | 0.9368 |
| 20261212 | 65536 | 1.2642 | 1.2748 | 1.2729 |
| 20261212 | 1048576 | 1.9378 | 2.0850 | 2.1533 |

At the largest input the host-driven difference is modest, around 8–11%, rather than the much larger ratio against bit-serial arithmetic. At smaller inputs dispatch and challenge handling dominate. This narrows the supported arithmetic claim; it does not establish a speedup for an upstream complete prover.

## Captured scheduling and input overlap

The [graph builder](build_optimized_graph.py) changes only the multiplication backend inside the same complete captured transcript, including the device SHA-256 challenge chain. [The input study](input_overlap_study.py) compares serial ordinary buffers, serial reusable pinned buffers, overlapping pinned upload with CPU hashing, and a staged path that includes copying caller ordinary buffers into pinned memory. Both concurrent operations only read immutable input bytes. The upload and hash finish before the dependent transcript executes. Each proof and each binding digest must match the CPU oracle.

Pinned allocation/preparation is measured separately. Full wall timing includes the canonical input hash, both uploads, digest update, captured proof, proof download and decoding. It excludes a real application claim, commitment protocol and application verifier. The resident arithmetic comparison uses identical captured scheduling across CLMAD and both optimized comparators. Complete median/p95 results are in [followup-summary.csv](results/followup-summary.csv).

The first input-overlap campaign exposed a timing defect: CUDA events were not explicitly attached to the graph's launch stream and some resident samples were implausibly short. Its original source and raw records remain as a negative instrumentation control. Those resident timings are invalid and the original input timing is not used as authoritative complete-path evidence. [The corrected driver](input_validated_study.py) attaches events/uploads/downloads to explicit streams, waits for graph completion inside the host boundary, and clears every output to a sentinel outside each timer so stale matching outputs cannot pass. Only its independently replicated results support the input/graph conclusions below. The retained legacy probe confirms that creating the old graph changed the caller's current stream in both cohorts; that graph itself replays correctly when explicitly fenced. This is a caller/measurement-stream defect, not evidence that the old field arithmetic is wrong.

| Seed | Entries | Ordinary serial, ms | Pinned serial, ms | Pinned overlap, ms | Staged overlap, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20261112 | 65536 | 1.6095 | 1.5747 | 1.2564 | 1.3276 |
| 20261112 | 1048576 | 21.9587 | 21.8678 | 16.8209 | 18.7861 |
| 20261212 | 65536 | 1.6157 | 1.5843 | 1.2656 | 1.3290 |
| 20261212 | 1048576 | 21.9577 | 21.8565 | 16.8275 | 18.7847 |

At 1,048,576 entries, overlap lowers the pinned input-inclusive median from about 21.9 ms to 16.8 ms in both cohorts, approximately 23%. Including caller staging reduces the gain to about 14%. The corrected captured resident arithmetic comparison is:

| Seed | CLMAD, ms | ctmul32, ms | ctmul64, ms |
| --- | ---: | ---: | ---: |
| 20261112 | 0.46163 | 0.75789 | 0.84222 |
| 20261212 | 0.46157 | 0.75776 | 0.84122 |

Under identical captured scheduling CLMAD reduces resident arithmetic time by about 39% versus ctmul32 and 45% versus ctmul64. At the full input boundary the corresponding difference is only about 2%, because input hashing/movement dominate. The useful complete-path result is safe overlap; the scalar-arithmetic headline does not describe this input-fed application boundary. Host-wall resident measurements are retained alongside the explicitly recorded device events as a separate timing sanity check.


## Decision and scope

Use the optimized same-field comparison to assess arithmetic, and the input-inclusive comparison to assess host-fed work. Reusable pinned buffers are an application precondition, not an automatic free optimization. The result remains an exact algebraic sumcheck/transcript experiment. Integrating it into a hiding/commitment proof system is not a source-compatible backend swap: field basis, transcript and application verifier contracts must all agree. No complete-prover security or market claim follows from this experiment. Larger composition polynomials, different proof systems and heterogeneous multi-GPU execution are separate workloads, not implied successes.

Reproduce `python build_optimized.py UPSTREAM --nvcc NVCC`, generate each CPU oracle with [optimized_oracle.py](optimized_oracle.py), run [optimized_study.py](optimized_study.py), then build the optimized graphs and run `python input_validated_study.py results/optimized-oracle-20261112.json`; repeat for seed 20261212 and separate replication filenames. Source, oracle, binary and pinned-header hashes are retained. `python verify_followups.py` validates all new source/result bindings, sample counts and exact-result assertions; CI also retains the original CPU arithmetic/altered-transcript checks. These checks replay evidence, not GPU timing.

## Server timing boundary

GPU campaigns run serially on the same RTX 5060 Ti. Clocks, power limits and unrelated server services are unchanged. Measurements describe this server and retained workload; small percentage differences are not treated as universal improvements. No energy measurement or heterogeneous multi-GPU throughput is inferred.
