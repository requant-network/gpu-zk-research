# Research results

**Completed campaign:** Completed Stage 2: exact GF(2^128) CLMAD arithmetic, native CPU and fused GPU suffixes, complete device-resident transcript graphs, and input-inclusive timing. The million-entry resident transcript takes about 0.538 ms; canonical host binding/uploads raise the measured path to about 21.91 ms. This is an algebraic public-input prover, not a complete ZK system.

The [final report](FINAL_REPORT.md) supersedes the preliminary scope and next-step list below. It contains the completed experiments, reproducible commands, negative outcomes and remaining work beyond this campaign. Historical Stage-1 details follow.

Research date: 2026-11-09, Asia/Yekaterinburg. Machine-readable records use UTC.

## CPU correctness

15 honest proofs verified; 45 altered-coefficient, incorrect-claim, and truncated-transcript cases rejected. The full-input verifier provides a baseline for future authenticated polynomial openings.

Both Windows/Python 3.12.10 and Linux x86_64/Python 3.12.3 runs passed. Raw records include source hashes and timing scope:

- [Windows baseline](results/cpu-baseline.json)
- [Linux baseline](results/cpu-linux.json)

## GPU campaign

The RTX 5060 Ti campaign has now executed successfully. The initial `cuda/probe.cu` diagnostic passed against its exact CPU fixture. The extended study and independent-seed replication are documented in [GPU_REPORT.md](GPU_REPORT.md), with full source hashes, timings, correctness checks, negative results, and next research decisions.

- [Primary GPU observations](results/gpu-study.json)
- [Independent input-seed replication](results/gpu-replication.json)
- [Environment](results/gpu-environment.json)
- [Reproduction and measurement rules](METHODS.md)

These experiments measure arithmetic and systems behavior. They do not establish full-protocol security, upstream verifier acceptance, or mining revenue.
