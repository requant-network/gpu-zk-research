# Initial research results

Research date: 2026-11-09, Asia/Yekaterinburg. Machine-readable records use UTC.

## CPU correctness

15 honest proofs verified; 45 altered-coefficient, incorrect-claim, and truncated-transcript cases rejected. The full-input verifier provides a baseline for future authenticated polynomial openings.

Both Windows/Python 3.12.10 and Linux x86_64/Python 3.12.3 runs passed. Raw records include source hashes and timing scope:

- [Windows baseline](results/cpu-baseline.json)
- [Linux baseline](results/cpu-linux.json)

## CUDA preparation

`cuda/probe.cu` compiled successfully with NVIDIA CUDA 12.8.61, C++17, optimization O3, and `sm_120`. The compiler returned exit code zero. The diagnostic executable was not run because the owner reserved the occupied GPU for another job. GPU correctness, latency, energy, and complete-pipeline speedup remain unmeasured.

- [Compiler output](results/cuda-build.log)
- [Reproduction and measurement rules](METHODS.md)

The CPU measurements establish a reproducible arithmetic baseline. They do not establish GPU performance, protocol security, network acceptance, profitability, or a production implementation. Next-stage experiments are recorded in [RESEARCH.md](RESEARCH.md).
