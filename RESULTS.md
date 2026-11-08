# Research results

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
