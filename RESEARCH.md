# Research plan

## Question

Can a memory-efficient sumcheck folding pipeline reduce complete proof-generation cost while preserving every algebraic check?

## Hypothesis and boundaries

The first experiment implements a degree-two dot-product sumcheck over the BabyBear prime 2013265921. It binds the transcript to the public inputs with SHA-256, checks every round identity, and recomputes the final multilinear evaluation from the public inputs. Verification is linear in the input size. This is an algebraic research baseline, not a zero-knowledge or succinct proof system. Modulo reduction of SHA-256 challenges is a deterministic prototype choice, not a formal Fiat-Shamir security claim.

## Completed first experiment

python run.py validates 15 honest proofs and rejects 45 altered coefficients, wrong claims, and truncated transcripts. Input sizes are 2, 16, 256, 4096, and 16384. Patterns include zero, maximum canonical field elements, and seeded random values.

## Completed GPU campaign

[GPU_REPORT.md](GPU_REPORT.md) records the executed two-seed campaign, mathematical contract, independent checks, raw timings, ablations, negative findings and next decisions. All study sources and records are public. This extends the CPU baseline without claiming a complete cryptographic protocol.

## Next experiments

1. Establish a complete prover profile with a fixed upstream release and verifier.
2. Extend the now-tested round fusion and coefficient reduction to a pinned proof system; investigate complete binary-field rounds using the tested CUDA 13.3 carryless primitive.
3. Measure data movement, peak memory, proof verification, and joules per verified proof.

## Evidence required

Report a full-operation comparison against a pinned baseline, independent correctness checks, memory requirements, and energy observations. Distinguish algorithmic improvement from changed workloads or weaker validation. An upstream verifier must accept any claimed protocol improvement.

## Sources

- [ZKProphet](https://arxiv.org/abs/2509.22684)
- [FlashProve research prototype](https://github.com/anthropics/FlashProve)
- [Succinct prover documentation](https://docs.succinct.xyz/docs/provers/introduction)
