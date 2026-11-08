# Research program and decisions

A research project by [Requant Network](https://requant.network).

## Completed campaign

Can finite-field arithmetic and transcript scheduling reduce the complete measured sumcheck path without changing algebraic checks?

The [final report](FINAL_REPORT.md) documents the hypotheses, mathematical contract, two-seed measurements, independent checks, negative outcomes and conclusions. Completion refers to this bounded experimental campaign; it does not mean the field has no open questions.

| Completed experiment | Executable source | Primary evidence |
| --- | --- | --- |
| Exact binary-field arithmetic | [binary_study.py](binary_study.py) | [results/binary-study.json](results/binary-study.json) |
| Independent large-input CPU prover | [binary_scale.py](binary_scale.py) | [results/binary-scale.json](results/binary-scale.json) |
| Python versus native CPU suffix | [binary_tail_study.py](binary_tail_study.py) | [results/binary-tail.json](results/binary-tail.json) |
| Fused GPU suffix and SHA-256 | [binary_fused_study.py](binary_fused_study.py) | [results/binary-fused.json](results/binary-fused.json) |
| Full device-resident transcript | [binary_graph_study.py](binary_graph_study.py) | [results/binary-graph.json](results/binary-graph.json) |
| Input-inclusive path | [binary_input_study.py](binary_input_study.py) | [results/binary-input.json](results/binary-input.json) |

The matching replication records and complete median/p95 data are indexed in [stage2-summary.csv](results/stage2-summary.csv). [GPU_REPORT.md](GPU_REPORT.md) preserves the preliminary Stage-1 experiments.

## Work beyond this campaign

The completed campaign supports device-resident transcript scheduling for this sumcheck and rejects a universal suffix cutoff. The next distinct study would integrate these choices into a pinned binary-field proof system with a genuine application verifier, hiding/commitment costs and input-binding pipeline. An optimized bitsliced baseline, larger composition polynomials, multi-GPU behavior and measured energy remain open. No claim of research novelty, production security or mining rewards follows from this campaign.
