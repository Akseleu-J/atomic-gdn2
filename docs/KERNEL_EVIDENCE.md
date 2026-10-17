# Kernel evidence and model-training provenance

Evidence status: 17 October 2026. Model quality and kernel performance have separate protocols.

## Source identities

| Experiment | Kernel revision | Numerical scope |
|---|---|---|
| Four completed language-model runs | `0dbc59cf74883315d7adbebe9229c520663e0d58` | Original resolved training settings; both architectures share this kernel |
| Four-path speed experiment | `11616acd8a061768a38feb12c58e5e329952f5d5` | Zero damping; measured precision, leaf sizes and rematerialization policies |

The implementation has three forward stages, a segment-fused custom backward and selective
recomputation. Saved arrays scale with chunk count. The current package is `atomic_gdn2`;
the earlier `atomic_ops` path is a separate comparator. `wy_eps` rescales triangular
coupling in `[I + (1-eps)T]^-1`; it is not diagonal Tikhonov loading.

## Current performance evidence

Single TPU v5e-1, three seeds, three rounds per seed, two warmups and four timed calls
per round for each of four paths. Full return trees are synchronized; compilation and
host checks are outside timing. Each call replays the initial state.

| Workload | OLD/NEW | JAX_REF/NEW |
|---|---:|---:|
| Six-input gradient microbenchmark, B4/H6/L2048/D128 | 411.75 ± 5.13× | 34.50 ± 0.43× |
| Complete synthetic LM optimizer step, 34,873,952 parameters, B8/L2048 | 117.76 ± 0.15× | 4.531 ± 0.004× |

SD is sample SD across three seed summaries, not a confidence interval or cross-machine
variation. Ratios are summarized from paired round ratios. OLD uses block rematerialization
in the full step. OLD and JAX_REF are my implementations, not external library baselines.
The experiment has 1008 timed calls and 504 warmups; all six kernel shapes are reported.

[Protocol and all timing samples](https://github.com/Akseleu-J/atomic-ops/blob/main/benchmarks/README.md).

## Numerical agreement

Recorded kernel-gradient and full-step parameter/state/loss norm gates pass. NEW versus
JAX_REF parameter-update checks pass all three seeds. NEW versus OLD update checks fail
at seeds 0 and 1 and pass at seed 2. Passing parameter norms is not equivalent to passing
update comparisons or proving equal optimizer trajectories.

A full-model coordinate capture documents severe cancellation near AdamW's epsilon scale.
Exact summation of its captured FP32 factors retains 98.2542% of the selected seed-0
OLD/NEW gradient difference: the final convolution reduction alone does not explain it.
The exact TPU audit verifies frozen factors (27 cross contractions, nine decompositions,
nine permutation checks, eight controls), not the unrounded full-network gradient.
It does not establish the exclusive cause of every upstream discrepancy. A separate
seed-1 decay_a discrepancy remains outside that coordinate investigation.

[Numerical methods, controls and scope](https://github.com/Akseleu-J/atomic-ops/tree/main/numerical).

## Historical evidence and the earlier paper

Earlier scaling, finite differences, MQAR and model integration remain useful evidence for
their recorded revisions. The mixed historical aggregate reports 145 PASS and one OLD
MQAR OOM; these are not 146 independent numerical proofs of the current benchmark.
Earlier 468×/200.17× figures are historical measurements, superseded for current headline
reporting by the new protocol. They are not invalidated or silently redefined as the new run.

[Historical records](https://github.com/Akseleu-J/atomic-ops/tree/main/attestation) ·
[Original archive roles and hashes](https://github.com/Akseleu-J/atomic-ops/blob/main/artifacts/INDEX.json).

The 16 October model paper still contains the earlier kernel section. See
[paper status](../paper/README.md). Its revision is a separate editorial task.
The optional `scripts/reproduce_kernel_bench.py` in this model repository is a different
highest-precision diagnostic, not the harness that generated the current four-path table.
