# Strict publication audit — 2026-10-16

## Decision and scope

The revised article retains the author's kernel contribution and the historical **468× kernel**
and **200.17× end-to-end** results with their distinct configurations and sources. The four-run
language-model measurements are independently recalculated from the included metrics.
Editorial/source checks pass. Two artifact reproducibility findings remain open below.
This audit is not a new TPU benchmark, training run or evaluation of the safetensors exports.

## Completed corrections

- Sole author: **Akseleu Omirbay**, including PDF metadata. Authorial voice is I/my; no we/our.
- Kernel results restored in abstract, introduction, Section 3 and conclusion. Table 3 contains
  all six timing configurations; Table 4 gives selected numerical-validation records.
- Fig. 2A remains the main mean ± sample SD view. Fig. 4 is appendix **per-seed detail**;
  it is explicitly the same two seeds, not additional independent evidence.
- No “requires an equivalence audit” wording in the paper. Kernel checks are described by their
  actual measured quantities, inputs and tolerances.
- The final PDF has **13 pages**, **16 unique external URLs**
  with successful responses and **29 resolved internal links**.
  All six bibliography entries are cited. Every page was rendered and reviewed, with enlarged
  checks of kernel tables, method, algorithm, architecture and appendix. No overfull boxes or
  undefined references; one harmless underfull bibliography warning remains.

## Numerical and methodological checks

All source metric hashes and all 41 shared-step means/sample SD were recomputed. Both seed pairs
favor hybrid at all 41 common evaluations. At exactly 61,500 steps / 4,030,464,000 target tokens:
Baseline 1.005909 ± 0.000227; hybrid 0.993838 ± 0.000111; difference −0.012071 estimated BPB.
The requested plot cutoff (Baseline 63,000; hybrid 62,673/62,683) stays separately labelled;
its difference is −0.011617 and is not exactly token matched. Threshold statistics use first
logged per-seed crossings, not interpolation. SD uses ddof=1 over two seeds, not a confidence interval.

The article retains the 4.2 bytes/token proxy, repeated validation, single corpus/context,
unequal parameters/FLOPs, time-based learning-rate schedule and absence of an isolated DAR
ablation. Its claims concern the complete architectures. Exact byte-level BPB, causality of
the quality gain, downstream performance and broad generalization are not inferred.
Training source snapshots and actual FFN-inclusive block update ordering were checked.

## Kernel evidence ledger

| Claim | Evidence and scope | Audit result |
|---|---|---|
| Up to 468× kernel | `atomic-ops/attestation/scaling.json`: 467.64×, H=6, L=2048, B=4, D=128, inner block 64, single v5e; supplied JAX associative-scan wrapper | All six reported timing rows retained; reported ratios consistent with displayed-time rounding |
| Forward relative L2 about 2.5e-6 | Scaling JSON: H4/L2048 2.486e-6; H6/L2048 2.481e-6 | Source checked; these are forward errors for two shapes |
| Gradient-gate maximum | Aggregate `1.15.summary`: 3.427e-5 across six named cases | Corrected exact maximum; six cases, not six independent random seeds |
| Finite differences | Aggregate serial and kernel records with their respective tolerances | Selected checks retained; no claim of exact arithmetic equality |
| 200.17× end-to-end | Kernel README and `docs/ATTESTATION.md`: 27.5M LM, B8/L2048/H6/D128; 103.29 vs 20674.62 ms | Documentation checked, ratio compatible with rounded input timings; raw-record gap K1 |
| Aggregate validation | Summary 145 PASS / 1 FAIL; actual cell counts 145 PASS / 6 INFO / 1 FAIL / 1 ERROR | FAIL is OLD MQAR OOM; ERROR is a disk-full scale attempt documented as superseded; not 146 numerical tests |

The separate aggregate 27.49M/B8/L4096 measurement is 169.6 vs 26764.55 ms (~157.81×).
It is not the source of the 200.17× claim. Neither small-LM timing is assigned to the current
307M hybrid. The code uses selective recomputation plus saved chunk arrays; it does not have
constant training memory. Precision modes, damping and clipping are stated as implemented.

## Open findings and concrete acceptance criteria

### K1 — Major: exact end-to-end execution artifact not included

The 200.17× result is present in two original kernel documents. The inspected committed aggregate
does not contain its exact 27.5M/L2048 execution record, and original nb logs are not included.
The historical source identifiers also mix short Git identifiers and a labelled content hash.
This does not contradict the documented result or invalidate the numerical checks.

**Closure:** attach the original execution output, exact resolved code revision/configuration,
timing protocol (warm-up, synchronization, repeats), environment and loss/optimizer scope.
If that artifact cannot be recovered, a new run should be published as a new measurement with
its own record; it must not be presented as the recovered original run.

### K2 — Major: public scaling notebook wrapper is not executable as written

`notebooks/speedup_scaling.ipynb` defines eight-argument `fwd_old`/`fwd_new`, but precheck and
correctness calls expand seven input values and append scale plus h0: nine arguments.
The broad exception can turn the TypeError into missing timing. The OLD wrapper also ignores
supplied scale/h0, returns a zero final state and normalizes affine transitions; timing repeat
counts differ. These are concrete notebook-wrapper defects, not a demonstrated failure of the
Pallas numerical kernel or proof that historical timings are false.

An optional corrected harness already exists in this model repository:
`scripts/reproduce_kernel_bench.py`. It has not been run on TPU for this revision.
**Closure:** run a corrected equal-input/loss harness, retain forward/gradient checks, synchronized
warm timings and full outputs. Keep the original benchmark evidence unchanged. The atomic-ops
repository was not modified during this revision.

## Publication status

The article is editorially revised with source-backed historical kernel results and reproducible
four-run statistics. The two open findings concern independently replaying historical benchmark
artifacts. A claim of complete independent reproduction or universal production readiness would
exceed this audit. Additional seeds/component ablations are future scientific work, not invented
requirements for retaining the current measured architecture comparison.

Machine-readable receipt: `attestation/article-audit.json`. This report avoids repeating raw logs
inside the paper; full current training logs remain in `artifacts/four-runs-20261016.zip`.
