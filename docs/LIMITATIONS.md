# Limitations

- Two seeds provide an observed sample SD, not a robust confidence interval or a universal quality claim.
- One corpus/tokenizer, L=2048, validation only; no independent test or downstream/generation benchmark.
- MLA, routing and parameter counts differ; no component ablation isolating DAR, and no matched-FLOP comparison.
- Same time-based LR rule yields different LR values at the same step across models.
- Full training data pass was not completed. The near-step baseline cutoff was selected after the runs;
  actual baseline checkpoints at 65,000 and their logs remain available.
- BPB is estimated from CE with 4.2 bytes/token; validation drops a partial batch tail.
- First observed crossing is quantized by eval_every=1500; crossing SD=0 does not mean exact crossing time.
- Clip/nan_to_num frequency and causal effect on quality were not separately measured.
- Saved checkpoint markers do not on their own establish restoration or semantic correctness.
- A kernel microbenchmark is not a model-level speedup. Historical benchmark wrappers have known
  equivalence/reproduction issues documented in KERNEL_EVIDENCE.md.
