# Reproducing the reported protocol

1. Use the immutable model/kernel commits recorded in `attestation/training.json` and each raw provenance.
2. Match Linux dependency pins in `scripts/requirements.lock`, TPU v5e-8, bf16 compute/fp32 parameters.
3. Use windmaple/openwebtext-gpt2 uint16 GPT-2 token files. Available train=9,035,582,489 tokens;
   val=4,434,606. Raw provenance contains file sizes and edge hashes; these are not full-corpus hashes.
4. Set batch=32, sequence=2048, accum=1, epochs=1, init seeds 1/2, data seeds 43/44,
   dropout seeds 124/125, eval_every=1500, eval_batches=100, checkpoint_every=5000.
5. Fresh optimizer, no checkpoint resume. Peak LR 3e-4, weight decay .1, grad clip 1,
   Muon/AdamW split by parameter shape/type. LR is elapsed-time cosine with the first-100-update warmup bound.
6. Total notebook budget=32,040 seconds including setup/compile; reserve=600 seconds inside that budget.
   A stop at deadline is not a completed data pass. Record actual steps/tokens and checkpoint success.
7. Compare all common logged steps, then separately report the selected near-step endpoints.

The ZIP in `artifacts/` contains the original resolved config, console log, training log, metrics,
runtime preflight and provenance for all four runs. Tables/plots never interpolate missing evaluations.
Model configs in `configs/` are defaults; resolved run configs override them and define the experiment.
The original notebook bootstrap owned the setup-inclusive deadline; an ordinary CLI run does not.

The current repository is prepared from the measured implementation; current publication commits
are not misrepresented as the commits on which training happened.

## Public source recovery

The original model commit IDs identify training in private development history. Readers do
not need access to that history: `artifacts/four-runs-20261016.zip` includes each run's
`raw/<run>/<run>/source/atomic-gdn2/` and `source/atomic-ops/` payloads, plus hashes and
resolved run configs. Use the matching payload, not an assumed public commit URL.
Initialization seeds are 1/2, data seeds 43/44 and dropout seeds 124/125.
Portable metric paths and kernel experiment identities are in
[`attestation/current-evidence.json`](../attestation/current-evidence.json).
