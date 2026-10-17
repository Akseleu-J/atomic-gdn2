# Public evidence status — 17 October 2026

The [16 October audit](history/PUBLICATION_AUDIT-20261016.md) is a dated historical
receipt. Its paper checks apply to the unchanged 13-page PDF, not a new manuscript.

## Current model records

The four run metrics agree with the four HF metric files. All 41 shared-step means and
sample SD, per-seed ordering and first-observed token crossings were recalculated.
At 61,500 steps / 4,030,464,000 target tokens, Baseline is 1.005909 ± 0.000227 and
hybrid is 0.993838 ± 0.000111 BPB_est. Delta is −0.012071. Seed settings are
initialization 1/2, data 43/44 and dropout 124/125.

HF LFS metadata agrees with the four export SHA256/size records. Default aliases match
seed 2. This check does not download or reevaluate the full weights. There are no legacy
paths in the current HF tree. Original provenance records remain unchanged.

## Current kernel records

The new timing CSV and all four current raw archives agree with the author's evidence
bundle. Their protocol, results and numerical limits are mapped in
[KERNEL_EVIDENCE.md](KERNEL_EVIDENCE.md). Kernel revisions used by model training and
the timing experiment are distinct. Recorded OLD update failures remain visible.

## Remaining scope

- The [dated PDF](../paper/README.md) still needs its planned kernel-section revision.
- No fresh TPU timing, training, post-export model evaluation or Git-history secret scan
  was performed by this documentation review.
- Historical wrapper defects do not become current performance evidence. Use the frozen
  four-path notebook in atomic-ops for the current speed protocol.
- CPU/CI checks do not establish TPU timing or universal numerical correctness.

Public artifact locations and immutable experiment identities:
[`attestation/current-evidence.json`](../attestation/current-evidence.json).
