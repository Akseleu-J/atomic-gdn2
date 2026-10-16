# Final article review — 2026-10-16

The final paper has **11 pages**, including references and appendices.

## Five completed changes

1. Algorithm 1 specifies the 13-step block forward path: retrieval from earlier sources,
   mixer-output sanitization, residual sanitization, FFN update, and FFN-inclusive block summary.
2. Table 2 compares the original DAR formulation with this adaptation. Original sublayer/block
   grouping is credited to DAR; the hybrid hierarchy, embedding-source omission, backbone and
   sanitization differences are stated explicitly.
3. The notation x_(b,i), delta_(b,i), s_b, r_(b,i) is shared by equations, algorithm and vector diagram.
4. The zero-query observation derives uniform retrieval and a norm bound. It does not assert a
   zero update or identity-preserving conversion of a pretrained model.
5. The method now precedes implementation and experiments: idea, notation, equations, source
   hierarchy, comparison, algorithm, diagram, execution details, measurements and discussion.

## Source and result audit

- PaperDAR, PaperLayer and PaperBlock in `models/hybrid.py` specify the routing and update order.
  Sanitized local mixer outputs are distinguished from actual residual changes when clipping acts.
- `models/gdn2.py` supplies the unconstrained linear write projection, sigmoid erase projection,
  clipped negative-softplus decay and Q/K normalization. The kernel recurrence and numerical
  settings retain the scope documented in KERNEL_EVIDENCE.md.
- DAR source: https://arxiv.org/html/2605.18855v1 , Section 2.3 and Figure 3. Bibliographic
  authors/titles/years were checked against official arXiv, JAX and Muon sources.
- All 41 shared-step means/sample SD and source metric hashes were recomputed. Both seed pairs
  independently favor hybrid at all 41 shared evaluations. No training or post-export model
  evaluation was rerun for this editorial revision.
- All four export manifests agree with actual source commit, step and parameter count. The
  exact token match is 61,500 × 32 × 2048 = 4,030,464,000 target tokens. The separate near-step
  cutoff remains clearly labelled. BPB is a fixed bytes-per-token estimate; n=2 SD is not a CI.

## Links and layout

All **13 unique external PDF URLs** returned a successful HTTP response. All **25 internal PDF
link targets** resolved. The six bibliography entries are all cited; there are no missing or
uncited entries. Response checks show availability, not a guarantee that remote pages never change.
The complete URL-level receipt is `attestation/article-audit.json`.

Every page was rendered and reviewed via the contact sheet; method equations, comparison table,
algorithm, architecture and bibliography were inspected at larger size. No clipped or overlapping
elements, overfull boxes, undefined references, unresolved citation markers or near-empty orphan
pages remain. One nonfatal underfull-hbox warning remains in the bibliography. Main/threshold
result tables and numerical curve assets were not changed.

The original training source snapshots and raw logs remain available in the four-run evidence ZIP.
The PDF contains selected measurements and diagnostic figures, not raw log dumps. Limitations retain
the unequal parameters/FLOPs, time-based LR, single corpus, two seeds, no isolated DAR ablation,
no independent test/downstream evaluation, and unmeasured clipping frequency.
