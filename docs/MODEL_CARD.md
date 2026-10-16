---
license: apache-2.0
language:
- en
library_name: flax
pipeline_tag: text-generation
tags:
- jax
- flax
- pallas
- gated-deltanet
- research
- base-model
---

<div align="center">

# Atomic-GDN2

### Recurrent token mixing · Latent attention · Hierarchical delta routing

**307M hybrid · 334M Baseline · JAX / Pallas · TPU v5e**

[Technical Report](https://github.com/Akseleu-J/atomic-gdn2/blob/main/paper/atomic-gdn2.pdf) · [Code](https://github.com/Akseleu-J/atomic-gdn2) · [Kernel](https://github.com/Akseleu-J/atomic-ops) · [Run Evidence](https://github.com/Akseleu-J/atomic-gdn2/blob/main/attestation/training.json)

</div>

---

## 1. Overview

Atomic-GDN2 studies a language-model backbone that combines Gated DeltaNet-2 (GDN-2),
causal Multi-head Latent Attention (MLA), and hierarchical delta attention residuals (DAR).
This release provides **four checkpoints trained from scratch**: two seeds of the pure GDN-2
**Baseline** and two seeds of the **hybrid**.

At the last exactly shared evaluation, after **4.030B training target tokens**, hybrid reaches
**0.993838 estimated validation BPB**, compared with **1.005909** for Baseline, while using
approximately **8.09% fewer parameters**. The experiment compares complete architectures;
it does not isolate the effect of DAR.

### Release highlights

- **A sequential hybrid backbone.** Each of seven blocks contains three GDN-2 mixers,
  one MLA mixer and one SwiGLU FFN, with the ordinary residual stream retained.
- **Routing over depth changes.** Learned query vectors retrieve earlier local deltas
  and previous-block changes before the next mixer or FFN.
- **Two seeds per architecture.** hybrid has lower BPB in both seed pairs at every one
  of the 41 shared logged evaluation steps.
- **Traceable checkpoints.** FP32 safetensors preserve Flax tensor layout; export manifests,
  configs, source hashes and validation logs accompany each seed.

These are English base language models for research. They are not instruction-tuned assistants.

## 2. Model specification

| Specification | Baseline | hybrid |
|:---|:---:|:---:|
| Parameters | 334,156,800 | **307,131,392** |
| Block composition | 4 × GDN-2 + FFN | 3 × GDN-2 + MLA + FFN |
| Blocks / token mixers | 7 / 28 | 7 / 28 |
| GDN-2 / MLA mixers | 28 / 0 | 21 / 7 |
| SwiGLU FFNs | 7 | 7 |
| Hidden / FFN dimension | 1024 / 3584 | 1024 / 3584 |
| GDN-2 heads / head dimension | 8 / 128 | 8 / 128 |
| MLA KV rank | — | 512 |
| MLA non-RoPE / RoPE dimensions per head | — | 64 / 64 |
| Depth routing | Ordinary residuals | Hierarchical DAR + ordinary residuals |
| Training context | 2048 tokens | 2048 tokens |
| Vocabulary / tokenizer | 50,257 / GPT-2 | 50,257 / GPT-2 |
| Embeddings | Tied input/output | Tied input/output |
| Training projections / parameters | bf16 / fp32 | bf16 / fp32 |
| Released parameters | FP32 safetensors | FP32 safetensors |

### Hierarchical delta routing

For the earlier depth sources available before sublayer (i), routing is:

$$
a_{ij}=\operatorname{softmax}_{j\in\mathcal V_i}
\left(z_i^{\mathsf T}\operatorname{RMSNorm}_{\rm key}(v_j)\right),
\qquad r_i=\sum_{j\in\mathcal V_i}a_{ij}v_j.
$$

The next sublayer receives `Norm_pre(x_i + r_i)`; its output is added to the ordinary residual.
The sources combine **previous-block summaries** with **earlier within-block deltas**, at the same
token position. Each block summary includes the FFN change. No sublayer reads its own new delta.
Zero-initialized learned queries give uniform retrieval over a nonempty source set.

The scoring and additive retrieval follow [Delta Attention Residuals](https://arxiv.org/abs/2605.18855);
the source hierarchy and GDN-2/MLA integration are the adaptation studied here.
Mixer and FFN normalization/sanitization sites are specified in the
[architecture notes](https://github.com/Akseleu-J/atomic-gdn2/blob/main/docs/ARCHITECTURE.md).

<p align="center">
  <img src="https://huggingface.co/Akseleu/Atomic-GDN2/resolve/fb8fdf2e00b992f4041885d26b209fcedb5b5c0c/assets/architecture.png" alt="Sequential hybrid model and expanded block with additive depth routing" width="760" />
</p>

## 3. Evaluation results

### Exactly matched training tokens

**61,500 optimizer steps · 4,030,464,000 target tokens per run · two seeds per architecture**

| Metric | Baseline | hybrid | hybrid − Baseline |
|:---|---:|---:|---:|
| Validation BPB est, mean ± sample SD | 1.005909 ± 0.000227 | **0.993838 ± 0.000111** | **−0.012071** |

Sample SD uses `ddof=1` across two seeds; it is not a confidence interval.
Validation covers **4,390,912 target tokens** in 67 full batches. The metric is
`BPB_est = token cross-entropy / ln(2) / 4.2`: a fixed bytes-per-token estimate,
not exact byte-level BPB. No separate test-set or downstream result is reported.

![Validation learning curves and first-observed target-token requirements](https://huggingface.co/Akseleu/Atomic-GDN2/resolve/fb8fdf2e00b992f4041885d26b209fcedb5b5c0c/assets/learning-curves.png)

The learning-curve endpoint uses the selected near-step cutoff: Baseline **63,000**,
hybrid **62,673 / 62,683**. Its mean difference is **−0.011617 BPB**; this endpoint is
not exactly token matched. The common-step result above uses only observed, identical steps.

### Tokens and time to target

| Target BPB est ≤ | Baseline steps, mean ± SD | hybrid steps, mean ± SD | Fewer target tokens | Less notebook time |
|---:|---:|---:|---:|---:|
| 1.10 | 20,250 ± 1,061 | 16,500 ± 0 | **18.52%** | **13.48%** |
| 1.08 | 25,500 ± 0 | 21,000 ± 0 | **17.65%** | **13.55%** |
| 1.06 | 31,500 ± 0 | 27,000 ± 0 | **14.29%** | **10.38%** |
| 1.02 | 50,250 ± 1,061 | 42,000 ± 0 | **16.42%** | **12.51%** |

Each crossing is the first logged evaluation meeting the target, calculated separately per seed
and then averaged, without interpolation. Evaluation every 1,500 steps limits crossing resolution.
Notebook time includes setup, compilation, evaluation and checkpoint overhead; TPU queue time is excluded.

## 4. Checkpoint collection

| Checkpoint | Seed | Actual saved step | Training target tokens | Final validation BPB est |
|:---|---:|---:|---:|---:|
| [Baseline seed 1](https://huggingface.co/Akseleu/Atomic-GDN2/tree/main/baseline/seed-1) | 1 | 65,000 | 4,259,840,000 | 1.004101 |
| [Baseline seed 2](https://huggingface.co/Akseleu/Atomic-GDN2/tree/main/baseline/seed-2) | 2 | 65,000 | 4,259,840,000 | 1.003825 |
| [hybrid seed 1](https://huggingface.co/Akseleu/Atomic-GDN2/tree/main/hybrid/seed-1) | 1 | 62,673 | 4,107,337,728 | 0.993435 |
| [hybrid seed 2](https://huggingface.co/Akseleu/Atomic-GDN2/tree/main/hybrid/seed-2) | 2 | 62,683 | 4,107,993,088 | 0.993187 |

These are final saved checkpoints, whereas the main quality comparison uses the common step 61,500.
The default paths `baseline/model.safetensors` and `hybrid/model.safetensors` point to **seed 2**.
They are not merged or averaged models. This repository contains only this current experiment.

Each seed directory contains:

```text
model.safetensors       FP32 parameter tensors
config.json             model configuration and tensor-layout metadata
training_args.json      resolved run configuration
export.json             tensor shapes/counts, source step and SHA256
metrics.jsonl           recorded training and validation measurements
SUCCESS.json            training completion record
provenance.json         source/data provenance
runtime-preflight.json  original execution checks
```

All exported parameter values were finite; tensor bytes matched their original checkpoint values
after serialization. The official safetensors reader checked keys/shapes, and remote LFS SHA256
matched all four export manifests. A full model evaluation was not rerun after conversion.
Optimizer state is excluded; training resume requires the original Orbax checkpoint.

## 5. Load the weights

The files retain **original Flax parameter layout**: `/`-separated paths and Dense kernels in
`[input, output]` orientation. Use the matching package loader and configuration.
Transformers `AutoModelForCausalLM` loading is not implemented.

```bash
pip install "git+https://github.com/Akseleu-J/atomic-gdn2" safetensors huggingface_hub
```

```python
from atomic_gdn2_model.weights import load_hub_checkpoint

model, params, cfg = load_hub_checkpoint(
    repo_id="Akseleu/Atomic-GDN2",
    model="hybrid",  # or "baseline"
    seed=2,          # or 1
)
```

The model uses JAX/Flax and the external `atomic_gdn2` Pallas kernel. A compatible TPU environment
is recommended; CPU interpret execution is a slow debugging path. Input lengths must be padded
to a multiple of the kernel chunk size **128**. Use the external
[`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) tokenizer.
This loader provides the model, parameters and config; automatic chat/generation integration
and generation-throughput measurements are not part of the release.

## 6. Training and reproducibility

| Setting | Reported protocol |
|:---|:---|
| Corpus | `windmaple/openwebtext-gpt2` |
| Initialization / data seeds | 1 and 2 for each architecture |
| Global batch / sequence length | 32 / 2048 |
| Hardware per run | 8 TPU v5e devices |
| Notebook time limit | 8 h 54 min, including setup/compile and finalization |
| Optimizers | Muon / AdamW |
| Peak learning rate / weight decay | 3e-4 / 0.1 |
| Gradient clip | 1.0 |
| Schedule | 100-update warmup; cosine decay based on elapsed time |
| Resume | None; four fresh runs |
| Available train tokens | 9,035,582,489; pass interrupted by the time budget |

<details>
<summary><strong>Immutable training sources and detailed artifacts</strong></summary>

- Baseline: `bf80e315732837c4261f32faebbb052c51dee094`.
- hybrid: `697a10d3cf06be5a5b9cafa38fe272629814e64f`.
- Common kernel: `0dbc59cf74883315d7adbebe9229c520663e0d58`.
- [Resolved configurations and run logs](https://github.com/Akseleu-J/atomic-gdn2/tree/main/artifacts/runs).
- [Complete four-run evidence ZIP](https://github.com/Akseleu-J/atomic-gdn2/blob/main/artifacts/four-runs-20261016.zip).
- [Full learning curves, tables and figures](https://github.com/Akseleu-J/atomic-gdn2/tree/main/results).
- [Reproduction protocol](https://github.com/Akseleu-J/atomic-gdn2/blob/main/docs/REPRODUCING.md).

The publication repository is a release snapshot. Per-run manifests identify the actual training sources.

</details>

## 7. Scope and limitations

- **Attribution:** MLA, routing, normalization and parameter counts differ. No ablation isolates DAR,
  and matched tokens do not imply matched FLOPs. Both architectures use the same kernel.
- **Replication:** two seeds and one corpus characterize this experiment, not general model capability.
  Context beyond 2048, downstream tasks and an independent final test were not evaluated.
- **Schedule:** elapsed-time LR yields different LR histories at equal steps. Threshold estimates
  are limited by the logging grid. The near-step endpoint was selected after the runs.
- **Numerics:** clipping, damping and sanitization are used; their activation rates and separate
  quality effects were not measured. Serialization checks do not establish full backward equivalence.
- **Use:** research base models, without instruction or safety tuning. Outputs may be inaccurate,
  biased or inappropriate. No production-readiness or historical kernel-ratio-to-model-speedup claim is made.

See the [technical report](https://github.com/Akseleu-J/atomic-gdn2/blob/main/paper/atomic-gdn2.pdf)
and [detailed limitations](https://github.com/Akseleu-J/atomic-gdn2/blob/main/docs/LIMITATIONS.md).

## 8. License and citation

Model code and released weights: **Apache-2.0**. Dataset and tokenizer terms remain with their sources.

```bibtex
@misc{omirbay2026atomicgdn2,
  author = {Akseleu Omirbay},
  title = {Atomic-GDN2: TPU Kernels and Hierarchical Delta Routing for Hybrid Language Models},
  year = {2026},
  howpublished = {Technical report and research artifacts},
  url = {https://github.com/Akseleu-J/atomic-gdn2}
}
```
