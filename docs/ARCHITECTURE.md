# Current architecture

Both models are decoder-only LMs, d_model=1024, 8 heads, tied GPT-2 embeddings, context 2048,
bf16 model projections and fp32 parameters/kernel inputs. Seven blocks each contain four
token mixers and one SwiGLU FFN. Baseline uses 4× GDN2; hybrid uses 3× GDN2 + 1× causal MLA.
MLA KV rank=512, non-RoPE/RoPE query-key dimensions=64/64 per head.

## Hierarchical delta routing

For an earlier source delta v_j, keys are RMS-normalized and scored by a learned query vector w_i:

    k_j = RMSNorm_key(v_j)
    alpha_j = softmax_j(w_i^T k_j)
    r_i = sum_j alpha_j v_j
    delta_i = F_i(Norm_pre(x_i + r_i))
    x_{i+1} = sanitize(x_i + delta_i)

Sources are previous-block summaries plus earlier mixer deltas inside this block. Sources are
depth states at the same token position, not a memory between separate examples. With no sources,
r_i=0. Queries start at zero, giving uniform weights over a nonempty source set, not zero retrieval.
The token mixers use pre-RMSNorm; the FFN uses pre-LayerNorm and its own DAR router.
The block summary is x_out−x_in and includes FFN. No separate old IntraBlockAttention is used.
The FFN residual update is not clipped at the same site as mixer updates; inspect `models/hybrid.py`
for the exact sanitization locations. This detail is not hidden by the schematic notation above.

## Parameters and compatibility

Baseline: 334,156,800. Hybrid: 307,131,392, about 8.09% fewer. The obsolete 342M hybrid
and latent-Q/K DAR checkpoints are different architectures and must not be resumed as current weights.
Keys in the safetensors release preserve Flax names and Dense kernel orientation.
The reference equations are adapted from Delta Attention Residuals; the block grouping/backbone is my adaptation.
