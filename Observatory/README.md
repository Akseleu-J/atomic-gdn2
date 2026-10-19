# Atomic-GDN2 Observatory

**[Open the interactive 3D Observatory](https://huggingface.co/spaces/Akseleu/Atomic-GDN2-Observatory)** · [Capture a new prompt in Colab](https://colab.research.google.com/github/Akseleu-J/atomic-gdn2/blob/main/Observatory/notebooks/Atomic-Production-Capture.ipynb) · [Model weights](https://huggingface.co/Akseleu/Atomic-GDN2)

[![Atomic-GDN2 Observatory](assets/preview.png)](https://huggingface.co/spaces/Akseleu/Atomic-GDN2-Observatory)

The default **40k+** view contains **40,235 graph links** and **15,119 displayed nodes** at 32 groups / full density. The top toggle restores the lighter standard view (24,233 links). The official interface uses restrained translucent instrument panels, compact navigation and a token dock. The signal palette remains signed and magnitude-aware.

## Interface release · 2026-10-19

- Official Observatory branding, translucent side panels with close controls, grouped settings and responsive layouts.
- Curved strands with fixed endpoints, subtle sway and direction particles. Hover highlights incident links in white; a click pins the selection across tokens.
- Node brightness follows activation magnitude with a fixed scale within each bank across the captured token sequence. Structural junctions are visually distinct.
- Local/layer routes remain bright; lower block-delta routes are deliberately subdued. Settings expose density, glow, block-route light, sway and signal speed.
- The operating system's reduced-motion preference pauses illustrative motion on startup; users can re-enable it. Capture values and graph identities are unchanged.
- Renderer sources and build script are in `source/`. The Colab notebook embeds the same interface.

Motion is a visual aid, not measured neural or hardware timing. The baseline remains available in repository history before this release.

## What the data means

- Published **hybrid seed 2**, step **62,683**, **307,131,392 parameters**, 7 blocks, 21 GDN2 mixers, 7 MLA mixers, 7 dense SwiGLU FFNs and tied GPT-2 embeddings/head. Eight GPT-2 input tokens: `To be, or not to be.`
- Checkpoint revision: `efb564b842b8c001c91f222ebf453716006b73c4`. Weight SHA256: `4f44215338eaeab0fc043b9151ad7f32e798a0ae34ae83626b458c356e750ca8`.
- The data is a **PyTorch FP32 serial reference forward using the published trained weights**. It is not a capture of production TPU bf16/fused internal states and does not prove parity with that execution.
- Overview banks show **signed means of contiguous channel groups**, not individual neurons. The full model has 1024 channels; the overview can show 16, 32 or 64 groups.
- All **175 H-DAR vector-source/target pairs** are retained. Violet: preceding local layer deltas. Gold: all earlier block deltas. The residual stream remains separate. Block delta includes the FFN and both dependencies `block_end - block_start`.
- The default 40k+ overview retains **17,088 individual Dense products across 89 input projections**. Projection detail can inspect **40,320 saved products across all 210 projections**. For each projection, the capture stores 24 largest-absolute output activations and 8 largest-absolute input products per selected output. This is a selection, not the full Dense graph or a causal explanation.
- **Individual Dense products and GDN memory are captured for token 8 only.** Token playback changes residuals, deltas, routing and attention; the individual Dense sample remains fixed to token 8. The interface labels this scope.
- Blue operation/membership lines indicate **structural dependencies**, not measured scalar attributions through nonlinearities. Structural operation junctions have no scalar activation. The 40,235-link total includes these structural lines, residual/group lines, DAR contributions and selected Dense products.
- Compact GDN memory cells are signed means of 8x8 original state regions (128x128 becomes 16x16 per head). The full reference state is available when generating a full capture in Colab.
- Animated particle motion illustrates contribution direction; it is not measured processor execution time. This visualization does not establish language-model quality or DAR's isolated quality effect.

## Verification

`verification/graph-audit.json` checks the exported arithmetic and rendered topology for all 8 tokens, 16/32/64 groups and both overview modes: finite data, legal/source-ordered DAR routes, all endpoints, no duplicate graph edges, real saved Dense products, and both terms of block-delta subtraction. DAR reconstruction error on the grouped trace is 1.26e-8; selected product FP32/JSON rounding error is 1.04e-7.

`verification/source-audit.json` matches the four model source files against the checkpoint training provenance, allowing Windows CRLF versus Git LF line endings. `verification/independent-audit.json` records the separate NumPy/FP32 replay and comparisons against the weight file and capture. The historical tiny original-Flax comparison is also retained: maximum stage difference 0.001498; it is not full-model TPU parity evidence.

## Explore and reproduce

Left drag: orbit; right/middle drag or Shift+drag: pan. Wheel: zoom. WASD + Q/E: fly. Shift accelerates. Hover a node for its white incident-link fan; click to pin selection, Esc to release; double click or F focuses it. Home resets. Use the top **40k+** toggle to compare modes, and settings for DAR, GDN state, MLA attention and projection details. Lower density or choose the standard mode on slower GPUs.

The hosted site replays the frozen checkpoint trace. **New prompts run through the public Colab notebook**, with a CPU reference capture bounded to the first 8 GPT-2 tokens. No API token or private Drive notebook is needed for the public replay. Public weights are fetched without embedded credentials.

Serve locally:

```sh
python -m http.server 8767 --directory Observatory/web
```

Open `http://127.0.0.1:8767`. For a new capture install `torch safetensors huggingface_hub transformers numpy`, or run the notebook. To audit a downloaded copy of the pinned safetensors file:

```sh
python Observatory/verification/audit_checkpoint.py --weights model.safetensors --trace Observatory/web/trace.json --config Observatory/capture/config.json --report independent-audit.json
node Observatory/verification/validate-graph.cjs
```

`MANIFEST.json` records exact publication-file hashes. Rendering speed is device/load dependent; no performance improvement is claimed for this interface release.

Apache-2.0; see LICENSE. Tokenizer/data rights remain with their respective sources.
