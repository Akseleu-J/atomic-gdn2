# Safetensors checkpoints

Repository: https://huggingface.co/Akseleu/Atomic-GDN2

Four current checkpoints use folders `baseline/seed-1`, `baseline/seed-2`, `hybrid/seed-1`,
`hybrid/seed-2`. Each folder contains model.safetensors, model configuration, export manifest,
resolved training arguments and run evidence. See the HF model card for the published inventory.

`format=flax`: tensor keys are slash-separated original Flax paths, with Dense kernels stored
as [input, output], not transposed PyTorch matrices. Inference files contain parameters only.
They do not include the optimizer or create automatic compatibility with Transformers AutoModel.
Use `atomic_gdn2_model.weights.load_safetensors` and build_model with the matching ModelConfig.
The GPT-2 tokenizer is external; use `openai-community/gpt2`.

Default model aliases, where provided, refer to seed 2; they are not an average or merged checkpoint.
Check `export.json` for tensor shapes, finite-value checks, parameter count, exact roundtrip verification
and the file SHA256. The current HF tree contains only the four reported checkpoints and
their seed-2 default aliases. Superseded legacy paths were removed from main; previous HF
commits remain in ordinary revision history.

## Re-export an original Orbax checkpoint

The streaming exporter restores only parameter arrays, never optimizer state. Its slots refer to the
fixed published run inventory: 2/3 are Baseline seeds 1/2, 4/5 are hybrid seeds 1/2.

```bash
pip install tensorstore==0.1.85 safetensors==0.8.0 PyYAML==6.0.3
python scripts/export_checkpoint.py --slot 2 --checkpoint /absolute/path/to/checkpoints/65000/default
```

The output is written under `checkpoint-exports/`. It checks parameter count and finite values,
validates the official safetensors header/keys/shapes, and compares each tensor's original bytes with
its serialized bytes. The original Orbax checkpoint must be obtained separately; the public model
weights do not include optimizer state or reconstruct it. Set an `HF_EXPORT_TOKEN` environment
variable only when direct upload is intended; never commit credentials with export artifacts.

## Verified release inventory

All four were exported on Kaggle CPU and uploaded separately.

| Path | Bytes | SHA256 |
|---|---:|---|
| `baseline/seed-1/model.safetensors` | 1,336,666,296 | `2d1b79e067a3a4c9061337c171b8e97d44fbf7d5c1908ea3aa1cf329bcd48732` |
| `baseline/seed-2/model.safetensors` | 1,336,666,296 | `6984a9ea8eaef2b474eb3c9ca763b6ec18093fbb8376c8e6939f1807ce9ed02b` |
| `hybrid/seed-1/model.safetensors` | 1,228,565,080 | `f0c178b48424c208a499cc0e58912d985a0f3f1a1a39e9a02d5621054ddc463f` |
| `hybrid/seed-2/model.safetensors` | 1,228,565,080 | `4f44215338eaeab0fc043b9151ad7f32e798a0ae34ae83626b458c356e750ca8` |

HF release revision: `fb8fdf2e00b992f4041885d26b209fcedb5b5c0c`.

Temporary private env datasets were used only for the authorized export. They are not part of this repository or the weight files.

The current technical model card is also archived in [MODEL_CARD.md](MODEL_CARD.md).
