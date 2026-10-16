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
and the file SHA256. Legacy checkpoints are preserved under a separately labelled legacy path/history.
