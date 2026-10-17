# Release provenance

Training commits remain immutable: Baseline bf80e315732837c4261f32faebbb052c51dee094;
hybrid 697a10d3cf06be5a5b9cafa38fe272629814e64f; kernel 0dbc59cf74883315d7adbebe9229c520663e0d58.
Per-run source manifests and hashes are preserved in the raw evidence ZIP.
This public repository is a new publication snapshot with updated documentation and paper assets.
The previous repository/history remains preserved privately under a dated archive name.

Historical kernel attestations contain older commit/content hashes, different shapes and incomplete raw
notebook provenance. Their claimed speedups are not silently transferred to the four new runs.
The safetensors export is params-only; model weight publication does not recreate optimizer state.

## Separate current speed experiment

The 17 October four-path benchmark pins atomic-ops commit
11616acd8a061768a38feb12c58e5e329952f5d5, uses zero damping and measures a synthetic
34,873,952-parameter model for its complete-step comparison. It does not replace the
training kernel pin above. See [kernel evidence](KERNEL_EVIDENCE.md).
Original private-development source payloads are publicly available inside the four-run ZIP;
the private historical repositories are not required to read that evidence.
