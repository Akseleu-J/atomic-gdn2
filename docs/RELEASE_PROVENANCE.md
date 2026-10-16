# Release provenance

Training commits remain immutable: Baseline bf80e315732837c4261f32faebbb052c51dee094;
hybrid 697a10d3cf06be5a5b9cafa38fe272629814e64f; kernel 0dbc59cf74883315d7adbebe9229c520663e0d58.
Per-run source manifests and hashes are preserved in the raw evidence ZIP.
This public repository is a new publication snapshot with updated documentation and paper assets.
The previous repository/history remains preserved privately under a dated archive name.

Historical kernel attestations contain older commit/content hashes, different shapes and incomplete raw
notebook provenance. Their claimed speedups are not silently transferred to the four new runs.
The safetensors export is params-only; model weight publication does not recreate optimizer state.
