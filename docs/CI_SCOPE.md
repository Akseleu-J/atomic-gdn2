# CPU workflow scope

`Model CPU checks` runs Python 3.12/3.13 with JAX/JAXlib 0.11.2 and the model's
original pinned external kernel. It checks configuration, data loading, optimizer,
deadline handling and the tiny mocked-model timed loop.

Tests marked `kernel` are explicitly excluded from this CPU unit workflow. This is
not a TPU training, checkpoint-quality, generation-throughput or full-model Pallas
validation run. The four recorded training runs and exported weight evidence remain
separate artifacts. Dependency consistency is checked with `pip check`.
