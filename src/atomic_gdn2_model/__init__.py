"""atomic-gdn2: GDN-2 baseline vs hybrid GDN-2/MLA + hierarchical block-delta DAR language models (JAX/TPU)."""
__version__ = "0.1.0"

from .config import DataConfig, ExperimentConfig, ModelConfig, RuntimeConfig, TrainConfig

__all__ = ["ExperimentConfig", "ModelConfig", "DataConfig", "TrainConfig", "RuntimeConfig", "__version__"]
