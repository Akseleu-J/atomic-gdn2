"""Original-layout Flax safetensors loader; no Transformers model conversion."""
from pathlib import Path

def load_safetensors(path: str | Path):
    from safetensors.numpy import load_file
    from flax.traverse_util import unflatten_dict
    arrays = load_file(str(path))
    return unflatten_dict({tuple(key.split('/')): value for key,value in arrays.items()})

def load_hub_checkpoint(repo_id='Akseleu/Atomic-GDN2', model='hybrid', seed=2, *, revision=None, token=None):
    import json
    from huggingface_hub import hf_hub_download
    from .config import ModelConfig
    from .models import build_model
    folder=f'{model}/seed-{seed}'
    config=hf_hub_download(repo_id, f'{folder}/config.json', revision=revision, token=token)
    weights=hf_hub_download(repo_id, f'{folder}/model.safetensors', revision=revision, token=token)
    raw=json.loads(Path(config).read_text())
    fields=dict(raw['model'])
    fields['layer_types_override']=tuple(fields.get('layer_types_override',()))
    cfg=ModelConfig(**fields)
    return build_model(raw['arch'],cfg,True),load_safetensors(weights),cfg
