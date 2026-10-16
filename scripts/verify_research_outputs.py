"""Verify measured means/SD from the public source metrics, without running training."""
import json,statistics,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
att=json.loads((ROOT/'attestation/training.json').read_text())
curves={}
for run in att['runs']:
    for item in run['artifacts']:
        name=('baseline' if item['model']=='Baseline' else 'hybrid_darfix')+f"_seed{item['seed']}_b32"
        path=ROOT/'artifacts/runs'/name/'metrics.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        curves[item['model'],item['seed']]=[json.loads(s) for s in path.read_text().splitlines()]
for row in att['per_step_evolution']:
    for model,key in [('Baseline','baseline'),('hybrid','hybrid')]:
        values=[next(x['val_bpb_est'] for x in curves[model,s] if x['kind']=='eval' and x['step']==row['step']) for s in (1,2)]
        assert math.isclose(statistics.mean(values),row[key]['mean'],abs_tol=1e-12)
        assert math.isclose(statistics.stdev(values),row[key]['std'],abs_tol=1e-12)
print('Verified all shared-step means/SD and source metric hashes')
