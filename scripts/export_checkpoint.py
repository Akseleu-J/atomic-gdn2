"""Lossless params-only export of published Orbax OCDBT, no optimizer deserialization."""
import argparse,ctypes,hashlib,json,os,sys,struct,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--slot',type=int,choices=[2,3,4,5],required=True);p.add_argument('--remote',action='store_true');p.add_argument('--checkpoint',type=Path);args=p.parse_args()
import numpy as np
os.environ['TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS']='600'
import tensorstore as ts
from safetensors.numpy import save_file,load_file
from safetensors import safe_open
results=json.loads((ROOT/'attestation/checkpoint-sources.json').read_text())
run=next(r for r in results['runs'] if r['slot']==args.slot)
name=('baseline' if args.slot<4 else 'hybrid_darfix')+f'_seed{run["seed"]}_b32'
base=ROOT/'artifacts/runs'/name
cp=base/'checkpoints'/str(run['checkpoint_step'])/'default'
spec={'driver':'ocdbt','base':{'driver':'file','path':cp.as_posix()+'/'}}
if args.remote:
    from checkpoint_http import open_remote
    spec,server=open_remote(args.slot,run,name)
elif args.checkpoint:
    cp=args.checkpoint.resolve()
    assert (cp/'manifest.ocdbt').is_file(), 'Pass the Orbax default directory containing manifest.ocdbt'
    spec={'driver':'ocdbt','base':{'driver':'file','path':cp.as_posix()+'/'}}
else:
    raise ValueError('Use --checkpoint /path/to/step/default or --remote with ephemeral CHECKPOINT_URLS')
kv=ts.KvStore.open(spec).result();keys=[k.decode() for k in kv.list().result()]
paths=sorted(k[:-len('/.zarray')] for k in keys if k.startswith('params.') and k.endswith('/.zarray'))
assert paths
context=ts.Context({'cache_pool':{'total_bytes_limit':32*1024*1024},'http_request_concurrency':{'limit':24},'data_copy_concurrency':{'limit':2}})
model='baseline' if args.slot<4 else 'hybrid'
out=ROOT/'checkpoint-exports'/model/f'seed-{run["seed"]}'
out.mkdir(parents=True,exist_ok=True)
file=out/'model.safetensors'
manifest=[];header={};offset=0;stores=[]
for path in paths:
    store=ts.open({'driver':'zarr','kvstore':spec,'path':path},open=True,context=context).result()
    key='/'.join(path.split('.')[1:]);elements=int(np.prod(store.shape));nbytes=elements*4
    assert store.dtype==ts.float32,str(store.dtype)
    header[key]=dict(dtype='F32',shape=list(store.shape),data_offsets=[offset,offset+nbytes])
    offset+=nbytes;stores.append((key,store))
assert offset//4==run['parameters']
header['__metadata__']={'format':'flax','architecture':model,'seed':str(run['seed']),
    'step':str(run['checkpoint_step']),'source_commit':run['commit'],'kernel_commit':'0dbc59cf74883315d7adbebe9229c520663e0d58'}
encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
data_start=8+len(encoded)
stream=file.open('wb');stream.write(struct.pack('<Q',len(encoded)));stream.write(encoded)
pending={j:stores[j][1].read() for j in range(min(4,len(stores)))}
for i,(key,store) in enumerate(stores):
    array=np.asarray(pending.pop(i).result())
    assert np.isfinite(array).all(),key
    array=np.ascontiguousarray(array);raw=memoryview(array).cast('B')
    digest=hashlib.sha256()
    for start in range(0,len(raw),8*1024*1024):
        chunk=raw[start:start+8*1024*1024];stream.write(chunk);digest.update(chunk)
    manifest.append(dict(key=key,shape=list(array.shape),dtype=str(array.dtype),elements=int(array.size),
        sha256=digest.hexdigest()))
    del raw,array;gc.collect()
    if i+4<len(stores):pending[i+4]=stores[i+4][1].read()
    if i%10==0:print(f'params {i+1}/{len(paths)}',flush=True)
stream.close()
# Validate with the official reader and hash every stored tensor through its data offsets.
with safe_open(file,framework='np') as saved:
    assert set(saved.keys())=={m['key'] for m in manifest}
    for m in manifest:assert saved.get_slice(m['key']).get_shape()==m['shape']
with file.open('rb') as f:
    for m in manifest:
        begin,end=header[m['key']]['data_offsets'];f.seek(data_start+begin);remaining=end-begin;d=hashlib.sha256()
        while remaining:
            chunk=f.read(min(8*1024*1024,remaining));assert chunk;d.update(chunk);remaining-=len(chunk)
        assert d.hexdigest()==m['sha256'],m['key']
digest=hashlib.sha256()
with file.open('rb') as stream:
    while chunk:=stream.read(8*1024*1024):digest.update(chunk)
proof=dict(model=model,seed=run['seed'],step=run['checkpoint_step'],parameters=run['parameters'],
    tensor_count=len(manifest),dtype='float32',source_commit=run['commit'],kernel_commit='0dbc59cf74883315d7adbebe9229c520663e0d58',
    file_sha256=digest.hexdigest(),bytes=file.stat().st_size,
    params_only=True,optimizer_state_included=False,all_finite=True,roundtrip_bitwise_equal=True,tensors=manifest,
    validation=dict(loss=run['val_loss'],bpb_est=run['val_bpb_est'],tokens_seen=run['tokens']))
(out/'export.json').write_text(json.dumps(proof,indent=2))
cfg_path=base/'config.yaml'
import yaml
cfg=yaml.safe_load(cfg_path.read_text())
(out/'config.json').write_text(json.dumps(dict(arch=model,model=cfg['model'],format='flax-safetensors',
    tensor_key_separator='/',tokenizer='openai-community/gpt2',transformers_compatible=False),indent=2))
(out/'training_args.json').write_text(json.dumps(cfg,indent=2))
for n in ('SUCCESS.json','provenance.json','runtime-preflight.json','metrics.jsonl'):
    import shutil;shutil.copy2(base/n,out/n)
(ROOT/'checkpoint-exports'/f'safetensors-export-{args.slot}.json').write_text(json.dumps({k:v for k,v in proof.items() if k!='tensors'},indent=2))
print(json.dumps({k:v for k,v in proof.items() if k!='tensors'},indent=2))

if os.environ.get('HF_EXPORT_TOKEN'):
    import time
    from huggingface_hub import HfApi
    api=HfApi(token=os.environ['HF_EXPORT_TOKEN'])
    repo='Akseleu/Atomic-GDN2'
    for attempt in range(5):
        try:
            commit=api.upload_folder(repo_id=repo,folder_path=out,path_in_repo=f'{model}/seed-{run["seed"]}',
                commit_message=f'Verified lossless CPU export: {model} seed {run["seed"]}, step {run["checkpoint_step"]}')
            break
        except Exception:
            if attempt==4:raise
            time.sleep(10*(attempt+1))
    info=api.model_info(repo,files_metadata=True)
    item=next(s for s in info.siblings if s.rfilename==f'{model}/seed-{run["seed"]}/model.safetensors')
    assert item.lfs.sha256==proof['file_sha256']
    print('Uploaded and remote SHA256 verified',flush=True)
