"""Publication replay: mean-pooled overview, exact retained Dense products.

Run after production_runtime.run in Colab. Keeps the full capture separately.
"""
import copy,base64,json,gzip,hashlib
from pathlib import Path
import numpy as np

def publication_capture(result,root):
    root=Path(root)
    pub={**result};groups=64
    def pool(rows):
        a=np.asarray(rows,np.float32)
        return a.reshape(*a.shape[:-1],groups,-1).mean(-1,dtype=np.float64).tolist()
    pub['stages']=[{**s,'values':pool(s['values'])} for s in result['stages']]
    pub['modules']=[{**m,**{k:pool(m[k]) for k in ('residual','normalized','output') if m[k]}} for m in result['modules']]
    pub['routes']=[{**r,'residual':pool(r['residual']),'retrieved':pool(r['retrieved'])} for r in result['routes']]
    pub['memories']=[]
    for g in result['memories']:
        def matrix_mean(x):
            a=np.asarray(x,np.float32);h,d,_=a.shape;s=d//16
            return a.reshape(h,16,s,16,s).mean((2,4),dtype=np.float64).tolist()
        pub['memories'].append({**g,'states':matrix_mean(g['states']),'updates':matrix_mean(g['updates']),
            'dim':16,'original_dim':g['dim'],'provenance':'Each displayed cell is the signed mean of an 8×8 region of the FP32 reference state'})
    pub['publication']=dict(overview_groups=groups,channels_per_group=result['channels']//groups,
        memory_grid=16,memory_pooling='signed mean of 8×8 original cells',
        projection_selection='24 largest absolute output activations, 8 largest absolute input contributions per output; final captured token',
        individual_projection_values_rounded=False,full_capture_sha256=hashlib.file_digest((root/'production-trace.json').open('rb'),'sha256').hexdigest())
    raw=json.dumps(pub,separators=(',',':'),ensure_ascii=False).encode()
    path=root/'publication-trace.json.gz';path.write_bytes(gzip.compress(raw,compresslevel=6))
    (root/'publication-trace.json').write_bytes(raw)
    report=dict(raw_bytes=len(raw),gzip_bytes=path.stat().st_size,groups=groups,actual_channels=result['channels'],
        dense_edges_retained=sum(len(p['edges']) for p in pub['projections']),dense_edges_total=sum(p['total_edges'] for p in pub['projections']),
        dar_source_pairs=sum(len(r['sources']) for r in pub['routes']),checkpoint_verified=pub['checks']['checkpoint_verified'])
    (root/'publication-report.json').write_text(json.dumps(report,indent=2))
    print('PUBLICATION_TRACE_READY',json.dumps(report),flush=True)
    return pub,report
