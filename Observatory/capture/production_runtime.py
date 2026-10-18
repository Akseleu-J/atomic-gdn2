"""Bounded FP32 reference capture using published Flax safetensors.

Uses the trained parameters and project hierarchy. Recurrence is serial, all
normalizations and projections FP32. This is not TPU bf16/fused parity evidence.
"""
import os,json,time,hashlib,gzip,math
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open

REVISION='efb564b842b8c001c91f222ebf453716006b73c4'
REPO='Akseleu/Atomic-GDN2'
PREFIX='hybrid/seed-2'

def download(root):
    from huggingface_hub import hf_hub_download
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    cfg_path=hf_hub_download(REPO,PREFIX+'/config.json',revision=REVISION)
    proof_path=hf_hub_download(REPO,PREFIX+'/export.json',revision=REVISION)
    path=Path(hf_hub_download(REPO,PREFIX+'/model.safetensors',revision=REVISION))
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    proof=json.loads(Path(proof_path).read_text());assert digest==proof['file_sha256']
    cfg=json.loads(Path(cfg_path).read_text())['model']
    receipt=dict(repo=REPO,revision=REVISION,path=str(path),sha256=digest,parameters=proof['parameters'],step=proof['step'],verified=True)
    (root/'checkpoint-receipt.json').write_text(json.dumps(receipt,indent=2))
    print('CHECKPOINT_VERIFIED',json.dumps({k:v for k,v in receipt.items() if k!='path'}),flush=True)
    return path,cfg,receipt

class Capture:
    def __init__(self,path,cfg,receipt=None):
        self.file=safe_open(str(path),framework='pt',device='cpu');self.cfg=cfg
        self.receipt=receipt or {};self.device='cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_num_threads(2)
        self.eps=1e-6;self.collect=True
    def w(self,name):return self.file.get_tensor(name).to(self.device)
    def arr(self,x):return x.detach().float().cpu().tolist()
    def norm(self,x,prefix,rms=False):
        v=x.square().mean(-1,keepdim=True)
        if rms:return x*torch.rsqrt(v+self.eps)*self.w(prefix+'/scale')
        mean=x.mean(-1,keepdim=True);var=torch.clamp(v-mean.square(),min=0)
        return (x-mean)*torch.rsqrt(var+self.eps)*self.w(prefix+'/scale')+self.w(prefix+'/bias')
    def dense(self,x,path):
        w=self.w(path+'/kernel');y=x@w
        if path+'/bias' in self.file.keys():y=y+self.w(path+'/bias')
        if self.collect:
            # The detail frame retains individually indexed, exact products.
            a=x[-1];b=y[-1];out=torch.topk(b.abs(),min(24,len(b))).indices
            products=a[:,None]*w[:,out];src=torch.topk(products.abs(),min(8,len(a)),dim=0).indices
            rows=[]
            for k,j in enumerate(out.tolist()):
                for i in src[:,k].tolist():rows.append([i,j,float(a[i]),float(w[i,j]),float(products[i,k])])
            err=float((x[-1]@w-y[-1]+(self.w(path+'/bias') if path+'/bias' in self.file.keys() else 0)).abs().max())
            self.projections.append(dict(id=path,input_dim=x.shape[-1],output_dim=y.shape[-1],
                token=self.T-1,input=self.arr(a),output=self.arr(b),edges=rows,total_edges=w.numel(),max_error=err))
        return y
    def dar(self,x,sources,ids,prefix,target):
        if not sources:return x
        values=torch.stack(sources);keys=self.norm(values,prefix+'/key_norm',True)
        scores=torch.einsum('d,std->st',self.w(prefix+'/query'),keys)
        weights=torch.softmax(scores,dim=0);selected=torch.einsum('st,std->td',weights,values)
        assert torch.allclose(weights.sum(0),torch.ones(self.T,device=self.device),atol=1e-6)
        if self.collect:self.routes.append(dict(target=target,sources=ids,weights=self.arr(weights.T),scores=self.arr(scores.T),residual=self.arr(x),retrieved=self.arr(selected)))
        return x+selected
    def rope(self,x):
        d=x.shape[-1];inv=1/(self.cfg['rope_theta']**(torch.arange(0,d,2,device=self.device).float()/d))
        f=torch.arange(self.T,device=self.device).float()[:,None]*inv[None];angles=torch.cat([f,f],-1)[:,None,:]
        return x*angles.cos()+torch.cat([-x[...,d//2:],x[...,:d//2]],-1)*angles.sin()
    def gdn(self,x,path):
        H=self.cfg['n_heads'];D=self.cfg['d_model']//H
        q,k,v,w,b,g=[self.dense(x,path+'/Dense_'+str(i)).reshape(self.T,H,D) for i in range(6)]
        if self.cfg['l2norm_qk']:
            q=q*torch.rsqrt(q.square().sum(-1,keepdim=True)+self.eps)
            k=k*torch.rsqrt(k.square().sum(-1,keepdim=True)+self.eps)
        b=b.sigmoid();g=-torch.minimum(F.softplus(g),torch.tensor(160/self.cfg['gdn2_bs2'],device=self.device))
        state=torch.zeros(H,D,D,device=self.device);outputs=[];energies=[];last_update=None
        for t in range(self.T):
            decayed=state*g[t].exp()[:,:,None]
            erase=torch.einsum('hd,hdv->hv',b[t]*k[t],decayed)
            write=k[t,:,:,None]*(w[t]*v[t]-erase)[:,None,:]
            state=decayed+write;last_update=write
            outputs.append(torch.einsum('hd,hdv->hv',q[t]/math.sqrt(D),state))
            energies.append(state.square().mean((-1,-2)).sqrt())
        if self.collect:self.memories.append(dict(id=path,heads=H,dim=D,token=self.T-1,
            states=self.arr(state),updates=self.arr(last_update),rms=self.arr(torch.stack(energies)),
            provenance='FP32 serial reference state; full production bf16/fused parity not evaluated'))
        return self.dense(torch.stack(outputs).reshape(self.T,-1),path+'/Dense_6')
    def mla(self,x,path):
        H=self.cfg['n_heads'];dn=self.cfg['qk_nope_head_dim'];dr=self.cfg['qk_rope_head_dim'];D=dn+dr
        assert self.cfg.get('q_lora_rank') is None
        q=self.dense(x,path+'/Dense_0').reshape(self.T,H,D)
        latent=self.norm(self.dense(x,path+'/Dense_1'),path+'/LayerNorm_0')
        kn=self.dense(latent,path+'/Dense_2').reshape(self.T,H,dn)
        v=self.dense(latent,path+'/Dense_3').reshape(self.T,H,D)
        kr=self.rope(self.dense(x,path+'/Dense_4').reshape(self.T,1,dr)).expand(-1,H,-1)
        q=torch.cat([q[...,:dn],self.rope(q[...,dn:])],-1);k=torch.cat([kn,kr],-1)
        score=torch.einsum('thd,shd->hts',q,k)/math.sqrt(D)
        mask=torch.ones(self.T,self.T,device=self.device).tril().bool()
        a=torch.softmax(score.masked_fill(~mask,-1e9),-1)
        out=torch.einsum('hts,shd->thd',a,v)
        if self.collect:self.attention.append(dict(id=path,heads=H,attention=self.arr(a),latent=self.arr(latent)))
        return self.dense(out.reshape(self.T,-1),path+'/Dense_5')
    def ffn(self,x,path):
        g=self.dense(x,path+'/Dense_0');u=self.dense(x,path+'/Dense_1')
        return self.dense(F.silu(g)*u,path+'/Dense_2')
    def stage(self,id,x,label,block):
        if self.collect:self.stages.append(dict(id=id,values=self.arr(x),label=label,block=block))
    @torch.inference_mode()
    def capture(self,ids,labels=None,collect=True):
        started=time.perf_counter();self.collect=collect;self.T=len(ids)
        self.stages=[];self.routes=[];self.modules=[];self.projections=[];self.memories=[];self.attention=[]
        ids=torch.tensor(ids,device=self.device);embedding=self.w('Embed_0/embedding');x=embedding[ids]
        self.stage('embedding',x,'Embedding',-1);history=[];histids=[]
        c=self.cfg;pattern=c.get('layer_types_override') or (['gdn2']*(c['layers_per_block']-1)+['mla'])*c['num_blocks']
        for b in range(c['num_blocks']):
            start=x;local=[];localids=[]
            for l in range(c['layers_per_block']):
                sid=f'block_{b}_layer_{l}';prefix=f'block_{b}/layer_{l}';kind=pattern[b*c['layers_per_block']+l]
                routed=self.dar(x,history+local,histids+localids,prefix+'/dar',sid)
                norm=self.norm(routed,prefix+'/pre_sublayer_norm',True)
                delta=torch.nan_to_num(torch.clamp((self.gdn if kind=='gdn2' else self.mla)(norm,prefix+'/sublayer'),-1000,1000))
                out=torch.nan_to_num(torch.clamp(x+delta,-1000,1000))
                if collect:self.modules.append(dict(id=prefix+'/sublayer',stage=sid,kind=kind,block=b,label=f'B{b+1} · {kind.upper()} {l+1}',residual=self.arr(x),normalized=self.arr(norm),output=self.arr(out)))
                self.stage(sid,delta,f'B{b+1} Δ {kind} {l+1}',b)
                local.append(delta);localids.append(sid);x=out
            sid=f'block_{b}_ffn';routed=self.dar(x,history+local,histids+localids,f'block_{b}/ffn_dar',sid)
            norm=self.norm(routed,f'block_{b}/LayerNorm_0');delta=self.ffn(norm,f'block_{b}/ffn');out=x+delta
            if collect:self.modules.append(dict(id=f'block_{b}/ffn',stage=sid,kind='ffn',block=b,label=f'B{b+1} · SwiGLU',residual=self.arr(x),normalized=self.arr(norm),output=self.arr(out)))
            self.stage(sid,delta,f'B{b+1} Δ FFN',b);x=out;summary=x-start
            sid=f'block_{b}_summary';history.append(summary);histids.append(sid);self.stage(sid,summary,f'Δ BLOCK {b+1}',b)
            print('CAPTURE_BLOCK',b+1,'/',c['num_blocks'],flush=True)
        norm=self.norm(x,'LayerNorm_0');logits=norm@embedding.T
        if not collect:return logits.detach().cpu()
        self.modules.append(dict(id='lm_head',stage='logits',kind='head',block=c['num_blocks'],label='LM head · GPT-2',residual=self.arr(x),normalized=self.arr(norm),output=[]))
        probabilities=logits.softmax(-1);p,i=probabilities.topk(5,-1)
        top=[[dict(id=int(j),label=str(int(j)),prob=float(v)) for j,v in zip(rowi,rowp)] for rowi,rowp in zip(i,p)]
        result=dict(version=4,text=' '.join(labels or map(str,ids.tolist())),tokens=[dict(id=int(i),label=l) for i,l in zip(ids,labels or map(str,ids.tolist()))],channels=c['d_model'],metadata=dict(config=c,checkpoint=self.receipt,model='published hybrid seed-2 · FP32 reference'),
            modules=self.modules,stages=self.stages,routes=self.routes,projections=self.projections,memories=self.memories,attention=self.attention,top=top,
            checks=dict(finite=bool(torch.isfinite(logits).all()),checkpoint_verified=self.receipt.get('verified',False),projection_max_error=max(p['max_error'] for p in self.projections)),
            seconds=round(time.perf_counter()-started,3),device=self.device,detail_token=self.T-1,
            provenance='Published checkpoint; full-depth FP32 reference forward. Serial GDN; no production TPU bf16/fused parity claim. Projection detail retains selected exact channel products.')
        assert result['checks']['finite']
        return result

def run(root,text='To be, or not to be.'):
    from transformers import AutoTokenizer
    root=Path(root);path,cfg,receipt=download(root)
    tok=AutoTokenizer.from_pretrained('openai-community/gpt2')
    ids=tok.encode(text)[:8];labels=[tok.decode([i]) for i in ids]
    engine=Capture(path,cfg,receipt);result=engine.capture(ids,labels)
    for rows in result['top']:
        for item in rows:item['label']=tok.decode([item['id']])
    # Replay through the same equations without observation to catch collection side effects.
    plain=engine.capture(ids,labels,False)
    result['checks']['plain_top_ids_equal']=bool(torch.equal(plain.topk(5,-1).indices,torch.tensor([[p['id'] for p in row] for row in result['top']])))
    result['checks']['plain_top_ids_equal'] or (_ for _ in ()).throw(AssertionError('Observation changed top IDs'))
    raw=json.dumps(result,separators=(',',':'),ensure_ascii=False).encode()
    (root/'production-trace.json').write_bytes(raw)
    with gzip.open(root/'production-trace.json.gz','wb') as f:f.write(raw)
    report=dict(checks=result['checks'],tokens=len(ids),modules=len(result['modules']),routes=len(result['routes']),projections=len(result['projections']),bytes=len(raw),gzip_bytes=(root/'production-trace.json.gz').stat().st_size,seconds=result['seconds'],device=result['device'])
    (root/'capture-report.json').write_text(json.dumps(report,indent=2))
    print('PRODUCTION_CAPTURE_READY',json.dumps(report),flush=True)
    return result,engine,tok
