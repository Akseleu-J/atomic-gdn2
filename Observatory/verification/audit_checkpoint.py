"""Independent NumPy/FP32 serial replay audit of the exported Observatory trace.

This checks published weights and the reference capture, not TPU fused parity.
Only NumPy is needed; safetensors is read through a read-only memory map.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,gzip,hashlib,json,math,struct,time
from pathlib import Path
import numpy as np

EXPECTED_SHA='4f44215338eaeab0fc043b9151ad7f32e798a0ae34ae83626b458c356e750ca8'

class Audit:
    def __init__(self,weights,trace,config):
        self.path=Path(weights);self.trace=trace;self.cfg=config;self.T=len(trace['tokens'])
        with self.path.open('rb') as f:
            n=struct.unpack('<Q',f.read(8))[0];self.header=json.loads(f.read(n));self.offset=8+n
        self.cache={};self.errors={};self.projections={p['id']:p for p in trace['projections']}
    def w(self,name):
        if name not in self.cache:
            h=self.header[name];assert h['dtype']=='F32'
            self.cache[name]=np.memmap(self.path,dtype='<f4',mode='r',offset=self.offset+h['data_offsets'][0],shape=tuple(h['shape']))
        return self.cache[name]
    def compare(self,name,a,b):
        a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(name,a.shape,b.shape)
        assert np.isfinite(a).all() and np.isfinite(b).all(),name
        self.errors[name]=max(self.errors.get(name,0),float(np.max(np.abs(a-b))))
    def norm(self,x,prefix,rms=False):
        v=np.mean(x*x,-1,keepdims=True)
        if rms:return x/np.sqrt(v+np.float32(1e-6))*self.w(prefix+'/scale')
        mean=np.mean(x,-1,keepdims=True)
        return (x-mean)/np.sqrt(np.maximum(v-mean*mean,0)+np.float32(1e-6))*self.w(prefix+'/scale')+self.w(prefix+'/bias')
    def dense(self,x,path):
        y=x@self.w(path+'/kernel')
        if path+'/bias' in self.header:y=y+self.w(path+'/bias')
        p=self.projections[path]
        self.compare('projection_input',x[-1],p['input']);self.compare('projection_output',y[-1],p['output'])
        return y
    def stage(self,id,x):
        saved=next(s for s in self.trace['stages'] if s['id']==id)['values']
        self.compare('stage_group_means',x.reshape(self.T,64,-1).mean(-1,dtype=np.float64),saved)
    def module(self,id,residual,normalized,output=None):
        m=next(m for m in self.trace['modules'] if m['stage']==id)
        for key,x in [('residual',residual),('normalized',normalized),('output',output)]:
            if x is not None:self.compare('module_'+key+'_group_means',x.reshape(self.T,64,-1).mean(-1,dtype=np.float64),m[key])
    def softmax(self,x,axis):
        e=np.exp(x-np.max(x,axis=axis,keepdims=True));return e/np.sum(e,axis=axis,keepdims=True)
    def dar(self,x,sources,ids,prefix,target):
        if not sources:return x
        values=np.stack(sources);keys=self.norm(values,prefix+'/key_norm',True)
        score=np.einsum('d,std->st',self.w(prefix+'/query'),keys)
        weights=self.softmax(score,0);retrieved=np.einsum('st,std->td',weights,values)
        r=next(r for r in self.trace['routes'] if r['target']==target);assert r['sources']==ids
        self.compare('dar_weights',weights.T,r['weights'])
        self.compare('dar_scores',score.T,r['scores'])
        self.compare('dar_residual_group_means',x.reshape(self.T,64,-1).mean(-1,dtype=np.float64),r['residual'])
        self.compare('dar_retrieved_group_means',retrieved.reshape(self.T,64,-1).mean(-1,dtype=np.float64),r['retrieved'])
        return x+retrieved
    def rope(self,x):
        d=x.shape[-1];inv=1/(self.cfg['rope_theta']**(np.arange(0,d,2,dtype=np.float32)/d))
        f=np.arange(self.T,dtype=np.float32)[:,None]*inv[None];angles=np.concatenate([f,f],-1)[:,None,:]
        return x*np.cos(angles)+np.concatenate([-x[...,d//2:],x[...,:d//2]],-1)*np.sin(angles)
    def gdn(self,x,path):
        H=self.cfg['n_heads'];D=self.cfg['d_model']//H
        q,k,v,w,b,g=[self.dense(x,path+'/Dense_'+str(i)).reshape(self.T,H,D) for i in range(6)]
        if self.cfg['l2norm_qk']:
            q=q/np.sqrt(np.sum(q*q,-1,keepdims=True)+np.float32(1e-6))
            k=k/np.sqrt(np.sum(k*k,-1,keepdims=True)+np.float32(1e-6))
        b=1/(1+np.exp(-b));g=-np.minimum(np.logaddexp(np.float32(0),g),np.float32(160/self.cfg['gdn2_bs2']))
        state=np.zeros((H,D,D),np.float32);outputs=[];energies=[]
        for t in range(self.T):
            decayed=state*np.exp(g[t])[:,:,None]
            erase=np.einsum('hd,hdv->hv',b[t]*k[t],decayed)
            update=k[t,:,:,None]*(w[t]*v[t]-erase)[:,None,:]
            state=decayed+update
            energies.append(np.sqrt(np.mean(state*state,axis=(-1,-2))))
            outputs.append(np.einsum('hd,hdv->hv',q[t]/np.float32(math.sqrt(D)),state))
        saved=next(g for g in self.trace['memories'] if g['id']==path)
        assert saved['token']==self.T-1
        self.compare('memory_rms',np.stack(energies),saved['rms'])
        pooled=lambda a:a.reshape(H,16,D//16,16,D//16).mean((2,4),dtype=np.float64)
        self.compare('memory_group_means',pooled(state),saved['states'])
        self.compare('memory_update_group_means',pooled(update),saved['updates'])
        return self.dense(np.stack(outputs).reshape(self.T,-1),path+'/Dense_6')
    def mla(self,x,path):
        c=self.cfg;H=c['n_heads'];dn=c['qk_nope_head_dim'];dr=c['qk_rope_head_dim'];D=dn+dr
        q=self.dense(x,path+'/Dense_0').reshape(self.T,H,D)
        latent=self.norm(self.dense(x,path+'/Dense_1'),path+'/LayerNorm_0')
        kn=self.dense(latent,path+'/Dense_2').reshape(self.T,H,dn)
        v=self.dense(latent,path+'/Dense_3').reshape(self.T,H,D)
        kr=np.broadcast_to(self.rope(self.dense(x,path+'/Dense_4').reshape(self.T,1,dr)),(self.T,H,dr))
        q=np.concatenate([q[...,:dn],self.rope(q[...,dn:])],-1);k=np.concatenate([kn,kr],-1)
        score=np.einsum('thd,shd->hts',q,k)/np.float32(math.sqrt(D))
        a=self.softmax(np.where(np.tril(np.ones((self.T,self.T),bool)),score,np.float32(-1e9)),-1)
        saved=next(g for g in self.trace['attention'] if g['id']==path)
        self.compare('attention_probabilities',a,saved['attention'])
        self.compare('attention_latent',latent,saved['latent'])
        return self.dense(np.einsum('hts,shd->thd',a,v).reshape(self.T,-1),path+'/Dense_5')
    def run(self):
        c=self.cfg;emb=self.w('Embed_0/embedding');x=np.asarray(emb[[t['id'] for t in self.trace['tokens']]])
        self.stage('embedding',x);history=[];histids=[]
        pattern=c.get('layer_types_override') or (['gdn2']*(c['layers_per_block']-1)+['mla'])*c['num_blocks']
        for b in range(c['num_blocks']):
            start=x;local=[];localids=[]
            for l in range(c['layers_per_block']):
                sid=f'block_{b}_layer_{l}';prefix=f'block_{b}/layer_{l}'
                routed=self.dar(x,history+local,histids+localids,prefix+'/dar',sid)
                norm=self.norm(routed,prefix+'/pre_sublayer_norm',True)
                fn=self.gdn if pattern[b*c['layers_per_block']+l]=='gdn2' else self.mla
                delta=np.nan_to_num(np.clip(fn(norm,prefix+'/sublayer'),-1000,1000))
                out=np.nan_to_num(np.clip(x+delta,-1000,1000));self.module(sid,x,norm,out)
                self.stage(sid,delta);local.append(delta);localids.append(sid);x=out
            sid=f'block_{b}_ffn';prefix=f'block_{b}'
            routed=self.dar(x,history+local,histids+localids,prefix+'/ffn_dar',sid)
            norm=self.norm(routed,prefix+'/LayerNorm_0')
            g=self.dense(norm,prefix+'/ffn/Dense_0');u=self.dense(norm,prefix+'/ffn/Dense_1')
            delta=self.dense(g/(1+np.exp(-g))*u,prefix+'/ffn/Dense_2')
            self.module(sid,x,norm,x+delta);self.stage(sid,delta);x=x+delta;summary=x-start;sid=f'block_{b}_summary'
            self.stage(sid,summary);history.append(summary);histids.append(sid)
            print('NUMPY_AUDIT_BLOCK',b+1,flush=True)
        norm=self.norm(x,'LayerNorm_0');self.module('logits',x,norm)
        logits=norm@emb.T;p=self.softmax(logits,-1)
        ids=np.argsort(-p,axis=-1)[:,:5]
        saved=np.asarray([[q['id'] for q in row] for row in self.trace['top']])
        self.top_ids_equal=bool(np.array_equal(ids,saved))
        self.compare('top5_probability',np.take_along_axis(p,saved,axis=-1),[[q['prob'] for q in row] for row in self.trace['top']])
    def verify_products(self):
        maxw=0.;maxp=0.;gemv=0.;count=0
        for p in self.trace['projections']:
            w=self.w(p['id']+'/kernel');a=np.asarray(p['input'],np.float32)
            y=a@w
            if p['id']+'/bias' in self.header:y=y+self.w(p['id']+'/bias')
            gemv=max(gemv,float(np.max(np.abs(y-np.asarray(p['output'])))))
            for i,j,x,v,c in p['edges']:
                maxw=max(maxw,abs(float(w[i,j])-v));maxp=max(maxp,abs(float(np.float32(x)*w[i,j])-c));count+=1
        assert maxw==0 and maxp==0,(maxw,maxp)
        return dict(individual_products=count,selected_weight_byte_mismatches=0,selected_fp32_product_mismatches=0,projection_gemv_max_abs_error=gemv)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--weights',required=True);parser.add_argument('--trace',required=True);parser.add_argument('--config',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args();started=time.perf_counter()
    with Path(args.weights).open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    assert sha==EXPECTED_SHA
    raw=Path(args.trace).read_bytes();data=gzip.decompress(raw) if raw[:2]==b'\x1f\x8b' else raw
    trace=json.loads(data);cfg=json.loads(Path(args.config).read_text())['model']
    audit=Audit(args.weights,trace,cfg);products=audit.verify_products();audit.run()
    report=dict(checkpoint_sha256=sha,trace_file_sha256=hashlib.sha256(raw).hexdigest(),numpy_version=np.__version__,
        scope='Independent NumPy FP32 serial replay vs PyTorch FP32 capture. Not production TPU bf16/fused parity.',
        tokens=audit.T,modules=len(trace['modules']),projections=len(trace['projections']),**products,
        full_replay_max_abs_errors=audit.errors,top5_ids_equal=audit.top_ids_equal,seconds=round(time.perf_counter()-started,3))
    report['passed']=audit.top_ids_equal and max(audit.errors.values())<1e-3 and products['projection_gemv_max_abs_error']<1e-3
    Path(args.report).write_text(json.dumps(report,indent=2));print('INDEPENDENT_AUDIT',json.dumps(report),flush=True)
    assert report['passed'],report
if __name__=='__main__':main()
