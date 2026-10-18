/* The unified graph uses trace identities, never a proxy module for a block delta. */
(function(root){
function graph(trace,t,density=100){
 const nodes=[],edges=[],labels=[],banks=new Map(),routes=[],C=trace.channels;
 let total=0;const width=trace.modules.length*18+8,origin=-width/2;
 const values=id=>trace.stages.find(s=>s.id===id).values[t];
 const rgb=v=>v>=0?[1,.62,.39]:[.32,.96,.8];
 function bank(id,vals,x,z=0,y=0,h=13){
  if(banks.has(id))return banks.get(id);
  const max=Math.max(1e-8,...vals.map(Math.abs));
  const b=vals.map((v,c)=>{const n={id:id+':'+c,p:[x,y+(.5-c/Math.max(1,vals.length-1))*h,z],v,title:id+' ['+c+']',c:[...rgb(v),.55+.45*Math.sqrt(Math.abs(v)/max)]};nodes.push(n);return n;});banks.set(id,b);return b;
 }
 function link(a,b,v,kind,info,arc=0){edges.push({a:a.p,b:b.p,v,kind,info,arc,source:a.id,target:b.id});}
 function diagonal(a,b,kind,info,factor=1,arc=0){if(a.length!==b.length)throw Error('Channel mismatch '+info);a.forEach((n,c)=>link(n,b[c],n.v*factor,kind,info+'\nchannel '+c+' · contribution '+(n.v*factor).toPrecision(6),arc));}
 function operation(name,inputs,out,x,y=0,z=0){
  const n={id:name+"@"+x+","+y+","+z,p:[x,y,z],v:0,title:name+' · tensor operation (structural dependency)',c:[.5,.7,1,1]};nodes.push(n);
  inputs.forEach(b=>b.forEach(a=>link(a,n,0,'operation',name+' · tensor dependency, not a numerical attribution')));
  out.forEach(b=>link(n,b,0,'operation',name+' · measured output'));
  labels.push({p:[x,y+1.3,z],title:name,sub:'tensor operation'});
 }
 function dense(d,a,b){const w=d.weight,all=[];for(let i=0;i<a.length;i++)for(let j=0;j<b.length;j++)all.push({i,j,v:d.input[t][i]*w[i][j]});total+=all.length;
  all.sort((a,b)=>Math.abs(b.v)-Math.abs(a.v));for(const e of all.slice(0,Math.ceil(all.length*density/100)))link(a[e.i],b[e.j],e.v,'dense',`${d.id}\nx[${e.i}] × W[${e.i},${e.j}] = ${e.v.toPrecision(6)}\nx = ${d.input[t][e.i].toPrecision(5)} · W = ${w[e.i][e.j].toPrecision(5)}`);
 }
 let residual=bank('embedding',values('embedding'),origin,0),blockStart=residual;
 labels.push({p:[origin,9,0],title:'Embedding',sub:'residual stream'});
 for(let index=0;index<trace.modules.length;index++){
  const m=trace.modules[index],x=origin+5+index*18,ds=trace.dense.filter(d=>d.id.startsWith(m.id+'/'));
  const r=trace.routes.find(r=>r.target===m.stage);
  if(index===0||trace.modules[index-1].block!==m.block)blockStart=residual;
  let routed=residual;
  if(r){
   const retrieved=bank(m.stage+'/retrieved',r.retrieved[t],x,-3);
   routed=bank(m.stage+'/routed',r.residual[t].map((v,c)=>v+r.retrieved[t][c]),x+1,0);
   diagonal(residual,routed,'residual','Residual → routed input');diagonal(retrieved,routed,'retrieved','Retrieved Δ → routed input');
   r.sources.forEach((id,s)=>{
    const source=banks.get(id);if(!source)throw Error('Missing DAR source '+id);
    const history=id.endsWith('_summary'),alpha=r.weights[t][s];
    diagonal(source,retrieved,history?'history':'local',`${id} → ${m.stage}\nα = ${alpha.toPrecision(6)}`,alpha,history?-10-Math.abs(source[0].p[0]-x)*.05:5+Math.abs(source[0].p[0]-x)*.05);
    routes.push({source:id,target:m.stage,channels:C,alpha,kind:history?'history':'local'});
   });
   labels.push({p:[x,9,-3],title:'H-DAR',sub:r.sources.length+' sources'});
  }
  const normalized=bank(m.id+'/normalized',ds[0].input[t],x+2,0);
  operation(m.kind==='head'?'Head LayerNorm':m.kind==='ffn'?'LayerNorm':'RMSNorm',[routed],normalized,x+1.5,-9);
  labels.push({p:[x+8,13,0],title:m.label,sub:m.kind==='head'?'tied embedding':'delta + residual'});
  if(m.kind==='head'){
   const logits=bank('logits',ds[0].output[t],x+12,0,0,27);dense(ds[0],normalized,logits);continue;
  }
  const outputs=[];let outputInput;
  if(m.kind==='ffn'){
   for(let i=0;i<2;i++){const b=bank(ds[i].id,ds[i].output[t],x+6,(i-.5)*5,0,18);dense(ds[i],normalized,b);outputs.push(b);}
   outputInput=bank(m.id+'/gated',ds[2].input[t],x+10,0,0,18);
   operation('SiLU(g) × up',outputs,outputInput,x+8,-11);
  }else if(m.kind==='gdn2'){
   for(let i=0;i<ds.length-1;i++){const b=bank(ds[i].id,ds[i].output[t],x+6,(i-2.5)*2.3);dense(ds[i],normalized,b);outputs.push(b);}
   outputInput=bank(m.id+'/kernel_output',ds.at(-1).input[t],x+10,0);
   operation('GDN · normalize / gates / state',outputs,outputInput,x+8,-10);
  }else{
   const q=bank(ds[0].id,ds[0].output[t],x+6,-5),kv=bank(ds[1].id,ds[1].output[t],x+5,0),rope=bank(ds[4].id,ds[4].output[t],x+6,5);
   dense(ds[0],normalized,q);dense(ds[1],normalized,kv);dense(ds[4],normalized,rope);
   const latent=bank(m.id+'/latent_norm',ds[2].input[t],x+6,0);operation('KV LayerNorm',[kv],latent,x+5.5,-9);
   const k=bank(ds[2].id,ds[2].output[t],x+8,-2),v=bank(ds[3].id,ds[3].output[t],x+8,2);dense(ds[2],latent,k);dense(ds[3],latent,v);
   outputInput=bank(m.id+'/attention_output',ds.at(-1).input[t],x+10,0);
   operation('RoPE / causal attention',[q,k,v,rope],outputInput,x+9,-10);
  }
  const delta=bank(m.stage,values(m.stage),x+13,0);dense(ds.at(-1),outputInput,delta);
  const next=bank(m.stage+'/residual_out',residual.map((n,c)=>{let v=n.v+delta[c].v;return m.kind==='ffn'?v:Math.max(-1000,Math.min(1000,v));}),x+15,0);
  diagonal(residual,next,'residual','Residual skip',1,2);diagonal(delta,next,'residual','Layer Δ addition');
  labels.push({p:[x+13,8,0],title:'Δ '+m.label,sub:m.kind==='ffn'?'included in block delta':'source for subsequent local DAR'});
  residual=next;
  if(m.kind==='ffn'){
   const id=`block_${m.block}_summary`,summary=bank(id,values(id),x+14,3,-17,9);
   diagonal(blockStart,summary,'summary','Block Δ = x_end − x_start',-1,-2);diagonal(residual,summary,'summary','Block Δ = x_end − x_start',1,-2);
   labels.push({p:[x+14,-11,3],title:`Δ BLOCK ${m.block+1}`,sub:'x_end − x_start · includes FFN'});
  }
 }
 return {nodes,edges,labels,total,width,routes};
}
root.AtomicGraph=graph;if(typeof module==='object'&&module.exports)module.exports=graph;
})(typeof globalThis!=='undefined'?globalThis:this);

