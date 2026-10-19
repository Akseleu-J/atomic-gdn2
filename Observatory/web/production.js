/* Bounded graph for full-depth checkpoint captures. Groups are explicitly means. */
(function(root){
root.ProductionGraph=function(trace,t,detail,moduleIndex=0,groups=32,expanded=false){
 const nodes=[],edges=[],labels=[],banks=new Map(),contexts=[],blockStarts=new Map();let total=0,projectionCount=0;
 const col=v=>v>=0?[1,.64,.4,1]:[.32,.93,.8,1];
 function bank(id,vals,x,z=0,y=0,h=24,indices=null){const b=vals.map((v,i)=>{let n={id:id+':'+(indices?indices[i]:i),p:[x,y+(.5-i/Math.max(1,vals.length-1))*h,z],v,c:col(v),title:id+' ['+(indices?indices[i]:i)+']'+(!detail&&!indices?' · group mean':'')};nodes.push(n);return n;});banks.set(id,b);return b;}
 function line(a,b,v,kind,info,arc=0){edges.push({a:a.p,b:b.p,v,kind,info,arc,source:a.id,target:b.id});}
 function diag(a,b,kind,info,scale=1,arc=0){a.forEach((n,i)=>line(n,b[i],n.v*scale,kind,info+'\n'+(n.v*scale).toPrecision(6),arc));}
 function mean(v){const a=[];for(let g=0;g<groups;g++){const lo=Math.floor(g*v.length/groups),hi=Math.floor((g+1)*v.length/groups);let sum=0;for(let c=lo;c<hi;c++)sum+=v[c];a.push(sum/(hi-lo));}return a;}
 const stage=id=>trace.stages.find(s=>s.id===id).values[t];
 // Only these projections consume the normalized module input directly.
 // Latent MLA, GDN readout and SwiGLU output projections have other inputs.
 const inputProjectionIndices={gdn2:[0,1,2,3,4,5],mla:[0,1,4],ffn:[0,1]};
 function addInputProjection(ctx,p,slot=0){
  const {m,x,normalized,op}=ctx;
  const src=[...new Set(p.edges.map(e=>e[0]))].sort((a,b)=>a-b),dst=[...new Set(p.edges.map(e=>e[1]))].sort((a,b)=>a-b);
  const z=slot===0?4:(slot%2?-1:1)*(4+Math.floor(slot/2)*4);
  const a=bank(p.id+'/selected x',src.map(i=>p.input[i]),x+2.8,z,0,24,src),b=bank(p.id+'/selected y',dst.map(i=>p.output[i]),x+6.3,z,0,24,dst);
  const ai=new Map(src.map((v,i)=>[v,a[i]])),bi=new Map(dst.map((v,i)=>[v,b[i]]));
  for(const [i,j,v,w,c] of p.edges)line(ai.get(i),bi.get(j),c,'dense',`${p.id} · individual channels\nx[${i}] × W[${i},${j}] = ${c.toPrecision(6)}\nx=${v.toPrecision(5)} · W=${w.toPrecision(5)}`);
  a.forEach((n,i)=>line(normalized[Math.floor(src[i]*groups/p.input_dim)],n,0,'operation','Channel membership in normalized tensor group · structural, not a numerical mean-to-channel map'));
  b.forEach(n=>line(n,op,0,'operation','Projection participates in sublayer operation'));
  total+=p.total_edges;projectionCount++;
 }
 if(detail){
  const m=trace.modules[moduleIndex],ds=trace.projections.filter(p=>p.id.startsWith(m.id+'/'));
  if(!ds.length){labels.push({p:[0,0,0],title:'LM head',sub:'Full logits top-5 in inspector; 50,257 × 1024 not sent to browser'});return{nodes,edges,labels,total,width:30};}
  ds.forEach((p,k)=>{
   const src=[...new Set(p.edges.map(e=>e[0]))].sort((a,b)=>a-b),dst=[...new Set(p.edges.map(e=>e[1]))].sort((a,b)=>a-b);
   const x=(k-(ds.length-1)/2)*7,z=(k%2-.5)*4;
   const a=bank(p.id+'/x',src.map(i=>p.input[i]),x,z,0,20,src),b=bank(p.id+'/y',dst.map(i=>p.output[i]),x+4,z,0,20,dst);
   const ai=new Map(src.map((i,k)=>[i,a[k]])),bi=new Map(dst.map((i,k)=>[i,b[k]]));
   for(const [i,j,v,w,c]of p.edges)line(ai.get(i),bi.get(j),c,'dense',`${p.id}\nx[${i}] × W[${i},${j}] = ${c.toPrecision(6)}\nx=${v.toPrecision(5)} · W=${w.toPrecision(5)}`);
   total+=p.total_edges;labels.push({p:[x+2,12,z],title:p.id.split('/').at(-1),sub:`${p.edges.length.toLocaleString()} / ${p.total_edges.toLocaleString()} exact products · token ${p.token}`});
  });
  labels.push({p:[0,16,0],title:m.label,sub:'Individual channels · selected strongest outputs and their inputs'});
  return{nodes,edges,labels,total,width:Math.max(35,ds.length*7),detail:true};
 }
 const width=trace.modules.length*12,origin=-width/2;
 let residual=bank('Embedding · channel means',mean(stage('embedding')),origin);
 for(let i=0;i<trace.modules.length;i++){
  const m=trace.modules[i],x=origin+i*12+3,r=trace.routes.find(r=>r.target===m.stage);
  if(!blockStarts.has(m.block))blockStarts.set(m.block,residual);
  if(m.kind==='head'){
   const a=bank('Head LayerNorm · means',mean(m.normalized[t]),x+2),op=bank('Tied LM head',[0],x+6,0,0,0);
   op[0].structural=true;
   diag(residual,a,'operation','LayerNorm tensor dependency');a.forEach(n=>line(n,op[0],0,'operation','Tied embedding projection; probabilities in inspector'));
   labels.push({p:[x+3,10,0],title:'GPT-2 head',sub:'50,257 vocabulary'});continue;
  }
  let routed=residual;
  if(r){const got=bank(m.stage+'/retrieved · means',mean(r.retrieved[t]),x,-2),sum=bank(m.stage+'/routed · means',mean(r.residual[t].map((v,c)=>v+r.retrieved[t][c])),x+.8);
   diag(residual,sum,'residual','Residual contribution');diag(got,sum,'retrieved','Retrieved Δ');
   r.sources.forEach((id,s)=>{const src=banks.get(id),hist=id.endsWith('_summary');if(!src)throw Error('Missing production DAR source '+id);diag(src,got,hist?'history':'local',id+' → '+m.stage+' · α='+r.weights[t][s].toPrecision(6),r.weights[t][s],hist?-12-Math.abs(src[0].p[0]-x)*.025:7+Math.abs(src[0].p[0]-x)*.02);});routed=sum;
  }
  const normalized=bank(m.stage+'/normalized · means',mean(m.normalized[t]),x+2),delta=bank(m.stage,mean(stage(m.stage)),x+7.2),next=bank(m.stage+'/residual · means',mean(m.output[t]),x+9.4);
  diag(routed,normalized,'operation','RMSNorm/LayerNorm · structural dependency');
  const op=bank(m.id+' operation',[0],x+4.5,0,-10,0)[0];op.structural=true;normalized.forEach(n=>line(n,op,0,'operation',m.label+' tensor input'));delta.forEach(n=>line(op,n,0,'operation',m.label+' measured Δ output'));
  const ctx={m,x,normalized,op};contexts.push(ctx);
  if(groups>=32||expanded){const p=trace.projections.find(p=>p.id.startsWith(m.id+'/'));if(p)addInputProjection(ctx,p);}
diag(residual,next,'residual','Residual skip',1,2);diag(delta,next,'residual','Δ addition');residual=next;
  labels.push({p:[x+4,17,0],title:m.label,sub:groups+' channel-group means · '+trace.channels+' actual channels'});
  if(m.kind==='ffn'){const id=`block_${m.block}_summary`,summary=bank(id,mean(stage(id)),x+6,4,-19,9);diag(next,summary,'operation','Block end contribution to captured delta');diag(blockStarts.get(m.block),summary,'operation','Subtract block start to form captured delta',-1,-3);labels.push({p:[x+6,-12,4],title:'Δ BLOCK '+(m.block+1),sub:'all earlier blocks available to later H-DAR'});}
 }
 // Add genuine captured products across modules, keeping every existing DAR route.
 // Stop after reaching 40k graph links; no duplicate or decorative links are added.
 if(expanded){
  for(let slot=1;slot<6&&edges.length<40000;slot++){
   for(const ctx of contexts){
    const index=inputProjectionIndices[ctx.m.kind]?.[slot];if(index===undefined)continue;
    const p=trace.projections.find(p=>p.id===ctx.m.id+'/Dense_'+index);if(!p)continue;
    addInputProjection(ctx,p,slot);
    if(edges.length>=40000)break;
   }
  }
 }
 nodes.forEach(n=>{n.p[0]*=.72;n.p[1]*=1.45;n.p[2]*=1.5;});labels.forEach(l=>{l.p[0]*=.72;l.p[1]*=1.45;l.p[2]*=1.5;});return{nodes,edges,labels,total,width:width*.72,groups,expanded,projectionCount};
};
})(typeof globalThis!=='undefined'?globalThis:this);
