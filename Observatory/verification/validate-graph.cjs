/* Validate exported checkpoint evidence, routing, and rendered graph endpoints. */
const fs=require('node:fs'),assert=require('node:assert/strict'),path=require('node:path');
require('../web/production.js');
const t=JSON.parse(fs.readFileSync(path.join(__dirname,'../web/trace.json'),'utf8'));
const stages=new Map(t.stages.map(s=>[s.id,s]));
assert.equal(t.channels,1024);assert.equal(t.modules.length,36);assert(t.checks.checkpoint_verified);
assert.equal(t.metadata.checkpoint.sha256,'4f44215338eaeab0fc043b9151ad7f32e798a0ae34ae83626b458c356e750ca8');
let finite=0,darError=0,productError=0,weightError=0,pairs=0,products=0;
function checkFinite(x){if(typeof x==='number'){assert(Number.isFinite(x));finite++;}else if(Array.isArray(x))x.forEach(checkFinite);else if(x&&typeof x==='object')Object.values(x).forEach(checkFinite);}
checkFinite(t);
for(const r of t.routes){
 const m=t.modules.find(m=>m.stage===r.target);assert(m);
 const local=t.modules.filter(s=>s.block===m.block&&t.modules.indexOf(s)<t.modules.indexOf(m)).map(s=>s.stage);
 const history=Array.from({length:m.block},(_,b)=>`block_${b}_summary`);
 assert.deepEqual(r.sources,[...history,...local]);pairs+=r.sources.length;
 for(let tok=0;tok<t.tokens.length;tok++){
  weightError=Math.max(weightError,Math.abs(r.weights[tok].reduce((a,b)=>a+b,0)-1));
  for(let c=0;c<64;c++){
   const v=r.sources.reduce((a,s,i)=>a+stages.get(s).values[tok][c]*r.weights[tok][i],0);
   darError=Math.max(darError,Math.abs(v-r.retrieved[tok][c]));
  }
 }
}
assert.equal(pairs,175);assert(darError<1e-5);assert(weightError<1e-6);
for(const p of t.projections)for(const [i,j,x,w,c] of p.edges){
 assert(i>=0&&i<p.input_dim&&j>=0&&j<p.output_dim);assert.equal(x,p.input[i]);
 productError=Math.max(productError,Math.abs(x*w-c));products++;
}
assert.equal(products,40320);assert(productError<1e-5);
const graphs=[];
for(const groups of [16,32,64])for(const expanded of [false,true])for(let tok=0;tok<t.tokens.length;tok++){
 const g=global.ProductionGraph(t,tok,false,0,groups,expanded),ids=new Set(g.nodes.map(n=>n.id)),unique=new Set();
 assert.equal(ids.size,g.nodes.length);
 for(const e of g.edges){
  assert(ids.has(e.source)&&ids.has(e.target));assert(e.a.every(Number.isFinite)&&e.b.every(Number.isFinite));
  const key=[e.source,e.target,e.kind].join('>');assert(!unique.has(key),'Duplicate edge '+key);unique.add(key);
  if(e.kind==='dense'){
   const [path,input]=e.source.split('/selected x:'),output=Number(e.target.split('/selected y:')[1]);
   const p=t.projections.find(p=>p.id===path),saved=p.edges.find(row=>row[0]===Number(input)&&row[1]===output);
   assert(saved);assert.equal(e.v,saved[4]);
  }
 }
 const dar=g.edges.filter(e=>e.kind==='local'||e.kind==='history');assert.equal(dar.length,pairs*groups);
 const starts=g.edges.filter(e=>e.info.startsWith('Subtract block start'));
 assert.equal(starts.length,7*groups);
 if(expanded)assert(g.edges.length>=40000);
 if(tok===7)graphs.push({groups,expanded,nodes:g.nodes.length,edges:g.edges.length,dense:g.edges.filter(e=>e.kind==='dense').length,dar_channels:dar.length,projections:g.projectionCount});
}
for(let i=0;i<t.modules.length;i++){
 const g=global.ProductionGraph(t,7,true,i,32),ids=new Set(g.nodes.map(n=>n.id));
 for(const e of g.edges)assert(ids.has(e.source)&&ids.has(e.target));
}
const report={checkpoint:t.metadata.checkpoint,finite_values_checked:finite,tokens:8,modules:36,dar_source_pairs:pairs,dar_mean_reconstruction_max_error:darError,softmax_sum_max_error:weightError,individual_dense_products:products,product_max_rounding_error:productError,all_graph_endpoints_valid:true,graph_sizes:graphs,scope:'Export and graph validation. Does not establish production TPU bf16/fused numerical parity.'};
fs.writeFileSync(path.join(__dirname,'validation-production.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify({...report,checkpoint:{sha256:report.checkpoint.sha256}}));
