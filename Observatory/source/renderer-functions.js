// Stable scales are measured across all captured tokens for each comparable bank.
function normalization(){
 const cacheKey=[trace.version,mode,selected,head,$('groups').value,expanded].join(':');
 if(normCache.has(cacheKey))return normCache.get(cacheKey);
 const result={banks:new Map(),dense:1e-8};
 const add=g=>{for(const n of g.nodes)result.banks.set(F.bank(n),Math.max(result.banks.get(F.bank(n))||1e-8,Math.abs(n.v)));for(const e of g.edges)if(e.kind==='dense')result.dense=Math.max(result.dense,Math.abs(e.v));};
 if(trace.version===4&&['network','projection'].includes(mode)){
  for(let t=0;t<trace.tokens.length;t++)add(ProductionGraph(trace,t,mode==='projection',selected,Number($('groups').value),expanded));
 }else{
  add({nodes,edges});
  // DAR uses signed source/retrieved vectors: one fixed bound across this capture.
  if(mode==='dar'){let max=1e-8;for(const s of trace.stages)for(const row of s.values)for(const v of row)max=Math.max(max,Math.abs(v));for(const k of result.banks.keys())result.banks.set(k,max);}
  if(mode==='attention')for(const k of result.banks.keys())result.banks.set(k,1);
 }
 normCache.set(cacheKey,result);return result;
}
function uploadGeometry(){
 const norms=normalization(),point=[],pulse=[];let size=0;geometryTier=camera.distance<150?1:0;
 nodeIndex=new Map(nodes.map((n,i)=>[F.key(n),i]));nodeByPosition=new Map(nodes.map(n=>[n.p,n]));adjacency=new Map(nodes.map(n=>[F.key(n),[]]));
 const work=edges.map((e,i)=>{
  const a=nodeByPosition.get(e.a),b=nodeByPosition.get(e.b),ia=a?nodeIndex.get(F.key(a)):-2,ib=b?nodeIndex.get(F.key(b)):-2;
  if(a)adjacency.get(F.key(a)).push(e);if(b&&b!==a)adjacency.get(F.key(b)).push(e);
  const curve=F.curve(e),arc=F.arc(e,curve),steps=e.kind==='frame'?1:(geometryTier?(e.arc?16:10):(e.arc?8:4));
  const col=F.palette(e,norms.dense),key=(e.source||a&&F.key(a)||e.a.join(','))+'>'+(e.target||b&&F.key(b)||e.b.join(','))+':'+e.kind;
  size+=steps*2*26;return {e,curve,arc,steps,col,key,ia,ib,i};
 });
 const line=new Float32Array(size);let offset=0;lastNodeColors=new Map();lastEdgeColors=new Map();
 function vertex(arr,w,t,col,old){
  const values=[...w.e.a,...w.e.b,...col,t,7/w.arc[3],w.curve[0],w.curve[1],w.ia,w.ib,w.curve[2],w.curve[3],...old,...w.arc.slice(0,3),F.blockEdge(w.e)?1:0];
  if(arr===line){line.set(values,offset);offset+=26;}else arr.push(...values);
 }
 for(const w of work){
  const old=previousEdgeColors.get(w.key)||w.col;lastEdgeColors.set(w.key,w.col);
  for(let j=0;j<w.steps;j++){vertex(line,w,j/w.steps,w.col,old);vertex(line,w,(j+1)/w.steps,w.col,old);}
  // Sparse trails on numerical paths only. Four samples form a tapered tail.
  if(w.e.kind!=='frame'&&w.e.kind!=='operation'&&w.i%Math.max(1,Math.ceil(edges.length/1000))===0){
   for(let tail=0;tail<4;tail++){
    const col=[...w.col.slice(0,3),(.24-tail*.05)*(F.blockEdge(w.e)?.1:1)],q=(w.i*.61803398875-tail*.35/w.arc[3]+1)%1;
    vertex(pulse,w,q,col,col);
   }
  }
 }
 for(const n of nodes){
  const id=F.key(n),col=F.nodeColor(n,norms.banks.get(F.bank(n))||1),index=nodeIndex.get(id);
  if(/block_\d+_summary/.test(id))col[3]*=.4;
  lastNodeColors.set(id,col);
  vertex(point,{e:{a:n.p,b:n.p,kind:'node',source:id},curve:[0,0,0,0],arc:[.25,.5,.75,1],ia:index,ib:index},0,col,previousNodeColors.get(id)||col);
 }
 for(const [name,data]of Object.entries({line,node:point,pulse})){
  gl.bindBuffer(gl.ARRAY_BUFFER,buffers[name]);gl.bufferData(gl.ARRAY_BUFFER,data instanceof Float32Array?data:new Float32Array(data),gl.STATIC_DRAW);counts[name]=data.length/26;
 }
 kept=edges.filter(e=>e.kind==='dense').length;pickDirty=true;
 canvas.dataset.style='filament-v2';canvas.dataset.lineVertices=String(counts.line);
 canvas.dataset.blockAlpha=JSON.stringify({history:.0012,blockOperations:.0025,local:.055});
}
function setSelection(n){
 const next=n?nodeIndex.get(F.key(n)):-1;
 if(next!==selectionId&&next>=0){selectionId=next;selectionFade=0;}
 selectionGoal=n?1:0;
 canvas.dataset.selection=n?F.key(n):'';canvas.dataset.pinned=focused?'true':'false';
}
function inspectNode(n){
 if(!n)return;
 const connected=adjacency.get(F.key(n))||[];
 $('inspectTitle').textContent=n.title;
 $('inspect').textContent=(n.structural?'Structural operation junction.':'Captured token value: '+n.v.toPrecision(7))+'\n'+connected.length+' saved incident links. '+(focused?'Pinned · Esc releases selection.':'Hover · click to pin.')+'\n'+connected.slice().sort((a,b)=>Math.abs(b.v)-Math.abs(a.v)).slice(0,4).map(e=>e.info).join('\n');
}
function focusCamera(n,distance=45){if(!n)return;cameraGoal={target:[...n.p],distance};}
function updateScreenNodes(){
 if(!matrix)return;screenNodes=[];
 for(const n of nodes){const p=project(n.p);if(p[2]>0&&p[0]>=0&&p[0]<=innerWidth&&p[1]>=0&&p[1]<=innerHeight)screenNodes.push({n,p});}
 pickDirty=false;const probe=screenNodes.filter(q=>q.n.id?.includes('/selected y')).reduce((a,b)=>!a||Math.hypot(b.p[0]-innerWidth/2,b.p[1]-innerHeight/2)<Math.hypot(a.p[0]-innerWidth/2,a.p[1]-innerHeight/2)?b:a,null);if(probe)canvas.dataset.pickTarget=JSON.stringify({x:probe.p[0],y:probe.p[1],id:F.key(probe.n)});
}
function pickNode(x,y,radius=10){
 if(pickDirty)updateScreenNodes();let best=null,bestDistance=radius;
 for(const {n,p}of screenNodes){const d=Math.hypot(p[0]-x,p[1]-y);if(d<bestDistance){best=n;bestDistance=d;}}
 return best;
}
