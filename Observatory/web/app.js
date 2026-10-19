'use strict';
(()=>{
const $=id=>document.getElementById(id), canvas=$('scene'), gl=canvas.getContext('webgl',{alpha:true,antialias:true,preserveDrawingBuffer:true});
if(!gl){$('message').textContent='WebGL недоступен. Включите аппаратное ускорение браузера.';return;}
let expanded=true,projectionCount=0;
let trace,token=0,mode='network',selected=0,head=0,edges=[],nodes=[],labels=[],total=0,kept=0,clock=0,last=0,play=false,signals=true,orbit=false,nextToken=0,frames=0,fpsStart=0,frameTimes=[];
const F=Filaments;let geometryTier=0;let moving=true,selectionId=-1,selectionFade=0,selectionGoal=0,transition=1,transitionStart=0,visualTime=0,cameraGoal=null;
let nodeIndex=new Map(),nodeByPosition=new Map(),adjacency=new Map(),normCache=new Map(),previousNodeColors=new Map(),previousEdgeColors=new Map(),lastNodeColors=new Map(),lastEdgeColors=new Map(),screenNodes=[],pickDirty=true,hoverX=-1,hoverY=-1,pickStamp=0;
let camera={yaw:-.14,pitch:.13,distance:62,target:[0,0,0]},matrix,drag=null,hover=null,focused=null;
const vs=`
attribute vec3 a;attribute vec3 b;attribute vec4 c;attribute vec4 d;
attribute vec4 meta;attribute vec4 prior;attribute vec4 lut;
uniform mat4 vp;uniform float time,particle,scale,selectedId,focusMix,transition,sway,blockLight;
varying vec4 color;
float parameter(float q){
 if(q<lut.x)return .25*q/max(lut.x,.00001);
 if(q<lut.y)return .25+.25*(q-lut.x)/max(lut.y-lut.x,.00001);
 if(q<lut.z)return .5+.25*(q-lut.y)/max(lut.z-lut.y,.00001);
 return .75+.25*(q-lut.z)/max(1.-lut.z,.00001);
}
void main(){
 float q=fract(time*d.y+d.x);float t=particle>.5?parameter(q):d.x;
 float envelope=sin(t*3.14159265),wave=sin(time*.63+meta.z+t*4.2)*meta.w*sway;
 vec3 p=mix(a,b,t);p.z+=envelope*(d.z+wave);p.y+=envelope*(d.w+wave*.7);
 gl_Position=vp*vec4(p,1.);
 color=mix(prior,c,transition);
 float hit=selectedId>=0.&&(abs(meta.x-selectedId)<.25||abs(meta.y-selectedId)<.25)?1.:0.;
 float amount=focusMix*hit;
 color.rgb=mix(color.rgb,vec3(1.,.98,.92),amount);
 color.a*=mix(1.,.055,focusMix*(1.-hit));
 color.a=mix(color.a,max(color.a,.66),amount);
 if(lut.w>.5)color.a*=mix(blockLight,1.,amount);
 if(particle>.5)color.a*=smoothstep(0.,.06,q)*(1.-smoothstep(.9,1.,q));
 gl_PointSize=scale*(particle>.5?2.1:mix(1.9,3.4,sqrt(c.a)))*clamp(100./gl_Position.w,.65,4.5)*(1.+amount*.45);
}`;
const fs=`precision mediump float;varying vec4 color;uniform float point,exposure;
void main(){float alpha=1.;if(point>.5){float r=length(gl_PointCoord-.5)*2.;if(r>1.)discard;alpha=exp(-r*r*5.5);}gl_FragColor=vec4(color.rgb,color.a*alpha*exposure);}`;

function shader(type,s){const sh=gl.createShader(type);gl.shaderSource(sh,s);gl.compileShader(sh);if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(sh));return sh;}
const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vs));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
const loc={};for(const k of ['a','b','c','d','meta','prior','lut'])loc[k]=gl.getAttribLocation(program,k);for(const k of ['vp','time','particle','scale','point','exposure','selectedId','focusMix','transition','sway','blockLight'])loc[k]=gl.getUniformLocation(program,k);
const buffers={line:gl.createBuffer(),node:gl.createBuffer(),pulse:gl.createBuffer()};let counts={};
function multiply(a,b){let o=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)o[c*4+r]+=a[k*4+r]*b[c*4+k];return o;}
const sub=(a,b)=>a.map((v,i)=>v-b[i]),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],norm=a=>a.map(v=>v/Math.hypot(...a));
function view(){const t=camera.target,eye=[t[0]+Math.sin(camera.yaw)*Math.cos(camera.pitch)*camera.distance,t[1]+Math.sin(camera.pitch)*camera.distance,t[2]+Math.cos(camera.yaw)*Math.cos(camera.pitch)*camera.distance],z=norm(sub(eye,t)),x=[Math.cos(camera.yaw),0,-Math.sin(camera.yaw)],y=cross(z,x);const v=new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1]),f=1/Math.tan(.58/2),asp=canvas.width/canvas.height,n=.05,far=5000;return multiply(new Float32Array([f/asp,0,0,0,0,f,0,0,0,0,(far+n)/(n-far),-1,0,0,2*far*n/(n-far),0]),v);}
function project(p){const q=[...p,1],o=[0,0,0,0];for(let r=0;r<4;r++)for(let c=0;c<4;c++)o[r]+=matrix[c*4+r]*q[c];return [(o[0]/o[3]*.5+.5)*innerWidth,(-o[1]/o[3]*.5+.5)*innerHeight,o[3]];}
const color=(v,a=1)=>v>=0?[1,.49,.42,a]:[.3,1,.7,a];
function bank(values,x,height,z,title){let out=[];const max=Math.max(.000001,...values.map(Math.abs));for(let i=0;i<values.length;i++){const p=[x,(.5-i/Math.max(1,values.length-1))*height,z];const n={p,v:values[i],title:title+' ['+i+']',c:color(values[i],.3+.7*Math.sqrt(Math.abs(values[i])/max))};nodes.push(n);out.push(n);}return out;}
function label(p,title,sub=''){labels.push({p,title,sub});}
function edge(a,b,value,info,arc=0,kind='dense'){edges.push({a:a.p||a,b:b.p||b,v:value,info,arc,kind});}
function denseGraph(d,x,z=0,spacing=2.2,h=8){const input=d.input[token],out=d.output[token],a=bank(input,x,h,z,d.id+' input'),b=bank(out,x+spacing,h*Math.sqrt(out.length/input.length),z,d.id+' output');let all=[];for(let i=0;i<input.length;i++)for(let j=0;j<out.length;j++)all.push({i,j,v:input[i]*d.weight[i][j]});total+=all.length;all.sort((a,b)=>Math.abs(b.v)-Math.abs(a.v));let count=Math.max(1,Math.round(all.length*Number($('density').value)/100));for(const e of all.slice(0,count)){edge(a[e.i],b[e.j],e.v,`${d.id}\nx[${e.i}] × W[${e.i},${e.j}] = ${e.v.toPrecision(5)}\nx=${input[e.i].toPrecision(4)} · W=${d.weight[e.i][e.j].toPrecision(4)}`,0,'dense');}label([x+spacing/2,h/2+1.2,z],d.id.split('/').pop(),`${input.length} → ${out.length}`);return {a,b};}
function frame(x,y,z,w,h){const p=[[x-w/2,y-h/2,z],[x+w/2,y-h/2,z],[x+w/2,y+h/2,z],[x-w/2,y+h/2,z]];for(let i=0;i<4;i++)edge(p[i],p[(i+1)%4],0,'Граница модуля',0,'frame');}
function moduleDenses(m){return trace.dense.filter(d=>d.id.startsWith(m.id+'/'));}
function network(){const g=AtomicGraph(trace,token,Number($('density').value));nodes=g.nodes;edges=g.edges;labels=g.labels;total=g.total;camera.distance=g.width/(2*Math.tan(.29)*(innerWidth/innerHeight))*1.12;camera.target=[3,-3,0];$('explain').textContent='Единый residual stream. Фиолетовые связи — Δ прошлых слоёв текущего блока; золотые — суммарные Δ прошлых блоков. Все '+g.routes.length+' маршрутов H-DAR × '+trace.channels+' каналов сохранены. Синие узлы обозначают tensor operations, их линии — структурные зависимости. Кликните нейрон, чтобы проследить все его связи.';}
function productionScene(){
 if(mode==='dar'){dar();return;}
 if(mode==='memory'){const m=trace.modules[selected],g=trace.memories.find(g=>g.id===m.id);if(!g){$('explain').textContent='Выберите GDN-2.';return;}const h=Math.min(head,g.heads-1),a=g.states[h],mx=Math.max(1e-8,...a.flat().map(Math.abs));for(let i=0;i<a.length;i++)for(let j=0;j<a[i].length;j++)nodes.push({p:[(j/(a[i].length-1)-.5)*20,(.5-i/(a.length-1))*20,0],v:a[i][j],title:`H head ${h} [${i},${j}] · token ${g.token} · pooled region mean`,c:color(a[i][j],.2+.8*Math.sqrt(Math.abs(a[i][j])/mx))});heatmap(a,`FP32 serial reference · head ${h+1} · token ${g.token}`);label([0,12,0],'GDN STATE · '+m.label,'published checkpoint · '+g.dim+' × '+g.dim+(g.original_dim?' pooled from '+g.original_dim+' × '+g.original_dim:'' ));camera.distance=65;camera.target=[0,0,0];$('explain').textContent='FP32 reference-память из опубликованных весов. В компактном файле каждая клетка — среднее области 8×8 исходной матрицы. Сохранён последний токен захвата. Проверка совпадения с TPU bf16/fused пока отсутствует.';return;}
 if(mode==='attention'){const m=trace.modules[selected],a=trace.attention.find(g=>g.id===m.id);if(!a){$('explain').textContent='Выберите MLA.';return;}const h=Math.min(head,a.heads-1),row=a.attention[h][token],src=bank(row,-10,16,0,'Attention probability'),out=bank([row.reduce((a,b)=>a+b,0)],10,0,0,'Σ attention');src.forEach((n,i)=>{if(i<=token)edge(n,out[0],n.v,`A[${token},${i}]=${n.v}`,0,'local');});heatmap(a.attention[h],'Captured FP32 MLA causal attention · head '+(h+1));label([0,10,0],'MLA · '+m.label,'query token '+token);camera.distance=60;camera.target=[0,0,0];$('explain').textContent='Причинное внимание по всем захваченным GPT-2 токенам. Вероятности вычислены из весов опубликованного checkpoint.';return;}
 const g=ProductionGraph(trace,token,mode==='projection',selected,Number($('groups').value),expanded);projectionCount=g.projectionCount||0;nodes=g.nodes;edges=g.edges;labels=g.labels;total=g.total;camera.distance=Math.max(mode==='projection'?95:0,g.width/(2*Math.tan(.29)*(innerWidth/innerHeight))*1.12);camera.target=[0,-3,0];
 $('explain').textContent=mode==='projection'?`Индивидуальные каналы выбранного модуля. Точные xᵢWᵢⱼ; сохранены 24 сильнейших выхода и 8 сильнейших входов каждого. Детализация для токена ${trace.detail_token+1}; остальные связи не экспортированы.`:`Полные ${trace.metadata.config.num_blocks} блоков, ${trace.channels} реальных каналов. Узлы обзора — средние по ${Number($('groups').value)} группам. Все ${trace.routes.reduce((n,r)=>n+r.sources.length,0)} источников H-DAR сохранены. ${expanded?'Режим 40k+: добавлены реальные вклады '+projectionCount+' входных Dense-проекций; 40 000+ связей при плотности 100%.':'Обычный режим.'} Индивидуальные Dense-вклады зафиксированы на токене ${trace.detail_token+1}. Переключите «Проекции модуля» для отдельных нейронов.`;
}
function projection(){const m=trace.modules[selected],ds=moduleDenses(m),width=ds.length*5;ds.forEach((d,i)=>denseGraph(d,-width/2+i*5,(i%2?1:-1)*1.8,3.5,12));label([0,11,0],m.label,'Measured xᵢWᵢⱼ · bias is included in output, not edges');camera.distance=Math.max(85,width*1.9);camera.target=[-1,0,0];$('explain').textContent='Связь = xᵢ × Wᵢⱼ на выбранном токене. Цвет — знак, яркость — величина. Выход узла содержит сумму вкладов и bias. Q/K нормализация, RoPE и SwiGLU не изображаются как Dense-связи.';}
function dar(){const m=trace.modules[selected];const r=trace.routes.find(r=>r.target===m.stage);if(!r){$('explain').textContent='У первого слоя нет предшествующих DAR-источников.';camera.distance=40;return;}const target=bank(r.retrieved[token],9,15,0,'DAR retrieved'),res=bank(r.residual[token],14,15,0,'Residual'),sum=bank(r.residual[token].map((v,i)=>v+r.retrieved[token][i]),20,15,0,'Residual + retrieved');r.sources.forEach((id,s)=>{let vals=trace.stages.find(v=>v.id===id).values[token],z=(s-(r.sources.length-1)/2)*4.5;const b=bank(vals,-12,15,z,id);for(let c=0;c<b.length;c++){edge(b[c],target[c],vals[c]*r.weights[token][s],`DAR ${id}\n${trace.publication?'group mean':'channel'} ${c} · α=${r.weights[token][s].toPrecision(6)}\ncontribution=${(vals[c]*r.weights[token][s]).toPrecision(6)}`,0,'dar');}label([-12,9,z],id,'α = '+r.weights[token][s].toFixed(4));});for(let c=0;c<sum.length;c++){edge(target[c],sum[c],r.retrieved[token][c],'retrieved contribution',0,'retrieved');edge(res[c],sum[c],r.residual[token][c],'residual contribution',0,'residual');}label([9,9,0],'Retrieved Δ');label([14,9,0],'Residual');label([20,9,0],'Routed input');camera.distance=70;camera.target=[3,0,0];$('explain').textContent='Paper-DAR: learned query · RMSNorm keys → softmax по глубине. Линии связывают только одинаковые каналы: αₛΔₛ. Затем retrieved прибавляется к residual; нормализация идёт перед подслоем.';}
function memory(){const m=trace.modules[selected],g=trace.gdn.find(g=>g.id===m.id);if(!g){$('explain').textContent='Выберите модуль GDN-2 для просмотра его памяти.';return;}const h=Math.min(head,g.heads-1),state=g.states[token][h],dim=state.length;const q=bank(g.q[token][h],-13,14,0,'q'),k=bank(g.k[token][h],-9,14,-3,'k'),v=bank(g.write_value[token][h],-9,14,3,'w·v − erase'),read=bank(g.measured[token][h],13,14,0,'Measured kernel output');let max=Math.max(1e-6,...state.flat().map(Math.abs));for(let i=0;i<dim;i++)for(let j=0;j<state[i].length;j++){const p=[(j/(state[i].length-1)-.5)*8,(.5-i/(dim-1))*14,0],n={p,v:state[i][j],title:`H[${i},${j}] reference`,c:color(state[i][j],.3+.7*Math.abs(state[i][j])/max)};nodes.push(n);const contribution=g.q[token][h][i]/Math.sqrt(dim)*state[i][j];edge(q[i],n,contribution,`Read q[${i}]H[${i},${j}]/√d = ${contribution}`,0,'dense');edge(n,read[j],contribution,'Reference read contribution',0,'dense');total+=2;}
 label([0,9,0],'RECURRENT STATE','Hₜ = decay(Hₜ₋₁) + k ⊗ (w·v − erase)');label([-13,9,0],'Query');label([-9,9,-3],'Key');label([-9,9,3],'Write value');label([13,9,0],'Kernel output');camera.distance=65;camera.target=[0,0,0];heatmap(state,'GDN Hₜ · serial reference · max output error '+g.max_error.toExponential(2));$('explain').textContent='Матрица памяти восстановлена последовательной формулой; это не захват внутренних состояний fused-ядра. Справа — измеренный выход ядра. Ошибка сравнения сохранена. Key и write value показаны отдельными банками.';}
function attention(){const m=trace.modules[selected],a=trace.mla.find(g=>g.id===m.id);if(!a){$('explain').textContent='Выберите MLA для просмотра внимания.';return;}const h=Math.min(head,a.heads-1),weights=a.attention[h][token],dim=a.v[0][h].length,context=Array(dim).fill(0);for(let t=0;t<=token;t++)for(let d=0;d<dim;d++)context[d]+=weights[t]*a.v[t][h][d];const out=bank(context,12,14,0,'Attention context Σ A·V');for(let t=0;t<=token;t++){const z=(t-token/2)*1.8,vals=a.v[t][h],v=bank(vals,-10,14,z,'V token '+t);for(let d=0;d<dim;d++){edge(v[d],out[d],weights[t]*vals[d],`A[${token},${t}]V[${t},${d}] = ${weights[t]*vals[d]}`,0,'dense');total++;}label([-10,9,z],'Byte '+t,'α = '+weights[t].toFixed(4));}label([12,9,0],'ATTENTION CONTEXT','Σ Aₜₛ Vₛ');camera.distance=Math.max(65,token*2);camera.target=[0,0,0];heatmap(a.attention[h],'MLA causal attention · query '+token+' · max output error '+a.max_error.toExponential(2));$('explain').textContent='Линия связывает канал V предыдущего токена с тем же каналом контекста: AₜₛVₛ. Будущие токены исключены причинной маской. Q/K после RoPE определяют A; выход сверяется с моделью.';}
function heatmap(data,caption){const c=$('heat'),cx=c.getContext('2d');c.style.display='block';let max=Math.max(1e-8,...data.flat().map(Math.abs));cx.clearRect(0,0,256,256);data.forEach((row,i)=>row.forEach((v,j)=>{const col=color(v),s=Math.sqrt(Math.abs(v)/max);cx.fillStyle=`rgb(${col.slice(0,3).map(v=>Math.round(12+v*s*235)).join(',')})`;cx.fillRect(j*256/row.length,i*256/data.length,Math.ceil(256/row.length),Math.ceil(256/data.length));}));$('heatCaption').textContent=caption;}
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
 $('inspect').textContent=(n.structural?'Структурный узел операции.':'Значение записанного токена: '+n.v.toPrecision(7))+'\n'+connected.length+' сохранённых связей. '+(focused?'Закреплено · Esc снимает выбор.':'Наведение · клик закрепляет.')+'\n'+connected.slice().sort((a,b)=>Math.abs(b.v)-Math.abs(a.v)).slice(0,4).map(e=>e.info).join('\n');
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

function build(reset=true){if(!trace)return;const old=JSON.parse(JSON.stringify(camera)),selectedKey=focused?F.key(focused):null;previousNodeColors=lastNodeColors;previousEdgeColors=lastEdgeColors;focused=null;hover=null;frameTimes=[];edges=[];nodes=[];labels=[];total=0;document.body.classList.toggle('detail-view',mode!=='network');$('heat').style.display='none';$('heatCaption').textContent='';if(trace.version===4)productionScene();else ({network,projection,dar,memory,attention}[mode])();if(!reset)camera=old;if(trace.version===4){const dense=edges.filter(e=>e.kind==='dense').sort((a,b)=>Math.abs(b.v)-Math.abs(a.v)),keep=new Set(dense.slice(0,Math.ceil(dense.length*Number($('density').value)/100)));edges=edges.filter(e=>e.kind!=='dense'||keep.has(e));}focused=selectedKey?nodes.find(n=>F.key(n)===selectedKey)||null:null;uploadGeometry();setSelection(focused);transition=previousNodeColors.size?0:1;transitionStart=visualTime;pickDirty=true;$('labels').replaceChildren();for(const l of labels){const el=document.createElement('div');el.className='scene-label';const title=document.createElement('strong');title.textContent=l.title;const sub=document.createElement('span');sub.className='sub';sub.textContent=l.sub;el.append(title,sub);$('labels').append(el);l.el=el;}$('stats').textContent=`${trace.version===4?'HF · FP32 · ':''}${nodes.length.toLocaleString()} узлов · ${edges.filter(e=>e.kind!=='frame').length.toLocaleString()} связей${total?' · Dense: '+kept.toLocaleString()+' / '+total.toLocaleString():''} · реальные ${trace.channels} каналов`;$('tokenLabel').textContent=`${trace.version===4?'TOKEN':'BYTE'} ${token+1} / ${trace.tokens.length} · ${trace.tokens[token].label}`;$('token').value=token;[...$('tokens').children].forEach((b,i)=>b.classList.toggle('active',i===token));$('predictions').replaceChildren();const p=document.createElement('p');p.className='eyebrow';p.textContent=trace.version===4?'NEXT GPT-2 TOKEN / TOP 5':'NEXT BYTE / TOP 5';$('predictions').append(p);for(const t of trace.top[token]){const d=document.createElement('div');const a=document.createElement('span'),b=document.createElement('span');a.textContent=t.label.replaceAll('\n','⏎').replaceAll('\t','⇥').replaceAll(' ','␠');b.textContent=(t.prob*100).toFixed(2)+'%';d.append(a,b);$('predictions').append(d);}if(focused)inspectNode(focused);}
function load(data){if(![3,4].includes(data.version)||!data.modules?.length||(data.version===3&&!data.dense?.length))throw Error('Нужна полная трасса version 3');trace=data;normCache.clear();previousNodeColors.clear();previousEdgeColors.clear();lastNodeColors.clear();lastEdgeColors.clear();focused=null;if(document.body.dataset.replayOnly==='true'){$('text').readOnly=true;$('prompt').querySelector('button').textContent='Новый текст в Colab';}$('detail40').disabled=data.version!==4;if(data.version!==4)expanded=false;$('detail40').classList.toggle('active',expanded);$('detail40').setAttribute('aria-pressed',String(expanded));token=data.version===4?data.detail_token:0;selected=0;head=0;$('text').value=data.text;$('module').replaceChildren();data.modules.forEach((m,i)=>{const o=document.createElement('option');o.value=i;o.textContent=m.label;$('module').append(o);});$('token').max=data.tokens.length-1;$('tokens').replaceChildren();data.tokens.forEach((t,i)=>{const b=document.createElement('button');b.textContent=t.label;b.title=(data.version===4?'Token ':'Byte ')+i;b.onclick=()=>{token=i;build(false);};$('tokens').append(b);});$('badge').textContent=`${data.version===4?'HF FP32':'TINY'} · ${data.channels}D · ${data.modules.length} MODULES`;if(data.version===4){$('head').replaceChildren();for(let h=0;h<data.metadata.config.n_heads;h++){const o=document.createElement('option');o.value=h;o.textContent='Head '+(h+1);$('head').append(o);}}$('provenance').textContent=`Tiny checkpoint · ${(data.metadata.checkpoint_sha256||data.metadata.checkpoint?.sha256||'').slice(0,12)}. Dense error ${(data.checks.dense_max_error||data.checks.projection_max_error||0).toExponential(2)}. MLA error ${(data.checks.mla_max_error||0).toExponential(2)}. GDN reference error ${(data.checks.gdn_reference_max_error||0).toExponential(2)}. Частицы обозначают направление вклада; их скорость художественная, не время работы процессора.`;if(data.version===4)$('provenance').textContent=data.provenance+' · SHA256 '+data.metadata.checkpoint.sha256+' · '+data.seconds+' s';build();}
function drawBuffer(name,kind,point,particle,exposure,size){
 gl.bindBuffer(gl.ARRAY_BUFFER,buffers[name]);let off=0;
 for(const [k,n]of [['a',3],['b',3],['c',4],['d',4],['meta',4],['prior',4],['lut',4]]){gl.enableVertexAttribArray(loc[k]);gl.vertexAttribPointer(loc[k],n,gl.FLOAT,false,104,off*4);off+=n;}
 gl.uniform1f(loc.point,point);gl.uniform1f(loc.particle,particle);gl.uniform1f(loc.exposure,exposure);gl.uniform1f(loc.scale,size);gl.drawArrays(kind,0,counts[name]||0);
}
function animate(now){
 requestAnimationFrame(animate);const elapsed=now-last,dt=Math.min(.1,elapsed/1000||0);last=now;
 if(document.hidden||!trace)return;
 if(elapsed>0&&elapsed<10000){frameTimes.push(elapsed);if(frameTimes.length>180)frameTimes.shift();}
 moveCamera(dt);
 if(moving){visualTime+=dt;clock+=dt*Number($('speed').value)/65;}
 if(cameraGoal){const k=1-Math.exp(-dt*7);camera.target=camera.target.map((v,i)=>v+(cameraGoal.target[i]-v)*k);camera.distance+=(cameraGoal.distance-camera.distance)*k;pickDirty=true;if(Math.abs(camera.distance-cameraGoal.distance)<.03&&Math.hypot(...camera.target.map((v,i)=>v-cameraGoal.target[i]))<.03)cameraGoal=null;}
 if(!cameraGoal&&!drag&&geometryTier!==(camera.distance<150?1:0)){uploadGeometry();setSelection(focused||hover);}
 if(orbit&&moving){camera.yaw+=dt*.035;pickDirty=true;}
 if(play&&moving&&visualTime>nextToken){token=(token+1)%trace.tokens.length;build(false);nextToken=visualTime+1.6;}
 transition=moving?Math.min(1,(visualTime-transitionStart)/.42):1;const smooth=transition*transition*(3-2*transition);
 selectionFade+=(selectionGoal-selectionFade)*(1-Math.exp(-dt*16));if(selectionGoal===0&&selectionFade<.001){selectionFade=0;selectionId=-1;}
 const ratio=Math.min(1.5,devicePixelRatio),w=Math.round(innerWidth*ratio),h=Math.round(innerHeight*ratio);
 if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;pickDirty=true;}
 gl.viewport(0,0,w,h);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);gl.enable(gl.BLEND);matrix=view();if(pickDirty)updateScreenNodes();
 if(!focused&&!drag&&hoverX>=0&&now-pickStamp>65){pickStamp=now;const next=pickNode(hoverX,hoverY);if(next!==hover){hover=next;setSelection(next);inspectNode(next);}}
 gl.uniformMatrix4fv(loc.vp,false,matrix);gl.uniform1f(loc.time,clock);gl.uniform1f(loc.selectedId,selectionId);gl.uniform1f(loc.focusMix,selectionFade);gl.uniform1f(loc.transition,smooth);gl.uniform1f(loc.sway,Number($('sway').value)/40);gl.uniform1f(loc.blockLight,Number($('blockLight').value)/25);
 const exposure=Number($('glow').value)/55;
 gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);drawBuffer('line',gl.LINES,0,0,exposure,1);
 gl.blendFunc(gl.SRC_ALPHA,gl.ONE);drawBuffer('line',gl.LINES,0,0,exposure*.3,1);
 drawBuffer('node',gl.POINTS,1,0,.035*exposure,ratio*2.6);drawBuffer('node',gl.POINTS,1,0,.82,ratio*.85);
 if(signals){drawBuffer('pulse',gl.POINTS,1,1,exposure*.12,ratio*2);drawBuffer('pulse',gl.POINTS,1,1,exposure*.65,ratio*.65);}
 for(const l of labels){const p=project(l.p);l.el.style.transform=`translate(${p[0]}px,${p[1]}px)`;const major=(trace.version===4?trace.modules.some((m,i)=>m.label===l.title&&i%5===0):true)||l.title.startsWith('Δ BLOCK');l.el.style.display=p[2]>0&&p[0]>0&&p[0]<innerWidth&&p[1]>80&&p[1]<innerHeight-120&&!(mode==='network'&&camera.distance>140&&!major)?'':'none';}
 frames++;
 if(now-fpsStart>1000){
  $('fps').textContent=`${Math.round(frames*1000/(now-fpsStart))} FPS · WEBGL · ${w} × ${h}`;
  canvas.dataset.camera=JSON.stringify(camera);canvas.dataset.motion=JSON.stringify({moving,clock,transition:smooth,sway:Number($('sway').value),signals});
  if(frameTimes.length>30){const a=[...frameTimes].sort((a,b)=>a-b);canvas.dataset.performance=JSON.stringify({frames:a.length,median_ms:a[Math.floor(a.length*.5)],p95_ms:a[Math.floor(a.length*.95)],mean_fps:1000*a.length/a.reduce((a,b)=>a+b,0),nodes:nodes.length,edges:edges.length,viewport:[innerWidth,innerHeight]});}
  frames=0;fpsStart=now;
 }
}
canvas.onpointerdown=e=>{cameraGoal=null;drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,moved:false,button:e.button};canvas.setPointerCapture(e.pointerId);};
canvas.onpointerup=e=>{const clicked=drag&&!drag.moved&&drag.button===0;drag=null;if(clicked){const n=pickNode(e.clientX,e.clientY,12);focused=focused===n?null:n;hover=n;setSelection(focused||hover);inspectNode(focused||hover);if(focused){document.body.classList.add('info-open');}}};
canvas.onpointermove=e=>{
 hoverX=e.clientX;hoverY=e.clientY;
 if(drag){const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.moved=drag.moved||Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)>4;
  if(drag.moved){if(e.shiftKey||drag.button===1||drag.button===2)panCamera(-dx*camera.distance/1200,dy*camera.distance/1200,0);else{camera.yaw-=dx*.004;camera.pitch+=dy*.004;}pickDirty=true;}
  drag.x=e.clientX;drag.y=e.clientY;
 }
};
canvas.onpointerleave=()=>{hoverX=-1;hoverY=-1;hover=null;if(!focused)setSelection(null);};
canvas.onpointercancel=()=>{drag=null;};
canvas.addEventListener('wheel',e=>{e.preventDefault();cameraGoal=null;camera.distance=Math.max(.8,Math.min(1500,camera.distance*Math.exp(e.deltaY*.001)));pickDirty=true;},{passive:false});
canvas.ondblclick=e=>focusCamera(pickNode(e.clientX,e.clientY,15));
canvas.oncontextmenu=e=>e.preventDefault();
canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();$('message').textContent='Контекст WebGL потерян. Перезагрузите страницу.';});


$('mode').onchange=e=>{mode=e.target.value;if(mode==='memory')selected=trace.modules.findIndex(m=>m.kind==='gdn2');if(mode==='attention')selected=trace.modules.findIndex(m=>m.kind==='mla');$('module').value=selected;build();};$('module').onchange=e=>{selected=Number(e.target.value);build();if(mode==='network'){const id=trace.modules[selected].stage,n=nodes.find(n=>n.id===id+':'+Math.floor(trace.version===4?Number($('groups').value)/2:trace.channels/2));if(n){camera.target=[n.p[0]-6,0,0];camera.distance=52;}}};$('head').onchange=e=>{head=Number(e.target.value);build(false);};$('density').oninput=e=>{$('densityValue').textContent=e.target.value+'%';build(false);};$('token').oninput=e=>{token=Number(e.target.value);build(false);};$('reset').onclick=()=>{cameraGoal=null;camera.yaw=-.14;camera.pitch=.13;build();};$('signals').onclick=()=>{signals=!signals;$('signals').textContent='Сигналы: '+(signals?'вкл':'выкл');};$('orbit').onclick=()=>{orbit=!orbit;$('orbit').textContent='Облёт: '+(orbit?'вкл':'выкл');};$('play').onclick=()=>{play=!play;$('play').textContent=play?'Ⅱ Пауза':'▶ Токены';};$('info').onclick=()=>document.body.classList.toggle('info-open');$('controls').onclick=()=>document.body.classList.toggle('controls-open');$('cinema').onclick=()=>{document.body.classList.toggle('cinematic');$('cinema').textContent=document.body.classList.contains('cinematic')?'Панели':'Кино';};document.addEventListener('keydown',e=>{if(e.key==='Escape'){focused=null;hover=null;hoverX=-1;hoverY=-1;setSelection(null);document.body.classList.remove('cinematic');}});
$('export').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(trace)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='atomic-trace-v'+trace.version+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);};$('detail40').onclick=()=>{expanded=!expanded;const reset=mode!=='network';mode='network';$('mode').value=mode;$('detail40').classList.toggle('active',expanded);$('detail40').setAttribute('aria-pressed',String(expanded));build(reset);};$('groups').onchange=()=>build();$('import').onchange=async e=>{try{const file=e.target.files[0];const raw=file.name.endsWith('.gz')?await new Response(file.stream().pipeThrough(new DecompressionStream('gzip'))).text():await file.text();load(JSON.parse(raw));$('message').textContent='Трасса загружена';}catch(err){$('message').textContent=err.message;}};
$('prompt').onsubmit=async e=>{e.preventDefault();if(document.body.dataset.replayOnly==='true'){window.open(document.body.dataset.colabUrl,'_blank','noopener');return;}const b=e.target.querySelector('button');b.disabled=true;$('message').textContent='Вычисление модели… первый проход может занять несколько минут.';try{let d;if(trace.version===4&&!window.google?.colab?.kernel)throw Error('Этот захват опубликованного checkpoint воспроизводится в Colab. Новый текст задаётся в ноутбуке захвата.');if(window.google?.colab?.kernel){const r=await google.colab.kernel.invokeFunction(trace.version===4?'atomic.production.capture':'atomic.capture',[$('text').value],{});d=r.data['application/json'];}else{const r=await fetch('/api/trace',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:$('text').value})});if(!r.ok)throw Error(await r.text());d=await r.json();}load(d);$('message').textContent='Готово · '+d.seconds+' с';}catch(err){$('message').textContent='Не удалось вычислить: '+err.message;}finally{b.disabled=false;}};

const keys=new Set();
function panCamera(right,up,forward){const y=camera.yaw,p=camera.pitch,r=[Math.cos(y),0,-Math.sin(y)],f=[-Math.sin(y)*Math.cos(p),-Math.sin(p),-Math.cos(y)*Math.cos(p)],u=cross(r,f);camera.target=camera.target.map((v,i)=>v+r[i]*right+u[i]*up+f[i]*forward);}
function moveCamera(dt){if(!keys.size)return;cameraGoal=null;pickDirty=true;const speed=Math.max(2,camera.distance*.35)*dt*(keys.has('Shift')?3:1);panCamera(((keys.has('d')?1:0)-(keys.has('a')?1:0))*speed,((keys.has('e')?1:0)-(keys.has('q')?1:0))*speed,((keys.has('w')?1:0)-(keys.has('s')?1:0))*speed);}
window.addEventListener('keydown',e=>{if(/INPUT|TEXTAREA|SELECT/.test(e.target.tagName))return;const k=e.key==='Shift'?'Shift':e.key.toLowerCase();if(['w','a','s','d','q','e','Shift'].includes(k)){keys.add(k);e.preventDefault();}if(k==='f'&&(focused||hover)){focusCamera(focused||hover);}if(k==='home')$('reset').click();});
window.addEventListener('keyup',e=>keys.delete(e.key==='Shift'?'Shift':e.key.toLowerCase()));window.addEventListener('blur',()=>keys.clear());
const navigation=document.createElement('div');navigation.id='navigation';navigation.innerHTML='<button data-view="front">Спереди</button><button data-view="side">Сбоку</button><button data-view="top">Сверху</button><button data-view="back">Сзади</button><button data-view="focus">Фокус F</button><span>ЛКМ: вращать · ПКМ / Shift: сдвиг · колесо: зум · WASD: полёт · Q/E: высота · Shift: быстрее · Home: всё</span>';document.body.append(navigation);
navigation.onclick=e=>{const v=e.target.dataset.view;if(!v)return;if(v==='focus'){if(focused||hover){focusCamera(focused||hover);}return;}cameraGoal=null;pickDirty=true;[camera.yaw,camera.pitch]=({front:[0,0],side:[Math.PI/2,0],top:[0,Math.PI/2-.001],back:[Math.PI,0]})[v];};

$('motion').onclick=()=>{moving=!moving;$('motion').textContent='Движение: '+(moving?'вкл':'пауза');};$('near').onclick=()=>{const candidates=nodes.filter(n=>n.id?.includes('/selected y'));const target=focused||candidates.reduce((a,b)=>!a||Math.abs(b.p[0])+Math.abs(b.p[1])<Math.abs(a.p[0])+Math.abs(a.p[1])?b:a,null);focusCamera(target,125);};
window.addEventListener('error',e=>{$('message').textContent=e.message;});requestAnimationFrame(animate);(async()=>{try{let initial=window.ATOMIC_TRACE;if(!initial&&window.ATOMIC_TRACE_GZIP){const bytes=Uint8Array.from(atob(window.ATOMIC_TRACE_GZIP),c=>c.charCodeAt(0));initial=JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());}load(initial||await (await fetch('trace.json')).json());}catch(e){$('message').textContent=e.message;}})();
})();
