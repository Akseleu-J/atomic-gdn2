/* Pure rendering helpers. Geometry never changes trace values or graph membership. */
(function(root){
 const key=n=>n.id||n.title;
 const bank=n=>key(n).replace(/:\d+$|\s*\[\d+(?:,\d+)?\].*$/,'');
 const signed=(v,a)=>v>=0?[1,.055,.035,a]:[.025,1,.12,a];
 function hash(s){let h=2166136261;for(let i=0;i<s.length;i++)h=Math.imul(h^s.charCodeAt(i),16777619);return(h>>>0)/4294967296;}
 function blockEdge(e){return e.kind==='history'||e.kind==='summary'||/block_\d+_summary/.test((e.source||'')+' '+(e.target||''))||/Block end contribution|Subtract block start/.test(e.info||'');}
 function curve(e){
  if(e.kind==='frame')return [0,0,0,0];
  const id=(e.source||e.a.join(','))+'>'+(e.target||e.b.join(','))+e.kind;
  const h=hash(id),v=hash(id+'y'),length=Math.hypot(...e.a.map((x,i)=>e.b[i]-x));
  return [(h-.5)*Math.min(9,length*.36),(e.arc||0)*.5+(v-.5)*Math.min(5,length*.28),h*6.283185,Math.min(.38,length*.025)*(blockEdge(e)?.35:1)];
 }
 function sample(e,c,t,time=0,sway=1){
  const envelope=Math.sin(Math.PI*t),w=Math.sin(time*.63+c[2]+t*4.2)*c[3]*sway;
  return e.a.map((v,i)=>v+(e.b[i]-v)*t+(i===1?envelope*(c[1]+w*.7):i===2?envelope*(c[0]+w):0));
 }
 function arc(e,c){
  let previous=e.a,length=0,check=[];
  for(let j=1;j<=32;j++){const p=sample(e,c,j/32,0,0);length+=Math.hypot(...p.map((x,i)=>x-previous[i]));previous=p;if(j%8===0)check.push(length);}
  return [...check.slice(0,3).map(v=>v/Math.max(length,1e-8)),Math.max(length,.001)];
 }
 function palette(e,max=1){
  // Golden block routes retain their independent full-light default.
  if(blockEdge(e))return [1,.76,.025,e.kind==='history'?.0012:.0025];
  if(e.kind==='frame')return [.37,.5,.7,.02];
  if(['dar','local'].includes(e.kind))return [.68,.22,1,.07];
  if(e.kind==='operation')return [.64,.79,1,.008];
  if(['residual','retrieved'].includes(e.kind))return [.5,1,.78,.065];
  return signed(e.v,.05+.22*Math.pow(Math.min(1,Math.abs(e.v)/Math.max(max,1e-8)),.38));
 }
 function nodeColor(n,scale){return n.structural?[.65,.8,1,.38]:signed(n.v,.08+.82*Math.sqrt(Math.min(1,Math.abs(n.v)/Math.max(scale,1e-8))));}
 const api={key,bank,hash,blockEdge,curve,sample,arc,palette,nodeColor};root.Filaments=api;if(typeof module==='object'&&module.exports)module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
