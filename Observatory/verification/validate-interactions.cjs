/* Runs the actual application event handlers with a minimal DOM/WebGL adapter.
 * This verifies state transitions, not GPU output or browser performance. */
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
class Element{
 constructor(id=''){this.id=id;this.children=[];this.style={};this.dataset={};this.value='';this.textContent='';this.listeners={};const values=new Set();this.classList={add:x=>values.add(x),remove:x=>values.delete(x),contains:x=>values.has(x),toggle:(x,on)=>{if(on===undefined)on=!values.has(x);on?values.add(x):values.delete(x);return on;}};}
 append(...children){this.children.push(...children);}
 replaceChildren(...children){this.children=[...children];}
 querySelector(){return this.button||(this.button=new Element());}
 setAttribute(k,v){this[k]=v;}
 addEventListener(k,f){(this.listeners[k]??=[]).push(f);}
 setPointerCapture(){}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
for(const [id,value]of Object.entries({mode:'network',groups:'32',density:'100',glow:'55',sway:'40',blockLight:'100',speed:'65',module:'0',head:'0'}))get(id).value=value;
const uniforms={},draws=[];let attrib=0;const gl=new Proxy({}, {get:(_,name)=>{
 if(name==='getAttribLocation')return()=>attrib++;
 if(name==='getUniformLocation')return(_,s)=>s;
 if(name==='uniform1f')return(k,v)=>{uniforms[k]=v;};
 if(name==='drawArrays')return()=>draws.push({...uniforms});
 if(name==='getShaderParameter'||name==='getProgramParameter')return()=>true;
 if(name.startsWith('create'))return()=>({});
 if(name.toUpperCase()===name)return 1;
 return()=>{};
}});
get('scene').getContext=()=>gl;
get('heat').getContext=()=>({clearRect(){},fillRect(){}});
const body=new Element();body.dataset={replayOnly:'true',colabUrl:'https://colab.research.google.com/github/Akseleu-J/atomic-gdn2/blob/main/Observatory/notebooks/Atomic-Production-Capture.ipynb'};
const handlers={},document={body,hidden:false,getElementById:get,createElement:()=>new Element(),addEventListener:(k,f)=>(handlers[k]??=[]).push(f)};
let frame,now=0;const ctx={document,innerWidth:1280,innerHeight:720,devicePixelRatio:1,requestAnimationFrame:f=>{frame=f;},console,Float32Array,Uint8Array,Math,Map,Set,JSON,Blob,URL,setTimeout,addEventListener(){}};
ctx.window=ctx;ctx.ATOMIC_TRACE=JSON.parse(fs.readFileSync(path.join(__dirname,'../web/trace.json'),'utf8'));vm.createContext(ctx);
for(const file of ['production.js','topology.js','filaments.js','app.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,'../web',file),'utf8'),ctx,{filename:file});
function tick(count=40){for(let i=0;i<count;i++){now+=16.667;frame(now);}}
tick();assert.equal(get('message').textContent,'');
const html=fs.readFileSync(path.join(__dirname,'../web/index.html'),'utf8');
assert.match(html, /id="blockLight"[^>]*value="100"/);
const effectiveBlockLight=()=>uniforms.blockLight*uniforms.exposure/.65;
const maximum=4*150/55;
assert(Math.abs(effectiveBlockLight()-maximum)<1e-10);
const referencePasses=[150/55,150/55*.3,.035*150/55,.82,150/55*.12,150/55*.65].map(v=>4*v);
for(const glow of ['10','55','150']){
 get('glow').value=glow;tick(1);assert(Math.abs(effectiveBlockLight()-maximum)<1e-10);
 draws.slice(-6).forEach((pass,i)=>assert(Math.abs(pass.blockLight*pass.exposure-referencePasses[i])<1e-10));
}
get('blockLight').value='25';tick(1);assert(Math.abs(effectiveBlockLight()-maximum/4)<1e-10);
get('blockLight').value='0';tick(1);assert.equal(effectiveBlockLight(),0);
get('blockLight').value='100';get('glow').value='55';tick(1);
const probe=JSON.parse(get('scene').dataset.pickTarget),event={clientX:probe.x,clientY:probe.y,button:0,pointerId:1};
get('scene').onpointermove(event);tick();const hoverId=get('scene').dataset.selection;
assert(hoverId);assert.equal(get('scene').dataset.pinned,'false');assert(uniforms.focusMix>.95);
get('scene').onpointerdown(event);get('scene').onpointerup(event);tick();assert.equal(get('scene').dataset.pinned,'true');
const pinned=get('scene').dataset.selection;
get('tokens').children[0].onclick();tick();assert.equal(get('scene').dataset.selection,pinned);assert.equal(get('scene').dataset.pinned,'true');assert.match(get('tokenLabel').textContent,/TOKEN 1/);
for(const h of handlers.keydown||[])h({key:'Escape'});tick();assert.equal(get('scene').dataset.selection,'');assert.equal(uniforms.selectedId,-1);
get('motion').onclick();tick();const paused=uniforms.time;tick(100);assert.equal(uniforms.time,paused);
get('motion').onclick();tick();assert(uniforms.time>paused);
get('scene').onpointermove(event);tick();get('scene').onpointerleave();tick();assert.equal(get('scene').dataset.selection,'');
get('near').onclick();tick(100);assert(get('scene').dataset.lineVertices);assert(Number(get('scene').dataset.lineVertices)>0);
const result={passed:true,block_delta_default_maximum:true,block_delta_independent_of_scene_glow:true,block_delta_manual_dimming:true,hover_white_fan_state:true,click_pins:true,pin_survives_token:true,escape_clears:true,pointer_leave_clears_hover:true,motion_pause_freezes_clock:true,scope:'Actual app handlers with mock DOM/WebGL; GPU appearance checked separately in Chrome'};
fs.writeFileSync(path.join(__dirname,'validation-interactions.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
