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

