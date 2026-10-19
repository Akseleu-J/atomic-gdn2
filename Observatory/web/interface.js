/* Panel behaviour is isolated from the scientific viewer and its rendering. */
(()=>{
 const byId=id=>document.getElementById(id);
 const panels=[['controls','settings','controls-open','closeSettings'],['info','inspector','info-open','closeInspector']];
 for(const [button,panel,state,close]of panels){
  byId(close).onclick=()=>{document.body.classList.remove(state);byId(button).focus();};
  // Space is limited on phones: opening one panel dismisses the other.
  byId(button).addEventListener('click',()=>{if(innerWidth<=640&&document.body.classList.contains(state))for(const other of panels)if(other[2]!==state)document.body.classList.remove(other[2]);});
 }
 const sync=()=>{for(const [button,panel,state]of panels){byId(button).setAttribute('aria-expanded',String(document.body.classList.contains(state)&&!document.body.classList.contains('cinematic')));}};
 new MutationObserver(sync).observe(document.body,{attributes:true,attributeFilter:['class']});sync();
 document.addEventListener('keydown',event=>{if(event.key==='Escape'){document.body.classList.remove('controls-open','info-open');sync();}});
 if(matchMedia('(prefers-reduced-motion: reduce)').matches){byId('sway').value='0';byId('motion').click();}
})();
