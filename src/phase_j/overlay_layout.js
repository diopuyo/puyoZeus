'use strict';
const overlayOptions=OverlayOptions.resolve(window.overlayDefaults,location.search);
const FULL_HISTORY_LIMIT=2048, SECOND_MS=1000;
const NARROW_SCALE=83, PANEL_MARGIN=20, FULL_GRAPH_WIDTH=1000, FULL_GRAPH_HEIGHT=74, FULL_GRAPH_PAD=4;
let fullHistory=[],fullStart=null;
/** @param {object} options */
function applyOverlayOptions(options) {
  const stage=el('stage'),panel=el('panel'),band=el('band');
  const layouts={center:'side',card:'ribbon',ribbon:'side',minimal:'mini',vertical:'side',l:'l','l-left':'l','l-right':'l'};
  stage.dataset.layout=options.layout.startsWith('l-')?'l':layouts[options.layout]||'ribbon';
  stage.dataset.theme=options.theme==='pop-cream'?'pop':options.theme;
  stage.dataset.pop=options.theme==='pop-cream'?'cream':'black';
  stage.dataset.place=options.layout==='center'?'center':'edge';
  stage.dataset.vgraph=['center','vertical'].includes(options.layout)?'1':'0';stage.dataset.state='wait';
  stage.dataset.midline=options.midline;stage.dataset.font=options.font;
  if(options.theme==='stylish') stage.dataset.flat='light';
  panel.dataset.layout=options.layout;stage.dataset.mascot=String(options.mascot);
  const isL=stage.dataset.layout==='l',g=OverlayOptions.geometry(options);
  document.querySelector('.scores').hidden=!options.numbers;
  document.querySelector('.trend').hidden=!options.recent;
  el('details').hidden=true;el('whole').hidden=!isL||!options.full;
  band.hidden=!isL||!options.band;band.append(el('whole'));
  if(isL) {
    Object.assign(panel.style,{left:`${g.columnX}px`,top:`${g.y}px`,width:`${g.columnWidth}px`,height:`${g.height}px`});
    Object.assign(band.style,{top:`${g.bandY}px`,height:`${g.bandHeight}px`});
  } else {
    for(const property of ['left','top','width','height']) panel.style[property]='';
    if(options.x!==undefined) panel.style.left=`${options.x}px`;
    if(options.y!==undefined) panel.style.top=`${options.y}px`;
  }
}

/** @returns {void} */
function clearFullHistory() {
  fullHistory=[];fullStart=null;el('whole-trace').setAttribute('d','');
  el('whole-time').textContent='受信開始から';
}
/** @param {number} p @param {number} time */
function updateFullHistory(p,time) {
  if(!Number.isFinite(time)) return;
  if(fullStart===null) fullStart=time;
  if(fullHistory.length&&fullHistory.at(-1).time===time) fullHistory.pop();
  fullHistory.push({p,time});
  // 長時間でも開始点を残し、保持数を制限する。直近30秒の系列とは独立。
  if(fullHistory.length>FULL_HISTORY_LIMIT) fullHistory=fullHistory.filter((_,i)=>i%2===0||i===fullHistory.length-1);
  const duration=Math.max(SECOND_MS,time-fullStart);
  const x=time=>FULL_GRAPH_PAD+(time-fullStart)/duration*(FULL_GRAPH_WIDTH-2*FULL_GRAPH_PAD);
  const y=p=>FULL_GRAPH_PAD+(1-p)*(FULL_GRAPH_HEIGHT-2*FULL_GRAPH_PAD);
  const path=fullHistory.map((item,i)=>`${i?'L':'M'}${x(item.time).toFixed(1)} ${y(item.p).toFixed(1)}`);
  el('whole-trace').setAttribute('d',path.join(' '));
  el('whole-time').textContent=`受信開始 0秒 → ${Math.round((time-fullStart)/SECOND_MS)}秒`;
}
applyOverlayOptions(overlayOptions);
