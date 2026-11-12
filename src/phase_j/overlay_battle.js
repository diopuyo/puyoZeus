'use strict';
// 値・履歴は既存の描画関数から受け取り、見た目だけを切り替える。
window.PuyoBattle=(()=>{
  const HOLD_DELAY=600,HOLD_MIN=800,WINDOW_MS=30000,SAMPLE_LIMIT=4096,DOT_RADIUS=6;
  const BAND_MARGIN=28,CENTER_GAP=8,CENTER_SIZE=120,RESULT_LEFT=200,RESULT_WIDTH=150;
  const stage=el('stage'),panel=el('panel'),mascot=el('mascot');
  const speech=document.createElement('div'),tab=document.createElement('div');
  let holdTimer=null,holdSince=null,heldAt=null,desiredHold=false;
  let samples=[];
  speech.id='battle-speech';speech.className='bubble';mascot.append(speech);
  tab.id='result-tab';tab.className='tab';tab.innerHTML='<div class="tab-in"><span id="result-text"></span></div>';
  panel.append(tab);
  document.querySelectorAll('.player').forEach(node=>{
    for(const name of ['psh','pbg','pmute','dash']) {
      const span=document.createElement('span');span.className=name;span.setAttribute('aria-hidden','true');node.prepend(span);
    }
  });
  const veil=document.createElement('div');veil.className='g-veil';el('bar').append(veil);
  const split=document.createElement('div');split.id='gSplit';split.className='g-split';
  split.innerHTML='<i class="solid"></i><i class="hollow"></i>';el('bar').append(split);
  const base=document.createElement('div');base.className='mbase';mascot.prepend(base);
  new ResizeObserver(()=>{
    const graph=el('trend');
    if(!graph.clientWidth||!graph.clientHeight) return;
    el('tip').setAttribute('rx',String(DOT_RADIUS*GRAPH_WIDTH/graph.clientWidth));
    el('tip').setAttribute('ry',String(DOT_RADIUS*GRAPH_HEIGHT/graph.clientHeight));
  }).observe(el('trend'));
  /** @returns {boolean} */
  function placeMascot() {
    const layout=stage.dataset.layout;
    const boxes={ribbon:[184,-130,132],mini:[184,-90,92],side:[48,648,128]};
    let [x,y,size]=boxes[layout]||[30,892,160];
    if(stage.dataset.place==='center') size=CENTER_SIZE;
    size=({small:96,large:224})[overlayOptions.mascot_size]||size;
    if(layout!=='l') {
      x=panel.offsetLeft+(panel.offsetWidth-size)/2;
      y=layout==='side'?panel.offsetTop+panel.offsetHeight+(stage.dataset.place==='center'?CENTER_GAP:BAND_MARGIN):panel.offsetTop-size+2;
    }
    else {
      const g=OverlayOptions.geometry(overlayOptions);
      size=Math.min(size,g.bandHeight-2*BAND_MARGIN);
      y=g.bandY+(g.bandHeight-size)/2;
      Object.assign(tab.style,{top:`${g.bandY-panel.offsetTop+BAND_MARGIN}px`,bottom:'auto',
        left:`${Math.min(RESULT_LEFT,g.columnWidth-RESULT_WIDTH)}px`});
    }
    Object.assign(mascot.style,{left:`${x}px`,top:`${y}px`,width:`${size}px`,height:`${size}px`});
    mascot.querySelector('img').style.scale=overlayOptions.mascot_flip_right&&x>960?'-1 1':'1 1';
    return true;
  }
  /** @returns {void} */
  function resetHold() {
    clearTimeout(holdTimer);holdTimer=null;holdSince=null;heldAt=null;desiredHold=false;
  }
  /** @returns {void} */
  function settleHold() {
    clearTimeout(holdTimer);const now=performance.now();
    if(desiredHold) {
      const remaining=HOLD_DELAY-(now-holdSince);
      if(remaining>0) {holdTimer=setTimeout(settleHold,remaining);return;}
      if(heldAt===null) heldAt=now;
      stage.dataset.state='hold';
    } else {
      const remaining=heldAt===null?0:HOLD_MIN-(now-heldAt);
      if(remaining>0) {holdTimer=setTimeout(settleHold,remaining);return;}
      heldAt=null;stage.dataset.state='live';
    }
  }
  /** @param {number} p @param {number} time */
  function record(p,time) {
    if(!Number.isFinite(time)) return;
    PuyoGraphScale.observe(p);
    if(samples.at(-1)?.time===time) samples.pop();
    samples.push({p,time});
    while(samples.length>1&&samples[1].time<=time-WINDOW_MS) samples.shift();
    samples=samples.slice(-SAMPLE_LIMIT);
    const recent=samples.filter(item=>item.time>=time-WINDOW_MS);
    const vertical=stage.dataset.vgraph==='1';
    const x=t=>GRAPH_PAD+(1-(time-t)/WINDOW_MS)*(GRAPH_WIDTH-2*GRAPH_PAD);
    const y=v=>GRAPH_PAD+(1-PuyoEvaluation.graph(v))*(GRAPH_HEIGHT-2*GRAPH_PAD);
    drawHistory(recent,time,vertical,x,y);
  }
  /** @param {object[]} points @param {number} time @param {boolean} vertical @param {Function} x @param {Function} y */
  function drawHistory(points,time,vertical,x,y) {
    // 縦向きでは表示中の履歴を枠の上端から下端まで配置する。
    const timeSpan=vertical&&points.length>1?time-points[0].time:WINDOW_MS;
    const xy=points.map(item=>vertical?
      [GRAPH_PAD+PuyoEvaluation.graph(item.p)*(GRAPH_WIDTH-2*GRAPH_PAD),(time-item.time)/timeSpan*GRAPH_HEIGHT]:[x(item.time),y(item.p)]);
    const line=xy.map(([a,b],i)=>`${i?'L':'M'}${a.toFixed(1)} ${b.toFixed(1)}`).join(' ');
    el('trace').setAttribute('d',line);
    el('trend').querySelector('.guide').setAttribute('d',vertical?`M${GRAPH_WIDTH/2} 0V${GRAPH_HEIGHT}`:`M0 ${GRAPH_HEIGHT/2}H${GRAPH_WIDTH}`);
    el('trend').setAttribute('aria-label',vertical?'1P勝率の推移。右ほど1P有利、下ほど過去、中央線は50%':'1P勝率の推移。上ほど1P有利、中央線は50%');
    if(PuyoEvaluation.isEvaluation()) el('trend').setAttribute('aria-label',
      vertical?'1P評価値の推移。右ほど1P有利、下ほど過去、中央は0、両端は±3000':'1P評価値の推移。上ほど1P有利、中央は0、両端は±3000');
    for(const [side,edge] of [['one',0],['two',GRAPH_WIDTH]]) {
      let area=el('trend').querySelector('.history-'+side);
      if(!area) {area=document.createElementNS('http://www.w3.org/2000/svg','path');area.setAttribute('class','history-'+side);el('trend').prepend(area);}
      area.setAttribute('d',vertical&&xy.length?`M${edge} ${xy[0][1]} ${line.replace(/^M/,'L')} L${edge} ${xy.at(-1)[1]}Z`:'');
    }
    if(xy.length) {el('tip').setAttribute('cx',xy.at(-1)[0]);el('tip').setAttribute('cy',xy.at(-1)[1]);}
  }
  /** @param {object} data @returns {void} */
  function update(data={}) {
    const empty=panel.dataset.empty==='true',winner=stage.dataset.winner;
    for(const id of ['p1','p2']) el(id).closest('.plate').dataset.len=String(el(id).textContent.length);
    const waiting=empty&&['試合待ち',PuyoI18n.t('試合待ち')].includes(el('state').textContent);
    speech.textContent=waiting?'次の試合まだかな':'';
    speech.classList.toggle('on',waiting);speech.hidden=!waiting;
    tab.classList.toggle('on',Boolean(winner));
    el('result-text').textContent=winner==='1P'?'◀ WIN!':winner==='2P'?'WIN! ▶':'';
    if(empty||winner) {
      resetHold();stage.dataset.state=panel.dataset.mode==='offline'?'offline':empty?'wait':'result';
    } else {
      desiredHold=panel.dataset.mode==='waiting';
      if(desiredHold&&holdSince===null) holdSince=performance.now();
      if(!desiredHold) holdSince=null;
      settleHold();
    }
    if(!empty) record(shown,data.timing?.source_available_ms);
    else el('trend').querySelectorAll('.history-one,.history-two').forEach(area=>area.setAttribute('d',''));
    placeMascot();
  }
  // 旧リボンと同じく数字は公開値を即時表示。帯だけCSSで補間する。
  animateProbability=(p,immediate=false)=>{cancelAnimationFrame(animation);paintProbability(p);};
  const reset=resetThemeMotion;resetThemeMotion=()=>{reset();samples=[];};
  const render=renderSnapshot;renderSnapshot=data=>{render(data);update(data);};
  const absent=unavailable;unavailable=(...args)=>{absent(...args);update();};
  const sync=syncMatch;syncMatch=data=>{
    const key=JSON.stringify([data.identity?.session_id,data.identity?.match_id]);
    if(key!==matchKey||data.timing?.source_available_ms<lastSourceMs) resetHold();
    sync(data);
  };
  /** @returns {void} */
  function redraw() {
    const last=samples.at(-1);
    if(last) record(last.p,last.time);
  }
  update();return {placeMascot,redraw};
})();
