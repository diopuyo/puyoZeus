'use strict';
// 試合内の最大の振れを保持し、表示座標だけを補間する。
window.PuyoGraphScale=(()=>{
  const MIN_HALF=.10, MIN_MARGIN=.03, MARGIN_RATE=.15, FULL_HALF=.5, DURATION_MS=300;
  let peak=0,from=MIN_HALF,target=MIN_HALF,start=0,frame=0;
  /** @returns {number} */
  function half() {
    const progress=Math.min(1,Math.max(0,(performance.now()-start)/DURATION_MS));
    return from+(target-from)*progress;
  }
  /** @returns {void} */
  function redraw() {
    frame=0;PuyoEvaluation.refresh();
    if(half()<target) frame=requestAnimationFrame(redraw);
  }
  /** @param {number} p @returns {void} */
  function observe(p) {
    if(!Number.isFinite(p)) return;
    peak=Math.max(peak,Math.abs(p-FULL_HALF));
    const next=Math.min(FULL_HALF,Math.max(MIN_HALF,peak+Math.max(MIN_MARGIN,peak*MARGIN_RATE)));
    if(next<=target) return;
    from=half();target=next;start=performance.now();
    if(!frame) frame=requestAnimationFrame(redraw);
  }
  /** @returns {void} */
  function reset() {
    cancelAnimationFrame(frame);frame=0;peak=0;from=target=MIN_HALF;start=0;
  }
  /** @param {number} p @returns {number} */
  function graph(p) {
    const evaluation=PuyoEvaluation.isEvaluation(),fixed=!overlayOptions.graph_autoscale;
    const width=fixed?FULL_HALF:half();
    const limit=evaluation?(fixed?PuyoEvaluation.GRAPH_LIMIT:
      Math.min(PuyoEvaluation.GRAPH_LIMIT,PuyoEvaluation.value(FULL_HALF+width))):width;
    const distance=evaluation?PuyoEvaluation.value(p):p-FULL_HALF;
    return FULL_HALF+Math.max(-limit,Math.min(limit,distance))/(2*limit);
  }
  return {observe,reset,graph,half};
})();
// 公開された勝率を表示時だけ換算する。判定・平滑化には戻さない。
window.PuyoEvaluation=(()=>{
  const SCALE=600, LIMIT=9999, GRAPH_LIMIT=3000, NEUTRAL_LIMIT=100;
  /** @param {number} p @returns {number} */
  function value(p) {
    if(!Number.isFinite(p)) return 0;
    const distance=Math.abs(p-0.5),low=0.5-distance;
    const magnitude=low<=0?LIMIT:Math.min(LIMIT,Math.round(SCALE*Math.log((1-low)/low)));
    return magnitude===0?0:(p<0.5?-magnitude:magnitude);
  }
  /** @param {string} mode @returns {boolean} */
  function isEvaluation(mode=overlayOptions.value_display) {
    return mode==='evaluation';
  }
  /** @param {number} p @returns {number} */
  function graph(p) {
    return PuyoGraphScale.graph(p);
  }
  /** @param {number} p @returns {void} */
  function paint(p) {
    const score=value(p),evaluation=isEvaluation();
    for(const [id,percent] of [['p1',Math.round(p*100)],['p2',100-Math.round(p*100)]]) {
      el(id).textContent=String(percent);
      el(id).closest('.plate').dataset.len=String(el(id).textContent.length);
    }
    const text=score>0?`+${score}`:String(score);
    el('evaluation-value').textContent=text;
    el('evaluation').style.setProperty('--evaluation-digits',String(text.length));
    el('evaluation').dataset.advantage=Math.abs(score)<=NEUTRAL_LIMIT?'neutral':score>0?'one':'two';
    if(evaluation) el('bar').setAttribute('aria-label',`1P評価値 ${score}`);
  }
  /** @returns {void} */
  function refresh() {
    if(isProbability(shown)) paint(shown);
    window.PuyoBattle?.redraw();
    const last=fullHistory.at(-1);
    if(last) updateFullHistory(last.p,last.time);
  }
  return {value,graph,paint,refresh,isEvaluation,SCALE,LIMIT,GRAPH_LIMIT,NEUTRAL_LIMIT};
})();
