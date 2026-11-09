'use strict';
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
    return isEvaluation()?
      0.5+Math.max(-GRAPH_LIMIT,Math.min(GRAPH_LIMIT,value(p)))/(2*GRAPH_LIMIT):p;
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
