'use strict';
// 数字を描く経路を観測し、入力DTOや勝率の計算は変更しない。
window.PuyoExpressions=(()=>{
  const HOLD_MS=1500, BANDS=[10,30,50,70,90], PERCENT=100;
  const startedAt=(window.overlayDefaults?.app_started_at||Date.now()/1000)*1000;
  let current=null,changed=-Infinity,timer=null,target='50';
  /** @param {number} p @returns {string} */
  function band(p) {return String(BANDS[overlayOptions.mascot_band_edges.filter(edge=>p>=edge/PERCENT).length]);}
  /** @param {string} value @returns {void} */
  function show(value) {
    const node=el('mascot'),img=node.querySelector('img');
    const same=current===value;
    current=value;if(!same) changed=performance.now();node.dataset.expression=value;
    const file={win:'90',lose:'10'}[value]||value;
    let name=value==='off'?'overlay':file;
    if(['70','90'].includes(name)&&overlayOptions.mascot_dopamine!=='normal') name+='_'+overlayOptions.mascot_dopamine;
    const minutes=(Date.now()-startedAt)/60000;
    const stage=overlayOptions.mascot_bloodshot?[30,45,60].filter(t=>minutes>=t).length:0;
    if(stage) name+='_b'+stage;
    if(overlayOptions.mascot_speed!=='1') name+='_s'+overlayOptions.mascot_speed;
    // 目を止める設定だけ静止素材を使い、デザインによって動きを変えない。
    const source=!overlayOptions.mascot_eyes?
      '/mascot/battle-still.webp':`/mascot/${name}.webp`;
    // 同じ顔のまま決着した場合、アニメーションを先頭に戻さない。
    if(img.getAttribute('src')!==source) img.src=source;
  }
  /** @param {boolean} immediate @returns {void} */
  function settle(immediate=false) {
    clearTimeout(timer);timer=null;
    const remaining=HOLD_MS-(performance.now()-changed);
    if(immediate||remaining<=0) show(target);
    else if(current!==target) timer=setTimeout(()=>settle(),remaining);
  }
  /** @returns {void} */
  function update() {
    const enabled=overlayOptions.mascot&&overlayOptions.mascot_expressions;
    const available=el('panel').dataset.empty==='false'&&isProbability(shown);
    const winner=el('stage').dataset.winner;
    target=!enabled?'off':!available?'50':winner?(winner==='1P'?'win':'lose'):band(shown);
    // 待機・決着は通常の帯とは別状態。試合境界で前試合の表情を残さない。
    settle(!enabled||!available||Boolean(winner));
  }
  const paint=paintProbability;paintProbability=p=>{paint(p);update();};
  if(overlayOptions.mascot_bloodshot) setInterval(()=>show(current||'off'),1000);
  return {update,band};
})();
