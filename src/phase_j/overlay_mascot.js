'use strict';
// 全デザイン共通。目と表情は画像、運動はドーパミンの帯だけで決める。
window.PuyoMascot=(()=>{
  const SIZES={small:96,medium:160,large:224};
  const WIDTH=OverlayOptions.WIDTH,HEIGHT=OverlayOptions.HEIGHT,MARGIN=24,GAP=16;
  const TENSION={
    normal:{'70':{lift:4,tilt:3,scale:1.04,duration:900},'90':{lift:8,tilt:6,scale:1.08,duration:720}},
    high:{'70':{lift:7,tilt:5,scale:1.07,duration:620},'90':{lift:12,tilt:9,scale:1.12,duration:480}}
  };
  let tension=null,tensionLevel=null;
  const node=document.createElement('div'),img=document.createElement('img');
  node.id='mascot';node.style.cssText='position:absolute;z-index:20;pointer-events:none';
  img.style.cssText='width:100%;height:100%;object-fit:contain';node.append(img);el('stage').append(node);
  img.src='/mascot/overlay.webp';img.alt=PuyoI18n.t('ゼウスくん');
  /** @returns {void} */
  function resetMotion() {
    tension?.cancel();tension=null;tensionLevel=null;
    node.getAnimations().forEach(animation=>animation.cancel());
  }
  /** @returns {void} */
  function updateTension() {
    const available=el('panel').dataset.empty==='false'&&isProbability(shown);
    const winner=el('stage').dataset.winner;
    const band=winner?(winner==='1P'?'90':null):window.PuyoExpressions?.band(shown);
    const enabled=overlayOptions.mascot&&overlayOptions.mascot_motion&&!reducedMotion.matches;
    const level=enabled&&available?TENSION[overlayOptions.mascot_dopamine]?.[band]||null:null;
    if(!enabled||!available) resetMotion();
    if(level===tensionLevel) return;
    tension?.cancel();tension=null;tensionLevel=level;
    if(!level) return;
    const {lift,tilt,scale,duration}=level;
    // 配置・左右反転を維持したまま、画像だけを動かす。
    img.style.transformOrigin='50% 100%';
    const rest='translateY(0) rotate(0deg) scale(1)';
    tension=img.animate([
      {transform:rest},
      {transform:`translateY(-${lift}%) rotate(-${tilt}deg) scale(${scale})`},
      {transform:rest},
      {transform:`translateY(-${lift}%) rotate(${tilt}deg) scale(${scale})`},
      {transform:rest}
    ],{duration,iterations:Infinity,easing:'ease-in-out'});
  }
  /** @returns {void} */
  function place() {
    if(window.PuyoBattle?.placeMascot()) return;
    let size=SIZES[overlayOptions.mascot_size];const panel=el('panel');
    const isL=overlayOptions.layout.startsWith('l');
    const game=OverlayOptions.geometry(overlayOptions);
    if(isL)size=Math.min(size,game.bandHeight-GAP);
    node.style.width=node.style.height=`${size}px`;
    const x=isL&&!overlayOptions.band?game.x+game.width-size-MARGIN:panel.offsetLeft+(panel.offsetWidth-size)/2;
    const y=isL?game.bandY+(game.bandHeight-size)/2:Math.max(0,panel.offsetTop-size+4);
    node.style.left=`${x}px`;node.style.top=`${y}px`;
    img.style.scale=overlayOptions.mascot_flip_right&&x+size/2>WIDTH/2?'-1 1':'1 1';
  }
  /** @param {object} data @returns {void} */
  function update(data) {
    node.hidden=!overlayOptions.mascot;window.PuyoExpressions?.update();updateTension();if(node.hidden) return;
    place();
  }
  const original=renderSnapshot;renderSnapshot=data=>{original(data);update(data);};
  const originalUnavailable=unavailable;unavailable=(...args)=>{originalUnavailable(...args);update({});};
  const originalSync=syncMatch;syncMatch=data=>{
    const key=JSON.stringify([data.identity?.session_id,data.identity?.match_id]),time=data.timing?.source_available_ms;
    if(key!==matchKey||(Number.isFinite(time)&&lastSourceMs!==null&&time<lastSourceMs)) resetMotion();
    originalSync(data);
  };
  const paint=paintProbability;paintProbability=p=>{paint(p);updateTension();};
  reducedMotion.addEventListener('change',updateTension);
  update({});return {update,place};
})();
