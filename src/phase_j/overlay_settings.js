'use strict';
const form=document.getElementById('controls'),get=id=>document.getElementById(id);
const PERCENT=100;
let imageUrl=null;
for(const [kind,name] of [['themes','theme'],['layouts','layout']]) {
  for(const [value,label] of Object.entries(OverlayOptions.catalog[kind]))
    form.elements.namedItem(name).add(new Option(label,value));
}
for(const [value,label] of Object.entries(OverlayOptions.catalog.themes)) {
  const button=document.createElement('button');button.type='button';button.dataset.theme=value;
  button.textContent=label;button.onclick=()=>{form.elements.theme.value=value;updatePreview();};
  get('theme-samples').append(button);
}
/** @param {object} values */
function setControls(values) {
  const options=OverlayOptions.resolve(values);
  options.layout=({data:'card',ribbon:'center'})[options.layout]||options.layout;
  for(const [key,value] of Object.entries(options)) {
    const input=form.elements.namedItem(key);
    if(!input) continue;
    if(input.type==='checkbox') input.checked=value;else input.value=String(value);
  }
  updatePreview();
}
/** @returns {object} */
function readControls() {
  const result={};
  for(const input of form.elements) if(input.name) result[input.name]=input.type==='checkbox'?input.checked:input.value;
  result.scale=Number(result.scale);
  return OverlayOptions.normalize(result);
}
/** @returns {void} */
function updatePreview() {
  const options=readControls(),query=new URLSearchParams(options),g=OverlayOptions.geometry(OverlayOptions.resolve(options));
  const isL=options.layout.startsWith('l');
  const placementHint=get('placement-hint');
  if(placementHint) {
    placementHint.hidden=!isL;
    const placementEnglish=options.language==='en'||(options.language==='auto'&&!String(window.overlayDefaults?.os_language||navigator.language).startsWith('ja'));
    placementHint.textContent=placementEnglish?
      `In OBS, scale the game video to ${options.scale}% and place it at (${g.x}, ${g.y}).`:
      `OBS でゲームの映像を ${options.scale}% に縮小し、位置 (${g.x}, ${g.y}) に置いてください`;
  }
  form.elements.mascot_dopamine.disabled=!options.mascot_dopamine_mode;
  form.elements.mascot_speed.disabled=!options.mascot_eyes;
  for(const button of get('theme-samples').children) button.setAttribute('aria-pressed',String(button.dataset.theme===options.theme));
  get('l-controls').disabled=!isL;get('scale-value').value=`${options.scale}%`;
  const base=new URL('/',location.href);base.search=query.toString();get('url').value=base.href;
  if(window.overlayDefaults.distribution_mode) get('url').value=new URL('/',location.href).href;
  get('json').value=JSON.stringify({overlay:options},null,2);
  get('copied').textContent='';
  get('placement').textContent=`ブラウザソース：1920 × 1080、位置 (0, 0)、最上面\nゲーム：位置 (${g.x}, ${g.y})、${g.width} × ${g.height}、縮小率 ${isL?options.scale:100}%\n`+
    (isL?'ゲームの部分は透明です。下寄せ時の横帯は上側に表示します。\n':'')+
    '仮想カメラ：ゲームのキャプチャだけをソース出力。認識入力は1920 × 1080を維持します。';
  const x=value=>`${value/OverlayOptions.WIDTH*PERCENT}%`,y=value=>`${value/OverlayOptions.HEIGHT*PERCENT}%`;
  Object.assign(get('frame').style,{left:x(g.x),top:y(g.y),width:x(g.width),height:y(g.height)});
  const live=get('scene').value==='live';
  get('sample-label').textContent=live?'数値：現在の対戦データ／背景：選択した静止画':'数値と推移は見本用のサンプルです。';
  base.searchParams.set('preview',live?'live':'sample');
  // input と change が続いても、同じ見本の読み込みをやり直さない。
  if(get('overlay').src!==base.href) get('overlay').src=base.href;else renderSample();
  window.PuyoI18n?.setLanguage(options.language);
  window.scheduleAppearanceSave?.();
}
/** @returns {void} */
function renderSample() {
  const frame=get('overlay'),doc=frame.contentDocument;
  if(get('scene').value==='live'||doc?.readyState!=='complete'||doc.URL!==frame.src) return;
  const child=get('overlay').contentWindow;
  if(typeof child.renderSnapshot!=='function') return;
  const scene=get('scene').value,COUNT=61,STEP_MS=2000;
  for(let i=0;i<COUNT;i++) {
    const p=scene==='early'?.5+i/1000:.6-i/200+Math.sin(i/4)*.07;
    const terminal=scene==='result'&&i===COUNT-1;
    child.renderSnapshot({identity:{session_id:'preview',match_id:'preview'},
      timing:{source_available_ms:i*STEP_MS},display:{input_status:'ready',match_phase:'playing'},
      evaluations:{practical:{availability:'available',p1_win_probability:terminal?0:p,
        calibration_id:'b3-stage2-display',source:terminal?'confirmed_death':scene==='early'?'static':'exchange'},
        display_layers:{current_p1:p+.02,predicted_p1:null}}});
  }
}
form.addEventListener('input',updatePreview);
form.elements.layout.addEventListener('change',updatePreview);
form.elements.theme.addEventListener('change',()=>{
  updatePreview();
});
form.addEventListener('submit',event=>event.preventDefault());
get('scene').addEventListener('change',updatePreview);
get('overlay').addEventListener('load',renderSample);
// 非表示タブで読み込んだ見本も、表示・サイズ・画像・フォントの確定後に配置する。
get('overlay').addEventListener('load',()=>{
  const child=get('overlay').contentWindow,doc=child.document;
  const style=doc.createElement('style');
  // 小さな見本でも数字と単位を別々の箱に保ち、互いに縮めない。
  style.textContent='.num{display:inline-flex;align-items:baseline;gap:2px}.num>span,.num>small{flex:none}.num>small{margin-left:0}';
  doc.head.append(style);
  const place=()=>child.PuyoMascot?.place();
  const observer=new ResizeObserver(place);
  for(const id of ['stage','panel','mascot']) {
    const node=doc.getElementById(id);if(node)observer.observe(node);
  }
  child.addEventListener('resize',place);
  doc.fonts.ready.then(place);
  doc.querySelectorAll('img').forEach(img=>img.addEventListener('load',place));
  child.addEventListener('pagehide',()=>observer.disconnect(),{once:true});
  place();
});
get('reset').onclick=()=>setControls(window.overlayDefaults);
get('image').onchange=()=>{
  const file=get('image').files[0];if(!file) return;
  get('image-name').textContent=file.name;
  if(imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl=URL.createObjectURL(file);get('frame').src=imageUrl;
  get('frame').hidden=false;get('empty').hidden=true;
};
get('choose-image').onclick=()=>get('image').click();
get('copy').onclick=async()=>{
  try {await navigator.clipboard.writeText(get('url').value);get('copied').textContent=' コピーしました';}
  catch {get('url').select();get('copied').textContent=' 選択したURLを Ctrl+C でコピーしてください';}
};
setControls(window.overlayDefaults);
