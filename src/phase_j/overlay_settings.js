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
  options.layout=({card:'center',data:'center',ribbon:'center'})[options.layout]||options.layout;
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
  base.searchParams.set('preview',live?'live':'sample');get('overlay').src=base.href;
  window.PuyoI18n?.setLanguage(options.language);
  window.scheduleAppearanceSave?.();
}
/** @returns {void} */
function renderSample() {
  if(get('scene').value==='live') return;
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
