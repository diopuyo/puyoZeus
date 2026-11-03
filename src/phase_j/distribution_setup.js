'use strict';
const $=id=>document.getElementById(id);let state,language='ja';
/** @param {string} ja @param {string} en @returns {string} */
const text=(ja,en)=>language==='ja'?ja:en;
const optionEnglish={pop:'Pop (black)','pop-cream':'Pop (cream)',center:'Center column',minimal:'Mini strip','l-left':'L-shaped',vertical:'Side column'};
const STATUS_POLL_MS=5000;
let stopped=false;
for(const [kind,name] of [['themes','theme'],['layouts','layout']]) {
  for(const [value,label] of Object.entries(OverlayOptions.catalog[kind])) {
    const option=new Option(label,value);option.dataset.ja=label;option.dataset.en=optionEnglish[value];$(name).add(option);
  }
}
/** @param {string} path @param {object} data @returns {Promise<object>} */
async function post(path,data={}) {
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const result=await response.json();if(!response.ok) throw Error(result.error);return result;
}
/** @returns {void} */
function translate() {
  language=$('language').value==='auto'?state.os_language:$('language').value;document.documentElement.lang=language;
  document.title=text('puyoZeus Live — 初期設定','puyoZeus Live — Setup');
  for(const node of document.querySelectorAll('[data-ja]')) node.textContent=node.dataset[language];preview();renderRuntime();
}
/** @returns {void} */
function preview() {
  for(const button of $('design-samples').children)
    button.setAttribute('aria-pressed',String(button.dataset.theme===$('theme').value));
  const query=new URLSearchParams({layout:$('layout').value,theme:$('theme').value,mascot:$('mascot').checked,mascot_position:'card',language,preview:'sample'});
  $('preset').src='/?'+query;
  const g=OverlayOptions.geometry(OverlayOptions.resolve({layout:$('layout').value}));
  $('preview-placement').textContent=text('OBS ブラウザ: 1920 × 1080。ゲーム: ','OBS Browser: 1920 × 1080. Game: ')+
    `${g.width} × ${g.height} / (${g.x}, ${g.y})`;
}
$('preset').onload=()=>{
  const child=$('preset').contentWindow;if(typeof child.renderSnapshot!=='function') return;
  child.renderSnapshot({identity:{session_id:'setup',match_id:'sample'},display:{input_status:'ready',match_phase:'playing'},
    timing:{source_available_ms:0},evaluations:{practical:{availability:'available',p1_win_probability:.68,source:'static',calibration_id:'b3-stage2-display'}}});
  const g=child.OverlayOptions.geometry(child.OverlayOptions.resolve({layout:$('layout').value}));
  const frame=child.document.createElement('div');
  frame.textContent=text('ゲームの表示範囲','Game display area');
  frame.style.cssText=`position:absolute;pointer-events:none;z-index:-1;box-sizing:border-box;border:3px dashed #8396b1;color:#b9c9db;padding:28px;font:28px sans-serif;left:${g.x}px;top:${g.y}px;width:${g.width}px;height:${g.height}px`;
  child.document.getElementById('stage').append(frame);
};
/** @returns {Promise<void>} */
async function scan() {
  $('scan').disabled=true;$('test').disabled=true;
  try {
    const result=await post('/api/discover');$('device').replaceChildren();
    for(const device of result.devices) {const option=new Option(`${device.name} (${device.index})`,device.index);option.dataset.name=device.name;$('device').append(option);}
    const selected=result.selected;
    if(selected) $('device').value=selected.index;
    $('camera-help').hidden=Boolean(selected);$('test').disabled=!result.devices.length;
    $('device-status').textContent=result.message.split(' / ')[language==='en'?1:0]||result.message;
  } catch(error) {$('device-status').textContent=String(error);$('camera-help').hidden=false;}
  $('scan').disabled=false;
}
$('scan').onclick=scan;
$('test').onclick=async()=>{
  $('test').disabled=true;
  try {const result=await post('/api/camera-test',{index:Number($('device').value)});$('input-test').src=result.image;$('input-test').hidden=false;}
  catch(error) {$('device-status').textContent=String(error);}
  finally {$('test').disabled=!$('device').selectedOptions.length;}
};
$('start').onclick=async()=>{
  $('start').disabled=true;
  try {
    if(!$('device').selectedOptions.length) throw Error(text('入力を選んでください','Select an input'));
    const values={language:$('language').value,device_index:Number($('device').value),device_name:$('device').selectedOptions[0].dataset.name,
      port:Number($('port').value),relay_url:$('relay_url').value,releases_url:$('releases_url').value,
      overlay:{...state.settings.overlay,layout:$('layout').value,theme:$('theme').value,mascot:$('mascot').checked,language:$('language').value}};
    if(values.overlay.layout!==state.settings.overlay.layout) {
      delete values.overlay.x;delete values.overlay.y;
      values.overlay.corner=values.overlay.layout==='l-left'?'top-right':'top-left';
      const defaults=OverlayOptions.resolve({layout:values.overlay.layout,theme:values.overlay.theme});
      for(const key of ['recent','full','details']) values.overlay[key]=defaults[key];
    }
    for(const key of ['desktop_shortcut','start_menu_shortcut','autostart','check_updates']) values[key]=$(key).checked;
    await post('/api/settings',values);const launched=await post('/api/start');
    if(!launched.started) throw Error(launched.input?.message||'Input unavailable');
    $('status').textContent=text('保存しました。起動しています。表示ページで画面調整・試合待ちを確認してください。','Saved. Starting up. Open the overlay to check calibration and match waiting.');
    await refreshRuntime();
  } catch(error) {$('status').textContent=String(error);}
  $('start').disabled=false;
};
$('copy').onclick=async()=>{try{await navigator.clipboard.writeText($('url').value);$('status').textContent=text('コピーしました','Copied');}catch{$('url').select();$('status').textContent='Ctrl+C';}};
$('language').onchange=translate;for(const id of ['theme','layout','mascot']) $(id).onchange=preview;
$('layout').onchange=preview;
$('theme').onchange=preview;
/** @returns {Promise<void>} */
async function initialize() {
  state=await (await fetch('/api/state')).json();$('root').textContent=state.root;$('url').value=state.url;
  for(const id of ['language','port','relay_url','releases_url']) $(id).value=state.settings[id];
  for(const id of ['desktop_shortcut','start_menu_shortcut','autostart','check_updates']) $(id).checked=state.settings[id];
  const options=OverlayOptions.resolve(state.settings.overlay);
  $('layout').value=({card:'center',data:'center',ribbon:'center',l:'l-left','l-right':'l-left'})[options.layout]||options.layout;
  $('theme').value=options.theme;$('mascot').checked=options.mascot;
  translate();await scan();
}
initialize().catch(error=>{$('status').textContent=String(error);});
/** @returns {void} */
function renderRuntime() {
  if(!state) return;
  const status=DistributionStatus(state,language);
  $('runtime-title').textContent=status.title;$('runtime-next').textContent=status.next;
  $('runtime-card').dataset.state=status.kind;$('retry').hidden=state.running||!state.error;
}
/** @returns {Promise<void>} */
async function refreshRuntime() {
  if(stopped) return;
  try {const response=await fetch('/api/state');if(!response.ok) throw Error(response.statusText);
    const latest=await response.json();if(stopped) return;state=latest;renderRuntime();
  } catch {$('runtime-title').textContent=text('接続を確認してください','Check connection');
    $('runtime-next').textContent=text('アプリが起動しているか確認してください。','Check that the app is running.');}
}
$('retry').onclick=async()=>{
  $('retry').disabled=true;
  try {const result=await post('/api/start');if(!result.started) throw Error(result.input?.message||text('入力を確認してください','Check input'));await refreshRuntime();}
  catch(error) {$('status').textContent=String(error);}
  finally {$('retry').disabled=false;}
};
$('stop-setup').onclick=async()=>{
  $('stop-setup').disabled=true;
  try {await post('/api/stop');stopped=true;$('runtime-title').textContent=text('終了しました','App closed');
    $('runtime-next').textContent=text('このページを閉じてください。再開はアプリを起動してください。','Close this page. Launch the app to resume.');
    for(const id of ['retry','start','scan','test']) $(id).disabled=true;
  } catch(error) {$('status').textContent=String(error);$('stop-setup').disabled=false;}
};
setInterval(refreshRuntime,STATUS_POLL_MS);

// 配置と独立した三つの見本。選択後は既存の保存・URL生成を使う。
for(const [value,label] of Object.entries(OverlayOptions.catalog.themes)) {
  const button=document.createElement('button');button.type='button';button.dataset.theme=value;
  button.dataset.ja=label;button.dataset.en=optionEnglish[value];button.textContent=button.dataset.ja;
  button.onclick=()=>{$('theme').value=value;preview();};$('design-samples').append(button);
}
