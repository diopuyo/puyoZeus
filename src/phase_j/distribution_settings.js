'use strict';
// 確認済みのプレビューIDだけを送信する。再起動時の自動再送はしない。
let reportPreview=null;
const reportLocalNotice='送信先を準備中です。確認した内容は手元 (logs/reports) に保存します / The destination is being prepared. Reviewed reports are saved locally (logs/reports).';
const localMessage=value=>{const parts=(value||'').split(' / ');return parts[document.documentElement.lang==='en'&&parts.length>1?1:0];};
const errorMessage=error=>error instanceof Error?error.message:String(error);
// 翻訳後のDOMを再利用せず、日英の原文と接頭辞を保持する。
/** @type {Map<string, {message: string, prefix: string}>} */
const localMessages=new Map();
/** @param {string} id @param {string} message @param {string} prefix @returns {void} */
function setLocalMessage(id,message,prefix='') {
  localMessages.set(id,{message:message||'',prefix});
  get(id).textContent=prefix+localMessage(message);
}
/** @returns {void} */
function renderLocalMessages() {
  for(const [id,{message,prefix}] of localMessages) get(id).textContent=prefix+localMessage(message);
}
// 日英の原文がある結果は出し直し、片方しかない結果は消す。
document.addEventListener('puyo-language-change',()=>{
  for(const node of document.querySelectorAll('[data-i18n-transient]')) {
    node.textContent='';
    if(!localMessages.get(node.id)?.message.includes(' / ')) localMessages.delete(node.id);
  }
  renderLocalMessages();
});
/** @param {string} path @param {object} data @returns {Promise<object>} */
async function distributionPost(path,data) {
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const result=await response.json();if(!response.ok) throw Error(result.error||response.statusText);return result;
}
document.getElementById('save-settings').onclick=async()=>{
  try {const result=await distributionPost('/api/settings',applicationSettings());setLocalMessage('save-status',result.warning||'保存しました / Saved');}
  catch(error) {setLocalMessage('save-status',errorMessage(error));}
};
get('send-logs').onclick=async()=>{
  setLocalMessage('cancel-logs','キャンセル / Cancel');
  setLocalMessage('log-status','');get('confirm-logs').disabled=true;get('log-dialog').showModal();
  try {
    reportPreview=await distributionPost('/api/logs/preview',{});
    get('log-preview').textContent=JSON.stringify(reportPreview,null,2);get('confirm-logs').disabled=false;
  } catch(error) {setLocalMessage('log-status',errorMessage(error));}
};
get('cancel-logs').onclick=()=>{reportPreview=null;get('log-dialog').close();};
get('confirm-logs').onclick=async()=>{
  get('confirm-logs').disabled=true;
  try {const result=await distributionPost('/api/logs/send',{token:reportPreview.token,confirmed:true});
    setLocalMessage('log-status',reportPreview.local_only?'手元 (logs/reports) に保存しました / Saved locally (logs/reports).':(result.sent?reportReceipt(result):reportFailure(result.status)));
    setLocalMessage('cancel-logs','閉じる / Close');
  } catch(error) {setLocalMessage('log-status',errorMessage(error));}
  reportPreview=null;
};
get('dismiss-intro').onclick=()=>{get('getting-started').hidden=true;};
const APP_CHECKBOXES={'desktop-shortcut':'desktop_shortcut','start-menu-shortcut':'start_menu_shortcut',autostart:'autostart','check-updates':'check_updates'};
/** @returns {object} */
function applicationSettings() {
  const values={completed:true,overlay:readControls(),port:Number(get('app-port').value)};
  for(const [id,key] of Object.entries(APP_CHECKBOXES)) values[key]=get(id).checked;
  return values;
}
get('save-app-settings').onclick=async()=>{
  try {const result=await distributionPost('/api/settings',applicationSettings());setLocalMessage('app-save-status',result.warning||'保存しました / Saved');}
  catch(error) {setLocalMessage('app-save-status',errorMessage(error));}
};
get('copy-obs-url').onclick=async()=>{
  try {await navigator.clipboard.writeText(get('url').value);setLocalMessage('obs-copied','コピーしました / Copied');}
  catch(error) {setLocalMessage('obs-copied',errorMessage(error));}
};
/** @param {object} settings @returns {void} */
function loadApplicationSettings(settings) {
  for(const [id,key] of Object.entries(APP_CHECKBOXES)) get(id).checked=settings[key];
  get('app-port').value=settings.port;
  get('getting-started').hidden=Boolean(settings.onboarding_seen||settings.completed);
  if(!settings.onboarding_seen) distributionPost('/api/settings',{onboarding_seen:true}).catch(error=>setLocalMessage('save-status',errorMessage(error)));
}
/** @returns {Promise<void>} */
async function appNotice() {
  try {
    const response=await fetch('/api/state');if(!response.ok) return;
    const state=await response.json();
    // 既存の設定応答がある一般配布版だけで専用操作を表示する。
    if(!state.settings) return;
    let notice=get('report-local-notice');
    if(!notice) {notice=document.createElement('p');notice.id='report-local-notice';get('send-logs').after(notice);}
    setLocalMessage('report-local-notice',new URL(state.settings.relay_url).hostname.endsWith('.invalid')?reportLocalNotice:'');
    document.querySelectorAll('[data-distribution-only]').forEach(element=>{element.hidden=false;});
    if(!window.distributionReady) {
      window.distributionReady=true;loadApplicationSettings(state.settings);
      if(location.hash==='#logs') get('send-logs').click();
    }
    get('open-feedback').hidden=state.feedback_enabled!==true;
    window.lastDiagnostics=state.diagnostics||{};renderDiagnostics();
    const d=window.lastDiagnostics;
    setLocalMessage('input-reason',d.input_waiting?(d.input?.message||'試合が映るのを待っています / Waiting for a match to appear'):d.scanning?'入力を確認しています / Checking inputs':d.input?.message);
    if(d.calibration==='overlay_feedback') setLocalMessage('input-reason','OBS の仮想カメラの出力を「ソース」にして、ゲームの映像だけを選んでください。 / Set OBS Virtual Camera output to Source and select only the game capture.');
    if(!window.obsLoaded) {get('obs-enabled').checked=state.settings.obs_enabled;get('obs-port').value=state.settings.obs_port;window.obsLoaded=true;}

    window.lastDistributionState=state;renderAppNotice();
    window.renderDistributionUX?.(state);
  } catch {}
}
const APP_POLL_MS=1000;
appNotice();setInterval(appNotice,APP_POLL_MS);
/** @returns {void} */
function renderAppNotice() {
  renderLocalMessages();
  const state=window.lastDistributionState;if(!state) return;
  const language=document.documentElement.lang,notice=DistributionStatus(state,language);
  const en=language==='en',scanning=state.diagnostics?.scanning,waiting=state.diagnostics?.input_waiting;
  get('toggle-runtime').textContent=(state.running||waiting)?(en?'Stop':'停止'):(en?'Start':'起動');
  get('toggle-runtime').disabled=runtimePending||(scanning&&!waiting);
  if(window.renderDistributionUX) {window.renderDistributionUX(state);return;}
  get('runtime-state').textContent=waiting?(en?'Waiting for a match':'試合待ち'):scanning?(en?'Checking input':'入力確認中'):state.running?(en?'Running':'動作中'):(en?'Stopped':'停止中');
  const update=language==='en'?'New version available':'新しい版があります';
  get('app-notice').textContent=`${notice.title} — ${notice.next}`+
    (state.update?.available?` / ${update}: ${state.update.version}`:'')+
    (state.diagnostics?.warning?` / ${localMessage(state.diagnostics.warning)}`:'');
}
form.addEventListener('input',renderAppNotice);
let runtimePending=false;
get('toggle-runtime').onclick=async()=>{
  runtimePending=true;get('toggle-runtime').disabled=true;
  try {
    const running=window.lastDistributionState?.running||window.lastDistributionState?.diagnostics?.input_waiting;
    const result=await distributionPost(running?'/api/pause':'/api/start',{});
    setLocalMessage('input-reason',result.input?.message);
  } catch(error) {setLocalMessage('input-reason',errorMessage(error));}
  finally {runtimePending=false;await appNotice();}
};
/** @param {boolean} apply @returns {Promise<void>} */
async function obsAction(apply) {
  get('obs-check').disabled=get('obs-apply').disabled=true;
  try {
    if(apply) get('obs-enabled').checked=true;
    const settings={obs_enabled:get('obs-enabled').checked,obs_port:Number(get('obs-port').value)};
    if(get('obs-password').value) settings.obs_password=get('obs-password').value;
    await distributionPost('/api/settings',settings);get('obs-password').value='';
    const result=await distributionPost('/api/obs',{apply});
    const en=document.documentElement.lang==='en';
    if(apply) {
      const success=result.url_updated?(en?`Updated the overlay URL in OBS scene “${result.scene}” to ${result.url}.`:`OBS の「${result.scene}」の表示の URL を ${result.url} に直しました。`):result.already_added?(en?`The overlay is already in OBS scene “${result.scene}”. The virtual camera is running.`:`OBS の「${result.scene}」にはすでに追加済みです。仮想カメラを開始しました。`):
        (en?`Added the overlay to OBS scene “${result.scene}” and started the virtual camera.`:`OBS の「${result.scene}」に表示を追加し、仮想カメラを開始しました。`);
      setLocalMessage('obs-result',success+'\n'+(en?'Set Virtual Camera output using the gear → Source → game video (to keep the overlay out of the input).':'仮想カメラの出力は、歯車 →「ソース」→ ゲームの映像にしてください (表示が映り込むのを防ぎます)'));
    } else setLocalMessage('obs-result',result.warning,`${result.scene}: ${result.browser_sources.join(', ')}\n`);
  } catch(error) {setLocalMessage('obs-result',errorMessage(error));}
  finally {get('obs-check').disabled=get('obs-apply').disabled=false;}
}
get('obs-check').onclick=()=>obsAction(false);get('obs-apply').onclick=()=>obsAction(true);
get('obs-password-clear').onclick=async()=>{
  try {await distributionPost('/api/settings',{obs_password:''});get('obs-password').value='';
    setLocalMessage('obs-result','パスワードを消去しました / Password cleared');}
  catch(error) {setLocalMessage('obs-result',errorMessage(error));}
};

function renderDiagnostics() {
 const d=window.lastDiagnostics||{},t=window.PuyoI18n.t;
    get('live-diagnostics').textContent=[
      `${t('入力の解像度')}: ${d.input_resolution?.join(' × ')||'—'}`,
      `FPS: ${d.fps==null?'—':Number(d.fps).toFixed(1)}`, `${t('較正')}: ${d.calibration||'—'}`,
      `${t('判定の遅れ')}: ${d.delay_ms==null?'—':Number(d.delay_ms).toFixed(1)+' ms'}`,
      `${t('最後の更新')}: ${d.last_update?new Date(d.last_update*1000).toLocaleTimeString():'—'}`
    ].join('\n');
}
form.addEventListener('input',renderDiagnostics);

/** @param {object} result @returns {string} */
function reportReceipt(result) {
  const accepted=Number.isInteger(result.accepted)?result.accepted:'?';
  const duplicates=Number.isInteger(result.duplicates)?result.duplicates:'?';
  if(accepted==='?'&&duplicates==='?') return '送信しました / Sent';
  return `受理 ${accepted} 件、重複 ${duplicates} 件 / Accepted: ${accepted}; duplicates: ${duplicates}`;
}
/** @param {number} status @returns {string} */
function reportFailure(status) {
  if(status===429) return '送信回数の上限です。時間をおいて再度確認・送信してください。手元に保存しました。 / Rate limit reached. Review and retry later. Saved locally.';
  if(status===413) return '報告が大きすぎます。フォルダから少数ずつフォームで送ってください。手元に保存しました。 / Reports are too large. Open the folder and submit fewer reports through the form. Saved locally.';
  return '送信できませんでした。手元に保存しました。次回確認して再送できます。 / Could not send. Saved locally for your next manual review and retry.';
}
for(const [id,path] of [['open-reports','/api/reports/open'],['open-feedback','/api/feedback/open']]) {
  get(id).onclick=async()=>{
    try {await distributionPost(path,{});setLocalMessage('save-status','開きました / Opened');}
    catch(error) {setLocalMessage('save-status',errorMessage(error));}
  };
}

// 窓で同じ設定ページのログ欄へ移ったときも確認ダイアログを開く。
window.addEventListener('hashchange',()=>{
  if(location.hash==='#logs'&&!get('log-dialog').open) get('send-logs').click();
});
