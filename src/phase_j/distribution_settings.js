'use strict';
// 確認済みのプレビューIDだけを送信する。再起動時の自動再送はしない。
let reportPreview=null;
const reportLocalNotice='0.6.0 では送信先を準備中です。確認した内容は手元 (logs/reports) に保存します / The destination is being prepared for 0.6.0. Reviewed reports are saved locally (logs/reports).';
const localMessage=value=>{const parts=(value||'').split(' / ');return parts[document.documentElement.lang==='en'&&parts.length>1?1:0];};
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
/** @param {string} path @param {object} data @returns {Promise<object>} */
async function distributionPost(path,data) {
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const result=await response.json();if(!response.ok) throw Error(result.error||response.statusText);return result;
}
document.getElementById('save-settings').onclick=async()=>{
  try {await distributionPost('/api/settings',{overlay:readControls()});get('save-status').textContent='保存しました';}
  catch(error) {get('save-status').textContent=String(error);}
};
get('send-logs').onclick=async()=>{
  setLocalMessage('log-status','');get('confirm-logs').disabled=true;get('log-dialog').showModal();
  try {
    reportPreview=await distributionPost('/api/logs/preview',{});
    get('log-preview').textContent=JSON.stringify(reportPreview,null,2);get('confirm-logs').disabled=false;
  } catch(error) {setLocalMessage('log-status',String(error));}
};
get('cancel-logs').onclick=()=>{reportPreview=null;get('log-dialog').close();};
get('confirm-logs').onclick=async()=>{
  get('confirm-logs').disabled=true;
  try {const result=await distributionPost('/api/logs/send',{token:reportPreview.token,confirmed:true});
    setLocalMessage('log-status',reportPreview.local_only?'手元 (logs/reports) に保存しました / Saved locally (logs/reports).':(result.sent?'送信しました / Sent':'送信できませんでした。手元に保存しました。次回確認して再送できます。 / Could not send. Saved locally for your next manual review and retry.'));
  } catch(error) {setLocalMessage('log-status',String(error));}
  reportPreview=null;
};
get('stop-app').onclick=async()=>{
  try {await distributionPost('/api/stop',{});get('save-status').textContent='終了しました。このページを閉じてください。';}
  catch(error) {get('save-status').textContent=String(error);}
};
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
      window.distributionReady=true;
      if(location.hash==='#logs') get('send-logs').click();
    }
    window.lastDiagnostics=state.diagnostics||{};renderDiagnostics();
    const d=window.lastDiagnostics;
    setLocalMessage('input-reason',d.scanning?'入力を確認しています / Checking inputs':d.input?.message);
    if(d.calibration==='overlay_feedback') setLocalMessage('input-reason','OBS の仮想カメラの出力を「ソース」にして、ゲームの映像だけを選んでください。 / Set OBS Virtual Camera output to Source and select only the game capture.');
    if(!window.obsLoaded) {get('obs-enabled').checked=state.settings.obs_enabled;get('obs-port').value=state.settings.obs_port;window.obsLoaded=true;}

    window.lastDistributionState=state;renderAppNotice();
  } catch {}
}
appNotice();setInterval(appNotice,5000);
/** @returns {void} */
function renderAppNotice() {
  renderLocalMessages();
  const state=window.lastDistributionState;if(!state) return;
  const language=document.documentElement.lang,notice=DistributionStatus(state,language);
  const update=language==='en'?'New version available':'新しい版があります';
  get('app-notice').textContent=`${notice.title} — ${notice.next}`+
    (state.update?.available?` / ${update}: ${state.update.version}`:'');
}
form.addEventListener('input',renderAppNotice);
get('uninstall-app').onclick=async()=>{
  const t=window.PuyoI18n.t;
  if(!confirm(t('アプリを終了し、アンインストールを開始しますか？'))) return;
  try {await distributionPost('/api/uninstall',{confirmed:true});
    get('uninstall-status').textContent=t('開いた画面で続けてください。このページは閉じられます。');
  } catch(error) {get('uninstall-status').textContent=String(error);}
};

get('rescan-input').onclick=async()=>{
  get('rescan-input').disabled=true;
  try {const result=await distributionPost('/api/start',{});setLocalMessage('input-reason',result.input?.message);}
  catch(error) {setLocalMessage('input-reason',String(error));}
  finally {get('rescan-input').disabled=false;appNotice();}
};
/** @param {boolean} apply @returns {Promise<void>} */
async function obsAction(apply) {
  get('obs-check').disabled=get('obs-apply').disabled=true;
  try {
    const settings={obs_enabled:get('obs-enabled').checked,obs_port:Number(get('obs-port').value)};
    if(get('obs-password').value) settings.obs_password=get('obs-password').value;
    await distributionPost('/api/settings',settings);get('obs-password').value='';
    const result=await distributionPost('/api/obs',{apply});
    setLocalMessage('obs-result',result.warning,`${result.scene}: ${result.browser_sources.join(', ')}\n`);
  } catch(error) {setLocalMessage('obs-result',String(error));}
  finally {get('obs-check').disabled=get('obs-apply').disabled=false;}
}
get('obs-check').onclick=()=>obsAction(false);get('obs-apply').onclick=()=>obsAction(true);

function renderDiagnostics() {
 const d=window.lastDiagnostics||{},t=window.PuyoI18n.t;
    get('live-diagnostics').textContent=[
      `${t('入力の解像度')}: ${d.input_resolution?.join(' × ')||'—'}`,
      `FPS: ${d.fps||'—'}`, `${t('較正')}: ${d.calibration||'—'}`,
      `${t('判定の遅れ')}: ${d.delay_ms==null?'—':d.delay_ms+' ms'}`,
      `${t('最後の更新')}: ${d.last_update?new Date(d.last_update*1000).toLocaleTimeString():'—'}`
    ].join('\n');
}
form.addEventListener('input',renderDiagnostics);
