'use strict';
// 配布アプリの操作だけを3つのタブにまとめる。開発用の設定画面は維持する。
if(window.overlayDefaults.distribution_mode) {

  const main=document.querySelector('main'),columns=document.querySelector('.columns');
  main.append(get('log-dialog'));
  document.body.classList.add('distribution-ux');
  document.querySelector('h1').textContent='puyoZeus Live';
  document.querySelector('h1+p').textContent='配信を始める、見た目を変える、アプリを設定する。';
  const tabs=document.createElement('nav');tabs.className='ux-tabs';tabs.setAttribute('role','tablist');
  main.insertBefore(tabs,columns);
  const panels={};
  for(const [key,title] of [['stream','配信'],['appearance','見た目'],['other','その他']]) {
    const button=document.createElement('button');button.type='button';button.id=`tab-${key}`;
    button.textContent=title;button.setAttribute('role','tab');button.setAttribute('aria-controls',`page-${key}`);
    const panel=document.createElement('section');panel.id=`page-${key}`;panel.className='ux-page';
    panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby',button.id);panels[key]=panel;
    tabs.append(button);main.append(panel);
    button.onclick=()=>{for(const [name,page] of Object.entries(panels)) {
      page.hidden=name!==key;get(`tab-${name}`).setAttribute('aria-selected',String(name===key));
      get(`tab-${name}`).tabIndex=name===key?0:-1;
    }};
    button.onkeydown=event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
      event.preventDefault();const list=[...tabs.children],at=list.indexOf(button);
      const next=event.key==='Home'?0:event.key==='End'?2:(at+(event.key==='ArrowRight'?1:2))%3;
      list[next].click();list[next].focus();};
  }
  const stream=panels.stream,other=panels.other;
  const start=get('toggle-runtime').parentElement;stream.append(start);
  const exit=start.querySelector('small');other.append(exit);
  stream.append(get('app-notice'));
  const locationNotice=document.createElement('p');locationNotice.id='location-notice';
  locationNotice.setAttribute('role','status');locationNotice.hidden=true;stream.append(locationNotice);
  const obs=get('obs-apply').closest('section');stream.append(obs);obs.classList.add('ux-obs');
  obs.querySelector('h2').textContent='OBS を自動で設定';
  const manual=document.createElement('details');manual.innerHTML='<summary>自分で設定する場合</summary>';
  const firstAuto=obs.querySelectorAll('h3')[1];
  while(obs.querySelector('h2').nextSibling!==firstAuto) manual.append(obs.querySelector('h2').nextSibling);
  firstAuto.remove();
  const manualCamera=obs.querySelector('h3');
  const cameraSteps=[];while(manualCamera.nextSibling) cameraSteps.push(obs.removeChild(manualCamera.nextSibling));
  manual.append(manualCamera,...cameraSteps);obs.append(manual);
  const preparation=document.createElement('p');preparation.innerHTML='OBS の「ツール → WebSocket サーバー設定」でサーバーを有効にします。<br>そこで表示されるパスワードを下に入力します。';
  obs.querySelector('h2').after(preparation);get('obs-apply').textContent='OBS を自動で設定';
  const urlRow=document.createElement('div');urlRow.className='ux-obs-url';
  const shortURL=document.createElement('input');shortURL.id='stream-url';shortURL.readOnly=true;
  shortURL.setAttribute('aria-label','OBS URL');shortURL.value=new URL('/',location.href).href;
  urlRow.append(shortURL,get('copy-obs-url'),get('obs-copied'));obs.querySelector('h2').after(urlRow);
  const obsDetails=document.createElement('details');obsDetails.innerHTML='<summary>詳しく</summary>';
  obsDetails.append(get('obs-enabled').closest('label'));obs.append(obsDetails);
  const video=document.createElement('section');video.className='output ux-video';
  video.innerHTML='<h2>アプリが見ている映像</h2><img id="input-preview" width="320" height="180" alt="アプリが見ている映像" hidden><p id="input-placeholder">まだ映像がありません</p><p id="input-device"></p><label>入力を切り替える<select id="input-select"><option value="">自動で選ぶ</option></select></label><button id="scan-input" type="button">映像を確認</button>';
  stream.append(video);
  const diagnostic=get('live-diagnostics').closest('section');
  const detail=document.createElement('details');detail.innerHTML='<summary>詳しく</summary>';detail.append(diagnostic);stream.append(detail);
  panels.appearance.append(columns);columns.classList.add('ux-appearance');
  const sampleColumn=get('preview').parentElement,sceneLabel=get('scene').closest('label');
  const background=document.createElement('details');background.innerHTML='<summary>見本にゲーム画像を重ねる</summary>';
  while(sampleColumn.firstChild!==sceneLabel) background.append(sampleColumn.firstChild);
  sampleColumn.append(background);
  const language=form.elements.language;language.setAttribute('form','controls');other.prepend(language.closest('label'));
  language.addEventListener('input',()=>{updatePreview();renderAppNotice();window.renderDistributionUX?.(window.lastDistributionState);});
  other.append(get('desktop-shortcut').closest('section'),get('send-logs').closest('.output'));
  for(const id of ['save-settings','save-app-settings']) get(id).remove();
  const save=get('save-status');tabs.after(save);save.classList.add('ux-saved');
  const placement=get('placement').closest('.output');const advanced=document.createElement('details');
  advanced.innerHTML='<summary>詳しく</summary>';placement.before(advanced);advanced.append(placement);
  const placementHint=document.createElement('p');placementHint.id='placement-hint';
  const mascot=form.querySelector('fieldset');const more=document.createElement('details');
  more.innerHTML='<summary>ゼウスくんの細かな設定</summary>';
  const size=form.elements.mascot_size.closest('label');while(size.nextSibling) more.append(size.nextSibling);more.prepend(size);mascot.append(more);
  form.prepend(form.elements.layout.closest('label'),form.elements.theme.closest('label'),get('theme-samples'));
  form.elements.layout.closest('label').after(placementHint);updatePreview();
  const hint=document.createElement('small');hint.textContent='変更は自動で保存され、OBS に反映されます。';tabs.after(hint);
  get('tab-stream').click();
  installUXSave();installInputPreview();PuyoI18n.setLanguage(readControls().language);
}

/** @returns {void} */
function installUXSave() {
  let timer,queue=Promise.resolve(),last='';
  window.scheduleAppearanceSave=()=>{
    if(!window.distributionReady) return;
    clearTimeout(timer);timer=setTimeout(()=>{
      const values=applicationSettings();delete values.completed;
      const encoded=JSON.stringify(values);if(encoded===last) return;
      queue=queue.catch(()=>{}).then(async()=>{
        try {const result=await distributionPost('/api/settings',values);last=encoded;
          setLocalMessage('save-status',result.warning||'保存しました / Saved');
          setTimeout(()=>setLocalMessage('save-status',''),2500);
        } catch(error) {setLocalMessage('save-status',errorMessage(error));}
      });
    },250);
  };
  for(const id of [...Object.keys(APP_CHECKBOXES),'app-port']) get(id).addEventListener('input',window.scheduleAppearanceSave);
}

/** @returns {void} */
function installInputPreview() {
  let previousURL='';
  const refresh=async()=>{
    try {const response=await fetch('/api/input-preview.jpg');if(response.status!==200) return;
      const blob=await response.blob();if(!blob.size) return;
      const url=URL.createObjectURL(blob);get('input-preview').src=url;get('input-preview').hidden=false;
      get('input-placeholder').hidden=true;if(previousURL) URL.revokeObjectURL(previousURL);previousURL=url;
    } catch {}
  };
  const scan=async()=>{get('scan-input').disabled=true;window.inputChecking=true;
    window.renderDistributionUX?.(window.lastDistributionState);
    try {await distributionPost('/api/discover',{});await appNotice();await refresh();}
    catch(error) {setLocalMessage('save-status',errorMessage(error));}finally {
      window.inputChecking=false;get('scan-input').disabled=false;await appNotice();}
  };
  get('scan-input').onclick=scan;
  get('input-select').onchange=async()=>{
    const select=get('input-select');select.disabled=true;
    try {await distributionPost('/api/input/select',{index:select.value===''?null:Number(select.value)});await appNotice();}
    catch(error) {setLocalMessage('save-status',errorMessage(error));}finally {select.disabled=false;}
  };
  distributionPost('/api/devices',{}).then(result=>{for(const device of result.devices) {
    const option=document.createElement('option');option.value=device.index;option.textContent=device.name;get('input-select').append(option);
  }}).catch(()=>{});
  setInterval(refresh,1000);refresh();
}

/** @param {object} state @returns {void} */
window.renderDistributionUX=state=>{
  if(!state||!get('input-device')) return;
  const d=state.diagnostics||{},en=document.documentElement.lang==='en';
  const attention=state.error||['overlay_feedback','obs_stopped'].includes(d.calibration)||['feedback','wrong_size','obs_stopped'].includes(d.input?.reason);
  const playing=state.running&&d.calibration==='ready'&&!['between_matches','no_puyo'].includes(d.match_phase);
  renderPortConflict(state,en);
  const key=d.restarting?'waiting':(window.inputChecking||(d.scanning&&!d.input_waiting))?'checking':attention?'attention':playing?'playing':(state.running||d.input_waiting)?'waiting':'stopped';
  const labels={checking:['確認中','Checking'],attention:['要対応','Needs attention'],playing:['解析中','Analyzing'],waiting:['試合待ち','Waiting for a match'],stopped:['停止中','Stopped']};
  const badge=get('runtime-state');badge.textContent=labels[key][en?1:0];badge.dataset.status=key;
  const messages={attention:state.error?(en?'Could not start. Check Details.':'起動できませんでした。「詳しく」を確認してください。'):localMessage(d.input?.message||'映像の入力を確認してください / Check the video input'),
    checking:en?'Looking for video input…':'映像の入力を探しています…',
    playing:en?'Watching the game video.':'ゲームの映像を見ています。',
    waiting:d.restarting?(en?'Restarting analysis':'解析を再開しています'):localMessage(d.input_waiting&&d.input?.reason!=='ready'&&d.input?.message||'試合が映るのを待っています / Waiting for a match to appear'),
    stopped:en?'Press Start when you are ready.':'準備ができたら「起動」を押してください。'};
  get('app-notice').textContent=messages[key];
  const locationNotice=get('location-notice');locationNotice.hidden=!d.onedrive;
  locationNotice.textContent=d.onedrive?(en?
    'Caution: The app is extracted inside OneDrive. Syncing may slow it down or cause write conflicts. Extract it again outside OneDrive (for example, C:\\puyoZeus).':
    '注意: OneDrive の中に展開されています。同期で重くなったり書き込みがぶつかったりすることがあるので、OneDrive の外 (例 C:\\puyoZeus) に展開し直してください。'):'';
  get('input-device').textContent=d.input?.selected?.name||d.input?.devices?.[0]?.name||(state.running?state.settings.device_name:'—');
  get('input-select').value=state.selected_input==null?'':String(state.selected_input);
  get('scan-input').disabled=Boolean(state.running||d.scanning);
};

/** @param {object} state @param {boolean} en @returns {void} */
function renderPortConflict(state,en) {
  let panel=get('port-conflict');
  if(!panel) {
    panel=document.createElement('section');panel.id='port-conflict';
    panel.innerHTML='<strong></strong><p></p><button type="button"></button>';
    get('page-stream').prepend(panel);
    panel.querySelector('button').onclick=async()=>{
      const button=panel.querySelector('button');button.disabled=true;
      try {const result=await distributionPost('/api/port/switch',{});location.assign(result.url);}
      catch(error) {panel.querySelector('p').textContent=errorMessage(error);}
      finally {button.disabled=false;}
    };
  }
  const conflict=state.port_conflict;panel.hidden=!conflict||conflict.kind==='same';
  if(panel.hidden) return;
  panel.querySelector('strong').textContent=en?'Needs attention':'要対応';
  panel.querySelector('p').textContent=conflict.kind==='unresponsive'?
    (en?`Unresponsive puyoZeus Live (${conflict.root}) is using the port.`:`応答しない puyoZeus Live (${conflict.root}) がポートを使っています`):conflict.kind==='different'?
    (en?`puyoZeus Live in another location (${conflict.root}) is running, and OBS is connected to it.`:`別の場所 (${conflict.root}) の puyoZeus Live が動いていて、OBS の表示はそちらにつながっています`):
    (en?`Port ${conflict.port} is used by other software, so this app uses ${new URL(state.url).port}. Set up OBS automatically or change its URL to ${state.url}`:`ポート ${conflict.port} がほかのソフトに使われているので ${new URL(state.url).port} を使っています。OBS を自動で設定し直すか、OBS の URL を ${state.url} にしてください`);
  const button=panel.querySelector('button');button.hidden=!['different','unresponsive'].includes(conflict.kind);
  button.textContent=en?'Stop that app and switch to this one':'そちらを終了してこちらに切り替える';
}
