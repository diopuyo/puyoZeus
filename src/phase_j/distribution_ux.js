'use strict';
// 配布アプリの操作だけを3つのタブにまとめる。開発用の設定画面は維持する。
if(window.overlayDefaults.distribution_mode) {

  const main=document.querySelector('main'),columns=document.querySelector('.columns');
  main.append(get('log-dialog'));
  document.body.classList.add('distribution-ux');
  document.querySelector('h1').textContent='puyoZeus Live';
  document.querySelector('h1+p').remove();
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
  installStreamSteps(stream,start);
  panels.appearance.append(columns);columns.classList.add('ux-appearance');
  const sampleColumn=get('preview').parentElement,sceneLabel=get('scene').closest('label');
  const background=document.createElement('details');background.innerHTML='<summary>見本にゲーム画像を重ねる</summary>';
  while(sampleColumn.firstChild!==sceneLabel) background.append(sampleColumn.firstChild);
  sampleColumn.append(background);
  const language=form.elements.language;language.setAttribute('form','controls');other.prepend(language.closest('label'));
  language.addEventListener('input',()=>{updatePreview();renderAppNotice();window.renderDistributionUX?.(window.lastDistributionState);});
  other.append(get('autostart').closest('section'),get('send-logs').closest('.output'));
  for(const id of ['save-settings','save-app-settings']) get(id).remove();
  const save=get('save-status');tabs.after(save);save.classList.add('ux-saved');
  // URL と JSON は内部処理用に保持し、配布画面には表示しない。
  get('placement').closest('.output').hidden=true;
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
  const messages={attention:state.error?(en?'Could not start. Check Details.':'解析を始められませんでした。「くわしく」を確認してください。'):localMessage(d.input?.message||'映像の入力を確認してください / Check the video input'),
    checking:en?'Looking for video input…':'映像の入力を探しています…',
    playing:en?'Watching the game video.':'ゲームの映像を見ています。',
    waiting:d.restarting?(en?'Restarting analysis':'解析を再開しています'):localMessage(d.input_waiting&&d.input?.reason!=='ready'&&d.input?.message||'試合が映るのを待っています / Waiting for a match to appear'),
    stopped:en?'When ready, press Start analysis.':'準備ができたら「解析を始める」を押してください。'};
  get('app-notice').textContent=messages[key];
  renderStreamSteps(state,key);
  refreshObsStatus(state);
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

const OBS_STATUS_INTERVAL_MS=12000;
/** @param {HTMLElement} stream @param {HTMLElement} start @returns {void} */
function installStreamSteps(stream,start) {
  start.classList.add('stream-actions');get('getting-started').style.display='none';
  const detail=document.createElement('details');detail.id='stream-details';
  detail.innerHTML='<summary>くわしく</summary>';detail.append(get('live-diagnostics').closest('section'));
  const status=document.createElement('p');status.id='obs-connection';status.setAttribute('role','status');stream.append(status);
  stream.append(detail);
  const steps=document.createElement('details');steps.id='stream-steps';steps.open=true;
  steps.innerHTML='<summary id="preparation-summary">はじめに</summary><div id="stream-step-list"></div>';
  stream.append(steps);const list=steps.querySelector('div');
  const obs=get('obs-apply').closest('section');obs.classList.add('ux-obs');list.append(obs);
  const old=[...obs.childNodes];obs.replaceChildren();
  obs.innerHTML='<h2><span id="step-obs-mark">①</span> OBS と連携</h2><p>OBS の「ツール → WebSocket サーバー設定」で有効にし、「接続情報を表示」のパスワードを入れてください。</p>';
  // 取り外した要素は文書のID検索で取得できないため、元の節から集める。
  const find=id=>old.flatMap(n=>n.nodeType===1?[n,...n.querySelectorAll('*')]:[]).find(n=>n.id===id);
  obs.append(find('obs-password').closest('label'),find('obs-apply'),find('obs-result'));
  const manual=document.createElement('details');manual.id='obs-manual';manual.innerHTML='<summary>手動で設定する・その他</summary>';
  const url=document.createElement('input');url.id='stream-url';url.readOnly=true;url.setAttribute('aria-label','OBS URL');url.value=new URL('/',location.href).href;
  manual.append(url,find('copy-obs-url'),find('obs-copied'));
  manual.insertAdjacentHTML('beforeend','<p>OBS にブラウザソースを追加し、幅 1920・高さ 1080 に設定します。</p>');
  manual.append(find('obs-enabled').closest('label'),find('obs-port').closest('label'),find('obs-password-clear'),find('obs-check'));
  manual.insertAdjacentHTML('beforeend','<p>空欄なら保存済みのパスワードを使います。パスワードはこのPCだけに保存し、ログには含めません。</p>');
  obs.append(manual);find('obs-apply').textContent='OBS を自動で設定';find('obs-check').textContent='OBS の状態を確かめる';
  installCameraStep(list);installStreamVideo(stream);
  const content=document.createElement("div");content.className="stream-content";
  stream.append(content);content.append(steps,stream.querySelector(".ux-video"));
}
/** @param {HTMLElement} list @returns {void} */
function installCameraStep(list) {
  const camera=document.createElement('section');camera.id='step-camera';camera.className='output';camera.tabIndex=-1;
  camera.innerHTML='<h2><span id="step-camera-mark">②</span> 仮想カメラの出力をゲームの映像に</h2><p>OBS の「仮想カメラ開始」の横の歯車 → 出力の種類「ソース」→ ゲームのキャプチャを選んでください。</p><div class="camera-diagram" role="img" aria-label="仮想カメラの歯車 → ソース → ゲームの映像"><div class="camera-controls"><strong>仮想カメラ開始</strong><b>⚙</b></div><span aria-hidden="true">↓</span><div class="camera-output-window"><div><span>出力の種類</span><strong>ソース</strong></div><div><span>出力するソース</span><strong>ゲームの映像</strong></div></div></div><p>この操作は OBS で行います。自動設定では変更できません。</p><p id="camera-step-status" role="status"></p>';
  const analysis=document.createElement('section');analysis.className='output';
  analysis.innerHTML='<h2><span id="step-analysis-mark">③</span> 解析を始める</h2><p>準備ができたら、画面上部の「解析を始める」を押してください。試合が映ると解析中になります。</p><a href="#toggle-runtime">上の操作へ</a>';
  list.append(camera,analysis);
}
/** @param {HTMLElement} stream @returns {void} */
function installStreamVideo(stream) {
  const video=document.createElement('section');video.className='output ux-video';
  video.innerHTML='<h2>アプリが見ている映像</h2><img id="input-preview" width="320" height="180" alt="アプリが見ている映像" hidden><p id="input-placeholder">まだ映像がありません</p><p id="input-device"></p><details><summary>入力を変える</summary><label>入力を切り替える<select id="input-select"><option value="">自動で選ぶ</option></select></label><button id="scan-input" type="button" aria-describedby="scan-input-help">映像を確認</button><p id="scan-input-help">解析中は確かめられません。解析を止めてから押してください。</p></details>';
  const frame=document.createElement('div');frame.className='ux-video-frame';
  frame.append(video.querySelector('#input-preview'),video.querySelector('#input-placeholder'));
  const controls=document.createElement('div');controls.className='ux-video-controls';
  controls.append(video.querySelector('#input-device'),video.querySelector('details'));
  video.append(frame,controls);stream.append(video);
}
/** @param {object} state @param {string} key @returns {void} */
function renderStreamSteps(state,key) {
  const settings=state.settings||{},d=state.diagnostics||{},en=document.documentElement.lang==='en';
  const feedback=d.calibration==='overlay_feedback'||d.input?.reason==='feedback';
  const running=state.running||d.input_waiting;
  get('toggle-runtime').textContent=running?(en?'Stop analysis':'解析を止める'):(en?'Start analysis':'解析を始める');
  const done=settings.stream_obs_done&&settings.stream_camera_done&&settings.stream_analysis_done;
  const steps=get('stream-steps');
  if(feedback) steps.open=true;
  else if(steps.dataset.done!==String(Boolean(done))) steps.open=!done;
  steps.dataset.done=String(Boolean(done));
  get('preparation-summary').textContent=done&&!feedback?(en?'Ready ✓ · Show getting started again':'準備: 済み ✓ ・ はじめにをもう一度見る'):(en?'Getting started':'はじめに');
  for(const [id,flag,number] of [['obs','stream_obs_done','①'],['camera','stream_camera_done','②'],['analysis','stream_analysis_done','③']]) {
    get(`step-${id}-mark`).textContent=id==='camera'&&feedback?'!':settings[flag]?'✓':number;
  }
  get('step-camera').classList.toggle('needs-attention',feedback);
  get('camera-step-status').textContent=feedback?(en?'Needs attention: the overlay is appearing in the input.':'要対応: 表示が入力の映像に映り込んでいます。'):settings.stream_camera_done?(en?'Checked during analysis':'解析中に確認済み'):'';
  if(feedback) {
    get('app-notice').textContent=en?'Overlay detected in the input. Check step ② below.':'表示が映り込んでいます。下の手順②を確認してください。';
    const link=document.createElement('a');link.href='#step-camera';link.textContent=en?' Go to step ②':' 手順②へ';
    link.onclick=()=>{steps.open=true;get('step-camera').focus();};get('app-notice').append(link);
  }
  if(key==='playing'&&!feedback&&(!settings.stream_camera_done||!settings.stream_analysis_done)&&!window.streamSaving) {
    window.streamSaving=true;
    distributionPost('/api/settings',{stream_camera_done:true,stream_analysis_done:true})
      .then(()=>appNotice()).catch(error=>setLocalMessage('save-status',errorMessage(error))).finally(()=>{window.streamSaving=false;});
  }
}
/** @param {object} state @returns {Promise<void>} */
async function refreshObsStatus(state) {
  if(!get('obs-connection')) return;
  const configured=state.settings?.obs_enabled&&state.obs_password_saved;
  if(!configured) {window.streamObsStatus=null;window.streamObsChecked=0;renderObsStatus(false);return;}
  renderObsStatus(true);
  if(document.hidden||window.streamObsBusy||Date.now()-(window.streamObsChecked||0)<OBS_STATUS_INTERVAL_MS) return;
  window.streamObsBusy=true;
  try {window.streamObsStatus=await distributionPost('/api/obs',{apply:false});}
  catch {window.streamObsStatus={failed:true};}
  finally {window.streamObsBusy=false;window.streamObsChecked=Date.now();renderObsStatus(true);}
}
/** @param {boolean} configured @returns {void} */
function renderObsStatus(configured) {
  const s=window.streamObsStatus,en=document.documentElement.lang==='en';
  const obs=!configured?(en?'× Not set up':'× 未設定'):!s?(en?'Checking':'確認中'):s.failed?(en?'× Cannot connect to OBS':'× OBS につながりません'):(en?'✓ Connected':'✓ 連携済み');
  const browser=s&&!s.failed?(s.applied||s.already_added?'✓':en?'× Not added':'× 未追加'):'—';
  const camera=s&&!s.failed?(s.virtual_camera?(en?'✓ Running':'✓ 開始中'):(en?'× Stopped':'× 停止中')):'—';
  const separator=en?' · ':' ・ ';
  get('obs-connection').textContent=[`OBS ${obs}`,`${en?'Browser source':'ブラウザソース'} ${browser}`,`${en?'Virtual camera':'仮想カメラ'} ${camera}`].join(separator);
}
