'use strict';
// 設定ページと製品表示で同じ設定解決・座標計算を使う。
window.OverlayOptions=(()=>{
  const DEFAULT_VALUE_DISPLAY='probability';
  const WIDTH=1920, HEIGHT=1080, MIN_SCALE=70, MAX_SCALE=85;
  const DEFAULT_SCALE=80, PERCENT=100, BAND_EDGES=[10,30,70,90], EDGE_COUNT=4;
  // 設定ページに出す選択肢はここだけで管理する。
  const catalog={themes:{pop:'ポップ (黒)','pop-cream':'ポップ (クリーム)'},
    layouts:{center:'真ん中の柱',minimal:'小さい帯',card:'カード',vertical:'縦','l-top-left':'L字・左上','l-top-right':'L字・右上','l-bottom-left':'L字・左下','l-bottom-right':'L字・右下'}};
  const aliases={'battle-a':'pop',esports:'pop',normal:'pop',classic:'pop',noir:'pop',neon:'pop',arena:'pop',stylish:'pop',plain:'pop'};
  const enums={layout:['center','card','ribbon','minimal','data','l-right','l-left','l','vertical','l-top-left','l-top-right','l-bottom-left','l-bottom-right'],
    corner:['top-left','top-right','bottom-left','bottom-right'],font:['small','medium','large'],value_display:['probability','evaluation'],
    midline:['split','line','glow','adaptive'],theme:['pop',...Object.keys(catalog.themes),...Object.keys(aliases)],
    language:['auto','ja','en'],mascot_position:['left','right','card'],mascot_size:['small','medium','large'],mascot_dopamine:['low','normal','high'],mascot_speed:['0.5','1','1.5','2']};
  const contentFlags=['content_pending','content_sent','content_offset','content_rate',
    'content_hands','content_return','content_margin','content_bound','content_unknown',
    'content_active','content_step','content_length','content_remaining','content_seconds',
    'content_k1','content_colors','content_ojama','content_model'];
  const flags=[...contentFlags,'band','numbers','recent','full','status','details','mascot','mascot_dopamine_mode','mascot_motion','mascot_eyes','mascot_expressions','mascot_flip_right','mascot_bloodshot','debug_panel','vgraph'];
  // 配信で読む主要項目だけを既定表示し、補足は個別設定で追加する。
  const defaultContents=['pending','return','margin','step','length','k1'].map(key=>`content_${key}`);
  /** @param {object} input @returns {object} */
  function valid(input) {
    const legacyDisplay={evaluation_main:'evaluation',probability_main:'probability'};
    input={...input,value_display:legacyDisplay[input.value_display]||input.value_display};
    const result={};
    for(const [key,choices] of Object.entries(enums)) if(choices.includes(input[key])) result[key]=input[key];
    if(result.theme) result.theme=aliases[result.theme]||result.theme;
    if(result.layout==='data') {result.layout='card';result.details=true;}
    if(result.layout==='ribbon') result.layout='center';
    for(const [key,limit] of [['x',WIDTH],['y',HEIGHT]]) {
      if(input[key]!==undefined&&input[key]!==''&&Number.isFinite(Number(input[key])))
        result[key]=Math.round(Math.max(0,Math.min(limit,Number(input[key]))));
    }
    for(const key of flags) {
      if([true,'true','1'].includes(input[key])) result[key]=true;
      if([false,'false','0'].includes(input[key])) result[key]=false;
    }
    const scale=Number(input.scale);
    if(input.scale!==''&&Number.isFinite(scale)&&scale>=MIN_SCALE&&scale<=MAX_SCALE) result.scale=scale;
    const raw=input.mascot_band_edges;
    const edges=typeof raw==='string'?raw.split(',').map(v=>v.trim()===''?NaN:Number(v)):raw;
    if(Array.isArray(edges)&&edges.length===EDGE_COUNT&&edges.every((v,i)=>
      typeof v==='number'&&Number.isFinite(v)&&v>0&&v<PERCENT&&(i===0||v>edges[i-1])))
      result.mascot_band_edges=[...edges];
    return normalize(result);
  }
  /** @param {object} input @returns {object} */
  function normalize(input) {
    const result={...input};
    if(['l','l-left','l-right'].includes(result.layout))
      result.layout='l-'+(result.corner||(result.layout==='l-left'?'top-right':'top-left'));
    if(result.mascot_dopamine_mode===undefined&&
      ('mascot_motion' in result||'mascot_expressions' in result))
      result.mascot_dopamine_mode=result.mascot_motion!==false&&result.mascot_expressions!==false;
    for(const key of ['x','y','corner','mascot_flip_right','mascot_band_edges','numbers','recent','full','mascot_motion','mascot_expressions']) delete result[key];
    return result;
  }
  /** @param {object} defaults @param {string} query @returns {object} */
  function resolve(defaults={},query='') {
    if(defaults.distribution_mode&&!new URLSearchParams(query).has('preview')) query='';
    const selected={...valid(defaults),...valid(Object.fromEntries(new URLSearchParams(query)))};
    // 旧URLの指定を記録しつつ、見た目は対応する採用デザインに統一する。
    const requestedTheme=new URLSearchParams(query).get('theme')??defaults.theme;
    selected.legacyTheme=!requestedTheme||Boolean(aliases[requestedTheme]);
    const layout=selected.layout||'center', isL=layout.startsWith('l-');
    const mode=selected.mascot_dopamine_mode!==false;
    return {...Object.fromEntries(contentFlags.map(key=>[key,defaultContents.includes(key)])),layout,midline:'adaptive',theme:'pop',corner:isL?layout.slice(2):'top-left',scale:DEFAULT_SCALE,font:'medium',
      mascot_dopamine:'normal',mascot_speed:'1',mascot_bloodshot:false,debug_panel:false,vgraph:false,mascot_band_edges:[...BAND_EDGES],language:'auto',mascot:true,mascot_dopamine_mode:mode,mascot_motion:mode,mascot_eyes:true,mascot_expressions:mode,mascot_flip_right:false,mascot_position:'card',mascot_size:'medium',
      value_display:DEFAULT_VALUE_DISPLAY,band:true,numbers:true,recent:!['minimal','card'].includes(layout),full:isL||layout==='card',status:true,
      details:isL||layout==='data',...selected};
  }
  /** @param {object} options @returns {object} */
  function geometry(options) {
    const l=options.layout.startsWith('l'), ratio=l?options.scale/PERCENT:1;
    const width=Math.round(WIDTH*ratio),height=Math.round(HEIGHT*ratio);
    const right=options.corner.endsWith('right'),bottom=options.corner.startsWith('bottom');
    return {x:right?WIDTH-width:0,y:bottom?HEIGHT-height:0,width,height,
      columnX:right?0:width,columnWidth:WIDTH-width,
      bandY:bottom?0:height,bandHeight:HEIGHT-height};
  }
  /** @param {object} options @returns {boolean} */
  function isBattle(options) {
    return !options.debug_panel&&['pop','pop-cream','stylish','plain'].includes(options.theme);
  }
  return {resolve,normalize,geometry,flags,catalog,WIDTH,HEIGHT,isBattle,DEFAULT_VALUE_DISPLAY};
})();

// 入力・実行状態を同じ言葉で案内する。判定値には触れない。
/** @param {object} state @param {string} language @returns {object} */
window.DistributionStatus=(state,language='ja')=>{
  const d=state.diagnostics||{},en=language==='en';
  const value=(kind,ja,english,next,nextEnglish)=>({kind,title:en?english:ja,next:en?nextEnglish:next});
  if(state.error) return value('error','起動・実行エラー','Startup or runtime error',
    '入力を確認してください。実行中ならトレイの「終了」で停止して開き直してください。続く場合は「表示設定・ログを送る」で記録を確認できます。',
    'Check input. If the app is running, quit and reopen it. If this continues, review the report in Display settings / Send logs.');
  if(d.input_waiting) {
    const parts=(d.input?.message||'試合が映るのを待っています / Waiting for a match to appear').split(' / ');
    return value('waiting','試合待ち','Waiting for a match',parts[0],parts[1]||parts[0]);
  }
  if(d.scanning) return value('checking','入力を確かめています','Checking input',
    'OBS の仮想カメラを探しています。少しお待ちください。','Looking for OBS Virtual Camera. Please wait.');
  if(!state.running) return value('stopped','起動前','Ready to start',
    'OBS の仮想カメラを開始し、上の「起動」を押してください。','Start OBS Virtual Camera, then click Start above.');
  if(d.calibration==='obs_stopped') return value('error','要対応','Needs attention',
    'OBS の仮想カメラが止まっています。OBS の「仮想カメラ開始」を押してください',
    'OBS Virtual Camera has stopped. Click Start Virtual Camera in OBS.');
  if(d.calibration==='overlay_feedback') return value('error','入力の設定を確認','Check input settings',
    'OBS 仮想カメラの出力を「ソース」にして、ゲームのキャプチャだけを選んでください。',
    'Set OBS Virtual Camera output to Source and select only the game capture.');
  if(d.calibration==='no_puyo_screen'||d.match_phase==='between_matches') return value('waiting','試合待ち','Waiting for a match',
    '対戦画面が映ると自動で認識を始めます。','Recognition starts automatically when a match appears.');
  if(d.calibration!=='ready') return value('checking','入力を確かめています','Checking input',
    'ゲームの対戦画面を映してください。画面調整は自動で進みます。',
    'Show a match screen. Screen calibration runs automatically.');
  if(d.display_status==='hold') return value('hold','認識中 · 更新を保留','Recognizing · update on hold',
    '入力を確認しています。勝率は前回の値を表示しています。',
    'Checking input. The win rate shows the previous value.');
  return value('live','認識中','Recognizing',
    '対戦に合わせて勝率を更新しています。','The win rate updates with the match.');
};
