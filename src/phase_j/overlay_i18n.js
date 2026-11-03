'use strict';
// 公開DTOと数値はそのままに、表示用テキストとアクセシビリティ属性だけ翻訳する。
const PUYO_TRANSLATIONS={
 'デザイン':'Design','リボン':'Ribbon','ミニ帯':'Mini strip','サイド柱':'Side column',
 '目を動かす':'Animate eyes','表情を変える':'Change expressions',
 'ドーパミンの動きを有効にする':'Enable dopamine motion',
 '配置と表示内容を選んで、OBSに貼るURLを作成します。既定は真ん中の柱・ポップ (黒) です。':'Choose a layout and contents, then copy the URL into OBS. The default is Center column with Pop (black) design.',
 'ゼウスくん':'Zeus-kun','eスポーツ':'Esports','ポップ':'Pop','ポップ (黒)':'Pop (black)','真ん中の柱':'Center column','小さい帯':'Mini strip','縦':'Side column','L字':'L-shaped','ポップ (クリーム)':'Pop (cream)',
 'グラフを縦にする (サイド柱だけ・試験)':'Vertical graph (side column only, experimental)','ノーマル':'Normal',
 '中央カード':'Center card','リボン（細い帯）':'Ribbon (thin strip)','最小表示':'Minimal',
 '縦ゲージ':'Vertical gauge','L字・左':'L-shaped left',' · 前回値':' · Previous value',
 'A 対戦ゲージ':'A Battle gauge','スタイリッシュ':'Stylish','ふつう':'Plain','直近 30 秒':'Last 30 seconds','直近 30 s':'Last 30 seconds','変化なし':'No change',
 '表示の位置':'Overlay position','表示をドラッグして移動':'Drag to move the overlay',
 '見本の表示をドラッグ。端に吸着します。座標はパネル左上です。':'Drag the preview. Snaps to edges. Coordinates refer to the panel top left.',
 '中央に戻す':'Reset to center','左上に戻す':'Top left','右上に戻す':'Top right','左下に戻す':'Bottom left','右下に戻す':'Bottom right',
 '注意：盤面・NEXT・得点・予告・WINと重なる可能性があります。実画面で確認してください。':'Warning: may overlap boards, NEXT, scores, incoming garbage or WIN. Check the actual game.',
 '主要なゲーム表示を避けた位置です。実画面でも確認してください。':'Outside the main game areas. Check the actual game too.',

 '表示の映り込み':'Overlay in input',
 '表示の映り込み：仮想カメラの出力をソースにしてください':'Overlay in input: set Virtual Camera output to Source',
"いまの状態":"Current status",
"入力を自動判定して起動":"Detect input and start",
"OBS 自動連携（任意）":"OBS integration (optional)",
"OBS 連携をオンにする":"Enable OBS integration",
"WebSocket ポート":"WebSocket port",
"OBS パスワード":"OBS password",
"空欄なら保存済みのパスワードを使います。パスワードはこのPCだけに保存し、ログには含めません。":"Leave blank to keep the saved password. It stays on this PC and is excluded from reports.",
"接続・シーンを確認":"Check connection and scene",
"今のシーンに表示を追加・仮想カメラ開始":"Add overlay to current scene and start Virtual Camera",
"仮想カメラの出力":"Virtual Camera output",
"OBS の「仮想カメラ開始」の横の歯車を開く":"Open the gear next to Start Virtual Camera in OBS",
"出力の種類を「ソース」にする":"Set Output type to Source",
"ゲーム映像のキャプチャを選ぶ（表示用ブラウザは選ばない）":"Choose the game capture source (exclude the overlay browser)",
"🎮 ゲーム映像":"\ud83c\udfae Game capture",
"⚙ ソース":"\u2699 Source",
"仮想カメラ":"Virtual Camera",
"WebSocket では出力の種類を確認・変更できません。シーンにブラウザがある場合は映り込みに注意してください。":"WebSocket cannot inspect or change the output type. A scene containing a browser may include the overlay in the input.",
"入力の解像度":"Input resolution",
"較正":"Calibration",
"判定の遅れ":"Analysis delay",
"最後の更新":"Last update",
"入力を確認しています":"Checking inputs",
"OBS の仮想カメラの出力を「ソース」にして、ゲームの映像だけを選んでください。":"Set OBS Virtual Camera output to Source and select only the game capture.",
    'ゼウスくんのドーパミン':"Zeus-kun's dopamine", '低':'Low', '普通':'Normal', '高':'High',
    '高い':'High', 'エレクション':'Election',
    '目の充血':'Bloodshot eyes over time', '目のスピード':'Eye speed',
    '起動から30分・45分・60分で充血が強くなります。再起動でリセット。':'Increases at 30, 45 and 60 minutes after app startup. Restart to reset.',

    'アンインストール':'Uninstall',
    'アンインストールを開始':'Start uninstall',
    'アプリを終了し、アンインストールを開始しますか？':'Exit the app and start uninstalling?',
    '開いた画面で続けてください。このページは閉じられます。':'Continue in the window that opened. You can close this page.',
    'インストーラー版は下のボタン、スタートメニュー、または Windows の「アプリ」から削除できます。zip版は下のボタンか同梱の Uninstall.bat（アンインストール.bat）を実行し、登録解除後に展開フォルダを削除してください。設定・ログ・記録は既定で残します。削除を選んだ場合だけ消します。':'For the installer edition, use the button below, Start menu, or Windows Apps. For the ZIP edition, use the button below or run the included Uninstall.bat, then delete the extracted folder after cleanup. Settings, logs and recordings are kept by default and removed only if you choose to delete them.',
    'puyoZeus Live — 対戦分析':'puyoZeus Live — Match Analysis',
    'puyoZeus Live':'puyoZeus Live',
    '表情の境界（1P勝率%、昇順4値）':'Expression thresholds (1P win %, four ascending values)',
    'ゼウスくんの表情を勝率で変える':'Change Zeus-kun expressions with win probability',
    '右側に置いたゼウスくんを左右反転する':'Mirror the Zeus-kun when placed on the right',
    '1P視点の5段階。通常の表情は最低1.5秒保持し、決着・待機はすぐ反映します。':'Five bands from the 1P perspective. Hold each expression for at least 1.5 seconds; results and waiting apply immediately.',
    'puyoZeus Live 表示設定':'puyoZeus Live Display Settings',
    '配置と表示内容を選んで、OBSに貼るURLを作成します。既定は中央カードです。':'Choose a layout and contents, then copy the URL into OBS. The default is a centered card.',
    'おじゃまは「個」、連鎖は「連鎖」、猶予は「手」で表示。不明な値は「—」。猶予と返し火力は受ける側の予想です。「(目安)」は予想できないときの返し火力の上限を示します。':'Garbage is measured in puyos, chains in links, and time to respond in moves. Unknown values show —. Response estimates refer to the receiving player. (Upper bound) marks the maximum response when prediction is unavailable.',
    '見えている2手と、その次に理想のツモが来た場合の1手を合わせ、3手以内で撃てる最大火力':'Maximum attack within three moves: the two visible pieces followed by one ideal piece.',
    '画面の式から読み取った現在の連鎖数。読めない間は直前の値を保持':'Current chain link read from the screen; the previous value is held when unreadable.',
    '読み取れた連鎖数。予想できない場合は実際の何連鎖目かと一致しないことがあります':'Observed chain count; it may differ from the actual link when prediction is unavailable.',
    '予想できないため、返し火力の上限を目安として表示':'Prediction unavailable; showing the upper bound of the response attack.',
    '返しの過不足 (＋なら返せる・−なら足りない)':'Response margin (+ enough to counter / − shortfall)',
    '予想を使わず、確定した盤面で判定':'Using the confirmed board without prediction',
    '勝率は変動をならして表示':'Win probability is smoothed',
    '変動をならす前の勝率':'Unsmoothed win probability','確定した盤面の勝率':'Confirmed-board win probability',
    '判定の中身と試合全体の推移':'Analysis details and full-match history',
    '試合全体の1P勝率。中央線は50%':'Full-match 1P win probability. Center line: 50%',
    '試合開始からの合計':'Total since match start',
    'OBSブラウザソース用URL':'OBS browser source URL','OBSの配置':'OBS placement',
    '仮想カメラ：ゲームのキャプチャだけをソース出力。認識入力は1920 × 1080を維持します。':'Virtual camera: output only the game capture as Source. Keep recognition input at 1920 × 1080.',
    'ゲームの部分は透明です。下寄せ時の横帯は上側に表示します。':'The game area is transparent. Bottom-aligned layouts put the band above the game.',
    'ブラウザソース：':'Browser source: ','ゲーム：位置':'Game: position','縮小率':'Scale','、位置':' / position','、最上面':' / top layer',
    '見本用の実フレーム（1920×1080推奨）':'Optional local preview image (1920×1080 recommended)',
    '画像を選ぶ':'Choose image','未選択':'No image selected',
    '画像はこのブラウザ内だけで使用し、送信・保存しません。':'Images stay in this browser and are neither uploaded nor saved.',
    '実フレームを選ぶと、ゲーム画面を含めた配置を確認できます。':'Choose a local image to preview the layout, or use this diagram background.',
    '選択したゲームの実フレーム':'Selected local preview image',
    '数値：現在の対戦データ／背景：選択した静止画':'Values: current match / Background: selected still image',
    '数値と推移は見本用のサンプルです。':'Values and history are illustrative samples.',
    '設定ファイルの既定値として使う':'Use as configuration defaults',
    'puyo_live.json の最上位に以下の overlay を追加・置換し、次回起動時に反映します。他の設定は残してください。':'Add or replace the overlay object in puyo_live.json. Keep other settings. Changes apply on next launch.',
    '選択したURLを Ctrl+C でコピーしてください':'Press Ctrl+C to copy the selected URL',
    '設定ファイルの既定に戻す':'Reset to saved defaults','コピーしました':'Copied',
    '表示言語':'Display language','OSの言語':'OS language','日本語':'Japanese',
    'ゼウスくんを表示':'Show Zeus-kun','ゼウスくん':'Mascot','勝率が動いたときの動き':'Animate large probability changes',
    'カード横':'Beside card','左下':'Bottom left','右下':'Bottom right','大きさ':'Size',
    'テーマ':'Theme','従来（既定）':'Classic (default)','ネオンガラス':'Neon glass','大会中継':'Tournament','ぷよの丸み':'Round shapes','黒と白金':'Black and platinum',
    '50%の示し方':'50% reference','上下の淡い塗り分け':'Two-tone shading','細い実線＋50目盛り':'Thin line + 50 mark','柔らかな光帯':'Soft glow','勝率線の上下色分け（既定）':'Two-color probability line (default)',
    '重ねる：中央カード':'Overlay: centered card','重ねる：リボン':'Overlay: ribbon','重ねる：最小':'Overlay: minimal','重ねる：判定の詳細付き':'Overlay: analysis details',
    'L字型の配置':'L-shaped placement','L字型':'L-shaped','配置':'Layout','ゲームを寄せる隅':'Game corner',
    '左上（データは右と下）':'Top left (data right and below)','右上（データは左と下）':'Top right (data left and below)',
    '左下（データは右と上）':'Bottom left (data right and above)','右下（データは左と上）':'Bottom right (data left and above)',
    'ゲーム縮小率':'Game scale','横帯を表示（下寄せ時は上側）':'Show band (above bottom-aligned game)',
    '文字の大きさ':'Text size','表示する内容':'Visible contents','勝率の数字':'Win probability numbers',
    '直近30秒の推移':'Last 30 seconds','試合全体の推移':'Full-match history','判定の中身':'Analysis details',
    '予告ぷよ':'Incoming garbage','送ったおじゃま':'Garbage sent','相殺':'Garbage countered','おじゃまレート':'Garbage conversion rate',
    '猶予':'Time to respond','返し火力':'Response attack','返しの過不足':'Response margin','(目安)':'(Upper bound)',
    '予想できず':'Prediction unavailable','連鎖中':'Chain active','連鎖の進み':'Chain progress','予想連鎖':'Predicted chain','残り連鎖':'Remaining chain',
    '残り時間':'Time remaining','すぐ撃てる火力':'Immediate attack potential','色ぷよ数':'Colored puyo count','おじゃま数':'Garbage count',
    '色ぷよ':'Colored puyos','盤面・火力':'Board / attack','通常判定':'Build evaluation','撃ち合い判定':'Exchange evaluation','判定方法':'Evaluation method',
    '見本の数値':'Preview values','序盤（サンプル）':'Opening (sample)','撃ち合い（サンプル）':'Exchange (sample)','決着（サンプル）':'Result (sample)',
    '現在の対戦データ':'Current match data','表示の見本':'Overlay preview','URLをコピー':'Copy URL',
    '保存しました':'Saved','設定を保存':'Save settings','ログを送る':'Send logs','初期設定':'Initial setup','送信内容の確認':'Review report',
    'アプリを終了':'Quit app','終了しました。このページを閉じてください。':'Stopped. You can close this page.',
    '起動・実行エラー。ログを送るから内容を確認できます。':'Startup or runtime error. Select Send logs to review the report.',
    '新しい版があります:':'New version available:',
    '自動送信はしません。保存済みの未送信ログも含みます。':'Nothing is sent automatically. Previously saved unsent reports are included.',
    '確認して送信':'Confirm and send','キャンセル':'Cancel','送信しました':'Sent',
    '送信できませんでした。手元に保存しました。次回確認して再送できます。':'Could not send. Saved locally for your next manual review and retry.',
    '画面調整中':'Calibrating','試合待ち':'Waiting for a match','入力確認中':'Checking input','判定エラー':'Evaluation error','判定待ち':'Waiting for evaluation','判定中':'Evaluating',
    '対戦中':'Playing','試合結果':'Match result','決着を確認':'Result confirmed','決着':'Result','更新待ち':'Waiting for update','接続待ち':'Waiting for connection',
    '勝率は未取得':'Win probability unavailable','1Pの勝率':'1P win probability','勝率':'Win probability','データ待ち':'Waiting for data','試合全体':'Full match','直近30秒':'Last 30 seconds',
    '連鎖後の予想':'Post-chain prediction','予想込み':'Includes prediction','受信開始から':'Since connection','受信開始':'Connected at',
    '五分':'Even','有利':'Ahead','不利':'Behind','勝ち':'Win','負け':'Loss',
    '（合計）':' (total)','（N手）':' (moves)','（N連鎖目）':' (link number)','（秒）':' (seconds)',
    'の印':' indicator','連鎖目':' links','連鎖':' chain','予想 ':'Predicted ','残り ':'Remaining ','基準 ':'At ',
    '1P推移':'1P history','1P勝率':'1P win probability','現在':'Now','位置':'Position','対戦の勝率':'Match win probability',
    '返し':'Response','おじゃま':'Garbage','状態':'Status','点/個':' points/puyo','個':' puyos','秒':' s','手':' moves'
  };
window.PuyoI18n=(()=>{
  const dictionary=PUYO_TRANSLATIONS;
  const exact={'小':'Small','中':'Medium','大':'Large','あり':'Yes','なし':'No','連鎖':'Chain'};
  const pattern=new RegExp(Object.keys(dictionary).sort((a,b)=>b.length-a.length).map(s=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|'),'g');
  let language='ja';const originals=new WeakMap(),attributes=new WeakMap();
  const originalTitle=document.title;
  /** @param {string} text @returns {string} */
  function t(text) {return language==='ja'?text:exact[text]||text.replace(pattern,key=>dictionary[key]);}
  /** @param {Node} root @returns {void} */
  function translate(root=document.body) {
    if(!root) return;
    const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
    while(walker.nextNode()) {
      const node=walker.currentNode;if(node.parentElement?.closest('script,style,textarea,pre')) continue;
      const saved=originals.get(node),source=saved&&node.nodeValue===saved.output?saved.source:node.nodeValue;
      const output=t(source);originals.set(node,{source,output});if(node.nodeValue!==output) node.nodeValue=output;
    }
    for(const node of root.querySelectorAll('[title],[aria-label],[alt]')) {
      for(const attr of ['title','aria-label','alt']) if(node.hasAttribute(attr)) {
        const saved=attributes.get(node)||{},old=saved[attr],value=node.getAttribute(attr);
        const source=old&&value===old.output?old.source:value,output=t(source);
        saved[attr]={source,output};attributes.set(node,saved);if(value!==output) node.setAttribute(attr,output);
      }
    }
  }
  /** @param {string} requested @returns {void} */
  function setLanguage(requested) {
    language=requested==='auto'?(String(window.overlayDefaults?.os_language||navigator.language).startsWith('ja')?'ja':'en'):requested;
    if(!['ja','en'].includes(language)) language='ja';document.documentElement.lang=language;document.title=t(originalTitle);translate();
  }
  const observer=new MutationObserver(()=>translate());
  setLanguage(OverlayOptions.resolve(window.overlayDefaults,location.search).language);
  observer.observe(document.body,{subtree:true,childList:true,characterData:true});
  const style=document.createElement('style');style.textContent='html[lang=en] .content-row{grid-template-columns:minmax(130px,1.7fr) 1fr 1fr}html[lang=en] .content-row>span:first-child{overflow-wrap:anywhere}dialog{color:#eef4ff;background:#142238;max-width:90vw;border:1px solid #758ba8;border-radius:12px}dialog pre{max-height:50vh;overflow:auto;white-space:pre-wrap}';document.head.append(style);
  return {t,setLanguage,translate,dictionary};
})();
