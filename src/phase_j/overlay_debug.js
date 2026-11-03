'use strict';
// 試験用の情報パネル。overlay.debug_panel=true かつ L字配置のときだけ有効。
// 各側の状態(日本語)・連鎖数・確定盤と読取盤・置ける手数の見込みなどを、受け取った値のまま表示する。
// 勝率・判定の表示には触れない。値は新しく計算せず、公開 payload の既存欄を読むだけ。
window.PuyoDebug=(()=>{
  const ROWS=13,COLS=6,HIDDEN_ROW=0,CELL_PX=26,READABLE_CELL_PX=18,MIN_CELL_PX=10,BOARD_GAP_PX=8,STALE_MS=3500,TICK_MS=500;
  const FILL={'.':'#17212f','1':'#e5484d','2':'#3b82f6','3':'#2fa56a','4':'#f2cf1d','5':'#a259e6','o':'#9aa3b2','?':'#ff3df5'};
  const LETTER={'1':'赤','2':'青','3':'緑','4':'黄','5':'紫','o':'お','?':'?'};
  const DARK_LETTER=new Set(['4','o']);
  const MISMATCH='#ffffff',DIVIDER='#6b7a90';
  const SIDES=['p1','p2'],LABEL={p1:'1P',p2:'2P'};
  // 状態の日本語表示。値の出どころ: debug_panel.pN.state (src/board_state_machine.py の BoardState 名)。
  const STATE_LABEL={TSUMO_FALL:['ツモ落下中','tsumo'],OJAMA_FALL:['おじゃま落下中','ojama'],
    GRAVITY_SETTLE:['連鎖後の着地待ち','settle'],EFFECT:['演出中(全消し等)','effect']};
  const CSS=`
#stage.debug-on #panel .trend,#stage.debug-on #panel #contents-extra,
#stage.debug-on #panel #details,#stage.debug-on #panel #whole{display:none!important}
#stage.debug-on #panel .meta{display:flex;justify-content:space-between;align-items:center;gap:8px}
#stage.debug-on #panel .brand{margin:0;font-size:22px}#stage.debug-on #panel .state{font-size:22px}
#stage.debug-on #panel .scores{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:4px 0}
#stage.debug-on #panel .player{display:flex;margin:0;justify-content:space-between;align-items:baseline}
#stage.debug-on #panel .name{font-size:30px}#stage.debug-on #panel .number{font-size:46px}
#stage.debug-on #panel .unit{font-size:20px}
#debug{margin-top:6px;border-top:2px solid #526783;padding-top:4px;font-weight:700;min-height:0;overflow:hidden}
#debug .dbg-head{font-size:15px;color:#ffda9a;line-height:1.2}
#debug .dbg-status{display:grid;grid-template-columns:1fr;gap:3px;margin-top:4px}
#debug .dbg-line{display:flex;min-width:0;overflow:hidden;align-items:baseline;gap:8px;padding:2px 6px;border-left:6px solid #6b7a90;background:#14202f}
#debug .dbg-who{font-size:20px}#debug .dbg-state{font-size:24px;line-height:1.15;white-space:nowrap}
#debug .dbg-sub{font-size:12px;color:#c0cede;white-space:nowrap;overflow:hidden;min-width:0;margin-left:auto}
#debug .dbg-line.dbg-p1 .dbg-who{color:var(--p1)}#debug .dbg-line.dbg-p2 .dbg-who{color:var(--p2)}
#debug .dbg-line[data-kind="chain"]{border-left-color:#ff9f43;background:#3a2a14}
#debug .dbg-line[data-kind="ojama"]{border-left-color:#c084fc;background:#2c1b3d}
#debug .dbg-line[data-kind="tsumo"]{border-left-color:#4aa8ff;background:#14283d}
#debug .dbg-line[data-kind="stable"]{border-left-color:#3ddc97;background:#12301f}
#debug .dbg-line[data-kind="unstable"]{border-left-color:#ff5d5d;background:#3d1616}
#debug .dbg-line[data-kind="settle"],#debug .dbg-line[data-kind="effect"]{border-left-color:#f2cf1d;background:#38330f}
#debug .dbg-line[data-kind="out"]{border-left-color:#6b7a90;color:#9fb0c6}
#debug .dbg-side{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:3px}
#debug .dbg-title{font-size:13px;line-height:1.15;white-space:nowrap}
#debug .dbg-title.dbg-p1{color:var(--p1)}#debug .dbg-title.dbg-p2{color:var(--p2)}
#debug canvas{display:block;margin-top:1px;background:#0b121c}
#debug .dbg-rows{display:grid;grid-template-columns:96px minmax(0,1fr) minmax(0,1fr);gap:0 6px;margin-top:4px;
  font-size:17px;line-height:1.2;font-variant-numeric:tabular-nums}
#debug .dbg-rows span:nth-child(3n+1){color:#c0cede;font-size:13px;white-space:nowrap}
#debug .dbg-rows span:nth-child(3n+2){color:var(--p1);text-align:right}
#debug .dbg-rows span:nth-child(3n){color:var(--p2);text-align:right}
#debug .dbg-note{font-size:12px;color:#9fb0c6;margin-top:3px;line-height:1.2;white-space:nowrap;overflow:hidden}
#debug.tight .dbg-note{display:none}
#debug[data-stale="true"] .dbg-head{color:#ff8a8a}
#stage.narrow #debug .dbg-rows{font-size:14px;grid-template-columns:70px minmax(0,1fr) minmax(0,1fr)}
#stage.narrow #debug .dbg-state{font-size:20px}`;
  let root=null,received=null;
  /** @param {string} tag @param {string} cls @param {string} text @returns {HTMLElement} */
  function make(tag,cls='',text='') {
    const node=document.createElement(tag);if(cls) node.className=cls;node.textContent=text;return node;
  }
  /** @param {unknown} value @param {string} suffix @returns {string} */
  function num(value,suffix='') {
    return Number.isFinite(value)?`${Number(Number(value).toFixed(1))}${suffix}`:'—';
  }
  /** 連鎖数。画面の式から読んだ段 chain_display_step を優先し、無ければ既存の観測段数 chain_n_steps。 */
  function chainText(content) {
    if(Number.isFinite(content?.chain_display_step)) return `${content.chain_display_step}連鎖目`;
    if(Number.isFinite(content?.chain_n_steps)&&content.chain_n_steps>=1) return `式${content.chain_n_steps}段観測`;
    return '段不明';
  }
  /** 状態の日本語表示 {text, kind}。出どころ: debug_panel.state / match_active / confirmed と b3_contents の連鎖値。 */
  function statusOf(side,content,matchActive) {
    if(!side) return {text:'—',kind:'none'};
    if(matchActive===false||side.state==='MENU') return {text:'試合外',kind:'out'};
    if(side.state==='CHAIN'||content?.chain_active) return {text:`連鎖中 ${chainText(content)}`,kind:'chain'};
    if(STATE_LABEL[side.state]) return {text:STATE_LABEL[side.state][0],kind:STATE_LABEL[side.state][1]};
    if(side.state==='STABLE'&&side.confirmed) return {text:'安定 (確定盤)',kind:'stable'};
    return {text:'認識不安定',kind:'unstable'};
  }
  /** 確定盤と読取盤で違うマスの数。どちらか無ければ null。 */
  function diffCount(side) {
    if(!side?.confirmed||!side?.reading) return null;
    let n=0;
    for(let r=0;r<ROWS;r++) for(let c=0;c<COLS;c++) if(side.confirmed[r][c]!==side.reading[r][c]) n++;
    return n;
  }
  /** @param {HTMLCanvasElement} canvas @param {string[]|null} board @param {string[]|null} other @param {number} cell */
  function drawBoard(canvas,board,other,cell) {
    canvas.width=COLS*cell;canvas.height=ROWS*cell;
    const g=canvas.getContext('2d');g.clearRect(0,0,canvas.width,canvas.height);
    g.font=`800 ${Math.round(cell*0.7)}px "Yu Gothic UI",Meiryo,sans-serif`;
    g.textAlign='center';g.textBaseline='middle';
    for(let r=0;r<ROWS;r++) for(let c=0;c<COLS;c++) {
      const ch=board?board[r][c]:'.';
      g.globalAlpha=r===HIDDEN_ROW?0.55:1;
      g.fillStyle=FILL[ch]||FILL['?'];g.fillRect(c*cell+1,r*cell+1,cell-2,cell-2);
      if(ch!=='.') {
        g.fillStyle=DARK_LETTER.has(ch)?'#101820':'#ffffff';
        g.fillText(LETTER[ch]||'?',c*cell+cell/2,r*cell+cell/2+1);
      }
      g.globalAlpha=1;
      if(other&&other[r][c]!==ch) {
        g.strokeStyle=MISMATCH;g.lineWidth=2;g.strokeRect(c*cell+2,r*cell+2,cell-4,cell-4);
      }
    }
    g.strokeStyle=DIVIDER;g.lineWidth=1;g.setLineDash([4,3]);
    g.beginPath();g.moveTo(0,cell);g.lineTo(canvas.width,cell);g.stroke();g.setLineDash([]);
  }
  /** 横幅に収まる最大のマス寸法 (1行に確定盤と読取盤の2枚)。 */
  function widthLimit() {
    return Math.floor((root.node.clientWidth-BOARD_GAP_PX)/2/COLS);
  }
  /** 試験用欄が列の下端からはみ出していないか (レイアウト px)。 */
  function overflowing() {
    const panel=el('panel');
    const bottom=panel.clientHeight-parseFloat(getComputedStyle(panel).paddingBottom||'0');
    return root.node.offsetTop+root.node.offsetHeight>bottom||root.node.scrollWidth>root.node.clientWidth;
  }
  /** 4枚の盤面 (1P確定,1P読取,2P確定,2P読取) を描く。列から出るあいだ、文字が読める大きさ (18px) まで
   *  マスを小さくし、まだ出るなら注記を隠し、それでも出るときだけ 18px より小さくする。 */
  function drawFitted(boards) {
    root.node.classList.remove('tight');
    let cell=Math.max(MIN_CELL_PX,Math.min(CELL_PX,widthLimit()));
    for(;;) {
      boards.forEach((shown,i)=>drawBoard(root.canvas[i],shown.board,shown.other,cell));
      if(cell<=MIN_CELL_PX||!overflowing()) break;
      if(cell<=READABLE_CELL_PX) root.node.classList.add('tight');
      cell-=1;
    }
  }
  /** @param {object} data */
  function update(data) {
    if(!root) return;
    received=performance.now();
    const panel=data.debug_panel||null, contents=data.evaluations?.b3_contents||null;
    const display=data.display||{}, practical=data.evaluations?.practical||{};
    const input=display.input_status&&display.input_status!=='ready'?`入力:${display.message||display.input_status}`:'入力:正常';
    const quality=data.input?.recognition_quality?.status;
    root.head.textContent=`${input} / 認識:${{trusted:'信頼',partial:'一部欠け'}[quality]||'—'} / 判定元:${practical.source||'—'}`;
    const boards=[];
    SIDES.forEach((key,i)=>{
      const side=panel?.[key],status=statusOf(side,contents?.[key],panel?.match_active);
      root.line[i].dataset.kind=status.kind;root.state[i].textContent=status.text;
      const diff=diffCount(side);
      root.sub[i].textContent=side?[side.state||'—',side.formula?`式${side.formula.left}×${side.formula.right}`:null,
        diff===null?null:`読取差${diff}`].filter(Boolean).join(' '):'';
      root.title[2*i].textContent=`${LABEL[key]} ${side?.confirmed?'確定盤':'確定盤なし'}`;
      root.title[2*i+1].textContent=`${LABEL[key]} ${side?.reading?'読取盤':'読取なし'}`;
      boards.push({board:side?.confirmed||null,other:null},{board:side?.reading||null,other:side?.confirmed||null});
    });
    const rows=[
      ['予想連鎖',s=>contents?.[s]?.chain_active?`${num(contents[s].chain_pred_l,'連鎖')} ${num(contents[s].chain_rem_sec,'秒')}`:'—'],
      ['置ける手数',s=>num(contents?.[s]?.ta_H_med,'手')],
      ['予告おじゃま',s=>num(contents?.[s]?.sl_pending,'個')],
      ['得点',s=>num(panel?.[s]?.score)]];
    root.rows.replaceChildren(...rows.flatMap(([label,fn])=>[make('span','',label),...SIDES.map(s=>make('span','',fn(s)))]));
    const frame=panel?`フレーム ${panel.frame??'—'} · ${num(panel.t_sec,'秒')}`:'盤面の追加欄なし';
    root.note.textContent=`${frame} · 白枠=読取≠確定 · 最上段=隠し段`;
    drawFitted(boards);
    root.node.dataset.stale='false';
  }
  /** 受信が止まったら赤字で知らせる。 */
  function tick() {
    if(!root||received===null) return;
    const age=performance.now()-received;
    root.node.dataset.stale=String(age>STALE_MS);
    if(age>STALE_MS) root.head.textContent=`更新が止まっています(${Math.round(age/1000)}秒前が最後)`;
  }
  /** DOM を作り、L字配置で有効なときだけ表示する。 */
  function init() {
    const options=typeof overlayOptions==='undefined'?null:overlayOptions;
    if(!options||!options.debug_panel||!options.layout.startsWith('l')||OverlayOptions.isBattle(options)) return;
    const style=document.createElement('style');style.textContent=CSS;document.head.append(style);
    const node=make('div');node.id='debug';
    const head=make('div','dbg-head','試験用パネル · データ待ち');
    const status=make('div','dbg-status'),line=[],state=[],sub=[];
    const title=[],canvas=[],pairs=[];
    SIDES.forEach(key=>{
      const row=make('div',`dbg-line dbg-${key}`),who=make('span','dbg-who',LABEL[key]);
      const st=make('span','dbg-state','—'),sb=make('span','dbg-sub');
      row.append(who,st,sb);status.append(row);line.push(row);state.push(st);sub.push(sb);
      const pair=make('div','dbg-side');
      for(let k=0;k<2;k++) {
        const box=make('div'),t=make('div',`dbg-title dbg-${key}`,LABEL[key]),c=document.createElement('canvas');
        box.append(t,c);pair.append(box);title.push(t);canvas.push(c);drawBoard(c,null,null,CELL_PX);
      }
      pairs.push(pair);
    });
    const rows=make('div','dbg-rows'),note=make('div','dbg-note');
    node.append(head,status,...pairs,rows,note);el('panel').append(node);
    el('stage').classList.add('debug-on');
    root={node,head,line,state,sub,title,canvas,rows,note};
    setInterval(tick,TICK_MS);
  }
  return {init,update,statusOf,chainText,get enabled() {return root!==null;}};
})();
