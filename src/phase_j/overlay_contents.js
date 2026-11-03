'use strict';
// 連鎖の進みだけ最新画面観測。他の値は採用済みB3行の生値。
const CONTENT_GROUPS=[
  ['おじゃま',[
    ['pending','予告ぷよ','sl_pending','ojama'],['sent','送ったおじゃま','sl_sent','ojama'],
    ['offset','相殺','sl_offset','ojama'],['rate','おじゃまレート','sl_rate','rate']]],
  ['返し',[
    ['hands','猶予','ta_H_med','hands'],['return','返し火力','ta_q_med','ojama'],
    ['margin','返しの過不足','ta_m_sl_med','margin'],['bound','(目安)','ta_bound','flag'],
    ['unknown','予想できず','unpredictable','flag']]],
  ['連鎖',[
    ['active','連鎖中','chain_active','flag'],['step','連鎖の進み','chain_display_step','step'],
    ['length','予想連鎖','chain_pred_l','prediction'],['remaining','残り連鎖','chain_rem_stages','remaining'],
    ['seconds','残り時間','chain_rem_sec','seconds']]],
  ['盤面・火力',[
    ['k1','すぐ撃てる火力','k1_ojama','ojama'],['colors','色ぷよ','cells_color','count'],
    ['ojama','おじゃま','cells_ojama','ojama']]],
];
const CONTENT_DECIMALS=1;
/** @param {unknown} value @param {string} unit @returns {string} */
function contentValue(value,unit) {
  if(value===null||value===undefined) return '—';
  if(unit==='flag') return value?'あり':'なし';
  if(!Number.isFinite(value)) return '—';
  const number=n=>String(Number(n.toFixed(CONTENT_DECIMALS)));
  if(unit==='margin') return `${value>0?'＋':value<0?'−':''}${number(Math.abs(value))}個`;
  if(unit==='step') return `${number(value)}連鎖目`;
  if(unit==='prediction') return `予想 ${number(value)}連鎖`;
  if(unit==='remaining') return `残り ${number(value)}連鎖`;
  return number(value)+({ojama:'個',hands:'手',seconds:'秒',count:'個',rate:'点/個'}[unit]||'');
}
/** @param {object|null} contents */
function renderContents(contents) {
  const primary=el('contents-main'),extra=el('contents-extra');
  primary.replaceChildren();extra.replaceChildren();
  CONTENT_GROUPS.forEach(([title,fields],index)=>{
    const selected=fields.filter(([option])=>overlayOptions[`content_${option}`]);
    if(!selected.length) return;
    const group=document.createElement('div');group.className='content-group';
    const heading=document.createElement('div');heading.className='content-row content-head';
    for(const text of [title,'1P','2P']) {
      const cell=document.createElement('span');cell.textContent=text;heading.append(cell);
    }
    group.append(heading);
    for(const [option,label,key,unit] of selected) {
      const row=document.createElement('div');row.className='content-row';row.dataset.field=option;
      if(option==='k1') row.title='見えている2手と、その次に理想のツモが来た場合の1手を合わせ、3手以内で撃てる最大火力';
      if(option==='step') row.title='画面の式から読み取った現在の連鎖数。読めない間は直前の値を保持';
      if(option==='margin') row.title='返しの過不足 (＋なら返せる・−なら足りない)';
      if(option==='bound') row.title='予想できないため、返し火力の上限を目安として表示';
      if(option==='sent'||option==='offset') row.title='試合開始からの合計';
      const values=['p1','p2'].map(side=>contentValue(key==='sl_rate'?contents?.[key]:contents?.[side]?.[key],unit));
      for(const text of [label,...values]) {
        const cell=document.createElement('span');cell.textContent=text;row.append(cell);
      }
      group.append(row);
    }
    (index<2?primary:extra).append(group);
  });
  el('contents-model').hidden=!overlayOptions.content_model;
  el('margin-note').hidden=!overlayOptions.content_margin;
  el('contents-model').textContent=`${{static:'通常判定',exchange:'撃ち合い判定'}[contents?.model]||'判定方法：—'}`+
    (Number.isFinite(contents?.t_sec)?` · 基準 ${contents.t_sec.toFixed(1)}秒`:'');
}
