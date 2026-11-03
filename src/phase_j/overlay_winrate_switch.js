/* 24f: 公開時に記録した選択値を数字・短期・試合全体の同じ描画へ渡す。 */
(() => {
  'use strict';
  const original = window.renderSnapshot;
  document.querySelector('#whole .whole-meta span').textContent = OverlayOptions.isBattle(overlayOptions)?
    '試合の流れ · 1P勝率':'撃ち合い+B3 · 1P勝率 · 試合全体';
  window.renderSnapshot = function(data) {
    const selected = data.evaluations?.winrate_switch;
    if (!selected) return original(data);
    const view = {...data, evaluations: {...data.evaluations,
      practical: {...data.evaluations.practical, p1_win_probability: selected.line1}}};
    original(view);
    const heading = document.getElementById('heading');
    if(!OverlayOptions.isBattle(overlayOptions)) heading.textContent = '撃ち合い+B3';
    heading.title = `出所: ${selected.source1} / B3: ${selected.b3 == null ? '—' : (selected.b3*100).toFixed(1)+'%'}`;
  };
})();
