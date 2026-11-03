/* 24h: 数字とグラフは同じpracticalを使い、表示名だけを設定する。 */
(() => {
  'use strict';
  const original = window.renderSnapshot;
  window.renderSnapshot = function(data) {
    original(data);
    const unified = data.evaluations?.live24h;
    if (!unified || data.terminal?.state === 'confirmed') return;
    document.getElementById('heading').textContent =
      unified.ama ? '撃ち合い+B3' : '実戦勝率: B3 (C2)';
  };
})();
