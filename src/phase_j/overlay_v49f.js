/* OBS試験用の独立欄。B3の数値・色分け・推移は既存描画に任せる。 */
(() => {
  'use strict';
  const card = document.createElement('section');
  card.id = 'v49f-card';
  card.setAttribute('aria-label', '撃ち合い勝率 v4.9f');
  card.innerHTML = '<div class="v49f-label">撃ち合い勝率 <b>v4.9f</b><small>OBS 試験版</small></div>' +
    '<div class="v49f-values"><span>1P <strong id="v49f-p1">—</strong></span>' +
    '<span>2P <strong id="v49f-p2">—</strong></span></div><div id="v49f-status">入力待ち</div>';
  document.getElementById('panel').appendChild(card);
  const style = document.createElement('style');
  style.textContent = '#v49f-card{margin-top:14px;padding:12px 14px;border:2px solid #f6c969;' +
    'border-radius:10px;background:#172334;color:#fff;font:600 18px/1.35 "Meiryo",sans-serif}' +
    '.v49f-label{color:#f6d797}.v49f-label small{float:right;font-size:13px;margin-top:4px}' +
    '.v49f-values{display:flex;justify-content:space-between;margin:5px 0;font-size:18px}' +
    '.v49f-values strong{font-size:32px;font-variant-numeric:tabular-nums}' +
    '#v49f-status{font-size:15px;color:#d5dfec;white-space:normal}' +
    '#v49f-card[data-death="true"]{border-color:#fff;background:#344256}';
  document.head.appendChild(style);
  const original = window.renderSnapshot;
  window.renderSnapshot = function(data) {
    original(data);
    document.getElementById('heading').textContent = 'B3 勝率';
    const v = data.evaluations && data.evaluations.v49f;
    const fmt = p => Number.isFinite(p) ? (p === 0 || p === 1 ? (100*p).toFixed(0) : (100*p).toFixed(1))+'%' : '—';
    document.getElementById('v49f-p1').textContent = fmt(v && v.p1);
    document.getElementById('v49f-p2').textContent = fmt(v && v.p2);
    const dead = v && v.death_sides && v.death_sides.length === 1;
    card.dataset.death = String(!!dead);
    document.getElementById('v49f-status').textContent = v ?
      (dead ? '死亡断定 · ' : '') + (Number.isFinite(v.age_sec) ? v.age_sec.toFixed(1)+'秒前 · ' : '') + v.reason : '入力待ち';
  };
})();
