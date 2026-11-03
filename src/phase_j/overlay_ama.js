/* 本人用試験の3枠。amaの保持値・打てない状態を勝率と一緒に表示する。 */
(() => {
  'use strict';
  const card = document.createElement('section');
  card.id = 'ama-card';
  card.setAttribute('aria-label', 'ama 想定の勝率');
  card.innerHTML = '<div class="ama-label">ama 想定の勝率 <small>別枠 · 各側1回読み</small></div>' +
    '<div class="ama-values"><span>1P <strong id="ama-p1">—</strong></span>' +
    '<span>2P <strong id="ama-p2">—</strong></span></div>' +
    '<div id="ama-status">初回入力待ち</div><div id="ama-plan"></div>';
  document.getElementById('panel').appendChild(card);
  const style = document.createElement('style');
  style.textContent = '#ama-card{margin-top:10px;padding:10px 14px;border:2px solid #72d9bc;' +
    'border-radius:10px;background:#122b2c;color:#fff;font:600 18px/1.35 Meiryo,sans-serif}' +
    '.ama-label{color:#97ead1}.ama-label small{display:block;font-size:12px;color:#aecac1}' +
    '.ama-values{display:flex;justify-content:space-between;margin:4px 0}.ama-values strong{font-size:30px}' +
    '#ama-status,#ama-plan{font-size:14px;white-space:normal;color:#d0e6df}' +
    '.l-layout #panel .brand{margin-bottom:8px;font-size:22px}' +
    '#stage.l-layout #panel .scores{margin:8px 0}#stage.l-layout #panel .player{margin:4px 0}' +
    '#stage.l-layout #panel .number{font-size:54px}#stage.l-layout #panel .name{font-size:28px}' +
    '#stage.l-layout #panel .trend{margin-top:10px}#stage.l-layout #panel #trend{height:54px;margin-top:6px}' +
    '#stage.l-layout #panel .trend-meta{font-size:16px}';
  document.head.appendChild(style);
  const original = window.renderSnapshot;
  window.renderSnapshot = function(data) {
    original(data);
    const evaluations = data.evaluations || {};
    document.getElementById('heading').textContent = evaluations.b3_model_label || '実戦勝率: B3';
    const value = evaluations.ama_plain;
    const fmt = p => Number.isFinite(p) ? (100*p).toFixed(p === 0 || p === 1 ? 0 : 1)+'%' : '—';
    document.getElementById('ama-p1').textContent = fmt(value && value.p1);
    document.getElementById('ama-p2').textContent = fmt(value && value.p2);
    document.getElementById('ama-status').textContent = value ?
      (Number.isFinite(value.age_sec) ? value.age_sec.toFixed(1)+'秒保持 · ' : '')+value.reason : '初回入力待ち';
    document.getElementById('ama-plan').textContent = value && value.sides ? value.sides.map((p,i) =>
      (i+1)+'P: '+(p.no_safe_move ? '打てない' : p.no_main ? '本線なし' : '本線あり')).join(' / ') : '';
  };
})();
