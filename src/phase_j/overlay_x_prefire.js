/* 撃つ直前の暫定 X を文字 1 行だけ出す試験用の欄 (2026-10-22・既定OFF・見た目は後で整える)。
   評価の公開内容 evaluations.x_provisional が無いときは、DOM に何も足さない。
   取り下げは公開内容に状態として来ない (暫定が消えるだけ) ので、暫定だった側が消えたら取り下げとみなす。
   確定は status === 'confirmed' で来るが、次の組が動くまで残るので、見せるのは HOLD_MS だけにする。 */
(() => {
  'use strict';
  const LINE_ID = 'x-prefire-line';
  const SIDE_NAMES = ['1P', '2P'];
  const HOLD_MS = 1500;           // 「確定」「取り下げ」を見せておく時間 (user 指定の 1〜2 秒の中間)
  const PENDING_STALE_MS = 3500;  // 暫定のまま更新が途絶えたら黙って消す (表示側の更新待ちと同じ長さ)
  const NUMBER_LOCALE = 'ja-JP';
  const STATUS_TEXT = {pending: '確定待ち', confirmed: '確定', retracted: '取り下げ'};
  // 既存の補助文 (.detail-note) と同じ控えめな文字。太さはパネルから継承する。
  const LINE_STYLE = 'margin-top:8px;font-size:18px;line-height:1.4;color:#c0cede;white-space:normal';
  // 側ごとの記憶: {key, value, phase, seenAt, until}。phase は pending / confirmed / retracted / done。
  const memory = [null, null];
  let timer = null;

  const usable = entry => Boolean(entry) && ['pending', 'confirmed'].includes(entry.status) &&
    Number.isFinite(entry.score) && Number.isFinite(entry.chain) && Number.isFinite(entry.landing_t);
  const hold = (value, phase, now) => ({key: value.landing_t, value, phase, until: now + HOLD_MS});

  /** 側ごとの記憶を、今回の公開内容で進める。 */
  function observe(index, entry, now) {
    const saved = memory[index];
    if (usable(entry) && entry.status === 'pending') {
      memory[index] = {key: entry.landing_t, value: entry, phase: 'pending', seenAt: now};
    } else if (usable(entry)) {
      // 確定は残り続けるので、同じ暫定から確定へ移った時か、初見の時だけ出し直す。
      if (!saved || saved.key !== entry.landing_t || saved.phase === 'pending') {
        memory[index] = hold(entry, 'confirmed', now);
      }
    } else if (saved && saved.phase === 'pending') {
      memory[index] = hold(saved.value, 'retracted', now);
    } else if (saved && saved.phase === 'done') {
      memory[index] = null;  // 確定の表示が済み、公開内容からも消えた
    }
  }

  /** 見せる時間が過ぎた記憶を片付ける。確定は公開内容に残る間、再表示しないよう done で覚えておく。 */
  function expire(now) {
    memory.forEach((saved, index) => {
      if (!saved) return;
      if (saved.phase === 'pending' && now - saved.seenAt >= PENDING_STALE_MS) memory[index] = null;
      else if (['confirmed', 'retracted'].includes(saved.phase) && now >= saved.until) {
        memory[index] = saved.phase === 'confirmed' ? {key: saved.key, phase: 'done'} : null;
      }
    });
  }

  const sideText = (index, saved) => `暫定 X (${SIDE_NAMES[index]}): ${saved.value.chain}連鎖 ` +
    `${saved.value.score.toLocaleString(NUMBER_LOCALE)}点 ${STATUS_TEXT[saved.phase]}`;

  function removeLine() {
    const line = document.getElementById(LINE_ID);
    if (line) line.remove();
  }

  function showLine(text, status) {
    const panel = document.getElementById('panel');
    if (!panel) return;
    let line = document.getElementById(LINE_ID);
    if (!line) {
      line = document.createElement('div');
      line.id = LINE_ID;
      line.setAttribute('role', 'status');
      line.style.cssText = LINE_STYLE;
      panel.appendChild(line);
    }
    if (line.textContent !== text) line.textContent = text;
    line.dataset.status = status;
  }

  function render() {
    const shown = memory.map((saved, index) => [index, saved]).filter(([, saved]) => saved && saved.phase !== 'done');
    if (!shown.length) return removeLine();
    showLine(shown.map(([index, saved]) => sideText(index, saved)).join(' / '), shown[0][1].phase);
  }

  /** 次に状態が変わる時刻 (見せ終わり・暫定の失効) に、公開内容が来なくても描き直す。 */
  function schedule(now) {
    clearTimeout(timer);
    const deadlines = memory.filter(saved => saved && saved.phase !== 'done')
      .map(saved => saved.phase === 'pending' ? saved.seenAt + PENDING_STALE_MS : saved.until);
    if (!deadlines.length) return;
    timer = setTimeout(tick, Math.max(0, Math.min(...deadlines) - now) + 1);
  }

  function tick() {
    const now = performance.now();
    expire(now);
    render();
    schedule(now);
  }

  function reset() {
    clearTimeout(timer);
    memory[0] = memory[1] = null;
    removeLine();
  }

  function update(data) {
    const field = data && data.evaluations && data.evaluations.x_provisional;
    if (!field || !Array.isArray(field.sides)) return reset();
    const now = performance.now();
    SIDE_NAMES.forEach((_, index) => observe(index, field.sides[index], now));
    expire(now);
    render();
    schedule(now);
  }

  const original = window.renderSnapshot;
  window.renderSnapshot = function(data) {
    original(data);
    // この 1 行の不具合で、既存の勝率表示まで「入力確認中」に落とさない。
    try { update(data); } catch (error) { reset(); }
  };
})();
