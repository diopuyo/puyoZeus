'use strict';
// 決着の側だけを記録する。急変や決着で板を動かす演出は使わない。
const THEME_ROLL_MS=360;
/** @returns {void} */
function resetThemeMotion() {
  delete el('stage').dataset.motion;delete el('stage').dataset.winner;
}
/** @param {number} p @param {string|null} winner @param {boolean} pending */
function updateThemeMotion(p,winner,pending) {
  const stage=el('stage');
  if(winner) stage.dataset.winner=winner;else delete stage.dataset.winner;
}
