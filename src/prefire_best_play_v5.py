"""Phase 5: 完了時刻厳守、保持なし、欠測は両側予測を抑止する既定OFF層。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src import prefire_v5_search as search
from src import prefire_v5_value as value
from src.prefire_best_play_stable import StableBestPlayLayer
from src.prefire_best_play_layer import PREFIRE_SOURCE, logit_mean
from src.prefire_exchange_layer import _now_ms

WAITING = 4
TRACE_COLUMNS = ('t_sec', 'game_idx', 'p_current', 'p_shown', 'v_1p', 'v_2p',
                 'hand_1p', 'hand_2p', 'status_1p', 'status_2p', 'ready_sec', 'used', 'held')


def combine_ready(best: tuple) -> float:
    """従来の先着優先・同手数logit平均を維持。各枝の待機値は設置後評価済み。"""
    first, second = best
    if first[1].consumed != second[1].consumed:
        return first[0] if first[1].consumed < second[1].consumed else second[0]
    return logit_mean(first[0], second[0])


class BestPlayV5Layer(StableBestPlayLayer):
    """Phase 4 の入力安定化だけを再利用する。表示保持は呼ばない。"""

    def __init__(self, latency_sec: float = 0.0) -> None:
        if not np.isfinite(latency_sec) or latency_sec < 0:
            raise ValueError('遅れは有限の非負秒が必要')
        super().__init__(latency_sec=latency_sec)
        self._active_key: tuple | None = None

    def apply(self, overlay: Any, t_sec: float, game_idx: int) -> None:
        """現在入力の完了済み結果のみ反映。入力変更中に古い補正を保持しない。"""
        self.queues.update(overlay)
        tracker, latest = overlay.tracker, self._latest(overlay)
        if tracker.current is not None or tracker.source != 'G_fe' or tracker.probability is None or latest is None:
            self._active_key = None
            return
        sides, known = self._stable_sides(latest)
        pending = value.pending_counts(overlay)
        states = tuple(search.Position(s.board._grid.astype(np.int8).tobytes(), k, p)
                       for s, k, p in zip(sides, known, pending or (0, 0)))
        statuses = tuple(search.status(p.board, p.queue) for p in states)
        if pending is None:
            statuses = tuple(s if s != search.OK else search.MISSING for s in statuses)
        elapsed = max(0.0, t_sec-overlay._start)
        key = (states, elapsed, overlay._game)
        # 経過秒は評価器の入力。丸めて別局面の結果を流用しない。
        identity = (states, statuses, search.sim.effective_rate(elapsed), overlay._game)
        if identity != self._active_key:
            self._active_key = identity
            self._shown_key = key
            self._schedule_v5(overlay, key, states, statuses, elapsed, t_sec)
        self._show_v5(tracker, self._shown_key, statuses, t_sec, game_idx)

    def _schedule_v5(self, overlay: Any, key: tuple, states: tuple, statuses: tuple,
                     elapsed: float, t_sec: float) -> None:
        """完了予定時刻を結果とは別に保存する。欠測にも状態コードを残す。"""
        if key in self._cache:
            return
        started = _now_ms()
        result = (None, None)
        if all(s == search.OK for s in statuses):
            values: dict[tuple, float] = {}
            def evaluator(exchange: search.Exchange, time: float) -> float:
                # 同一計算内の左右探索で、評価器に渡す入力が同一なら再利用する。
                state_key = (tuple((p.board, p.queue, p.pending) for p in exchange.sides), time)
                if state_key not in values:
                    values[state_key] = value.evaluate(overlay, exchange, time)
                return values[state_key]
            result = tuple(value.choose(states, side, elapsed, evaluator) for side in (0, 1))
        self._cache[key] = result
        self._ready[key] = t_sec + self.latency_sec
        self.computes.append((t_sec, _now_ms()-started))

    def _show_v5(self, tracker: Any, key: tuple, statuses: tuple, t_sec: float, game_idx: int) -> None:
        """片側欠測なら合成しない。保持を通さず、現在入力の完了済み値だけ出す。"""
        p0 = tracker.probability
        ready = self._ready[key]
        complete = ready <= t_sec
        best = self._cache[key] if complete else (None, None)
        used = complete and all(s == search.OK for s in statuses) and all(b is not None for b in best)
        shown = combine_ready(best) if used else p0
        codes = statuses if complete or any(s != search.OK for s in statuses) else (WAITING, WAITING)
        self.trace.append((t_sec, game_idx, p0, shown,
            *(b[0] if b is not None else np.nan for b in best),
            *(b[1].consumed if b is not None else 0 for b in best), *codes, ready, used, False))
        if used:
            self._written = (shown, p0, tracker.source)
            tracker.probability, tracker.source = shown, PREFIRE_SOURCE

    def save(self, path: Path) -> None:
        """欠測・遷移・計算待ち・適用の母数を再現できる記録。"""
        np.savez_compressed(path, columns=np.asarray(TRACE_COLUMNS),
            values=np.asarray(self.trace, float).reshape(-1, len(TRACE_COLUMNS)),
            computes=np.asarray(self.computes, float).reshape(-1, 2))
