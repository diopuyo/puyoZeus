"""発火前の最善手の値 Phase 4: 入力の安定化・片側ごとの再利用・値の保持 (2026-10-01、既定 OFF)。

Phase 3 (src/prefire_best_play_layer.py) との違い:
1. 計算の鍵 = 片側ごとに (盤面, 手に持つ組・NEXT・NEXT2, S3 に渡す NEXT 読み)。NEXT は STABLE_FRAMES フレーム
   続いた読みだけを使い、手に持つ組と NEXT は記録の NEXT のずれを考慮して割り当てる (src/prefire_stable_queue.py)。
   探索・応手・死亡証明の結果は片側の入力だけで使い回す (prefire_best_play の lru と換算率鍵のキャッシュ)。
2. 評価器は s3 に固定する (Phase 3 の遅れ0の診断で唯一 q 全体を改善した構成)。
3. 値の保持: 最善の選択 (採った側と手数) が HOLD_SEC 続くまで新しい選択を表示に出さない。
   出すまでの間は、直前に表示した選択の補正量 (logit の差) を今の現在値に足す。
   撃ち合い中 (tracker.source != 'G_fe') は層を使わず、保持も捨てる。発火・着弾・確定死亡は評価器の値がそのまま出る。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from src import prefire_exchange_sim as sim
from src.exchange_event_overlay import ConfirmedSide
from src.indicators_v2 import SEC_PER_HAND
from src.prefire_best_play_layer import (BOTH_SIDES, NO_SIDE, PREFIRE_SOURCE, SIDES, BestPlayPrefireLayer,
                                         build_context, combine, side_best)
from src.prefire_exchange_layer import _now_ms
from src.prefire_stable_queue import StableQueues

HOLD_SEC = SEC_PER_HAND / 2       # 最善の選択がこの時間続いたら表示に出す (1手の半分。採点前に固定)
EVALUATOR = 's3'
EPS = 1e-6
ZERO_QUEUE = np.zeros(4, dtype=int)   # 採用読みがまだ無い側の S3 入力 (未読と同じ扱い、ちらつく生の読みは使わない)
TRACE_COLUMNS = ('t_sec', 'game_idx', 'p_current', 'p_shown', 'v_1p', 'v_2p', 'hand_1p', 'hand_2p',
                 'lethal_1p', 'lethal_2p', 'chosen', 'chosen_hand', 'accepted', 'held', 'shown_lethal', 'ready_sec')


def _logit(p: float) -> float:
    q = min(1 - EPS, max(EPS, p))
    return math.log(q / (1 - q))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@dataclass
class DecisionHold:
    """最善の選択の切り替えを HOLD_SEC だけ遅らせる (値の安定化、状態は wrapper のここだけ)。"""

    hold_sec: float = HOLD_SEC
    accepted: tuple[int, int] = (NO_SIDE, 0)
    delta: float = 0.0
    candidate: tuple[int, int] | None = None
    since: float = 0.0
    lethal: bool = False      # 表示中の補正が E35 の確実な勝ち由来か (監査用)

    def reset(self) -> None:
        """撃ち合い・試合境界で保持を捨てる。"""
        self.accepted, self.delta, self.candidate, self.since, self.lethal = (NO_SIDE, 0), 0.0, None, 0.0, False

    def update(self, t_sec: float, p0: float, decision: tuple[int, int] | None, value: float,
               lethal: bool = False) -> tuple[float, bool]:
        """表示する値と、保持中 (新しい選択を待っている) かを返す。decision=None は結果がまだ無い。"""
        if decision is not None and decision == self.accepted:
            self.candidate = None
            active = decision[0] != NO_SIDE
            self.delta, self.lethal = (_logit(value) - _logit(p0), lethal) if active else (0.0, False)
            return (value if active else p0), False
        if decision is not None:
            if decision != self.candidate:
                self.candidate, self.since = decision, t_sec
            if t_sec - self.since >= self.hold_sec:
                self.accepted, self.candidate = decision, None
                return self.update(t_sec, p0, decision, value, lethal)
        return (p0 if self.delta == 0.0 else _sigmoid(_logit(p0) + self.delta)), True


def _decision_detail(best: tuple, chosen: int) -> tuple[int, bool]:
    """採った選択の手数と、それが確実な勝ちか (同時なら両側とも同じ手数、確実な勝ちとは扱わない)。"""
    if chosen == NO_SIDE:
        return 0, False
    if chosen == BOTH_SIDES:
        return min(b.hand for b in best if b is not None), False
    return best[chosen].hand, bool(best[chosen].lethal)


class StableBestPlayLayer(BestPlayPrefireLayer):
    """Phase 4 の予測層 (外部 wrapper。評価器の状態は変えない)。"""

    def __init__(self, latency_sec: float = 0.0, hold_sec: float = HOLD_SEC) -> None:
        super().__init__(latency_sec=latency_sec, evaluator=EVALUATOR)
        self.queues = StableQueues()
        self.hold = DecisionHold(hold_sec)
        self.computes: list[tuple[float, float]] = []
        self._colors: set[int] = set()
        self._color_game: int | None = None
        self._hold_game: int | None = None
        self._pending: list[tuple[float, tuple]] = []
        self._shown_key: tuple | None = None

    def apply(self, overlay: Any, t_sec: float, game_idx: int) -> None:
        """毎フレーム呼ぶ。静止区間なら保持つきの最善手の値で表示値を差し替える。"""
        self.queues.update(overlay)
        if overlay._game != self._hold_game:
            self._drop()
            self._hold_game = overlay._game
        tracker = overlay.tracker
        latest = self._latest(overlay)
        if tracker.current is not None or tracker.source != 'G_fe' or tracker.probability is None or latest is None:
            self._drop()
            return
        sides, known = self._stable_sides(latest)
        colors = self._game_colors(overlay, latest, known)
        elapsed = max(0.0, t_sec - overlay._start)
        key = (tuple((s.board._grid.tobytes(), k, s.queue.tobytes()) for s, k in zip(sides, known)),
               colors, sim.effective_rate(elapsed), overlay._game)
        self._schedule(overlay, key, sides, elapsed, known, colors, t_sec)
        self._show(tracker, t_sec, game_idx)

    def _drop(self) -> None:
        """撃ち合い・試合境界: 保持と、届く前の結果・表示中の結果を捨てる (撃ち合い前の盤面の結果を後で使わない)。"""
        self.hold.reset()
        self._pending.clear()
        self._shown_key = None

    def _schedule(self, overlay: Any, key: tuple, sides: tuple, elapsed: float, known: tuple,
                  colors: tuple[int, ...], t_sec: float) -> None:
        """非同期の模擬: 新しい入力の結果は計算開始から latency_sec 後に届き、届いた最新の結果を表示に使う。

        届くまでの間に入力が変わっても、届いた結果は使う (実時間の作業者は最新の完了結果を出す。時刻 t までの観測だけ)。
        計算済みの入力に戻ったときは、その結果をすぐ使う (再計算しない。まだ届いていない古い入力の結果は捨てる)。
        """
        if key in self._cache:
            self._shown_key = key        # 今の入力の結果が既にある: 届く前の古い入力の結果は要らない
            self._pending.clear()
        else:
            started = _now_ms()
            self._cache[key] = self._compute_stable(overlay, sides, elapsed, known, colors)
            self.computes.append((t_sec, _now_ms() - started))
            self._ready[key] = t_sec + self.latency_sec
            self._pending.append((self._ready[key], key))
        while self._pending and self._pending[0][0] <= t_sec:
            self._shown_key = self._pending.pop(0)[1]

    def _show(self, tracker: Any, t_sec: float, game_idx: int) -> None:
        """保持を通して表示値を決め、内訳を trace に残す。"""
        p0 = tracker.probability
        key = self._shown_key
        ready = key is not None
        best = self._cache[key] if ready else (None, None)
        value, chosen = combine(p0, best) if ready else (p0, NO_SIDE)
        hand, lethal = _decision_detail(best, chosen)
        shown, held = self.hold.update(t_sec, p0, (chosen, hand) if ready else None, value, lethal)
        shown_lethal = self.hold.accepted[0] + 1 if self.hold.lethal else 0   # 1=1P の確実な勝ち、2=2P
        self.trace.append((t_sec, game_idx, p0, shown, *(np.nan if b is None else b.value for b in best),
                           *(0 if b is None else b.hand for b in best),
                           *(False if b is None else b.lethal for b in best),
                           chosen, hand, self.hold.accepted[0], held, shown_lethal,
                           self._ready.get(key, np.nan) if ready else np.nan))
        if shown == p0:
            return   # 補正なしなら表示も由来も書き換えない (OFF とビット一致)
        self._written = (shown, p0, tracker.source)
        tracker.probability, tracker.source = shown, PREFIRE_SOURCE

    def _stable_sides(self, latest: tuple) -> tuple[tuple, tuple]:
        """両側の (盤面 + 採用読みの NEXT) と、(手に持つ組, NEXT, NEXT2)。"""
        sides, known = [], []
        for side, queue in zip(latest, self.queues.sides):
            reading = queue.reading()
            sides.append(ConfirmedSide(side.t_sec, side.board, ZERO_QUEUE if reading is None else reading))
            known.append(queue.known())
        return tuple(sides), tuple(known)

    def _game_colors(self, overlay: Any, latest: tuple, known: tuple) -> tuple[int, ...]:
        """試合の中で今までに見えた色 (盤面と採用した組。増えるだけなので鍵がちらつかない)。"""
        if overlay._game != self._color_game:
            self._colors, self._color_game = set(), overlay._game
        grids = [np.asarray(s.board._grid, dtype=np.int8) for s in latest]
        self._colors |= set(sim.seen_colors(grids, list(known)))
        return tuple(sorted(self._colors))

    def _compute_stable(self, overlay: Any, sides: tuple, elapsed: float, known: tuple,
                        colors: tuple[int, ...]) -> tuple:
        """両側の最善の発火 (s3)。探索・応手・証明は片側の入力で使い回される。"""
        try:
            ctx = build_context(overlay, sides, known, elapsed, self._features)
        except (ValueError, TypeError, FloatingPointError):
            return (None, None)
        ctx.colors = colors
        result = []
        for attacker in SIDES:
            try:
                result.append(side_best(ctx, attacker, EVALUATOR))
            except (ValueError, TypeError, FloatingPointError):
                result.append(None)
        return tuple(result)

    def save(self, path: Path) -> None:
        """由来の記録と、計算した回の所要 (遅れがあっても残す)。"""
        data = np.asarray(self.trace, dtype=float).reshape(-1, len(TRACE_COLUMNS))
        computes = np.asarray(self.computes, dtype=float).reshape(-1, 2)
        np.savez_compressed(path, columns=np.asarray(TRACE_COLUMNS), values=data, computes=computes)
