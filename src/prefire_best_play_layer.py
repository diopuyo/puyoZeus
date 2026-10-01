"""発火前の最善手の値を表示勝率へ出す予測層 (2026-10-01、Phase 3、既定 OFF)。

user 決定 (2026-10-01): 撃つかどうかは指し手の選択にすぎない。表示する勝率は各側の最善の選択肢の値。
発火タイミングの学習 (Phase 2 の hazard) は使わない。人のミスは考慮しない。

1つの静止時刻 (tracker.source == 'G_fe') で、各側の選択肢は
  待つ (= 現在値 p0) / 既知の3手以内で撃つ (手に持つ組・NEXT・NEXT2、各手数の最大発火)。
撃つ選択肢の値 (1P 視点) は、
  1. 送り量 = 既存 score_to_ojama (マージンタイムは両者の早い方 = overlay._start 起点)
  2. 受け側の猶予手数 = 連鎖演出の手数 (既存 remaining_hands) + 攻撃側が撃つまでに置く手数
  3. 受け側が「受ける」か「最善応手を返す」の良い方を選ぶ (応手量は既存 MC 応手探索)
  4. 相殺した純受け量で、既存と同じ合成 logit_mean(S3, 仮想着弾 G_fe) を求める
     (ExchangeLandingProjection._evaluate_single / _counter_projection と同じ合成)
  5. 受け側がどう応手しても窒息すると E35 で証明できれば、既存の回避不能死と同じ値 (0.02/0.98)
両側の合成 (二人の意思決定、因果的・単純):
  - 各側は自分に良い方を選ぶ: 1P は max(p0, 撃つ値)、2P は min(p0, 撃つ値)。撃たないと窒息なら待てない。
  - 片側だけが待つより良くなるなら、その値。
  - 両側とも良くなるなら、先に撃てる側 (手数が小さい側) の値。撃つ値には相手の最善応手が入っているので、
    先に撃った側の値がそのまま撃ち合いの帰結になる。同じ手数なら、どちらが先かは観測からは決まらないので
    両者の値の等重み logit 平均。
評価器の内部状態は変えない (Phase 2 と同じく表示の直前だけ差し替え、update の前に戻す)。
計算は非同期を想定する: 盤面・NEXT が変わった時刻から latency_sec 経つまでは結果を出さない (既定 0 = 同期)。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from src import prefire_best_play as search
from src import prefire_exchange_sim as sim
from src.prefire_exchange_layer import PrefireExchangeLayer, known_pairs, _now_ms

PREFIRE_SOURCE = 'G_fe+bestplay'
EVALUATORS = ('full', 'fast', 's3', 'gfe')
SIDES = (0, 1)
NO_SIDE, BOTH_SIDES = -1, 2
TRACE_COLUMNS = ('t_sec', 'game_idx', 'p_current', 'p_shown', 'v_1p', 'v_2p', 'hand_1p', 'hand_2p',
                 'lethal_1p', 'lethal_2p', 'forced_1p', 'forced_2p', 'chosen', 'compute_ms', 'ready_sec')


@dataclass(frozen=True)
class SideBest:
    """片側の最善の発火 (1P 視点の値)。forced=撃たないと窒息 (待つ選択肢がない)。"""

    value: float
    hand: int
    lethal: bool
    forced: bool


def logit_mean(left: float, right: float) -> float:
    """既存の等重み logit 平均 (exchange_event_landing.logit_mean) をそのまま使う。"""
    from src.exchange_event_landing import logit_mean as existing
    return existing(left, right)


def combine(p0: float, best: tuple[SideBest | None, SideBest | None]) -> tuple[float, int]:
    """二人の最善の選択を合成する。戻り値は (1P 勝率, 採った側: -1=どちらも待つ / 0 / 1 / 2=同時)。"""
    first, second = best
    v1 = first.value if first is not None and (first.forced or first.value > p0) else None
    v2 = second.value if second is not None and (second.forced or second.value < p0) else None
    if v1 is None and v2 is None:
        return p0, NO_SIDE
    if v2 is None:
        return v1, 0
    if v1 is None:
        return v2, 1
    if first.hand != second.hand:
        return (v1, 0) if first.hand < second.hand else (v2, 1)
    return logit_mean(v1, v2), BOTH_SIDES


@dataclass
class Context:
    """1回の計算で共有する、時刻 t 以前の観測だけから作った評価入力。"""

    overlay: Any
    latest: tuple
    known: tuple
    elapsed: float
    colors: tuple[int, ...]
    static: Any
    prefire: np.ndarray
    count: Any


def build_context(overlay: Any, latest: tuple, known: tuple, elapsed: float, cache: dict) -> Context:
    """S3 の入力 (発火前の盤面から作る static・側特徴・count) を overlay._fire と同じ作り方で一度だけ作る。"""
    from src.exchange_event_count_features import CountObservation
    from src.exchange_event_overlay import UNUSED_S1_M0
    snapshot = overlay._snapshots[-1][1]
    static = overlay._build_static(tuple(s.board for s in latest), snapshot, elapsed, UNUSED_S1_M0)
    prefire = np.stack([side_features_native(overlay, s, elapsed, cache) for s in latest])
    models = overlay.tracker.models
    count = (CountObservation(np.stack([s.board._grid for s in latest]), np.stack([s.queue for s in latest]),
                              elapsed) if getattr(models, 'count_features', False) else None)
    grids = [np.asarray(s.board._grid, dtype=np.int8) for s in latest]
    return Context(overlay, latest, known, elapsed, sim.seen_colors(grids, list(known)), static, prefire, count)


def side_features_native(overlay: Any, side: Any, elapsed: float, cache: dict) -> np.ndarray:
    """発火前の側特徴 (S3 の入力)。overlay の表にあればそれを読み、無ければ native 連鎖計算で求める。

    native (exchange_event_count_features.NativeSimulator、同じ隠し段規則) は浮きぷよ・UNKNOWN のない盤面だけで使う
    (search.feature_simulator)。Python 版と同じ値になることを記録の盤面で確かめた
    (scripts/prefire_native_feature_parity_20261001.py)。
    """
    from src.exchange_event_features import prefire_side_features
    from src.scoring import compute_effective_rate
    grid = side.board._grid
    key = (grid.tobytes(), grid.dtype.str, side.queue.tobytes(), compute_effective_rate(elapsed))
    shared = getattr(overlay, '_feature_cache', {})
    if key in shared:
        return shared[key]
    if key not in cache:
        cache[key] = prefire_side_features(grid, side.queue, elapsed, simulator=search.feature_simulator(grid))
    return cache[key]


def s3_value(ctx: Context, scores: np.ndarray, elapsed: float) -> float:
    """既存 S3 (撃ち合い確定) を、探索で求めた両側の得点で評価する (Phase 2 の branch_value と同じ入力)。"""
    from src.exchange_event_evaluator import ExchangeEndInput, FiringInput, evaluate_exchange_event
    firing = tuple(bool(scores[idx] > 0) for idx in SIDES)
    event = ExchangeEndInput(FiringInput(ctx.static, ctx.prefire, firing, ctx.count), np.zeros(2), scores, elapsed)
    return float(evaluate_exchange_event(event, ctx.overlay.tracker.models))


def landing_gfe(ctx: Context, boards: tuple, incoming: tuple[int, int]) -> float:
    """仮想着弾 G_fe。ExchangeLandingProjection._landing_gfe と同じ4行 (着弾・M0・static・評価)。"""
    from src.exchange_event_evaluator import evaluate_exchange_event
    from src.exchange_virtual_board import land_pending_ojama_onto_board
    overlay = ctx.overlay
    virtual = tuple(land_pending_ojama_onto_board(b, boards[1 - i], incoming[i])[0] for i, b in enumerate(boards))
    queues = np.stack([s.queue for s in ctx.latest])
    m0 = overlay._m0(np.stack([b._grid for b in virtual]), queues)
    event = overlay._build_static(virtual, overlay._snapshots[-1][1], ctx.elapsed, m0)
    return float(evaluate_exchange_event(event, overlay.tracker.models))


def exchange_inputs(ctx: Context, attacker: int, line: search.FireLine, counter: float,
                    elapsed: float) -> tuple[np.ndarray, tuple, tuple[int, int]]:
    """attacker が line を撃ち、受け側が counter 個 (おじゃま) を返した撃ち合いの (両側得点, 盤面, 純受け量)。

    相殺は既存 cancel_own_pending_then_send_surplus と同じ (送り合いの差だけが降る)。攻撃側の盤面は連鎖後、
    受け側は現在の盤面 (既存 _counter_projection も応手前の盤面で着弾を評価する)。
    """
    from src.ojama_accounting import cancel_own_pending_then_send_surplus
    receiver = 1 - attacker
    sent = int(sim.send_ojama(line.score, elapsed))
    reply = int(math.floor(counter))
    left, surplus = cancel_own_pending_then_send_surplus(reply, sent, 0)
    incoming = [0, 0]
    incoming[receiver], incoming[attacker] = left, surplus
    scores = np.zeros(2)
    scores[attacker], scores[receiver] = line.score, reply * sim.effective_rate(elapsed)
    boards = [s.board for s in ctx.latest]
    boards[attacker] = search.board_from(line.final)
    return scores, tuple(boards), (incoming[0], incoming[1])


def exchange_value(ctx: Context, attacker: int, line: search.FireLine, counter: float, elapsed: float) -> float:
    """撃ち合いの 1P 勝率。合成は既存 _evaluate_single と同じ logit_mean(S3, 仮想着弾 G_fe)。"""
    scores, boards, incoming = exchange_inputs(ctx, attacker, line, counter, elapsed)
    return logit_mean(s3_value(ctx, scores, elapsed), landing_gfe(ctx, boards, incoming))


def exchange_s3(ctx: Context, attacker: int, line: search.FireLine, counter: float, elapsed: float) -> float:
    """撃ち合いの 1P 勝率を S3 だけで求める (仮想着弾 G_fe を省く。盤面特徴の再計算がないので速い)。"""
    return s3_value(ctx, exchange_inputs(ctx, attacker, line, counter, elapsed)[0], elapsed)


def exchange_gfe(ctx: Context, attacker: int, line: search.FireLine, counter: float, elapsed: float) -> float:
    """撃ち合いの 1P 勝率を仮想着弾 G_fe だけで求める (待つ値 p0 と同じ評価器なので較正のずれがない)。"""
    _, boards, incoming = exchange_inputs(ctx, attacker, line, counter, elapsed)
    return landing_gfe(ctx, boards, incoming)


def fire_elapsed(ctx: Context, line: search.FireLine) -> float:
    """撃つ時刻の経過秒 (撃つまでに置く手数ぶん後ろへずらす。マージンタイムの換算に使う)。"""
    from src.indicators_v2 import SEC_PER_HAND
    return ctx.elapsed + (line.hand - 1) * SEC_PER_HAND


def counter_hands(line: search.FireLine) -> int:
    """受け側の猶予手数 = 連鎖演出中の手数 (既存 remaining_hands、着地前の最後の1手込み) + 攻撃側が撃つまでの手数。"""
    from src.exchange_event_landing import remaining_hands
    return remaining_hands(line.chain_count, 0.0, 0.0) + line.hand - 1


@dataclass(frozen=True)
class LineEval:
    """1つの発火の評価。counter は受け側が選んだ応手量 (標本平均、受けるを選べば 0)。"""

    value: float
    counter: float
    lethal: bool


def _lethal_value(attacker: int) -> float:
    """確実な勝ちの値 (既存の回避不能死と同じ 0.02 / 0.98、1P 視点)。"""
    from src.exchange_event_landing import UNAVOIDABLE_DEATH_PROBABILITY
    win = 1 - UNAVOIDABLE_DEATH_PROBABILITY
    return win if attacker == 0 else 1 - win


def line_eval(ctx: Context, attacker: int, line: search.FireLine, value_fn=None) -> LineEval:
    """受け側は応手標本ごとに「受ける/返す」の良い方を選ぶ (value_fn で評価、既定は S3 と仮想着弾 G_fe の合成)。"""
    value_fn = exchange_value if value_fn is None else value_fn
    receiver = 1 - attacker
    elapsed, hands = fire_elapsed(ctx, line), counter_hands(line)
    raw = np.asarray(ctx.latest[receiver].board._grid, dtype=np.int8).tobytes()
    if search.lethal(raw, int(sim.send_ojama(line.score, elapsed)), hands, ctx.colors, elapsed):
        return LineEval(_lethal_value(attacker), 0.0, True)
    better = (lambda a, b: a > b) if receiver == 0 else (lambda a, b: a < b)   # 受け側 (receiver) に良い方
    passive = value_fn(ctx, attacker, line, 0.0, elapsed)
    values, counters = [], []
    for c in search.counter_quantiles(raw, ctx.known[receiver], hands, ctx.colors, elapsed):
        reply = value_fn(ctx, attacker, line, c, elapsed) if c > 0 else passive
        chosen = c > 0 and better(reply, passive)
        values.append(reply if chosen else passive)
        counters.append(c if chosen else 0.0)
    return LineEval(float(np.mean(values)), float(np.mean(counters)), False)


def line_value(ctx: Context, attacker: int, line: search.FireLine) -> tuple[float, bool]:
    """1つの発火の値 (1P 視点) と、確実な勝ちか (構成 full)。"""
    ev = line_eval(ctx, attacker, line)
    return ev.value, ev.lethal


def _improves(attacker: int, value: float, best: float | None) -> bool:
    """攻撃側に良い値か (1P は大きい方、2P は小さい方)。同じ値なら先の手を残す。"""
    return best is None or (value > best if attacker == 0 else value < best)


def side_best(ctx: Context, attacker: int, evaluator: str = 'full') -> SideBest | None:
    """片側の最善の発火。撃たないと窒息なら、手に持つ組で撃つ発火だけが選べる。

    full: 各発火を S3 と仮想着弾 G_fe の合成で評価する。
    fast: 発火の選択と受け側の選択は S3 だけで行い、選んだ1つだけを合成で評価する (盤面特徴の再計算を減らす)。
    s3:   S3 だけで評価する。
    gfe:  仮想着弾 G_fe だけで評価する (S3 は実際に撃たれた撃ち合いで学習しているので、撃つ仮定の評価に
          選択の偏りが入りうる。待つ値と同じ G_fe で比べる対照)。
    """
    grid = np.asarray(ctx.latest[attacker].board._grid, dtype=np.int8)
    lines, forced = search.fire_lines(grid.tobytes(), tuple(int(v) for v in ctx.known[attacker]))
    if forced:
        lines = tuple(line for line in lines if line.hand == 1)
    if evaluator in ('full', 'gfe'):
        value_fn = exchange_value if evaluator == 'full' else exchange_gfe
        best = None
        for line in lines:
            ev = line_eval(ctx, attacker, line, value_fn)
            value, certain = ev.value, ev.lethal
            if _improves(attacker, value, None if best is None else best.value):
                best = SideBest(value, line.hand, certain, forced)
        return best
    evals = [(line, line_eval(ctx, attacker, line, exchange_s3)) for line in lines]
    pick = None
    for line, ev in evals:
        if _improves(attacker, ev.value, None if pick is None else pick[1].value):
            pick = (line, ev)
    if pick is None:
        return None
    line, ev = pick
    if evaluator == 'fast' and not ev.lethal:
        scores, boards, incoming = exchange_inputs(ctx, attacker, line, ev.counter, fire_elapsed(ctx, line))
        return SideBest(logit_mean(ev.value, landing_gfe(ctx, boards, incoming)), line.hand, False, forced)
    return SideBest(ev.value, line.hand, ev.lethal, forced)


class BestPlayPrefireLayer(PrefireExchangeLayer):
    """表示の直前だけ最善手の値を出す外部 wrapper (評価器の状態は持たない・変えない)。"""

    def __init__(self, latency_sec: float = 0.0, evaluator: str = 'full') -> None:
        if evaluator not in EVALUATORS:
            raise ValueError(f'evaluator は {EVALUATORS} のいずれか: {evaluator!r}')
        super().__init__(hazard=None)
        self.latency_sec = float(latency_sec)
        self.evaluator = evaluator
        self._ready: dict[tuple, float] = {}

    def apply(self, overlay: Any, t_sec: float, game_idx: int) -> None:
        """静止区間なら最善手の値で表示値を差し替え、内訳を trace に残す。"""
        tracker = overlay.tracker
        if tracker.current is not None or tracker.source != 'G_fe' or tracker.probability is None:
            return
        latest = self._latest(overlay)
        if latest is None:
            return
        known = tuple(known_pairs(h) for h in overlay._history)
        elapsed = max(0.0, t_sec - overlay._start)
        key = (tuple(s.board._grid.tobytes() for s in latest), known, sim.effective_rate(elapsed), overlay._game)
        spent = 0.0
        if key not in self._cache:
            started = _now_ms()
            self._cache[key] = self._compute(overlay, latest, elapsed, known)
            spent = _now_ms() - started
            self._ready[key] = t_sec + self.latency_sec   # 非同期の結果は計算開始から latency_sec 後に届く
        if t_sec < self._ready[key]:
            return
        best = self._cache[key]
        shown, chosen = combine(tracker.probability, best)
        self.trace.append((t_sec, game_idx, tracker.probability, shown,
                           *(np.nan if b is None else b.value for b in best),
                           *(0 if b is None else b.hand for b in best),
                           *(False if b is None else b.lethal for b in best),
                           *(False if b is None else b.forced for b in best), chosen, spent, self._ready[key]))
        if chosen == NO_SIDE:
            return   # どちらも待つのが最善なら表示も由来も書き換えない (OFF とビット一致)
        self._written = (shown, tracker.probability, tracker.source)
        tracker.probability, tracker.source = shown, PREFIRE_SOURCE

    def restore(self, overlay: Any) -> None:
        """update の前に、書き換えた表示値を元の値へ戻す (Phase 2 と同じ規則、由来名だけ違う)。"""
        tracker = overlay.tracker
        if self._written is not None and tracker.source == PREFIRE_SOURCE and tracker.probability == self._written[0]:
            tracker.probability, tracker.source = self._written[1], self._written[2]
        self._written = None

    def _compute(self, overlay: Any, latest: tuple, elapsed: float, known: tuple) -> tuple:
        """両側の最善の発火を求める。評価入力が作れない側は選択肢なし (欠測を推測で埋めない)。"""
        try:
            ctx = build_context(overlay, latest, known, elapsed, self._features)
        except (ValueError, TypeError, FloatingPointError):
            return (None, None)
        result = []
        for attacker in SIDES:
            try:
                result.append(side_best(ctx, attacker, self.evaluator))
            except (ValueError, TypeError, FloatingPointError):
                result.append(None)
        return tuple(result)

    def save(self, path: Path) -> None:
        """由来の記録 (フレームごとの現在値・表示値・各側の最善・所要)。"""
        data = np.asarray(self.trace, dtype=float).reshape(-1, len(TRACE_COLUMNS))
        np.savez_compressed(path, columns=np.asarray(TRACE_COLUMNS), values=data)
