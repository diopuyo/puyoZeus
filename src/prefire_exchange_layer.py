"""発火前の撃ち合い予測を表示勝率へ反映する予測層 (2026-09-30、Phase 2、既定 OFF)。

撃ち合いが起きていない静止区間 (tracker.source == 'G_fe') に限り、次を混ぜる:
    p = w0·p_current + Σ_a w_a·v_a
- v_a: 側 a が既知2手以内の最大発火を撃ち、受け側が猶予手数内の最善応手で返したと仮定した撃ち合いの
  S3 勝率。受け側の応手は割り引かない (user 決定 9/30: 両者が正しく指した局面の価値)。
- w_a: 側 a が近く撃つ確率 (148動画で学習した発火タイミング hazard)。撃たないと窒息するなら 1。
評価器の内部状態は変えない。update の前に `restore`、後に `apply` を呼び、表示の直前だけ値を差し替える
(OFF と内部状態がバイト一致する)。由来は source='G_fe+prefire' と trace に残す。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src import prefire_exchange_sim as sim

MODEL_DIR = Path('models/prefire_hazard_v1')
STATIC_SOURCE = 'G_fe'
PREFIRE_SOURCE = 'G_fe+prefire'
SIDES = (0, 1)
TRACE_COLUMNS = ('t_sec', 'game_idx', 'p_current', 'p_shown', 'pi_1p', 'pi_2p', 'v_1p', 'v_2p',
                 'forced_1p', 'forced_2p', 'compute_ms')


class HazardModel(Protocol):
    """発火タイミングの確率モデル (差し替え可能)。"""

    def predict(self, features: np.ndarray) -> float: ...


@dataclass(frozen=True)
class FileHazardModel:
    """manifest の列順・sha256 を照合してから読み込む sklearn 分類器。"""

    estimator: Any
    horizon_placements: int

    @classmethod
    def load(cls, directory: Path = MODEL_DIR) -> FileHazardModel:
        """列順が sim.HAZARD_COLUMNS と一致し、ファイルの sha256 が manifest と一致する場合だけ読む。"""
        import joblib
        meta = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if tuple(meta['columns']) != sim.HAZARD_COLUMNS:
            raise ValueError('hazard モデルの列順が不一致')
        path = directory / meta['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('hazard モデルのファイルが不一致')
        return cls(joblib.load(path), int(meta['horizon_placements']))

    def predict(self, features: np.ndarray) -> float:
        """1行の確率。"""
        return float(self.estimator.predict_proba(features[None, :])[0, 1])


@dataclass(frozen=True)
class Branch:
    """片側が撃つ仮定の枝。"""

    weight: float
    value: float
    forced: bool


def mix(current: float, branches: tuple[Branch | None, Branch | None]) -> tuple[float, tuple[float, float]]:
    """確率空間で混合する。両側の重みの和が1を超える場合は和で割る (撃たない枝は 0 になる)。"""
    weights = [0.0 if b is None else b.weight for b in branches]
    scale = max(1.0, sum(weights))
    normalized = [w / scale for w in weights]
    value = (1 - sum(normalized)) * current
    value += sum(w * b.value for w, b in zip(normalized, branches) if b is not None)
    return float(min(1.0, max(0.0, value))), (normalized[0], normalized[1])


def _queue(side: Any) -> tuple[int, ...]:
    """履歴の NEXT/NEXT2 を int の4要素タプルにする (未読は 0)。"""
    return tuple(int(v) for v in side.queue)


def counter_send(receiver: Any, option: sim.FireOption, elapsed: float) -> float:
    """受け側の最善応手 (おじゃま個数)。猶予手数 = 攻撃の連鎖時間の手数 + 攻撃側が発火までに使う手数。"""
    from src.exchange_event_landing import future_send, remaining_hands
    grid = np.asarray(receiver.board._grid, dtype=np.int8)
    queue = _queue(receiver)
    if not sim.queue_valid(queue):
        return 0.0   # 応手の探索に NEXT が要る。未読なら応手なし (欠測を楽観で埋めない)
    hands = remaining_hands(option.chain_count, 0.0, 0.0) + option.hand - 1
    return float(future_send(grid.tobytes(), grid.shape, grid.dtype.str, queue, hands, elapsed))


def branch_value(overlay: Any, latest: tuple, attacker: int, option: sim.FireOption,
                 elapsed: float) -> float:
    """側 attacker が option を撃ち、受け側が最善応手で返した撃ち合いの S3 勝率 (1P)。

    入力は現在の確定盤面・NEXT だけ (時刻 t 以前)。S3 の入力の作り方は overlay._fire と同じで、
    得点だけを探索結果で与える。
    """
    from src.exchange_event_count_features import CountObservation
    from src.exchange_event_evaluator import ExchangeEndInput, FiringInput, evaluate_exchange_event
    from src.exchange_event_features import prefire_side_features
    from src.exchange_event_overlay import UNUSED_S1_M0
    receiver = 1 - attacker
    counter = counter_send(latest[receiver], option, elapsed)
    scores = np.zeros(2)
    scores[attacker] = option.score
    scores[receiver] = counter * sim.effective_rate(elapsed)
    snapshot = overlay._snapshots[-1][1]
    static = overlay._build_static(tuple(s.board for s in latest), snapshot, elapsed, UNUSED_S1_M0)
    prefire = np.stack([prefire_side_features(s.board._grid, s.queue, elapsed) for s in latest])
    models = overlay.tracker.models
    count = (CountObservation(np.stack([s.board._grid for s in latest]), np.stack([s.queue for s in latest]),
                              elapsed) if getattr(models, 'count_features', False) else None)
    firing = tuple(bool(idx == attacker or scores[idx] > 0) for idx in SIDES)
    event = ExchangeEndInput(FiringInput(static, prefire, firing, count), np.zeros(2), scores, elapsed)
    return evaluate_exchange_event(event, models)


class PrefireExchangeLayer:
    """表示の直前だけ発火前予測を混ぜる外部 wrapper (評価器の状態は持たない・変えない)。"""

    def __init__(self, hazard: HazardModel) -> None:
        self.hazard = hazard
        self._written: tuple[float, float, str] | None = None   # (書いた値, 元の値, 元の由来)
        self._cache: dict[tuple, tuple] = {}
        self.trace: list[tuple] = []

    def restore(self, overlay: Any) -> None:
        """update の前に、書き換えた表示値を評価器の元の値へ戻す (評価器がまだ上書きしていなければ)。"""
        tracker = overlay.tracker
        if self._written is not None and tracker.source == PREFIRE_SOURCE and tracker.probability == self._written[0]:
            tracker.probability, tracker.source = self._written[1], self._written[2]
        self._written = None

    def apply(self, overlay: Any, t_sec: float, game_idx: int) -> None:
        """静止区間なら予測を混ぜて表示値を差し替え、由来と内訳を trace に残す。"""
        tracker = overlay.tracker
        if tracker.current is not None or tracker.source != STATIC_SOURCE or tracker.probability is None:
            return
        latest = self._latest(overlay)
        if latest is None:
            return
        elapsed = max(0.0, t_sec - overlay._start)
        started = _now_ms()
        branches = self._branches(overlay, latest, elapsed)
        shown, weights = mix(tracker.probability, branches)
        self.trace.append((t_sec, game_idx, tracker.probability, shown, *weights,
                           *(np.nan if b is None else b.value for b in branches),
                           *(False if b is None else b.forced for b in branches), _now_ms() - started))
        self._written = (shown, tracker.probability, tracker.source)
        tracker.probability, tracker.source = shown, PREFIRE_SOURCE

    @staticmethod
    def _latest(overlay: Any) -> tuple | None:
        """両側の最新確定盤面 (履歴の末尾)。どちらかが無ければ予測しない。"""
        if overlay._start is None or not all(overlay._history) or not overlay._snapshots:
            return None
        return tuple(h[-1] for h in overlay._history)

    def _branches(self, overlay: Any, latest: tuple, elapsed: float) -> tuple:
        """同じ盤面・NEXT・換算率の組では探索と S3 を一度だけ行う。"""
        key = (tuple(s.board._grid.tobytes() for s in latest), tuple(_queue(s) for s in latest),
               sim.effective_rate(elapsed), overlay._game)
        if key not in self._cache:
            self._cache[key] = self._compute(overlay, latest, elapsed)
        return self._cache[key]

    def _compute(self, overlay: Any, latest: tuple, elapsed: float) -> tuple:
        """両側の発火候補・hazard・枝の勝率を求める。"""
        grids = [np.asarray(s.board._grid, dtype=np.int8) for s in latest]
        options = [sim.fire_options(g.tobytes(), _queue(s)) for g, s in zip(grids, latest)]
        result = []
        for a in SIDES:
            best = options[a].best()
            if best is None:
                result.append(None)
                continue
            features = sim.hazard_features(options[a], options[1 - a], grids[a], grids[1 - a], elapsed)
            weight = 1.0 if options[a].forced else self.hazard.predict(features)
            try:
                value = branch_value(overlay, latest, a, best, elapsed)
            except (ValueError, TypeError, FloatingPointError):
                result.append(None)   # 評価入力が作れない枝は混ぜない (欠測を推測で埋めない)
                continue
            result.append(Branch(weight, value, options[a].forced))
        return tuple(result)

    def save(self, path: Path) -> None:
        """由来の記録 (フレームごとの現在値・表示値・重み・枝の勝率・所要)。"""
        data = np.asarray(self.trace, dtype=float).reshape(-1, len(TRACE_COLUMNS))
        np.savez_compressed(path, columns=np.asarray(TRACE_COLUMNS), values=data)


def _now_ms() -> float:
    """所要時間の計測 (ms)。"""
    from time import perf_counter
    return perf_counter() * 1e3
