"""既定OFFのPhase 6。軽量選択後の本番値を学習済みa(z)で反映する。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.prefire_best_play_v5 import BestPlayV5Layer, combine_ready
from src.prefire_best_play_layer import PREFIRE_SOURCE
from src.prefire_v5c_value import EvaluationCache
from src import prefire_v5b_search as exact
from src import prefire_v6_value as value
from src.prefire_v6_production import ProductionValue
from src.prefire_stable_queue import STABLE_FRAMES

DEFAULT_STRENGTH = Path('logs/prefire_prediction/v6/strength.json')
FEATURE_NAMES = ('next_assignment_confidence', 'missing', 'truncated', 'disagreement')
EXTRA_COLUMNS = ('t_sec', 'game_idx', 'p_current', 'p_raw', 'alpha', *FEATURE_NAMES, 'proof_1p', 'proof_2p')
LOGIT_LIMIT = 32.


def assignment_confidence(queues: Any) -> float:
    """採用読みとの一致継続と繰上げ確認の証拠割合。人の発火確率ではない。"""
    evidence = []
    for side in queues.sides:
        readings = [min(1., p.count/STABLE_FRAMES) if p.accepted == p.run else 0.
                    for p in side.pairs]
        evidence.append(float(np.mean([float(side.shifted), *readings])))
    return min(evidence)


class Strength:
    """4変数と切片だけのロジスティック。学習資料の分離をファイルで明示する。"""

    def __init__(self, coefficients: tuple[float, ...], identity: bool = False) -> None:
        if len(coefficients) != len(FEATURE_NAMES)+1 or not np.isfinite(coefficients).all():
            raise ValueError('a(z)には有限の切片と4係数が必要')
        self.coefficients, self.identity = coefficients, identity

    @classmethod
    def load(cls, path: Path = DEFAULT_STRENGTH) -> Strength:
        """採点時はセット1学習済みのモデルだけを読む。"""
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['training_set'] != 'zenchi_set1_57' or data['features'] != list(FEATURE_NAMES):
            raise ValueError('a(z)の学習母集団または特徴定義が不一致')
        return cls(tuple(data['coefficients']))

    def __call__(self, features: tuple) -> float:
        """データ収集時だけidentityを明示的に使う。"""
        if self.identity:
            return 1.0
        x = float(np.dot(self.coefficients, (1., *features)))
        return float(1/(1+np.exp(-np.clip(x, -LOGIT_LIMIT, LOGIT_LIMIT))))


class BestPlayV6Layer(BestPlayV5Layer):
    """5Cの全列挙・入力・完了時刻規則を継承し、比較と反映だけを変更する。"""

    def __init__(self, latency_sec: float = 0., strength: Strength | None = None) -> None:
        super().__init__(latency_sec)
        self.strength = strength if strength is not None else Strength.load()
        self.extra: list[tuple] = []
        self._gaps: dict[tuple, float] = {}
        self._last_gap = 0.0
        self._last_certainty = (False, False)
        self._certainties: dict[tuple, tuple] = {}

    def _compute_v5(self, overlay: Any, states: tuple, elapsed: float) -> tuple:
        """各側の選択手と待機だけ本番評価。未知3組目は5Cと同じ8標本。"""
        if self._evaluation is None or not self._evaluation.matches(overlay):
            self._evaluation = EvaluationCache(overlay)
        self._evaluation.bind(overlay)
        evaluator = ProductionValue(overlay, states, elapsed, self._evaluation)
        results = tuple(value.choose(states, side, elapsed, self._transitions, evaluator)
                        for side in (0, 1))
        self._last_gap = max(r[1] for r in results)
        self._last_certainty = self._selected_certainty(results, elapsed, evaluator)
        return tuple(r[0] for r in results)

    def _selected_certainty(self, results: tuple, elapsed: float, evaluator: ProductionValue) -> tuple:
        """選択後のE35根拠だけ記録する。候補段階の証明を表示の証明にしない。"""
        from src.indicators_v2 import SEC_PER_HAND
        from src.prefire_best_play_layer import _lethal_value
        proofs = []
        for side, (choice, _) in enumerate(results):
            if choice is None:
                return (False, False)
            probability, attack, reply = choice
            pair = (attack, reply) if side == 0 else (reply, attack)
            time = elapsed+max(0, attack.consumed-1)*SEC_PER_HAND
            exchange = self._transitions.resolve(*pair, side, time)
            proof = evaluator.certainties.get(exchange, (False, False))
            proofs.append(tuple(proof[i] and abs(probability-_lethal_value(i)) < value.PROBABILITY_EPSILON
                                for i in (0, 1)))
        hands = [r[0][1].consumed for r in results]
        if hands[0] != hands[1]:
            return proofs[int(hands[1] < hands[0])]
        return tuple(all(p[i] for p in proofs) for i in (0, 1))

    def _schedule_v5(self, overlay: Any, key: tuple, states: tuple, statuses: tuple,
                     elapsed: float, t_sec: float) -> None:
        """食い違い幅はその計算要求に結び付け、別入力へ転用しない。"""
        super()._schedule_v5(overlay, key, states, statuses, elapsed, t_sec)
        self._gaps[key] = self._last_gap
        self._certainties[key] = self._last_certainty

    def _show_v5(self, tracker: Any, key: tuple, statuses: tuple, t_sec: float, game_idx: int) -> None:
        """通常値と予測値のlogit差へa(z)を掛ける。欠測と計算待ちは補正しない。"""
        p0, source = tracker.probability, tracker.source
        super()._show_v5(tracker, key, statuses, t_sec, game_idx)
        row = self.trace[-1]
        raw, used = row[3], bool(row[-2])
        known = [exact.known_depth(p.queue) for p in key[0]]
        confidence = assignment_confidence(self.queues)
        features = (confidence, float(min(known) < exact.MAX_KNOWN_HANDS), 0.,
                    self._gaps.get(key, 0.)/(1+self._gaps.get(key, 0.)))
        alpha = self.strength(features) if used else 0.
        shown = p0
        if used:
            log_odds = value.logit(p0)+alpha*(value.logit(raw)-value.logit(p0))
            shown = float(1/(1+np.exp(-log_odds)))
            tracker.probability, tracker.source = shown, PREFIRE_SOURCE
            self._written = (shown, p0, source)
        self.trace[-1] = (*row[:3], shown, *row[4:])
        certainty = tuple(bool(used and p) for p in self._certainties.get(key, (False, False)))
        self.extra.append((t_sec, game_idx, p0, raw, alpha, *features, *certainty))

    def save(self, path: Path) -> None:
        """本体traceに加えて学習に必要な生の増分とzを保存する。"""
        super().save(path)
        np.savez_compressed(path.with_name('prefire_v6_features.npz'),
                            columns=np.asarray(EXTRA_COLUMNS),
                            values=np.asarray(self.extra).reshape(-1, len(EXTRA_COLUMNS)))
