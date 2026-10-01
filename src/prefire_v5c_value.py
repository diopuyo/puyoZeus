"""5C専用の値キャッシュ。既存の指標とG_feの演算を再利用する。"""
from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import numpy as np

from src import prefire_v5_search as search
from src import prefire_v5_value as value
from src.board import Board
from src.prefire_v5b_value import BOARD_CACHE_SIZE, CachedM0, CachedStatic

VALUE_CACHE_SIZE = 65536


def articulation(board: Board) -> Any:
    """発火候補の選択だけnativeへ委譲し、関節点の反実仮想は既存実装で解く。"""
    from scripts import build_labeled_win_from_npz as builder
    iv = builder.iv
    if not (builder._PUYO_CORE_AVAILABLE and builder._board_is_gravity_consistent(board)):
        return iv.chain_articulation_point_count(board)
    chains, best, _ = builder._native_takapt_best_drop(board)
    if best is None or chains < 2:
        return iv.IndicatorV2Value(score=0., raw=0.)
    simulator = iv._SHARED_SIMULATOR
    result = simulator.simulate(best)
    raw = float(iv._count_critical_erase_groups(result.steps, simulator)) if len(result.steps) >= 2 else 0.
    return iv.IndicatorV2Value(score=iv._clamp01(raw / iv.NORM_CHAIN_ARTICULATION_POINT), raw=raw)


def board_features(raw: bytes, shape: tuple[int, int], dtype: str) -> dict[str, float]:
    """学習変換で検証済みのnative指標を使用し、未対応列は既存Pythonへ委譲する。"""
    from scripts import build_labeled_win_from_npz as builder
    from scripts import visualize_advantage_overlay as renderer
    board = Board()
    board._grid = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    registry = {**builder.GRID_ONLY_HEAVY_INDICATORS_NATIVE, 'chain_articulation_point_count': articulation}
    row = {name: registry.get(name, fn)(board).score
           for name, fn in renderer.FULL_MODEL_GRID_REGISTRY.items()}
    total, _ = renderer.iv.connectivity_observation(board)
    row.update(conn_pair_count=float(total.pair_count), conn_triple_count=float(total.triple_count),
               conn_max_group_size=float(total.max_group_size))
    return row


class StaticCache(CachedStatic):
    """通常表示の関数を置換せず、探索専用の片側特徴だけ高速化する。"""

    def __init__(self, original: Any) -> None:
        super().__init__(original)
        self.features = lru_cache(maxsize=BOARD_CACHE_SIZE)(board_features)


class EvaluationCache:
    """同一モデル内の着弾後評価を、通知をまたいで保存する。"""

    def __init__(self, overlay: Any) -> None:
        self.original_m0 = overlay._m0
        self.original_static = overlay._build_static
        self.models = overlay.tracker.models
        self.proxy = SimpleNamespace(_m0=CachedM0(overlay._m0),
            _build_static=StaticCache(overlay._build_static), _snapshots=overlay._snapshots,
            tracker=overlay.tracker)
        self.values: OrderedDict[tuple, float] = OrderedDict()
        self.hits = self.misses = 0

    def matches(self, overlay: Any) -> bool:
        """モデルやビルダーを交換した場合は古い値を再利用しない。"""
        return (self.original_m0 is overlay._m0 and self.original_static is overlay._build_static
                and self.models is overlay.tracker.models)

    def bind(self, overlay: Any) -> None:
        """現在通知のスナップショットを参照する。"""
        self.proxy._snapshots = overlay._snapshots

    def __call__(self, exchange: search.Exchange, elapsed: float) -> float:
        """G_feに実際に入る盤面・残組・予告・経過位相が一致するときだけ引く。"""
        standard = self.original_static is self.proxy._build_static.renderer._exchange_static_input
        if not standard:
            return value.evaluate(self.proxy, exchange, elapsed)
        phase = int(np.searchsorted(self.models.elapsed_thresholds, elapsed, side='left'))
        key = (tuple((p.board, p.queue[:4], p.pending) for p in exchange.sides), phase)
        if key in self.values:
            self.hits += 1
            self.values.move_to_end(key)
            return self.values[key]
        self.misses += 1
        result = value.evaluate(self.proxy, exchange, elapsed)
        self.values[key] = result
        if len(self.values) > VALUE_CACHE_SIZE:
            self.values.popitem(last=False)
        return result
