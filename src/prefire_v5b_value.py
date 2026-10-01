"""同一モデルの片側encoderを再用する。モデル係数・浮動小数点の計算順を変えない。"""
from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from typing import Any

import numpy as np
import torch

from src.advantage_m0_current_cnn_v1 import board_categories_from_raw
from src.exchange_event_m0 import PROBABILITY_EPSILON, LOGIT_LIMIT

ENCODER_CACHE_SIZE = 8192
BOARD_CACHE_SIZE = 16384


class CachedStatic:
    """全列挙内の盤面特徴を再利用する。通常表示用128件キャッシュとは分離する。"""

    def __init__(self, original: Any) -> None:
        from scripts import visualize_advantage_overlay as renderer
        self.original = original
        self.renderer = renderer
        self.features = lru_cache(maxsize=BOARD_CACHE_SIZE)(renderer._exchange_board_features.__wrapped__)

    def __call__(self, boards: tuple, snapshot: Any, elapsed_sec: float, m0_probability: float) -> Any:
        """元のG_fe特徴生成をそのまま使い、片側特徴のキャッシュ容量だけを変える。"""
        if self.original is not self.renderer._exchange_static_input:
            return self.original(boards, snapshot, elapsed_sec, m0_probability)
        own, opponent = (self.features(b._grid.tobytes(), b._grid.shape, b._grid.dtype.str) for b in boards)
        features = self.renderer._side_feats_full(own, opponent, snapshot.net_balance_capped, snapshot.forecast_p1)
        return self.renderer.StaticInput(np.array([features[name] for name in self.renderer.D_COLUMNS]),
                                         m0_probability, elapsed_sec, source_side=0)


class CachedM0:
    """片側の盤面・残ツモ・側を鍵にした、単一モデルインスタンス専用キャッシュ。"""

    def __init__(self, original: Any) -> None:
        self.original = original
        self.cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()

    def __call__(self, boards: np.ndarray, queues: np.ndarray) -> float:
        """encoder以外は元のforwardと同じshapeと演算順で確率を算出する。"""
        keys = tuple((i, b.tobytes(), b.dtype.str, b.shape, q.tobytes(), q.dtype.str)
                     for i, (b, q) in enumerate(zip(boards, queues)))
        with torch.inference_mode():
            sides = self._sides(keys, boards, queues)
            model = self.original.model
            direct = model._score_pair(sides[0], sides[1])
            swapped = model._score_pair(sides[1], sides[0])
            probability = float(torch.sigmoid(0.5 * (direct - swapped)).item())
        probability = np.clip(probability, PROBABILITY_EPSILON, 1-PROBABILITY_EPSILON)
        logit = np.log(probability/(1-probability)) * self.original.slope
        return float(1/(1+np.exp(-np.clip(logit, -LOGIT_LIMIT, LOGIT_LIMIT))))

    def _sides(self, keys: tuple, boards: np.ndarray, queues: np.ndarray) -> tuple:
        """元と同じ2盤面batchでmissを計算する。batch変更による丸め差を避ける。"""
        if any(key not in self.cache for key in keys):
            categories = np.stack([board_categories_from_raw(b) for b in boards])
            queue = np.where((queues >= 1) & (queues <= 5), queues, 0)
            encoded = self.original.model.side_encoder(
                torch.as_tensor(categories, dtype=torch.long), torch.as_tensor(queue, dtype=torch.long))
            for i, key in enumerate(keys):
                self.cache[key] = encoded[i:i+1].clone()
            while len(self.cache) > ENCODER_CACHE_SIZE:
                self.cache.popitem(last=False)
        for key in keys:
            self.cache.move_to_end(key)
        return tuple(self.cache[key] for key in keys)
