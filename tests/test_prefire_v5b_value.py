"""M0片側再利用は出力を変えず、モデル個体間のキャッシュも共有しない。"""
from types import SimpleNamespace

import numpy as np
import torch

from src.advantage_m0_current_cnn_v1 import AdvantageM0CurrentCNNV2
from src.exchange_event_m0 import FileM0Predictor
from src.prefire_v5b_value import CachedM0, CachedStatic
from src.board import Board


def test_cached_encoder_is_bit_identical() -> None:
    torch.manual_seed(20261001)
    torch.set_num_threads(1)
    predictor = FileM0Predictor.__new__(FileM0Predictor)
    predictor.model, predictor.slope = AdvantageM0CurrentCNNV2().eval(), 1.3
    cached = CachedM0(predictor)
    boards = np.zeros((2, 13, 6), dtype=np.int8)
    queues = np.array([[1, 2, 3, 4], [2, 2, 5, 1]])
    for side in (0, 1, 0, 1):
        boards[side, -1, side] += 1
        expected = predictor(boards, queues)
        assert cached(boards, queues) == expected
        assert cached(boards, queues) == expected
    queues[0] = 0
    assert cached(boards, queues) == predictor(boards, queues)
    assert CachedM0(predictor).cache == {}


def test_static_cache_preserves_all_columns() -> None:
    from scripts.visualize_advantage_overlay import _exchange_static_input
    cached = CachedStatic(_exchange_static_input)
    boards = (Board(), Board())
    boards[0]._grid[-1, :3] = (1, 2, 3)
    boards[1]._grid[-1, :3] = (3, 4, 2)
    for pending in (0, 6, 31):
        snapshot = SimpleNamespace(net_balance_capped=pending, forecast_p1=pending)
        for elapsed in (0., 15., 34.):
            actual = cached(boards, snapshot, elapsed, .3)
            expected = _exchange_static_input(boards, snapshot, elapsed, .3)
            assert np.array_equal(actual.d_features, expected.d_features, equal_nan=True)
            assert actual.elapsed_sec == expected.elapsed_sec
            assert actual.m0_probability_1p == expected.m0_probability_1p
    assert cached.features.cache_info().misses == 2
