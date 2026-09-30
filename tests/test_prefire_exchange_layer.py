"""発火前予測層 (src/prefire_exchange_layer.py) の単体テスト (段1)。評価器の状態を変えないことを確かめる。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src import prefire_exchange_layer as layer_module
from src.prefire_exchange_layer import Branch, PrefireExchangeLayer, mix


class ConstantHazard:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, features: np.ndarray) -> float:
        return self.value


def fake_overlay(source: str = 'G_fe', probability: float = 0.4, current=None) -> SimpleNamespace:
    grid = np.zeros((13, 6), dtype=np.int8)
    grid[10:13, 0] = 1
    side = SimpleNamespace(board=SimpleNamespace(_grid=grid), queue=np.array([1, 1, 2, 3]), t_sec=1.0)
    tracker = SimpleNamespace(source=source, probability=probability, current=current, models=None)
    return SimpleNamespace(tracker=tracker, _start=0.0, _history=[[side], [side]], _snapshots=[(1.0, None)],
                           _game=0)


@pytest.fixture
def fixed_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """S3 の評価を固定値にして、混合と状態管理だけを検査する (1P が撃てば .9、2P が撃てば .1)。"""
    monkeypatch.setattr(layer_module, 'branch_value', lambda overlay, latest, a, option, elapsed: .9 if a == 0 else .1)


def test_mix_without_branches_is_identity() -> None:
    assert mix(0.37, (None, None)) == (0.37, (0.0, 0.0))


def test_mix_normalizes_when_weights_exceed_one() -> None:
    value, weights = mix(0.5, (Branch(1.0, 0.9, True), Branch(1.0, 0.1, False)))
    assert weights == (0.5, 0.5) and abs(value - 0.5) < 1e-12


def test_apply_then_restore_returns_exact_original(fixed_branch: None) -> None:
    overlay = fake_overlay()
    layer = PrefireExchangeLayer(ConstantHazard(0.5))
    layer.apply(overlay, 2.0, 0)
    assert overlay.tracker.source == layer_module.PREFIRE_SOURCE and overlay.tracker.probability != 0.4
    layer.restore(overlay)
    assert overlay.tracker.source == 'G_fe' and overlay.tracker.probability == 0.4


def test_no_double_blending_across_frames(fixed_branch: None) -> None:
    overlay = fake_overlay()
    layer = PrefireExchangeLayer(ConstantHazard(0.5))
    layer.apply(overlay, 2.0, 0)
    first = overlay.tracker.probability
    layer.restore(overlay)
    layer.apply(overlay, 2.1, 0)
    assert overlay.tracker.probability == first
    assert len(layer._cache) == 1   # 同じ盤面・NEXT は探索を再用する


def test_not_applied_during_exchange_or_other_source(fixed_branch: None) -> None:
    for overlay in (fake_overlay(current=object()), fake_overlay(source='S3'), fake_overlay(probability=None)):
        before = (overlay.tracker.source, overlay.tracker.probability)
        PrefireExchangeLayer(ConstantHazard(0.5)).apply(overlay, 2.0, 0)
        assert (overlay.tracker.source, overlay.tracker.probability) == before


def test_restore_keeps_evaluator_update(fixed_branch: None) -> None:
    """評価器が update で新しい値を書いたら、それを上書きで戻さない。"""
    overlay = fake_overlay()
    layer = PrefireExchangeLayer(ConstantHazard(0.5))
    layer.apply(overlay, 2.0, 0)
    overlay.tracker.source, overlay.tracker.probability = 'S1', 0.77
    layer.restore(overlay)
    assert (overlay.tracker.source, overlay.tracker.probability) == ('S1', 0.77)


def test_trace_records_provenance(fixed_branch: None, tmp_path) -> None:
    overlay = fake_overlay()
    layer = PrefireExchangeLayer(ConstantHazard(0.2))
    layer.apply(overlay, 2.0, 0)
    layer.save(tmp_path / 'trace.npz')
    with np.load(tmp_path / 'trace.npz') as data:
        row = dict(zip(data['columns'], data['values'][0]))
    assert row['p_current'] == 0.4 and row['v_1p'] == 0.9 and row['v_2p'] == 0.1
    assert abs(row['p_shown'] - (0.6 * 0.4 + 0.2 * 0.9 + 0.2 * 0.1)) < 1e-12
