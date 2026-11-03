"""F1の個数・視点・CV集計の回帰テスト。"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from scripts import train_exchange_event_models_v2_20260927 as f1


def test_orientation_preserves_absolute_values() -> None:
    """側の絶対値を負にせず、差の符号だけ交換する。"""
    sides = np.arange(12, dtype=float).reshape(2, 6)
    direct, reverse = f1.orient(sides, 1), f1.orient(sides, -1)
    np.testing.assert_array_equal(direct[:5], reverse[5:10])
    np.testing.assert_array_equal(direct[10:15], -reverse[10:15])
    np.testing.assert_array_equal(direct[-2:], reverse[-2:][::-1])


def test_log_counts_do_not_saturate() -> None:
    """正規化上限72個を超えても大小と差が残る。"""
    values = np.log1p(np.array([72., 144., 720.]))
    assert np.all(np.diff(values) > 0)


def test_bootstrap_pairs_match_duplicated_videos() -> None:
    """同点と同じ動画の再抽出を含むAUCをsklearnで照合する。"""
    y = np.array([0, 1, 1, 0, 0, 1])
    p = np.array([.2, .5, .5, .7, .8, .9])
    groups = np.array([0, 0, 1, 1, 2, 2])
    pairs = f1.auc_pair_matrix(y, p, groups, 3)
    draw = np.array([2, 0, 3])
    weight = draw[groups]
    expected = roc_auc_score(y, p, sample_weight=weight)
    actual = draw @ pairs @ draw / ((weight*y).sum()*(weight*(1-y)).sum())
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("delta,upper,overall,accepted", [
    (-.002, -.0001, .001, True), (-.0019, -.0001, 0., False),
    (-.004, .0001, 0., False), (-.004, -.001, .0011, False)])
def test_preregistered_thresholds(delta: float, upper: float, overall: float,
                                 accepted: bool) -> None:
    """点推定・有意性・全体非劣性の三条件を別々に確認する。"""
    middle = {"log_loss": dict(delta=delta, ci95=[-.01, upper]),
              "auc": dict(delta=0., ci95=[-.01, .01])}
    report = dict(differences=dict(middle=dict(S1_prime=middle),
        overall=dict(S1_prime={"log_loss": dict(ci95=[-.01, overall])})))
    assert f1.decisions(report)["S1_prime"]["adopt"] is accepted


def test_native_fire_matches_python() -> None:
    """探索のnative化で火力個数・候補順が変わらない。"""
    board = f1.Board()
    board._grid[-1, :3] = 1
    queue, levels = (1, 2, 2, 1), (-1, 0, 1)
    expected = f1.iv.near_future_fire_power(board, queue[:2], queue[2:], k_levels=levels)
    f1.init_worker()
    actual = f1.fire(board._grid.astype(np.int8).tobytes(), queue, 0., levels)
    np.testing.assert_array_equal(actual, [expected.values[k].raw for k in levels])


def test_response_uses_one_hand_at_exchange_end() -> None:
    """両者演出終了では着地を起こす最後の一手のみが残る。"""
    assert f1.remaining_hands(0, 0., 0.) == 1


def test_counter_margin_uses_net_received(monkeypatch: pytest.MonkeyPatch) -> None:
    """自側送りを相殺済み純受け量に一度だけ算入する。"""
    raw = dict(grids=np.zeros((2, 13, 6), dtype=np.int8))
    raw.update({key: np.ones(2, dtype=int) for key in f1.QUEUE})
    levels_seen = []

    def fake_fire(grid: bytes, queue: tuple, elapsed: float,
                  levels: tuple, response: bool = False) -> np.ndarray:
        levels_seen.append((levels, response))
        return np.full(len(levels), 100. if response else 144.)

    monkeypatch.setattr(f1, "fire", fake_fire)
    values = f1.stage_features(raw, np.array([0, 1]), np.array([30., 180.]),
                               np.array([0, 0]), 0.)
    assert values[0, -1] == pytest.approx(-np.log1p(50.))
    assert values[1, -1] == pytest.approx(np.log1p(100.))
    assert ((-1,), True) in levels_seen
    assert values[0, 0] == pytest.approx(np.log1p(144.))


def test_completion_applies_only_firing_side() -> None:
    """非発火側の潜在群を確定送量とみなさない。"""
    raw = dict(grids=np.zeros((2, 13, 6), dtype=np.int8))
    raw["grids"][:, -1, :4] = 1
    sent, counts, boards = f1.completion(raw, np.array([0, 1]), np.array([1, 0]), 0.)
    np.testing.assert_array_equal(counts, [1, 0])
    assert sent[1] == 0
    assert not boards[0]._grid.any()
    assert boards[1]._grid.any()


def test_bootstrap_checkpoint_without_pickle(tmp_path: Path) -> None:
    """動画IDを含めてallow_pickle=Falseで集計・再開できる。"""
    frame = pd.DataFrame(dict(video_id=["a", "a", "b", "b", "c", "c"],
        label=[0, 1, 0, 1, 0, 1], S1=[.2, .5, .7, .5, .8, .9]))
    frame.to_pickle(tmp_path / "middle.pkl")
    job = (str(tmp_path), "middle", "S1")
    path = Path(f1.bootstrap_job(job))
    data = f1.read_npz(path)
    assert data["videos"].dtype.kind == "U"
    assert data["values"].shape == (f1.BOOTSTRAPS, 2)
    assert f1.bootstrap_job(job) == str(path)
