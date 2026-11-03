"""F1bの未来盤面・NEXT・時刻の参照禁止を確認する。"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from scripts import audit_retrain_f1b_20260927 as f1b


def inputs() -> dict[str, np.ndarray]:
    """発火前と未来の行を分けた最小の保存データ。"""
    raw = dict(grids=np.zeros((4, 13, 6), dtype=np.int8),
        score=np.array([0, 0, 700, 1400]), game_idx=np.zeros(4, dtype=int),
        t_sec=np.array([1., 2., 20., 21.]))
    raw.update({key: np.ones(4, dtype=np.int8) for key in f1b.f1.QUEUE})
    return raw


def observed_features(raw: dict, ids: np.ndarray, sends: np.ndarray, counts: np.ndarray,
                      elapsed: float, boards: list | None = None) -> np.ndarray:
    """参照した入力を直接出力し、未来入力変更の不変性を検証する。"""
    return np.array([[raw["grids"][i].sum(), raw["next1_a"][i], raw["dnext_b"][i],
                      raw["t_sec"][i], elapsed, sends[side]] for side, i in enumerate(ids)])


@pytest.mark.parametrize("field", ["grids", "next1_a", "next1_b", "dnext_a", "dnext_b", "t_sec"])
def test_post_inputs_do_not_change_s3(field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """終点の盤面・NEXT・時刻を変えても是正S3が変化しない。"""
    raw = inputs()
    monkeypatch.setattr(f1b.f1, "stage_features", observed_features)
    args = (np.array([0, 1]), np.array([2, 3]), 10., {0: 0.}, np.array([1, 0]))
    before = f1b.corrected_event(raw, *args)
    raw[field][2:] = 5
    after = f1b.corrected_event(raw, *args)
    np.testing.assert_array_equal(before, after)


def test_score_remains_an_allowed_s3_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """得点差だけはS3の確定送り量として保持する。"""
    raw = inputs()
    monkeypatch.setattr(f1b.f1, "stage_features", observed_features)
    args = (np.array([0, 1]), np.array([2, 3]), 10., {0: 0.}, np.array([1, 0]))
    before = f1b.corrected_event(raw, *args)
    raw["score"][2] += 700
    after = f1b.corrected_event(raw, *args)
    assert after[0, -1]-before[0, -1] == 10


def test_audit_does_not_invent_s3_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """欠測評価時刻を終点時刻で埋めて全件合格にしない。"""
    raw = inputs()
    pre = dict(row_id=np.array([0, 1]), pre_ids=np.array([[0, 1], [-1, 1]]),
               trigger=np.array([10., 10.]), endpoint=np.array([21., 21.]))
    post = dict(row_id=np.array([0, 1]), post_ids=np.array([[2, 3], [2, 3]]))

    def read(path: Path) -> dict:
        if "timeline_s3" in str(path):
            return post
        if "timeline_" in str(path):
            return pre
        return raw

    monkeypatch.setattr(f1b.f1, "read_npz", read)
    frame = f1b.audit_video("video_test")
    assert frame.s3_evaluation_sec.isna().all()
    assert frame.s1_both_available.tolist() == [True, False]
    assert frame.s1_any_missing.tolist() == [False, True]
    assert not frame.s1_any_future.any()
    assert frame.old_s3_oracle_endpoint.all()


def test_missing_ids_do_not_read_last_row() -> None:
    """未取得盤面の-1を未来の末尾行に変換しない。"""
    assert np.isnan(f1b.board_times(inputs(), np.array([-1]))[0])


def test_legacy_entrypoint_uses_corrected_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """旧起動名でもリーク版の学習処理へ戻らない。"""
    calls = []
    monkeypatch.setattr(f1b, "main", lambda: calls.append("F1b"))
    f1b.f1.main()
    assert calls == ["F1b"]
