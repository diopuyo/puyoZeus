"""実表示の密なタイムラインdumpの回帰テスト。"""
from __future__ import annotations

import inspect

import numpy as np

import scripts.visualize_advantage_overlay as vao


def _row(source: str = "resolved_hold") -> vao.DisplayTimelineRow:
    return vao.DisplayTimelineRow(
        t_sec=1.25, game_idx=2, display_adv=87.0, display_p1=0.935,
        adv_raw_last=12.0, source=source, resolved_active=True,
        settled_ran=False, state1="CHAIN", state2="STABLE",
        score1=1200, score2=800,
    )


def test_display_source_priority() -> None:
    assert vao._display_timeline_source(True, True, True, True) == "episode_guard"
    assert vao._display_timeline_source(
        True, True, True, True, True) == "episode_guard"
    assert vao._display_timeline_source(
        True, True, True, False, True) == "minimum_prediction_guard"
    assert vao._display_timeline_source(True, True, True) == "resolved_hold"
    assert vao._display_timeline_source(False, True, False) == "resolved_release"
    assert vao._display_timeline_source(False, False, True) == "settled"
    assert vao._display_timeline_source(False, False, False) == "frozen"


def test_save_display_timeline_roundtrip(tmp_path) -> None:
    path = tmp_path / "dense" / "display.npz"
    vao.save_display_timeline(path, "video_x", [_row()])
    with np.load(path, allow_pickle=False) as data:
        assert str(data["video_id"]) == "video_x"
        assert data["t_sec"].tolist() == [1.25]
        assert data["display_adv"].tolist() == [87.0]
        assert data["source"].tolist() == ["resolved_hold"]
        assert data["resolved_active"].tolist() == [True]


def test_dense_dump_is_optional_and_cli_wired() -> None:
    prm = inspect.signature(vao.generate).parameters["dump_display_timeline_path"]
    assert prm.default is None
    src = inspect.getsource(vao.main)
    assert "--dump-display-timeline" in src
    assert "dump_display_timeline_path=a.dump_display_timeline_path" in src


def test_minimum_prediction_guard_is_optional_and_cli_wired() -> None:
    prm = inspect.signature(vao.generate).parameters[
        "enable_resolved_minimum_prediction_guard"]
    assert prm.default is False
    src = inspect.getsource(vao.main)
    assert "--resolved-minimum-prediction-guard" in src
    assert "enable_resolved_minimum_prediction_guard" in src
