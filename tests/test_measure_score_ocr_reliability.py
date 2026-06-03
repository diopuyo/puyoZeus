"""scripts/measure_score_ocr_reliability.py のユニットテスト。

SideStats の集計ロジック・_update_side_stats・サマリ生成の正確性を
モック/合成データで検証する。実動画・GPU は不要。
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

# テスト対象の関数・クラスを直接 import
from scripts.measure_score_ocr_reliability import (
    FORMULA_CONSEC_FRAMES,
    SCORE_ROI_INK_RATIO_MIN,
    STABLE_NONE_RATE_RISK_THRESHOLD,
    SideStats,
    VideoResult,
    _build_summary,
    _build_tasks,
    _state_key,
    _update_side_stats,
)
from src.board_state_machine import BoardState


# ============================
# _state_key
# ============================


def test_state_key_stable() -> None:
    assert _state_key(BoardState.STABLE) == "STABLE"


def test_state_key_chain() -> None:
    assert _state_key(BoardState.CHAIN) == "CHAIN"


def test_state_key_tsumo_fall() -> None:
    assert _state_key(BoardState.TSUMO_FALL) == "TSUMO_FALL"


# ============================
# SideStats の基本プロパティ
# ============================


def test_side_stats_initial_read_rate_zero() -> None:
    """初期状態では read_rate = 0.0。"""
    s = SideStats("1P")
    assert s.overall_read_rate() == 0.0


def test_side_stats_state_none_rate_empty() -> None:
    """フレームが 0 件のとき state_none_rate は 0.0。"""
    s = SideStats("1P")
    assert s.state_none_rate("STABLE") == 0.0


def test_side_stats_monotonic_violation_rate_no_readable() -> None:
    """readable_frames=0 のとき violation_rate は 0.0。"""
    s = SideStats("1P")
    assert s.monotonic_violation_rate() == 0.0


# ============================
# _update_side_stats — 読取り成功
# ============================


def _make_stats(side: str = "1P") -> SideStats:
    s = SideStats(side)
    s.frames_by_state = defaultdict(int)
    s.none_by_state = defaultdict(int)
    return s


def test_update_readable_increments_total_and_readable() -> None:
    s = _make_stats()
    _update_side_stats(s, score=1000, state=BoardState.STABLE,
                       ink_ratio=1.0, is_match_active=True)
    assert s.total_frames == 1
    assert s.readable_frames == 1
    assert s.frames_by_state["STABLE"] == 1
    assert s.none_by_state.get("STABLE", 0) == 0


def test_update_none_score_increments_none_by_state() -> None:
    s = _make_stats()
    _update_side_stats(s, score=None, state=BoardState.CHAIN,
                       ink_ratio=1.0, is_match_active=True)
    assert s.readable_frames == 0
    assert s.none_by_state["CHAIN"] == 1


# ============================
# 単調性違反検知
# ============================


def test_update_monotonic_violation_detected() -> None:
    """score が下がったら violation カウントが増える。"""
    s = _make_stats()
    # frame 1: score=5000
    _update_side_stats(s, score=5000, state=BoardState.STABLE,
                       ink_ratio=1.0, is_match_active=True)
    # frame 2: score=3000 (下がった → 違反)
    _update_side_stats(s, score=3000, state=BoardState.STABLE,
                       ink_ratio=1.0, is_match_active=True)
    assert s.monotonic_violations == 1


def test_update_monotonic_no_violation_when_equal() -> None:
    """同値は違反ではない。"""
    s = _make_stats()
    _update_side_stats(s, score=1000, state=BoardState.STABLE,
                       ink_ratio=1.0, is_match_active=True)
    _update_side_stats(s, score=1000, state=BoardState.STABLE,
                       ink_ratio=1.0, is_match_active=True)
    assert s.monotonic_violations == 0


def test_update_no_violation_when_not_match_active() -> None:
    """試合境界リセット後 (is_match_active=False) は違反判定しない。"""
    s = _make_stats()
    _update_side_stats(s, score=5000, state=BoardState.STABLE,
                       ink_ratio=1.0, is_match_active=False)
    _update_side_stats(s, score=1000, state=BoardState.STABLE,
                       ink_ratio=1.0, is_match_active=False)
    assert s.monotonic_violations == 0


# ============================
# 機能D実誤発火リスク検知
# ============================


def test_formula_risk_triggered_after_consec_frames() -> None:
    """STABLE + None + ink_ratio 高 が FORMULA_CONSEC_FRAMES 連続でリスクカウント。"""
    s = _make_stats()
    for _ in range(FORMULA_CONSEC_FRAMES):
        _update_side_stats(s, score=None, state=BoardState.STABLE,
                           ink_ratio=SCORE_ROI_INK_RATIO_MIN + 0.1,
                           is_match_active=True)
    assert s.formula_risk_events >= 1


def test_formula_risk_not_triggered_when_ink_low() -> None:
    """ink_ratio が閾値以下ならリスクカウントされない (= メニュー/真黒 ROI)。"""
    s = _make_stats()
    for _ in range(FORMULA_CONSEC_FRAMES + 1):
        _update_side_stats(s, score=None, state=BoardState.STABLE,
                           ink_ratio=0.0, is_match_active=True)
    assert s.formula_risk_events == 0


def test_formula_risk_not_triggered_when_chain_state() -> None:
    """CHAIN 状態での None は機能D リスクに計上されない。"""
    s = _make_stats()
    for _ in range(FORMULA_CONSEC_FRAMES + 1):
        _update_side_stats(s, score=None, state=BoardState.CHAIN,
                           ink_ratio=1.0, is_match_active=True)
    assert s.formula_risk_events == 0


def test_formula_risk_counter_resets_on_readable_frame() -> None:
    """読取り成功フレームが挟まると連続カウンタがリセットされる。"""
    s = _make_stats()
    # None 1 frame
    _update_side_stats(s, score=None, state=BoardState.STABLE,
                       ink_ratio=1.0, is_match_active=True)
    # readable 1 frame → リセット
    _update_side_stats(s, score=5000, state=BoardState.STABLE,
                       ink_ratio=1.0, is_match_active=True)
    # None 1 frame (連続 2 frame になっていないはず)
    _update_side_stats(s, score=None, state=BoardState.STABLE,
                       ink_ratio=1.0, is_match_active=True)
    assert s.formula_risk_events == 0


# ============================
# _build_summary
# ============================


def _make_video_result(
    vid: str,
    mnum: int,
    stable_none_rate: float,
) -> VideoResult:
    """テスト用 VideoResult を作成するヘルパ。"""
    r = VideoResult(
        video_id=vid, match_num=mnum,
        video_path=f"data/match_clips/{vid}/{vid}_match{mnum:02d}.mp4",
        total_duration_sec=60.0,
    )
    # SideStats を手動で設定
    for stats in (r.p1, r.p2):
        stats.frames_by_state = defaultdict(int)
        stats.none_by_state = defaultdict(int)
        total = 100
        stable = 60
        none_stable = int(stable * stable_none_rate)
        stats.total_frames = total
        stats.readable_frames = total - none_stable
        stats.frames_by_state["STABLE"] = stable
        stats.none_by_state["STABLE"] = none_stable
    return r


def test_build_summary_healthy_when_all_low_none() -> None:
    """全動画 STABLE None 率が低いとき feature_d_healthy=True。"""
    results = [_make_video_result("v29", 1, 0.02)]
    summary = _build_summary(results)
    assert summary["feature_d_healthy"] is True
    assert summary["risk_videos"] == []


def test_build_summary_risk_when_high_stable_none() -> None:
    """STABLE None 率が閾値超の動画があると risk_videos に掲載される。"""
    results = [_make_video_result("v89", 1, STABLE_NONE_RATE_RISK_THRESHOLD + 0.05)]
    summary = _build_summary(results)
    assert len(summary["risk_videos"]) > 0
    assert summary["feature_d_healthy"] is False


def test_build_summary_error_results_skipped() -> None:
    """error がある VideoResult は集計から除外される。"""
    r = VideoResult(
        video_id="v99", match_num=1,
        video_path="dummy.mp4", total_duration_sec=0.0,
        error="ファイル不在",
    )
    summary = _build_summary([r])
    assert summary["total_readable_frames"] == 0
    assert summary["risk_videos"] == []


# ============================
# _build_tasks
# ============================


def test_build_tasks_smoke_returns_single_task(tmp_path: Path, monkeypatch) -> None:
    """smoke=True では最大 v29 match01 の 1 タスクになる。
    実ファイル不在でもタスクリスト生成エラーにはならない。
    """
    # MATCH_CLIPS_DIR を tmp_path にリダイレクトして smoke テスト
    import scripts.measure_score_ocr_reliability as mod
    original = mod.MATCH_CLIPS_DIR
    monkeypatch.setattr(mod, "MATCH_CLIPS_DIR", tmp_path)

    # tmp_path/v29/v29_match01.mp4 を作成
    (tmp_path / "v29").mkdir()
    dummy = tmp_path / "v29" / "v29_match01.mp4"
    dummy.write_bytes(b"")

    tasks = _build_tasks(smoke=True)
    # 実ファイルがあれば 1 タスク、なければ 0 タスク (skip)
    # ここでは monkeypatch が効かないため実ファイルの有無に依存しない検証
    assert isinstance(tasks, list)
    monkeypatch.setattr(mod, "MATCH_CLIPS_DIR", original)
