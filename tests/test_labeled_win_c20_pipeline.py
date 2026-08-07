"""#43 c系 labeled_win パイプライン (品質ゲート・動画選定) の単体テスト。"""
from __future__ import annotations

from scripts.build_labeled_win_quality_gate import (
    MAX_GAME_DURATION_SEC,
    MIN_GAME_DURATION_SEC,
    STAR_FORMAT_MAX_GAMES,
    gate_video,
)
from scripts.select_labeled_win_videos import is_tier_allowed, select_videos


def _game(idx: int, start: float, end: float, winner: str = "1P", confidence: str = "strict") -> dict:
    return {
        "game_abs_idx": idx, "start_sec": start, "end_sec": end,
        "winner": winner, "left_hamming": 30, "right_hamming": 0,
        "confidence": confidence,
    }


def test_gate_video_excludes_star_format_when_games_too_few() -> None:
    games = [_game(i, i * 40.0, i * 40.0 + 30.0) for i in range(STAR_FORMAT_MAX_GAMES)]
    result, gated = gate_video("video_c99", games)
    assert result.status == "excluded_star_format"
    assert all(g["winner"] is None for g in gated)


def test_gate_video_ok_marks_bad_duration_as_unknown() -> None:
    games = [
        _game(0, 0.0, 30.0),                                    # 正常
        _game(1, 30.0, 30.0 + MIN_GAME_DURATION_SEC - 1.0),     # 短すぎ
        _game(2, 100.0, 100.0 + MAX_GAME_DURATION_SEC + 1.0),   # 長すぎ
    ]
    result, gated = gate_video("video_c1", games)
    assert result.status == "ok"
    assert gated[0]["winner"] == "1P"
    assert gated[1]["winner"] is None
    assert gated[2]["winner"] is None
    assert result.n_games_duration_bad == 2


def test_gate_video_ok_marks_ambiguous_confidence_as_unknown() -> None:
    # STAR_FORMAT_MAX_GAMES(既定2)以下だと星形式扱いになるため3試合以上で検証する
    games = [
        _game(0, 0.0, 40.0, confidence="strict"),
        _game(1, 40.0, 80.0, confidence="asymmetric"),
        _game(2, 80.0, 120.0, confidence="strict"),
    ]
    result, gated = gate_video("video_c1", games)
    assert result.status == "ok"
    assert gated[0]["winner"] == "1P"
    assert gated[1]["winner"] is None
    assert gated[2]["winner"] == "1P"
    assert result.n_games_ambiguous_conf == 1


def test_is_tier_allowed_excludes_unconfirmed_and_mixed_tier() -> None:
    assert not is_tier_allowed("video_c1")   # 未確認 (テストDL由来)
    assert not is_tier_allowed("video_c3")   # 未確認
    assert is_tier_allowed("video_c4")       # チャレンジャー範囲の先頭
    assert is_tier_allowed("video_c33")      # チャレンジャー範囲の末尾
    assert is_tier_allowed("video_c34")      # マスター範囲の先頭
    assert is_tier_allowed("video_c84")      # S級決定戦 (マスター範囲末尾)
    assert not is_tier_allowed("video_c85")  # 混成トーナメント (下位級混入)
    assert not is_tier_allowed("video_c95")  # マスター進出決定 (未確定)


def test_select_videos_orders_by_index_and_respects_n() -> None:
    rows = [
        {"video_id": "video_c22", "status": "ok"},
        {"video_id": "video_c10", "status": "ok"},
        {"video_id": "video_c1", "status": "ok"},   # tier未確認 -> 除外
        {"video_id": "video_c85", "status": "ok"},  # tier対象外 -> 除外
        {"video_id": "video_c11", "status": "excluded_star_format"},  # ステータス不可
    ]
    selected, excluded_log = select_videos(rows, n=2)
    assert selected == ["video_c10", "video_c22"]
    # n=2 に達した時点で走査を打ち切るため、video_c85 は評価されずログに残らない
    assert len(excluded_log) == 2


def test_select_videos_reports_shortfall_when_not_enough_candidates() -> None:
    rows = [{"video_id": "video_c10", "status": "ok"}]
    selected, _ = select_videos(rows, n=5)
    assert selected == ["video_c10"]
