"""scripts/extract_match_winners.py の --panel-diff-mode 関連ロジックのテスト。

detect_panel_increments 自体は cv2.VideoCapture 依存のためここでは対象にしない
(既存 detect_match_starts も同様に統合テスト対象外)。ここでは新規に切り出した
純粋関数 (_panel_diff_step / game_records_from_panel_diff) を検証する。
"""
from __future__ import annotations

from scripts.extract_match_winners import (
    PANEL_DIFF_CONFIRM_COUNT,
    PanelIncrementEvent,
    _panel_diff_step,
    _resolve_confidence,
    game_records_from_panel_diff,
)


def test_panel_diff_step_no_change_resets_pending() -> None:
    """変化なし (winner=None) は pending をリセットし未確定を返す。"""
    confirmed, pending_winner, pending_count = _panel_diff_step(
        winner=None, pending_winner="1P", pending_count=1,
    )
    assert confirmed is False
    assert pending_winner is None
    assert pending_count == 0


def test_panel_diff_step_confirms_after_n_consecutive() -> None:
    """同じ勝者が confirm_count 回連続すると確定する。"""
    confirmed, pending_winner, pending_count = _panel_diff_step(
        winner="1P", pending_winner=None, pending_count=0, confirm_count=2,
    )
    assert confirmed is False
    assert pending_winner == "1P"
    assert pending_count == 1

    confirmed, pending_winner, pending_count = _panel_diff_step(
        winner="1P", pending_winner=pending_winner, pending_count=pending_count, confirm_count=2,
    )
    assert confirmed is True
    assert pending_winner is None
    assert pending_count == 0


def test_panel_diff_step_switching_side_restarts_count() -> None:
    """1 回目 1P、2 回目 2P のように勝者側が入れ替わると再カウントする。"""
    confirmed, pending_winner, pending_count = _panel_diff_step(
        winner="1P", pending_winner=None, pending_count=0, confirm_count=2,
    )
    assert confirmed is False and pending_winner == "1P" and pending_count == 1

    confirmed, pending_winner, pending_count = _panel_diff_step(
        winner="2P", pending_winner=pending_winner, pending_count=pending_count, confirm_count=2,
    )
    assert confirmed is False
    assert pending_winner == "2P"
    assert pending_count == 1


def test_panel_diff_step_default_confirm_count_matches_constant() -> None:
    """confirm_count 省略時は PANEL_DIFF_CONFIRM_COUNT を使う。"""
    pending_winner: str | None = None
    pending_count = 0
    confirmed = False
    for _ in range(PANEL_DIFF_CONFIRM_COUNT):
        confirmed, pending_winner, pending_count = _panel_diff_step(
            winner="1P", pending_winner=pending_winner, pending_count=pending_count,
        )
    assert confirmed is True


def test_resolve_confidence_strict() -> None:
    assert _resolve_confidence(dl=20, dr=0, winner="1P") == "strict"


def test_resolve_confidence_asymmetric() -> None:
    # strict の条件(片側<=DIGIT_SAME_HAMMING)を満たさないが winner はある想定
    assert _resolve_confidence(dl=15, dr=6, winner="1P") == "asymmetric"


def test_resolve_confidence_none_winner() -> None:
    assert _resolve_confidence(dl=30, dr=29, winner=None) == "none"


def _event(t: float, winner: str | None = "1P") -> PanelIncrementEvent:
    return PanelIncrementEvent(
        event_sec=t, winner=winner, left_hamming=20, right_hamming=0, confidence="strict",
    )


def test_game_records_from_panel_diff_basic_sequence() -> None:
    """増分イベント 3 件 -> 3 試合、start は直前イベント時刻を継承。"""
    events = [_event(100.0, "1P"), _event(160.0, "2P"), _event(250.0, "1P")]
    records = game_records_from_panel_diff(events, first_visible_sec=50.0)
    assert len(records) == 3
    assert records[0].start_sec == 50.0
    assert records[0].end_sec == 100.0
    assert records[0].winner == "1P"
    assert records[1].start_sec == 100.0
    assert records[1].end_sec == 160.0
    assert records[1].winner == "2P"
    assert records[2].start_sec == 160.0
    assert records[2].end_sec == 250.0
    assert records[2].game_abs_idx == 2


def test_game_records_from_panel_diff_empty_events() -> None:
    assert game_records_from_panel_diff([], first_visible_sec=10.0) == []


def test_game_records_from_panel_diff_no_panel_visible() -> None:
    events = [_event(100.0, "1P")]
    assert game_records_from_panel_diff(events, first_visible_sec=None) == []
