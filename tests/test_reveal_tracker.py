"""Phase I: RevealTracker のテスト."""
from __future__ import annotations

import pytest

from src.board import (
    BOARD_COLS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_RED,
    Board,
)
from src.probabilistic_board import ProbabilisticBoard
from src.self_supervised.reveal_tracker import (
    DEFAULT_MAX_PENDING_AGE_SEC,
    PendingInference,
    RevealEvent,
    RevealTracker,
)


def _make_board(cells: dict[tuple[int, int], int]) -> Board:
    b = Board()
    for (r, c), color in cells.items():
        b.set(r, c, color)
    return b


def _make_pboard_hidden(col: int, color: int, prob: float = 1.0) -> ProbabilisticBoard:
    """row 0 の col 列に color を確率 prob で置いた ProbabilisticBoard."""
    pb = ProbabilisticBoard()
    if prob >= 1.0:
        pb.set_certain(0, col, color)
    else:
        pb.set_distribution(
            0, col, {color: prob, COLOR_EMPTY: 1.0 - prob},
        )
    return pb


# ============================
# 基本動作
# ============================


def test_init_pending_is_empty() -> None:
    tr = RevealTracker()
    assert len(tr) == 0
    assert tr.pending == []


def test_invalid_max_age_raises() -> None:
    with pytest.raises(ValueError):
        RevealTracker(max_pending_age_sec=0.0)
    with pytest.raises(ValueError):
        RevealTracker(max_pending_age_sec=-1.0)


def test_add_inference_with_colored_cell_appends() -> None:
    tr = RevealTracker()
    pb = _make_pboard_hidden(col=2, color=COLOR_RED, prob=1.0)
    tr.add_inference(t=1.0, side="1P", inferred=pb, state_snapshot={"k": 1})
    assert len(tr) == 1
    entry = tr.pending[0]
    assert entry.timestamp == 1.0
    assert entry.side == "1P"
    assert (0, 2) in entry.inferred_dist


def test_add_inference_all_empty_skipped() -> None:
    """色情報の無い (= 隠し段が EMPTY 確定) 推論は pending に入れない."""
    tr = RevealTracker()
    pb = ProbabilisticBoard()
    for c in range(BOARD_COLS):
        pb.set_certain(0, c, COLOR_EMPTY)
    tr.add_inference(t=1.0, side="1P", inferred=pb)
    assert len(tr) == 0


# ============================
# Reveal 検出
# ============================


def test_update_without_chain_or_drop_no_event() -> None:
    """was_chain_or_drop=False なら reveal 判定しない."""
    tr = RevealTracker()
    pb = _make_pboard_hidden(col=2, color=COLOR_RED)
    tr.add_inference(t=1.0, side="1P", inferred=pb)
    prev = _make_board({})
    cur = _make_board({(1, 2): COLOR_RED})  # row 1 に出現
    events = tr.update(
        t=2.0, side="1P",
        current_board=cur, prev_board=prev,
        was_chain_or_drop=False,
    )
    assert events == []
    assert len(tr) == 1  # pending は維持


def test_update_with_chain_drop_detects_match() -> None:
    """連鎖直後の遷移で row 1 に推論色が現れたら match RevealEvent を返す."""
    tr = RevealTracker()
    pb = _make_pboard_hidden(col=2, color=COLOR_RED)
    tr.add_inference(t=1.0, side="1P", inferred=pb)
    prev = _make_board({})
    cur = _make_board({(1, 2): COLOR_RED})
    events = tr.update(
        t=2.0, side="1P",
        current_board=cur, prev_board=prev,
        was_chain_or_drop=True,
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.side == "1P"
    assert ev.col == 2
    assert ev.observed_color == COLOR_RED
    assert ev.match is True
    assert ev.original_inference_t == 1.0
    # pending は消化済
    assert len(tr) == 0


def test_update_with_chain_drop_detects_mismatch() -> None:
    """推論色 != 観測色 なら match=False の RevealEvent を返す."""
    tr = RevealTracker()
    pb = _make_pboard_hidden(col=2, color=COLOR_RED)
    tr.add_inference(t=1.0, side="1P", inferred=pb)
    prev = _make_board({})
    cur = _make_board({(1, 2): COLOR_BLUE})  # 推論は RED だが観測は BLUE
    events = tr.update(
        t=2.0, side="1P",
        current_board=cur, prev_board=prev,
        was_chain_or_drop=True,
    )
    assert len(events) == 1
    assert events[0].match is False
    assert events[0].observed_color == COLOR_BLUE


def test_update_filters_by_side() -> None:
    """side が違う pending は消化されない."""
    tr = RevealTracker()
    pb = _make_pboard_hidden(col=2, color=COLOR_RED)
    tr.add_inference(t=1.0, side="2P", inferred=pb)
    prev = _make_board({})
    cur = _make_board({(1, 2): COLOR_RED})
    events = tr.update(
        t=2.0, side="1P",
        current_board=cur, prev_board=prev,
        was_chain_or_drop=True,
    )
    assert events == []
    assert len(tr) == 1  # 2P pending は残る


def test_update_no_reveal_columns() -> None:
    """row 1 が変化しなければ reveal イベントなし."""
    tr = RevealTracker()
    pb = _make_pboard_hidden(col=2, color=COLOR_RED)
    tr.add_inference(t=1.0, side="1P", inferred=pb)
    cells = {(12, 2): COLOR_BLUE}
    prev = _make_board(cells)
    cur = _make_board(cells)
    events = tr.update(
        t=2.0, side="1P",
        current_board=cur, prev_board=prev,
        was_chain_or_drop=True,
    )
    assert events == []


def test_update_ojama_at_row1_not_treated_as_reveal() -> None:
    """row 1 が OJAMA に変わったケースは reveal とみなさない."""
    tr = RevealTracker()
    pb = _make_pboard_hidden(col=2, color=COLOR_RED)
    tr.add_inference(t=1.0, side="1P", inferred=pb)
    prev = _make_board({})
    cur = _make_board({(1, 2): COLOR_OJAMA})
    events = tr.update(
        t=2.0, side="1P",
        current_board=cur, prev_board=prev,
        was_chain_or_drop=True,
    )
    assert events == []


def test_update_prev_was_ojama_now_color_treated_as_reveal() -> None:
    """prev row 1 が OJAMA → cur 色付き (連鎖でおじゃまが消えて隠し段から puyo 落下) は reveal."""
    tr = RevealTracker()
    pb = _make_pboard_hidden(col=3, color=COLOR_GREEN)
    tr.add_inference(t=1.0, side="1P", inferred=pb)
    prev = _make_board({(1, 3): COLOR_OJAMA})
    cur = _make_board({(1, 3): COLOR_GREEN})
    events = tr.update(
        t=2.0, side="1P",
        current_board=cur, prev_board=prev,
        was_chain_or_drop=True,
    )
    assert len(events) == 1
    assert events[0].col == 3
    assert events[0].match is True


# ============================
# 期限切れ pending の破棄
# ============================


def test_cleanup_stale_drops_old_entries() -> None:
    tr = RevealTracker(max_pending_age_sec=5.0)
    pb = _make_pboard_hidden(col=0, color=COLOR_RED)
    tr.add_inference(t=0.0, side="1P", inferred=pb)
    tr.add_inference(t=10.0, side="1P", inferred=pb)
    n_dropped = tr.cleanup_stale(current_t=20.0)
    # 0.0 と 10.0 のうち、cutoff = 20 - 5 = 15 → 0.0 と 10.0 両方 drop
    assert n_dropped == 2
    assert len(tr) == 0


def test_cleanup_stale_keeps_recent() -> None:
    tr = RevealTracker(max_pending_age_sec=5.0)
    pb = _make_pboard_hidden(col=0, color=COLOR_RED)
    tr.add_inference(t=10.0, side="1P", inferred=pb)
    n_dropped = tr.cleanup_stale(current_t=12.0)
    assert n_dropped == 0
    assert len(tr) == 1


def test_update_calls_cleanup() -> None:
    """update() を呼ぶと cleanup_stale が走る."""
    tr = RevealTracker(max_pending_age_sec=2.0)
    pb = _make_pboard_hidden(col=0, color=COLOR_RED)
    tr.add_inference(t=0.0, side="1P", inferred=pb)
    # 5 秒後の update で 0.0 の pending は drop される
    prev = _make_board({})
    cur = _make_board({})
    _ = tr.update(
        t=5.0, side="1P",
        current_board=cur, prev_board=prev,
        was_chain_or_drop=True,
    )
    assert len(tr) == 0


def test_clear_removes_all() -> None:
    tr = RevealTracker()
    pb = _make_pboard_hidden(col=0, color=COLOR_RED)
    tr.add_inference(t=0.0, side="1P", inferred=pb)
    tr.add_inference(t=1.0, side="2P", inferred=pb)
    tr.clear()
    assert len(tr) == 0


# ============================
# 確率分布最尤色の選択
# ============================


def test_argmax_prefers_colored_over_empty() -> None:
    """EMPTY が最大確率でも 色付き色 を最尤として返す."""
    tr = RevealTracker()
    pb = ProbabilisticBoard()
    # RED 0.3 / EMPTY 0.7 でも、color comparison では RED が選ばれる
    pb.set_distribution(
        0, 2, {COLOR_RED: 0.3, COLOR_EMPTY: 0.7},
    )
    tr.add_inference(t=1.0, side="1P", inferred=pb)
    prev = _make_board({})
    cur = _make_board({(1, 2): COLOR_RED})
    events = tr.update(
        t=2.0, side="1P",
        current_board=cur, prev_board=prev,
        was_chain_or_drop=True,
    )
    assert len(events) == 1
    assert events[0].match is True
