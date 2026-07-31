"""VideoChainTracker の連鎖検出テスト。"""
from __future__ import annotations

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_RED,
    Board,
)
from src.chain import ChainSimulator
from src.chain_detector import (
    ChainEvent,
    VideoChainTracker,
    count_non_empty,
    track_chains,
)


def _empty_grid() -> list[list[int]]:
    return [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]


def _board_with_red4() -> Board:
    grid = _empty_grid()
    for r in range(9, 13):
        grid[r][0] = COLOR_RED
    return Board.from_list(grid)


def _empty_board() -> Board:
    return Board.from_list(_empty_grid())


def test_count_non_empty() -> None:
    assert count_non_empty(_empty_board()) == 0
    assert count_non_empty(_board_with_red4()) == 4


def test_tracker_no_event_on_initial_frame() -> None:
    tracker = VideoChainTracker()
    assert tracker.update(0.0, _board_with_red4()) is None


def test_tracker_detects_1chain_on_erasure() -> None:
    """赤 4 消去で 1 連鎖検出。全消しは「次連鎖発火時」に持ち越し（公式仕様）。"""
    tracker = VideoChainTracker()
    tracker.update(0.0, _board_with_red4())
    event = tracker.update(0.5, _empty_board())
    assert event is not None
    assert event.chain_count == 1
    assert event.total_erased == 4
    # 1 連鎖の素点 40 のみ。全消しボーナスは次連鎖に持ち越し
    assert event.base_score == 40
    assert event.total_score == 40
    assert event.all_clear_bonus_applied == 0
    assert event.is_all_clear is True
    # 次回発火に持ち越しフラグ
    assert tracker.all_clear_pending is True
    assert event.trigger_sec == 0.0
    assert event.end_sec == 0.5


def test_tracker_ignores_small_drops() -> None:
    """3 個以下の減少は連鎖扱いしない（ノイズ）。"""
    tracker = VideoChainTracker(erasure_min_drop=4)
    grid = _empty_grid()
    grid[12][0] = COLOR_RED
    grid[12][1] = COLOR_RED
    grid[12][2] = COLOR_RED
    tracker.update(0.0, Board.from_list(grid))
    tracker.update(0.5, _empty_board())  # 3 個減少
    # 消去ありだが ChainSimulator は発火しない（4 個未満）
    # → chain_count=0 で event=None
    # update は統計更新するが event は返さない
    # もう一度更新しても変わらない
    event = tracker.update(1.0, _empty_board())
    assert event is None


def test_tracker_2chain_event() -> None:
    """2 連鎖が発火する盤面で連鎖数 2 と得点が一致。"""
    # scoring テストの 2連鎖構成を流用
    grid = _empty_grid()
    grid[11][0] = COLOR_RED
    grid[12][0] = COLOR_RED
    grid[10][0] = COLOR_RED
    grid[9][0] = COLOR_RED
    grid[8][0] = COLOR_BLUE
    grid[7][0] = COLOR_BLUE
    grid[12][1] = COLOR_BLUE
    grid[11][1] = COLOR_BLUE
    grid[10][1] = COLOR_BLUE
    board = Board.from_list(grid)
    tracker = VideoChainTracker()
    tracker.update(0.0, board)
    ev = tracker.update(1.0, _empty_board())
    assert ev is not None
    assert ev.chain_count == 2
    # 素点 540、全消し持ち越しは次回（このイベントには適用されない）
    assert ev.base_score == 540
    assert ev.total_score == 540
    assert ev.all_clear_bonus_applied == 0
    assert ev.is_all_clear is True


def test_tracker_ojama_calculation() -> None:
    """得点が 70 点以上で ojama が発生する（持ち越しなし、素点のみ）。"""
    # 2 連鎖(540 点) → 540/70 = 7 余り 50
    tracker = VideoChainTracker()
    grid = _empty_grid()
    grid[11][0] = COLOR_RED
    grid[12][0] = COLOR_RED
    grid[10][0] = COLOR_RED
    grid[9][0] = COLOR_RED
    grid[8][0] = COLOR_BLUE
    grid[7][0] = COLOR_BLUE
    grid[12][1] = COLOR_BLUE
    grid[11][1] = COLOR_BLUE
    grid[10][1] = COLOR_BLUE
    tracker.update(0.0, Board.from_list(grid))
    ev = tracker.update(1.0, _empty_board())
    assert ev is not None
    assert ev.total_score == 540
    assert ev.ojama_sent == 7
    assert ev.leftover_score == 50


def test_track_chains_batch() -> None:
    """バッチ処理 API でも同じ結果。"""
    frames = [
        (0.0, _board_with_red4()),
        (0.5, _empty_board()),
    ]
    events = track_chains(frames)
    assert len(events) == 1
    assert events[0].chain_count == 1
    # 素点 40 のみ（全消しは次回に持ち越し）
    assert events[0].total_score == 40
    assert events[0].is_all_clear is True


def test_tracker_all_clear_carryover_to_next_chain() -> None:
    """全消し後の次連鎖発火で +2100 ボーナス加算（公式仕様）。"""
    tracker = VideoChainTracker()
    # 1 回目: 全消し（赤 4 個のみ）
    tracker.update(0.0, _board_with_red4())
    ev1 = tracker.update(0.5, _empty_board())
    assert ev1 is not None
    assert ev1.is_all_clear is True
    assert ev1.total_score == 40
    assert ev1.all_clear_bonus_applied == 0
    assert tracker.all_clear_pending is True

    # 2 回目: 別の赤 4 連鎖（盤面右端、これは全消しではない）
    grid = _empty_grid()
    for r in range(9, 13):
        grid[r][5] = COLOR_RED
    grid[12][0] = COLOR_BLUE  # 残ぷよで全消し回避
    tracker.update(1.0, Board.from_list(grid))
    ev2 = tracker.update(1.5, _empty_grid_blue_remnant())
    assert ev2 is not None
    # 素点 40 + 持ち越し全消しボーナス 2100 = 2140
    assert ev2.base_score == 40
    assert ev2.all_clear_bonus_applied == 2100
    assert ev2.total_score == 2140
    # 次回への持ち越しは消費済（is_all_clear=False、全消しでないので）
    assert ev2.is_all_clear is False
    assert tracker.all_clear_pending is False


def _empty_grid_blue_remnant() -> Board:
    """全消しではない、青 1 個だけ残った盤面。"""
    grid = _empty_grid()
    grid[12][0] = COLOR_BLUE
    return Board.from_list(grid)


# ============================
# 修正C (2026-07-24): debounce_confirm_frames テスト
# ============================


def test_debounce_default_is_bit_identical() -> None:
    """debounce_confirm_frames 未指定 (既定 1) は従来通り即時確定する。"""
    tracker = VideoChainTracker()
    tracker.update(0.0, _board_with_red4())
    event = tracker.update(0.5, _empty_board())
    assert event is not None
    assert event.chain_count == 1


def test_debounce_rejects_1frame_flicker() -> None:
    """1 フレームだけの色フリッカー(drop がすぐ解消)は debounce>1 で握りつぶす。"""
    tracker = VideoChainTracker(debounce_confirm_frames=2)
    tracker.update(0.0, _board_with_red4())
    # フリッカー: 1 フレームだけ非空セルが 4 個消えたように見える (drop 検出)
    assert tracker.update(0.1, _empty_board()) is None
    # フリッカー解消 (元の赤4に戻る) → 候補は破棄され event は出ない
    event = tracker.update(0.2, _board_with_red4())
    assert event is None


def test_debounce_confirms_persistent_drop() -> None:
    """drop が debounce_confirm_frames 回連続で続けば ChainEvent を確定する。"""
    tracker = VideoChainTracker(debounce_confirm_frames=2)
    tracker.update(0.0, _board_with_red4())
    # 1 回目の drop 観測 (候補として保留、まだ event なし)
    assert tracker.update(0.1, _empty_board()) is None
    # 2 回目 (debounce_confirm_frames=2 に到達) → 確定
    event = tracker.update(0.2, _empty_board())
    assert event is not None
    assert event.chain_count == 1
    assert event.total_erased == 4
    # trigger_sec は最初の (フリッカー前) stable board の時刻
    assert event.trigger_sec == 0.0


def test_debounce_does_not_swallow_fast_followup_chain() -> None:
    """本物の速い追撃連鎖(短時間差)は debounce 確認中でも取りこぼさない。"""
    tracker = VideoChainTracker(debounce_confirm_frames=2)
    tracker.update(0.0, _board_with_red4())
    # 1 回目 drop 観測 (保留)
    assert tracker.update(0.03, _empty_board()) is None
    # 2 回目: 短時間差でも drop が継続していれば確定する (取りこぼさない)
    event = tracker.update(0.06, _empty_board())
    assert event is not None
    assert event.chain_count == 1
