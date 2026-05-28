"""B1 PiecePersistenceGuard のテスト。

設計原則の遵守確認:
- STABLE 中に非 EMPTY cell を保護登録
- protected cell が EMPTY/UNKNOWN に変わる更新を block
- NON-STABLE 遷移で保護リセット
- 「ぷよを消す経路」 を構造的に作らない (EMPTY → 非 EMPTY 変換禁止)
"""
from __future__ import annotations

import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_RED,
    COLOR_UNKNOWN,
    Board,
)
from src.piece_persistence_guard import PiecePersistenceGuard


def _board_with(color_at: dict[tuple[int, int], int]) -> Board:
    """指定 cell のみ色を入れた Board を生成する。"""
    b = Board()
    for (r, c), v in color_at.items():
        b.set(r, c, v)
    return b


# ============================================================
# 初期化
# ============================================================

def test_init_empty_protected() -> None:
    """初期状態で _protected は空、 _in_stable=False。"""
    guard = PiecePersistenceGuard()
    assert len(guard._protected) == 0
    assert guard._in_stable is False


# ============================================================
# on_stable_confirmed
# ============================================================

def test_on_stable_confirmed_records_non_empty() -> None:
    """STABLE 確定で非 EMPTY cell を保護登録する。"""
    guard = PiecePersistenceGuard()
    board = _board_with({(5, 2): COLOR_RED, (6, 3): COLOR_BLUE})
    guard.on_stable_confirmed(board)
    assert (5, 2) in guard._protected
    assert guard._protected[(5, 2)] == COLOR_RED
    assert (6, 3) in guard._protected
    assert guard._protected[(6, 3)] == COLOR_BLUE
    assert guard._in_stable is True


def test_on_stable_confirmed_skips_empty() -> None:
    """EMPTY cell は保護登録しない。"""
    guard = PiecePersistenceGuard()
    board = _board_with({(5, 2): COLOR_EMPTY, (6, 3): COLOR_RED})
    guard.on_stable_confirmed(board)
    assert (5, 2) not in guard._protected
    assert (6, 3) in guard._protected


def test_on_stable_confirmed_skips_unknown() -> None:
    """UNKNOWN cell は保護登録しない。"""
    guard = PiecePersistenceGuard()
    board = _board_with({(3, 1): COLOR_UNKNOWN, (4, 2): COLOR_BLUE})
    guard.on_stable_confirmed(board)
    assert (3, 1) not in guard._protected
    assert (4, 2) in guard._protected


# ============================================================
# guard
# ============================================================

def test_guard_blocks_color_to_empty() -> None:
    """protected cell が EMPTY に変わる更新を block して元色を維持する。"""
    guard = PiecePersistenceGuard()
    board = _board_with({(5, 2): COLOR_RED})
    guard.on_stable_confirmed(board)
    # candidate で (5, 2) が EMPTY になる
    candidate = _board_with({})  # 全 EMPTY
    result = guard.guard(candidate)
    assert result.get(5, 2) == COLOR_RED  # block されて元色維持


def test_guard_blocks_color_to_unknown() -> None:
    """protected cell が UNKNOWN に変わる更新を block して元色を維持する。"""
    guard = PiecePersistenceGuard()
    board = _board_with({(4, 1): COLOR_BLUE})
    guard.on_stable_confirmed(board)
    candidate = _board_with({(4, 1): COLOR_UNKNOWN})
    result = guard.guard(candidate)
    assert result.get(4, 1) == COLOR_BLUE  # block されて元色維持


def test_guard_allows_color_to_color_change() -> None:
    """protected cell でも color → 別 color 変化は許容する (= 連鎖等の正常変化)。"""
    guard = PiecePersistenceGuard()
    board = _board_with({(5, 2): COLOR_RED})
    guard.on_stable_confirmed(board)
    # candidate で (5, 2) が BLUE に変わる
    candidate = _board_with({(5, 2): COLOR_BLUE})
    result = guard.guard(candidate)
    assert result.get(5, 2) == COLOR_BLUE  # 色変化は許容


def test_guard_allows_non_protected_cell() -> None:
    """保護対象外の cell は候補値をそのまま返す。"""
    guard = PiecePersistenceGuard()
    board = _board_with({(5, 2): COLOR_RED})
    guard.on_stable_confirmed(board)
    # (3, 0) は保護対象外
    candidate = _board_with({(3, 0): COLOR_BLUE})
    result = guard.guard(candidate)
    assert result.get(3, 0) == COLOR_BLUE  # そのまま


def test_guard_no_empty_to_non_empty_injection() -> None:
    """ぷよを消す経路を作らない: EMPTY → 非 EMPTY 変換は禁止。

    guard は protected cell への EMPTY → 元色 復元のみ行う。
    元々 EMPTY だった cell に非 EMPTY を注入することはしない。
    """
    guard = PiecePersistenceGuard()
    # 全空盤面を on_stable_confirmed に渡す (= 何も保護されない)
    empty_board = Board()
    guard.on_stable_confirmed(empty_board)
    candidate = _board_with({(5, 2): COLOR_EMPTY})
    result = guard.guard(candidate)
    # EMPTY → 非 EMPTY 注入は発生しない
    assert result.get(5, 2) == COLOR_EMPTY


# ============================================================
# on_non_stable_enter
# ============================================================

def test_on_non_stable_enter_clears_protected() -> None:
    """NON-STABLE 遷移で保護リセット。"""
    guard = PiecePersistenceGuard()
    board = _board_with({(5, 2): COLOR_RED})
    guard.on_stable_confirmed(board)
    assert len(guard._protected) > 0
    guard.on_non_stable_enter()
    assert len(guard._protected) == 0
    assert guard._in_stable is False


# ============================================================
# reset
# ============================================================

def test_reset_clears_all() -> None:
    """reset で完全リセット。"""
    guard = PiecePersistenceGuard()
    board = _board_with({(5, 2): COLOR_RED, (6, 3): COLOR_BLUE})
    guard.on_stable_confirmed(board)
    guard.reset()
    assert len(guard._protected) == 0
    assert guard._in_stable is False


# ============================================================
# to_dict
# ============================================================

def test_to_dict() -> None:
    """to_dict がシリアライズ可能な dict を返す。"""
    guard = PiecePersistenceGuard()
    board = _board_with({(5, 2): COLOR_RED})
    guard.on_stable_confirmed(board)
    d = guard.to_dict()
    assert isinstance(d, dict)
    assert d["n_protected"] == 1
    assert d["in_stable"] is True
    assert (5, 2) in d["protected_positions"]


# ============================================================
# guard が無効 (in_stable=False) の場合は素通し
# ============================================================

def test_guard_passthrough_when_not_in_stable() -> None:
    """on_stable_confirmed 未呼び出し (in_stable=False) のとき素通し。"""
    guard = PiecePersistenceGuard()
    candidate = Board()  # 全 EMPTY
    result = guard.guard(candidate)
    # 保護なしなので candidate そのまま返す
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            assert result.get(r, c) == COLOR_EMPTY
