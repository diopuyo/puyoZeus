"""ぷよぷよ物理ルール整合性チェック.

Phase I 自己教師あり学習の bootstrap 自己強化問題対策モジュール。

「5 frame 連続同色 = settle」の単純規則だけだと、誤認が連続して固定化する
リスクがあるため、ぷよぷよルール (4+連結消去・重力・色種数 ≤5 等) で
擬似ラベルをクロス検証する。

主な API:
    - check_color_count(board): 色種数 ≤ 5 か
    - check_gravity_rule(board): 各列で puyo が下端から連続 (空中 puyo なし)
    - check_no_pre_chain_4_plus_connection(board): STABLE で 4+ 連結が残ってない
    - check_cell_color_settle_consistency(color, board, pos): 上記 3 種を統合
    - filter_pseudo_labels_by_consistency(samples, board_lookup_fn):
        擬似ラベルリストを物理整合性で filter

注: 本モジュールは並列稼働中の cell collection (CellColorValidator) には
影響しない。CellColorFineTuner からのみ呼ばれる。
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
    HIDDEN_ROWS,
    Board,
)


# ============================
# 定数
# ============================

# ぷよぷよ通: 1 試合で同時に出現する puyo 色は最大 5 色 (おじゃま除く)
MAX_COLORS_IN_GAME: int = 5

# 4 連結以上は STABLE 状態で存在し得ない (即座に連鎖消去するため)
MIN_CONNECTED_TO_VANISH: int = 4

# BFS 隣接 4 方向 (上下左右)
_NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (1, 0), (-1, 0), (0, 1), (0, -1),
)

# 物理推論で対象外とする色 (空・おじゃま・不明)
_NON_PUYO_COLORS: frozenset[int] = frozenset({
    COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN,
})


# ============================
# 公開: 個別チェック関数
# ============================


def check_color_count(board: Board) -> tuple[bool, set[int]]:
    """盤面の色種数 ≤ 5 をチェック (ojama, empty, unknown 除く).

    Args:
        board: 検査対象 Board.

    Returns:
        (is_valid, colors): 5 色以下なら True, 違反時は False.
    """
    colors: set[int] = set()
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            v = int(board.get(r, c))
            if v in _NON_PUYO_COLORS:
                continue
            colors.add(v)
    return len(colors) <= MAX_COLORS_IN_GAME, colors


def check_gravity_rule(board: Board) -> tuple[bool, list[tuple[int, int]]]:
    """重力ルール: 各列で puyo は下端から連続 (空中 puyo なし).

    UNKNOWN セルは「観測不能」扱いで違反検出に使わない (連続性を遮断しない)。

    Args:
        board: 検査対象 Board.

    Returns:
        (is_valid, violations): 空中 puyo の (row, col) 一覧.
    """
    violations: list[tuple[int, int]] = []
    for col in range(BOARD_COLS):
        has_empty_below: bool = False
        # 下端から上に走査
        for row in range(BOARD_ROWS - 1, HIDDEN_ROWS - 1, -1):
            v = int(board.get(row, col))
            if v == COLOR_EMPTY:
                has_empty_below = True
            elif v == COLOR_UNKNOWN:
                # UNKNOWN は判定保留 (連続性は遮断しない)
                continue
            else:
                if has_empty_below:
                    violations.append((row, col))
    return len(violations) == 0, violations


def check_no_pre_chain_4_plus_connection(
    board: Board,
) -> tuple[bool, list[dict[str, Any]]]:
    """STABLE 確定盤面で 4+ 同色連結が無いことを確認.

    chain 中なら正当だが、STABLE 状態で 4+ 同色連結が残っていたら
    即消去されるはずなので、認識ミス (誤色付与) を示唆する。

    おじゃまぷよは消去対象外なので 4+ 連結があっても違反としない。

    Args:
        board: 検査対象 Board.

    Returns:
        (is_valid, violations): 違反クラスタ一覧 [{color, cells}].
    """
    visited = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)
    violations: list[dict[str, Any]] = []
    for r in range(HIDDEN_ROWS, BOARD_ROWS):
        for c in range(BOARD_COLS):
            if visited[r, c]:
                continue
            color = int(board.get(r, c))
            if color in _NON_PUYO_COLORS:
                visited[r, c] = True
                continue
            cluster = _bfs_same_color_cluster(board, r, c, color, visited)
            if len(cluster) >= MIN_CONNECTED_TO_VANISH:
                violations.append({"color": color, "cells": cluster})
    return len(violations) == 0, violations


def _bfs_same_color_cluster(
    board: Board, start_r: int, start_c: int,
    color: int, visited: np.ndarray,
) -> list[tuple[int, int]]:
    """BFS で同色連結成分を抽出 (可視領域のみ)."""
    stack: list[tuple[int, int]] = [(start_r, start_c)]
    cluster: list[tuple[int, int]] = []
    while stack:
        ar, ac = stack.pop()
        if visited[ar, ac]:
            continue
        if int(board.get(ar, ac)) != color:
            continue
        visited[ar, ac] = True
        cluster.append((ar, ac))
        for dr, dc in _NEIGHBOR_OFFSETS:
            nr, nc = ar + dr, ac + dc
            if HIDDEN_ROWS <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS:
                if not visited[nr, nc]:
                    stack.append((nr, nc))
    return cluster


def check_cell_color_settle_consistency(
    settled_color: int,
    settled_board: Board,
    settled_pos: tuple[int, int],
) -> tuple[bool, str]:
    """settle 候補 cell が物理ルール整合か確認 (3 種統合).

    Args:
        settled_color: settle 後の確定色 (主張).
        settled_board: settle 時点の確定盤面 (cell 込み).
        settled_pos: (row, col) の cell 座標.

    Returns:
        (is_valid, reason_if_invalid): 不整合なら理由文字列.
    """
    # 1. 色種数チェック
    color_valid, colors = check_color_count(settled_board)
    if not color_valid:
        return False, f"color_count_violation:{len(colors)}_colors"

    # 2. 重力チェック
    gravity_valid, airborne = check_gravity_rule(settled_board)
    if not gravity_valid:
        return False, f"gravity_violation:airborne={airborne[:3]}"

    # 3. 4+ 連結チェック (STABLE で消えていない 4+ 連結 = 認識ミス)
    no_4plus, violations = check_no_pre_chain_4_plus_connection(settled_board)
    if not no_4plus:
        return False, (
            f"4plus_connection_violation:{len(violations)}_clusters"
        )

    # settled_pos / settled_color はこの実装では追加チェックには使わないが、
    # 将来「settled cell が違反クラスタ内か?」等の判定に使えるよう interface 保持
    _ = settled_color
    _ = settled_pos
    return True, ""


# ============================
# 公開: 擬似ラベル filter
# ============================


def filter_pseudo_labels_by_consistency(
    samples: list[Any],
    board_lookup_fn: Optional[Callable[[float, str], Optional[Board]]],
) -> tuple[list[Any], dict[str, int]]:
    """擬似ラベルリストを物理整合性で filter.

    Args:
        samples: list[PseudoLabelSample].
        board_lookup_fn: callable(timestamp, side) -> Board | None.
            None または lookup 失敗時は当該 sample を保留せず通す
            (board 不明では物理 check 不能なため、過剰除外を避ける)。

    Returns:
        (filtered_samples, stats):
            stats = {n_in, n_out, n_color_violation, n_gravity_violation,
                     n_4plus_violation, n_other, n_no_board}
    """
    out: list[Any] = []
    stats: dict[str, int] = {
        "n_in": len(samples),
        "n_out": 0,
        "n_color_violation": 0,
        "n_gravity_violation": 0,
        "n_4plus_violation": 0,
        "n_other": 0,
        "n_no_board": 0,
    }
    if board_lookup_fn is None:
        # board 取得手段なし → no-op (後方互換)
        out.extend(samples)
        stats["n_out"] = len(samples)
        stats["n_no_board"] = len(samples)
        return out, stats

    for s in samples:
        keep, reason = _classify_one_sample(s, board_lookup_fn)
        _update_stats(stats, reason)
        if keep:
            out.append(s)
    stats["n_out"] = len(out)
    return out, stats


def _classify_one_sample(
    sample: Any,
    board_lookup_fn: Callable[[float, str], Optional[Board]],
) -> tuple[bool, str]:
    """1 サンプルを物理整合性で判定."""
    try:
        meta = getattr(sample, "metadata", {}) or {}
        input_data = getattr(sample, "input_data", None)
        # side / row / col は input_data 優先、なければ metadata から
        side, row, col = _extract_position(input_data, meta)
        color = _extract_int_label(getattr(sample, "label", None))
        if color is None or row is None or col is None:
            return False, "other"
        timestamp = float(getattr(sample, "timestamp", 0.0))
        board = board_lookup_fn(timestamp, side)
        if board is None:
            return False, "no_board"
        valid, reason = check_cell_color_settle_consistency(
            color, board, (row, col),
        )
        if valid:
            return True, "ok"
        return False, _categorize_reason(reason)
    except Exception:
        return False, "other"


def _extract_position(
    input_data: Any, meta: dict[str, Any],
) -> tuple[str, Optional[int], Optional[int]]:
    """side / row / col を input_data → metadata の順で抽出."""
    side: str = "1P"
    row: Optional[int] = None
    col: Optional[int] = None
    if isinstance(input_data, dict):
        side = str(input_data.get("side", side))
        row = _safe_int(input_data.get("row"))
        col = _safe_int(input_data.get("col"))
    if row is None:
        row = _safe_int(meta.get("row"))
    if col is None:
        col = _safe_int(meta.get("col"))
    if "side" in meta and isinstance(input_data, dict) is False:
        side = str(meta.get("side", side))
    return side, row, col


def _safe_int(v: Any) -> Optional[int]:
    """None 安全な int 変換."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _extract_int_label(label: Any) -> Optional[int]:
    """label を int に変換できればその値, 不可なら None."""
    if label is None:
        return None
    try:
        return int(label)
    except (TypeError, ValueError):
        return None


def _categorize_reason(reason: str) -> str:
    """check_cell_color_settle_consistency の reason を stat key に変換."""
    if "color_count" in reason:
        return "color_violation"
    if "gravity" in reason:
        return "gravity_violation"
    if "4plus" in reason:
        return "4plus_violation"
    return "other"


def _update_stats(stats: dict[str, int], reason: str) -> None:
    """stats dict を reason 文字列で更新 (副作用)."""
    if reason == "ok":
        return
    if reason == "no_board":
        stats["n_no_board"] += 1
        return
    if reason == "color_violation":
        stats["n_color_violation"] += 1
        return
    if reason == "gravity_violation":
        stats["n_gravity_violation"] += 1
        return
    if reason == "4plus_violation":
        stats["n_4plus_violation"] += 1
        return
    stats["n_other"] += 1


__all__ = [
    "MAX_COLORS_IN_GAME",
    "MIN_CONNECTED_TO_VANISH",
    "check_color_count",
    "check_gravity_rule",
    "check_no_pre_chain_4_plus_connection",
    "check_cell_color_settle_consistency",
    "filter_pseudo_labels_by_consistency",
]
