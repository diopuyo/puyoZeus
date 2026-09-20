"""
ぷよぷよのゲームルール（重力・接地）に基づく単一フレーム盤面補正。

目的:
    CNN / HSV 分類器は各セルを独立に分類するため、物理的に不可能な状態
    （非空セルの直下が空 = 浮遊ぷよ）を出力することがある。
    ぷよぷよの重力ルールでは、連鎖アニメ中以外は全ての非空ぷよが接地
    または別ぷよの上に乗っているはず。

    本モジュールは単一フレームの盤面に列単位で重力を適用して、
    各列の非空セルを最下段側へ詰めた盤面を返す。

前提（重要）:
    - 入力は「CNN 観測結果」であり、真の盤面ではない
    - 列ごとの非空セル「集合」は CNN が正しく検出した前提で、垂直位置のみ
      修正する（色分類の誤りまでは補正しない）
    - 連鎖アニメ中のフレームは上位ぷよが一時的に浮いていることがあるが、
      本補正は「最も自然な接地状態」を推定するだけなので、連鎖演出中の
      フレームでは過補正になる可能性がある（必要に応じて発動抑制）
    - HIDDEN_ROWS（row 0）は画面外のため補正対象外
    - COLOR_UNKNOWN は位置固定で扱う（補正しない）

使い方:
    corrected = apply_gravity(board)
    changes = diff_boards(original, corrected)   # 変化セル一覧
"""
from __future__ import annotations

from dataclasses import dataclass

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_UNKNOWN,
    HIDDEN_ROWS,
    Board,
)


@dataclass(frozen=True)
class CellChange:
    """補正による 1 セルの変化情報。"""
    row: int
    col: int
    before: int
    after: int


def apply_gravity(board: Board, skip_hidden: bool = True) -> Board:
    """
    各列の非空セルを最下段側へ詰めた盤面を返す。

    Args:
        board: 元盤面（非破壊、コピーを返す）
        skip_hidden: True なら HIDDEN_ROWS 行は触らず固定する

    Returns:
        重力適用後の盤面
    """
    start_row = HIDDEN_ROWS if skip_hidden else 0
    grid = [[board.get(r, c) for c in range(BOARD_COLS)] for r in range(BOARD_ROWS)]

    for col in range(BOARD_COLS):
        # 対象行範囲から UNKNOWN 以外の非空セルを収集（上→下の順）
        colored: list[int] = []
        unknown_positions: list[int] = []
        for row in range(start_row, BOARD_ROWS):
            val = grid[row][col]
            if val == COLOR_UNKNOWN:
                unknown_positions.append(row)
            elif val != COLOR_EMPTY:
                colored.append(val)
        # UNKNOWN は位置固定なので残す。それ以外の行を一旦空にしてから下詰め
        for row in range(start_row, BOARD_ROWS):
            if grid[row][col] != COLOR_UNKNOWN:
                grid[row][col] = COLOR_EMPTY
        # 下から詰める（UNKNOWN 行はスキップ）
        write_row = BOARD_ROWS - 1
        for color in reversed(colored):
            while write_row >= start_row and grid[write_row][col] == COLOR_UNKNOWN:
                write_row -= 1
            if write_row < start_row:
                break
            grid[write_row][col] = color
            write_row -= 1

    return Board.from_list(grid)


def clear_floating_above_gap(
    board: Board,
    min_gap: int = 2,
    skip_hidden: bool = True,
) -> Board:
    """
    列内の下部スタックから 2 行以上の空白で隔てられた上部セルを空にする。

    物理ルール: 接地していないぷよは重力で落ちる。static frame で
    「下部スタックがあり、その上に 2+ 行の空白があり、さらにその上に
    孤立した非空セル」がある状態は、以下のいずれか:
        - UI オーバーレイ（×マーク、連鎖演出等）の誤認
        - 連鎖中の中空落下アニメ（transient state）
        - 落下中のネクストぷよ（static 判定から除外すべき）
    いずれも static 盤面としては「無いもの」として扱うのが安全。

    Args:
        board: 元盤面
        min_gap: スタックから何行以上離れたら「浮遊」扱いにするか（既定 2）
        skip_hidden: True なら HIDDEN_ROWS は触らない

    Returns:
        浮遊セルを空に置き換えた新盤面（元は変更しない）
    """
    start_row = HIDDEN_ROWS if skip_hidden else 0
    grid = [[board.get(r, c) for c in range(BOARD_COLS)] for r in range(BOARD_ROWS)]

    for col in range(BOARD_COLS):
        # r12 から上に走査し、下部「接地スタック」の最上段を見つける
        stack_top = BOARD_ROWS  # 未発見マーカー
        for row in range(BOARD_ROWS - 1, start_row - 1, -1):
            val = grid[row][col]
            if val == COLOR_EMPTY or val == COLOR_UNKNOWN:
                break
            stack_top = row
        if stack_top == BOARD_ROWS:
            # 接地ぷよなし。列全体が空 or UNKNOWN のみ → 何もしない
            continue
        # スタック上方に空白が続いて、さらにその上に非空セルがあれば
        # 浮遊とみなす（min_gap 以上の空白が条件）
        gap_start = stack_top - 1
        if gap_start < start_row:
            continue
        # gap を数える
        gap = 0
        scan_row = gap_start
        while scan_row >= start_row and grid[scan_row][col] in (COLOR_EMPTY, COLOR_UNKNOWN):
            gap += 1
            scan_row -= 1
        if gap < min_gap:
            continue
        # scan_row はスタック外の非空セル or start_row-1
        # scan_row >= start_row の全非空セルを空にする
        for r in range(start_row, scan_row + 1):
            if grid[r][col] not in (COLOR_EMPTY, COLOR_UNKNOWN):
                grid[r][col] = COLOR_EMPTY

    return Board.from_list(grid)


def diff_boards(before: Board, after: Board) -> list[CellChange]:
    """2 つの盤面の差分セルを列挙する。"""
    changes: list[CellChange] = []
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            b = before.get(row, col)
            a = after.get(row, col)
            if b != a:
                changes.append(CellChange(row=row, col=col, before=b, after=a))
    return changes
