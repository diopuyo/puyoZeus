"""
盤面データ表現モジュール

ぷよぷよeスポーツの6列×13行の盤面をnumpy配列で管理する。
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

# ============================
# 定数定義
# ============================
BOARD_COLS: int = 6
BOARD_ROWS: int = 13
DEATH_COL: int = 2       # 窒息判定列 (0-indexed)
# 窒息判定行。
# user確定ルール (2026-07-22, docs/PUYO_RULES_CONFIRMED_2026-07-22.md):
#   死亡マスは「3列目の画面内一番上」= 隠し段(row0)を除いた可視最上段 = row1。
#   row0(隠し段)は回し入れ等で一時的にぷよが通過しうる非公開領域であり、
#   そこが埋まっただけでは窒息と判定しない (旧 DEATH_ROW=0 は判定が甘すぎた)。
DEATH_ROW: int = 1       # 窒息判定行 (可視最上段。隠し段 row0 は含まない)

# 可視領域の行数 (画面に表示される範囲)
# 隠し段 (row 0) は画面外で、回し入れ等で puyo が入ることがある
VISIBLE_ROWS: int = 12
HIDDEN_ROWS: int = BOARD_ROWS - VISIBLE_ROWS  # = 1

# 色定義
COLOR_EMPTY: int = 0
COLOR_RED: int = 1
COLOR_BLUE: int = 2
COLOR_GREEN: int = 3
COLOR_YELLOW: int = 4
COLOR_PURPLE: int = 5
COLOR_OJAMA: int = 9
# 画像から確定できないセル (隠し段・回し入れで不明、量子的未確定状態)
COLOR_UNKNOWN: int = 10

# ターミナル表示用カラーコード
_ANSI_COLORS: dict[int, str] = {
    COLOR_EMPTY:   "\033[0m  ",
    COLOR_RED:     "\033[41m R\033[0m",
    COLOR_BLUE:    "\033[44m B\033[0m",
    COLOR_GREEN:   "\033[42m G\033[0m",
    COLOR_YELLOW:  "\033[43m Y\033[0m",
    COLOR_PURPLE:  "\033[45m P\033[0m",
    COLOR_OJAMA:   "\033[47m O\033[0m",
    COLOR_UNKNOWN: "\033[100m ?\033[0m",
}

_COLOR_NAMES: dict[int, str] = {
    COLOR_EMPTY:   "空",
    COLOR_RED:     "赤",
    COLOR_BLUE:    "青",
    COLOR_GREEN:   "緑",
    COLOR_YELLOW:  "黄",
    COLOR_PURPLE:  "紫",
    COLOR_OJAMA:   "おじゃま",
    COLOR_UNKNOWN: "不明",
}

VALID_COLORS: frozenset[int] = frozenset(_COLOR_NAMES.keys())


class Board:
    """
    ぷよぷよeスポーツの盤面を表すクラス。

    内部表現: shape=(BOARD_ROWS, BOARD_COLS) の numpy.ndarray (uint8)
    行: 0=最上段(隠し段), 12=最下段
    列: 0=左端, 5=右端
    """

    def __init__(self) -> None:
        """空の盤面を生成する。"""
        self._grid: np.ndarray = np.zeros(
            (BOARD_ROWS, BOARD_COLS), dtype=np.uint8
        )

    # ============================
    # ファクトリメソッド
    # ============================

    @classmethod
    def from_list(cls, data: list[list[int]]) -> "Board":
        """
        2次元リストから盤面を生成する。

        Args:
            data: shape (BOARD_ROWS, BOARD_COLS) のリスト。
                  data[0] が最上段、data[12] が最下段。

        Returns:
            Board: 生成された盤面オブジェクト。

        Raises:
            ValueError: データのサイズまたは色値が不正な場合。
        """
        if len(data) != BOARD_ROWS:
            raise ValueError(
                f"行数が不正です: {len(data)} (期待値: {BOARD_ROWS})"
            )
        for row_idx, row in enumerate(data):
            if len(row) != BOARD_COLS:
                raise ValueError(
                    f"列数が不正です [行{row_idx}]: {len(row)} (期待値: {BOARD_COLS})"
                )
            for col_idx, val in enumerate(row):
                if val not in VALID_COLORS:
                    raise ValueError(
                        f"無効な色値 [{row_idx}][{col_idx}]: {val}"
                    )

        board = cls()
        board._grid = np.array(data, dtype=np.uint8)
        return board

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Board":
        """
        辞書からBoardを復元する (JSON読み込み用)。

        Args:
            data: to_dict() で生成した辞書。

        Returns:
            Board: 復元された盤面オブジェクト。
        """
        return cls.from_list(data["grid"])

    # ============================
    # セルアクセス
    # ============================

    def get(self, row: int, col: int) -> int:
        """指定セルの色値を返す。"""
        self._validate_pos(row, col)
        return int(self._grid[row, col])

    def set(self, row: int, col: int, color: int) -> None:
        """指定セルに色値をセットする。"""
        self._validate_pos(row, col)
        if color not in VALID_COLORS:
            raise ValueError(f"無効な色値: {color}")
        self._grid[row, col] = color

    # ============================
    # 盤面情報
    # ============================

    def height_of(self, col: int) -> int:
        """
        指定列のぷよの高さ（積み上げ数）を返す。
        UNKNOWN セルは高さ計算から除外 (隠し段が不確定でも可視分のみ評価)。

        Args:
            col: 列インデックス (0-5)

        Returns:
            int: 最下段から積み上げられたぷよの数 (UNKNOWN除く)。
        """
        if not (0 <= col < BOARD_COLS):
            raise ValueError(f"列インデックスが範囲外: {col}")
        column = self._grid[:, col]
        concrete = np.where((column != COLOR_EMPTY) & (column != COLOR_UNKNOWN))[0]
        if len(concrete) == 0:
            return 0
        top_row = int(concrete[0])
        return BOARD_ROWS - top_row

    def is_dead(self) -> bool:
        """
        窒息判定。3列目(index:2)の可視最上段(row=DEATH_ROW=1)にぷよがあればTrue。
        隠し段(row0)は判定に含めない (user確定ルール 2026-07-22)。
        COLOR_UNKNOWN は判定不能とみなして False (確定的には死んでいない)。

        【契約・厳守】本メソッドは STABLE (連鎖解決後) の静止盤面に対する
        静的判定としてのみ使うこと。**連鎖中 (CHAIN/GRAVITY_SETTLE 等の
        非STABLE state) の盤面には絶対に呼ばないこと。**
        呼出元が「同じ Board インスタンスを非STABLE中も保持し続けている」
        場合 (=state machine の設計原則「非STABLE中は前回STABLE盤面を
        凍結する」) は、その凍結中の呼び出しは実質的に「連鎖中の盤面へ
        is_dead() を呼ぶ」ことと等価になり、本契約に違反する。
        ぷよ生死のuser伝授「盤面が高い ≠ 窒息」(memory
        `reference_full_board_is_not_death_2026-08-22`) の通り、天井到達の
        設置が同時に連鎖の発火トリガーになった場合、連鎖解決前の一瞬だけ
        STABLE 判定される盤面 (=連鎖が解決すれば助かる) に対して呼ぶと、
        誤って True (死亡) と判定される (実測: CHAIN を含む区間で凍結する
        事故が試合時間の約8%・569秒、2026-08-24 是正、
        `docs/KNOWN_WEAKNESSES.md` 参照)。
        呼出元 (`scripts/visualize_advantage_overlay.py` の
        `enable_stable_confirmed_is_dead` 等) 側で「非STABLE中の凍結呼び出し
        を除外/遡及訂正する」ガードを設けること。

        ゲーム本来の判定タイミング (次ツモ・連鎖解決後) と整合する。

        Returns:
            bool: 窒息状態ならTrue。
        """
        c = int(self._grid[DEATH_ROW, DEATH_COL])
        return c != COLOR_EMPTY and c != COLOR_UNKNOWN

    def is_known(self, row: int, col: int) -> bool:
        """指定セルが確定状態か (UNKNOWN でないか) を返す。"""
        self._validate_pos(row, col)
        return int(self._grid[row, col]) != COLOR_UNKNOWN

    def has_any_unknown(self) -> bool:
        """盤面内に UNKNOWN セルがあれば True。"""
        return bool(np.any(self._grid == COLOR_UNKNOWN))

    def count_puyos(self) -> int:
        """
        盤面上の全ぷよ数（おじゃま含む、空・UNKNOWN除く）を返す。

        Returns:
            int: ぷよの総数。
        """
        grid = self._grid
        return int(((grid != COLOR_EMPTY) & (grid != COLOR_UNKNOWN)).sum())

    # ============================
    # コピー・シリアライズ
    # ============================

    def copy(self) -> "Board":
        """
        盤面の深コピーを返す。

        Returns:
            Board: コピーされた盤面オブジェクト。
        """
        new_board = Board()
        new_board._grid = self._grid.copy()
        return new_board

    def to_dict(self) -> dict[str, Any]:
        """
        盤面をJSON保存可能な辞書に変換する。

        Returns:
            dict: グリッドデータを含む辞書。
        """
        return {
            "grid": self._grid.tolist(),
            "rows": BOARD_ROWS,
            "cols": BOARD_COLS,
        }

    def to_json(self) -> str:
        """
        盤面をJSON文字列に変換する。

        Returns:
            str: JSON文字列。
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)

    # ============================
    # 表示
    # ============================

    def display(self) -> None:
        """色付きターミナルで盤面を表示する。"""
        print(f"  {'--' * BOARD_COLS}")
        for row in range(BOARD_ROWS):
            label = "H|" if row == DEATH_ROW else f"{row:2d}|"
            cells = "".join(
                _ANSI_COLORS.get(int(self._grid[row, col]), "??")
                for col in range(BOARD_COLS)
            )
            print(f"{label}{cells}|")
        print(f"  {'--' * BOARD_COLS}")
        col_heights = [self.height_of(c) for c in range(BOARD_COLS)]
        print(f"  高さ: {col_heights}")

    # ============================
    # 内部ユーティリティ
    # ============================

    def _validate_pos(self, row: int, col: int) -> None:
        """行・列インデックスの範囲チェック。"""
        if not (0 <= row < BOARD_ROWS):
            raise ValueError(f"行インデックスが範囲外: {row}")
        if not (0 <= col < BOARD_COLS):
            raise ValueError(f"列インデックスが範囲外: {col}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return NotImplemented
        return bool(np.array_equal(self._grid, other._grid))

    def grid_bytes(self) -> bytes:
        """盤面内容のハッシュ可能なバイト表現を返す (キャッシュキー用途)。

        内部 numpy 配列を直接晒さず、同一盤面内容なら常に同一バイト列を
        返すことだけを保証する (呼び出し側で盤面同値判定・辞書キー等に使う、
        例: scripts/compute_exchange_delta_winprob.py の重い指標計算キャッシュ)。
        """
        return self._grid.tobytes()

    def __repr__(self) -> str:
        return f"Board(puyos={self.count_puyos()}, dead={self.is_dead()})"
