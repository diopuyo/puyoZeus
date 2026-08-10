"""隠し段 (13段目 = row 0) の色可能性を確率として構成する (2026-08-08).

設計: docs/HIDDEN_ROW_PROBABILISTIC_DESIGN_2026-08-08.md

## なぜ必要か
現行は隠し段の読めないセルを `COLOR_UNKNOWN` という**単一の値**で表しており、
「赤かもしれない」「おじゃまかもしれない」という情報を持てない。 連鎖
シミュレーションは UNKNOWN を連結対象から外すため (src/chain.py:238)、
**連鎖数は常に 1 通りしか出ない**。 隠し段に何が入っているかで実際の連鎖数は
変わるので、 可能性を潰さずに保持する。

## 確率は憶測で埋めない
確率は **すでに分かっている物理・会計情報から** 構成する。 優先度の高い情報源で
確定したら、 そこで打ち止めにして下位の推測で上書きしない。 どの情報源で決まった
かを `HiddenCellSource` として保持し、 後から検証できるようにする。

  A. 重力による確定  — 可視最上段が空なら隠し段も空 (確定)
  B. おじゃま着弾会計 — 均等配分分は確定、 端数分は確率      (段階2 で追加)
  C. 画面外へ行ったツモ — そのツモの 2 色に限定              (段階3 で追加)
  D. 情報なし         — 試合の使用色 + おじゃまの一様分布

本モジュールは stateless な純関数群であり、 認識パイプラインに依存しない
(観測指標は stateless 実装という規約に従う)。 会計値やツモ色は引数で受け取る。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    Board,
)
from src.probabilistic_board import ProbabilisticCell

# 隠し段の行数 (row 0 のみ = 13段目)。可視段は row 1..12。
HIDDEN_ROWS: int = 1
# 可視最上段の行 (窒息判定と同じ DEATH_ROW=1)。
TOP_VISIBLE_ROW: int = HIDDEN_ROWS
# 試合で実際に使われる色数 (ぷよぷよeスポーツは 5 色中 1 色をランダム除外)。
COLORS_PER_MATCH: int = 4


class HiddenCellSource(str, Enum):
    """そのセルの確率がどの情報源で決まったか (検証用に必ず保持する)。"""

    GRAVITY_EMPTY = "gravity_empty"      # A: 重力より空で確定
    OJAMA_CERTAIN = "ojama_certain"      # B: おじゃま均等配分分 (確定)
    OJAMA_REMAINDER = "ojama_remainder"  # B: おじゃま端数分 (確率)
    TSUMO_COLORS = "tsumo_colors"        # C: 画面外へ行ったツモの 2 色
    UNINFORMED = "uninformed"            # D: 情報なし (一様分布)


@dataclass(frozen=True)
class HiddenCellProbability:
    """隠し段 1 セルの確率分布と、その根拠。"""

    col: int
    cell: ProbabilisticCell
    source: HiddenCellSource

    @property
    def is_certain(self) -> bool:
        """単一色にほぼ確定しているか (展開不要なセルの判定に使う)。"""
        return self.cell.is_certain()


@dataclass(frozen=True)
class HiddenRowProbabilities:
    """隠し段 1 行分 (6 列) の確率分布。

    `confirmed_board` (単一値の Board) と併走して持つための器。
    既存コードは本オブジェクトを参照しなければ従来通り動作する。
    """

    cells: tuple[HiddenCellProbability, ...] = field(default_factory=tuple)

    def get(self, col: int) -> HiddenCellProbability | None:
        """指定列のセルを返す (無ければ None)。"""
        for c in self.cells:
            if c.col == col:
                return c
        return None

    @property
    def uncertain_cols(self) -> tuple[int, ...]:
        """確定していない列の一覧 (組み合わせ展開の対象)。"""
        return tuple(c.col for c in self.cells if not c.is_certain)


def infer_match_colors(board: Board) -> tuple[int, ...]:
    """盤面に現れている色ぷよの集合を返す (おじゃま・空は含まない).

    試合は 4 色のみ使用され 1 色はランダムに除外される
    (memory `reference_four_colors_per_match`)。 使われていない色に確率を
    置かないだけで組み合わせ数が減るため、 観測された色に限定する。
    観測が足りず 4 色に満たない場合はそのまま返す (推測で足さない)。
    """
    seen: set[int] = set()
    for row in range(TOP_VISIBLE_ROW, BOARD_ROWS):
        for col in range(BOARD_COLS):
            color = board.get(row, col)
            if color not in (COLOR_EMPTY, COLOR_OJAMA) and color <= 5:
                seen.add(int(color))
    return tuple(sorted(seen))


def _uninformed_cell(match_colors: tuple[int, ...]) -> ProbabilisticCell:
    """情報源D: 使用色 + おじゃまの一様分布を返す.

    「分からない」ことを表す分布であり、 それらしい事前分布を作らない。
    使用色が未観測 (空盤面など) の場合は空で確定させる — 盤面が空なら
    隠し段も空だからで、 これは推測ではなく重力の帰結である。
    """
    if not match_colors:
        return ProbabilisticCell.certain(COLOR_EMPTY)
    candidates = tuple(match_colors) + (COLOR_OJAMA,)
    return ProbabilisticCell.uniform(candidates)


def build_hidden_row_probabilities(
    board: Board,
    match_colors: tuple[int, ...] | None = None,
) -> HiddenRowProbabilities:
    """盤面から隠し段の確率分布を構成する (段階1: 情報源 A + D)。

    Args:
        board: 可視段が読み取り済みの盤面。
        match_colors: その試合で使われている色。 None なら board から推定する。

    Returns:
        6 列分の HiddenRowProbabilities。

    注意:
        本段階では おじゃま会計 (B) とツモ色 (C) を使わないため、 情報源A で
        確定しない列はすべて一様分布 (D) になる。 これは現行の
        `COLOR_UNKNOWN` と情報量として等価であり、 **既存挙動を変えない**
        (単一値に落とせば UNKNOWN と同じ扱いになる)。
    """
    colors = infer_match_colors(board) if match_colors is None else match_colors
    cells: list[HiddenCellProbability] = []
    for col in range(BOARD_COLS):
        top_visible = board.get(TOP_VISIBLE_ROW, col)
        if top_visible == COLOR_EMPTY:
            # 情報源A: 可視最上段が空なら重力より隠し段も空 (確定)
            cells.append(HiddenCellProbability(
                col=col,
                cell=ProbabilisticCell.certain(COLOR_EMPTY),
                source=HiddenCellSource.GRAVITY_EMPTY,
            ))
            continue
        # 情報源D: 情報がないので使用色 + おじゃまの一様分布
        cells.append(HiddenCellProbability(
            col=col,
            cell=_uninformed_cell(colors),
            source=HiddenCellSource.UNINFORMED,
        ))
    return HiddenRowProbabilities(cells=tuple(cells))
