"""ぷよぷよ既知の戦法テンプレート (GTR / LLR / 階段 / 座布団) との一致度評価.

I-J (B-1) 実装。上級プレイヤーの定石パターンとの近さを 0〜1 で返す。
key_flexibility 二相性 (雑然 vs 柔軟保留) で、後者が定石組み中であることを
分離するための指標。

設計:
    - 各テンプレートは「セル相対位置 + 色等価クラス」で記述
    - 色等価クラス: 'A', 'B', 'C', ... 同一クラスは同色、異クラスは異色
    - 盤面上で最も合致する色割り当てを探し、一致セル比率を返す
    - 1P/2P 対応: テンプレート単体は左下基準。2P では水平ミラー (mirror=True で評価)

引数の Board は破壊しない。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
    Board,
)

# テンプレート不在時の fallback score
FORM_TEMPLATE_NEUTRAL_SCORE: float = 0.0
# テンプレートが盤外に出た場合の score (現状は 0 = 完全不一致)
FORM_TEMPLATE_OUT_OF_BOUNDS_SCORE: float = 0.0


@dataclass(frozen=True)
class FormTemplate:
    """戦法テンプレートの定義.

    Attributes:
        name: テンプレ名 (gtr / llr / staircase / zabuton)
        cells: 相対位置 → 色等価クラス. (row_from_bottom, col_from_left)
            row_from_bottom=0 が最下段、増えるほど上。
        anchor_col: テンプレ全体の col 開始位置 (盤上絶対 col).
            1P 用 = 0 (左端), 2P 用 = BOARD_COLS - max_col_offset - 1.
    """
    name: str
    cells: tuple[tuple[tuple[int, int], str], ...]
    # 注: 上の cells は dict ではなく tuple で immutable に


def _mirror_template_cells(
    cells: tuple[tuple[tuple[int, int], str], ...],
) -> tuple[tuple[tuple[int, int], str], ...]:
    """テンプレを水平方向にミラーする (2P 用).

    元の col 0..max_col を max_col-col に反転。
    """
    if not cells:
        return cells
    max_col = max(c for ((_, c), _) in cells)
    return tuple(
        (((r, max_col - c), cls) for ((r, c), cls) in cells)
    )


def template_score(
    board: Board, template: FormTemplate, mirror: bool = False,
) -> tuple[float, dict]:
    """テンプレートとの一致度を 0〜1 で返す.

    Args:
        board: 評価対象の盤面.
        template: テンプレート定義.
        mirror: True で水平ミラー (2P 側評価用).

    Returns:
        (score, detail). score は 0..1, detail は debug 情報.

    アルゴリズム:
        1. テンプレ各セルの絶対位置を計算 (row_from_bottom → 絶対 row).
        2. 等価クラス毎に観測色の最頻値を「割当色」として採用.
        3. 異クラスが同じ最頻色を共有する場合はその後発クラスは「不一致」.
        4. 各クラスのセルで割当色と一致するセル数 / 総セル数 = score.
    """
    cells = template.cells
    if mirror:
        cells = _mirror_template_cells(cells)
    # 絶対位置に変換 (row_from_bottom → 絶対 row)
    abs_cells_by_class: dict[str, list[tuple[int, int]]] = {}
    for (r_bot, c), cls in cells:
        abs_r = BOARD_ROWS - 1 - r_bot
        abs_c = c
        if not (0 <= abs_r < BOARD_ROWS and 0 <= abs_c < BOARD_COLS):
            return FORM_TEMPLATE_OUT_OF_BOUNDS_SCORE, {
                "reason": "out_of_bounds", "cell": (abs_r, abs_c),
            }
        abs_cells_by_class.setdefault(cls, []).append((abs_r, abs_c))

    total = 0
    matched = 0
    assigned_colors: dict[str, int] = {}
    used_colors: set[int] = set()
    for cls, cells_list in abs_cells_by_class.items():
        colors = [int(board.get(r, c)) for r, c in cells_list]
        non_empty = [
            col for col in colors
            if col not in (COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA)
        ]
        total += len(cells_list)
        if not non_empty:
            continue
        # 最頻色 (同数なら入力順=色番号小さい順)
        counter = Counter(non_empty)
        top_color, _top_count = counter.most_common(1)[0]
        if top_color in used_colors:
            # 異クラスが同じ色を共有 → このクラスは不一致扱い
            continue
        assigned_colors[cls] = top_color
        used_colors.add(top_color)
        # このクラスのセルで top_color と一致する数を加算
        matched += sum(1 for c in colors if c == top_color)
    score = matched / total if total > 0 else 0.0
    return score, {
        "matched": matched, "total": total,
        "assigned": assigned_colors,
    }


def best_template_score(
    board: Board, template: FormTemplate,
) -> tuple[float, bool]:
    """1P / 2P (mirror) の両方を試して最良の score を返す.

    Returns:
        (best_score, mirror_used)
    """
    s1, _ = template_score(board, template, mirror=False)
    s2, _ = template_score(board, template, mirror=True)
    if s2 > s1:
        return s2, True
    return s1, False


# ============================
# テンプレート定義
# ============================
# 表記: (row_from_bottom, col_from_left) -> equiv_class
# row 0 = 最下段, col 0 = 左端 (1P 側)。

# GTR (じーてぃーあーる): 左下 3-4 段の鍵型。3 色構成 (A/B/C)。
# 実形:
#   col 0, row 0..2: A, A, B (col 0 下から AAB)
#   col 1, row 0..2: A, B, C (col 1 下から ABC)
#   col 2, row 0..1: B, C    (col 2 下から BC)
#   col 3, row 0:    B       (col 3 下から B)
# (注: 簡略化、本来は GTR ヘッド 4 puyo + 折り返し)
GTR_TEMPLATE: FormTemplate = FormTemplate(
    name="gtr",
    cells=(
        ((0, 0), "A"), ((1, 0), "A"), ((2, 0), "B"),
        ((0, 1), "A"), ((1, 1), "B"), ((2, 1), "C"),
        ((0, 2), "B"), ((1, 2), "C"),
        ((0, 3), "B"),
    ),
)

# LLR (えるえるあーる): GTR の階段変形。3 色構成。
# 実形 (簡略):
#   col 0, row 0..1: A, A
#   col 1, row 0..2: B, A, C
#   col 2, row 0..1: B, C
#   col 3, row 0:    B
LLR_TEMPLATE: FormTemplate = FormTemplate(
    name="llr",
    cells=(
        ((0, 0), "A"), ((1, 0), "A"),
        ((0, 1), "B"), ((1, 1), "A"), ((2, 1), "C"),
        ((0, 2), "B"), ((1, 2), "C"),
        ((0, 3), "B"),
    ),
)

# 階段 (Staircase): 左から右に階段状に上がる古典形。
# 各列で 2 個ずつ上がる連鎖形。
#   col 0, row 0..1: A, B
#   col 1, row 0..1: A, B
#   col 2, row 0..1: C, D
#   col 3, row 0..1: C, D
# 4 色 (A,B,C,D)
STAIRCASE_TEMPLATE: FormTemplate = FormTemplate(
    name="staircase",
    cells=(
        ((0, 0), "A"), ((1, 0), "B"),
        ((0, 1), "A"), ((1, 1), "B"),
        ((0, 2), "C"), ((1, 2), "D"),
        ((0, 3), "C"), ((1, 3), "D"),
    ),
)

# 座布団 (Zabuton): 中央 col1-3, row 0-2 の 9-12 puyo 塊形.
# 簡略: 中央 4 列 × 下層 3 段で 4 色 (折り返し可能形)。
ZABUTON_TEMPLATE: FormTemplate = FormTemplate(
    name="zabuton",
    cells=(
        ((0, 1), "A"), ((1, 1), "B"), ((2, 1), "A"),
        ((0, 2), "B"), ((1, 2), "A"), ((2, 2), "B"),
        ((0, 3), "C"), ((1, 3), "D"), ((2, 3), "C"),
        ((0, 4), "D"), ((1, 4), "C"), ((2, 4), "D"),
    ),
)

# Sullen GTR (フキゲン GTR): GTR の頭部を不機嫌な配置にした派生 (2026-05-09 追加).
# citrus610/ama (Puyo Puyo Tsu AI) form.h の SGTR pattern を参照しつつ、
# 本プロジェクト互換の簡略形 (頭 + 折り返し下部) に翻訳。
# 特徴: col 1 (左から 2 列目) に同等価クラス B が縦 2 連で立つ点が GTR と異なる.
#   col 0, row 0..2: A, A, B   (col 0 下から AAB)
#   col 1, row 0..2: A, B, B   (col 1 下から ABB ← GTR は ABC、ここで分岐)
#   col 2, row 0..1: B, C      (col 2 下から BC)
#   col 3, row 0:    C         (col 3 下から C)
# 等価クラス 3 種 (A/B/C)、合計 9 cell。1P/2P mirror あり、色順序対称。
SULLEN_GTR_TEMPLATE: FormTemplate = FormTemplate(
    name="sullen_gtr",
    cells=(
        ((0, 0), "A"), ((1, 0), "A"), ((2, 0), "B"),
        ((0, 1), "A"), ((1, 1), "B"), ((2, 1), "B"),
        ((0, 2), "B"), ((1, 2), "C"),
        ((0, 3), "C"),
    ),
)

# Fron (フロン積み): プレイヤー Fron に由来する LLR 派生形 (2026-05-09 追加).
# citrus610/ama form.h の FRON pattern を参照しつつ、本プロジェクト互換の簡略形.
# 特徴: col 2 (左から 3 列目) の row 1 に B、row 2 に B (B 同色サンドではなく
# B/B 縦連) で、SGTR の col 1 縦連と対称な右寄り構造。
#   col 0, row 0..2: A, A, B   (col 0 下から AAB)
#   col 1, row 0..2: A, B, B   (col 1 下から ABB)
#   col 2, row 0..2: B, C, B   (col 2 下から BCB ← Fron 特徴: 中央 C 挟み)
#   col 3, row 0..1: C, D      (col 3 下から CD)
# 等価クラス 4 種 (A/B/C/D)、合計 11 cell。1P/2P mirror あり、色順序対称。
FRON_TEMPLATE: FormTemplate = FormTemplate(
    name="fron",
    cells=(
        ((0, 0), "A"), ((1, 0), "A"), ((2, 0), "B"),
        ((0, 1), "A"), ((1, 1), "B"), ((2, 1), "B"),
        ((0, 2), "B"), ((1, 2), "C"), ((2, 2), "B"),
        ((0, 3), "C"), ((1, 3), "D"),
    ),
)


ALL_FORM_TEMPLATES: tuple[FormTemplate, ...] = (
    GTR_TEMPLATE, LLR_TEMPLATE, STAIRCASE_TEMPLATE, ZABUTON_TEMPLATE,
    # 2026-05-09 追加 (B-1.b)
    SULLEN_GTR_TEMPLATE, FRON_TEMPLATE,
)


def all_template_scores(board: Board) -> dict[str, float]:
    """全テンプレートの best score (1P/2P いずれかの最良) を返す."""
    return {
        t.name: best_template_score(board, t)[0]
        for t in ALL_FORM_TEMPLATES
    }
