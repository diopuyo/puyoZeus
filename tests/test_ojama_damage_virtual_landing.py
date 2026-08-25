"""Q-04「おじゃまダメージの意味論不一致」修正のテスト (2026-08-24)。

fable architect レビュー確定の設計 (仮想着弾+着弾後の余裕評価) を検証する。
既存 `ojama_damage` の後方互換 (virtual_landing=False が既定・旧値と bit 一致)
を壊さないことと、新方式 (virtual_landing=True) が「発火点が見つからない
盤面で全6列平均に希釈され、窒息寸前の列でも無害帯に埋没する」バグ
(Q-04 本体) を修正していることの両方を確認する。

盤面ビルダーは既存 tests/test_indicators_v2.py の `_headroom_board` と同じ
方針 (非連結色 R/B/G 巡回で 1 手発火を作らせない) を踏襲する。
"""
from __future__ import annotations

import pytest

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_GREEN, COLOR_RED, Board
import src.indicators_v2 as iv


# ============================
# 盤面ビルダー
# ============================


def _empty_grid() -> list[list[int]]:
    return [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]


def _empty_board() -> Board:
    return Board.from_list(_empty_grid())


def _headroom_board(height: int) -> Board:
    """**全6列**の高さが height になるよう非連結色 (R/B/G 巡回) で積む。

    tests/test_indicators_v2.py の同名ヘルパーと同じ設計意図 (1 手追加でも
    4 連結が完成せず、発火点が決まらない盤面を安定して作る)。
    """
    g = _empty_grid()
    colors = [COLOR_RED, COLOR_BLUE, COLOR_GREEN]
    top = BOARD_ROWS - 1
    for col in range(BOARD_COLS):
        for i in range(height):
            g[top - i][col] = colors[(i + col) % 3]
    return Board.from_list(g)


def _col2_margin_board(headroom_dan: int) -> Board:
    """DEATH_COL(=2 列目)だけを積み、他5列は空にした盤面。

    headroom_dan = MAX_COL_HEIGHT(12) - height_of(col2) となるよう
    col2 の高さを (12 - headroom_dan) にする (headroom_dan>=1 前提、
    =0 だと既に窒息済みになるため呼び出し側で保証すること)。
    2色交互 (R/B) で積み、同色4連結を作らず _takapt_best_drop を
    撹乱しない (tests/test_expected_net_damage_step5.py と同方針)。
    """
    g = _empty_grid()
    colors = [COLOR_RED, COLOR_BLUE]
    height = 12 - headroom_dan
    top = BOARD_ROWS - 1
    for i in range(height):
        g[top - i][2] = colors[i % 2]
    return Board.from_list(g)


# ============================
# テスト2: 後方互換 (virtual_landing=False は既定・旧値と bit 一致)
# ============================


def test_virtual_landing_default_is_false_and_matches_legacy() -> None:
    """virtual_landing 省略時 (=False) は、明示的に False を渡した場合と

    完全に同じ結果になる (optional 引数追加のみ・既存呼び出し元は無改修で
    旧挙動を維持する、CLAUDE.md backwards compat 規約)。
    """
    boards = (_empty_board(), _headroom_board(3), _headroom_board(9), _col2_margin_board(1))
    for board in boards:
        for count in (0, 12, 18, 48, 60):
            default_result = iv.ojama_damage(board, count)
            explicit_false_result = iv.ojama_damage(board, count, virtual_landing=False)
            assert default_result.score == explicit_false_result.score
            assert default_result.raw == explicit_false_result.raw


# ============================
# テスト3: 同値化の再発防止ゲート (Q-04 本体の直接再現テスト)
# ============================


def test_virtual_landing_col2_margin_board_diverges_from_empty_board() -> None:
    """count=48 固定で、col2 余裕1段の盤面と空盤面の damage 差が

    0.5 以上あること (Q-04 バグの直接再現: 旧実装は両方とも 0.05 に
    同値化していた)。
    """
    count = 48
    empty_damage = iv.ojama_damage(_empty_board(), count, virtual_landing=True).score
    col2_damage = iv.ojama_damage(_col2_margin_board(1), count, virtual_landing=True).score
    assert col2_damage - empty_damage >= 0.5
    # 空盤面は「受けても無害」帯のまま (各列高さ8にしかならず窒息しない)。
    assert empty_damage == pytest.approx(iv.OJAMA_DAMAGE_FLOOR, abs=1e-6)
    # col2 が溢れて窒息するため最大ダメージ。
    assert col2_damage == pytest.approx(iv.OJAMA_DAMAGE_CEIL, abs=1e-6)


# ============================
# テスト4: 決定性 (端数ありでも乱数を使わない)
# ============================


def test_virtual_landing_deterministic_with_remainder() -> None:
    """端数あり (6 で割り切れない count) を含め、同一入力を2回評価しても

    完全に同値になること (端数は連続量で控除する決定的近似のため、seed 不要)。
    """
    board = _headroom_board(6)
    for count in (11, 17, 50, 73):
        first = iv.ojama_damage(board, count, virtual_landing=True)
        second = iv.ojama_damage(board, count, virtual_landing=True)
        assert first.score == second.score
        assert first.raw == second.raw


# ============================
# テスト5: 単調性 (count に対して非減少、層別3盤面)
# ============================


def test_virtual_landing_monotonic_in_count_stratified() -> None:
    """空 / 中 / 圧迫 の3層盤面それぞれで、おじゃま量が増えるほど

    score が単調非減少であること (feedback_stratify_before_pooling_
    2026-07-29: 代表値でなく層別で確認する)。
    """
    counts = (0, 3, 11, 12, 15, 17, 18, 30, 48, 60, 100)
    for board in (_empty_board(), _headroom_board(6), _headroom_board(10)):
        scores = [iv.ojama_damage(board, c, virtual_landing=True).score for c in counts]
        for prev, cur in zip(scores, scores[1:]):
            assert cur >= prev - 1e-9, (board, counts, scores)


# ============================
# テスト6: 単調性 (盤面の埋まり具合に対して非減少、同一 count)
# ============================


def test_virtual_landing_monotonic_in_board_fullness() -> None:
    """同じ count で、盤面が埋まっている(=headroom小さい)ほど score が

    非減少であること (逆転しないことを固定)。
    """
    heights_desc_headroom = (0, 3, 6, 9, 11)  # headroom = 12,9,6,3,1
    for count in (0, 12, 18, 40):
        scores = [
            iv.ojama_damage(_headroom_board(h), count, virtual_landing=True).score
            for h in heights_desc_headroom
        ]
        for prev, cur in zip(scores, scores[1:]):
            assert cur >= prev - 1e-9, (count, heights_desc_headroom, scores)


# ============================
# テスト7: 窒息物理 (col2 の余裕段数から CEIL 到達点を予測できる)
# ============================


@pytest.mark.parametrize("headroom_dan", [1, 2, 3, 4])
def test_virtual_landing_col2_margin_reaches_ceil_at_measured_threshold(
    headroom_dan: int,
) -> None:
    """col2 の余裕 headroom_dan 段の盤面に、6 列均等配分 (OJAMA_DAMAGE_PER_DAN

    =6個/段) で headroom_dan 段ぶん (=6*headroom_dan 個) 着弾させると、
    col2 が真上に積み上がって DEATH_ROW(=1) に到達し CEIL に達する。
    1段手前 (6*(headroom_dan-1) 個) ではまだ届かず CEIL 未満のままである
    ことも併せて固定する (境界の両側を固定、`Board.height_of` が隠し段
    row0 を高さに含む仕様と DEATH_ROW=1 の関係から実測で導出した境界)。
    """
    board = _col2_margin_board(headroom_dan)
    per_dan = iv.OJAMA_DAMAGE_PER_DAN
    count_reaches_ceil = int(per_dan * headroom_dan)
    result_at_ceil = iv.ojama_damage(board, count_reaches_ceil, virtual_landing=True)
    assert result_at_ceil.score == pytest.approx(iv.OJAMA_DAMAGE_CEIL, abs=1e-6)

    if headroom_dan >= 2:
        count_one_dan_short = int(per_dan * (headroom_dan - 1))
        result_before_ceil = iv.ojama_damage(board, count_one_dan_short, virtual_landing=True)
        assert result_before_ceil.score < iv.OJAMA_DAMAGE_CEIL


# ============================
# テスト8: 折れ点保存 (headroom3 の折れ線構造、virtual_landing=True 版)
# ============================


def test_virtual_landing_breakpoint_structure_preserved_headroom3() -> None:
    """既存 test_ojama_damage_breakpoints_headroom3 相当の折れ点構造

    (12個までは平坦・12〜18で急激に立ち上がる・大量でCEIL) が
    virtual_landing=True でも保存されていること。
    仮想着弾は物理的な is_dead() 判定を経由するため、旧方式 (引き算近似)
    と厳密に同じカーブにはならないが、「12個まで平坦→立ち上がる→
    大量でCEIL」という定性構造は保存される (根拠: 6列均等配分により
    12個/18個は全列の headroom_dan=3 の盤面でちょうど 2段/3段に相当し、
    3段で DEATH_ROW に到達する物理と一致する)。
    """
    board = _headroom_board(9)  # 全列 headroom_dan=3
    s0 = iv.ojama_damage(board, 0, virtual_landing=True).score
    s11 = iv.ojama_damage(board, 11, virtual_landing=True).score
    s12 = iv.ojama_damage(board, 12, virtual_landing=True).score
    s17 = iv.ojama_damage(board, 17, virtual_landing=True).score
    s18 = iv.ojama_damage(board, 18, virtual_landing=True).score
    s60 = iv.ojama_damage(board, 60, virtual_landing=True).score
    assert s11 == pytest.approx(s0, abs=1e-6)
    assert s12 == pytest.approx(s0, abs=1e-6)
    assert s17 > s12 + 0.1
    assert s18 > s17
    assert s60 == pytest.approx(iv.OJAMA_DAMAGE_CEIL, abs=1e-6)


# ============================
# テスト9: 非破壊 (stateless 原則)
# ============================


def test_virtual_landing_does_not_mutate_input_board() -> None:
    """virtual_landing=True でも、呼出前後で入力 board が変化しないこと

    (drop_ojama は内部で copy() するため非破壊のはずだが、仮想着弾の
    追加ロジックが誤って直接 board を書き換えないことを固定する)。
    """
    board = _headroom_board(9)
    before = board.copy()
    iv.ojama_damage(board, 48, virtual_landing=True)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            assert board.get(row, col) == before.get(row, col)
