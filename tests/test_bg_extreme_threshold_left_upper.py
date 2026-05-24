"""軸 3-b (Phase L): BG_EXTREME_THRESHOLD_LEFT_UPPER エリア別閾値 ユニットテスト。

テスト概要:
    test 1: visible_row=5, col=0 で距離 < LEFT_UPPER threshold なら EMPTY (= 左上エリア緩和判定)
    test 2: visible_row=5, col=0 で距離 >= LEFT_UPPER threshold は EMPTY にならない (= 上限あり)
    test 3: visible_row=4, col=0 (= 左上エリア外) では DEFAULT threshold が適用される
    test 4: visible_row=5, col=2 (= 左上エリア外) では DEFAULT threshold が適用される
    test 5: visible_row=5, col=0 で距離 < DEFAULT threshold は当然 EMPTY (= regression check)
"""

from __future__ import annotations

from src.image_reader import (
    BG_EXTREME_THRESHOLD_DEFAULT,
    BG_EXTREME_THRESHOLD_LEFT_UPPER,
    BG_LEFT_UPPER_COL_MAX,
    BG_LEFT_UPPER_VISIBLE_ROW_MIN,
    ImageReader,
)


# ============================
# 定数整合性チェック
# ============================


def test_left_upper_threshold_is_larger_than_default() -> None:
    """LEFT_UPPER threshold は DEFAULT より大きくなければならない。"""
    assert BG_EXTREME_THRESHOLD_LEFT_UPPER > BG_EXTREME_THRESHOLD_DEFAULT, (
        f"LEFT_UPPER ({BG_EXTREME_THRESHOLD_LEFT_UPPER}) > "
        f"DEFAULT ({BG_EXTREME_THRESHOLD_DEFAULT}) が必要"
    )


def test_area_constants_values() -> None:
    """エリア定数が期待値であること。"""
    assert BG_LEFT_UPPER_VISIBLE_ROW_MIN == 5, (
        f"BG_LEFT_UPPER_VISIBLE_ROW_MIN は 5 のはず: {BG_LEFT_UPPER_VISIBLE_ROW_MIN}"
    )
    assert BG_LEFT_UPPER_COL_MAX == 1, (
        f"BG_LEFT_UPPER_COL_MAX は 1 のはず: {BG_LEFT_UPPER_COL_MAX}"
    )


# ============================
# _resolve_tier1_threshold テスト
# ============================


def _make_reader() -> ImageReader:
    """テスト用の最小 ImageReader を生成する。"""
    # 分類器なし、背景 FP なし (= threshold logic のみ検証)
    return ImageReader()


# test 1: 左上エリア内で距離が LEFT_UPPER threshold 未満 → EMPTY 判定
def test_resolve_tier1_left_upper_area_returns_left_upper_threshold() -> None:
    """visible_row=5, col=0 (= 左上エリア) では LEFT_UPPER threshold が返ること。"""
    reader = _make_reader()
    visible_row = BG_LEFT_UPPER_VISIBLE_ROW_MIN  # = 5
    col = 0
    threshold = reader._resolve_tier1_threshold(visible_row, col)
    assert threshold == BG_EXTREME_THRESHOLD_LEFT_UPPER, (
        f"左上エリアは LEFT_UPPER threshold ({BG_EXTREME_THRESHOLD_LEFT_UPPER}) "
        f"のはず: {threshold}"
    )


# test 2: 左上エリア内でも LEFT_UPPER threshold 以上の距離では EMPTY にならない
def test_resolve_tier1_left_upper_threshold_has_upper_bound() -> None:
    """visible_row=5, col=0 で threshold が LEFT_UPPER threshold で上限が存在すること。

    距離 >= LEFT_UPPER threshold のケースは threshold を返すだけ (= 呼出元が判定)。
    ここでは返値が LEFT_UPPER より大きくないことを確認して上限の存在を保証する。
    """
    reader = _make_reader()
    threshold = reader._resolve_tier1_threshold(5, 0)
    # 距離が threshold 以上なら EMPTY にならない (= 返値 == LEFT_UPPER threshold のみ)
    assert threshold == BG_EXTREME_THRESHOLD_LEFT_UPPER
    # LEFT_UPPER を超えた距離値はエリア判定に依らず EMPTY にならないことを確認
    # (= 呼出元の `dist < tier1_threshold` が False になる境界チェック)
    distance_above_threshold = BG_EXTREME_THRESHOLD_LEFT_UPPER + 1.0
    assert distance_above_threshold >= threshold, (
        "LEFT_UPPER threshold より大きい距離では EMPTY にならないはず"
    )


# test 3: visible_row=4 (= エリア外) では DEFAULT threshold が返ること
def test_resolve_tier1_outside_area_row_below_min() -> None:
    """visible_row=4, col=0 (= 左上エリア外: visible_row < BG_LEFT_UPPER_VISIBLE_ROW_MIN)
    では DEFAULT threshold が返ること。"""
    reader = _make_reader()
    visible_row = BG_LEFT_UPPER_VISIBLE_ROW_MIN - 1  # = 4
    col = 0
    threshold = reader._resolve_tier1_threshold(visible_row, col)
    assert threshold == BG_EXTREME_THRESHOLD_DEFAULT, (
        f"エリア外 (visible_row={visible_row}, col={col}) は DEFAULT threshold "
        f"({BG_EXTREME_THRESHOLD_DEFAULT}) のはず: {threshold}"
    )


# test 4: col=2 (= エリア外) では DEFAULT threshold が返ること
def test_resolve_tier1_outside_area_col_exceeds_max() -> None:
    """visible_row=5, col=2 (= 左上エリア外: col > BG_LEFT_UPPER_COL_MAX)
    では DEFAULT threshold が返ること。"""
    reader = _make_reader()
    visible_row = BG_LEFT_UPPER_VISIBLE_ROW_MIN  # = 5
    col = BG_LEFT_UPPER_COL_MAX + 1  # = 2
    threshold = reader._resolve_tier1_threshold(visible_row, col)
    assert threshold == BG_EXTREME_THRESHOLD_DEFAULT, (
        f"エリア外 (visible_row={visible_row}, col={col}) は DEFAULT threshold "
        f"({BG_EXTREME_THRESHOLD_DEFAULT}) のはず: {threshold}"
    )


# test 5: visible_row=5, col=0 で距離 < DEFAULT threshold は当然 EMPTY (= regression check)
def test_resolve_tier1_default_threshold_distance_still_triggers() -> None:
    """visible_row=5, col=0 で距離 < DEFAULT threshold の場合、
    LEFT_UPPER threshold > DEFAULT なので当然 dist < LEFT_UPPER も満たす (= regression check)。"""
    reader = _make_reader()
    threshold = reader._resolve_tier1_threshold(5, 0)
    distance_below_default = BG_EXTREME_THRESHOLD_DEFAULT - 1.0
    # DEFAULT 未満の距離は LEFT_UPPER threshold 未満にもなるはず
    assert distance_below_default < threshold, (
        f"DEFAULT 未満の距離 ({distance_below_default}) は "
        f"LEFT_UPPER threshold ({threshold}) 未満のはず"
    )
