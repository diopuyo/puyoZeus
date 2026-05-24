"""V3.1 TelopDetector.cells_covered_by_bbox + ImageReader use_telop_mask テスト。"""
from __future__ import annotations

import numpy as np
import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_RED,
    COLOR_UNKNOWN,
    HIDDEN_ROWS,
)
from src.image_reader import (
    DEFAULT_P1_REGION,
    DEFAULT_P2_REGION,
    BoardRegion,
    ImageReader,
)
from src.telop_detector import TelopDetector


# --- TelopDetector.cells_covered_by_bbox 単体テスト ---

def test_cells_covered_by_bbox_no_overlap() -> None:
    """bbox が region と完全に分離 → 被覆 0 セル。"""
    region = DEFAULT_P1_REGION  # x=282..666
    bbox = (1500, 100, 200, 200)  # 右側の領域、被覆なし
    covered = TelopDetector.cells_covered_by_bbox(bbox, region)
    assert len(covered) == 0


def test_cells_covered_by_bbox_full_overlap() -> None:
    """bbox が region 全体を覆う → 全可視セル被覆 (12 行 × 6 列 = 72)。"""
    region = DEFAULT_P1_REGION
    # region 全体を含む大きな bbox
    bbox = (
        region.x - 100, region.y - 100,
        region.width + 200, region.height + 200,
    )
    covered = TelopDetector.cells_covered_by_bbox(bbox, region)
    assert len(covered) == BOARD_COLS * (BOARD_ROWS - HIDDEN_ROWS)


def test_cells_covered_by_bbox_partial_overlap() -> None:
    """bbox が region 右半分だけ覆う → 右側セルのみ被覆。"""
    region = DEFAULT_P1_REGION
    half_x = region.x + region.width // 2
    bbox = (half_x, region.y, region.width, region.height)
    covered = TelopDetector.cells_covered_by_bbox(bbox, region)
    # 少なくとも右半分の列 (col >= 3) の行が含まれる
    cols_covered = {c for _, c in covered}
    assert 3 in cols_covered or 4 in cols_covered or 5 in cols_covered
    assert 0 not in cols_covered  # 最左列は被覆しない


def test_cells_covered_by_bbox_excludes_hidden_row() -> None:
    """bbox が画面上端含めても隠し段 (row 0) は被覆対象外。"""
    region = DEFAULT_P1_REGION
    bbox = (
        region.x - 100, 0,  # y=0 から
        region.width + 200, region.y + region.height + 200,
    )
    covered = TelopDetector.cells_covered_by_bbox(bbox, region)
    rows_covered = {r for r, _ in covered}
    # row 0 は HIDDEN_ROWS 未満で対象外
    assert 0 not in rows_covered


def test_cells_covered_by_bbox_p2_region() -> None:
    """2P region でも動作する (bbox が中央テロップでも 2P の左端を覆う場合)。"""
    region = DEFAULT_P2_REGION  # x=1258..1642
    bbox = (1200, 400, 100, 100)  # 左端を少しかすめる
    covered = TelopDetector.cells_covered_by_bbox(bbox, region)
    # 1258 - 1200 = 58、bbox は 1200-1300 → 1258-1300 の 42px 重複
    # P2 region 左端 (col 0) の一部行を被覆するはず
    cols_covered = {c for _, c in covered}
    assert 0 in cols_covered


# --- ImageReader 統合テスト ---

def test_image_reader_use_telop_mask_default_off() -> None:
    """既定では telop_mask off。"""
    reader = ImageReader()
    assert reader._telop_detector is None


def test_image_reader_use_telop_mask_on() -> None:
    """use_telop_mask=True で TelopDetector がロードされる。"""
    reader = ImageReader(use_telop_mask=True)
    assert reader._telop_detector is not None


def test_image_reader_telop_mask_no_detection_no_change() -> None:
    """テロップが検出されないフレームでは挙動変わらず。

    黒一色フレーム → テロップ検出されない → telop_mask は何もしない。
    """
    reader_off = ImageReader(use_telop_mask=False)
    reader_on = ImageReader(use_telop_mask=True)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    b1_off, b2_off = reader_off.read_both_boards(frame)
    b1_on, b2_on = reader_on.read_both_boards(frame)

    # 黒フレームではテロップマッチ低 → bbox=None → 同じ結果
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            assert b1_off.get(row, col) == b1_on.get(row, col)
            assert b2_off.get(row, col) == b2_on.get(row, col)


def test_image_reader_telop_mask_unknownize_via_cache() -> None:
    """キャッシュ bbox を直接設定して被覆セルが UNKNOWN になることを確認。"""
    reader = ImageReader(use_telop_mask=True)
    # P1 region 中央あたりを覆う bbox
    region = DEFAULT_P1_REGION
    cx = region.x + region.width // 2
    cy = region.y + region.height // 2
    reader._cached_telop_bbox = (cx - 30, cy - 30, 60, 60)

    # ダミーフレーム (色は全部 RED 判定されるよう真っ赤)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:, :, 2] = 200  # B=0, G=0, R=200 → 赤
    # match_state など使わず read_board だけ呼ぶ (cached_bbox を直接利用)
    board = reader.read_board(frame, region)
    # 被覆位置のセル (中央付近) は UNKNOWN になっているはず
    covered = TelopDetector.cells_covered_by_bbox(
        reader._cached_telop_bbox, region,
    )
    assert len(covered) > 0
    for row, col in covered:
        assert board.get(row, col) == COLOR_UNKNOWN


def test_image_reader_telop_mask_no_cache_no_unknown() -> None:
    """cached_telop_bbox=None なら UNKNOWN 化されない。"""
    reader = ImageReader(use_telop_mask=True)
    reader._cached_telop_bbox = None
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:, :, 2] = 200
    board = reader.read_board(frame, DEFAULT_P1_REGION)
    # UNKNOWN は隠し段の物理推論で出る可能性はあるが、可視領域は赤一色なので UNKNOWN ではない
    # 可視領域の少なくとも 1 セルが UNKNOWN でないことを確認
    visible_unknown_count = 0
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board.get(row, col) == COLOR_UNKNOWN:
                visible_unknown_count += 1
    # 可視領域に UNKNOWN が混じらないこと (テロップ被覆ではないので)
    assert visible_unknown_count == 0


def test_telop_result_bbox_field_exists() -> None:
    """TelopResult に bbox フィールドが追加されている。"""
    from src.telop_detector import TelopResult
    r = TelopResult(is_visible=False, template_name=None, score=0.0)
    assert r.bbox is None
    r2 = TelopResult(
        is_visible=True, template_name="t", score=0.9, bbox=(10, 20, 100, 50),
    )
    assert r2.bbox == (10, 20, 100, 50)
