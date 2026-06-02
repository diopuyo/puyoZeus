"""
image_reader.py のテスト

合成画像（純色セルグリッド）を使って色分類・盤面読み取りをテストする。
"""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
    HIDDEN_ROWS,
    VISIBLE_ROWS,
    Board,
)
from src.image_reader import (
    CELL_SAMPLE_RATIO,
    BG_EXTREME_THRESHOLD_DEFAULT,
    BG_EXTREME_THRESHOLD_PRE_CAPTURE,
    RED_HUE_WRAP_THRESHOLD,
    RED_HUE_WRAP_CORRECTED_MAX,
    BoardRegion,
    ColorClassifier,
    HsvRange,
    ImageReader,
    DEFAULT_COLOR_RANGES,
)

# ============================
# テスト用BGR色定義 (OpenCV HSV上で確実に分類される純色)
# ============================
# BGR (Blue, Green, Red)
BGR_RED:    tuple[int, int, int] = (0,   0,   200)   # HSV H≈0
BGR_BLUE:   tuple[int, int, int] = (200, 50,  0)     # HSV H≈110
BGR_GREEN:  tuple[int, int, int] = (0,   200, 0)     # HSV H≈60
BGR_YELLOW: tuple[int, int, int] = (0,   200, 200)   # HSV H≈30
BGR_PURPLE: tuple[int, int, int] = (180, 0,  180)    # HSV H≈150
BGR_OJAMA:  tuple[int, int, int] = (200, 200, 200)   # HSV S≈0
BGR_EMPTY:  tuple[int, int, int] = (10,  10,  10)    # HSV V≈4

COLOR_BGR_MAP: dict[int, tuple[int, int, int]] = {
    COLOR_RED:    BGR_RED,
    COLOR_BLUE:   BGR_BLUE,
    COLOR_GREEN:  BGR_GREEN,
    COLOR_YELLOW: BGR_YELLOW,
    COLOR_PURPLE: BGR_PURPLE,
    COLOR_OJAMA:  BGR_OJAMA,
    COLOR_EMPTY:  BGR_EMPTY,
}


# ============================
# ヘルパー関数
# ============================

def make_synthetic_frame(
    region: BoardRegion,
    color_grid: list[list[int]],
    frame_h: int = 600,
    frame_w: int = 600,
) -> np.ndarray:
    """
    指定された色グリッドで各セルを塗りつぶした合成フレームを生成する。

    row < HIDDEN_ROWS (隠し段) の色は画面上には描画されない (画面外のため)。
    region は可視領域 (VISIBLE_ROWS=12 行) のみを表す。

    Args:
        region: 盤面領域 (可視領域のみ)。
        color_grid: BOARD_ROWS × BOARD_COLS の色コードグリッド。
        frame_h, frame_w: フレームサイズ。

    Returns:
        np.ndarray: BGR画像 (frame_h × frame_w × 3)。
    """
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    for row in range(BOARD_ROWS):
        if row < HIDDEN_ROWS:
            continue  # 隠し段は画面外
        visible_row = row - HIDDEN_ROWS
        for col in range(BOARD_COLS):
            color_code = color_grid[row][col]
            bgr = COLOR_BGR_MAP.get(color_code, BGR_EMPTY)

            x1 = int(region.x + col * region.cell_width)
            x2 = int(region.x + (col + 1) * region.cell_width)
            y1 = int(region.y + visible_row * region.cell_height)
            y2 = int(region.y + (visible_row + 1) * region.cell_height)

            frame[y1:y2, x1:x2] = bgr

    return frame


def all_same_color_grid(color: int) -> list[list[int]]:
    """全セルが同じ色のグリッドを生成する。"""
    return [[color] * BOARD_COLS for _ in range(BOARD_ROWS)]


# テスト用のコンパクトな盤面領域 (可視12行 × 50x40 セル)
TEST_REGION = BoardRegion(x=10, y=10, width=300, height=480)


# ============================
# HsvRange テスト
# ============================

class TestHsvRange:
    def test_default_s_max_is_255(self):
        rng = HsvRange(h_min=0, h_max=10)
        assert rng.s_max == 255

    def test_custom_range(self):
        rng = HsvRange(h_min=50, h_max=80, s_min=100, v_min=120)
        assert rng.h_min == 50
        assert rng.h_max == 80


# ============================
# BoardRegion テスト
# ============================

class TestBoardRegion:
    def test_cell_width(self):
        region = BoardRegion(x=0, y=0, width=120, height=260)
        assert region.cell_width == pytest.approx(20.0)

    def test_cell_height(self):
        # height/VISIBLE_ROWS (=12) で計算される
        region = BoardRegion(x=0, y=0, width=120, height=240)
        assert region.cell_height == pytest.approx(20.0)

    def test_cell_center_first_visible_row(self):
        # row=HIDDEN_ROWS (最初の可視行) の中心が領域内の最上部中央
        region = BoardRegion(x=0, y=0, width=60, height=120)
        cx, cy = region.cell_center(HIDDEN_ROWS, 0)
        assert cx == 5   # cell_w/2 = 10/2
        assert cy == 5   # cell_h/2 = 10/2

    def test_cell_sample_rect_within_cell(self):
        region = BoardRegion(x=0, y=0, width=120, height=260)
        x1, y1, x2, y2 = region.cell_sample_rect(0, 0)
        assert x1 < x2
        assert y1 < y2


# ============================
# ColorClassifier テスト
# ============================

class TestColorClassifier:
    """純色パッチが正しく分類されることを確認する。"""

    @pytest.fixture
    def clf(self) -> ColorClassifier:
        return ColorClassifier()

    def _make_patch(self, bgr: tuple[int, int, int], size: int = 20) -> np.ndarray:
        patch = np.zeros((size, size, 3), dtype=np.uint8)
        patch[:, :] = bgr
        return patch

    def test_classify_red(self, clf: ColorClassifier):
        assert clf.classify(self._make_patch(BGR_RED)) == COLOR_RED

    def test_classify_blue(self, clf: ColorClassifier):
        assert clf.classify(self._make_patch(BGR_BLUE)) == COLOR_BLUE

    def test_classify_green(self, clf: ColorClassifier):
        assert clf.classify(self._make_patch(BGR_GREEN)) == COLOR_GREEN

    def test_classify_yellow(self, clf: ColorClassifier):
        assert clf.classify(self._make_patch(BGR_YELLOW)) == COLOR_YELLOW

    def test_classify_purple(self, clf: ColorClassifier):
        assert clf.classify(self._make_patch(BGR_PURPLE)) == COLOR_PURPLE

    def test_classify_ojama(self, clf: ColorClassifier):
        assert clf.classify(self._make_patch(BGR_OJAMA)) == COLOR_OJAMA

    def test_classify_empty_dark(self, clf: ColorClassifier):
        assert clf.classify(self._make_patch(BGR_EMPTY)) == COLOR_EMPTY

    def test_classify_empty_patch(self, clf: ColorClassifier):
        """空パッチは COLOR_EMPTY を返す。"""
        assert clf.classify(np.zeros((0, 0, 3), dtype=np.uint8)) == COLOR_EMPTY

    def test_classify_hsv_shortcut(self, clf: ColorClassifier):
        """classify_hsv がclassify と同じ結果を返す。"""
        result = clf.classify_hsv(0, 200, 180)  # 赤寄り
        assert result == COLOR_RED

    def test_custom_ranges(self):
        """カスタム閾値が優先される。"""
        custom = {
            COLOR_RED: [HsvRange(h_min=50, h_max=70, s_min=100, v_min=80)]
        }
        clf = ColorClassifier(color_ranges=custom)
        # 緑の純色パッチ (H≈60) がカスタム閾値でCOLOR_REDに分類される
        patch = np.zeros((10, 10, 3), dtype=np.uint8)
        patch[:, :] = BGR_GREEN
        assert clf.classify(patch) == COLOR_RED


# ============================
# サイクル71: 投票方式 ColorClassifier テスト
# ============================

class TestColorClassifierVoteMode:
    """vote_mode=True での per-pixel 投票方式の動作確認."""

    @pytest.fixture
    def clf(self) -> ColorClassifier:
        return ColorClassifier(vote_mode=True)

    def _make_patch(
        self, bgr: tuple[int, int, int], size: int = 20,
    ) -> np.ndarray:
        patch = np.zeros((size, size, 3), dtype=np.uint8)
        patch[:, :] = bgr
        return patch

    def test_vote_classify_pure_red(self, clf: ColorClassifier):
        """純赤パッチは COLOR_RED."""
        assert clf.classify(self._make_patch(BGR_RED)) == COLOR_RED

    def test_vote_classify_pure_blue(self, clf: ColorClassifier):
        assert clf.classify(self._make_patch(BGR_BLUE)) == COLOR_BLUE

    def test_vote_classify_pure_green(self, clf: ColorClassifier):
        assert clf.classify(self._make_patch(BGR_GREEN)) == COLOR_GREEN

    def test_vote_classify_pure_yellow(self, clf: ColorClassifier):
        assert clf.classify(self._make_patch(BGR_YELLOW)) == COLOR_YELLOW

    def test_vote_classify_pure_purple(self, clf: ColorClassifier):
        assert clf.classify(self._make_patch(BGR_PURPLE)) == COLOR_PURPLE

    def test_vote_classify_pure_ojama(self, clf: ColorClassifier):
        """全面グレーは OJAMA."""
        assert clf.classify(self._make_patch(BGR_OJAMA)) == COLOR_OJAMA

    def test_vote_classify_pure_empty(self, clf: ColorClassifier):
        """全面暗色は EMPTY."""
        assert clf.classify(self._make_patch(BGR_EMPTY)) == COLOR_EMPTY

    def test_vote_classify_half_red_half_empty(self, clf: ColorClassifier):
        """半分赤・半分空 cell は puyo 票 50% で COLOR_RED 採用 (= 主目的)."""
        patch = np.zeros((20, 20, 3), dtype=np.uint8)
        patch[10:, :] = BGR_RED  # 下半分が赤
        patch[:10, :] = BGR_EMPTY  # 上半分が空
        assert clf.classify(patch) == COLOR_RED

    def test_vote_classify_quarter_blue_rest_empty(
        self, clf: ColorClassifier,
    ):
        """1/4 だけ青、 残り空 → 25% > 10% 閾値で COLOR_BLUE 採用."""
        patch = np.zeros((20, 20, 3), dtype=np.uint8)
        patch[:, :] = BGR_EMPTY
        patch[10:, 10:] = BGR_BLUE  # 右下 10x10 = 25% が青
        assert clf.classify(patch) == COLOR_BLUE

    def test_vote_classify_tiny_puyo_below_threshold(
        self, clf: ColorClassifier,
    ):
        """puyo 票が 10% 未満 (5x5=25 / 400=6%) → EMPTY (= ノイズ抑制)."""
        patch = np.zeros((20, 20, 3), dtype=np.uint8)
        patch[:, :] = BGR_EMPTY
        patch[0:5, 0:5] = BGR_RED  # 6% だけ赤
        assert clf.classify(patch) == COLOR_EMPTY

    def test_vote_classify_red_vs_yellow_mixed(self, clf: ColorClassifier):
        """赤と黄の混合: 赤の方が多ければ赤、 同数なら HSV 順位の方."""
        patch = np.zeros((20, 20, 3), dtype=np.uint8)
        patch[:, :10] = BGR_RED
        patch[:, 10:] = BGR_YELLOW
        result = clf.classify(patch)
        # 赤と黄が半々 → どちらか採用 (両方 puyo 色なので非 EMPTY 確認)
        assert result in (COLOR_RED, COLOR_YELLOW)

    def test_vote_classify_empty_patch_size_zero(self, clf: ColorClassifier):
        """空パッチは EMPTY."""
        assert clf.classify(
            np.zeros((0, 0, 3), dtype=np.uint8),
        ) == COLOR_EMPTY

    def test_vote_mode_default_is_false(self):
        """vote_mode 未指定 (= デフォルト False) は median 方式と同じ挙動."""
        clf_default = ColorClassifier()
        # 半分埋まり cell は median 方式では空に倒れる可能性が高い
        # (= 投票方式と挙動が違うことを確認、 backwards compat 担保)
        patch = np.zeros((20, 20, 3), dtype=np.uint8)
        patch[15:, :] = BGR_RED
        patch[:15, :] = BGR_EMPTY
        # median 方式は 75% empty で empty に倒れる
        assert clf_default.classify(patch) == COLOR_EMPTY


# ============================
# ImageReader テスト
# ============================

class TestImageReader:
    @pytest.fixture
    def reader(self) -> ImageReader:
        return ImageReader(p1_region=TEST_REGION, p2_region=TEST_REGION)

    def test_read_empty_board(self, reader: ImageReader):
        """全空の合成フレームから空盤面が返される。"""
        grid = all_same_color_grid(COLOR_EMPTY)
        frame = make_synthetic_frame(TEST_REGION, grid)
        board = reader.read_board(frame, TEST_REGION)
        assert board.count_puyos() == 0

    def test_read_all_red_board(self, reader: ImageReader):
        """全赤の合成フレームから、可視行が全赤・隠し段は UNKNOWN。"""
        grid = all_same_color_grid(COLOR_RED)
        frame = make_synthetic_frame(TEST_REGION, grid)
        board = reader.read_board(frame, TEST_REGION)
        # 可視領域のみ puyo (VISIBLE_ROWS * BOARD_COLS)
        assert board.count_puyos() == VISIBLE_ROWS * BOARD_COLS
        assert board.get(0, 0) == COLOR_UNKNOWN
        assert board.get(HIDDEN_ROWS, 0) == COLOR_RED
        assert board.get(12, 5) == COLOR_RED

    def test_read_all_blue_board(self, reader: ImageReader):
        grid = all_same_color_grid(COLOR_BLUE)
        frame = make_synthetic_frame(TEST_REGION, grid)
        board = reader.read_board(frame, TEST_REGION)
        assert board.get(6, 3) == COLOR_BLUE

    def test_read_all_green_board(self, reader: ImageReader):
        grid = all_same_color_grid(COLOR_GREEN)
        frame = make_synthetic_frame(TEST_REGION, grid)
        board = reader.read_board(frame, TEST_REGION)
        # 隠し段は UNKNOWN、可視領域は GREEN
        assert board.get(0, 0) == COLOR_UNKNOWN
        assert board.get(HIDDEN_ROWS, 0) == COLOR_GREEN

    def test_read_all_yellow_board(self, reader: ImageReader):
        grid = all_same_color_grid(COLOR_YELLOW)
        frame = make_synthetic_frame(TEST_REGION, grid)
        board = reader.read_board(frame, TEST_REGION)
        assert board.get(12, 5) == COLOR_YELLOW

    def test_read_all_purple_board(self, reader: ImageReader):
        grid = all_same_color_grid(COLOR_PURPLE)
        frame = make_synthetic_frame(TEST_REGION, grid)
        board = reader.read_board(frame, TEST_REGION)
        assert board.get(6, 2) == COLOR_PURPLE

    def test_read_all_ojama_board(self, reader: ImageReader):
        grid = all_same_color_grid(COLOR_OJAMA)
        frame = make_synthetic_frame(TEST_REGION, grid)
        board = reader.read_board(frame, TEST_REGION)
        assert board.get(0, 0) == COLOR_UNKNOWN
        assert board.get(HIDDEN_ROWS, 0) == COLOR_OJAMA

    def test_read_mixed_board(self, reader: ImageReader):
        """複数色が混在する盤面を正しく読み取れる。"""
        grid = all_same_color_grid(COLOR_EMPTY)
        grid[12][0] = COLOR_RED
        grid[12][1] = COLOR_BLUE
        grid[12][2] = COLOR_GREEN
        grid[12][3] = COLOR_YELLOW
        grid[12][4] = COLOR_PURPLE
        grid[12][5] = COLOR_OJAMA

        frame = make_synthetic_frame(TEST_REGION, grid)
        board = reader.read_board(frame, TEST_REGION)

        assert board.get(12, 0) == COLOR_RED
        assert board.get(12, 1) == COLOR_BLUE
        assert board.get(12, 2) == COLOR_GREEN
        assert board.get(12, 3) == COLOR_YELLOW
        assert board.get(12, 4) == COLOR_PURPLE
        assert board.get(12, 5) == COLOR_OJAMA

    def test_read_both_boards_same_region(self, reader: ImageReader):
        """read_both_boards が2つのBoardを返す。"""
        grid = all_same_color_grid(COLOR_RED)
        frame = make_synthetic_frame(TEST_REGION, grid)
        b1, b2 = reader.read_both_boards(frame)
        assert isinstance(b1, Board)
        assert isinstance(b2, Board)

    def test_read_board_clips_to_frame_boundary(self):
        """領域がフレーム端に近い場合もクラッシュしない。"""
        # 小さなフレームで領域が境界をはみ出す状況
        region = BoardRegion(x=0, y=0, width=60, height=130)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        reader = ImageReader(p1_region=region, p2_region=region)
        board = reader.read_board(frame, region)
        assert isinstance(board, Board)

    def test_debug_frame_returns_image(self, reader: ImageReader):
        """debug_frame が同じサイズの画像を返す。"""
        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        result = reader.debug_frame(frame, TEST_REGION)
        assert result.shape == frame.shape

    def test_is_dead_detection_via_reader(self, reader: ImageReader):
        """
        画像読み取りでは隠し段 (row 0) は物理推論される。
        可視最上段 (row 1) が空なら row 0 も空と確定。
        """
        grid = all_same_color_grid(COLOR_EMPTY)
        grid[0][2] = COLOR_RED  # 画面外なので描画されない
        frame = make_synthetic_frame(TEST_REGION, grid)
        board = reader.read_board(frame, TEST_REGION)
        # 可視最上段が空なので row 0 も empty 確定
        assert board.get(0, 2) == COLOR_EMPTY
        assert board.is_dead() is False

    def test_hidden_row_unknown_when_top_occupied(self, reader: ImageReader):
        """可視最上段に puyo があると同列の row 0 は UNKNOWN (回し入れ可能性)。"""
        grid = all_same_color_grid(COLOR_EMPTY)
        grid[HIDDEN_ROWS][0] = COLOR_RED  # 可視最上段
        frame = make_synthetic_frame(TEST_REGION, grid)
        board = reader.read_board(frame, TEST_REGION)
        assert board.get(0, 0) == COLOR_UNKNOWN
        # 他の列は row 1 が空なので row 0 も空
        assert board.get(0, 1) == COLOR_EMPTY


# ============================
# I1 対応 A: bg_fp 採取前保護モードテスト
# ============================


class TestPreCaptureModeI1A:
    """set_pre_capture_mode の挙動を検証するテスト群。"""

    def test_pre_capture_mode_default_is_false(self) -> None:
        """ImageReader のデフォルト pre_capture_mode は False。"""
        reader = ImageReader()
        assert reader._pre_capture_mode is False

    def test_set_pre_capture_mode_true(self) -> None:
        """set_pre_capture_mode(True) で _pre_capture_mode が True になる。"""
        reader = ImageReader()
        reader.set_pre_capture_mode(True)
        assert reader._pre_capture_mode is True

    def test_set_pre_capture_mode_false(self) -> None:
        """set_pre_capture_mode(False) で _pre_capture_mode が False に戻る。"""
        reader = ImageReader()
        reader.set_pre_capture_mode(True)
        reader.set_pre_capture_mode(False)
        assert reader._pre_capture_mode is False

    def test_resolve_tier1_threshold_pre_capture_returns_zero(self) -> None:
        """pre_capture_mode=True のとき _resolve_tier1_threshold は 0.0 を返す。"""
        reader = ImageReader()
        reader.set_pre_capture_mode(True)
        # 全セル位置で 0.0 (= tier1 スキップ) になることを確認
        for visible_row in range(12):
            for col in range(6):
                threshold = reader._resolve_tier1_threshold(visible_row, col)
                assert threshold == BG_EXTREME_THRESHOLD_PRE_CAPTURE, (
                    f"visible_row={visible_row}, col={col}: "
                    f"expected {BG_EXTREME_THRESHOLD_PRE_CAPTURE}, got {threshold}"
                )

    def test_resolve_tier1_threshold_normal_mode_returns_default(self) -> None:
        """pre_capture_mode=False のとき通常の DEFAULT threshold を返す。"""
        reader = ImageReader()
        reader.set_pre_capture_mode(False)
        # 左上エリア外のセル (row=0, col=5) は DEFAULT を返す
        threshold = reader._resolve_tier1_threshold(0, 5)
        assert threshold == BG_EXTREME_THRESHOLD_DEFAULT

    def test_pre_capture_mode_overrides_left_upper_threshold(self) -> None:
        """pre_capture_mode=True は左上エリア (軸 3-b) の threshold も上書きする。

        pre_capture_mode が優先度最高なので、左上エリア (visible_row>=5, col<=1)
        でも 0.0 を返すことを確認。
        """
        reader = ImageReader()
        reader.set_pre_capture_mode(True)
        # 左上エリアのセル (visible_row=5, col=0)
        threshold = reader._resolve_tier1_threshold(5, 0)
        assert threshold == BG_EXTREME_THRESHOLD_PRE_CAPTURE


# ============================
# 案 P2: use_highlight_override テスト
# ============================

class TestHighlightOverride:
    """ImageReader の use_highlight_override 引数テスト。"""

    def test_image_reader_highlight_override_default_false(self) -> None:
        """案 R3 改 (2026-05-28): 案 P2 撤回により default が False に変更。"""
        reader = ImageReader()
        assert reader._use_highlight_override is False

    def test_image_reader_highlight_override_explicit_true(self) -> None:
        """use_highlight_override=True で明示的に有効化できる (再評価用)。"""
        reader = ImageReader(use_highlight_override=True)
        assert reader._use_highlight_override is True

    def test_image_reader_highlight_override_disabled(self) -> None:
        """use_highlight_override=False で _use_highlight_override が False。"""
        reader = ImageReader(use_highlight_override=False)
        assert reader._use_highlight_override is False


class TestStaticMaskAndGuard:
    """T4 StaticBoardMask AND ガード: ImageReader.set_static_mask テスト。"""

    def test_set_static_mask_stores_values(self) -> None:
        """set_static_mask で _static_mask_p1 / _static_mask_p2 が設定される。"""
        from src.background_fingerprint import StaticBoardMask
        reader = ImageReader()
        bg = np.zeros((720, 384, 3), dtype=np.uint8)
        m = StaticBoardMask(bg_roi=bg, region_x=0, region_y=0, region_w=384, region_h=720)
        reader.set_static_mask(m, None)
        assert reader._static_mask_p1 is m
        assert reader._static_mask_p2 is None

    def test_set_static_mask_none_clears(self) -> None:
        """None をセットすると無効化される。"""
        from src.background_fingerprint import StaticBoardMask
        reader = ImageReader()
        bg = np.zeros((720, 384, 3), dtype=np.uint8)
        m = StaticBoardMask(bg_roi=bg, region_x=0, region_y=0, region_w=384, region_h=720)
        reader.set_static_mask(m, m)
        reader.set_static_mask(None, None)
        assert reader._static_mask_p1 is None
        assert reader._static_mask_p2 is None

    def test_initial_static_mask_is_none(self) -> None:
        """初期状態で static_mask は None。"""
        reader = ImageReader()
        assert reader._static_mask_p1 is None
        assert reader._static_mask_p2 is None

    def test_is_empty_static_mask_no_mask(self) -> None:
        """StaticBoardMask が None の場合は常に False (= 判定スキップ)。"""
        reader = ImageReader()
        frame = np.zeros((720, 384, 3), dtype=np.uint8)
        region = BoardRegion(x=0, y=0, width=384, height=720)
        hsv = np.zeros((6, 6, 3), dtype=np.float32)
        result = reader._is_empty_static_mask(frame, region, 0, 0, hsv)
        assert result is False

    def test_is_empty_static_mask_same_frame(self) -> None:
        """背景と同じフレームで diff=0 → True (色なし前提)。"""
        from src.background_fingerprint import StaticBoardMask
        bg_color = (50, 50, 50)
        bg = np.full((720, 384, 3), bg_color, dtype=np.uint8)
        mask = StaticBoardMask(
            bg_roi=bg, region_x=0, region_y=0, region_w=384, region_h=720,
        )
        reader = ImageReader()
        reader.set_static_mask(mask, None)
        frame = np.full((720, 384, 3), bg_color, dtype=np.uint8)
        region = reader._p1_region  # p1 と同一オブジェクト
        # HSV: 暗色 (彩度 0) → 色なし signal
        hsv = np.zeros((6, 6, 3), dtype=np.float32)
        result = reader._is_empty_static_mask(frame, region, 0, 0, hsv)
        assert result is True


# ===== skip_tier1 テスト群 =====

class TestSkipTier1:
    """read_board / read_both_boards の skip_tier1 引数のテスト。"""

    def test_read_board_skip_tier1_default_is_false(self) -> None:
        """skip_tier1 のデフォルト値が False であること (= 既存挙動維持)。"""
        import inspect
        sig = inspect.signature(ImageReader.read_board)
        default = sig.parameters["skip_tier1"].default
        assert default is False

    def test_read_board_skip_tier1_false_returns_board(self) -> None:
        """skip_tier1=False で Board が正常に返る (= 既存挙動)。"""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        reader = ImageReader()
        board = reader.read_board(frame, reader._p1_region, skip_tier1=False)
        assert board is not None

    def test_read_board_skip_tier1_true_returns_board(self) -> None:
        """skip_tier1=True でも Board が正常に返る (= tier1 skip は例外を投げない)。"""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        reader = ImageReader()
        board = reader.read_board(frame, reader._p1_region, skip_tier1=True)
        assert board is not None

    def test_read_both_boards_skip_tier1_defaults_false(self) -> None:
        """read_both_boards の skip_tier1_1p / skip_tier1_2p デフォルトが False。"""
        import inspect
        sig = inspect.signature(ImageReader.read_both_boards)
        assert sig.parameters["skip_tier1_1p"].default is False
        assert sig.parameters["skip_tier1_2p"].default is False

    def test_read_both_boards_skip_tier1_returns_tuple(self) -> None:
        """skip_tier1_1p=True, skip_tier1_2p=True でも (Board, Board) が返る。"""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        reader = ImageReader()
        result = reader.read_both_boards(
            frame, skip_tier1_1p=True, skip_tier1_2p=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2


# ============================
# 赤色相折り返し補正テスト (fix/v70-zeropatch-redyellow)
# ============================

class TestRedHueWrapFix:
    """enable_red_hue_wrap_fix の動作確認テスト。"""

    def _make_patch_from_hsv(
        self, h: int, s: int, v: int, size: int = 20,
    ) -> np.ndarray:
        """HSV 値から BGR パッチを生成する。"""
        hsv = np.full((size, size, 3), [h, s, v], dtype=np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _make_bimodal_red_patch(
        self, size: int = 20, high_h: int = 173, low_h: int = 2,
        s: int = 200, v: int = 200,
    ) -> np.ndarray:
        """赤の 2 峰 (H=2 と H=173) を半々に混ぜた BGRパッチを生成する。

        実際の赤ぷよのように明部 (H=0-4) と暗部 (H=166-179) が混在する状況を模擬。
        """
        half = size // 2
        patch = np.zeros((size, size, 3), dtype=np.uint8)
        # 上半分: H=low_h (明部赤)
        hsv_lo = np.full((half, size, 3), [low_h, s, v], dtype=np.uint8)
        patch[:half, :] = cv2.cvtColor(hsv_lo, cv2.COLOR_HSV2BGR)
        # 下半分: H=high_h (暗部赤)
        hsv_hi = np.full((size - half, size, 3), [high_h, s, v], dtype=np.uint8)
        patch[half:, :] = cv2.cvtColor(hsv_hi, cv2.COLOR_HSV2BGR)
        return patch

    def test_off_mode_unchanged(self) -> None:
        """OFF 時 (明示的 False): 単純 median と完全同一の挙動。

        default は True (ON) に変更済 (user viz 採用承認 2026-06-02)。
        OFF の後方互換性を回帰防止として保持。
        """
        clf_off = ColorClassifier(enable_red_hue_wrap_fix=False)
        # 2 峰分布 (H=2 と H=173 が半々)
        h_arr = np.array([2, 2, 2, 2, 2, 173, 173, 173, 173, 173], dtype=np.uint8)
        result_off = clf_off._compute_stable_h_median(h_arr)
        expected = int(np.median(h_arr))
        assert result_off == expected, (
            f"OFF 時に単純 median と一致しない: {result_off} != {expected}"
        )

    def test_on_bimodal_red_classified_as_red(self) -> None:
        """ON 時: 2 峰赤 (H=2 と H=173 半々) が安定して RED 判定される。

        真因: 単純 median が H=13/14 付近に乗り赤/黄ちらつきが起きる。
        修正後: 折り返し補正で median が H=0 付近に collapse → RED 判定。
        """
        clf = ColorClassifier(enable_red_hue_wrap_fix=True)
        patch = self._make_bimodal_red_patch()
        result = clf.classify(patch)
        assert result == COLOR_RED, (
            f"2 峰赤パッチが RED にならず {result} になった (折り返し補正が機能していない)"
        )

    def test_on_pure_yellow_unchanged(self) -> None:
        """ON 時: 純黄 (H=26, R≒G) は YELLOW のまま変わらない。"""
        clf = ColorClassifier(enable_red_hue_wrap_fix=True)
        # H=26 は黄のコア域、折り返し補正対象外
        patch = self._make_patch_from_hsv(h=26, s=220, v=200)
        assert clf.classify(patch) == COLOR_YELLOW, (
            "黄パッチが折り返し補正で誤 RED 判定された"
        )

    def test_on_purple_unchanged(self) -> None:
        """ON 時: 紫 (H=145) は PURPLE のまま変わらない。"""
        clf = ColorClassifier(enable_red_hue_wrap_fix=True)
        patch = self._make_patch_from_hsv(h=145, s=180, v=180)
        assert clf.classify(patch) == COLOR_PURPLE, (
            "紫パッチが折り返し補正で誤判定された"
        )

    def test_on_pure_high_h_red_classified_as_red(self) -> None:
        """ON 時: 純粋 H=170 (高 H 赤) は RED に分類される。"""
        clf = ColorClassifier(enable_red_hue_wrap_fix=True)
        patch = self._make_patch_from_hsv(h=170, s=200, v=200)
        assert clf.classify(patch) == COLOR_RED, (
            "高 H 赤パッチが RED にならなかった"
        )

    def test_compute_stable_h_median_off_equals_np_median(self) -> None:
        """OFF 時: _compute_stable_h_median は int(np.median) と完全一致。"""
        clf = ColorClassifier(enable_red_hue_wrap_fix=False)
        # 典型的な 2 峰分布 (H=2 と H=173 が半々)
        h_arr = np.array([2, 2, 2, 2, 2, 173, 173, 173, 173, 173], dtype=np.uint8)
        result = clf._compute_stable_h_median(h_arr)
        expected = int(np.median(h_arr))
        assert result == expected, (
            f"OFF 時に単純 median と一致しない: {result} != {expected}"
        )

    def test_compute_stable_h_median_on_collapses_bimodal(self) -> None:
        """ON 時: 2 峰 H (0-4 と 166-179 が半々) の median が赤域 (0-13) に入る。"""
        clf = ColorClassifier(enable_red_hue_wrap_fix=True)
        # H=2 (低端) と H=173 (高端) が半々 → 補正後 median は 0 付近になるはず
        h_arr = np.array([2, 2, 2, 2, 2, 173, 173, 173, 173, 173], dtype=np.uint8)
        result = clf._compute_stable_h_median(h_arr)
        assert 0 <= result <= 13, (
            f"折り返し補正後の median が赤域 (0-13) に入らない: {result}"
        )

