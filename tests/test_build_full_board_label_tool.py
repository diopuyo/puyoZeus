"""満杯盤面ラベル付けクリック操作HTMLツール (scripts/build_full_board_label_tool.py) の単体テスト。

実CSV/実動画は使わず合成データのみで、grid往復エンコード・クロップ座標系・
HTML生成の整合を検証する (label_sheet版 tests/test_build_full_board_label_sheet.py
と対を成す)。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.build_full_board_label_sheet import encode_grid_string
from scripts.build_full_board_label_tool import (
    CYCLE_ORDER,
    DISPLAY_WIDTH_PX,
    HIDDEN_PANEL_HEIGHT_PX,
    RESULT_CSV_HEADER,
    STATUS_FIXED,
    STATUS_OK,
    STATUS_SKIP,
    ToolCandidate,
    build_board_crop_image,
    build_color_palette,
    build_tool_candidate,
    build_tool_candidates,
    crop_visible_board_region,
    decode_grid_string,
    frame_basename_from_row,
    render_html_document,
    validate_generated_html,
)
from scripts.visualize_recognition import (
    CELL_H, CELL_W, COLOR_BGR, N_VISIBLE_ROWS, P1_ROI_X, P1_ROI_Y, P2_ROI_X,
    P2_ROI_Y, ROI_H, ROI_W,
)
from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, COLOR_RED, COLOR_UNKNOWN,
    HIDDEN_ROWS,
)


def _make_grid(n_filled: int, fill_color: int = COLOR_RED) -> np.ndarray:
    """非空セル数 n_filled の合成グリッドを作る (盤面下段から埋める)。"""
    grid = np.full((BOARD_ROWS, BOARD_COLS), COLOR_EMPTY, dtype=np.int64)
    flat = grid.reshape(-1)
    flat[-n_filled:] = fill_color if n_filled > 0 else COLOR_EMPTY
    return flat.reshape(BOARD_ROWS, BOARD_COLS)


def _make_csv_row(
    *, video_id: str = "video_c1", t_sec: str = "101.3", side: str = "1P",
    grid: "np.ndarray | None" = None,
) -> dict:
    g = grid if grid is not None else _make_grid(10)
    return {
        "video_id": video_id, "t_sec": t_sec, "side": side, "game_idx": "0",
        "occupancy": str(int(np.count_nonzero(g != COLOR_EMPTY))),
        "tier": "primary", "phase": "終盤",
        "recognized_grid": encode_grid_string(g),
    }


# =============================================================================
# decode_grid_string (encode_grid_string の逆変換)
# =============================================================================


class TestDecodeGridString:
    def test_round_trips_with_encode(self) -> None:
        grid = _make_grid(37)
        decoded = decode_grid_string(encode_grid_string(grid))
        assert np.array_equal(decoded, grid)

    def test_unknown_round_trips(self) -> None:
        grid = np.full((BOARD_ROWS, BOARD_COLS), COLOR_EMPTY, dtype=np.int64)
        grid[0, :] = COLOR_UNKNOWN
        decoded = decode_grid_string(encode_grid_string(grid))
        assert np.all(decoded[0, :] == COLOR_UNKNOWN)

    def test_ojama_round_trips(self) -> None:
        grid = np.full((BOARD_ROWS, BOARD_COLS), COLOR_EMPTY, dtype=np.int64)
        grid[5, 2] = COLOR_OJAMA
        decoded = decode_grid_string(encode_grid_string(grid))
        assert decoded[5, 2] == COLOR_OJAMA

    def test_empty_board_round_trips(self) -> None:
        grid = _make_grid(0)
        decoded = decode_grid_string(encode_grid_string(grid))
        assert np.array_equal(decoded, grid)


# =============================================================================
# crop_visible_board_region (座標系整合)
# =============================================================================


class TestCropVisibleBoardRegion:
    def _marker_frame(self) -> np.ndarray:
        """1080p黒画面に、P1/P2 ROI内側にだけ判別可能なマーカー値を置く。"""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[P1_ROI_Y:P1_ROI_Y + ROI_H, P1_ROI_X:P1_ROI_X + ROI_W] = (11, 22, 33)
        frame[P2_ROI_Y:P2_ROI_Y + ROI_H, P2_ROI_X:P2_ROI_X + ROI_W] = (44, 55, 66)
        return frame

    def test_crop_shape_matches_roi(self) -> None:
        crop = crop_visible_board_region(self._marker_frame(), "1P")
        assert crop.shape == (ROI_H, ROI_W, 3)

    def test_1p_crop_picks_p1_roi(self) -> None:
        crop = crop_visible_board_region(self._marker_frame(), "1P")
        assert tuple(crop[0, 0]) == (11, 22, 33)

    def test_2p_crop_picks_p2_roi_not_p1(self) -> None:
        crop = crop_visible_board_region(self._marker_frame(), "2P")
        assert tuple(crop[0, 0]) == (44, 55, 66)

    def test_crop_rows_match_visible_rows_times_cell_h(self) -> None:
        # ROI_H は可視12行×CELL_Hに一致するはず (visualize_recognition側の定義との整合)
        assert ROI_H == N_VISIBLE_ROWS * CELL_H
        assert ROI_W == BOARD_COLS * CELL_W

    def test_hidden_plus_visible_rows_equals_board_rows(self) -> None:
        assert HIDDEN_ROWS + N_VISIBLE_ROWS == BOARD_ROWS


# =============================================================================
# frame_basename_from_row
# =============================================================================


class TestFrameBasenameFromRow:
    def test_strips_video_prefix(self) -> None:
        row = _make_csv_row(video_id="video_c17", t_sec="101.3", side="1P")
        assert frame_basename_from_row(row) == "c17_t101.3_1P"

    def test_keeps_id_without_prefix(self) -> None:
        row = _make_csv_row(video_id="c17", t_sec="101.3", side="2P")
        assert frame_basename_from_row(row) == "c17_t101.3_2P"

    def test_formats_t_sec_to_one_decimal(self) -> None:
        row = _make_csv_row(video_id="video_c5", t_sec="290.0", side="1P")
        assert frame_basename_from_row(row) == "c5_t290.0_1P"


# =============================================================================
# build_board_crop_image / build_tool_candidate(s) (欠損時の graceful skip)
# =============================================================================


class TestBuildBoardCropImage:
    def test_missing_full_frame_returns_none(self, tmp_path) -> None:
        assert build_board_crop_image(tmp_path, "nope_t1.0_1P") is None

    def test_valid_full_frame_produces_crop_file(self, tmp_path) -> None:
        import cv2
        base = "c1_t1.0_1P"
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"{base}_full.png"), frame)
        out = build_board_crop_image(tmp_path, base)
        assert out is not None
        assert out.exists()
        crop = cv2.imread(str(out))
        assert crop.shape == (ROI_H, ROI_W, 3)


class TestBuildToolCandidates:
    def _write_full_frame(self, tmp_path, base: str) -> None:
        import cv2
        cv2.imwrite(str(tmp_path / f"{base}_full.png"), np.zeros((1080, 1920, 3), dtype=np.uint8))

    def test_skips_row_without_full_frame(self, tmp_path) -> None:
        row = _make_csv_row(video_id="video_nofile", t_sec="1.0", side="1P")
        cands = build_tool_candidates([row], tmp_path)
        assert cands == []

    def test_builds_candidate_when_frame_present(self, tmp_path) -> None:
        row = _make_csv_row(video_id="video_c1", t_sec="1.0", side="1P")
        self._write_full_frame(tmp_path, "c1_t1.0_1P")
        cands = build_tool_candidates([row], tmp_path)
        assert len(cands) == 1
        cand = cands[0]
        assert cand.image_rel_path == "frames/c1_t1.0_1P_board_crop.png"
        assert len(cand.init_grid) == BOARD_ROWS
        assert all(len(r) == BOARD_COLS for r in cand.init_grid)

    def test_image_rel_path_uses_posix_separator(self, tmp_path) -> None:
        row = _make_csv_row(video_id="video_c1", t_sec="1.0", side="1P")
        self._write_full_frame(tmp_path, "c1_t1.0_1P")
        cand = build_tool_candidate(row, tmp_path)
        assert cand is not None
        assert "\\" not in cand.image_rel_path
        assert cand.image_rel_path.startswith("frames/")

    def test_key_is_unique_per_video_t_sec_side(self, tmp_path) -> None:
        rows = [
            _make_csv_row(video_id="video_c1", t_sec="1.0", side="1P"),
            _make_csv_row(video_id="video_c1", t_sec="2.0", side="1P"),
        ]
        self._write_full_frame(tmp_path, "c1_t1.0_1P")
        self._write_full_frame(tmp_path, "c1_t2.0_1P")
        cands = build_tool_candidates(rows, tmp_path)
        assert len({c.key for c in cands}) == 2


# =============================================================================
# 色パレット・サイクル順 (既存凡例との整合)
# =============================================================================


class TestColorPalette:
    def test_cycle_order_matches_user_spec(self) -> None:
        # 空->赤->青->緑->黄->紫->おじゃま->不明 の順
        assert CYCLE_ORDER == (0, 1, 2, 3, 4, 5, 9, 10)

    def test_palette_covers_all_cycle_colors(self) -> None:
        palette = build_color_palette()
        assert set(palette.keys()) == {str(c) for c in CYCLE_ORDER}

    def test_palette_hex_matches_color_bgr(self) -> None:
        palette = build_color_palette()
        for color_id in CYCLE_ORDER:
            b, g, r = COLOR_BGR[color_id]
            assert palette[str(color_id)]["hex"] == f"#{r:02x}{g:02x}{b:02x}"


# =============================================================================
# render_html_document / validate_generated_html
# =============================================================================


def _sample_candidate(key: str = "video_c1|1.0|1P") -> ToolCandidate:
    grid = _make_grid(20).tolist()
    return ToolCandidate(
        key=key, video_id="video_c1", t_sec="1.0", side="1P", game_idx="0",
        occupancy="20", tier="primary", phase="終盤",
        image_rel_path="frames/c1_t1.0_1P_board_crop.png", init_grid=grid,
    )


class TestRenderHtmlDocument:
    def test_contains_valid_embedded_json(self) -> None:
        html = render_html_document([_sample_candidate()], "test_key")
        start = html.index("const CANDIDATES = ") + len("const CANDIDATES = ")
        end = html.index(";", start)
        parsed = json.loads(html[start:end])
        assert len(parsed) == 1
        assert parsed[0]["video_id"] == "video_c1"

    def test_image_path_is_relative(self) -> None:
        html = render_html_document([_sample_candidate()], "test_key")
        assert 'src="frames/c1_t1.0_1P_board_crop.png"' in html
        assert "C:\\" not in html

    def test_contains_three_status_buttons_per_candidate(self) -> None:
        html = render_html_document([_sample_candidate(), _sample_candidate("k2")], "test_key")
        body = html.split("<script>")[0]  # JS内のラベル文字列と混同しないよう本文のみ対象
        assert body.count("認識通りでOK") == 2
        assert body.count("修正完了") == 2
        assert body.count("非ゲーム画面") == 2

    def test_contains_hidden_row_note(self) -> None:
        html = render_html_document([_sample_candidate()], "test_key")
        assert "隠し段" in html
        assert "画面外" in html

    def test_download_button_present(self) -> None:
        html = render_html_document([_sample_candidate()], "test_key")
        assert "結果をダウンロード" in html
        assert "labeling_result.csv" in html

    def test_result_csv_header_matches_constant(self) -> None:
        html = render_html_document([_sample_candidate()], "test_key")
        assert ",".join(RESULT_CSV_HEADER) in html

    def test_storage_key_embedded(self) -> None:
        html = render_html_document([_sample_candidate()], "my_unique_key")
        assert "my_unique_key" in html

    def test_progress_shows_total_count(self) -> None:
        cands = [_sample_candidate(f"k{i}") for i in range(3)]
        html = render_html_document(cands, "test_key")
        assert "完了 0/3" in html

    def test_html_escapes_special_chars_in_video_id(self) -> None:
        cand = _sample_candidate()
        cand.video_id = 'video_<script>&"'
        html = render_html_document([cand], "test_key")
        assert "<script>&" not in html.split("const CANDIDATES")[0]


class TestValidateGeneratedHtml:
    def test_passes_when_images_exist(self, tmp_path) -> None:
        import cv2
        (tmp_path / "frames").mkdir()
        cv2.imwrite(str(tmp_path / "frames" / "c1_t1.0_1P_board_crop.png"),
                    np.zeros((ROI_H, ROI_W, 3), dtype=np.uint8))
        cand = _sample_candidate()
        html = render_html_document([cand], "test_key")
        html_path = tmp_path / "label_tool.html"
        html_path.write_text(html, encoding="utf-8")
        validate_generated_html(html_path, [cand], tmp_path)  # 例外なしを確認

    def test_raises_when_image_missing(self, tmp_path) -> None:
        cand = _sample_candidate()
        html = render_html_document([cand], "test_key")
        html_path = tmp_path / "label_tool.html"
        html_path.write_text(html, encoding="utf-8")
        with pytest.raises(AssertionError):
            validate_generated_html(html_path, [cand], tmp_path)


# =============================================================================
# 表示幾何 (座標系ずれ検出)
# =============================================================================


class TestDisplayGeometry:
    def test_hidden_panel_height_scales_with_display_width(self) -> None:
        scale = DISPLAY_WIDTH_PX / ROI_W
        assert HIDDEN_PANEL_HEIGHT_PX == round(CELL_H * HIDDEN_ROWS * scale)

    def test_status_constants_are_distinct(self) -> None:
        assert len({STATUS_OK, STATUS_FIXED, STATUS_SKIP}) == 3
