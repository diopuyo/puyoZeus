"""エフェクト有無セルラベル付けクリックHTMLツール第3弾 (scripts/build_effect_cell_label_tool_v3.py)
の単体テスト。

実CSV/実動画は使わず合成データのみで、パレット/座標系/HTML生成の整合、および
新設「対象外エフェクト」ボタンの配線を検証する
(第1弾 tests/test_build_effect_cell_label_tool.py と対を成す)。
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from scripts.build_effect_cell_label_tool_v3 import (
    BOARD_DISPLAY_HEIGHT_PX,
    CYCLE_ORDER,
    DISPLAY_WIDTH_PX,
    EFFECT_STATE_BURST,
    EFFECT_STATE_NONE,
    EFFECT_STATE_SMOKE,
    RESULT_CSV_HEADER,
    STATE_PALETTE,
    STATUS_MARKED,
    STATUS_NO_EFFECT,
    STATUS_OUT_OF_SCOPE,
    STATUS_SKIP,
    EffectToolCandidate,
    _rel_path_under_frames,
    build_tool_candidate,
    build_tool_candidates,
    render_html_document,
    validate_generated_html,
)
from scripts.visualize_recognition import CELL_H, N_VISIBLE_ROWS, ROI_W


def _make_csv_row(
    *, video_id: str = "video_c18", t_sec: str = "100.00", side: str = "2P",
    layer: str = "burst", note: str = "相手連鎖5連鎖の受け側",
    full_path: str = r"C:\data\frames\c18_t100.00_2P_burst_full.png",
    crop_path: str = r"C:\data\frames\c18_t100.00_2P_burst_board_crop.png",
) -> dict:
    return {
        "video_id": video_id, "t_sec": t_sec, "side": side, "layer": layer,
        "note": note, "image_full_frame": full_path, "image_board_crop": crop_path,
    }


# =============================================================================
# _rel_path_under_frames
# =============================================================================


class TestRelPathUnderFrames:
    def test_windows_path_to_frames_relative(self) -> None:
        rel = _rel_path_under_frames(r"C:\data\verify\effect_cell_label_v3_2026-08-04\frames\c18_x.png")
        assert rel == "frames/c18_x.png"

    def test_no_backslash_in_output(self) -> None:
        rel = _rel_path_under_frames(r"C:\some\path\img.png")
        assert "\\" not in rel


# =============================================================================
# 状態パレット・サイクル順・ステータス定数 (4状態: no_effect/marked/out_of_scope/skip)
# =============================================================================


class TestStatePalette:
    def test_cycle_order_is_none_burst_smoke(self) -> None:
        assert CYCLE_ORDER == (EFFECT_STATE_NONE, EFFECT_STATE_BURST, EFFECT_STATE_SMOKE)

    def test_palette_covers_all_cycle_states(self) -> None:
        assert set(STATE_PALETTE.keys()) == set(CYCLE_ORDER)

    def test_none_state_is_transparent(self) -> None:
        assert STATE_PALETTE[EFFECT_STATE_NONE]["rgba"] == "transparent"


class TestStatusConstants:
    def test_four_distinct_statuses(self) -> None:
        # 第1弾の3状態 (no_effect/marked/skip) + 新設 out_of_scope の4状態
        assert len({STATUS_NO_EFFECT, STATUS_MARKED, STATUS_OUT_OF_SCOPE, STATUS_SKIP}) == 4

    def test_out_of_scope_is_new_and_distinct_from_skip(self) -> None:
        # 前回はテロップ混同をskipで運用していた、今回は専用ステータスで区別する
        assert STATUS_OUT_OF_SCOPE != STATUS_SKIP


# =============================================================================
# 表示幾何
# =============================================================================


class TestDisplayGeometry:
    def test_board_display_height_scales_with_display_width(self) -> None:
        scale = DISPLAY_WIDTH_PX / ROI_W
        assert BOARD_DISPLAY_HEIGHT_PX == round((N_VISIBLE_ROWS * CELL_H) * scale)


# =============================================================================
# build_tool_candidate(s) (画像欠損時の graceful skip)
# =============================================================================


class TestBuildToolCandidates:
    def _write_pair(self, tmp_path, base: str) -> None:
        frames = tmp_path / "frames"
        frames.mkdir(exist_ok=True)
        cv2.imwrite(str(frames / f"{base}_full.png"), np.zeros((1080, 1920, 3), dtype=np.uint8))
        cv2.imwrite(str(frames / f"{base}_board_crop.png"), np.zeros((720, 384, 3), dtype=np.uint8))

    def test_missing_images_returns_none(self, tmp_path) -> None:
        row = _make_csv_row()
        cand = build_tool_candidate(row, tmp_path)
        assert cand is None

    def test_valid_images_builds_candidate(self, tmp_path) -> None:
        self._write_pair(tmp_path, "c18_t100.00_2P_burst")
        row = _make_csv_row()
        cand = build_tool_candidate(row, tmp_path)
        assert cand is not None
        assert cand.image_rel_path == "frames/c18_t100.00_2P_burst_board_crop.png"
        assert cand.layer == "burst"
        assert cand.note == "相手連鎖5連鎖の受け側"

    def test_skips_missing_and_keeps_valid(self, tmp_path) -> None:
        self._write_pair(tmp_path, "c18_t100.00_2P_burst")
        rows = [
            _make_csv_row(),
            _make_csv_row(video_id="video_missing", full_path=r"C:\x\missing_full.png",
                          crop_path=r"C:\x\missing_board_crop.png"),
        ]
        cands = build_tool_candidates(rows, tmp_path)
        assert len(cands) == 1


# =============================================================================
# render_html_document / validate_generated_html
# =============================================================================


def _sample_candidate(key: str = "video_c18|100.00|2P|burst") -> EffectToolCandidate:
    return EffectToolCandidate(
        key=key, video_id="video_c18", t_sec="100.00", side="2P", layer="burst",
        note="相手連鎖5連鎖の受け側", image_rel_path="frames/c18_t100.00_2P_burst_board_crop.png",
        full_rel_path="frames/c18_t100.00_2P_burst_full.png",
    )


class TestRenderHtmlDocument:
    def test_contains_valid_embedded_json(self) -> None:
        html = render_html_document([_sample_candidate()], "test_key")
        start = html.index("const CANDIDATES = ") + len("const CANDIDATES = ")
        end = html.index(";", start)
        parsed = json.loads(html[start:end])
        assert len(parsed) == 1
        assert parsed[0]["video_id"] == "video_c18"

    def test_four_control_buttons_present(self) -> None:
        html = render_html_document([_sample_candidate(), _sample_candidate("k2")], "test_key")
        body = html.split("<script>")[0]
        assert body.count("エフェクトなし") == 2
        assert body.count("マーク完了") == 2
        assert body.count("対象外エフェクト") == 2
        assert body.count("フレーム異常(スキップ)") == 2

    def test_out_of_scope_button_wired_in_script(self) -> None:
        html = render_html_document([_sample_candidate()], "test_key")
        script = html.split("<script>")[1]
        assert "btn-outofscope" in script
        assert "STATUS_OUT_OF_SCOPE" in script

    def test_single_pane_no_color_grid_comparison(self) -> None:
        html = render_html_document([_sample_candidate()], "test_key")
        assert 'class="board-wrap"' in html
        assert 'class="grid-overlay"' in html
        assert "grid-pane" not in html

    def test_download_button_and_result_header(self) -> None:
        html = render_html_document([_sample_candidate()], "test_key")
        assert "結果をダウンロード" in html
        assert ",".join(RESULT_CSV_HEADER) in html

    def test_result_csv_header_includes_note_column(self) -> None:
        # noteはburst/telop_negative等の補足情報、CSV再現性のため必須
        assert "note" in RESULT_CSV_HEADER

    def test_progress_shows_total(self) -> None:
        cands = [_sample_candidate(f"k{i}") for i in range(4)]
        html = render_html_document(cands, "test_key")
        assert "完了 0/4" in html

    def test_html_escapes_special_chars(self) -> None:
        cand = _sample_candidate()
        cand.video_id = 'video_<script>&"'
        html = render_html_document([cand], "test_key")
        assert "<script>&" not in html.split("const CANDIDATES")[0]


class TestValidateGeneratedHtml:
    def test_passes_when_images_exist(self, tmp_path) -> None:
        frames = tmp_path / "frames"
        frames.mkdir()
        cv2.imwrite(str(frames / "c18_t100.00_2P_burst_board_crop.png"), np.zeros((720, 384, 3), dtype=np.uint8))
        cv2.imwrite(str(frames / "c18_t100.00_2P_burst_full.png"), np.zeros((1080, 1920, 3), dtype=np.uint8))
        cand = _sample_candidate()
        html = render_html_document([cand], "test_key")
        html_path = tmp_path / "label_tool_v3.html"
        html_path.write_text(html, encoding="utf-8")
        validate_generated_html(html_path, [cand], tmp_path)  # 例外なし

    def test_raises_when_image_missing(self, tmp_path) -> None:
        cand = _sample_candidate()
        html = render_html_document([cand], "test_key")
        html_path = tmp_path / "label_tool_v3.html"
        html_path.write_text(html, encoding="utf-8")
        try:
            validate_generated_html(html_path, [cand], tmp_path)
            raised = False
        except AssertionError:
            raised = True
        assert raised
