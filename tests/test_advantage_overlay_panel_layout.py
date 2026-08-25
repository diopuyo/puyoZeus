"""パネルレイアウト (--layout panel、2026-08-10 user指示) の座標計算テスト。

レンダリング (PIL/cv2) は目視レビュー対象なので単体テストの対象外とし、
本ファイルは *座標・スケール計算* (panel_layout_regions / _graph_geometry /
_build_counter_text) のみを検証する (stateless な純関数、副作用なし)。
既存の overlay レイアウト側 (render_area=None) が一切変わっていないことも
併せて確認し、backwards compat を担保する。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402


def test_panel_layout_regions_cover_canvas_without_gap_or_overlap() -> None:
    """video/graph/info/subtitle の4領域が 1920x1080 を隙間・重複なく分割する。"""
    regions = vao.panel_layout_regions()
    vx, vy, vw, vh = regions["video"]
    gx, gy, gw, gh = regions["graph"]
    ix, iy, iw, ih = regions["info"]
    sx, sy, sw, sh = regions["subtitle"]
    # 左列 (video+graph) は幅が揃い、縦に隙間なく連結して上部コンテンツ高を埋める
    assert vw == gw
    assert vx == gx == 0
    assert vy == 0 and gy == vh and gy + gh == vao.PANEL_CONTENT_H
    # 右列 (info) は左列の右端から始まり、上部コンテンツ高だけを占める
    assert ix == vw == gw
    assert iy == 0 and ih == vao.PANEL_CONTENT_H
    assert ix + iw == vao.PANEL_CANVAS_W
    # 字幕帯は下端で全幅・上部コンテンツの直下から始まりキャンバス下端まで
    assert sx == 0 and sw == vao.PANEL_CANVAS_W
    assert sy == vao.PANEL_CONTENT_H and sy + sh == vao.PANEL_CANVAS_H


def test_panel_layout_regions_default_matches_legacy_values() -> None:
    """既定呼び出し (subtitle_h指定なし) は従来の4領域サイズと完全一致する

    (2026-08-21 グラフ拡張のための subtitle_h 引数追加。受け入れ条件#2:
    既定の呼び出しが従来値と完全一致することを保証する回帰テスト)。
    """
    regions = vao.panel_layout_regions()
    assert regions["video"] == (0, 0, 1408, 792)
    assert regions["graph"] == (0, 792, 1408, 148)
    assert regions["info"] == (1408, 0, 512, 940)
    assert regions["subtitle"] == (0, 940, 1920, 140)


def test_panel_layout_regions_subtitle_h_zero_covers_canvas_without_gap() -> None:
    """subtitle_h=0 でも4領域が 1920x1080 を隙間・重複なく分割する

    (2026-08-21 「グラフ広げて」対応。字幕帯を無くした分は左下グラフと
    右の情報パネルへ丸ごと回る)。
    """
    regions = vao.panel_layout_regions(subtitle_h=0)
    vx, vy, vw, vh = regions["video"]
    gx, gy, gw, gh = regions["graph"]
    ix, iy, iw, ih = regions["info"]
    sx, sy, sw, sh = regions["subtitle"]
    # 字幕帯は高さ0 (存在しない) になる
    assert sh == 0
    assert sy == vao.PANEL_CANVAS_H
    # グラフ・情報パネルはキャンバス下端まで伸びる
    assert gy + gh == vao.PANEL_CANVAS_H
    assert iy + ih == vao.PANEL_CANVAS_H
    assert gh == 288  # 148 + 140 (字幕帯分がそのまま加算される)
    assert ih == 1080  # 940 + 140
    # 隙間・重複なく分割する不変条件は subtitle_h に関わらず維持される
    assert vw == gw and vx == gx == 0
    assert vy == 0 and gy == vh
    assert ix == vw == gw and iy == 0
    assert ix + iw == vao.PANEL_CANVAS_W
    assert sx == 0 and sw == vao.PANEL_CANVAS_W


def test_panel_layout_video_region_keeps_16_9_aspect() -> None:
    """左上映像領域は元動画と同じ 16:9 を維持する (引き伸ばし歪み防止)。"""
    vx, vy, vw, vh = vao.panel_layout_regions()["video"]
    assert vw == 1408 and vh == 792
    assert math.isclose(vw / vh, 16 / 9, rel_tol=1e-6)


def test_panel_layout_subtitle_band_has_no_overlap_with_content() -> None:
    """字幕帯は video/graph/info のどの矩形とも y 方向で重ならない (無描画保証の前提)。"""
    regions = vao.panel_layout_regions()
    sx, sy, sw, sh = regions["subtitle"]
    for name in ("video", "graph", "info"):
        x, y, w, h = regions[name]
        assert y + h <= sy, f"{name} が字幕帯 y={sy} と重なる (y+h={y + h})"


def test_graph_geometry_overlay_mode_matches_legacy_formula() -> None:
    """render_area=None (overlay既定) は従来の TOP_H/OUT_W/OUT_H/CANVAS_H 由来の値と一致。"""
    gx0, gx1, gy0, gy1, title_y = vao._graph_geometry(None)
    game_bottom = vao.TOP_H + vao.OUT_H
    assert (gx0, gx1) == (40, vao.OUT_W - 40)
    assert (gy0, gy1) == (game_bottom + 26, vao.CANVAS_H - 12)
    assert title_y == gy0 - 20


def test_graph_geometry_panel_mode_stays_within_given_box() -> None:
    """render_area 指定時、算出される gx0/gx1/gy0/gy1 は矩形内に収まる。"""
    box = vao.panel_layout_regions()["graph"]
    x0, y0, w, h = box
    gx0, gx1, gy0, gy1, title_y = vao._graph_geometry(box)
    assert x0 <= gx0 < gx1 <= x0 + w
    assert y0 <= title_y < gy0 < gy1 <= y0 + h


def test_build_counter_text_empty_when_nan() -> None:
    """counter-reach 未計算 (nan) 時は空文字 (行を描かない)。"""
    assert vao._build_counter_text(float("nan"), float("nan")) == ""


def test_build_counter_text_formats_percentages_when_available() -> None:
    """有効時は両者の応手確率を % 表示する。"""
    text = vao._build_counter_text(0.3, 0.8)
    assert "30%" in text and "80%" in text
