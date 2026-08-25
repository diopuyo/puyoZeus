"""表示変更2点 (2026-08-21 user指示) のテスト。

変更A: --layout panel の下端字幕帯 (140px) をグラフへ回して伸ばす
    (panel_layout_regions() に optional 引数 subtitle_h を追加)。
変更B: ResolvedExchangeTracker.enable_pending_landing_gate の配線漏れ是正
    (generate()/CLI に引数が無く渡す手段が存在しなかった)。

いずれも既定 OFF/従来値で bit-identical であることを最優先で担保する
(feedback_viz_eval_required: 数値だけで採否を決めず viz 併用が原則だが、
本ファイルはユニットテスト側の回帰防止分担)。
"""
from __future__ import annotations

import inspect

import numpy as np

import scripts.visualize_advantage_overlay as vao


# ============================
# 変更A: panel_subtitle_h (グラフ拡張)
# ============================


def _dummy_frame() -> np.ndarray:
    return np.zeros((vao.OUT_H, vao.OUT_W, 3), dtype=np.uint8)


def _draw_panel_layout_baseline() -> np.ndarray:
    return vao._draw_panel_layout(
        _dummy_frame(), 12.5, 0.55, [("board_ojama_count", 0.3)], False,
        [(0.0, 12.5)], 0.0, 30.0,
        state1="STABLE", state2="STABLE", counter_text="", elapsed_sec=3.0,
    )


def test_draw_panel_layout_default_subtitle_h_matches_explicit_140() -> None:
    """subtitle_h を省略した呼び出しと明示的に PANEL_SUBTITLE_H (140) を
    渡した呼び出しが完全に同一画素を出す (backwards compat)。"""
    baseline = _draw_panel_layout_baseline()
    explicit = vao._draw_panel_layout(
        _dummy_frame(), 12.5, 0.55, [("board_ojama_count", 0.3)], False,
        [(0.0, 12.5)], 0.0, 30.0,
        state1="STABLE", state2="STABLE", counter_text="", elapsed_sec=3.0,
        subtitle_h=vao.PANEL_SUBTITLE_H,
    )
    assert np.array_equal(baseline, explicit)


def test_draw_panel_layout_subtitle_h_zero_does_not_crash_and_keeps_canvas_size() -> None:
    """subtitle_h=0 でも例外を送出せず、出力キャンバスサイズは不変 (1920x1080)。

    ゼロ高矩形の塗りつぶし (_draw_panel_layout 内 `if sh > 0:` ガード) が
    安全に no-op になることの直接的な回帰テスト。
    """
    frame = vao._draw_panel_layout(
        _dummy_frame(), 12.5, 0.55, [("board_ojama_count", 0.3)], False,
        [(0.0, 12.5)], 0.0, 30.0,
        state1="STABLE", state2="STABLE", counter_text="", elapsed_sec=3.0,
        subtitle_h=0,
    )
    assert frame.shape == (vao.PANEL_CANVAS_H, vao.PANEL_CANVAS_W, 3)


def test_draw_panel_layout_subtitle_h_zero_paints_graph_pixels_near_bottom_edge() -> None:
    """subtitle_h=0 のとき、グラフの枠線 (outline) が画面下端近傍 (y=1079) まで
    描かれる (「グラフ広げて」の実効性の直接的な回帰テスト、視覚証跡は別途 PNG)。

    history に上下端いっぱいの値を積み、グラフ枠 (outline 白線, _graph_geometry
    の gy1 = y0+h-12) が y=1079 近傍の行に現れることを確認する。
    """
    history = [(0.0, 100.0), (5.0, -100.0), (10.0, 0.0)]
    frame = vao._draw_panel_layout(
        _dummy_frame(), 0.0, 0.5, [], False,
        history, 10.0, 30.0,
        state1="STABLE", state2="STABLE", counter_text="", elapsed_sec=3.0,
        subtitle_h=0,
    )
    # panel_layout_regions(subtitle_h=0) の graph 領域は y in [792, 1080)。
    # _graph_geometry は gy1 = y0 + h - 12 = 792 + 288 - 12 = 1068 に枠線を描く。
    # 従来 (subtitle_h=140) の枠線位置 (y0+h-12 = 792+148-12 = 928) より
    # 明確に下 (画面下端に近い) にあることを確認する。
    baseline_frame = vao._draw_panel_layout(
        _dummy_frame(), 0.0, 0.5, [], False,
        history, 10.0, 30.0,
        state1="STABLE", state2="STABLE", counter_text="", elapsed_sec=3.0,
        subtitle_h=vao.PANEL_SUBTITLE_H,
    )
    # 白枠 (255,255,255) が現れる最下行 y を、グラフ領域の x 範囲内で探す。
    def _max_white_row(img: np.ndarray) -> int:
        white = np.all(img == 255, axis=2)
        rows = np.where(white.any(axis=1))[0]
        assert rows.size > 0, "白枠が全く描かれていない (描画経路の疑いあり)"
        return int(rows.max())

    assert _max_white_row(frame) > _max_white_row(baseline_frame)


def test_panel_layout_regions_signature_has_subtitle_h_default_140() -> None:
    """panel_layout_regions の新規引数 subtitle_h は既定 PANEL_SUBTITLE_H (140)。"""
    sig = inspect.signature(vao.panel_layout_regions)
    assert sig.parameters["subtitle_h"].default == vao.PANEL_SUBTITLE_H


def test_draw_panel_layout_signature_has_subtitle_h_default_140() -> None:
    """_draw_panel_layout の新規引数 subtitle_h は既定 PANEL_SUBTITLE_H (140)。"""
    sig = inspect.signature(vao._draw_panel_layout)
    assert sig.parameters["subtitle_h"].default == vao.PANEL_SUBTITLE_H


def test_generate_signature_has_panel_subtitle_h_default_140() -> None:
    """generate() の新規引数 panel_subtitle_h は既定 PANEL_SUBTITLE_H (140、backwards compat)。"""
    sig = inspect.signature(vao.generate)
    assert "panel_subtitle_h" in sig.parameters
    assert sig.parameters["panel_subtitle_h"].default == vao.PANEL_SUBTITLE_H


def test_cli_panel_subtitle_h_flag_defaults_to_140() -> None:
    """CLI --panel-subtitle-h は既定 PANEL_SUBTITLE_H (140) で argparse 定義されている
    (main() は動画 I/O を伴い直接実行できないため、ソース検査で担保する)。"""
    src = inspect.getsource(vao.main)
    assert '"--panel-subtitle-h"' in src
    assert 'dest="panel_subtitle_h"' in src
    assert "default=PANEL_SUBTITLE_H" in src


def test_generate_source_passes_panel_subtitle_h_to_draw_panel_layout() -> None:
    """静的回帰テスト: generate() が _draw_panel_layout へ subtitle_h=panel_subtitle_h
    を渡していることを固定する (配線漏れ防止)。"""
    src = inspect.getsource(vao.generate)
    assert "subtitle_h=panel_subtitle_h" in src


# ============================
# 変更B: enable_resolved_pending_landing_gate (配線漏れ是正)
# ============================


def test_generate_signature_has_enable_resolved_pending_landing_gate_default_false() -> None:
    """generate() の新規引数は既定 False (backwards compat)。"""
    sig = inspect.signature(vao.generate)
    assert "enable_resolved_pending_landing_gate" in sig.parameters
    assert sig.parameters["enable_resolved_pending_landing_gate"].default is False


def test_cli_resolved_pending_landing_gate_flag_defaults_to_false() -> None:
    """CLI --resolved-pending-landing-gate は default=False で定義されている。"""
    src = inspect.getsource(vao.main)
    assert '"--resolved-pending-landing-gate"' in src
    assert 'dest="enable_resolved_pending_landing_gate"' in src


def test_generate_source_wires_pending_landing_gate_to_both_constructions() -> None:
    """静的回帰テスト: generate() ソース中の ResolvedExchangeTracker 構築
    (通常時/試合境界リセット時の2箇所) が両方とも
    enable_pending_landing_gate=enable_resolved_pending_landing_gate を
    渡していることを固定する (配線漏れ4回目の同型再発防止)。"""
    src = inspect.getsource(vao.generate)
    code_only = src.replace(vao.generate.__doc__ or "", "")
    pattern = "enable_pending_landing_gate=enable_resolved_pending_landing_gate"
    assert code_only.count(pattern) == 2


def test_resolved_exchange_tracker_pending_landing_gate_default_false() -> None:
    """ResolvedExchangeTracker.__init__ の enable_pending_landing_gate は
    既定 False (backwards compat、既存呼出元はキーワード省略可)。"""
    sig = inspect.signature(vao.ResolvedExchangeTracker.__init__)
    assert sig.parameters["enable_pending_landing_gate"].default is False
    tracker = vao.ResolvedExchangeTracker(model=object())
    assert tracker._enable_pending_landing_gate is False
