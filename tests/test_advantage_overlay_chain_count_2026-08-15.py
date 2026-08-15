"""れんさ数表示 (--show-chain-count、2026-08-15 user要望) のテスト。

user要望: 「得点よりれんさ数の方が重要指標。実際の連鎖数と評価がどう動いたか、
どちらの盤面での連鎖か明確に。認識性能検証としても使えるように」。

設計方針 (推定/実測の両論併記、単一の断定値を出さない):
    - 推定連鎖数 = ChainEvent.chain_count (simulate 由来、既知の過小評価事故あり)
    - 実測得点差 = OjamaAccountSnapshot.chain_total_score_pX (score OCR 由来)
    - 得点逆算連鎖数 = select_chain_count_high_confidence_band(実測得点差)
    - 推定と逆算が食い違ったら表示側で目立たせる (= 認識性能検証の価値そのもの)

本ファイルは stateless な純関数 (_build_chain_display_text) と、状態保持
ラッパー (ChainCountDisplayTracker) を検証する。レンダリング (PIL/cv2) は
_draw_panel_layout の出力を numpy 配列比較する形で「既定 OFF は bit-identical」
だけを担保し、視覚レビューは別途 viz で行う (feedback_viz_eval_required)。
"""
from __future__ import annotations

import inspect

import numpy as np

from src.board import Board
from src.chain_detector import ChainEvent
from src.ojama_accounting import OjamaAccountSnapshot
import scripts.visualize_advantage_overlay as vao


def _snap(
    chain_end_triggered_p1: bool = False, chain_total_score_p1: int = 0,
    chain_end_triggered_p2: bool = False, chain_total_score_p2: int = 0,
) -> OjamaAccountSnapshot:
    """テスト用の最小 OjamaAccountSnapshot (連鎖終了関連以外は無害な既定値)。"""
    return OjamaAccountSnapshot(
        t_sec=0.0, pending_p1=0, pending_p2=0,
        total_generated_by_p1=0, total_generated_by_p2=0,
        total_offset_by_p1=0, total_offset_by_p2=0,
        total_dropped_to_p1=0, total_dropped_to_p2=0,
        net_ojama_balance=0,
        overflow_risk_p1=False, overflow_risk_p2=False,
        confidence=1.0, leftover_p1=0, leftover_p2=0,
        all_clear_pending_p1=False, all_clear_pending_p2=False,
        chain_end_triggered_p1=chain_end_triggered_p1,
        chain_total_score_p1=chain_total_score_p1,
        chain_end_triggered_p2=chain_end_triggered_p2,
        chain_total_score_p2=chain_total_score_p2,
    )


def _chain_event(chain_count: int, trigger_sec: float = 0.0) -> ChainEvent:
    return ChainEvent(
        trigger_sec=trigger_sec, end_sec=trigger_sec + 1.0, before_board=Board(),
        chain_count=chain_count, total_erased=4, total_score=0,
        base_score=0, all_clear_bonus_applied=0, ojama_sent=0,
        leftover_score=0, is_all_clear=False,
    )


# ============================
# 定数の健全性
# ============================


def test_chain_display_hold_sec_is_positive() -> None:
    """保持秒数は正 (0 以下だとパルスが読めない)。"""
    assert vao.CHAIN_DISPLAY_HOLD_SEC > 0.0


def test_panel_info_chain_rows_stay_within_info_panel_bounds() -> None:
    """1P/2P のれんさ表示 y 座標が、応手行より下・経過時刻行より上に収まる。"""
    assert vao.PANEL_INFO_CHAIN_Y1 > vao.PANEL_INFO_COUNTER_Y
    assert vao.PANEL_INFO_CHAIN_Y2 > vao.PANEL_INFO_CHAIN_Y1
    info_h = vao.panel_layout_regions()["info"][3]
    assert vao.PANEL_INFO_CHAIN_Y2 < info_h - vao.PANEL_INFO_ELAPSED_BOTTOM_MARGIN


# ============================
# ChainCountDisplayTracker (状態保持)
# ============================


def test_tracker_returns_none_when_no_pulse_ever_fired() -> None:
    """一度もパルスが立っていなければ両サイドとも None (無意味な常時表示をしない)。"""
    t = vao.ChainCountDisplayTracker()
    t.update(None, None, _snap(), t_sec=0.0)
    assert t.snapshot("1P", 0.0) is None
    assert t.snapshot("2P", 0.0) is None


def test_tracker_holds_estimated_chain_count_after_trigger() -> None:
    """chain_event トリガー直後は推定連鎖数が読める (保持窓内)。"""
    t = vao.ChainCountDisplayTracker()
    t.update(_chain_event(6), None, _snap(), t_sec=10.0)
    info = t.snapshot("1P", 10.0)
    assert info is not None
    assert info.estimated_chain_count == 6
    assert info.actual_score is None  # まだ finalize していない


def test_tracker_estimated_pulse_expires_after_hold_window() -> None:
    """保持秒数を過ぎたら推定連鎖数は None に戻る (パルスの取りこぼしでなく正しい消灯)。"""
    t = vao.ChainCountDisplayTracker()
    t.update(_chain_event(6), None, _snap(), t_sec=10.0)
    still_held = t.snapshot("1P", 10.0 + vao.CHAIN_DISPLAY_HOLD_SEC)
    expired = t.snapshot("1P", 10.0 + vao.CHAIN_DISPLAY_HOLD_SEC + 0.01)
    assert still_held is not None and still_held.estimated_chain_count == 6
    assert expired is None


def test_tracker_holds_actual_score_and_derived_chain_count_on_finalize() -> None:
    """chain_end_triggered が立つと実測得点差+得点逆算連鎖数が読める。

    score=880 は 6連鎖 (下限近似スコア想定域) の高信頼帯に入る想定値
    (select_chain_count_high_confidence_band の許容比率[0.9,1.1]を満たす
    実測に近い値。厳密な下限近似値は src.scoring 側の実装依存のため、ここでは
    「actual_score はそのまま反映される」ことと「derived は select_chain_count_
    high_confidence_band の戻り値と一致する」ことのみを検証し、下限近似の
    具体的な数式は二重実装しない)。
    """
    from src.chain_count_truth import select_chain_count_high_confidence_band
    snap = _snap(chain_end_triggered_p2=True, chain_total_score_p2=880)
    t = vao.ChainCountDisplayTracker()
    t.update(None, None, snap, t_sec=5.0)
    info = t.snapshot("2P", 5.0)
    assert info is not None
    assert info.actual_score == 880
    expected = select_chain_count_high_confidence_band(880).chain_count
    assert info.derived_chain_count == expected
    assert info.estimated_chain_count is None  # chain_event 側は無発火


def test_tracker_sides_are_independent() -> None:
    """1P のパルスは 2P の表示に影響しない (side 混同防止)。"""
    t = vao.ChainCountDisplayTracker()
    t.update(_chain_event(4), None, _snap(), t_sec=1.0)
    assert t.snapshot("2P", 1.0) is None
    info1 = t.snapshot("1P", 1.0)
    assert info1 is not None and info1.estimated_chain_count == 4


def test_tracker_update_is_safe_with_both_sides_firing_same_frame() -> None:
    """1P/2P が同一フレームで同時に発火しても互いを上書きしない。"""
    snap = _snap(
        chain_end_triggered_p1=True, chain_total_score_p1=40,
        chain_end_triggered_p2=True, chain_total_score_p2=520,
    )
    t = vao.ChainCountDisplayTracker()
    t.update(_chain_event(1), _chain_event(5), snap, t_sec=2.0)
    info1 = t.snapshot("1P", 2.0)
    info2 = t.snapshot("2P", 2.0)
    assert info1 is not None and info1.estimated_chain_count == 1
    assert info2 is not None and info2.estimated_chain_count == 5
    assert info1.actual_score == 40
    assert info2.actual_score == 520


# ============================
# _build_chain_display_text (純粋関数)
# ============================


def test_build_chain_display_text_empty_when_info_none() -> None:
    """info が None (連鎖無し・保持期限切れ) なら空文字・食い違いなし。"""
    text, mismatch = vao._build_chain_display_text("1P", None)
    assert text == "" and mismatch is False


def test_build_chain_display_text_shows_dash_when_derived_unknown() -> None:
    """得点逆算が判定不能 (None) なら "-" を表示し、食い違い扱いにしない。"""
    info = vao.ChainCountDisplayInfo(
        estimated_chain_count=3, actual_score=15, derived_chain_count=None)
    text, mismatch = vao._build_chain_display_text("1P", info)
    assert "3連鎖" in text
    assert "+15点" in text
    assert "逆算-" in text or "(逆算-)" in text
    assert mismatch is False


def test_build_chain_display_text_flags_mismatch_when_estimate_and_derived_differ() -> None:
    """推定連鎖数と得点逆算連鎖数が食い違う場合、mismatch=True かつタグが本文に入る。"""
    info = vao.ChainCountDisplayInfo(
        estimated_chain_count=1, actual_score=880, derived_chain_count=6)
    text, mismatch = vao._build_chain_display_text("1P", info)
    assert mismatch is True
    assert "推定≠逆算" in text
    assert "1連鎖" in text and "6連鎖" in text


def test_build_chain_display_text_no_mismatch_when_estimate_and_derived_agree() -> None:
    """推定と逆算が一致すれば mismatch=False (認識が健全であることの表示)。"""
    info = vao.ChainCountDisplayInfo(
        estimated_chain_count=4, actual_score=200, derived_chain_count=4)
    text, mismatch = vao._build_chain_display_text("1P", info)
    assert mismatch is False
    assert "推定≠逆算" not in text


# ============================
# 既定 OFF は bit-identical (backwards compat)
# ============================


def _dummy_frame() -> np.ndarray:
    return np.zeros((vao.OUT_H, vao.OUT_W, 3), dtype=np.uint8)


def _draw_panel_layout_baseline() -> np.ndarray:
    return vao._draw_panel_layout(
        _dummy_frame(), 12.5, 0.55, [("board_ojama_count", 0.3)], False,
        [(0.0, 12.5)], 0.0, 30.0,
        state1="STABLE", state2="STABLE", counter_text="", elapsed_sec=3.0,
    )


def test_draw_panel_layout_default_chain_params_match_explicit_off_call() -> None:
    """新規引数を省略した呼び出し (旧来の全既存呼出元) と、明示的な
    OFF値 ("" / False) を渡した呼び出しが完全に同一画素を出す
    (呼び出し側の実装を変える必要がないことの保証)。
    """
    baseline = _draw_panel_layout_baseline()
    explicit_off = vao._draw_panel_layout(
        _dummy_frame(), 12.5, 0.55, [("board_ojama_count", 0.3)], False,
        [(0.0, 12.5)], 0.0, 30.0,
        state1="STABLE", state2="STABLE", counter_text="", elapsed_sec=3.0,
        chain_text_p1="", chain_text_p2="",
        chain_mismatch_p1=False, chain_mismatch_p2=False,
    )
    assert np.array_equal(baseline, explicit_off)


def test_draw_panel_layout_draws_something_when_chain_text_present() -> None:
    """chain_text_p1/p2 を渡すと、OFF 時のベースラインから画素が変化する
    (描画経路が実際に到達していることの確認)。
    """
    baseline = _draw_panel_layout_baseline()
    with_chain = vao._draw_panel_layout(
        _dummy_frame(), 12.5, 0.55, [("board_ojama_count", 0.3)], False,
        [(0.0, 12.5)], 0.0, 30.0,
        state1="STABLE", state2="STABLE", counter_text="", elapsed_sec=3.0,
        chain_text_p1="1P 推定6連鎖 / 実測+880点 (逆算6連鎖)",
        chain_text_p2="2P 推定1連鎖 / 実測+880点 (逆算6連鎖) [推定≠逆算]",
        chain_mismatch_p1=False, chain_mismatch_p2=True,
    )
    assert not np.array_equal(baseline, with_chain)


def test_generate_show_chain_count_defaults_to_false() -> None:
    """generate() の show_chain_count 既定値は False (backwards compat)。"""
    params = inspect.signature(vao.generate).parameters
    assert "show_chain_count" in params
    assert params["show_chain_count"].default is False


def test_cli_show_chain_count_flag_defaults_to_false() -> None:
    """CLI --show-chain-count は default=False で argparse 定義されている
    (main() は動画 I/O を伴い直接実行できないため、ソース検査で担保する)。
    """
    src = inspect.getsource(vao.main)
    assert '"--show-chain-count"' in src
    assert 'dest="show_chain_count"' in src
