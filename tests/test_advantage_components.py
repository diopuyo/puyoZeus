"""有利不利オーバーレイの合成成分(圧力/得点リード)の単体テスト。

PressureTracker(着弾ダメージの持続記憶)と ScoreLeadTracker(得点リード)は
純ロジックなので認識なしで検証できる。回帰防止用。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import types

import numpy as np
import pytest

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, Board,
)
from src.board_state_machine import BoardState  # noqa: E402
from src.chain_detector import ChainEvent  # noqa: E402
from src.indicators_v2 import IndicatorV2Value  # noqa: E402
from src.ojama_accounting import CHAIN_TOTAL_MIN_SCORE  # noqa: E402
from src.probability_calibration import (  # noqa: E402
    PhaseCalibrationParams, PlattCalibrationParams,
)
from scripts.visualize_advantage_overlay import (  # noqa: E402
    PressureTracker, ScoreLeadTracker, RealtimeForecastTracker, adv_to_winprob,
    kill_override, board_room, _detect_score_reset, _apply_platt_to_display,
    EarlyFireTracker, EARLY_FIRE_CAP,
    _match_progress_for_boards, _resolve_display_platt,
)


def test_kill_no_effect_when_survivable() -> None:
    """降る量が受け容量に対し小さければ有利不利は不変(通常の攻めは上書きしない)。"""
    assert kill_override(20.0, inc1=10.0, inc2=0.0, room1=60, room2=60) == 20.0


def test_kill_pushes_to_survivor_when_1p_lethal() -> None:
    """1Pに致死量(pending≫空き)が降る → 有利不利は2P側(負)へ強制。"""
    v = kill_override(30.0, inc1=200.0, inc2=0.0, room1=8, room2=60)
    assert v <= -90.0  # ほぼ完全上書き(2P勝ち)


def test_kill_pushes_to_survivor_when_2p_lethal() -> None:
    """2Pに致死量 → 1P側(正)へ。"""
    v = kill_override(-30.0, inc1=0.0, inc2=200.0, room1=60, room2=8)
    assert v >= 90.0


def test_kill_ignores_small_attack() -> None:
    """1ターンで捌ける小さな攻め(pending<40)は空きを超えても致死扱いしない。"""
    assert kill_override(15.0, inc1=25.0, inc2=0.0, room1=12, room2=40) == 15.0


def test_kill_symmetric_mutual_no_override() -> None:
    """双方が同程度に致死(相打ち)なら致死度差が小さく上書きしない。"""
    assert kill_override(5.0, inc1=200.0, inc2=200.0, room1=8, room2=8) == 5.0


def test_board_room_full_and_empty() -> None:
    """空盤面は容量72、埋まった盤面は容量0(row0=隠し段は除外)。"""
    empty = types.SimpleNamespace(_grid=np.zeros((13, 6), dtype=np.uint8))
    full = types.SimpleNamespace(_grid=np.ones((13, 6), dtype=np.uint8))
    assert board_room(empty) == 72
    assert board_room(full) == 0
    assert board_room(None) == 72


def test_winprob_even_is_half() -> None:
    """有利不利0 → 勝率50%(較正sigmoid・直線どちらも対称で成立)。"""
    assert abs(adv_to_winprob(0.0) - 0.5) < 1e-6


def test_winprob_monotonic() -> None:
    """有利不利が上がるほど1P勝率も上がる(単調増加)。"""
    assert adv_to_winprob(-80) < adv_to_winprob(-20) < adv_to_winprob(20) < adv_to_winprob(80)


def test_winprob_range() -> None:
    """勝率は [0,1] に収まる。"""
    for a in (-100.0, -50.0, 0.0, 50.0, 100.0):
        assert 0.0 <= adv_to_winprob(a) <= 1.0


def test_forecast_opponent_attack_negative() -> None:
    """2P が発火(score増)→ 1Pへの pending 増 → 信号は負(2P有利)。"""
    ft = RealtimeForecastTracker()
    ft.update(0, 0, 0, 0)
    assert ft.update(0, 8000, 0, 0) < 0.0


def test_forecast_self_attack_positive() -> None:
    """1P が発火(score増)→ 2Pへの pending 増 → 信号は正(1P有利)。"""
    ft = RealtimeForecastTracker()
    ft.update(0, 0, 0, 0)
    assert ft.update(8000, 0, 0, 0) > 0.0


def test_forecast_game_reset_clears() -> None:
    """スコア大幅減(試合境界)で pending がクリアされ信号がほぼ0に戻る。"""
    ft = RealtimeForecastTracker()
    ft.update(0, 0, 0, 0)
    ft.update(0, 8000, 0, 0)  # 2P攻撃で偏り
    v = ft.update(0, 0, 0, 0)  # score 8000→0 のリセット
    assert abs(v) < 1.0


def test_forecast_clamped_range() -> None:
    """予告信号は [-100, 100] に収まる。"""
    ft = RealtimeForecastTracker()
    ft.update(0, 0, 0, 0)
    v = ft.update(0, 999999, 0, 0)
    assert -100.0 <= v <= 100.0


def test_forecast_delivery_drains_pending() -> None:
    """1Pが発火→2Pへ pending。2Pがツモを重ねる(配送)と pending が減り信号が0へ寄る。"""
    ft = RealtimeForecastTracker()
    ft.update(0, 0, 0, 0)
    v0 = ft.update(700, 0, 0, 0)  # 1P発火 → 2Pに pending ~10個 → 正
    assert v0 > 0.0
    v1 = v0
    for k in range(1, 6):  # 2Pがツモを置く=30個/ターン配送で pending が捌ける
        v1 = ft.update(700, 0, 0, k)
    assert abs(v1) < abs(v0)  # 配送で pending 減 → 信号が0へ


def test_forecast_offset_by_counter() -> None:
    """2P発火で1Pに pending。1Pが同等以上を発火(相殺)すると 1P pending が減る。"""
    ft = RealtimeForecastTracker()
    ft.update(0, 0, 0, 0)
    neg = ft.update(0, 7000, 0, 0)  # 2P発火 → 1Pに pending ~100 → 強い負
    after = ft.update(7000, 7000, 0, 0)  # 1Pも同等発火で相殺
    assert after > neg  # 相殺で 1P pending 減 → 信号が負から回復


def test_pressure_zero_when_no_ojama() -> None:
    """お邪魔が両者0のままなら圧力は0。"""
    pt = PressureTracker()
    for _ in range(10):
        assert pt.update(0.0, 0.0) == 0.0


def test_pressure_positive_when_opponent_buried() -> None:
    """相手(2P)盤面のお邪魔が増える=1Pが攻撃を通した→圧力は正(1P有利)。"""
    pt = PressureTracker()
    pt.update(0.0, 0.0)
    val = pt.update(0.0, 20.0)  # 2Pに20個着弾
    assert val > 0.0


def test_pressure_negative_when_self_buried() -> None:
    """自分(1P)盤面のお邪魔が増える→圧力は負(2P有利)。"""
    pt = PressureTracker()
    pt.update(0.0, 0.0)
    val = pt.update(20.0, 0.0)
    assert val < 0.0


def test_pressure_decays_toward_zero() -> None:
    """着弾後、追加が無ければ圧力は減衰して0へ近づく。"""
    pt = PressureTracker()
    pt.update(0.0, 0.0)
    peak = pt.update(0.0, 30.0)
    later = peak
    for _ in range(200):
        later = pt.update(0.0, 30.0)  # 増分なし(同じ値)→減衰のみ
    assert abs(later) < abs(peak)


def test_pressure_clamped_range() -> None:
    """圧力は [-100, 100] に収まる。"""
    pt = PressureTracker()
    pt.update(0.0, 0.0)
    v = pt.update(0.0, 72.0)
    assert -100.0 <= v <= 100.0


def test_score_lead_sign() -> None:
    """スコアが高い側が有利(正=1P)。"""
    lt = ScoreLeadTracker()
    assert lt.update(10000, 2000) > 0.0
    lt2 = ScoreLeadTracker()
    assert lt2.update(2000, 10000) < 0.0


def test_score_lead_even() -> None:
    """同点なら0。"""
    lt = ScoreLeadTracker()
    assert lt.update(5000, 5000) == 0.0


def test_score_lead_none_keeps_last() -> None:
    """score None は直前値を保持する。"""
    lt = ScoreLeadTracker()
    lt.update(10000, 2000)
    v = lt.update(None, None)  # 保持 → 同じ符号
    assert v > 0.0


def test_score_lead_clamped_range() -> None:
    """得点リードは [-100, 100] に収まる。"""
    lt = ScoreLeadTracker()
    v = lt.update(999999, 0)
    assert -100.0 <= v <= 100.0


def test_score_reset_detects_large_drop() -> None:
    """前フレーム比で片側スコアが大幅減少(新ゲーム開始等)→ リセット検知。"""
    assert _detect_score_reset(0, 3000, 12000, 3000) is True


def test_score_reset_detects_both_near_zero() -> None:
    """両者スコアが0付近(全消し直後/試合最初期)→ リセット検知。"""
    assert _detect_score_reset(0, 5, None, None) is True


def test_score_reset_ignores_normal_progress() -> None:
    """通常の得点増加(大幅減少なし、0付近でもない)→ リセット検知しない。"""
    assert _detect_score_reset(4200, 3100, 4000, 3000) is False


def test_score_reset_ignores_one_sided_near_zero() -> None:
    """片方だけ0付近(もう片方は進行中)→ リセット検知しない(誤爆防止)。"""
    assert _detect_score_reset(0, 8000, 0, 6000) is False


def test_score_reset_none_score_is_undetectable() -> None:
    """score が None(OCR失敗)の場合は判定不能として False を返す。"""
    assert _detect_score_reset(None, 5000, 3000, 5000) is False


def test_platt_display_noop_when_params_none() -> None:
    """platt_params=None (校正無効/フラグOFF相当) なら (adv, p1) は完全不変。

    これが「フラグOFFで旧挙動が完全に再現される」ことの直接的な回帰テスト。
    """
    adv, p1 = _apply_platt_to_display(42.0, 0.71, None)
    assert adv == 42.0
    assert p1 == 0.71


def test_platt_display_recomputes_adv_from_calibrated_prob() -> None:
    """校正が有効な場合、p1 が校正され、adv = (校正後p1-0.5)*200 に再構成される。"""
    params = PlattCalibrationParams(a=1.0, b=0.0)  # 恒等変換
    adv, p1 = _apply_platt_to_display(60.0, adv_to_winprob(60.0), params)
    assert abs(p1 - adv_to_winprob(60.0)) < 1e-6  # 恒等変換なのでp1は不変
    assert abs(adv - 60.0) < 1e-3  # adv も往復して不変


def test_platt_display_clamped_to_range() -> None:
    """校正後 adv も [-100,100] にクリップされる。"""
    params = PlattCalibrationParams(a=10.0, b=0.0)  # 極端に強く0/1へ寄せる係数
    adv, p1 = _apply_platt_to_display(90.0, 0.95, params)
    assert -100.0 <= adv <= 100.0
    assert 0.0 <= p1 <= 1.0


# ============================
# 位相別 Platt 選択 (2026-08-11 Phase1-2 追加)
# ============================

def _make_phase_params_for_display() -> PhaseCalibrationParams:
    return PhaseCalibrationParams(phases={
        "序盤": PlattCalibrationParams(a=0.6, b=0.0),
        "中盤": PlattCalibrationParams(a=0.75, b=0.0),
        "終盤": PlattCalibrationParams(a=0.5, b=0.0),
    })


def test_match_progress_for_boards_empty_boards_is_zero() -> None:
    """両者空盤面 (試合開始直後相当) は進行度 0 (序盤) になる。"""
    assert _match_progress_for_boards(Board(), Board()) == 0.0


def test_resolve_display_platt_falls_back_to_common_when_phase_none() -> None:
    """位相別パラメータが無い (None) 場合は従来通り全位相共通の platt_params を返す。"""
    common = PlattCalibrationParams(a=0.8, b=-0.1)
    chosen = _resolve_display_platt(0.5, common, None)
    assert chosen is common


def test_resolve_display_platt_both_none_returns_none() -> None:
    """両方 None (校正無効) なら None を返す (=無変換、後方互換)。"""
    assert _resolve_display_platt(0.5, None, None) is None


def test_resolve_display_platt_prefers_phase_over_common() -> None:
    """位相別パラメータがあれば、共通パラメータより優先して使われる。"""
    common = PlattCalibrationParams(a=0.8, b=-0.1)
    phase_params = _make_phase_params_for_display()
    chosen = _resolve_display_platt(0.1, common, phase_params)
    assert chosen.a == pytest.approx(0.6)  # 序盤の係数


def test_resolve_display_platt_selects_correct_phase_by_progress() -> None:
    """進行度に応じて序盤/中盤/終盤いずれかの係数が選ばれる。"""
    phase_params = _make_phase_params_for_display()
    assert _resolve_display_platt(0.0, None, phase_params).a == pytest.approx(0.6)
    assert _resolve_display_platt(0.5, None, phase_params).a == pytest.approx(0.75)
    assert _resolve_display_platt(0.99, None, phase_params).a == pytest.approx(0.5)


# ============================
# EarlyFireTracker (2026-07-29 userレビュー指摘1/2対処)
# ============================

def _make_chain_event(
    trigger_sec: float, before_board: Board | None = None, total_score: int = 0,
    score_estimated: bool = False,
) -> ChainEvent:
    """テスト用の最小 ChainEvent を組み立てる (before_board/total_score 以外は不使用値でよい)。

    score_estimated: 根治① (W7, 2026-08-13) の充填フラグ。既定 False
    (backwards compat、既存呼び出しは挙動不変)。
    """
    return ChainEvent(
        trigger_sec=trigger_sec, end_sec=trigger_sec + 1.0,
        before_board=before_board if before_board is not None else Board(),
        chain_count=1, total_erased=0, total_score=total_score, base_score=0,
        all_clear_bonus_applied=0, ojama_sent=0, leftover_score=0,
        is_all_clear=False, score_estimated=score_estimated,
    )


def test_early_fire_bias_zero_when_no_events() -> None:
    """chain_event が一度も来なければ bias は 0 のまま。"""
    ft = EarlyFireTracker()
    for _ in range(5):
        v = ft.update(None, None, Board(), Board(), 0.0)
    assert v == 0.0


def test_early_fire_bias_positive_on_1p_fire(monkeypatch) -> None:
    """1P の chain_event 検知 → bias は正 (1P有利) へ即座に動く。"""
    import scripts.visualize_advantage_overlay as vao
    monkeypatch.setattr(vao.iv, "immediate_fire_power",
                        lambda board, elapsed: IndicatorV2Value(score=0.5, raw=30.0))
    monkeypatch.setattr(vao.iv, "ojama_damage",
                        lambda board, ojama: IndicatorV2Value(score=0.5, raw=0.0))
    ft = EarlyFireTracker()
    ev1 = _make_chain_event(trigger_sec=10.0)
    v = ft.update(ev1, None, Board(), Board(), 0.0)
    assert v > 0.0
    assert v <= EARLY_FIRE_CAP


def test_early_fire_bias_negative_on_2p_fire(monkeypatch) -> None:
    """2P の chain_event 検知 → bias は負 (2P有利) へ即座に動く (1P/2P対称)。"""
    import scripts.visualize_advantage_overlay as vao
    monkeypatch.setattr(vao.iv, "immediate_fire_power",
                        lambda board, elapsed: IndicatorV2Value(score=0.5, raw=30.0))
    monkeypatch.setattr(vao.iv, "ojama_damage",
                        lambda board, ojama: IndicatorV2Value(score=0.5, raw=0.0))
    ft = EarlyFireTracker()
    ev2 = _make_chain_event(trigger_sec=10.0)
    v = ft.update(None, ev2, Board(), Board(), 0.0)
    assert v < 0.0
    assert v >= -EARLY_FIRE_CAP


def test_early_fire_same_trigger_not_double_counted(monkeypatch) -> None:
    """同一 trigger_sec の chain_event を複数フレームで受けても二重加算しない。"""
    import scripts.visualize_advantage_overlay as vao
    monkeypatch.setattr(vao.iv, "immediate_fire_power",
                        lambda board, elapsed: IndicatorV2Value(score=0.5, raw=30.0))
    monkeypatch.setattr(vao.iv, "ojama_damage",
                        lambda board, ojama: IndicatorV2Value(score=0.5, raw=0.0))
    ft = EarlyFireTracker()
    ev1 = _make_chain_event(trigger_sec=10.0)
    v0 = ft.update(ev1, None, Board(), Board(), 0.0)
    v1 = ft.update(ev1, None, Board(), Board(), 0.0)  # 同じ ev を再度渡す (毎フレームありうる)
    assert v1 <= v0  # 減衰のみで新規加算はない


def test_early_fire_decays_without_new_events(monkeypatch) -> None:
    """新規発火が無ければ bias は減衰して 0 へ近づく。"""
    import scripts.visualize_advantage_overlay as vao
    monkeypatch.setattr(vao.iv, "immediate_fire_power",
                        lambda board, elapsed: IndicatorV2Value(score=0.5, raw=30.0))
    monkeypatch.setattr(vao.iv, "ojama_damage",
                        lambda board, ojama: IndicatorV2Value(score=0.5, raw=0.0))
    ft = EarlyFireTracker()
    ev1 = _make_chain_event(trigger_sec=10.0)
    peak = ft.update(ev1, None, Board(), Board(), 0.0)
    later = peak
    for _ in range(300):
        later = ft.update(None, None, Board(), Board(), 0.0)
    assert 0.0 <= later < peak


def test_early_fire_on_settled_clears_bias(monkeypatch) -> None:
    """settled 再計算が入ったら on_settled() で bias が即座に 0 になる(二重計上防止)。"""
    import scripts.visualize_advantage_overlay as vao
    monkeypatch.setattr(vao.iv, "immediate_fire_power",
                        lambda board, elapsed: IndicatorV2Value(score=0.5, raw=30.0))
    monkeypatch.setattr(vao.iv, "ojama_damage",
                        lambda board, ojama: IndicatorV2Value(score=0.5, raw=0.0))
    ft = EarlyFireTracker()
    ev1 = _make_chain_event(trigger_sec=10.0)
    ft.update(ev1, None, Board(), Board(), 0.0)
    assert ft.bias != 0.0
    ft.on_settled()
    assert ft.bias == 0.0


def test_early_fire_bias_clamped_to_cap(monkeypatch) -> None:
    """複数回の連続発火でも bias は ±EARLY_FIRE_CAP でクリップされる。"""
    import scripts.visualize_advantage_overlay as vao
    monkeypatch.setattr(vao.iv, "immediate_fire_power",
                        lambda board, elapsed: IndicatorV2Value(score=1.0, raw=999.0))
    monkeypatch.setattr(vao.iv, "ojama_damage",
                        lambda board, ojama: IndicatorV2Value(score=1.0, raw=-99.0))
    ft = EarlyFireTracker()
    for i in range(5):
        v = ft.update(_make_chain_event(trigger_sec=float(i)), None, Board(), Board(), 0.0)
    assert v <= EARLY_FIRE_CAP


def test_early_fire_none_board_is_safe() -> None:
    """opponent/before board が None (未確定) でも例外にならず 0 扱い。"""
    ft = EarlyFireTracker()
    ev1 = _make_chain_event(trigger_sec=1.0)
    v = ft.update(ev1, None, None, None, 0.0)
    assert v == 0.0


# ============================
# ResolvedExchangeTracker (2026-08-13、docs/DEMO_REVIEW_2026-08-13.md #9)
# ============================
# 実動画 (review_demo_2026-08-12.mp4 source t=192-200s、両者発火シーン) で
# 診断したところ、既知バグクラス (OJAMA_FALL/CHAIN 高速振動系) の実害として
# 「chain_count=8 だが total_score=0」の幻連鎖が数秒間 trigger_sec を変えながら
# 再検知され続ける事象を実測した (本フラグの追加が原因ではなく既存の認識層の
# 挙動)。以下は実測で動機づけられた2つの防御 (ノイズ下限ゲート/再決着1回上限)
# を、実認識に依存せず決定論的に検証する。


def _make_snapshot(
    pending_p1: int = 0, pending_p2: int = 0,
    chain_end_triggered_p1: bool = False, chain_end_triggered_p2: bool = False,
    chain_total_score_p1: int = 0, chain_total_score_p2: int = 0,
) -> "OjamaAccountSnapshot":
    """テスト用の最小 OjamaAccountSnapshot を組み立てる (残りは無害な既定値)。"""
    from src.ojama_accounting import OjamaAccountSnapshot

    return OjamaAccountSnapshot(
        t_sec=0.0, pending_p1=pending_p1, pending_p2=pending_p2,
        total_generated_by_p1=0, total_generated_by_p2=0,
        total_offset_by_p1=0, total_offset_by_p2=0,
        total_dropped_to_p1=0, total_dropped_to_p2=0,
        net_ojama_balance=pending_p2 - pending_p1,
        overflow_risk_p1=False, overflow_risk_p2=False, confidence=1.0,
        leftover_p1=0, leftover_p2=0,
        all_clear_pending_p1=False, all_clear_pending_p2=False,
        chain_end_triggered_p1=chain_end_triggered_p1,
        chain_end_triggered_p2=chain_end_triggered_p2,
        chain_total_score_p1=chain_total_score_p1,
        chain_total_score_p2=chain_total_score_p2,
    )


def _make_signal(chain_event: "ChainEvent | None", score: int | None) -> types.SimpleNamespace:
    """テスト用の最小 Signals もどき (ResolvedExchangeTracker が読む2属性のみ)。"""
    return types.SimpleNamespace(chain_event=chain_event, score=score)


def _stub_score_advantage_factory():
    """呼び出し回数を数えつつ決定論的な (adv, p1, drivers) を返すスタブを作る。

    呼び出し順に adv=10,20,30... と単調変化させ、「何回 _score_advantage が
    実行されたか」をテスト側で直接検証できるようにする (再決着の1回上限等)。
    """
    calls: list[int] = []

    def _stub(model, b1, b2, snap, feature_cols=None, attribution_exclude=()) -> tuple:
        calls.append(1)
        n = len(calls)
        return float(n * 10), 0.5 + n * 0.05, []

    return _stub, calls


def test_resolved_inactive_when_only_one_side_fires() -> None:
    """片側のみの発火はトリガー対象外 (early-fire-reaction の領分のまま)。"""
    from scripts.visualize_advantage_overlay import ResolvedExchangeTracker

    tracker = ResolvedExchangeTracker(model=object())
    ev1 = _make_chain_event(trigger_sec=1.0)
    active, just_deactivated = tracker.update(
        _make_signal(ev1, 100), _make_signal(None, 0), _make_snapshot(), 0.0)
    assert active is False
    assert just_deactivated is False


def test_resolved_inactive_when_score_unfilled(monkeypatch) -> None:
    """total_score=0 かつ score_estimated=False (根治①未実装/OFF時、または
    simulate 失敗時の fail-safe 値) は「スコア未計算」としてトリガー対象外
    にする — 実動画で実測した cc=8 score=0 系のノイズへの回帰テスト。

    【訂正 2026-08-13 W7根治③】旧テスト名/docstring は「score=0 の
    ChainEvent = 幻連鎖 (連鎖不在)」と記していたが誤り。実際には
    chain_count=8 (simulate 検証済み、連鎖は実在) でも total_score が
    ハードコード0だった (docs/KNOWN_WEAKNESSES.md W7: 「score未計算」と
    「連鎖不在」の混同)。本テストの意図は「連鎖の実在性」ではなく
    「スコアが計算できていない (score_estimated=False かつ 0)」ことのみを
    ノイズ扱いする、という後者の意味論に正しく寄せた。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=0)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=0)
    assert ev1.score_estimated is False and ev2.score_estimated is False
    active, _ = tracker.update(
        _make_signal(ev1, 0), _make_signal(ev2, 0), _make_snapshot(), 0.0)
    assert active is False
    assert calls == []  # _score_advantage は一度も呼ばれない


def test_resolved_activates_on_minimum_estimated_score(monkeypatch) -> None:
    """根治① (W7, 2026-08-13) が充填する最小の推定スコア (=CHAIN_TOTAL_MIN_SCORE,
    4連結1色1連鎖の最小得点 4×10×1=40) は、既存ゲートを変更しなくても
    素通しで起動する — 根治③のゲート拡張が不要と判断した根拠の直接確認
    (tests/test_scoring.py の不変条件テストと対で W7 根治③の判断を裏付ける)。
    """
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    ev1 = _make_chain_event(
        trigger_sec=1.0, total_score=CHAIN_TOTAL_MIN_SCORE, score_estimated=True)
    ev2 = _make_chain_event(
        trigger_sec=1.0, total_score=CHAIN_TOTAL_MIN_SCORE, score_estimated=True)
    active, _ = tracker.update(
        _make_signal(ev1, CHAIN_TOTAL_MIN_SCORE),
        _make_signal(ev2, CHAIN_TOTAL_MIN_SCORE), _make_snapshot(), 0.0)
    assert active is True
    assert len(calls) == 1


def test_resolved_activates_on_mutual_fire_above_gate(monkeypatch) -> None:
    """両側とも CHAIN_TOTAL_MIN_SCORE 以上で同時発火 → 即座に決着計算する。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    active, just_deactivated = tracker.update(
        _make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert active is True
    assert just_deactivated is False
    assert len(calls) == 1
    assert tracker.hold_adv == 10.0  # スタブの1回目の戻り値


def test_resolved_holds_without_recompute_while_same_events_active(monkeypatch) -> None:
    """同一 ChainEvent が継続する間は毎フレーム呼ばれても再計算しない (ホールド)。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    snap = _make_snapshot()
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), snap, 0.0)
    held = tracker.hold_adv
    for _ in range(10):
        active, _ = tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), snap, 0.0)
        assert active is True
        assert tracker.hold_adv == held  # 保持中は不変
    assert len(calls) == 1  # 決着計算は1回のみ


def test_resolved_deactivates_and_retains_hold_value_when_both_clear(monkeypatch) -> None:
    """両側の chain_event が両方 None に戻ったら deactivate し、最後の決着値を保持する
    (呼出側が adv_ema へ引き継ぐための値)。"""
    import scripts.visualize_advantage_overlay as vao
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    held = tracker.hold_adv
    active, just_deactivated = tracker.update(
        _make_signal(None, 500), _make_signal(None, 300), _make_snapshot(), 0.0)
    assert active is False
    assert just_deactivated is True
    assert tracker.hold_adv == held  # 値そのものは維持 (引き継ぎ用)


def test_resolved_redecide_when_settled_score_exceeds_prediction(monkeypatch) -> None:
    """simulate 過小評価対策: 確定済み連鎖合計得点が予測を超えたら再決着する。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=100)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 100), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert len(calls) == 1
    # 1P の連鎖が settle し、真の得点(2500)が予測(100)を大幅に超過していたと判明。
    snap_settled = _make_snapshot(chain_end_triggered_p1=True, chain_total_score_p1=2500)
    tracker.update(_make_signal(ev1, 100), _make_signal(ev2, 300), snap_settled, 0.0)
    assert len(calls) == 2  # 下限として即時再決着


def test_resolved_redecide_at_most_once_per_side(monkeypatch) -> None:
    """実動画で実測した「同一保持中に同側の settle が繰り返し立つ」異常系でも
    再決着は1回までに抑える (実測に基づく回帰テスト、本文コメント参照)。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=100)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 100), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    snap_a = _make_snapshot(chain_end_triggered_p1=True, chain_total_score_p1=2500)
    tracker.update(_make_signal(ev1, 100), _make_signal(ev2, 300), snap_a, 0.0)
    assert len(calls) == 2
    # 同側の settle が再度 (異常に) 立っても3回目は起きない。
    snap_b = _make_snapshot(chain_end_triggered_p1=True, chain_total_score_p1=9999)
    tracker.update(_make_signal(ev1, 100), _make_signal(ev2, 300), snap_b, 0.0)
    assert len(calls) == 2


def test_resolved_no_redecide_when_settled_score_not_exceeding(monkeypatch) -> None:
    """確定済み得点が予測以下なら再決着しない (下限方式なので上回った時だけ動く)。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=5000)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 5000), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    snap_settled = _make_snapshot(chain_end_triggered_p1=True, chain_total_score_p1=100)
    tracker.update(_make_signal(ev1, 5000), _make_signal(ev2, 300), snap_settled, 0.0)
    assert len(calls) == 1  # 予測(5000) > 確定値(100) のため再決着しない


# ============================
# ホールド解放条件 (検収指摘⑤、2026-08-14)
# ============================
# 両者発火でホールド中、片方が先に連鎖を終えた (chain_event が None化) 瞬間に
# 判定が動く事象の回帰テスト。tracker.update 自体は「両側 chain_event が
# None」まで正しく hold を維持する (AND ゲート) ことを固定し、実際の漏洩点
# だった呼出側の settled 上書きヘルパー (resolved_hold_freezes_settled) も
# 併せて検証する。


def test_resolved_holds_when_only_one_side_ends_first(monkeypatch) -> None:
    """両者発火でホールド後、片方の chain_event だけ None になっても
    ホールドは維持される (deactivate しない) — 検収指摘⑤の核心。"""
    import scripts.visualize_advantage_overlay as vao
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    held = tracker.hold_adv
    # 1P (ev1) の連鎖だけ先に終わり chain_event が None 化、2P (ev2) は継続中。
    active, just_deactivated = tracker.update(
        _make_signal(None, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert active is True  # 部分解放しない
    assert just_deactivated is False
    assert tracker.hold_adv == held  # 保持値も不変


def test_resolved_deactivates_only_when_both_sides_end(monkeypatch) -> None:
    """両側とも chain_event が None に戻ったときだけ deactivate する。"""
    import scripts.visualize_advantage_overlay as vao
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    # 1P だけ終了 → まだ保持。
    active, _ = tracker.update(
        _make_signal(None, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert active is True
    # 続いて 2P も終了 → ここで初めて解放。
    active, just_deactivated = tracker.update(
        _make_signal(None, 500), _make_signal(None, 300), _make_snapshot(), 0.0)
    assert active is False
    assert just_deactivated is True


def test_resolved_hold_freezes_settled_only_while_active() -> None:
    """呼出側 (main ループ) の settled 上書きヘルパー: hold 中のみ True を
    返し、settled 判定 (per_side_settled の片側OR) を素通りさせない
    (検収指摘⑤の実漏洩点)。フラグ無効時/非hold中は常に False = 従来挙動。"""
    from scripts.visualize_advantage_overlay import resolved_hold_freezes_settled

    assert resolved_hold_freezes_settled(True, True) is True
    assert resolved_hold_freezes_settled(True, False) is False
    assert resolved_hold_freezes_settled(False, True) is False
    assert resolved_hold_freezes_settled(False, False) is False


# ============================
# 指摘11: 着弾完了までのホールド延長 (docs/DEMO_REVIEW_2026-08-13.md #11)
# ============================


def _board_with_ojama(n: int) -> Board:
    """可視領域 (隠し段除く) の下段から順に n 個のお邪魔を敷いた Board を返す
    (テスト専用。iv.board_ojama_count は隠し段を数えないため、n をそのまま
    raw 値として使えるよう下段=BOARD_ROWS-1 から積む=窒息列を巻き込まない)。"""
    from src.board import HIDDEN_ROWS

    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    placed = 0
    for row in range(BOARD_ROWS - 1, HIDDEN_ROWS - 1, -1):
        for col in range(BOARD_COLS):
            if placed >= n:
                break
            grid[row][col] = COLOR_OJAMA
            placed += 1
        if placed >= n:
            break
    return Board.from_list(grid)


def _make_signal_full(
    chain_event: "ChainEvent | None", score: "int | None",
    state: BoardState = BoardState.STABLE,
    confirmed_board: "Board | None" = None,
) -> types.SimpleNamespace:
    """`_make_signal` に state/confirmed_board を足した完全版 (着弾完了判定②用)。"""
    return types.SimpleNamespace(
        chain_event=chain_event, score=score, state=state,
        confirmed_board=confirmed_board if confirmed_board is not None else Board(),
    )


def test_landing_complete_true_immediately_when_pending_already_drained() -> None:
    """既存動作の保存: pending が既に両者0なら board 状態を一切見ずに True
    (r_p1/r_p2 に state/confirmed_board が無くても安全 = 短絡評価の確認)。"""
    import scripts.visualize_advantage_overlay as vao

    tracker = vao.ResolvedExchangeTracker(model=object())
    tracker._incoming_total_p1 = 10.0
    tracker._incoming_total_p2 = 0.0
    tracker._target_ojama_p1 = 10.0
    assert tracker._landing_complete(object(), object(), _make_snapshot()) is True


def test_landing_complete_false_while_board_not_yet_stable() -> None:
    """pending がまだ残っており、受け側盤面も STABLE でない間は未完了。"""
    import scripts.visualize_advantage_overlay as vao

    tracker = vao.ResolvedExchangeTracker(model=object())
    tracker._incoming_total_p1 = 10.0
    tracker._incoming_total_p2 = 0.0
    tracker._target_ojama_p1 = 10.0
    r1 = _make_signal_full(None, 0, state=BoardState.OJAMA_FALL)
    r2 = _make_signal_full(None, 0, state=BoardState.STABLE)
    snap = _make_snapshot(pending_p1=5, pending_p2=0)
    assert tracker._landing_complete(r1, r2, snap) is False


def test_landing_complete_true_when_board_reaches_predicted_target() -> None:
    """②: pending は未だ0でなくても、受け側 STABLE 盤面のおじゃま数が
    予測着弾量に達していれば True (会計 pending の遅れに対する保険経路)。"""
    import scripts.visualize_advantage_overlay as vao

    tracker = vao.ResolvedExchangeTracker(model=object())
    tracker._incoming_total_p1 = 10.0
    tracker._incoming_total_p2 = 0.0
    tracker._target_ojama_p1 = 10.0
    r1 = _make_signal_full(None, 0, state=BoardState.STABLE, confirmed_board=_board_with_ojama(10))
    r2 = _make_signal_full(None, 0, state=BoardState.STABLE)
    snap = _make_snapshot(pending_p1=5, pending_p2=0)  # 会計はまだ残っている想定
    assert tracker._landing_complete(r1, r2, snap) is True


def test_resolved_hold_extends_until_pending_drains(monkeypatch) -> None:
    """指摘11本体: 両側 chain_event が None化しても pending>0 の間は保持を
    延長し、pending が0になって初めて解放する。"""
    import scripts.visualize_advantage_overlay as vao
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    held = tracker.hold_adv
    # 両側の連鎖アニメが終了。しかしまだ着弾中 (pending>0) → 延長。
    r1 = _make_signal_full(None, 0, state=BoardState.OJAMA_FALL)
    r2 = _make_signal_full(None, 0, state=BoardState.OJAMA_FALL)
    active, just_deactivated = tracker.update(r1, r2, _make_snapshot(pending_p1=8), 1.0)
    assert active is True
    assert just_deactivated is False
    assert tracker.hold_adv == held  # 保持値も不変
    # 着弾完了 (pending が0に到達) → ここで初めて解放。
    active, just_deactivated = tracker.update(
        _make_signal(None, 0), _make_signal(None, 0), _make_snapshot(), 2.0)
    assert active is False
    assert just_deactivated is True


def test_resolved_hold_safety_valve_forces_release(monkeypatch) -> None:
    """安全弁: 着弾完了シグナルが成立しないまま物理最大時間を超えたら強制解放する。"""
    import scripts.visualize_advantage_overlay as vao
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    r1 = _make_signal_full(None, 0, state=BoardState.OJAMA_FALL)
    r2 = _make_signal_full(None, 0, state=BoardState.OJAMA_FALL)
    stuck_snap = _make_snapshot(pending_p1=999, pending_p2=999)  # 着弾が永遠に完了しない想定
    active, just_deactivated = tracker.update(r1, r2, stuck_snap, elapsed_sec=1.0)
    assert active is True and just_deactivated is False
    just_before = 1.0 + vao.RESOLVED_HOLD_LANDING_MAX_WAIT_SEC - 0.01
    active, just_deactivated = tracker.update(r1, r2, stuck_snap, elapsed_sec=just_before)
    assert active is True and just_deactivated is False  # まだ安全弁未満
    just_after = 1.0 + vao.RESOLVED_HOLD_LANDING_MAX_WAIT_SEC + 0.01
    active, just_deactivated = tracker.update(r1, r2, stuck_snap, elapsed_sec=just_after)
    assert active is False and just_deactivated is True  # 安全弁で強制解放


def test_resolved_hold_bit_identical_when_pending_already_zero(monkeypatch) -> None:
    """回帰確認: pending が最初から0 (既存テスト群の前提) なら従来通り
    chain_event が両方 None になった瞬間に即解放する (延長ロジック追加前と
    ビット一致、既存テスト test_resolved_deactivates_only_when_both_sides_end
    と同じ前提の明示版)。"""
    import scripts.visualize_advantage_overlay as vao
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    active, just_deactivated = tracker.update(
        _make_signal(None, 500), _make_signal(None, 300), _make_snapshot(), 1.0)
    assert active is False
    assert just_deactivated is True


# ============================
# 指摘10: 決定度の増幅 (docs/DEMO_REVIEW_2026-08-13.md #10)
# ============================


def test_decisive_amplify_default_off_is_bit_identical(monkeypatch) -> None:
    """enable_decisive_amplify 既定 False では #9 単独の決着値と完全同一
    (backwards compat、新規サブフラグ未指定の既存呼出元は無変化)。"""
    import scripts.visualize_advantage_overlay as vao
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())  # enable_decisive_amplify省略=False
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert tracker.hold_adv == 10.0  # スタブの1回目の戻り値そのまま (増幅なし)
    assert tracker.hold_p1 == pytest.approx(0.55)


def test_decisive_amplify_adds_to_decisive_side_when_enabled(monkeypatch) -> None:
    """指摘10: enable_decisive_amplify=True で受け側応手不能度ぶんが決着値に
    加算される (CounterReachTracker/_counter_defender_adv 自体は既存実装を
    そのまま呼ぶだけなので、統合の配線だけを固定値で検証する)。"""
    import scripts.visualize_advantage_overlay as vao
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    monkeypatch.setattr(
        vao.CounterReachTracker, "update",
        lambda self, b1, b2, budget, **kw: (0.0, 0.1, float("nan")),
    )
    monkeypatch.setattr(vao, "_counter_defender_adv", lambda *a, **k: -8.0)
    tracker = vao.ResolvedExchangeTracker(model=object(), enable_decisive_amplify=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert tracker.hold_adv == pytest.approx(10.0 - 8.0)
    assert tracker.hold_p1 == pytest.approx(vao.adv_to_winprob(2.0))


def test_decisive_amplify_noop_when_no_incoming(monkeypatch) -> None:
    """飛来量が無い (両者ともゼロ) 局面では増幅対象の受け側が無く無効果。"""
    import scripts.visualize_advantage_overlay as vao
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: types.SimpleNamespace(
        board_p1_after=Board(), board_p2_after=Board(),
        board_p1_pre_landing=Board(), board_p2_pre_landing=Board(),
        dropped_to_p1=0, dropped_to_p2=0, leftover_p1=0, leftover_p2=0,
        p1_dead=False, p2_dead=False,
        chain_result_p1=types.SimpleNamespace(chain_count=1),
        chain_result_p2=types.SimpleNamespace(chain_count=1),
    ))
    tracker = vao.ResolvedExchangeTracker(model=object(), enable_decisive_amplify=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert tracker.hold_adv == 10.0  # 増幅対象なし = スタブの戻り値そのまま


# ============================
# 指摘12 修正4: 応手確率MCへの入力は着弾前盤面 (意味論バグ対処、2026-08-14)
# ============================
# 指摘12対処 (修正1: 時間予算統一) の後もなお応手0%が残っていた根因。
# _amplify_decisive が CounterReachTracker.update に着弾**後**仮想盤面
# (board_p1_after/board_p2_after、余剰おじゃまが既に降り切っている) を渡して
# いたため、実際にはまだ空中のはずのおじゃまが盤面を埋めた状態で応手可否を
# 判定してしまっていた。着弾**前**盤面 (board_p1_pre_landing/
# board_p2_pre_landing) を渡すよう修正する。


def test_amplify_decisive_passes_pre_landing_board_to_counter_mc(monkeypatch) -> None:
    """CounterReachTracker.update に渡る盤面は着弾前 (pre_landing) であり、
    着弾後 (after、既に降り切って埋まった盤面) ではないことを直接検証する。"""
    import scripts.visualize_advantage_overlay as vao

    monkeypatch.setattr(vao, "_load_chain_length_conditional_table", lambda *a, **k: {})
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)

    after_board = _board_with_ojama(30)     # 着弾後=既に埋まっている想定
    pre_landing_board = _board_with_ojama(0)  # 着弾前=まだ空いている想定
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: types.SimpleNamespace(
        board_p1_after=after_board, board_p2_after=Board(),
        board_p1_pre_landing=pre_landing_board, board_p2_pre_landing=Board(),
        dropped_to_p1=30, dropped_to_p2=0, leftover_p1=0, leftover_p2=0,
        p1_dead=False, p2_dead=False,
        chain_result_p1=types.SimpleNamespace(chain_count=1),
        chain_result_p2=types.SimpleNamespace(chain_count=1),
    ))
    captured: dict = {}

    def fake_counter_update(self, b1, b2, budget, **kw):
        captured["b1"] = b1
        captured["b2"] = b2
        return (0.0, 0.5, float("nan"))

    monkeypatch.setattr(vao.CounterReachTracker, "update", fake_counter_update)
    monkeypatch.setattr(vao, "_counter_defender_adv", lambda *a, **k: 0.0)

    tracker = vao.ResolvedExchangeTracker(model=object(), enable_decisive_amplify=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)

    assert captured["b1"] is pre_landing_board  # 着弾前を渡している (着弾後ではない)
    assert captured["b1"] is not after_board


def test_amplify_decisive_damage_calc_still_uses_after_board(monkeypatch) -> None:
    """ダメージ計算 (_counter_defender_adv) は着弾後盤面のまま (混同していない
    ことの確認、応手確率MCとは意味論が異なる)。"""
    import scripts.visualize_advantage_overlay as vao

    monkeypatch.setattr(vao, "_load_chain_length_conditional_table", lambda *a, **k: {})
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)

    after_board = _board_with_ojama(30)
    pre_landing_board = _board_with_ojama(0)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: types.SimpleNamespace(
        board_p1_after=after_board, board_p2_after=Board(),
        board_p1_pre_landing=pre_landing_board, board_p2_pre_landing=Board(),
        dropped_to_p1=30, dropped_to_p2=0, leftover_p1=0, leftover_p2=0,
        p1_dead=False, p2_dead=False,
        chain_result_p1=types.SimpleNamespace(chain_count=1),
        chain_result_p2=types.SimpleNamespace(chain_count=1),
    ))
    monkeypatch.setattr(
        vao.CounterReachTracker, "update",
        lambda self, b1, b2, budget, **kw: (0.0, 0.5, float("nan")),
    )
    captured: dict = {}

    def fake_counter_defender_adv(defender_side, defender_prob, incoming, b1, b2, **kw):
        captured["b1"] = b1
        captured["b2"] = b2
        return 0.0

    monkeypatch.setattr(vao, "_counter_defender_adv", fake_counter_defender_adv)

    tracker = vao.ResolvedExchangeTracker(model=object(), enable_decisive_amplify=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)

    assert captured["b1"] is after_board  # ダメージ計算は着弾後のまま (修正対象外)
    assert captured["b1"] is not pre_landing_board


# ============================
# 指摘12: 増幅の時間予算統一 + 専用強度 + ホールド中表示
# (docs/DEMO_REVIEW_2026-08-13.md #12、2026-08-14)
# ============================
# 実測 (logs/diag_issue12_breakdown_2026-08-14_v5.log): _amplify_decisive が
# 旧式 iv.estimate_chain_anim_duration_sec(観測連鎖数=2連鎖×0.4秒=2.4秒) を
# 直呼びしていたため、実演出8.1秒に対し残り時間を過小評価 → 応手0%と誤断 →
# 決定度増幅(旧COUNTER_SCALE全量)が発動し 84%→97% に飽和していた。


def _stub_mutual_exchange_result(
    dropped_to_p1: int = 0, dropped_to_p2: int = 0,
    leftover_p1: int = 0, leftover_p2: int = 0,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        board_p1_after=Board(), board_p2_after=Board(),
        board_p1_pre_landing=Board(), board_p2_pre_landing=Board(),
        dropped_to_p1=dropped_to_p1, dropped_to_p2=dropped_to_p2,
        leftover_p1=leftover_p1, leftover_p2=leftover_p2,
        p1_dead=False, p2_dead=False,
        chain_result_p1=types.SimpleNamespace(chain_count=1),
        chain_result_p2=types.SimpleNamespace(chain_count=1),
    )


def test_amplify_decisive_source_never_calls_legacy_time_budget_directly() -> None:
    """静的回帰テスト (修正1の再発防止): _amplify_decisive の実装ソースに
    旧式 `iv.estimate_chain_anim_duration_sec` の直呼びが無く、#3で実装済みの
    `_chain_remaining_time_budget_sec` 経由のみで時間予算を求めていることを
    ソース検査で固定する (「時間予算はこの関数以外で計算しない」の徹底)。"""
    import inspect
    import scripts.visualize_advantage_overlay as vao

    method = vao.ResolvedExchangeTracker._amplify_decisive
    src = inspect.getsource(method)
    code_only = src.replace(method.__doc__ or "", "")  # docstring内の言及を除外しコード本体のみ検査
    assert "estimate_chain_anim_duration_sec" not in code_only
    assert "_chain_remaining_time_budget_sec" in code_only


def test_amplify_decisive_budget_matches_unified_function_and_differs_from_legacy(
    monkeypatch,
) -> None:
    """修正1本体: _amplify_decisive が CounterReachTracker.update に渡す budget は
    `_chain_remaining_time_budget_sec` の出力と一致し、旧式 (観測連鎖数×0.4秒、
    経過時間控除なし) とは異なる値になる (実測との整合、指摘12直撃の再発防止)。"""
    import scripts.visualize_advantage_overlay as vao

    monkeypatch.setattr(vao, "_load_chain_length_conditional_table", lambda *a, **k: {})
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    monkeypatch.setattr(
        vao, "resolve_mutual_exchange",
        lambda *a, **k: _stub_mutual_exchange_result(dropped_to_p1=30),
    )
    captured: dict = {}

    def fake_counter_update(self, b1, b2, budget, **kw):
        captured["budget"] = budget
        return (0.0, 0.3, float("nan"))

    monkeypatch.setattr(vao.CounterReachTracker, "update", fake_counter_update)
    monkeypatch.setattr(vao, "_counter_defender_adv", lambda *a, **k: -3.0)

    tracker = vao.ResolvedExchangeTracker(model=object(), enable_decisive_amplify=True)
    # 攻撃側 (2P、defender=1P なので attacker_event=ev2) は6連鎖・t=100.0発火。
    ev1 = _make_chain_event(trigger_sec=100.0, total_score=500)
    ev2 = ChainEvent(
        trigger_sec=100.0, end_sec=101.0, before_board=Board(),
        chain_count=6, total_erased=0, total_score=300, base_score=0,
        all_clear_bonus_applied=0, ojama_sent=0, leftover_score=0, is_all_clear=False,
    )
    # elapsed_sec(試合内経過)は0.0のままにし、raw t_sec=101.0 (発火から1秒後)
    # を新規引数で渡す (2026-08-14 修正1、update() 側スレッド配線の確認)。
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(),
                   0.0, t_sec=101.0)

    expected = vao._chain_remaining_time_budget_sec(6, 100.0, 101.0, {})
    legacy = vao.iv.estimate_chain_anim_duration_sec(6.0)
    assert captured["budget"] == pytest.approx(expected)
    assert captured["budget"] != pytest.approx(legacy)  # 旧式とビット一致しない


def test_resolved_update_t_sec_defaults_to_elapsed_sec_when_omitted() -> None:
    """update() に t_sec を渡さない既存呼出し (4引数のまま) は内部 _t_sec が
    elapsed_sec にフォールバックする (backwards compat、既存テスト群は
    enable_decisive_amplify=False のため実害なし)。"""
    import scripts.visualize_advantage_overlay as vao

    tracker = vao.ResolvedExchangeTracker(model=object())
    tracker.update(_make_signal(None, 0), _make_signal(None, 0), _make_snapshot(), 3.5)
    assert tracker._t_sec == pytest.approx(3.5)


def test_resolved_amplify_scale_derived_from_existing_constants() -> None:
    """修正2: RESOLVED_AMPLIFY_SCALE は既存定数 (COUNTER_SCALE, W_COUNTER) の
    積のみで定義され、シーンからの逆算で決めていない
    (feedback_overfitting_awareness_2026-08-04 準拠、新規マジックナンバー無し)。
    ライブ per-frame 経路で counter_adv が W_COUNTER 倍されてから合成される
    のと同じ実効上限に揃えることで、決着増幅の二重計上 (指摘12の根因) を防ぐ。"""
    import scripts.visualize_advantage_overlay as vao

    assert vao.RESOLVED_AMPLIFY_SCALE == pytest.approx(vao.COUNTER_SCALE * vao.W_COUNTER)
    assert vao.RESOLVED_AMPLIFY_SCALE < vao.COUNTER_SCALE  # 全量加算だった旧バグは再発しない


def test_counter_defender_adv_default_scale_is_bit_identical_to_live_path() -> None:
    """scale 省略時はライブ per-frame 用 COUNTER_SCALE と完全に一致する
    (backwards compat、既存呼出元 = ライブ経路は挙動不変)。"""
    import scripts.visualize_advantage_overlay as vao

    b = Board()
    omitted = vao._counter_defender_adv("2P", 0.3, 20.0, b, b)
    explicit = vao._counter_defender_adv("2P", 0.3, 20.0, b, b, scale=vao.COUNTER_SCALE)
    assert omitted == explicit


def test_counter_defender_adv_resolved_scale_shrinks_magnitude_proportionally() -> None:
    """修正2: RESOLVED_AMPLIFY_SCALE を渡すと、振幅が COUNTER_SCALE 比で縮む
    (二重計上防止の実効値を直接確認)。"""
    import scripts.visualize_advantage_overlay as vao

    b = Board()
    full = vao._counter_defender_adv("2P", 0.0, 20.0, b, b, scale=vao.COUNTER_SCALE)
    resolved = vao._counter_defender_adv("2P", 0.0, 20.0, b, b, scale=vao.RESOLVED_AMPLIFY_SCALE)
    assert resolved == pytest.approx(full * (vao.RESOLVED_AMPLIFY_SCALE / vao.COUNTER_SCALE))


def test_amplify_decisive_populates_hold_display_fields(monkeypatch) -> None:
    """修正3: 決着計算時に受け側の side/飛来量/応手確率をホールド表示専用
    フィールドへ公開する (ResolvedExchangeTracker.hold_defender_side 等)。"""
    import scripts.visualize_advantage_overlay as vao

    monkeypatch.setattr(vao, "_load_chain_length_conditional_table", lambda *a, **k: {})
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    monkeypatch.setattr(
        vao, "resolve_mutual_exchange",
        lambda *a, **k: _stub_mutual_exchange_result(dropped_to_p1=30),
    )
    monkeypatch.setattr(
        vao.CounterReachTracker, "update",
        lambda self, b1, b2, budget, **kw: (0.0, 0.15, float("nan")),
    )
    tracker = vao.ResolvedExchangeTracker(model=object(), enable_decisive_amplify=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)

    assert tracker.hold_defender_side == "1P"  # dropped_to_p1=30 のみ = 1Pが受け側
    assert tracker.hold_incoming_ojama == pytest.approx(30.0)
    assert tracker.hold_defender_prob == pytest.approx(0.15)


def test_amplify_decisive_resets_hold_display_fields_when_no_incoming(monkeypatch) -> None:
    """飛来量が無い局面では表示フィールドも None/nan にリセットされる
    (前回保持の受け側情報を持ち越さない)。"""
    import scripts.visualize_advantage_overlay as vao

    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    monkeypatch.setattr(
        vao, "resolve_mutual_exchange", lambda *a, **k: _stub_mutual_exchange_result())
    tracker = vao.ResolvedExchangeTracker(model=object(), enable_decisive_amplify=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)

    assert tracker.hold_defender_side is None
    assert tracker.hold_incoming_ojama == 0.0
    assert math.isnan(tracker.hold_defender_prob)


def test_resolved_hold_display_fields_stay_default_when_amplify_disabled(monkeypatch) -> None:
    """enable_decisive_amplify=False (既定) では _amplify_decisive 自体が
    呼ばれないため、表示フィールドは __init__ の既定値のまま
    (#9 単独構成に追加コストも追加副作用も無いことの確認)。"""
    import scripts.visualize_advantage_overlay as vao

    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())  # enable_decisive_amplify省略=False
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)

    assert tracker.hold_defender_side is None
    assert math.isnan(tracker.hold_defender_prob)


def test_amplify_decisive_hold_display_fields_do_not_affect_judgment(monkeypatch) -> None:
    """修正3の要件: ホールド表示専用フィールドの公開は判定値
    (hold_adv/hold_p1) に一切影響しない (表示と判定の分離を固定)。"""
    import scripts.visualize_advantage_overlay as vao

    monkeypatch.setattr(vao, "_load_chain_length_conditional_table", lambda *a, **k: {})
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    monkeypatch.setattr(
        vao, "resolve_mutual_exchange",
        lambda *a, **k: _stub_mutual_exchange_result(dropped_to_p1=30),
    )
    monkeypatch.setattr(
        vao.CounterReachTracker, "update",
        lambda self, b1, b2, budget, **kw: (0.0, 0.15, float("nan")),
    )
    tracker = vao.ResolvedExchangeTracker(model=object(), enable_decisive_amplify=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    hold_adv_before, hold_p1_before = tracker.hold_adv, tracker.hold_p1

    # 表示フィールドを読むだけの操作 (パネル描画相当) を挟んでも判定値は不変。
    _ = (tracker.hold_defender_side, tracker.hold_incoming_ojama, tracker.hold_defender_prob)
    assert tracker.hold_adv == hold_adv_before
    assert tracker.hold_p1 == hold_p1_before


def test_resolve_counter_text_for_display_uses_hold_values_while_active() -> None:
    """修正3: ホールド中 (resolved_hold_active=True) かつ defender-only モードは
    ライブ per-frame の古い値 (指摘12 副次バグ) ではなく、決着計算の内部値
    (hold_defender_side/hold_incoming_ojama/hold_defender_prob) を描画する。"""
    from scripts.visualize_advantage_overlay import _resolve_counter_text_for_display

    text = _resolve_counter_text_for_display(
        True, True,
        "2P", 0.42, 55.0,  # hold_* (決着計算の内部値、これが使われるべき)
        "1P", 0.99, float("nan"), 3.0,  # ライブ per-frame の古い値 (無視される)
    )
    assert "2P応手 42%" in text
    assert "55" in text
    assert "1P" not in text  # ライブ側の古い受け側 (1P) が漏れていない


def test_resolve_counter_text_for_display_falls_back_when_not_holding() -> None:
    """ホールド非アクティブ時は従来通りライブ per-frame の値を使う
    (_resolve_counter_text とビット一致、backwards compat)。"""
    from scripts.visualize_advantage_overlay import (
        _resolve_counter_text, _resolve_counter_text_for_display,
    )

    live = _resolve_counter_text(True, "1P", 0.6, float("nan"), 12.0)
    via_helper = _resolve_counter_text_for_display(
        True, False, "2P", 0.42, 55.0, "1P", 0.6, float("nan"), 12.0)
    assert via_helper == live


def test_resolve_counter_text_for_display_ignores_hold_when_defender_only_disabled() -> None:
    """enable_defender_only=False の場合は resolved_hold_active に関わらず
    従来の両側常時表示のまま (backwards compat、hold中でも例外扱いしない)。"""
    from scripts.visualize_advantage_overlay import (
        _resolve_counter_text, _resolve_counter_text_for_display,
    )

    live = _resolve_counter_text(False, "1P", 0.6, 0.4, 12.0)
    via_helper = _resolve_counter_text_for_display(
        False, True, "2P", 0.42, 55.0, "1P", 0.6, 0.4, 12.0)
    assert via_helper == live


# ============================
# 評価済みモデル成果物の直読み (2026-08-14 coordinator指示)
# ============================
# 「評価したモデル (AUC 0.657/終盤0.839) = デモが使うモデル」を構造的に
# 一致させるための _acquire_model/_load_artifact_model/_score_advantage_full_row
# の配線テスト (実joblib/実CSV学習は重いためモック中心、成果物ロード成功系は
# 実ファイルが存在する前提のsmokeテストのみ実行)。


def test_load_artifact_model_returns_none_when_files_missing(monkeypatch, tmp_path) -> None:
    """成果物 (joblib) が存在しない環境では None を返す (fail-safe)。"""
    import scripts.visualize_advantage_overlay as vao
    monkeypatch.setattr(vao, "MODEL_ARTIFACT_PATH", tmp_path / "no_such_model.joblib")
    monkeypatch.setattr(
        vao, "MODEL_ARTIFACT_FEATURE_COLS_PATH", tmp_path / "no_such_cols.json")
    assert vao._load_artifact_model() is None


def test_load_artifact_model_returns_none_when_cols_missing(monkeypatch, tmp_path) -> None:
    """joblib はあるが隣接列リストJSONが無い場合も None (片方欠損もfail-safe)。"""
    import scripts.visualize_advantage_overlay as vao
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"dummy")  # 内容は読まれない (列リスト欠損で先にNone)
    monkeypatch.setattr(vao, "MODEL_ARTIFACT_PATH", model_path)
    monkeypatch.setattr(
        vao, "MODEL_ARTIFACT_FEATURE_COLS_PATH", tmp_path / "no_such_cols.json")
    assert vao._load_artifact_model() is None


def test_acquire_model_falls_back_to_train_model_when_artifact_missing(
    monkeypatch, tmp_path,
) -> None:
    """成果物が無い環境では従来の _train_model にフォールバックする (fail-safe)。"""
    import scripts.visualize_advantage_overlay as vao
    monkeypatch.setattr(vao, "MODEL_ARTIFACT_PATH", tmp_path / "no_such_model.joblib")
    monkeypatch.setattr(
        vao, "MODEL_ARTIFACT_FEATURE_COLS_PATH", tmp_path / "no_such_cols.json")
    sentinel = object()
    calls: list[object] = []
    monkeypatch.setattr(
        vao, "_train_model", lambda exclude_video=None: (calls.append(exclude_video), sentinel)[1])
    result = vao._acquire_model(None)
    assert result is sentinel
    assert calls == [None]


def test_acquire_model_skips_artifact_when_exclude_video_given(monkeypatch) -> None:
    """exclude_video 指定時は成果物 (全144動画で学習済み・除外不可) を使わず、
    必ず _train_model にフォールバックする (リーク防止、fail-silent警戒)。"""
    import scripts.visualize_advantage_overlay as vao
    load_calls: list[int] = []
    monkeypatch.setattr(
        vao, "_load_artifact_model", lambda: (load_calls.append(1), object())[1])
    sentinel = object()
    train_calls: list[object] = []
    monkeypatch.setattr(
        vao, "_train_model",
        lambda exclude_video=None: (train_calls.append(exclude_video), sentinel)[1])
    result = vao._acquire_model("video_29")
    assert result is sentinel
    assert train_calls == ["video_29"]
    assert load_calls == []  # 成果物ロード自体を試みない


def test_score_advantage_dispatches_to_full_row_when_artifact_mode(monkeypatch) -> None:
    """model._puyo_feature_mode == 'full_row' なら _score_advantage_full_row に
    完全委譲する (従来 diff ベース経路とは別関数、混在しない)。"""
    import scripts.visualize_advantage_overlay as vao
    calls: list[tuple] = []

    def _stub_full_row(model, b1, b2, snap, attribution_exclude):
        calls.append((model, b1, b2, snap, attribution_exclude))
        return 42.0, 0.71, [("dummy", 1.0)]

    monkeypatch.setattr(vao, "_score_advantage_full_row", _stub_full_row)
    model = types.SimpleNamespace(_puyo_feature_mode="full_row")
    b1, b2 = Board(), Board()
    snap = _make_snapshot()
    adv, p1, drivers = vao._score_advantage(model, b1, b2, snap)
    assert (adv, p1, drivers) == (42.0, 0.71, [("dummy", 1.0)])
    assert len(calls) == 1


def test_score_advantage_full_row_symmetric_on_identical_boards() -> None:
    """評価済み成果物モデルの実smoke: 完全に対称な局面 (同一盤面・pending無し)
    では adv=0/p1=0.5 に近い値を返す (対称化式 0.5*(p_1p+(1-p_2p)) の健全性)。

    成果物 (data/verify/retrain148_2026-08-14) が無い環境ではスキップする。
    """
    import scripts.visualize_advantage_overlay as vao
    if not vao.MODEL_ARTIFACT_PATH.exists() or not vao.MODEL_ARTIFACT_FEATURE_COLS_PATH.exists():
        pytest.skip("評価済みモデル成果物が未配置の環境のためスキップ")
    model = vao._load_artifact_model()
    assert model is not None
    assert model._puyo_feature_mode == "full_row"
    snap = _make_snapshot()
    adv, p1, drivers = vao._score_advantage(model, Board(), Board(), snap)
    assert adv == pytest.approx(0.0, abs=1e-9)  # 同一盤面同一入力 → 完全対称
    assert p1 == pytest.approx(0.5, abs=1e-9)
    assert isinstance(drivers, list)


def test_side_feats_full_matches_direct_indicator_calls() -> None:
    """_side_feats_full_base が FULL_MODEL_GRID_REGISTRY の各関数を board に
    直接適用した値と完全一致することを確認する (委譲のみで新規ロジックが
    無いことの回帰テスト)。"""
    import scripts.visualize_advantage_overlay as vao
    board = _board_with_ojama(5)
    base = vao._side_feats_full_base(board)
    for name, fn in vao.FULL_MODEL_GRID_REGISTRY.items():
        assert base[name] == pytest.approx(fn(board).score, nan_ok=True)
    total_conn, _ = vao.iv.connectivity_observation(board)
    assert base["conn_pair_count"] == float(total_conn.pair_count)
    assert base["conn_triple_count"] == float(total_conn.triple_count)
    assert base["conn_max_group_size"] == float(total_conn.max_group_size)


def test_side_feats_full_diff_targets_own_removed() -> None:
    """own→diff完全置換対象 (DIFF_REPLACE_OWN_COLUMNS) は最終featにown列が
    残らず、diff_ 列だけが入ることを確認する (b-2決定の再現)。"""
    import scripts.visualize_advantage_overlay as vao
    base_self = vao._side_feats_full_base(_board_with_ojama(3))
    base_opp = vao._side_feats_full_base(_board_with_ojama(1))
    feat = vao._side_feats_full(base_self, base_opp, net=0, forecast=0)
    for c in vao.DIFF_REPLACE_OWN_COLUMNS:
        assert c not in feat
        assert f"diff_{c}" in feat
        assert feat[f"diff_{c}"] == pytest.approx(base_self[c] - base_opp[c])
    for c in vao.DIFF_KEEP_OWN_PAIR_COLUMNS + vao.DIFF_KEEP_OWN_HEAVY_COLUMNS:
        assert c in feat  # own側は残る (own+diff両方)
        assert f"diff_{c}" in feat
