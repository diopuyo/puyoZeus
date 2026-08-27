"""有利不利オーバーレイの合成成分(圧力/得点リード)の単体テスト。

PressureTracker(着弾ダメージの持続記憶)と ScoreLeadTracker(得点リード)は
純ロジックなので認識なしで検証できる。回帰防止用。
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace as dataclass_replace
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
    _kill_override_chain_completion_inputs, _kill_override_attribution_entry,
    _drivers_for_display, KILL_OVERRIDE_DRIVER_KEY_P1, KILL_OVERRIDE_DRIVER_KEY_P2,
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
# EarlyFireTracker.finalized_since_last_check / on_settled(finalized=...)
# (2026-08-22 修正②)
# ============================

def test_on_settled_default_still_clears_bias_unconditionally() -> None:
    """既定 (引数省略) は従来通り無条件クリア (backwards compat, bit-identical)。"""
    ft = EarlyFireTracker()
    ft.bias = 12.3
    ft.on_settled()
    assert ft.bias == 0.0


def test_on_settled_finalized_true_clears_bias() -> None:
    """finalized=True (旧来と同じ意味) は無条件クリア。"""
    ft = EarlyFireTracker()
    ft.bias = 12.3
    ft.on_settled(finalized=True)
    assert ft.bias == 0.0


def test_on_settled_finalized_false_preserves_bias() -> None:
    """finalized=False は bias を維持する (修正②の核心)。"""
    ft = EarlyFireTracker()
    ft.bias = 12.3
    ft.on_settled(finalized=False)
    assert ft.bias == 12.3


def test_finalized_since_last_check_first_call_is_false() -> None:
    """初回呼び出しは判断材料が無いため False (安全側、誤ってクリアしない)。"""
    ft = EarlyFireTracker()
    assert ft.finalized_since_last_check(1260, 0) is False


def test_finalized_since_last_check_detects_change() -> None:
    """chain_total_score_p1/p2 の値が変化したら finalize が起きたと判定する。"""
    ft = EarlyFireTracker()
    ft.finalized_since_last_check(0, 0)  # 初回 (基準値確立)
    assert ft.finalized_since_last_check(0, 0) is False  # 変化なし
    assert ft.finalized_since_last_check(4020, 0) is True  # p1側finalize検知
    assert ft.finalized_since_last_check(4020, 0) is False  # 以後は変化なし
    assert ft.finalized_since_last_check(4020, 1260) is True  # p2側finalize検知


def test_finalized_since_last_check_new_instance_per_match_is_safe() -> None:
    """試合境界で EarlyFireTracker を作り直す既存設計 (_fresh_trackers) と
    整合し、新インスタンスは前試合の値を引き継がない。"""
    ft_prev_match = EarlyFireTracker()
    ft_prev_match.finalized_since_last_check(9999, 9999)
    ft_new_match = EarlyFireTracker()
    # 新インスタンスは未確認 (None) から始まるため初回は必ず False。
    assert ft_new_match.finalized_since_last_check(0, 0) is False


# ============================
# _kill_override_chain_completion_inputs (2026-08-22 修正①)
# + ChainGenerationAccumulator (2026-08-22 改良②)
# ============================

def _busy_side(chain_event) -> types.SimpleNamespace:
    return types.SimpleNamespace(chain_event=chain_event, state=BoardState.CHAIN)


def _idle_side() -> types.SimpleNamespace:
    return types.SimpleNamespace(chain_event=None, state=BoardState.STABLE)


def _fake_snap_pending(pending_p1: int, pending_p2: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(pending_p1=pending_p1, pending_p2=pending_p2)


def _make_4connect_board() -> Board:
    """最下段に赤4個 (1連鎖確定) を並べた盤面 (tests/test_exchange_virtual_board.py
    make_4connect_board と同一方式)。"""
    b = Board()
    for col in range(4):
        b.set(BOARD_ROWS - 1, col, 1)
    return b


# ---- _chain_event_gen_ojama (単発イベントのお邪魔換算、module-level関数) ----

def test_chain_event_gen_ojama_uses_total_score_directly_when_reliable(
    monkeypatch,
) -> None:
    """total_score が信頼できる場合 (>=CHAIN_TOTAL_MIN_SCORE) は追加simulateを
    一切行わない (無駄な再計算を避ける、2026-08-22 user指摘の核心)。"""
    import scripts.visualize_advantage_overlay as vao
    calls = {"n": 0}
    orig_simulate = vao._CHAIN_COMPLETION_SIMULATOR.simulate

    def _spy_simulate(board):
        calls["n"] += 1
        return orig_simulate(board)

    monkeypatch.setattr(vao._CHAIN_COMPLETION_SIMULATOR, "simulate", _spy_simulate)
    before1 = _make_4connect_board()
    ev1 = _make_chain_event(trigger_sec=1.0, before_board=before1, total_score=100_000)
    gen = vao._chain_event_gen_ojama(ev1, elapsed_sec=0.0)
    assert calls["n"] == 0  # フォールバック simulate は呼ばれない
    assert gen > 0.0


def test_chain_event_gen_ojama_w7_fallback_simulates_when_total_score_zero(
    monkeypatch,
) -> None:
    """[2026-08-22 user指摘対応・核心テスト] total_score=0 (formula/landing経路
    + enable_pseudo_chain_score_fill=False、現行本番既定) でも、before_board
    に本物の連鎖があれば自前simulateで得点を確定する。t=6717.5 (画面に
    「50×386」の掛け算式表示=score OCR不能) の実運用条件を再現する。
    ChainEvent.ojama_sent は使わない設計のため ojama_sent=0 のままでも
    成立することも確認する。"""
    import scripts.visualize_advantage_overlay as vao
    before1 = _make_4connect_board()
    ev1 = _make_chain_event(trigger_sec=1.0, before_board=before1, total_score=0)
    assert ev1.ojama_sent == 0  # 現行本番のpseudo ChainEventと同条件

    calls = {"n": 0}
    orig_calc = vao.calculate_chain_score

    def _spy_calc(result):
        calls["n"] += 1
        r = orig_calc(result)
        return type(r)(steps=r.steps, total_score=100_000, is_all_clear=r.is_all_clear)

    monkeypatch.setattr(vao, "calculate_chain_score", _spy_calc)
    gen = vao._chain_event_gen_ojama(ev1, elapsed_sec=0.0)
    assert calls["n"] == 1  # フォールバックsimulate経路が実際に発動した証拠
    assert gen > 0.0


def test_chain_event_gen_ojama_zero_when_before_board_has_no_real_chain() -> None:
    """total_score が低く、before_board にも本物の連鎖が無ければ 0。"""
    import scripts.visualize_advantage_overlay as vao
    ev = _make_chain_event(trigger_sec=1.0, total_score=CHAIN_TOTAL_MIN_SCORE - 1)
    assert vao._chain_event_gen_ojama(ev, elapsed_sec=0.0) == 0.0


# ---- ChainGenerationAccumulator (複数トリガーにまたがる累積、改良②) ----

def test_chain_gen_accumulator_zero_when_idle() -> None:
    """busy でない側は常に (0.0, None)。"""
    import scripts.visualize_advantage_overlay as vao
    acc = vao.ChainGenerationAccumulator()
    gen1, before1, gen2, before2 = acc.update(_idle_side(), _idle_side(), 0.0)
    assert (gen1, before1, gen2, before2) == (0.0, None, 0.0, None)


def test_chain_gen_accumulator_single_trigger_matches_single_event() -> None:
    """単発トリガーのみなら _chain_event_gen_ojama と同じ値になる。"""
    import scripts.visualize_advantage_overlay as vao
    before1 = _make_4connect_board()
    ev1 = _make_chain_event(trigger_sec=10.0, before_board=before1, total_score=100_000)
    acc = vao.ChainGenerationAccumulator()
    gen1, got_before1, gen2, _ = acc.update(_busy_side(ev1), _idle_side(), 0.0)
    expected = vao._chain_event_gen_ojama(ev1, 0.0)
    assert gen1 == expected
    assert got_before1 is before1
    assert gen2 == 0.0


def test_chain_gen_accumulator_accumulates_across_multiple_triggers_2026_08_22() -> None:
    """[根治②・核心テスト] t=6717.5 で実測した根本原因の再現:
    formula 機構は「アクティブな疑似イベントがあれば新規発火しない」ため、
    長い連鎖はホールド期限切れのたびに新しい ChainEvent (別の trigger_sec)
    に置き換わる。単発イベントだけを見ると連鎖全体を大幅に過小評価する
    (実測: 216のpendingに対し1個目の断片だけでは84個しか生成が見えず
    KILL_RATIO_FULL=1.5を大きく超えたまま=誤爆継続)。本アキュムレータは
    trigger_sec の変化ごとに加算し、断片の合計を返さねばならない。"""
    import scripts.visualize_advantage_overlay as vao
    before_a = _make_4connect_board()
    ev_a = _make_chain_event(trigger_sec=10.0, before_board=before_a, total_score=1000)
    before_b = _make_4connect_board()
    ev_b = _make_chain_event(trigger_sec=10.5, before_board=before_b, total_score=2000)

    # [2026-08-22 user判断] 既定は accumulate=False (対症療法の実測欠陥により
    # 非累積が既定に変更された)。本テストは「累積モードを明示指定した場合の
    # 挙動」を固定するため accumulate=True を明示する。
    acc = vao.ChainGenerationAccumulator(accumulate=True)
    gen_a, _, _, _ = acc.update(_busy_side(ev_a), _idle_side(), 0.0)
    gen_ab, before_ab, _, _ = acc.update(_busy_side(ev_b), _idle_side(), 0.0)

    expected_a = vao._chain_event_gen_ojama(ev_a, 0.0)
    expected_b = vao._chain_event_gen_ojama(ev_b, 0.0)
    assert gen_a == expected_a
    # 2個目のトリガー (trigger_sec が変化) で累積加算される (二重計上ではなく合算)。
    assert gen_ab == pytest.approx(expected_a + expected_b)
    assert before_ab is before_b  # room算出用は直近のbefore_boardを使う


def test_chain_gen_accumulator_default_replaces_instead_of_accumulating_2026_08_22() -> None:
    """[2026-08-22 user判断・核心テスト] 既定 (accumulate=False) は複数トリガーに
    またがって合算せず、直近1件の chain_event の値に**置き換える**。

    背景: 累積 (旧既定) は「まだ画面に見えていない残り連鎖ぶんまで既に生成
    し終えた」という架空の完了状態を仮定するため、実測 (全編再走査) で
    raw モデルとの間に新しい不一致時間帯 (t=6717.5 直前に6.63秒) を作ることが
    判明した。既定を非累積に変更し、根治 (CHAIN保持時間の実測較正配線) と
    併用したときに二重計上しない設計にした。"""
    import scripts.visualize_advantage_overlay as vao
    before_a = _make_4connect_board()
    ev_a = _make_chain_event(trigger_sec=10.0, before_board=before_a, total_score=1000)
    before_b = _make_4connect_board()
    ev_b = _make_chain_event(trigger_sec=10.5, before_board=before_b, total_score=2000)

    acc = vao.ChainGenerationAccumulator()  # 既定 accumulate=False
    gen_a, _, _, _ = acc.update(_busy_side(ev_a), _idle_side(), 0.0)
    gen_b, before_b_out, _, _ = acc.update(_busy_side(ev_b), _idle_side(), 0.0)

    expected_a = vao._chain_event_gen_ojama(ev_a, 0.0)
    expected_b = vao._chain_event_gen_ojama(ev_b, 0.0)
    assert gen_a == expected_a
    assert gen_b == expected_b  # 合算されず直近値に置き換わる (expected_a + expected_bにはならない)
    assert before_b_out is before_b


def test_chain_gen_accumulator_same_trigger_not_double_counted() -> None:
    """同一 trigger_sec の chain_event を複数フレームで受けても二重加算しない
    (EarlyFireTracker と同じ規約)。"""
    import scripts.visualize_advantage_overlay as vao
    before1 = _make_4connect_board()
    ev1 = _make_chain_event(trigger_sec=10.0, before_board=before1, total_score=1000)
    acc = vao.ChainGenerationAccumulator()
    gen_first, _, _, _ = acc.update(_busy_side(ev1), _idle_side(), 0.0)
    gen_second, _, _, _ = acc.update(_busy_side(ev1), _idle_side(), 0.0)
    assert gen_first == gen_second  # 同じイベントの再提示では増えない


def test_chain_gen_accumulator_resets_when_leaving_busy_state() -> None:
    """busy 状態を離れたら (=真の連鎖終了) 累積をリセットする。"""
    import scripts.visualize_advantage_overlay as vao
    before1 = _make_4connect_board()
    ev1 = _make_chain_event(trigger_sec=10.0, before_board=before1, total_score=100_000)
    acc = vao.ChainGenerationAccumulator()
    gen1, _, _, _ = acc.update(_busy_side(ev1), _idle_side(), 0.0)
    assert gen1 > 0.0
    gen1_after_stable, before1_after_stable, _, _ = acc.update(
        _idle_side(), _idle_side(), 0.0)
    assert gen1_after_stable == 0.0
    assert before1_after_stable is None
    # 同じ trigger_sec が再度来ても (通常は起きないが) 新規カウントとして扱える
    # ようリセットされていること (内部状態の直接確認)。
    assert acc._last_trigger["1p"] is None


def test_chain_gen_accumulator_two_sides_independent() -> None:
    """1P/2P の累積は互いに独立している。"""
    import scripts.visualize_advantage_overlay as vao
    before1 = _make_4connect_board()
    before2 = _make_4connect_board()
    ev1 = _make_chain_event(trigger_sec=1.0, before_board=before1, total_score=100_000)
    ev2 = _make_chain_event(trigger_sec=2.0, before_board=before2, total_score=0)
    acc = vao.ChainGenerationAccumulator()
    gen1, _, gen2, _ = acc.update(_busy_side(ev1), _busy_side(ev2), 0.0)
    assert gen1 > 0.0
    assert gen2 == 0.0  # before_boardが空盤面 (連鎖なし) のためW7フォールバックも0


# ---- _kill_override_chain_completion_inputs (簡素化後、gen/before_board を直接受け取る) ----

def test_chain_completion_inputs_noop_when_neither_side_firing() -> None:
    """どちらの側も生成量0 (発火していない) なら入力は完全不変。"""
    b1, b2 = Board(), Board()
    snap = _fake_snap_pending(216, 0)
    r1, r2, p1, p2 = _kill_override_chain_completion_inputs(
        snap, b1, b2, room1=5, room2=59,
        gen1=0.0, before1=None, gen2=0.0, before2=None)
    assert (r1, r2, p1, p2) == (5, 59, 216.0, 0.0)


def test_chain_completion_inputs_fully_cancels_when_own_gen_large() -> None:
    """t=6717.5 の再現: 自分の連鎖が pending を相殺しきる → 残存 pending は0。

    余剰 (pending を超えた分) は実ゲーム仕様通り相手へ送られる
    (cancel_own_pending_then_send_surplus と同じ規約、二重会計を避けるため
    独自の相殺ロジックは作らず resolve_mutual_exchange をそのまま使う)。
    """
    before1 = _make_4connect_board()
    b2 = Board()
    snap = _fake_snap_pending(216, 0)
    room1_eff, room2_eff, pending1_eff, pending2_eff = (
        _kill_override_chain_completion_inputs(
            snap, before1, b2, room1=5, room2=59,
            gen1=9999.0, before1=before1, gen2=0.0, before2=None))
    assert pending1_eff == 0.0  # 216個をはるかに超える生成量で完全相殺
    assert pending2_eff > 0.0  # 余剰は2Pへの反撃として送られる (相殺前より悪化はしない=1P有利の裏付け)
    assert room1_eff == 72  # 4連結が消えて盤面が完全に空になった


def test_chain_completion_inputs_partial_residual_when_own_gen_small() -> None:
    """自分の連鎖が小さく相殺しきれない場合は残存 pending が残る (発火継続)。"""
    before1 = _make_4connect_board()
    b2 = Board()
    snap = _fake_snap_pending(216, 0)
    _, _, pending1_eff, _ = _kill_override_chain_completion_inputs(
        snap, before1, b2, room1=5, room2=59,
        gen1=1.0, before1=before1, gen2=0.0, before2=None)
    assert 0.0 < pending1_eff < 216.0  # 一部は相殺されるが残存 (未対策時と同じ方向)


def test_chain_completion_inputs_falls_back_to_board_now_when_before_none() -> None:
    """gen>0 だが before_board が None (呼出側の防御的既定) の場合は
    現在の確定盤面 (board_now) を使う (b1/b2 引数のフォールバック)。"""
    b1, b2 = _make_4connect_board(), Board()
    snap = _fake_snap_pending(0, 0)
    room1_eff, room2_eff, pending1_eff, pending2_eff = (
        _kill_override_chain_completion_inputs(
            snap, b1, b2, room1=5, room2=59,
            gen1=10.0, before1=None, gen2=0.0, before2=None))
    assert room1_eff == 72  # b1 (4連結あり) がそのまま simulate された証拠


# ============================
# _kill_override_attribution_entry / _drivers_for_display (2026-08-22 修正④)
# ============================

def test_kill_override_attribution_entry_identifies_dying_side() -> None:
    """adv が悪化 (1P不利化) した場合は1P側キー、改善した場合は2P側キー。"""
    key1, val1 = _kill_override_attribution_entry(
        adv_before=42.0, adv_after=-100.0, pending1=216.0, pending2=0.0,
        room1=5, room2=59)
    assert key1 == KILL_OVERRIDE_DRIVER_KEY_P1
    assert val1 == 216.0
    key2, val2 = _kill_override_attribution_entry(
        adv_before=-30.0, adv_after=100.0, pending1=0.0, pending2=200.0,
        room1=60, room2=8)
    assert key2 == KILL_OVERRIDE_DRIVER_KEY_P2
    assert val2 == 200.0


def test_kill_override_attribution_entry_key_is_valid_jp_label() -> None:
    """合成キーは JP_LABEL に登録済み (描画時の KeyError 防止)。"""
    import scripts.visualize_advantage_overlay as vao
    assert KILL_OVERRIDE_DRIVER_KEY_P1 in vao.JP_LABEL
    assert KILL_OVERRIDE_DRIVER_KEY_P2 in vao.JP_LABEL


def test_drivers_for_display_none_note_is_bit_identical() -> None:
    """kill_override_note が None (未発火 or フラグ既定OFF) なら drivers は不変。"""
    drivers = [("board_ojama_count", 1.0), ("current_max_chain", 0.5)]
    assert _drivers_for_display(drivers, None) is drivers


def test_drivers_for_display_prepends_note_and_caps_at_three() -> None:
    """発火時は先頭に挿入され、全体は3件までに収まる。"""
    drivers = [("board_ojama_count", 1.0), ("current_max_chain", 0.5),
               ("diff_max_column_height", 0.3)]
    note = (KILL_OVERRIDE_DRIVER_KEY_P1, 216.0)
    result = _drivers_for_display(drivers, note)
    assert result[0] == note
    assert len(result) == 3
    assert result[1:] == drivers[:2]


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


def test_minimum_prediction_guard_only_during_physical_chain(monkeypatch) -> None:
    """1連鎖40点の下限予測は物理CHAIN中だけ未確定として扱う。"""
    import scripts.visualize_advantage_overlay as vao
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    minimum = _make_chain_event(
        trigger_sec=1.0, total_score=CHAIN_TOTAL_MIN_SCORE,
        score_estimated=True)
    normal = _make_chain_event(trigger_sec=1.0, total_score=1960)
    tracker.update(
        _make_signal(minimum, CHAIN_TOTAL_MIN_SCORE),
        _make_signal(normal, 1960), _make_snapshot(), 0.0)

    assert tracker.has_untrusted_minimum_active_chain(
        BoardState.CHAIN, BoardState.CHAIN)
    assert not tracker.has_untrusted_minimum_active_chain(
        BoardState.STABLE, BoardState.CHAIN)


def test_minimum_prediction_guard_does_not_block_real_multi_chain(monkeypatch) -> None:
    """40点でも chain_count>1 なら最小値への縮退ではないため保留しない。"""
    import scripts.visualize_advantage_overlay as vao
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    minimum = _make_chain_event(
        trigger_sec=1.0, total_score=CHAIN_TOTAL_MIN_SCORE,
        score_estimated=True)
    multi = dataclass_replace(minimum, chain_count=2)
    normal = _make_chain_event(trigger_sec=1.0, total_score=1960)
    tracker.update(
        _make_signal(multi, CHAIN_TOTAL_MIN_SCORE),
        _make_signal(normal, 1960), _make_snapshot(), 0.0)

    assert not tracker.has_untrusted_minimum_active_chain(
        BoardState.CHAIN, BoardState.CHAIN)


def test_minimum_prediction_guard_does_not_freeze_mutual_minimum(monkeypatch) -> None:
    """双方40点なら非対称な過小評価ではなく、小連鎖同士として通常評価する。"""
    import scripts.visualize_advantage_overlay as vao
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    minimum1 = _make_chain_event(
        trigger_sec=1.0, total_score=CHAIN_TOTAL_MIN_SCORE,
        score_estimated=True)
    minimum2 = _make_chain_event(
        trigger_sec=1.0, total_score=CHAIN_TOTAL_MIN_SCORE,
        score_estimated=True)
    tracker.update(
        _make_signal(minimum1, CHAIN_TOTAL_MIN_SCORE),
        _make_signal(minimum2, CHAIN_TOTAL_MIN_SCORE), _make_snapshot(), 0.0)

    assert not tracker.has_untrusted_minimum_active_chain(
        BoardState.CHAIN, BoardState.CHAIN)


def test_minimum_prediction_guard_ends_after_confirmed_score_growth(monkeypatch) -> None:
    """確定得点で再決着した後は、root eventが40点でも新しい予測を採用する。"""
    import scripts.visualize_advantage_overlay as vao
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    minimum = _make_chain_event(
        trigger_sec=1.0, total_score=CHAIN_TOTAL_MIN_SCORE,
        score_estimated=True)
    normal = _make_chain_event(trigger_sec=1.0, total_score=1960)
    tracker.update(
        _make_signal(minimum, CHAIN_TOTAL_MIN_SCORE),
        _make_signal(normal, 1960), _make_snapshot(), 0.0)
    tracker._pred_score1 = 1260.0

    assert not tracker.has_untrusted_minimum_active_chain(
        BoardState.CHAIN, BoardState.CHAIN)


def test_minimum_prediction_display_guard_requires_extreme_direction_flip(
    monkeypatch,
) -> None:
    """下限予測でも、極端でない値や直前STABLEと同方向なら表示を保留しない。"""
    import scripts.visualize_advantage_overlay as vao
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    minimum = _make_chain_event(
        trigger_sec=1.0, total_score=CHAIN_TOTAL_MIN_SCORE,
        score_estimated=True)
    normal = _make_chain_event(trigger_sec=1.0, total_score=1960)
    tracker.update(
        _make_signal(minimum, CHAIN_TOTAL_MIN_SCORE),
        _make_signal(normal, 1960), _make_snapshot(), 0.0)

    tracker.hold_adv = -80.0
    assert vao._minimum_prediction_guard_applies(
        tracker, BoardState.CHAIN, BoardState.CHAIN, 5.0)
    assert not vao._minimum_prediction_guard_applies(
        tracker, BoardState.CHAIN, BoardState.CHAIN, -5.0)
    tracker.hold_adv = -20.0
    assert not vao._minimum_prediction_guard_applies(
        tracker, BoardState.CHAIN, BoardState.CHAIN, 5.0)


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


def test_episode_physical_redecide_follows_unresolved_net_growth(monkeypatch) -> None:
    """進行中の台帳純残量5→21を、連鎖終了を待たず決着へ順次反映する。"""
    import scripts.visualize_advantage_overlay as vao

    stub, score_calls = _stub_score_advantage_factory()
    original = vao.resolve_mutual_exchange
    exchange_calls: list[tuple[int, int, int, int]] = []

    def capture(b1, b2, gen1, gen2, pending1, pending2, simulator=None):
        exchange_calls.append((gen1, gen2, pending1, pending2))
        return original(b1, b2, gen1, gen2, pending1, pending2, simulator)

    monkeypatch.setattr(vao, "_score_advantage", stub)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", capture)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_episode_physical_redecide=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=100)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    s1, s2, snap = _make_signal(ev1, 100), _make_signal(ev2, 300), _make_snapshot()

    tracker.update(s1, s2, snap, 0.0)
    tracker.update(
        s1, s2, snap, 0.1, physical_net_raw=5.0,
        physical_is_unresolved=True)
    tracker.update(
        s1, s2, snap, 0.2, physical_net_raw=5.0,
        physical_is_unresolved=True)
    tracker.update(
        s1, s2, snap, 0.3, physical_net_raw=21.0,
        physical_is_unresolved=True)

    assert len(score_calls) == 3  # 初回 + 5個 + 21個。同値5個では再計算しない
    assert exchange_calls[-2:] == [(5, 0, 0, 0), (21, 0, 0, 0)]
    assert tracker.episode_physical_redecide_count == 2


def test_episode_physical_redecide_default_off_ignores_net(monkeypatch) -> None:
    """新フラグOFFでは台帳純残量を渡しても従来の1回計算だけを維持する。"""
    import scripts.visualize_advantage_overlay as vao

    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=100)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    s1, s2, snap = _make_signal(ev1, 100), _make_signal(ev2, 300), _make_snapshot()
    tracker.update(s1, s2, snap, 0.0)
    tracker.update(
        s1, s2, snap, 0.1, physical_net_raw=21.0,
        physical_is_unresolved=True)
    assert len(calls) == 1


def test_episode_physical_stats_survive_tracker_recreation() -> None:
    """試合境界で本体を再生成しても動画全体の保留母数を保持する。"""
    import scripts.visualize_advantage_overlay as vao

    first = vao.ResolvedExchangeTracker(
        model=object(), enable_episode_physical_consistency_guard=True)
    first._t_sec = 10.0
    first.apply_episode_consistency(
        -20.0, 0.2, 15.0, 0.7, 10.0, 5.0,
        is_unresolved=True, allows_hard_override=False)
    second = vao.ResolvedExchangeTracker(
        model=object(), enable_episode_physical_consistency_guard=True,
        episode_physical_stats=first.episode_physical_stats)
    second._t_sec = 20.0
    second.apply_episode_consistency(
        -30.0, 0.1, 25.0, 0.8, 12.0, 8.0,
        is_unresolved=True, allows_hard_override=False)

    assert second.episode_consistency_fallback_count == 2
    assert second.episode_consistency_fallback_times == [10.0, 20.0]


def test_episode_consistency_holds_three_way_conflict() -> None:
    """40秒局面: 台帳+5・直前モデル+25に対するhold -44を直前値へ保留する。"""
    import scripts.visualize_advantage_overlay as vao

    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_episode_physical_consistency_guard=True)
    tracker._t_sec = 126.467
    adv, p1, applied = tracker.apply_episode_consistency(
        -44.11, 0.0991, 24.20, 0.7684, 25.98, 5.0,
        is_unresolved=True, allows_hard_override=False)
    assert (adv, p1) == pytest.approx((24.20, 0.7684))
    assert applied is True
    assert tracker.episode_consistency_fallback_times == [126.467]


@pytest.mark.parametrize("stable", [2.9, -2.9, -84.0])
def test_episode_consistency_uses_model_when_stable_memory_is_polluted(
    stable: float,
) -> None:
    """旧hold由来のstable値が弱い/逆でも、台帳+生モデル一致を採用する。"""
    import scripts.visualize_advantage_overlay as vao

    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_episode_physical_consistency_guard=True)
    adv, p1, applied = tracker.apply_episode_consistency(
        -44.0, 0.1, stable, 0.7, 25.0, 5.0,
        is_unresolved=True, allows_hard_override=False)
    assert adv == 25.0
    assert p1 == pytest.approx(vao.adv_to_winprob(25.0))
    assert applied is True


@pytest.mark.parametrize(
    ("stable", "model"), [(24.0, 2.9), (24.0, -2.9)],
)
def test_episode_consistency_does_not_use_even_model_as_direction_vote(
    stable: float, model: float,
) -> None:
    """生モデルがEVEN帯なら、台帳だけを根拠にholdを差し替えない。"""
    import scripts.visualize_advantage_overlay as vao

    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_episode_physical_consistency_guard=True)
    adv, p1, applied = tracker.apply_episode_consistency(
        -44.0, 0.1, stable, 0.7, model, 5.0,
        is_unresolved=True, allows_hard_override=False)
    assert (adv, p1) == (-44.0, 0.1)
    assert applied is False


@pytest.mark.parametrize(
    "candidate,stable,model,net,unresolved,allows",
    [
        (44.0, 24.0, 25.0, 5.0, True, False),   # holdも台帳と同方向
        (-44.0, 24.0, -25.0, 5.0, True, False), # 直前モデルは2P支持
        (-44.0, 24.0, 25.0, 5.0, False, False), # episode解決済み
        (-44.0, 24.0, 25.0, 5.0, True, True),   # 物理的なhard確定あり
    ],
)
def test_episode_consistency_does_not_mask_supported_or_final_result(
    candidate, stable, model, net, unresolved, allows,
) -> None:
    """同方向・別根拠・解決済み・物理確定の結論は保留しない。"""
    import scripts.visualize_advantage_overlay as vao

    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_episode_physical_consistency_guard=True)
    adv, _p1, applied = tracker.apply_episode_consistency(
        candidate, 0.2, stable, 0.7, model, net,
        is_unresolved=unresolved, allows_hard_override=allows)
    assert adv == candidate
    assert applied is False


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
# 指摘13: 片側のみ連鎖中のライブ受け側再評価
# (docs/DEMO_REVIEW_2026-08-13.md #13、2026-08-15)
# ============================
# 「84%で固定でなく、2Pのみ連鎖中なら受け側の状況は0.5秒ごとに変わるので
# 動くのが普通では」への対処。攻撃側の帰結 (飛来量・仮想盤面 board_pX_after)
# は凍結維持しつつ、受け側の現在盤面 (呼出側 generate() の sticky b1/b2) +
# 残り時間予算の逓減でモデル評価/決定度増幅を再計算する。新フラグ
# enable_live_defender_reeval (既定OFF)。


def _stub_exchange_result_distinct_boards(
    dropped_to_p1: int = 0, dropped_to_p2: int = 0,
) -> types.SimpleNamespace:
    """board_p1_after/board_p2_after/pre_landing の4枚全てに識別可能な別々の
    Board を持つ stub (どの盤面が実際に使われたかを `is` で厳密確認できる)。"""
    return types.SimpleNamespace(
        board_p1_after=_board_with_ojama(11), board_p2_after=_board_with_ojama(12),
        board_p1_pre_landing=_board_with_ojama(13), board_p2_pre_landing=_board_with_ojama(14),
        dropped_to_p1=dropped_to_p1, dropped_to_p2=dropped_to_p2,
        leftover_p1=0, leftover_p2=0,
        p1_dead=False, p2_dead=False,
        chain_result_p1=types.SimpleNamespace(chain_count=1),
        chain_result_p2=types.SimpleNamespace(chain_count=1),
    )


def test_live_defender_reeval_default_off_is_bit_identical(monkeypatch) -> None:
    """既定 False では b1/b2 を渡しても片側終了フェーズは完全凍結のまま
    (backwards compat、既存 test_resolved_holds_when_only_one_side_ends_first
    と同じ前提を b1/b2 明示指定で再確認)。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())  # enable_live_defender_reeval省略=False
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    held = tracker.hold_adv
    assert len(calls) == 1
    live_board = _board_with_ojama(5)
    active, _ = tracker.update(
        _make_signal(None, 500), _make_signal(ev2, 300), _make_snapshot(), 1.0,
        b1=live_board, b2=live_board)
    assert active is True
    assert tracker.hold_adv == held  # 生盤面を渡しても凍結が優先される
    assert len(calls) == 1  # _score_advantage は再度呼ばれない


def test_live_defender_reeval_skipped_when_defender_is_physically_busy(monkeypatch) -> None:
    """イベントが両側に残っていても、受け側が物理連鎖中なら凍結する。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_live_defender_reeval=True,
        enable_live_defender_strict=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert len(calls) == 1
    live_board = _board_with_ojama(1)
    signal1 = types.SimpleNamespace(
        chain_event=ev1, score=500, state=BoardState.CHAIN)
    signal2 = types.SimpleNamespace(
        chain_event=ev2, score=300, state=BoardState.CHAIN)
    tracker.update(signal1, signal2, _make_snapshot(), 1.0,
                   t_sec=1.0, b1=live_board, b2=live_board)
    assert len(calls) == 1  # 受け側の物理 state が busy の間は再評価しない


def test_live_defender_reeval_runs_when_event_is_stale_but_defender_is_free(
    monkeypatch,
) -> None:
    """両側eventが残っても、受け側STABLEなら現在盤面を再評価する。"""
    import scripts.visualize_advantage_overlay as vao

    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_live_defender_reeval=True,
        enable_live_defender_strict=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert len(calls) == 1

    signal1 = types.SimpleNamespace(
        chain_event=ev1, score=500, state=BoardState.STABLE)
    signal2 = types.SimpleNamespace(
        chain_event=ev2, score=300, state=BoardState.CHAIN)
    tracker.update(
        signal1, signal2, _make_snapshot(), 1.0, t_sec=1.0,
        b1=_board_with_ojama(1), b2=_board_with_ojama(1))

    assert len(calls) == 2
    assert tracker.hold_adv == 20.0


def test_live_defender_reeval_feeds_live_defender_board_and_frozen_attacker_board(
    monkeypatch,
) -> None:
    """片側のみ連鎖中 (1P終了=受け側、2P継続=攻撃側) の間、モデル評価に渡る
    盤面は 1P=受け側の未着弾分を物理着弾させた仮想盤面 (方向反転修正、
    2026-08-15、W12対処)・2P=凍結仮想盤面 (board_p2_after) になる
    (攻撃側の生盤面は連鎖アニメ中で信用できないため使わない)。生盤面を
    そのまま渡すと「降るまでは無傷」誤判定になる (指摘13方向反転の根因)
    ため、`land_pending_ojama_onto_board` で別オブジェクトへ差し替わる。"""
    import scripts.visualize_advantage_overlay as vao
    from src.indicators_v2 import board_ojama_count

    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    captured_boards: list[tuple] = []

    def _stub(model, b1, b2, snap, feature_cols=None, attribution_exclude=()):
        captured_boards.append((b1, b2))
        return 0.0, 0.5, []

    monkeypatch.setattr(vao, "_score_advantage", _stub)
    tracker = vao.ResolvedExchangeTracker(model=object(), enable_live_defender_reeval=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert len(captured_boards) == 1  # 初回決着計算

    live_board = _board_with_ojama(9)
    # 1P (受け側=dropped_to_p1=30) の chain_event だけ None化、2P (攻撃側) は継続中。
    # 会計 snap は既定 (pending=0) のため板差分フォールバックが効く:
    # incoming_total_p1=30 (dropped30+leftover0)、base(ev1.before_board)=0、
    # target=30。current(live_board)=9 → remaining=21 (<OJAMA_MAX_DROP_PER_TURN)。
    tracker.update(_make_signal(None, 500), _make_signal(ev2, 300), _make_snapshot(), 1.0,
                   t_sec=1.0, b1=live_board, b2=None)

    assert len(captured_boards) == 2  # ライブ再評価でもう1回呼ばれる
    b1_used, b2_used = captured_boards[1]
    assert b1_used is not live_board  # 生盤面のままではない(物理着弾済み仮想盤面)
    assert board_ojama_count(b1_used).raw == pytest.approx(30.0)  # 9(既着弾)+21(今回着弾)=30
    assert b2_used is result_stub.board_p2_after  # 攻撃側 (2P) は凍結仮想盤面のまま
    assert b2_used is not result_stub.board_p2_pre_landing


def test_live_defender_reeval_noop_when_defender_board_not_yet_observed(monkeypatch) -> None:
    """受け側の STABLE 盤面をまだ一度も観測していない (b1/b2 が None) 間は
    再評価せず直前の保持値を維持する (安全側、段差回避)。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object(), enable_live_defender_reeval=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    held = tracker.hold_adv
    active, _ = tracker.update(
        _make_signal(None, 500), _make_signal(ev2, 300), _make_snapshot(), 1.0, t_sec=1.0)
    assert active is True
    assert tracker.hold_adv == held
    assert len(calls) == 1  # ライブ再評価は呼ばれない (盤面欠損でno-op)


def test_live_defender_reeval_noop_when_no_threat(monkeypatch) -> None:
    """相殺で飛来量が無い局面ではライブ再評価対象の受け側が無く no-op
    (指摘10の noop テストと同じ前提を指摘13ライブ経路でも確認)。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    monkeypatch.setattr(vao, "resolve_mutual_exchange",
                        lambda *a, **k: _stub_mutual_exchange_result())
    tracker = vao.ResolvedExchangeTracker(model=object(), enable_live_defender_reeval=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    held = tracker.hold_adv
    live_board = _board_with_ojama(1)
    tracker.update(_make_signal(None, 500), _make_signal(ev2, 300), _make_snapshot(), 1.0,
                   t_sec=1.0, b1=live_board)
    assert tracker.hold_adv == held
    assert len(calls) == 1


def test_live_defender_reeval_respects_recompute_interval_throttle(monkeypatch) -> None:
    """再評価は COUNTER_RECOMPUTE_INTERVAL_SEC (0.5秒) 未満の連続呼び出しでは
    間引かれ、0.5秒以上経過すると再計算される (既存の応手判定周期と同一)。"""
    import scripts.visualize_advantage_overlay as vao

    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    calls: list[int] = []

    def _stub(model, b1, b2, snap, feature_cols=None, attribution_exclude=()):
        calls.append(1)
        return float(len(calls)), 0.5, []

    monkeypatch.setattr(vao, "_score_advantage", _stub)
    tracker = vao.ResolvedExchangeTracker(model=object(), enable_live_defender_reeval=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert len(calls) == 1

    live_board = _board_with_ojama(1)
    tracker.update(_make_signal(None, 500), _make_signal(ev2, 300), _make_snapshot(), 1.0,
                   t_sec=1.0, b1=live_board)
    assert len(calls) == 2  # 決着直後は間引きなしで即評価 (段差回避)

    # 0.5秒未満の再呼び出しは間引かれる (再計算されない)。
    tracker.update(_make_signal(None, 500), _make_signal(ev2, 300), _make_snapshot(), 1.2,
                   t_sec=1.2, b1=live_board)
    assert len(calls) == 2

    # 0.5秒以上経過した呼び出しでは再計算される。
    tracker.update(_make_signal(None, 500), _make_signal(ev2, 300), _make_snapshot(), 1.6,
                   t_sec=1.6, b1=live_board)
    assert len(calls) == 3


def test_live_defender_reeval_budget_decreases_with_elapsed_time_and_updates_display(
    monkeypatch,
) -> None:
    """決定度増幅も併用時、時間予算は `_chain_remaining_time_budget_sec` に
    現在の t_sec を都度渡すことで経過時間ぶん自然に減り (残り時間の逓減)、
    ホールド表示専用フィールド (hold_defender_prob 等) も追従する。"""
    import scripts.visualize_advantage_overlay as vao

    monkeypatch.setattr(vao, "_load_chain_length_conditional_table", lambda *a, **k: {})
    stub, _ = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    captured_budgets: list[float] = []

    def fake_counter_update(self, b1, b2, budget, **kw):
        captured_budgets.append(budget)
        return (0.0, 0.4, float("nan"))

    monkeypatch.setattr(vao.CounterReachTracker, "update", fake_counter_update)
    monkeypatch.setattr(vao, "_counter_defender_adv", lambda *a, **k: -2.0)

    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_decisive_amplify=True, enable_live_defender_reeval=True)
    ev1 = _make_chain_event(trigger_sec=100.0, total_score=500)
    # 攻撃側 (2P、defender=1P なので attacker_event=ev2) は6連鎖・t=100.0発火。
    ev2 = ChainEvent(
        trigger_sec=100.0, end_sec=101.0, before_board=Board(),
        chain_count=6, total_erased=0, total_score=300, base_score=0,
        all_clear_bonus_applied=0, ojama_sent=0, leftover_score=0, is_all_clear=False,
    )
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(),
                   0.0, t_sec=100.0)
    assert len(captured_budgets) == 1  # 初回決着計算内の増幅

    live_board = _board_with_ojama(1)
    # 1P(受け側)が先に終了、2Pの連鎖アニメはまだ続いている。発火から1.5秒経過。
    tracker.update(_make_signal(None, 500), _make_signal(ev2, 300), _make_snapshot(),
                   1.0, t_sec=101.5, b1=live_board)
    assert len(captured_budgets) == 2

    expected = vao._chain_remaining_time_budget_sec(6, 100.0, 101.5, {})
    assert captured_budgets[1] == pytest.approx(expected)
    assert captured_budgets[1] < captured_budgets[0]  # 経過時間ぶん残り予算が減っている
    assert tracker.hold_defender_prob == pytest.approx(0.4)  # 表示フィールドも追従
    assert tracker.hold_defender_side == "1P"


def test_live_defender_snap_forecast_uses_leftover_after_live_landing() -> None:
    """[方向反転修正、2026-08-15] 受け側の forecast は「このフレームで物理
    着弾させた後に残った未着弾分」(leftover_now、呼出側が
    `land_pending_ojama_onto_board` から得る) に差し替わる。旧実装 (全量
    self._incoming_total_pX を forecast に積む方式) はモデルが forecast を
    ほぼ無視する (W12) ため方向反転を解消できず撤回した。凍結経路
    `_resolve()` と同じ「forecast=leftover」意味論に揃う。"""
    import scripts.visualize_advantage_overlay as vao
    from src.ojama_accounting import OjamaAccountSnapshot

    tracker = vao.ResolvedExchangeTracker(model=object())
    tracker._incoming_total_p1 = 42.0  # dropped_to_p1(30) + leftover_p1(12) 相当
    tracker._incoming_total_p2 = 5.0
    tracker._resolved_snap = OjamaAccountSnapshot(
        t_sec=0.0, pending_p1=0, pending_p2=0,
        total_generated_by_p1=0, total_generated_by_p2=0,
        total_offset_by_p1=0, total_offset_by_p2=0,
        total_dropped_to_p1=0, total_dropped_to_p2=0,
        net_ojama_balance=0,
        overflow_risk_p1=False, overflow_risk_p2=False, confidence=1.0,
        leftover_p1=0, leftover_p2=0,
        all_clear_pending_p1=False, all_clear_pending_p2=False,
        chain_end_triggered_p1=False, chain_end_triggered_p2=False,
        chain_total_score_p1=0, chain_total_score_p2=0,
        net_balance_capped=5.0 - 12.0, forecast_p1=12.0, forecast_p2=5.0,
    )

    # 例: 今回のライブ再評価で12個中12個すべて着弾済み(leftover_now=0)とする。
    snap_1p_defender = tracker._live_defender_snap("1P", leftover_now=0.0)
    assert snap_1p_defender.forecast_p1 == pytest.approx(0.0)  # 着弾済み=盤面が語る、forecastは0
    assert snap_1p_defender.forecast_p2 == pytest.approx(5.0)  # 攻撃側は元の leftover のまま
    assert snap_1p_defender.net_balance_capped == pytest.approx(5.0 - 0.0)

    # OJAMA_MAX_DROP_PER_TURN 超過等で一部が着弾しきれず残った場合 (leftover_now=8)。
    snap_2p_defender = tracker._live_defender_snap("2P", leftover_now=8.0)
    assert snap_2p_defender.forecast_p1 == pytest.approx(12.0)  # 攻撃側(1P)は元のまま
    assert snap_2p_defender.forecast_p2 == pytest.approx(8.0)  # defender側だけ残量に差し替わる
    assert snap_2p_defender.net_balance_capped == pytest.approx(8.0 - 12.0)


def test_live_defender_reeval_passes_landed_leftover_forecast_to_score_advantage(
    monkeypatch,
) -> None:
    """`_reevaluate_live_defender` が `_score_advantage` へ渡す snap は
    `_live_defender_snap` 差し替え後の値であることを配線レベルで確認する
    (self._resolved_snap をそのまま渡す旧実装への回帰を防止)。方向反転修正
    (2026-08-15) 後は forecast が「物理着弾させた残り (leftover_now)」に
    揃う (盤面側で着弾済み分を表現するため forecast は全量ではない)。"""
    import scripts.visualize_advantage_overlay as vao

    # dropped_to_p1=30, leftover_p1=10 -> incoming_total_p1=40 (leftoverのみの10とは異なる値)
    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    result_stub.leftover_p1 = 10
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    captured_snaps: list = []

    def _stub(model, b1, b2, snap, feature_cols=None, attribution_exclude=()):
        captured_snaps.append(snap)
        return 0.0, 0.5, []

    monkeypatch.setattr(vao, "_score_advantage", _stub)
    tracker = vao.ResolvedExchangeTracker(model=object(), enable_live_defender_reeval=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert len(captured_snaps) == 1
    assert captured_snaps[0].forecast_p1 == pytest.approx(10.0)  # 初回決着 = leftoverのまま

    live_board = _board_with_ojama(9)
    # incoming_total_p1 = dropped(30)+leftover(10) = 40。base(ev1.before_board)=0、
    # target=40。current(live_board)=9 → remaining=31 (OJAMA_MAX_DROP_PER_TURN=30で
    # キャップされ dropped_now=30、leftover_now=1)。
    tracker.update(_make_signal(None, 500), _make_signal(ev2, 300), _make_snapshot(), 1.0,
                   t_sec=1.0, b1=live_board, b2=None)

    assert len(captured_snaps) == 2
    # 旧実装 (forecast=incoming全量40) への回帰でないこと、かつ初回決着の
    # leftover(10)のままでもないこと (=盤面着弾+forecast残りの二重計上防止)。
    assert captured_snaps[1].forecast_p1 == pytest.approx(1.0)
    assert captured_snaps[1].forecast_p1 != pytest.approx(40.0)
    assert captured_snaps[1].forecast_p1 != pytest.approx(10.0)


# ============================
# _live_remaining_incoming: 会計優先・盤面差分フォールバック
# (方向反転修正、2026-08-15、docs/KNOWN_WEAKNESSES.md W12)
# ============================


def test_live_remaining_incoming_prefers_accounting_snapshot_when_available() -> None:
    """会計スナップショット (`snap.pending_pX`) が正の値を持つ間は、実際に
    tsumo 設置で降り進んだ分が自然に反映された値をそのまま使う (盤面差分
    フォールバックより優先、二重計上防止の一次ソース)。"""
    import scripts.visualize_advantage_overlay as vao

    tracker = vao.ResolvedExchangeTracker(model=object())
    tracker._incoming_total_p1 = 40.0
    tracker._target_ojama_p1 = 40.0  # base(0)+incoming_total(40)
    snap = _make_snapshot(pending_p1=15, pending_p2=0)  # 40個中25個は既に降り進んだ想定
    live_board = _board_with_ojama(0)  # 盤面はまだ空(会計が正なら盤面は見ない)

    remaining = tracker._live_remaining_incoming("1P", live_board, snap)
    assert remaining == pytest.approx(15.0)  # 会計値をそのまま採用


def test_live_remaining_incoming_caps_accounting_value_at_frozen_incoming_total() -> None:
    """会計値が凍結時の飛来予測を上回っても、凍結時飛来総量を超えない
    (今回の交換由来分だけを対象にする、無関係な後続予告の混入防止)。"""
    import scripts.visualize_advantage_overlay as vao

    tracker = vao.ResolvedExchangeTracker(model=object())
    tracker._incoming_total_p1 = 40.0
    tracker._target_ojama_p1 = 40.0
    snap = _make_snapshot(pending_p1=999, pending_p2=0)
    live_board = _board_with_ojama(0)

    remaining = tracker._live_remaining_incoming("1P", live_board, snap)
    assert remaining == pytest.approx(40.0)  # incoming_total 上限でクリップ


def test_live_remaining_incoming_falls_back_to_board_delta_when_accounting_zeroed() -> None:
    """会計が0を示す (baseline reset 等でこの交換を追跡できていない) のに
    凍結時飛来予測が正の場合、盤面上で既に増えた分を凍結時飛来量から
    控除した残りへフォールバックする (二重計上防止)。"""
    import scripts.visualize_advantage_overlay as vao

    tracker = vao.ResolvedExchangeTracker(model=object())
    tracker._incoming_total_p1 = 40.0
    tracker._target_ojama_p1 = 40.0  # base(0)+incoming_total(40)
    snap = _make_snapshot(pending_p1=0, pending_p2=0)  # 会計リセット済み(取れない)
    live_board = _board_with_ojama(9)  # 盤面には既に9個降り進んでいる

    remaining = tracker._live_remaining_incoming("1P", live_board, snap)
    assert remaining == pytest.approx(31.0)  # 40 - 9


def test_live_remaining_incoming_none_snapshot_uses_board_delta_fallback() -> None:
    """snap=None (省略、backwards compat) でも盤面差分フォールバックで安全に
    動作する (会計未配線の呼出元でも no-op にならない)。"""
    import scripts.visualize_advantage_overlay as vao

    tracker = vao.ResolvedExchangeTracker(model=object())
    tracker._incoming_total_p1 = 40.0
    tracker._target_ojama_p1 = 40.0
    live_board = _board_with_ojama(9)

    remaining = tracker._live_remaining_incoming("1P", live_board, None)
    assert remaining == pytest.approx(31.0)


def test_live_remaining_incoming_zero_when_defender_side_has_no_incoming() -> None:
    """今回の交換で受け側でない側 (incoming_total<=0) は常に0
    (会計・盤面のノイズに関わらず脅威を作り出さない)。"""
    import scripts.visualize_advantage_overlay as vao

    tracker = vao.ResolvedExchangeTracker(model=object())
    tracker._incoming_total_p2 = 0.0
    tracker._target_ojama_p2 = 0.0
    snap = _make_snapshot(pending_p1=0, pending_p2=999)  # 別枠の予告(無関係)混入を模擬
    live_board = _board_with_ojama(9)

    remaining = tracker._live_remaining_incoming("2P", live_board, snap)
    assert remaining == pytest.approx(0.0)


def test_reevaluate_live_defender_landed_board_plus_leftover_equals_target_no_double_count(
    monkeypatch,
) -> None:
    """[二重計上防止の直接テスト] 物理着弾後の盤面おじゃま数 + forecast
    (leftover_now) の和が、凍結時に見積もった着弾目標総量
    (`_target_ojama_p1` = base + incoming_total) とちょうど一致すること
    (盤面が語る分と forecast が語る分が重複も欠落もしていない不変条件)。"""
    import scripts.visualize_advantage_overlay as vao
    from src.indicators_v2 import board_ojama_count

    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    result_stub.leftover_p1 = 10  # incoming_total_p1 = 40
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    captured: list = []

    def _stub(model, b1, b2, snap, feature_cols=None, attribution_exclude=()):
        captured.append((b1, snap))
        return 0.0, 0.5, []

    monkeypatch.setattr(vao, "_score_advantage", _stub)
    tracker = vao.ResolvedExchangeTracker(model=object(), enable_live_defender_reeval=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    live_board = _board_with_ojama(9)
    tracker.update(_make_signal(None, 500), _make_signal(ev2, 300), _make_snapshot(), 1.0,
                   t_sec=1.0, b1=live_board, b2=None)

    assert len(captured) == 2
    b1_used, snap_used = captured[1]
    assert board_ojama_count(b1_used).raw + snap_used.forecast_p1 == pytest.approx(
        tracker._target_ojama_p1)


# ============================
# 指摘14 案1: enable_live_defender_strict (既定OFF)
# ============================
# 【重要】案1は実装→A/B実測→coordinator指摘 (「ev1/ev2 ベースの向きが逆」)
# →実機構の計装確認→やり直し、という経緯を辿った (2026-08-15)。
# 計装 (scripts/_diag_issue14_reeval_calls_2026-08-15.py、対象=
# review_demo_2026-08-12.mp4 絶対t=195.3秒) で誤爆の初回発生フレームを
# 直接確認したところ:
#   ev1_cc=9(1P、継続中) / ev2=None(defender=2P) → 旧XOR条件は成立
#   だが **defender(2P)自身の状態機械 state は BoardState.GRAVITY_SETTLE**
#   (直前の小連鎖の消去は終わったが重力settle中でまだ静止していない) で
#   あり、次のフレームで2Pは別の本物の7連鎖を新規発火させていた。
# 根因: ChainEvent は「trigger 検知フレームで1度だけ発行され
# chain_hold_base_sec+chain_hold_per_step_sec×chain_count 秒だけ保持後
# None に戻る」パルス方式 (src/chain_detector.py VideoChainTracker) のため、
# 旧連鎖の hold が切れてから新連鎖の trigger が検出されるまでの settle gap
# で ev が None になる瞬間が生じる。この gap は defender が真に「自由」
# なのではなく重力settleの真っ最中であり、chain_event の有無だけでは
# 判定できない (案1初版はここを見誤り、A/B実測で baseline と1桁も
# 変わらず=無効化していた)。
# 対処: 状態機械の state (毎フレーム直接観測、chain_event のようなパルス
# +hold-window方式ではない) を使う。defender_side 自身の state が
# `_LIVE_DEFENDER_BUSY_STATES` (= {CHAIN, GRAVITY_SETTLE}) に含まれる間は
# 再評価をスキップする。TSUMO_FALL/OJAMA_FALL は busy 扱いしない
# (指摘13が意図した「受け側は連鎖中も置き続ける」正当な自由行動を
# 塞がないため)。


def test_live_defender_strict_default_off_still_misfires_on_settle_gap(monkeypatch) -> None:
    """strict省略時 (False、backwards compat) は、defender_side (2P) 自身が
    実際には GRAVITY_SETTLE 中 (=直前の連鎖の重力settleの真っ最中、実測
    595.3秒地点の settle gap を再現) でも、従来の XOR 条件だけで再評価が
    起動してしまう (指摘14の誤爆が既定 OFF 時は温存されていることの確認、
    退行検出用)。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p2=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    tracker = vao.ResolvedExchangeTracker(model=object(), enable_live_defender_reeval=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert len(calls) == 1
    live_board = _board_with_ojama(1)
    # 1P (攻撃側) は継続中。2P (defender=_decisive_defender の判定、
    # dropped_to_p2=30) は ev2 が settle gap で None だが実際は
    # GRAVITY_SETTLE 中 (実測パターンの再現)。
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(None, 300, state=BoardState.GRAVITY_SETTLE),
        _make_snapshot(), 1.0, t_sec=1.0, b1=None, b2=live_board)
    assert len(calls) == 2  # strict省略=False なので誤爆再評価が起きる (旧経路と同一)


def test_live_defender_strict_skips_reeval_during_settle_gap(monkeypatch) -> None:
    """[指摘14 案1本体、実機構の直接再現] strict=True では、defender_side
    (2P) 自身が GRAVITY_SETTLE 中 (実測 t=195.3秒の settle gap と同一パターン、
    ev2 は None だが state2 は busy) の間は再評価をスキップし直前の保持値を
    維持する (誤爆修正の直接確認)。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p2=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_live_defender_reeval=True, enable_live_defender_strict=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert len(calls) == 1
    held = tracker.hold_adv
    live_board = _board_with_ojama(1)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(None, 300, state=BoardState.GRAVITY_SETTLE),
        _make_snapshot(), 1.0, t_sec=1.0, b1=None, b2=live_board)
    assert len(calls) == 1  # strict=True で誤爆を回避、再評価されない
    assert tracker.hold_adv == held  # 保持値も直前のまま (退行なし)


def test_live_defender_strict_skips_when_defender_state_is_chain(monkeypatch) -> None:
    """defender_side 自身の state が CHAIN (ev がまだ発行されていない/検知
    ラグ中でも、盤面上は直接 CHAIN と分かる) でも同様にスキップする
    (busy 判定は CHAIN と GRAVITY_SETTLE の両方が対象)。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p2=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_live_defender_reeval=True, enable_live_defender_strict=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    held = tracker.hold_adv
    live_board = _board_with_ojama(1)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(None, 300, state=BoardState.CHAIN),
        _make_snapshot(), 1.0, t_sec=1.0, b1=None, b2=live_board)
    assert len(calls) == 1
    assert tracker.hold_adv == held


def test_live_defender_strict_still_reevaluates_when_defender_state_stable(
    monkeypatch,
) -> None:
    """[指摘14 案1の副作用チェック] defender_side (1P) 自身の state が
    STABLE (=本当に自由行動中) の正当なケースは、strict=True でも従来通り
    再評価される (指摘13が意図した挙動を壊さないことの確認、既存の
    test_live_defender_reeval_feeds_live_defender_board_and_frozen_attacker_board
    と同じ盤面組み合わせを strict=True で再確認)。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_live_defender_reeval=True, enable_live_defender_strict=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert len(calls) == 1
    live_board = _board_with_ojama(9)
    # 1P (defender、dropped_to_p1=30) 自身の state が STABLE (=本当に自由
    # 行動中)、2P (攻撃側) は継続中。
    tracker.update(
        _make_signal_full(None, 500, state=BoardState.STABLE),
        _make_signal_full(ev2, 300, state=BoardState.CHAIN),
        _make_snapshot(), 1.0, t_sec=1.0, b1=live_board, b2=None)
    assert len(calls) == 2  # 正当なケースは strict でも再評価される


def test_live_defender_strict_reevaluates_stable_nondefender_while_defender_busy(
    monkeypatch,
) -> None:
    """[2026-08-27 userレビュー 3:50] 初回STABLE復帰は基準化だけにし、
    hold後に開始→終了を両方観測した新規side-local chainの前後差分だけを
    decisive holdへ加える。chain_endは補正トリガーでなくARMにだけ使う。"""
    import scripts.visualize_advantage_overlay as vao

    captured: list[tuple[Board, Board, object]] = []

    def _stub(model, b1, b2, snap, feature_cols=None, attribution_exclude=()):
        captured.append((b1, b2, snap))
        n = len(captured)
        return float(n * 10), 0.5 + n * 0.05, []

    monkeypatch.setattr(vao, "_score_advantage", _stub)
    # decisive defender=1P。ただし1PはまだCHAIN中で、2PだけSTABLEへ復帰する。
    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_live_defender_reeval=True,
        enable_live_defender_strict=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(
        _make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0,
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    first_stable_p2 = _board_with_ojama(3)

    # 初回STABLE復帰 (実動画 source=312.933) はARMだけで、値を変えない。
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 300, state=BoardState.STABLE),
        _make_snapshot(chain_end_triggered_p2=True, chain_total_score_p2=300),
        1.0, t_sec=1.0, b1=None, b2=first_stable_p2,
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    assert len(captured) == 1
    assert tracker.hold_adv == pytest.approx(10.0)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 300, state=BoardState.TSUMO_FALL),
        _make_snapshot(), 1.5, t_sec=1.5, b1=None, b2=first_stable_p2,
        physical_chain_id_p1=11, physical_chain_id_p2=12)

    # その後の新規1連鎖 (実動画 source=316.5→318.467) を開始から観測。
    before_chain_p2 = _make_4connect_board()
    follow_ev2 = _make_chain_event(
        trigger_sec=2.0, before_board=before_chain_p2, total_score=50)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(follow_ev2, 300, state=BoardState.CHAIN),
        _make_snapshot(), 2.0, t_sec=2.0, b1=None, b2=before_chain_p2,
        physical_chain_id_p1=11, physical_chain_id_p2=13)
    after_chain_p2 = Board()
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(follow_ev2, 350, state=BoardState.STABLE),
        _make_snapshot(chain_end_triggered_p1=True, chain_total_score_p1=700),
        3.0, t_sec=3.0, b1=None, b2=after_chain_p2,
        physical_chain_id_p1=11, physical_chain_id_p2=13)

    # 相手側の確定得点で再決着しても追跡を失わない。paired-delta は仮想の
    # result側盤面でなく、同じ実盤面系列の before→after を比較する。
    assert len(captured) == 4
    assert captured[2][0] is result_stub.board_p1_after  # busy側だけ同じ盤面に凍結
    assert captured[3][0] is result_stub.board_p1_after
    assert captured[2][1] == before_chain_p2
    assert captured[2][1] is not result_stub.board_p2_after
    assert captured[3][1] is after_chain_p2
    assert captured[2][2] is captured[3][2] is tracker._resolved_snap
    assert tracker.hold_adv == pytest.approx(30.0)
    assert tracker.nondef_cycle_stats["applied"] == 1


def test_live_defender_strict_does_not_use_nonstable_nondefender_board(
    monkeypatch,
) -> None:
    """反対側も非STABLEなら、sticky盤面が渡っていても新しい確定盤面とは
    見なさず、両側とも決着値を凍結する。"""
    import scripts.visualize_advantage_overlay as vao

    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_live_defender_reeval=True,
        enable_live_defender_strict=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(
        _make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)

    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 300, state=BoardState.TSUMO_FALL),
        _make_snapshot(), 1.0, t_sec=1.0, b1=None, b2=_board_with_ojama(3))

    assert len(calls) == 1


def test_nondefender_cycle_does_not_arm_on_fragment_stable_without_chain_end(
    monkeypatch,
) -> None:
    """同一本線の断片間STABLE gapだけではARMせず、後続CHAINを新しい応手
    連鎖として数えない。物理連鎖終了の確認が必要。"""
    import scripts.visualize_advantage_overlay as vao

    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_live_defender_reeval=True,
        enable_live_defender_strict=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(
        _make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0,
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 300, state=BoardState.STABLE),
        _make_snapshot(), 1.0, t_sec=1.0, b1=None, b2=_board_with_ojama(4))
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(
            _make_chain_event(2.0, before_board=_board_with_ojama(4), total_score=1000),
            300, state=BoardState.CHAIN),
        _make_snapshot(), 2.0, t_sec=2.0, b1=None, b2=_board_with_ojama(4))
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 1300, state=BoardState.STABLE),
        _make_snapshot(), 3.0, t_sec=3.0, b1=None, b2=_board_with_ojama(1))

    assert len(calls) == 1
    assert tracker.nondef_cycle_stats["armed"] == 0
    assert tracker.nondef_cycle_stats["started"] == 0
    assert tracker.nondef_cycle_stats["applied"] == 0


def test_nondefender_cycle_rejects_direct_reopen_without_new_hand(monkeypatch) -> None:
    """終了候補でARMしても、新ツモ設置を挟まず直接CHAINへ戻る段継続は
    新しい応手連鎖として開始しない。"""
    import scripts.visualize_advantage_overlay as vao

    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_live_defender_reeval=True,
        enable_live_defender_strict=True)
    ev1 = _make_chain_event(1.0, total_score=500)
    ev2 = _make_chain_event(1.0, total_score=300)
    tracker.update(
        _make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0,
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 300, state=BoardState.STABLE),
        _make_snapshot(chain_end_triggered_p2=True, chain_total_score_p2=300),
        1.0, t_sec=1.0, b1=None, b2=_make_4connect_board(),
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(
            _make_chain_event(2.0, before_board=_make_4connect_board(), total_score=1000),
            300, state=BoardState.CHAIN),
        _make_snapshot(), 2.0, t_sec=2.0, b1=None, b2=_make_4connect_board(),
        physical_chain_id_p1=11, physical_chain_id_p2=13)

    assert len(calls) == 1
    assert tracker.nondef_cycle_stats["armed"] == 1
    assert tracker.nondef_cycle_stats["started"] == 0
    assert tracker.nondef_cycle_stats["rejected_no_new_hand"] == 1


def test_nondefender_cycle_ignores_stale_chain_end_and_subminimum_score(
    monkeypatch,
) -> None:
    """残留したchain_endフラグだけでは補正せず、開始を観測しても得点増分が
    40未満なら新規連鎖として採用しない。終了後のSTABLE反復でも再適用しない。"""
    import scripts.visualize_advantage_overlay as vao

    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_live_defender_reeval=True,
        enable_live_defender_strict=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(
        _make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0,
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    stale_end = _make_snapshot(chain_end_triggered_p2=True, chain_total_score_p2=300)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 300, state=BoardState.STABLE),
        stale_end, 1.0, t_sec=1.0, b1=None, b2=_board_with_ojama(4))
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 300, state=BoardState.TSUMO_FALL),
        _make_snapshot(), 1.5, t_sec=1.5, b1=None, b2=_board_with_ojama(4),
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(
            _make_chain_event(2.0, before_board=_board_with_ojama(4), total_score=39),
            300, state=BoardState.CHAIN),
        stale_end, 2.0, t_sec=2.0, b1=None, b2=_board_with_ojama(4),
        physical_chain_id_p1=11, physical_chain_id_p2=13)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 339, state=BoardState.STABLE),
        stale_end, 3.0, t_sec=3.0, b1=None, b2=_board_with_ojama(1),
        physical_chain_id_p1=11, physical_chain_id_p2=13)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 339, state=BoardState.STABLE),
        stale_end, 4.0, t_sec=4.0, b1=None, b2=_board_with_ojama(0),
        physical_chain_id_p1=11, physical_chain_id_p2=13)

    assert len(calls) == 1
    assert tracker.nondef_cycle_stats["rejected_score"] == 1
    assert tracker.nondef_cycle_stats["applied"] == 0


def test_nondefender_cycle_fallback_is_once_and_fresh_board_replaces(monkeypatch) -> None:
    """終了盤面がstaleなら整合済みsimulateを一度だけ使い、同じSTABLE中に
    fresh盤面が来た時は元anchorから置換する（fallbackへ加算しない）。"""
    import scripts.visualize_advantage_overlay as vao

    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_live_defender_reeval=True,
        enable_live_defender_strict=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    same = _make_4connect_board()
    same_event = _make_chain_event(2.0, before_board=same, total_score=50)
    tracker.update(
        _make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0,
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 300, state=BoardState.STABLE),
        _make_snapshot(chain_end_triggered_p2=True, chain_total_score_p2=300),
        1.0, t_sec=1.0, b1=None, b2=same,
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 300, state=BoardState.TSUMO_FALL),
        _make_snapshot(), 1.5, t_sec=1.5, b1=None, b2=same,
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(same_event, 300, state=BoardState.CHAIN),
        _make_snapshot(), 2.0, t_sec=2.0, b1=None, b2=same,
        physical_chain_id_p1=11, physical_chain_id_p2=13)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(same_event, 350, state=BoardState.STABLE),
        _make_snapshot(), 3.0, t_sec=3.0, b1=None, b2=same.copy(),
        physical_chain_id_p1=11, physical_chain_id_p2=13)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(same_event, 350, state=BoardState.STABLE),
        _make_snapshot(), 3.6, t_sec=3.6, b1=None, b2=same.copy(),
        physical_chain_id_p1=11, physical_chain_id_p2=13)
    assert len(calls) == 3
    assert tracker.hold_adv == pytest.approx(20.0)
    assert tracker.nondef_cycle_stats["fallback_applied"] == 1

    # 後着fresh盤面はfallback値へ足さず、同じanchor=10から置換する。
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(same_event, 350, state=BoardState.STABLE),
        _make_snapshot(), 3.7, t_sec=3.7, b1=None, b2=Board(),
        physical_chain_id_p1=11, physical_chain_id_p2=13)
    assert len(calls) == 5
    assert tracker.hold_adv == pytest.approx(20.0)
    assert tracker.nondef_cycle_stats["fresh_replaced"] == 1

    # 同じevent idは再開しない。
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(same_event, 350, state=BoardState.TSUMO_FALL),
        _make_snapshot(), 3.8, t_sec=3.8, b1=None, b2=Board(),
        physical_chain_id_p1=11, physical_chain_id_p2=13)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(same_event, 350, state=BoardState.CHAIN),
        _make_snapshot(), 4.0, t_sec=4.0, b1=None, b2=Board(),
        physical_chain_id_p1=11, physical_chain_id_p2=13)

    assert len(calls) == 5
    assert tracker.nondef_cycle_stats["rejected_stale_event"] == 1
    assert tracker.nondef_cycle_stats["applied"] == 1


def test_nondefender_cycle_rejects_simulation_score_mismatch(monkeypatch) -> None:
    """fallback simulateの得点が観測増分と許容差を超える場合は表示を変えない。"""
    import scripts.visualize_advantage_overlay as vao

    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_live_defender_reeval=True,
        enable_live_defender_strict=True)
    ev1 = _make_chain_event(1.0, total_score=500)
    ev2 = _make_chain_event(1.0, total_score=300)
    chain_board = _make_4connect_board()  # simulate得点40
    mismatch_event = _make_chain_event(2.0, before_board=chain_board, total_score=500)
    tracker.update(
        _make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0,
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 300, state=BoardState.STABLE),
        _make_snapshot(chain_end_triggered_p2=True, chain_total_score_p2=300),
        1.0, t_sec=1.0, b1=None, b2=chain_board,
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 300, state=BoardState.TSUMO_FALL),
        _make_snapshot(), 1.5, t_sec=1.5, b1=None, b2=chain_board,
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(mismatch_event, 300, state=BoardState.CHAIN),
        _make_snapshot(), 2.0, t_sec=2.0, b1=None, b2=chain_board,
        physical_chain_id_p1=11, physical_chain_id_p2=13)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(mismatch_event, 500, state=BoardState.STABLE),
        _make_snapshot(), 3.0, t_sec=3.0, b1=None, b2=chain_board,
        physical_chain_id_p1=11, physical_chain_id_p2=13)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(mismatch_event, 500, state=BoardState.STABLE),
        _make_snapshot(), 3.6, t_sec=3.6, b1=None, b2=chain_board,
        physical_chain_id_p1=11, physical_chain_id_p2=13)

    assert len(calls) == 1
    assert tracker.hold_adv == pytest.approx(10.0)
    assert tracker.nondef_cycle_stats["rejected_sim_mismatch"] == 1


def test_nondefender_cycle_rejects_same_physical_chain_after_state_noise(
    monkeypatch,
) -> None:
    """TSUMO/STABLEが途中に揺れても、決着を構成した同一chain_idの再開は
    新しい応手連鎖として補正しない。実動画t=302.833の再現を固定する。"""
    import scripts.visualize_advantage_overlay as vao

    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_live_defender_reeval=True,
        enable_live_defender_strict=True)
    ev1 = _make_chain_event(1.0, total_score=500)
    ev2 = _make_chain_event(1.0, total_score=300)
    tracker.update(
        _make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0,
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 300, state=BoardState.STABLE),
        _make_snapshot(chain_end_triggered_p2=True, chain_total_score_p2=300),
        1.0, t_sec=1.0, b1=None, b2=_make_4connect_board(),
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 300, state=BoardState.TSUMO_FALL),
        _make_snapshot(), 1.5, t_sec=1.5, b1=None, b2=_make_4connect_board(),
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(
            _make_chain_event(2.0, before_board=_make_4connect_board(), total_score=1000),
            300, state=BoardState.CHAIN),
        _make_snapshot(), 2.0, t_sec=2.0, b1=None, b2=_make_4connect_board(),
        physical_chain_id_p1=11, physical_chain_id_p2=12)

    assert len(calls) == 1
    assert tracker.nondef_cycle_stats["started"] == 0
    assert tracker.nondef_cycle_stats["rejected_stale_event"] == 1


def test_nondefender_cycle_rejects_direction_reversal_correction(monkeypatch) -> None:
    """モデル差分だけでholdの符号を反転させる補正は適用せず監査へ残す。"""
    import scripts.visualize_advantage_overlay as vao

    values = iter((10.0, 20.0, -80.0))

    def _stub(model, b1, b2, snap, feature_cols=None, attribution_exclude=()):
        adv = next(values)
        return adv, vao.adv_to_winprob(adv), []

    monkeypatch.setattr(vao, "_score_advantage", _stub)
    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    tracker = vao.ResolvedExchangeTracker(model=object())
    ev1 = _make_chain_event(1.0, total_score=500)
    ev2 = _make_chain_event(1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)

    applied = tracker._apply_nondefender_cycle_after(
        "2P", Board(), 10.0, "test", before=_make_4connect_board())

    assert applied is False
    assert tracker.hold_adv == pytest.approx(10.0)
    assert tracker.nondef_cycle_stats["rejected_direction"] == 1


def test_stable_nondefender_update_freezes_ledger_and_decisive_amplify(
    monkeypatch,
) -> None:
    """非decisive側の盤面更新は、交換結果・会計・incomingを変更せず、応手MCも
    再実行しない。決着時の増幅差分だけを新しい基礎値へ固定加算する。"""
    import scripts.visualize_advantage_overlay as vao

    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    monkeypatch.setattr(vao, "_load_chain_length_conditional_table", lambda: {})
    monkeypatch.setattr(vao, "_counter_defender_adv", lambda *a, **k: 7.0)
    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_decisive_amplify=True,
        enable_live_defender_reeval=True, enable_live_defender_strict=True)
    tracker._counter_tracker.update = lambda *a, **k: (0.0, 0.5, 0.5)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(
        _make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0,
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    frozen_result = tracker._result
    frozen_snap = tracker._resolved_snap
    frozen_incoming = (tracker._incoming_total_p1, tracker._incoming_total_p2)
    assert tracker.hold_adv == pytest.approx(17.0)  # 基礎10 + 増幅7

    def _mc_must_not_run(*args, **kwargs):
        raise AssertionError("非decisive側の更新で応手MCを再実行してはならない")

    tracker._counter_tracker.update = _mc_must_not_run
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 300, state=BoardState.STABLE),
        _make_snapshot(chain_end_triggered_p2=True, chain_total_score_p2=300),
        1.0, t_sec=1.0, b1=None, b2=_board_with_ojama(3),
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 300, state=BoardState.TSUMO_FALL),
        _make_snapshot(), 1.5, t_sec=1.5, b1=None, b2=_board_with_ojama(3),
        physical_chain_id_p1=11, physical_chain_id_p2=12)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(
            _make_chain_event(2.0, before_board=_make_4connect_board(), total_score=50),
            300, state=BoardState.CHAIN),
        _make_snapshot(), 2.0, t_sec=2.0, b1=None, b2=_make_4connect_board(),
        physical_chain_id_p1=11, physical_chain_id_p2=13)
    tracker.update(
        _make_signal_full(ev1, 500, state=BoardState.CHAIN),
        _make_signal_full(ev2, 350, state=BoardState.STABLE),
        _make_snapshot(), 3.0, t_sec=3.0, b1=None, b2=Board(),
        physical_chain_id_p1=11, physical_chain_id_p2=13)

    assert len(calls) == 3
    assert tracker.hold_adv == pytest.approx(27.0)  # 元17 + paired-delta(30-20)
    assert tracker._result is frozen_result
    assert tracker._resolved_snap is frozen_snap
    assert (tracker._incoming_total_p1, tracker._incoming_total_p2) == frozen_incoming


def test_live_defender_strict_still_reevaluates_when_defender_state_tsumo_fall(
    monkeypatch,
) -> None:
    """defender_side が TSUMO_FALL (ツモ設置中、指摘13が意図した「受け側は
    連鎖中も置き続ける」正当な自由行動そのもの) の間は busy 扱いにせず
    strict=True でも再評価される (TSUMO_FALL/OJAMA_FALL を busy 集合に
    含めない設計判断の直接確認)。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_live_defender_reeval=True, enable_live_defender_strict=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert len(calls) == 1
    live_board = _board_with_ojama(9)
    tracker.update(
        _make_signal_full(None, 500, state=BoardState.TSUMO_FALL),
        _make_signal_full(ev2, 300, state=BoardState.CHAIN),
        _make_snapshot(), 1.0, t_sec=1.0, b1=live_board, b2=None)
    assert len(calls) == 2  # TSUMO_FALL は busy 扱いしないため再評価される


def test_live_defender_strict_backward_compat_when_signal_lacks_state_attribute(
    monkeypatch,
) -> None:
    """[後方互換] `.state` 属性を持たない軽量な信号オブジェクト (既存テスト群が
    使う `_make_signal`) を渡しても、`update()` 側の `getattr(..., None)`
    フォールバックにより例外を出さず「busy でない」扱いになる (defender_state
    が None のため `_LIVE_DEFENDER_BUSY_STATES` に含まれない=素通り)。既存の
    `_make_signal` ベースのテスト群 (strict=False) を壊さないための保証。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    result_stub = _stub_exchange_result_distinct_boards(dropped_to_p1=30)
    monkeypatch.setattr(vao, "resolve_mutual_exchange", lambda *a, **k: result_stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_live_defender_reeval=True, enable_live_defender_strict=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=500)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 500), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert len(calls) == 1
    live_board = _board_with_ojama(9)
    # _make_signal (state属性なし) をそのまま使う。例外を出さず再評価される。
    tracker.update(_make_signal(None, 500), _make_signal(ev2, 300), _make_snapshot(), 1.0,
                   t_sec=1.0, b1=live_board, b2=None)
    assert len(calls) == 2


def test_live_defender_strict_flag_default_is_false() -> None:
    """コンストラクタ既定値の後方互換確認 (省略時 False)。"""
    import scripts.visualize_advantage_overlay as vao
    tracker = vao.ResolvedExchangeTracker(model=object())
    assert tracker._enable_live_defender_strict is False


# ============================
# 指摘14 案2: enable_resolved_kill_override (既定OFF)
# ============================
# 背景: kill_override はライブ per-frame 経路にのみ配線されており、決着
# ホールド中 (hold_adv/hold_p1 がそのまま disp_adv/disp_p1 に代入される経路)
# には未配線だった。pending/room 比が致死水準 (実測 589/50≈11.8 ≫
# KILL_RATIO_FULL=1.5) でも安全弁が発火しない事故の直接対処。


def test_hold_after_kill_override_noop_when_not_lethal() -> None:
    """飛来量が小さい (KILL_MIN_PENDING 未満) 間は hold_adv/hold_p1 を変えない
    (kill_override 本体の既存ノーオップ条件をそのまま踏襲、新規判定を足さない)。"""
    import scripts.visualize_advantage_overlay as vao
    tracker = vao.ResolvedExchangeTracker(model=object())
    tracker.hold_adv, tracker.hold_p1 = 20.0, 0.6
    tracker._incoming_total_p1, tracker._incoming_total_p2 = 10.0, 0.0
    b1, b2 = _board_with_ojama(0), _board_with_ojama(0)
    adv, p1 = tracker.hold_after_kill_override(b1, b2)
    assert adv == pytest.approx(20.0)
    assert p1 == pytest.approx(0.6)


def test_hold_after_kill_override_overrides_lethal_hold_toward_survivor(monkeypatch) -> None:
    """[指摘14 案2本体] pending/room 比が致死水準 (実測ケースを模した
    589 pending / room≈50) では hold_adv/hold_p1 が生存側 (1P) へ完全に
    上書きされる (kill_override(g=1) と同値になることを直接確認)。"""
    import scripts.visualize_advantage_overlay as vao
    tracker = vao.ResolvedExchangeTracker(model=object())
    # 誤爆時に表示されていた値を模す (2P の実際の生存率18.9%相当、1P視点+62)。
    tracker.hold_adv, tracker.hold_p1 = 62.2, 0.811
    tracker._incoming_total_p1, tracker._incoming_total_p2 = 0.0, 589.0
    b1 = _board_with_ojama(0)
    b2 = _board_with_ojama(22)  # room2 = 72-22 = 50 (実測に近似)
    adv, p1 = tracker.hold_after_kill_override(b1, b2)
    expected_adv = vao.kill_override(
        tracker.hold_adv, tracker._incoming_total_p1, tracker._incoming_total_p2,
        vao.board_room(b1), vao.board_room(b2))
    assert adv == pytest.approx(expected_adv)
    assert adv == pytest.approx(100.0)  # 致死度差が KILL_RATIO_FULL 以上 → 完全上書き
    assert p1 == pytest.approx(vao.adv_to_winprob(100.0))


def test_hold_after_kill_override_reuses_existing_room_and_pending_no_new_heuristic() -> None:
    """室/pending 比の材料は既存の観測量のみ再利用する (新しいヒューリスティクス
    を増やさない): pending=self._incoming_total_p1/p2 (指摘11の着弾完了判定と
    同一値)、room=モジュール既存の board_room(b1)/board_room(b2)。"""
    import scripts.visualize_advantage_overlay as vao
    tracker = vao.ResolvedExchangeTracker(model=object())
    tracker.hold_adv, tracker.hold_p1 = 0.0, 0.5
    tracker._incoming_total_p1, tracker._incoming_total_p2 = 50.0, 60.0
    b1, b2 = _board_with_ojama(3), _board_with_ojama(7)
    adv, p1 = tracker.hold_after_kill_override(b1, b2)
    expected = vao.kill_override(
        0.0, 50.0, 60.0, vao.board_room(b1), vao.board_room(b2))
    assert adv == pytest.approx(expected)
    if adv == 0.0:
        assert p1 == pytest.approx(0.5)
    else:
        assert p1 == pytest.approx(vao.adv_to_winprob(adv))


def test_generate_source_gates_hold_kill_override_calls_by_flag() -> None:
    """静的回帰テスト: generate() ソース中の hold_after_kill_override 呼び出し
    2箇所 (resolved_active時/just_deactivated時) が両方とも
    `enable_resolved_kill_override` の if ブロック配下にあることを固定する
    (既定 OFF 時に絶対に呼ばれないことの構造的保証)。"""
    import inspect
    import scripts.visualize_advantage_overlay as vao

    src = inspect.getsource(vao.generate)
    code_only = src.replace(vao.generate.__doc__ or "", "")  # docstring内の言及を除外
    call_pattern = "resolved_tracker.hold_after_kill_override("
    assert code_only.count(call_pattern) == 2
    # 各出現の直前行が enable_resolved_kill_override の if であることを確認。
    lines = code_only.splitlines()
    hit_lines = [i for i, line in enumerate(lines) if call_pattern in line]
    assert len(hit_lines) == 2
    for idx in hit_lines:
        preceding = "\n".join(lines[max(0, idx - 2):idx])
        assert "if enable_resolved_kill_override" in preceding


def test_generate_signature_new_flags_default_false() -> None:
    """generate() の新規フラグ2つは既定 False (backwards compat、既存呼出元
    はキーワード省略可)。"""
    import inspect
    import scripts.visualize_advantage_overlay as vao

    sig = inspect.signature(vao.generate)
    assert sig.parameters["enable_resolved_live_defender_strict"].default is False
    assert sig.parameters["enable_resolved_kill_override"].default is False
    assert sig.parameters["enable_resolved_kill_override_counter_aware"].default is False


# ============================
# 指摘19: enable_kill_override_counter_aware (既定OFF、状態ゲート方式)
# ============================
# 背景 (実測、logs/_diag_issue19_2026-08-15.log): kill_override は pending/room
# 比のみで致死を断定し、受け側が実際に応手可能かを一切見ていない。ドメイン
# ルール「おじゃまは連鎖完了+受け側のツモ着地時に降る」により受け側には撃ち
# 返す時間があるため、受け側がSTABLEで応手可能な局面 (t=201.4-203.4、実際に
# 撃ち返し score 42065 vs 19729 で勝利) まで 1P 0.7% と致死断定していた。
#
# [設計変更の経緯、coordinator判断 2026-08-16]
# 当初は既存 CounterReachTracker/mc_counter_estimator の応手確率で連続
# ブレンドする案を実装したが、実測 (logs/_diag_issue19_dampen_trace_
# 2026-08-15.log) で応手確率が25-40%止まり (mc_counter_estimator の既知の
# 推定精度限界、docs/KNOWN_WEAKNESSES.md W15) と判明し、target=±100 固定の
# 線形ブレンドでは合格水準(56%前後)に届かなかった。kill_override は
# 「モデルが致死量を見落とす」ことへの安全弁であり勝率の微調整装置では
# ないため、確率推定の精度に依存しない**状態ゲート (二値)** 方式へ切替。
# 指摘14案1が chain_event のパルス方式依存(誤爆)から状態機械ベースへ切替
# えて解決した前例と同じパターン (`_LIVE_DEFENDER_BUSY_STATES` 再利用)。


def _tracker_with_lethal_setup(model=None) -> "vao.ResolvedExchangeTracker":
    """589 pending / room≈50 (実測ケース、完全上書き g=1) を模したトラッカーを返す。

    victim_side は "2P" (589 pending を受ける側) になる。"""
    import scripts.visualize_advantage_overlay as vao

    tracker = vao.ResolvedExchangeTracker(model=model or object())
    tracker.hold_adv, tracker.hold_p1 = 62.2, 0.811
    tracker._incoming_total_p1, tracker._incoming_total_p2 = 0.0, 589.0
    return tracker


def test_kill_override_counter_aware_off_is_bit_identical_to_baseline() -> None:
    """[既定OFF] enable_kill_override_counter_aware=False の間は state1/state2
    (victim="2P" が STABLE=自由行動中) を渡していても一切参照せず、従来
    (#14案2) と bit-identical な結果を返す (backwards compat)。"""
    import scripts.visualize_advantage_overlay as vao

    tracker = _tracker_with_lethal_setup()
    b1, b2 = _board_with_ojama(0), _board_with_ojama(22)
    adv, p1 = tracker.hold_after_kill_override(
        b1, b2, state1=BoardState.STABLE, state2=BoardState.STABLE)
    expected_adv = vao.kill_override(
        tracker.hold_adv, tracker._incoming_total_p1, tracker._incoming_total_p2,
        vao.board_room(b1), vao.board_room(b2))
    assert adv == pytest.approx(expected_adv)
    assert adv == pytest.approx(100.0)  # 従来通り完全上書き(フラグ未参照の証拠)
    assert p1 == pytest.approx(vao.adv_to_winprob(100.0))


def test_kill_override_counter_aware_suppresses_when_victim_is_free() -> None:
    """[本体] victim_side ("2P") が STABLE (自由行動中=busyでない) なら
    致死断定を完全に取り消し hold_adv/hold_p1 のまま返す (指摘19が解消
    したい退行の直接対処。実動画では STABLE で応手可能だった)。"""
    tracker = _tracker_with_lethal_setup()
    tracker._enable_kill_override_counter_aware = True
    b1, b2 = _board_with_ojama(0), _board_with_ojama(22)
    adv, p1 = tracker.hold_after_kill_override(
        b1, b2, state1=BoardState.STABLE, state2=BoardState.STABLE)
    assert adv == pytest.approx(tracker.hold_adv)
    assert p1 == pytest.approx(tracker.hold_p1)


@pytest.mark.parametrize("busy_state", [BoardState.CHAIN, BoardState.GRAVITY_SETTLE])
def test_kill_override_counter_aware_still_fires_when_victim_is_busy(busy_state) -> None:
    """victim_side ("2P") が `_LIVE_DEFENDER_BUSY_STATES` (CHAIN/
    GRAVITY_SETTLE、物理的に動けない) なら従来通り完全発火する
    (counter_aware=True でも安全弁の効き目を弱めない、指摘14窓の
    99.3%維持に対応)。"""
    import scripts.visualize_advantage_overlay as vao

    tracker = _tracker_with_lethal_setup()
    tracker._enable_kill_override_counter_aware = True
    b1, b2 = _board_with_ojama(0), _board_with_ojama(22)
    adv, p1 = tracker.hold_after_kill_override(
        b1, b2, state1=BoardState.STABLE, state2=busy_state)
    expected_adv = vao.kill_override(
        tracker.hold_adv, tracker._incoming_total_p1, tracker._incoming_total_p2,
        vao.board_room(b1), vao.board_room(b2))
    assert adv == pytest.approx(expected_adv)
    assert adv == pytest.approx(100.0)


@pytest.mark.parametrize("free_state", [BoardState.TSUMO_FALL, BoardState.OJAMA_FALL])
def test_kill_override_counter_aware_suppresses_for_non_busy_states(free_state) -> None:
    """TSUMO_FALL/OJAMA_FALL は `_LIVE_DEFENDER_BUSY_STATES` に含まれない
    (= busy でない) ため、STABLE と同様に致死断定を取り消す (指摘13が意図
    した「受け側は連鎖中も置き続ける」正当な自由行動を塞がない設計を継承、
    `_LIVE_DEFENDER_BUSY_STATES` docstring 参照)。"""
    tracker = _tracker_with_lethal_setup()
    tracker._enable_kill_override_counter_aware = True
    b1, b2 = _board_with_ojama(0), _board_with_ojama(22)
    adv, p1 = tracker.hold_after_kill_override(
        b1, b2, state1=BoardState.STABLE, state2=free_state)
    assert adv == pytest.approx(tracker.hold_adv)
    assert p1 == pytest.approx(tracker.hold_p1)


def test_kill_override_counter_aware_falls_back_when_state_unknown() -> None:
    """victim_side の state が None (呼出元が未対応/省略、backwards compat)
    の場合は busy かどうか判定不能として従来通り発火する (fail-silent を
    避け警戒を緩めない)。"""
    import scripts.visualize_advantage_overlay as vao

    tracker = _tracker_with_lethal_setup()
    tracker._enable_kill_override_counter_aware = True
    b1, b2 = _board_with_ojama(0), _board_with_ojama(22)
    adv, p1 = tracker.hold_after_kill_override(b1, b2)  # state1/state2 省略
    expected_adv = vao.kill_override(
        tracker.hold_adv, tracker._incoming_total_p1, tracker._incoming_total_p2,
        vao.board_room(b1), vao.board_room(b2))
    assert adv == pytest.approx(expected_adv)
    assert adv == pytest.approx(100.0)


def test_kill_override_counter_aware_noop_when_not_lethal() -> None:
    """致死断定自体が発火しない (KILL_MIN_PENDING未満) 場合は state ゲートに
    すら到達しない (既存の早期return分岐をそのまま通る、victim側が
    STABLE=本来なら抑制対象の state でも無関係)。"""
    import scripts.visualize_advantage_overlay as vao

    tracker = vao.ResolvedExchangeTracker(model=object())
    tracker._enable_kill_override_counter_aware = True
    tracker.hold_adv, tracker.hold_p1 = 20.0, 0.6
    tracker._incoming_total_p1, tracker._incoming_total_p2 = 10.0, 0.0
    b1, b2 = _board_with_ojama(0), _board_with_ojama(0)
    adv, p1 = tracker.hold_after_kill_override(
        b1, b2, state1=BoardState.STABLE, state2=BoardState.STABLE)
    assert adv == pytest.approx(20.0)
    assert p1 == pytest.approx(0.6)


def test_resolved_exchange_tracker_constructor_new_flag_default_false() -> None:
    """ResolvedExchangeTracker.__init__ の新規フラグは既定 False
    (backwards compat、既存呼出元はキーワード省略可)。"""
    import inspect
    import scripts.visualize_advantage_overlay as vao

    sig = inspect.signature(vao.ResolvedExchangeTracker.__init__)
    assert sig.parameters["enable_kill_override_counter_aware"].default is False
    tracker = vao.ResolvedExchangeTracker(model=object())
    assert tracker._enable_kill_override_counter_aware is False


def test_hold_after_kill_override_signature_new_state_args_default_none() -> None:
    """hold_after_kill_override の新規引数 state1/state2 は既定 None
    (backwards compat、既存呼出元はキーワード省略可)。"""
    import inspect
    import scripts.visualize_advantage_overlay as vao

    sig = inspect.signature(vao.ResolvedExchangeTracker.hold_after_kill_override)
    assert sig.parameters["state1"].default is None
    assert sig.parameters["state2"].default is None


def test_generate_source_wires_kill_override_counter_aware_to_both_constructions() -> None:
    """静的回帰テスト: generate() ソース中の ResolvedExchangeTracker 構築
    (通常時/試合境界リセット時の2箇所) が両方とも
    enable_kill_override_counter_aware=enable_resolved_kill_override_counter_aware
    を渡していることを固定する (新フラグの配線漏れ防止)。"""
    import inspect
    import scripts.visualize_advantage_overlay as vao

    src = inspect.getsource(vao.generate)
    code_only = src.replace(vao.generate.__doc__ or "", "")
    pattern = (
        "enable_kill_override_counter_aware=enable_resolved_kill_override_counter_aware")
    assert code_only.count(pattern) == 2


def test_generate_source_passes_state_to_hold_kill_override_calls() -> None:
    """静的回帰テスト: generate() ソース中の hold_after_kill_override 呼び
    出し2箇所が両方とも state1=r.p1.state, state2=r.p2.state を渡している
    ことを固定する (state 未配線=常にNone=常に安全側発火、に戻る退行を防ぐ)。"""
    import inspect
    import scripts.visualize_advantage_overlay as vao

    src = inspect.getsource(vao.generate)
    code_only = src.replace(vao.generate.__doc__ or "", "")
    assert code_only.count("resolved_tracker.hold_after_kill_override(") == 2
    assert code_only.count("state1=r.p1.state, state2=r.p2.state") == 2


# ============================
# 指摘19 根治: enable_resolved_victim_gen_live (既定OFF、2026-08-16)
# ============================
# 背景 (実測、logs/_diag_issue19_root_cause_trace_2026-08-16.log): 従来の
# `_maybe_redecide` は `chain_end_triggered_pX` が True の**最初の1フレーム**
# だけ `chain_total_score_pX` を latch (`_redecided1/2`) し、以後は無視する。
# しかし OjamaAccountingTracker (src/ojama_accounting.py) の実装では
# `chain_end_triggered_pX` は settle 開始の瞬間に True になり、同一の連鎖が
# coalesce window 内で複数回に分けて finalize されるたびに
# `chain_total_score_pX` を段階的に上書きしながら True であり続ける
# (実測: 0→1260→4020 と3段階で確定)。「1回きり」latch は運悪く未確定
# (0や小さい途中値) の瞬間に固定してしまい、真の確定値を二度と拾わなかった。
# 本節は `enable_resolved_victim_gen_live=True` でこの latch を
# 「chain_end_triggered_pX が True の間 COUNTER_RECOMPUTE_INTERVAL_SEC ごとに
# 追従する」方式へ緩和したことの回帰テスト。


def test_resolved_exchange_tracker_victim_gen_live_default_false() -> None:
    """ResolvedExchangeTracker.__init__ の新規フラグは既定 False
    (backwards compat、既存呼出元はキーワード省略可)。"""
    import inspect
    import scripts.visualize_advantage_overlay as vao

    sig = inspect.signature(vao.ResolvedExchangeTracker.__init__)
    assert sig.parameters["enable_resolved_victim_gen_live"].default is False
    tracker = vao.ResolvedExchangeTracker(model=object())
    assert tracker._enable_resolved_victim_gen_live is False


def test_generate_signature_victim_gen_live_default_false() -> None:
    """generate() の新規フラグも既定 False。"""
    import inspect
    import scripts.visualize_advantage_overlay as vao

    sig = inspect.signature(vao.generate)
    assert sig.parameters["enable_resolved_victim_gen_live"].default is False


def test_victim_gen_live_off_is_bit_identical_to_at_most_once(monkeypatch) -> None:
    """[既定OFF・指摘19根治バグの直接再現] `enable_resolved_victim_gen_live
    =False` の間は、同一 chain_end_triggered_p1 継続中の最初の観測 (実測
    パターン通り未確定=0) で `_redecided1` が永久に latch されるため、
    以後 total が真に育っても (1260→4020) 二度と拾わない (=victim の gen が
    過小評価されたまま固定される、指摘19 の根本原因そのもの)。
    `test_resolved_redecide_at_most_once_per_side` と同じ「1回きり」結論の
    直接確認 (bit-identical)。ON 版 (`test_victim_gen_live_on_follows_
    growing_confirmed_total`) が同じ入力列で追従することとの対比。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(model=object())  # 既定 False
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=100)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 100), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert len(calls) == 1
    for step_sec, total in ((0.6, 0), (1.2, 1260), (1.8, 4020)):
        snap = _make_snapshot(chain_end_triggered_p1=True, chain_total_score_p1=total)
        tracker.update(_make_signal(ev1, 100), _make_signal(ev2, 300), snap, step_sec)
    assert len(calls) == 1  # 最初の (未確定=0の) 観測で latch、真の4020も永久に無視
    assert tracker._pred_score1 == pytest.approx(100.0)  # 過小評価のまま固定


def test_victim_gen_live_on_follows_growing_confirmed_total(monkeypatch) -> None:
    """[本体] True の場合、同一 chain_end_triggered_p1 継続中に
    chain_total_score_p1 が段階的に育つたび (0.5秒以上間隔をあけて) 追従し、
    予測を都度更新する (実測パターン 0→1260→4020 の再現)。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_resolved_victim_gen_live=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=100)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 100), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert len(calls) == 1
    # t=0.6: 最初の settle 観測 (総額まだ0、latchせず「未確定」として素通り)。
    snap0 = _make_snapshot(chain_end_triggered_p1=True, chain_total_score_p1=0)
    tracker.update(_make_signal(ev1, 100), _make_signal(ev2, 300), snap0, 0.6)
    assert len(calls) == 1  # 0は100を超えないため再決着なし
    # t=1.2 (前回確定観測から0.6秒後): 1260に成長、100を超えるため再決着。
    snap1 = _make_snapshot(chain_end_triggered_p1=True, chain_total_score_p1=1260)
    tracker.update(_make_signal(ev1, 100), _make_signal(ev2, 300), snap1, 1.2)
    assert len(calls) == 2
    assert tracker._pred_score1 == pytest.approx(1260.0)
    # t=1.8 (前回確定から0.6秒後): 4020にさらに成長、追従する。
    snap2 = _make_snapshot(chain_end_triggered_p1=True, chain_total_score_p1=4020)
    tracker.update(_make_signal(ev1, 100), _make_signal(ev2, 300), snap2, 1.8)
    assert len(calls) == 3
    assert tracker._pred_score1 == pytest.approx(4020.0)


def test_victim_gen_live_throttles_within_half_second(monkeypatch) -> None:
    """0.5秒 (COUNTER_RECOMPUTE_INTERVAL_SEC) 未満の連続呼び出しは、
    総額が育っていても追従をスキップする (間引き、`_reevaluate_live_defender`
    と同じ周期を再利用)。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_resolved_victim_gen_live=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=100)
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 100), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    snap1 = _make_snapshot(chain_end_triggered_p1=True, chain_total_score_p1=1260)
    tracker.update(_make_signal(ev1, 100), _make_signal(ev2, 300), snap1, 0.3)
    assert len(calls) == 2  # 最初の確定観測は latch 済みでなくても即座に通る
    # 0.5秒未満 (0.3→0.6=0.3秒後) の再成長はスキップされる。
    snap2 = _make_snapshot(chain_end_triggered_p1=True, chain_total_score_p1=4020)
    tracker.update(_make_signal(ev1, 100), _make_signal(ev2, 300), snap2, 0.6)
    assert len(calls) == 2  # 間引きにより追従しない
    assert tracker._pred_score1 == pytest.approx(1260.0)
    # 0.5秒以上経過 (0.3→0.9=0.6秒後) すれば追従する。
    tracker.update(_make_signal(ev1, 100), _make_signal(ev2, 300), snap2, 0.9)
    assert len(calls) == 3
    assert tracker._pred_score1 == pytest.approx(4020.0)


def test_victim_gen_live_attacker_side_unaffected_when_already_settled(monkeypatch) -> None:
    """攻撃側 (chain_total_score_pX が最初から確定値で以後変化しない) は、
    フラグ ON でも追加の再決着を起こさない (「攻撃側は従来通り即時確定値の
    まま」がクラス docstring の主張通りノーオペで成り立つことの確認)。"""
    import scripts.visualize_advantage_overlay as vao
    stub, calls = _stub_score_advantage_factory()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tracker = vao.ResolvedExchangeTracker(
        model=object(), enable_resolved_victim_gen_live=True)
    ev1 = _make_chain_event(trigger_sec=1.0, total_score=5000)  # 攻撃側、既に大きい
    ev2 = _make_chain_event(trigger_sec=1.0, total_score=300)
    tracker.update(_make_signal(ev1, 5000), _make_signal(ev2, 300), _make_snapshot(), 0.0)
    assert len(calls) == 1
    # 攻撃側 (1P) は settle 完了済み総額が最初の予測と同じ (追加成長なし)。
    for step_sec in (0.6, 1.2, 1.8):
        snap = _make_snapshot(chain_end_triggered_p1=True, chain_total_score_p1=5000)
        tracker.update(_make_signal(ev1, 5000), _make_signal(ev2, 300), snap, step_sec)
    assert len(calls) == 1  # 5000は5000を超えないため再決着は一度も起きない


def test_generate_source_wires_victim_gen_live_to_both_constructions() -> None:
    """静的回帰テスト: generate() ソース中の ResolvedExchangeTracker 構築
    (通常時/試合境界リセット時の2箇所) が両方とも
    enable_resolved_victim_gen_live=enable_resolved_victim_gen_live
    を渡していることを固定する (新フラグの配線漏れ防止)。"""
    import inspect
    import scripts.visualize_advantage_overlay as vao

    src = inspect.getsource(vao.generate)
    code_only = src.replace(vao.generate.__doc__ or "", "")
    pattern = "enable_resolved_victim_gen_live=enable_resolved_victim_gen_live"
    assert code_only.count(pattern) == 2


def test_main_source_wires_victim_gen_live_cli_flag() -> None:
    """静的回帰テスト: main() ソースが --resolved-victim-gen-live の argparse
    定義と generate() 呼び出しへの受け渡しの両方を含むことを固定する。"""
    import inspect
    import scripts.visualize_advantage_overlay as vao

    src = inspect.getsource(vao.main)
    assert '"--resolved-victim-gen-live"' in src
    assert "dest=\"enable_resolved_victim_gen_live\"" in src
    assert "enable_resolved_victim_gen_live=a.enable_resolved_victim_gen_live" in src


def test_episode_physical_redecide_defaults_and_wiring() -> None:
    """新機能は既定OFFで、CLI→generate→trackerの2生成箇所まで配線される。"""
    import inspect
    import scripts.visualize_advantage_overlay as vao

    tracker_sig = inspect.signature(vao.ResolvedExchangeTracker.__init__)
    generate_sig = inspect.signature(vao.generate)
    assert tracker_sig.parameters["enable_episode_physical_redecide"].default is False
    assert tracker_sig.parameters[
        "enable_episode_physical_consistency_guard"].default is False
    assert generate_sig.parameters[
        "enable_resolved_episode_physical_redecide"].default is False
    assert generate_sig.parameters[
        "enable_resolved_episode_physical_consistency_guard"].default is False
    source = inspect.getsource(vao.generate)
    assert source.count("enable_episode_physical_redecide=") == 2
    assert source.count("enable_episode_physical_consistency_guard=") == 2
    main_source = inspect.getsource(vao.main)
    assert '"--resolved-episode-physical-redecide"' in main_source
    assert '"--resolved-episode-physical-consistency-guard"' in main_source
    assert "a.enable_resolved_episode_physical_redecide" in main_source
    assert "a.enable_resolved_episode_physical_consistency_guard" in main_source


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


# ============================
# --model-dir オプション (2026-08-18 追加)
# ============================
# user要望「新しく収集したデータで学習したモデルを使って解析動画を作りたい」
# に対応するため、モデル成果物ディレクトリを CLI から差し替え可能にした。
# 既定 (未指定) は従来通り MODEL_ARTIFACT_DIR を使う (後方互換)。
# 明示指定時はファイル欠如・特徴量列不一致を fail-silent フォールバックせず
# 即座に例外化する (無言で旧モデルにフォールバックする事故を防ぐ)。


def test_load_artifact_model_default_dir_unchanged(monkeypatch, tmp_path) -> None:
    """model_dir 省略時は従来通り MODEL_ARTIFACT_PATH/_FEATURE_COLS_PATH を
    そのまま使う (後方互換、既存呼出元・既存テストの monkeypatch と完全互換)。"""
    import scripts.visualize_advantage_overlay as vao
    monkeypatch.setattr(vao, "MODEL_ARTIFACT_PATH", tmp_path / "no_such_model.joblib")
    monkeypatch.setattr(
        vao, "MODEL_ARTIFACT_FEATURE_COLS_PATH", tmp_path / "no_such_cols.json")
    assert vao._load_artifact_model() is None  # 引数無しでも従来通り fail-safe


def test_load_artifact_model_explicit_dir_missing_is_fail_safe_by_default(
    tmp_path,
) -> None:
    """model_dir 指定時でも strict=False (既定) なら従来通り None を返す
    (`_load_artifact_model` 単体は fail-safe のまま、strict 化は呼出元
    `_acquire_model` の責務)。"""
    import scripts.visualize_advantage_overlay as vao
    assert vao._load_artifact_model(tmp_path / "no_such_dir") is None


def test_load_artifact_model_explicit_dir_missing_raises_when_strict(tmp_path) -> None:
    """model_dir 指定 + strict=True で成果物が無い場合は
    ModelArtifactMissingError を送出する (フォールバックしない、fail-silent禁止)。"""
    import scripts.visualize_advantage_overlay as vao
    with pytest.raises(vao.ModelArtifactMissingError):
        vao._load_artifact_model(tmp_path / "no_such_dir", strict=True)


def test_load_artifact_model_explicit_dir_partial_files_raises_when_strict(
    tmp_path,
) -> None:
    """model_dir 配下に joblib はあるが feature_cols_full.json が無い場合も
    strict=True では例外 (片方欠損も見逃さない)。"""
    import scripts.visualize_advantage_overlay as vao
    model_dir = tmp_path / "custom_model"
    model_dir.mkdir()
    (model_dir / "model_full148_full_features.joblib").write_bytes(b"dummy")
    with pytest.raises(vao.ModelArtifactMissingError):
        vao._load_artifact_model(model_dir, strict=True)


def test_load_artifact_model_feature_count_mismatch_raises(tmp_path) -> None:
    """モデルが期待する特徴量数 (n_features_in_) と feature_cols_full.json の
    列数が食い違う場合、strict の値に関わらず ModelArtifactFeatureMismatchError
    を送出する (成果物ペアの整合性は常に検証する)。"""
    import json as _json

    import joblib

    import scripts.visualize_advantage_overlay as vao
    model_dir = tmp_path / "mismatched_model"
    model_dir.mkdir()
    fake_model = types.SimpleNamespace(n_features_in_=47)
    joblib.dump(fake_model, model_dir / "model_full148_full_features.joblib")
    (model_dir / "feature_cols_full.json").write_text(
        _json.dumps(["col_a", "col_b"]), encoding="utf-8")  # 2列 != 47
    with pytest.raises(vao.ModelArtifactFeatureMismatchError):
        vao._load_artifact_model(model_dir)  # strict=False でも不一致は必ずエラー


def test_acquire_model_explicit_model_dir_missing_raises_without_fallback(
    monkeypatch, tmp_path,
) -> None:
    """`--model-dir` 明示指定先に成果物が無い場合、_acquire_model は
    _train_model へフォールバックせず即座に例外を送出する (fail-silent禁止、
    「無言で古いモデルを使う」事故の再発防止)。"""
    import scripts.visualize_advantage_overlay as vao

    def _fail_if_called(exclude_video=None):
        raise AssertionError("_train_model が呼ばれた = fail-silent フォールバックが復活している")

    monkeypatch.setattr(vao, "_train_model", _fail_if_called)
    with pytest.raises(vao.ModelArtifactMissingError):
        vao._acquire_model(None, tmp_path / "no_such_dir")


def test_acquire_model_exclude_video_overrides_model_dir(monkeypatch, tmp_path) -> None:
    """exclude_video と model_dir が同時指定された場合、リーク防止を優先し
    model_dir を無視して CSV 起動時学習にフォールバックする (model_dir が
    存在しないディレクトリでも例外にならない = 無視されている証拠)。"""
    import scripts.visualize_advantage_overlay as vao
    sentinel = object()
    calls: list[object] = []
    monkeypatch.setattr(
        vao, "_train_model", lambda exclude_video=None: (calls.append(exclude_video), sentinel)[1])
    result = vao._acquire_model("video_29", tmp_path / "no_such_dir")
    assert result is sentinel
    assert calls == ["video_29"]


def test_generate_model_dir_default_is_none() -> None:
    """generate() の model_dir 既定が None であること
    (後方互換、未指定時は MODEL_ARTIFACT_DIR を使う従来挙動)。"""
    import inspect

    import scripts.visualize_advantage_overlay as vao
    params = inspect.signature(vao.generate).parameters
    assert "model_dir" in params
    assert params["model_dir"].default is None


def test_main_wires_model_dir_cli_option_through_to_generate() -> None:
    """main() の argparse に --model-dir が配線され、generate() まで
    届いていること (2026-08-18 追加のCLIオプション配線の回帰テスト)。"""
    import inspect

    import scripts.visualize_advantage_overlay as vao
    src = inspect.getsource(vao.main)
    assert "--model-dir" in src
    assert 'dest="model_dir"' in src
    assert "model_dir=a.model_dir" in src


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


# ============================
# 2026-08-22 修正: kill_override ライブ経路の入力差し替え + elapsed_sec の
# game_start_sec 対応化 (配線事故、致死の安全弁に誤値が渡っていた)
# ============================
def test_generate_source_wires_kill_override_to_confirmed_accounting() -> None:
    """静的回帰テスト: generate() のライブ per-frame 経路 (通常4成分ブレンド)
    にある kill_override 呼び出しが、確定会計 (OjamaAccountingTracker.snap.
    pending_p1/p2) を使っており、旧世代の粗い推定 (RealtimeForecastTracker.
    fctracker.inc1/inc2) を使っていないことを固定する。

    背景 (2026-08-22 実測、scripts/_diag_kill_override_wiring_2026-08-22.py):
    fctracker.inc1/inc2 は得点差÷70ヒューリスティック+ツモ毎30個減衰の粗い
    推定で、t=886.5s において真値は2Pに216個 pending なのに inc1=616.73/
    inc2=0.00 と1P側へ**逆方向**に出る事故を起こしていた。安全弁自体
    (kill_override 関数) は正しく動作しており、入力が誤っていたことが原因。
    fc (4成分ブレンドの予告項、fctracker.update() 呼び出し自体) は本修正の
    対象外 (据え置き)。
    """
    import inspect
    import scripts.visualize_advantage_overlay as vao

    src = inspect.getsource(vao.generate)
    code_only = src.replace(vao.generate.__doc__ or "", "")  # docstring内の言及を除外
    # (2026-08-22 修正① 追加) 既定 (enable_kill_override_chain_completion=False)
    # では kpending1/kpending2 は snap.pending_p1/p2 の単純代入であることを固定
    # (bit-identical fallback)。
    assert "kpending1, kpending2 = float(snap.pending_p1), float(snap.pending_p2)" in code_only
    assert "adv = kill_override(adv, kpending1, kpending2," in code_only
    # 旧・誤配線 (逆方向発火/見落としの原因) が復活していないことを固定する。
    assert "kill_override(adv, fctracker.inc1, fctracker.inc2," not in code_only
    assert "kill_override(adv, snap.pending_p1, snap.pending_p2," not in code_only
    # fc成分自体 (fctracker.update呼び出し) は本修正の対象外、残っていること。
    assert "fctracker.update(" in code_only


def test_generate_source_uses_relative_elapsed_sec_for_panel_display() -> None:
    """静的回帰テスト: panel レイアウトの「経過X秒」表示 (_draw_panel_layout
    の elapsed_sec) が、グラフ横軸と同じ試合相対時間 t_rel を使っており、
    試合境界を反映しない動画全体の絶対経過秒 (t - start_sec) に戻っていない
    ことを固定する。

    背景 (2026-08-22): 区間分割の継ぎ目 (t=893.7s 等) で「経過893秒」の直後に
    「経過6秒」へ逆行して見えるバグだった。境界が一度も起きない動画では
    game_start_sec=0.0 のままなので t_rel は従来の t - start_sec と完全一致
    し (backwards compat)、この修正は表示バグの解消のみで挙動を変えない。
    """
    import inspect
    import scripts.visualize_advantage_overlay as vao

    src = inspect.getsource(vao.generate)
    code_only = src.replace(vao.generate.__doc__ or "", "")
    assert "elapsed_sec=t_rel," in code_only
    assert "elapsed_sec=t - start_sec," not in code_only


# ============================
# 2026-08-22 修正①②④: kill_override 連鎖完走後是正 / EarlyFireTracker
# finalize連動クリア / 主因表示への安全弁理由明示 (静的回帰テスト)
# ============================

class TestKillOverrideChainCompletionWiring:
    """修正①: 新フラグの既定値・配線・bit-identical フォールバックを固定する。"""

    def test_generate_has_new_flags_default_false(self) -> None:
        import inspect
        import scripts.visualize_advantage_overlay as vao
        sig = inspect.signature(vao.generate)
        for name in (
            "enable_kill_override_chain_completion",
            "enable_kill_override_attribution",
            "enable_early_fire_clear_on_finalize",
        ):
            assert name in sig.parameters, f"{name} が generate() に無い"
            assert sig.parameters[name].default is False

    def test_cli_flags_wired_to_main(self) -> None:
        import scripts.visualize_advantage_overlay as vao
        text = Path(vao.__file__).read_text(encoding="utf-8")
        assert "--kill-override-chain-completion" in text
        assert (
            "enable_kill_override_chain_completion="
            "a.enable_kill_override_chain_completion" in text
        )
        assert "--kill-override-attribution" in text
        assert (
            "enable_kill_override_attribution=a.enable_kill_override_attribution"
            in text
        )
        assert "--early-fire-clear-on-finalize" in text
        assert (
            "enable_early_fire_clear_on_finalize="
            "a.enable_early_fire_clear_on_finalize" in text
        )

    def test_call_site_falls_back_to_raw_snap_pending_when_flag_off(self) -> None:
        """既定 False では kpending1/kpending2 = snap.pending_p1/p2 の単純代入
        (=修正前と bit-identical) であることをソースで固定する。"""
        import inspect
        import scripts.visualize_advantage_overlay as vao
        src = inspect.getsource(vao.generate)
        code_only = src.replace(vao.generate.__doc__ or "", "")
        assert (
            "if enable_kill_override_chain_completion:" in code_only
        )
        assert (
            "kroom1, kroom2 = room1, room2" in code_only
        )

    def test_call_site_early_fire_default_path_still_unconditional_clear(self) -> None:
        """既定 False では efire_tracker.on_settled() (引数無し=無条件クリア)
        が実行されるパスが残っていることを固定する (backwards compat)。"""
        import inspect
        import scripts.visualize_advantage_overlay as vao
        src = inspect.getsource(vao.generate)
        code_only = src.replace(vao.generate.__doc__ or "", "")
        assert "efire_tracker.on_settled()  # 確定計算が入ったので速報バイアスをクリア" in code_only
        assert "efire_tracker.on_settled(finalized=_finalized)" in code_only


class TestChainGenerationAccumulatorWiring:
    """2026-08-22 改良②: ChainGenerationAccumulator が generate() から実際に
    使われ、その累積値が kill_override まで届く配線を固定する (静的回帰テスト)。

    背景 (coordinator指摘、2026-08-22 20:40): 単発検証 (t=6664.17起点の短区間)
    では accum_gen1=840 まで累積したのに、全編再走査 (t=6131.6起点) では
    効果が出ず (112→112、v1単発版とほぼ同一)。配線漏れの可能性が疑われた
    ため、以下の4点をソーステキストで固定する
    (本日 (2026-08-22) 同種の配線漏れを複数捕まえている前例に倣う形)。
    """

    def test_fresh_trackers_signature_includes_chain_gen_accumulator(self) -> None:
        """_fresh_trackers() の戻り値型注釈に ChainGenerationAccumulator が
        含まれること (未使用のまま追加しただけ、で終わっていないかの入口確認)。"""
        import inspect
        import scripts.visualize_advantage_overlay as vao
        src = inspect.getsource(vao._fresh_trackers)
        assert "ChainGenerationAccumulator" in src
        assert "return (tracker" in src
        assert "ChainGenerationAccumulator(accumulate=enable_chain_gen_accumulate)" in src

    def test_generate_source_creates_chain_gen_tracker_instance(self) -> None:
        """generate() 内でインスタンスが実際に作られている (初期生成+match境界
        再生成の両方)。"""
        import inspect
        import scripts.visualize_advantage_overlay as vao
        src = inspect.getsource(vao.generate)
        code_only = src.replace(vao.generate.__doc__ or "", "")
        assert (
            "chain_gen_tracker = ChainGenerationAccumulator(\n"
            "        accumulate=enable_kill_override_chain_gen_accumulate)"
            in code_only
        )
        assert "efire_tracker, chain_gen_tracker) = _fresh_trackers(" in code_only

    def test_generate_source_calls_update_unconditionally_every_frame(self) -> None:
        """settled ゲートの外側 (毎フレーム実行される箇所) で
        chain_gen_tracker.update() が呼ばれていることを固定する
        (EarlyFireTracker.update と同じ間引き回避パターン)。呼び出しが
        settled ブロックの**内側**に紛れ込むと、per-side-settled 下での
        高頻度更新のたびに再計算はされても、settled=False の間の
        trigger_sec 変化を取りこぼす退行になる。"""
        import inspect
        import scripts.visualize_advantage_overlay as vao
        src = inspect.getsource(vao.generate)
        code_only = src.replace(vao.generate.__doc__ or "", "")
        # 「settled 内側の呼出し箇所 (_kill_override_chain_completion_inputs)」
        # より前に「毎フレーム呼出し箇所 (chain_gen_tracker.update)」が
        # 出現することをテキスト位置で固定する (settled ブロックの外側にある
        # ことの代理検査)。
        # 実際の呼出し文だけにマッチする (コメント中の言及と区別するため代入文
        # まるごとを検索文字列にする)。
        idx_update = code_only.index(
            "chain_gen2, chain_gen_before2) = chain_gen_tracker.update(")
        idx_settled_use = code_only.index("_kill_override_chain_completion_inputs(\n                        snap,")
        assert idx_update < idx_settled_use, (
            "chain_gen_tracker.update() が settled 内側の使用箇所より後ろに"
            "ある = settled ゲートの外側で毎フレーム呼ばれていない疑い"
        )
        # settled ブロックの外側 (efire_tracker.update と同じ if ブロック群) に
        # あることも直接固定する: efire_tracker.update 呼出しと
        # chain_gen_tracker.update 呼出しの間に settled 判定
        # (`if b1 is not None and b2 is not None and settled:`) が
        # 挟まっていないこと。
        idx_efire = code_only.index("efire_tracker.update(")
        idx_settled_gate = code_only.index(
            "if b1 is not None and b2 is not None and settled:")
        assert idx_efire < idx_update < idx_settled_gate

    def test_generate_source_wires_accumulator_output_into_kill_override_inputs(
        self,
    ) -> None:
        """settled ブロック内で chain_gen_tracker.update() の戻り値
        (chain_gen1/chain_gen_before1/chain_gen2/chain_gen_before2) が
        そのまま _kill_override_chain_completion_inputs へ渡り、その戻り値
        (kroom1/kroom2/kpending1/kpending2) がそのまま kill_override() へ
        渡ることを固定する (途中で握りつぶされていないか)。"""
        import inspect
        import scripts.visualize_advantage_overlay as vao
        src = inspect.getsource(vao.generate)
        code_only = src.replace(vao.generate.__doc__ or "", "")
        # (2026-08-24 更新) A案「規模の比較」で pending_p1/p2_override が末尾に
        # 追加されたため、終端は `))` ではなく `,` になった (accumulator 出力
        # 4値がそのまま渡る事実は不変)。
        assert (
            "chain_gen1, chain_gen_before1,\n                        "
            "chain_gen2, chain_gen_before2," in code_only
        )
        assert "adv = kill_override(adv, kpending1, kpending2," in code_only
        assert "kroom1, kroom2)" in code_only

    def test_generate_has_accumulate_flag_default_false_and_wired(self) -> None:
        """[2026-08-22 user判断] 累積モード切替フラグが generate() の両方の
        ChainGenerationAccumulator 生成箇所 (初期生成+match境界再生成) へ
        正しく届いていることを固定する。"""
        import inspect
        import scripts.visualize_advantage_overlay as vao
        sig = inspect.signature(vao.generate)
        assert "enable_kill_override_chain_gen_accumulate" in sig.parameters
        assert sig.parameters["enable_kill_override_chain_gen_accumulate"].default is False
        src = inspect.getsource(vao.generate)
        code_only = src.replace(vao.generate.__doc__ or "", "")
        assert (
            "accumulate=enable_kill_override_chain_gen_accumulate)" in code_only
        )
        assert (
            "enable_chain_gen_accumulate=enable_kill_override_chain_gen_accumulate)"
            in code_only
        )

    def test_cli_flag_wired_to_main(self) -> None:
        import scripts.visualize_advantage_overlay as vao
        text = Path(vao.__file__).read_text(encoding="utf-8")
        assert "--kill-override-chain-gen-accumulate" in text
        assert (
            "enable_kill_override_chain_gen_accumulate=(\n"
            "                 a.enable_kill_override_chain_gen_accumulate)"
            in text
        )


class TestChainHoldCalibrationWiring:
    """2026-08-22 修正②根治: CHAIN 保持時間の実測較正値 (src/recognition_
    pipeline.py:731-736、2.61+1.17×N) が generate() から RecognitionPipeline
    へ渡る配線を固定する (静的回帰テスト)。

    背景: この較正値は 2026-07-24 に実測済みだったが、visualize_advantage_
    overlay.py には一度も CLI フラグ化されておらず、本番は常にライブラリ既定
    (base=0.0, per_step=0.3固定) のままだった。これが t=6717.5 の formula
    再トリガー間隔 (~1.4秒 ≒ 0.3×5) と一致する断片化の直接原因だった。
    """

    def test_generate_has_new_params_default_none(self) -> None:
        import inspect
        import scripts.visualize_advantage_overlay as vao
        sig = inspect.signature(vao.generate)
        for name in ("chain_hold_base_sec", "chain_hold_per_step_sec"):
            assert name in sig.parameters, f"{name} が generate() に無い"
            assert sig.parameters[name].default is None

    def test_cli_flags_wired_to_main(self) -> None:
        import scripts.visualize_advantage_overlay as vao
        text = Path(vao.__file__).read_text(encoding="utf-8")
        assert "--chain-hold-base-sec" in text
        assert "chain_hold_base_sec=a.chain_hold_base_sec" in text
        assert "--chain-hold-per-step-sec" in text
        assert "chain_hold_per_step_sec=a.chain_hold_per_step_sec" in text

    def test_generate_source_passes_through_only_when_not_none(self) -> None:
        """既定 None ではキー自体を RecognitionPipeline.load_default に渡さず
        (=ライブラリ既定 0.0/0.3 のまま、bit-identical)、明示指定時のみ渡る。"""
        import inspect
        import scripts.visualize_advantage_overlay as vao
        src = inspect.getsource(vao.generate)
        code_only = src.replace(vao.generate.__doc__ or "", "")
        assert '"chain_hold_base_sec": chain_hold_base_sec' in code_only
        assert '"chain_hold_per_step_sec": chain_hold_per_step_sec' in code_only
        assert "if chain_hold_base_sec is not None else {}" in code_only
        assert "if chain_hold_per_step_sec is not None else {}" in code_only
