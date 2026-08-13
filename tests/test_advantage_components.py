"""有利不利オーバーレイの合成成分(圧力/得点リード)の単体テスト。

PressureTracker(着弾ダメージの持続記憶)と ScoreLeadTracker(得点リード)は
純ロジックなので認識なしで検証できる。回帰防止用。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import types

import numpy as np
import pytest

from src.board import Board  # noqa: E402
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
