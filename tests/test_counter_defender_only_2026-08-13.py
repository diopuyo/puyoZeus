"""#4/#5 修正 (--counter-defender-only) の意味論テスト。

docs/DEMO_REVIEW_2026-08-13.md #4/#5: (a) 固定閾値を両者常時計算・表示する
問題 (b) 応手100%でも勝率がほぼ死亡表示になる統合欠陥、を修正する。
受け側限定判定・実飛来量ベースの閾値・非線形ダメージ統合・表示行の
それぞれを単体テストで検証する。
"""
from __future__ import annotations

import math

from src.board import Board
from src.chain_detector import ChainEvent
from src.ojama_accounting import OjamaAccountSnapshot
from src.scoring import score_to_ojama
from scripts.visualize_advantage_overlay import (
    COUNTER_SCALE,
    CounterReachTracker,
    _ChainAttackObservation,
    _build_counter_text_defender_only,
    _counter_defender_adv,
    _incoming_ojama_for_defender,
    _resolve_counter_text,
    _resolve_defender_threat,
)


def _snap(pending_p1: int = 0, pending_p2: int = 0) -> OjamaAccountSnapshot:
    """テスト用の最小 OjamaAccountSnapshot (pending 以外は無害な既定値)。"""
    return OjamaAccountSnapshot(
        t_sec=0.0, pending_p1=pending_p1, pending_p2=pending_p2,
        total_generated_by_p1=0, total_generated_by_p2=0,
        total_offset_by_p1=0, total_offset_by_p2=0,
        total_dropped_to_p1=0, total_dropped_to_p2=0,
        net_ojama_balance=pending_p2 - pending_p1,
        overflow_risk_p1=False, overflow_risk_p2=False,
        confidence=1.0, leftover_p1=0, leftover_p2=0,
        all_clear_pending_p1=False, all_clear_pending_p2=False,
    )


def _chain_event(total_score: int, trigger_sec: float = 0.0, chain_count: int = 1) -> ChainEvent:
    return ChainEvent(
        trigger_sec=trigger_sec, end_sec=trigger_sec + 1.0, before_board=Board(),
        chain_count=chain_count, total_erased=4, total_score=total_score,
        base_score=total_score, all_clear_bonus_applied=0, ojama_sent=0,
        leftover_score=0, is_all_clear=False,
    )


def _obs(attacker_side, attacker_event=None, chain_count=0) -> _ChainAttackObservation:
    return _ChainAttackObservation(
        chain_count=chain_count, trigger_sec=0.0,
        attacker_side=attacker_side, attacker_event=attacker_event,
    )


# ============================
# _resolve_defender_threat / _incoming_ojama_for_defender
# ============================


def test_no_attack_no_pending_returns_no_defender() -> None:
    obs = _obs(None)
    side, incoming = _resolve_defender_threat(obs, _snap(), elapsed_sec=0.0)
    assert side is None
    assert incoming == 0.0


def test_pending_only_without_active_chain_event_still_detected() -> None:
    """連鎖イベントが無くても、予告おじゃま (pending) > 0 だけで脅威が成立する
    (docs/DEMO_REVIEW_2026-08-13.md #5 item1)。"""
    obs = _obs(None)  # 攻撃側 chain_event は無い
    side, incoming = _resolve_defender_threat(obs, _snap(pending_p1=8), elapsed_sec=0.0)
    assert side == "1P"
    assert incoming == 8.0


def test_active_chain_event_without_pending_detected() -> None:
    ev = _chain_event(total_score=700)  # score_to_ojama(700)=10個 (70点/個)
    obs = _obs("1P", attacker_event=ev, chain_count=3)
    side, incoming = _resolve_defender_threat(obs, _snap(), elapsed_sec=0.0)
    assert side == "2P"
    expected = float(score_to_ojama(700, elapsed_sec=0.0).ojama_count)
    assert incoming == expected
    assert incoming > 0.0


def test_chain_event_and_pending_are_summed() -> None:
    ev = _chain_event(total_score=700)
    obs = _obs("1P", attacker_event=ev, chain_count=3)
    side, incoming = _resolve_defender_threat(obs, _snap(pending_p2=5), elapsed_sec=0.0)
    assert side == "2P"
    chain_ojama = score_to_ojama(700, elapsed_sec=0.0).ojama_count
    assert incoming == chain_ojama + 5.0


def test_incoming_ojama_for_defender_ignores_unrelated_attacker_side() -> None:
    """1Pが攻撃中でも、2P向け (defender_side="1P") の chain_ojama は0
    (自分の攻撃を自分への脅威に数えない)。"""
    ev = _chain_event(total_score=700)
    obs = _obs("1P", attacker_event=ev, chain_count=3)
    incoming_for_1p = _incoming_ojama_for_defender("1P", obs, _snap(), elapsed_sec=0.0)
    assert incoming_for_1p == 0.0


# ============================
# _counter_defender_adv (方向性・極端化抑制)
# ============================


def test_counter_defender_adv_1p_defender_is_negative_direction() -> None:
    """1Pが受け側 (=2Pが攻撃中) の脅威は 1P視点でマイナス方向に効く。"""
    b1, b2 = Board(), Board()
    adv = _counter_defender_adv("1P", defender_prob=0.0, incoming_ojama=20.0, b1=b1, b2=b2)
    assert adv < 0.0


def test_counter_defender_adv_2p_defender_is_positive_direction() -> None:
    b1, b2 = Board(), Board()
    adv = _counter_defender_adv("2P", defender_prob=0.0, incoming_ojama=20.0, b1=b1, b2=b2)
    assert adv > 0.0


def test_counter_defender_adv_shrinks_toward_zero_as_prob_increases() -> None:
    """受け側が高確率で返せるほど、統合成分は 0 に近づく (極端化抑制の核心、
    docs/DEMO_REVIEW_2026-08-13.md #4)。"""
    b1, b2 = Board(), Board()
    adv_low_prob = _counter_defender_adv("1P", defender_prob=0.0, incoming_ojama=20.0, b1=b1, b2=b2)
    adv_high_prob = _counter_defender_adv("1P", defender_prob=0.99, incoming_ojama=20.0, b1=b1, b2=b2)
    assert abs(adv_high_prob) < abs(adv_low_prob)


def test_counter_defender_adv_prob_one_gives_exactly_zero() -> None:
    b1, b2 = Board(), Board()
    adv = _counter_defender_adv("1P", defender_prob=1.0, incoming_ojama=50.0, b1=b1, b2=b2)
    assert adv == 0.0


def test_counter_defender_adv_nan_prob_returns_zero() -> None:
    b1, b2 = Board(), Board()
    adv = _counter_defender_adv("1P", defender_prob=float("nan"), incoming_ojama=20.0, b1=b1, b2=b2)
    assert adv == 0.0


def test_counter_defender_adv_bounded_by_counter_scale() -> None:
    """値域は既存 COUNTER_SCALE を超えない (新規定数を作らない設計の確認)。"""
    b1, b2 = Board(), Board()
    adv = _counter_defender_adv("2P", defender_prob=0.0, incoming_ojama=1000.0, b1=b1, b2=b2)
    assert abs(adv) <= COUNTER_SCALE + 1e-9


# ============================
# 表示行 (#5)
# ============================


def test_build_counter_text_defender_only_empty_when_no_threat() -> None:
    assert _build_counter_text_defender_only(None, float("nan"), 0.0) == ""


def test_build_counter_text_defender_only_formats_when_threat() -> None:
    text = _build_counter_text_defender_only("2P", 0.42, 18.0)
    assert "2P" in text
    assert "42" in text
    assert "18" in text


def test_resolve_counter_text_uses_legacy_when_flag_off() -> None:
    text = _resolve_counter_text(False, None, 0.3, 0.7, 0.0)
    assert "1P" in text and "2P" in text  # 従来の両側表示


def test_resolve_counter_text_defender_only_hides_when_no_threat() -> None:
    text = _resolve_counter_text(True, None, float("nan"), float("nan"), 0.0)
    assert text == ""


# ============================
# CounterReachTracker.update(defender_side=...)
# ============================


def _tracker_with_stub_reach() -> tuple[CounterReachTracker, dict]:
    tracker = CounterReachTracker()
    calls: list[tuple] = []

    def _stub_reach(board, budget_sec, known_pairs, threshold_ojama=0.0):
        calls.append((budget_sec, threshold_ojama))
        return 0.6, 2.0

    tracker._reach = _stub_reach  # type: ignore[method-assign]
    return tracker, {"calls": calls}


def test_update_defender_only_computes_single_side() -> None:
    tracker, state = _tracker_with_stub_reach()
    b1, b2 = Board(), Board()
    adv, p1, p2 = tracker.update(
        b1, b2, budget_sec=1.0, t_sec=0.0, defender_side="1P", threshold_ojama=18.0)
    assert adv == 0.0  # 呼び出し側 (generate) が別途計算するため 0.0 固定
    assert p1 == 0.6
    assert math.isnan(p2)
    assert len(state["calls"]) == 1  # 1回のみ (両側計算より半分)
    assert state["calls"][0] == (1.0, 18.0)


def test_update_defender_only_budget_zero_returns_nan() -> None:
    tracker, state = _tracker_with_stub_reach()
    b1, b2 = Board(), Board()
    adv, p1, p2 = tracker.update(
        b1, b2, budget_sec=0.0, t_sec=0.0, defender_side="2P", threshold_ojama=12.0)
    assert adv == 0.0
    assert math.isnan(p1)
    assert math.isnan(p2)
    assert len(state["calls"]) == 0


def test_update_defender_side_none_preserves_legacy_symmetric_path() -> None:
    """defender_side 省略 (None) は従来通り両側計算する (backwards compat)。

    b1/b2 はキャッシュキー衝突 (盤面内容が同一だと1回に丸められる、
    tests/test_counter_reach_time_throttle.py と同じ既知の仕様) を避ける
    ため異なる盤面にする。
    """
    from src.board import COLOR_BLUE, COLOR_RED
    tracker, state = _tracker_with_stub_reach()
    b1, b2 = Board(), Board()
    b1.set(12, 0, COLOR_RED)
    b2.set(12, 5, COLOR_BLUE)
    tracker.update(b1, b2, budget_sec=1.0, t_sec=0.0)
    assert len(state["calls"]) == 2
