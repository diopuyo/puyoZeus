"""±100 張り付き/1フレーム符号反転 (2026-08-24 根治) のテスト。

対象 (memory project_pm100_display_flip_2026-08-24):
  - B案: `KillOverrideConfidenceGate` (--kill-override-hysteresis)。
    根因③ (ChainEvent 断片化による1フレーム ±100→∓100 反転) の構造的禁止。
  - A案: 「規模の比較」(--kill-override-scale-compare)。
    根因① (撃ち返しが会計に載っていない = finalize 遅延中の送付分欠落) を
    `PostChainUnregisteredSentTracker` が、根因② (PENDING_ABS_CAP=216 が相殺の
    引き算を汚染) を会計の cap 切り捨てなし並行帳簿 (pending_p1/p2_uncapped)
    が是正する。
  - 静的配線回帰 (feedback_wiring_gap_vs_wiring_error_2026-08-22:
    配線「漏れ」と「間違い」の両方を検出するため、配線先をソーステキストで固定)。

全フラグ既定 OFF で従来挙動と bit-identical であることが絶対条件
(別途 実dump の md5 比較でも検収する。ここでは単体レベルの不変条件を固定)。
"""
from __future__ import annotations

import inspect
import re
import types
from pathlib import Path

import pytest

from src.board import BOARD_ROWS, Board
from src.board_state_machine import BoardState
from src.ojama_accounting import (
    CHAIN_TOTAL_SANITY_MAX,
    PENDING_ABS_CAP,
    PENDING_UNCAPPED_SANITY_MAX,
    OjamaAccountingTracker,
)
from src.scoring import OJAMA_RATE_STANDARD

import scripts.mc_counter_estimator as mc_counter
import scripts.visualize_advantage_overlay as vao


VAO_SOURCE = Path(vao.__file__).read_text(encoding="utf-8")


# ============================
# ヘルパー
# ============================

def _busy_side() -> types.SimpleNamespace:
    return types.SimpleNamespace(chain_event=None, state=BoardState.CHAIN)


def _idle_side() -> types.SimpleNamespace:
    return types.SimpleNamespace(chain_event=None, state=BoardState.STABLE)


def _snap_unc(p1: float = 0.0, p2: float = 0.0) -> types.SimpleNamespace:
    """PostChainUnregisteredSentTracker が読む最小スナップショット。"""
    return types.SimpleNamespace(pending_p1_uncapped=p1, pending_p2_uncapped=p2)


def _board_with_ojama(n: int) -> Board:
    """可視領域 (row1-12) に n 個のおじゃまを置いた盤面。"""
    b = Board()
    placed = 0
    for row in range(BOARD_ROWS - 1, 0, -1):
        for col in range(6):
            if placed >= n:
                return b
            b.set(row, col, 9)
            placed += 1
    return b


def _fire_chain(
    tracker: OjamaAccountingTracker, side: str, chain_score: int,
    score_before: int = 0, t_sec: float = 5.0,
) -> None:
    """連鎖開始→終了→score settle を一連でシミュレート (tests/test_ojama_
    accounting.py の同名ヘルパーと同一方式)。"""
    from src.ojama_accounting import K_SETTLE_FRAMES
    score_after = score_before + chain_score
    tracker.on_state_transition(
        side, BoardState.STABLE, BoardState.CHAIN, score_before, t_sec)
    tracker.on_state_transition(
        side, BoardState.CHAIN, BoardState.STABLE, score_after, t_sec + 2.0)
    _dt = 0.001
    for i in range(K_SETTLE_FRAMES):
        tracker.on_state_transition(
            side, BoardState.STABLE, BoardState.STABLE,
            score_after, t_sec + 2.0 + (i + 1) * _dt)


# ============================
# 定数の導出テスト (物理量からの導出、シーン逆算でないことの固定)
# ============================

def test_gate_constants_derived_from_physical_quantities() -> None:
    """B案の定数は既存の物理実測値からの導出 (マジックナンバー禁止)。"""
    assert vao.KILL_FLIP_COOLDOWN_SEC == mc_counter.BEAM_ROLLOUT_AVG_STEP_TIME_SEC
    assert vao.KILL_CONFIRM_PERSIST_SEC == pytest.approx(
        2.0 * mc_counter.BEAM_ROLLOUT_AVG_STEP_TIME_SEC + 8.0 / 30.0)
    assert vao.KILL_UNCONFIRMED_ABS_CAP == 90.0
    # 1手 (≈0.348s) < 確定 (≈0.963s) < 断片化周期 (1.37s) の関係が崩れていない
    assert vao.KILL_FLIP_COOLDOWN_SEC < vao.KILL_CONFIRM_PERSIST_SEC


def test_uncapped_sanity_cap_derived_from_existing_constants() -> None:
    """並行帳簿のサニティ上限は既存定数からの導出 (200,000点 ÷ 70点/個)。"""
    assert PENDING_UNCAPPED_SANITY_MAX == CHAIN_TOTAL_SANITY_MAX // OJAMA_RATE_STANDARD
    assert PENDING_UNCAPPED_SANITY_MAX > PENDING_ABS_CAP


# ============================
# B案: KillOverrideConfidenceGate
# ============================

def test_gate_passthrough_when_not_fired() -> None:
    """kill_override 未発火 (adv_post == adv_pre) では素通し。"""
    gate = vao.KillOverrideConfidenceGate()
    assert gate.apply(12.5, 12.5, 0.0) == 12.5


def test_gate_first_fire_capped_at_90() -> None:
    """初回発火は持続確認前なので ±90 で頭打ち (±100 を即断しない)。"""
    gate = vao.KillOverrideConfidenceGate()
    assert gate.apply(20.0, 100.0, 0.0) == vao.KILL_UNCONFIRMED_ABS_CAP


def test_gate_full_override_after_persistence() -> None:
    """同一方向が KILL_CONFIRM_PERSIST_SEC 持続したら完全上書きを許可。"""
    gate = vao.KillOverrideConfidenceGate()
    gate.apply(20.0, 100.0, 0.0)
    assert gate.apply(20.0, 100.0, 0.5) == vao.KILL_UNCONFIRMED_ABS_CAP  # まだ未満
    assert gate.apply(20.0, 100.0, 1.0) == 100.0  # 1.0s ≥ 0.963s


def test_gate_one_frame_reversal_blocked_2026_08_24() -> None:
    """[核心・根因③再現] 確定済み +100 の直後、1フレームだけ逆方向 (−100) の
    上書きが来ても表示は反転しない (納品動画 t=211.40→211.43 の再現:
    ChainEvent 断片化で gen1 442→101、実世界では何も起きていないのに
    +100→−100 へ反転した)。クールダウン中は上書き前のブレンド値へ戻す。"""
    gate = vao.KillOverrideConfidenceGate()
    gate.apply(20.0, 100.0, 0.0)
    assert gate.apply(20.0, 100.0, 1.2) == 100.0  # 持続確認済み
    # 1フレーム後 (33ms) に逆方向: ブロックされ adv_pre (+18) が出る
    assert gate.apply(18.0, -100.0, 1.233) == 18.0


def test_gate_genuine_reversal_confirms_after_cooldown_and_persistence() -> None:
    """本物の反転はクールダウン (1手) → ±90 → 持続確認後に ±100 の順で確定する
    (真の致死の見逃し防止: 遅延は最大でも約1秒)。"""
    gate = vao.KillOverrideConfidenceGate()
    gate.apply(20.0, 100.0, 0.0)
    gate.apply(20.0, 100.0, 1.2)  # +方向確定
    t_flip = 1.233
    assert gate.apply(18.0, -100.0, t_flip) == 18.0  # 反転直後: 保留
    t_after_cooldown = t_flip + vao.KILL_FLIP_COOLDOWN_SEC + 0.02
    assert gate.apply(18.0, -100.0, t_after_cooldown) == -vao.KILL_UNCONFIRMED_ABS_CAP
    t_confirmed = t_flip + vao.KILL_CONFIRM_PERSIST_SEC + 0.02
    assert gate.apply(18.0, -100.0, t_confirmed) == -100.0


def test_gate_direction_memory_resets_after_fire_gap() -> None:
    """発火が1手時間を超えて途切れたら方向の記憶を破棄する (次の発火は新規
    事象として再び持続確認から始まる)。"""
    gate = vao.KillOverrideConfidenceGate()
    gate.apply(20.0, 100.0, 0.0)
    gate.apply(30.0, 30.0, 1.0)  # 未発火が1手時間超 → リセット
    # リセットされていなければ 1.2-0.0=1.2s ≥ 0.963s で 100 が出てしまう
    assert gate.apply(20.0, 100.0, 1.2) == vao.KILL_UNCONFIRMED_ABS_CAP


def test_gate_does_not_weaken_below_adv_pre() -> None:
    """adv_pre 自身が ±90 を超えている場合、ゲートがモデルより弱い方向へ
    働かない (上限は max(|adv_pre|, 90))。"""
    gate = vao.KillOverrideConfidenceGate()
    assert gate.apply(95.0, 100.0, 0.0) == 95.0


def test_gate_partial_override_within_cap_untouched() -> None:
    """±90 未満の部分上書き (g<1) は制限に掛からずそのまま通る。"""
    gate = vao.KillOverrideConfidenceGate()
    assert gate.apply(20.0, 55.0, 0.0) == 55.0


# ============================
# A案(i-a): PostChainUnregisteredSentTracker
# ============================

def test_sent_tracker_captures_on_completion() -> None:
    """busy→非busy 遷移 (連鎖完走) で生成量を「宛先への未登録分」として保持。"""
    tr = vao.PostChainUnregisteredSentTracker()
    tr.update(_busy_side(), _idle_side(), _snap_unc(), None, None, 517.0, 0.0, 0.0)
    extra_p1, extra_p2 = tr.update(
        _idle_side(), _idle_side(), _snap_unc(), None, None, 0.0, 0.0, 1.0)
    assert (extra_p1, extra_p2) == (0.0, 517.0)  # 1Pが送った分は2P宛て


def test_sent_tracker_reduces_on_accounting_registration() -> None:
    """宛先の pending (並行帳簿) が増えたら会計が追い付いたとみなし減額。"""
    tr = vao.PostChainUnregisteredSentTracker()
    tr.update(_busy_side(), _idle_side(), _snap_unc(), None, None, 517.0, 0.0, 0.0)
    tr.update(_idle_side(), _idle_side(), _snap_unc(), None, None, 0.0, 0.0, 1.0)
    extra_p1, extra_p2 = tr.update(
        _idle_side(), _idle_side(), _snap_unc(p2=517.0), None, None, 0.0, 0.0, 2.0)
    assert (extra_p1, extra_p2) == (0.0, 0.0)


def test_sent_tracker_reduces_on_board_landing() -> None:
    """宛先の盤面おじゃまセルが増えたら着弾済みとみなし減額。"""
    tr = vao.PostChainUnregisteredSentTracker()
    b2_before = _board_with_ojama(0)
    b2_after = _board_with_ojama(30)
    tr.update(_busy_side(), _idle_side(), _snap_unc(), None, b2_before,
              517.0, 0.0, 0.0)
    tr.update(_idle_side(), _idle_side(), _snap_unc(), None, b2_before,
              0.0, 0.0, 1.0)
    extra_p1, extra_p2 = tr.update(
        _idle_side(), _idle_side(), _snap_unc(), None, b2_after, 0.0, 0.0, 2.0)
    assert extra_p2 == pytest.approx(517.0 - 30.0)


def test_sent_tracker_mutual_offset_on_counter_completion_2026_08_24() -> None:
    """[核心・根因①再現] 納品動画 seg01 game2 の撃ち合い: 1P が517個を送付
    (会計未登録) → 2P が720個で即撃ち返し。従来は517が計算から欠落し720全量が
    1Pへの新規攻撃になった (kpend1=690, 比31 ≫ 1.5 → −100へ反転)。
    本トラッカーは完走時に相互相殺し、1Pへ向かうのは差分 203 だけになる。"""
    tr = vao.PostChainUnregisteredSentTracker()
    tr.update(_busy_side(), _idle_side(), _snap_unc(), None, None, 517.0, 0.0, 0.0)
    tr.update(_idle_side(), _idle_side(), _snap_unc(), None, None, 0.0, 0.0, 1.0)
    tr.update(_idle_side(), _busy_side(), _snap_unc(), None, None, 0.0, 720.0, 2.0)
    extra_p1, extra_p2 = tr.update(
        _idle_side(), _idle_side(), _snap_unc(), None, None, 0.0, 0.0, 3.0)
    assert extra_p1 == pytest.approx(720.0 - 517.0)  # 1P宛ては差分203のみ
    assert extra_p2 == 0.0  # 517は相殺で消えた


def test_sent_tracker_expires_after_deadline() -> None:
    """期限 (UNREGISTERED_SENT_EXPIRE_SEC) までにどの観測にも現れない保持分は
    破棄する (観測エラー扱い、架空の攻撃を残さない)。"""
    tr = vao.PostChainUnregisteredSentTracker()
    tr.update(_busy_side(), _idle_side(), _snap_unc(), None, None, 517.0, 0.0, 0.0)
    tr.update(_idle_side(), _idle_side(), _snap_unc(), None, None, 0.0, 0.0, 1.0)
    t_expired = 1.0 + vao.UNREGISTERED_SENT_EXPIRE_SEC + 1.0
    extra_p1, extra_p2 = tr.update(
        _idle_side(), _idle_side(), _snap_unc(), None, None, 0.0, 0.0, t_expired)
    assert (extra_p1, extra_p2) == (0.0, 0.0)


def test_sent_tracker_subtracts_own_pending_at_capture() -> None:
    """完走側の生成はまず自分宛て pending の相殺に使われる (実会計と同じ向き)。
    保持へ登録するのは相殺後の余剰だけ (二重相殺防止)。"""
    tr = vao.PostChainUnregisteredSentTracker()
    tr.update(_busy_side(), _idle_side(), _snap_unc(p1=40.0), None, None,
              100.0, 0.0, 0.0)
    extra_p1, extra_p2 = tr.update(
        _idle_side(), _idle_side(), _snap_unc(p1=40.0), None, None, 0.0, 0.0, 1.0)
    assert extra_p2 == pytest.approx(60.0)


def test_sent_tracker_keeps_max_gen_against_fragmentation() -> None:
    """busy 中に gen が瞬間的に落ちても (根因③ 断片化: 442→101)、同一連鎖内の
    最大値を「少なくとも生成した量」として捕捉する。"""
    tr = vao.PostChainUnregisteredSentTracker()
    tr.update(_busy_side(), _idle_side(), _snap_unc(), None, None, 442.0, 0.0, 0.0)
    tr.update(_busy_side(), _idle_side(), _snap_unc(), None, None, 101.0, 0.0, 0.5)
    extra_p1, extra_p2 = tr.update(
        _idle_side(), _idle_side(), _snap_unc(), None, None, 0.0, 0.0, 1.0)
    assert extra_p2 == pytest.approx(442.0)


# ============================
# A案(ii): 会計の cap 切り捨てなし並行帳簿
# ============================

def test_accounting_uncapped_snapshot_defaults_zero() -> None:
    """初期状態では並行帳簿も 0 (既存フィールドと同様)。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    snap = tracker.get_snapshot(t_sec=0.0)
    assert snap.pending_p1_uncapped == 0
    assert snap.pending_p2_uncapped == 0


def test_accounting_uncapped_keeps_real_value_beyond_abs_cap() -> None:
    """517個送付: 表示用 pending は216で丸まるが、並行帳簿は実額517を保持。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    _fire_chain(tracker, "p1", chain_score=517 * OJAMA_RATE_STANDARD, t_sec=5.0)
    snap = tracker.get_snapshot(t_sec=8.0)
    assert snap.pending_p2 == PENDING_ABS_CAP  # 表示用は従来通り216 (不変)
    assert snap.pending_p2_uncapped == 517


def test_accounting_uncapped_avoids_phantom_surplus_2026_08_24() -> None:
    """[核心・根因②再現] 1Pが517送付 → 2Pが720で撃ち返し。
    capped 帳簿は 216 に丸めた後で相殺するため架空の余剰 720−216=504 を
    1Pへ送る (さらに216へ再クランプ)。並行帳簿は実額で相殺し、1Pへ向かうのは
    真値 720−517=203 になる。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    _fire_chain(tracker, "p1", chain_score=517 * OJAMA_RATE_STANDARD, t_sec=5.0)
    _fire_chain(tracker, "p2", chain_score=720 * OJAMA_RATE_STANDARD, t_sec=15.0)
    snap = tracker.get_snapshot(t_sec=18.0)
    assert snap.pending_p1 == PENDING_ABS_CAP  # capped 帳簿は架空余剰504→216 (従来挙動)
    assert snap.pending_p1_uncapped == 203  # 並行帳簿は真値
    assert snap.pending_p2_uncapped == 0


def test_accounting_uncapped_drains_independently() -> None:
    """drain は各帳簿の残量から独立に計算する (実残量517側は30/手で減る)。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    _fire_chain(tracker, "p1", chain_score=517 * OJAMA_RATE_STANDARD, t_sec=5.0)
    tracker.on_tsumo_settled("p2", t_sec=9.0)
    snap = tracker.get_snapshot(t_sec=9.5)
    assert snap.pending_p2 == PENDING_ABS_CAP - 30
    assert snap.pending_p2_uncapped == 517 - 30


def test_accounting_uncapped_resets_on_match_boundary() -> None:
    """試合境界 (score 大幅減少) で並行帳簿も必ずゼロに戻る。

    境界リセットは既存仕様どおりサイド単位 (score が落ちた側の帳簿を消す)
    なので、pending_p2 (=p2側の帳簿) を消すには p2 側の score 減少が必要。
    capped 帳簿と同じリセット意味論に並行帳簿が従うことを固定する。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    _fire_chain(tracker, "p1", chain_score=517 * OJAMA_RATE_STANDARD, t_sec=5.0)
    # p2 に score を持たせてから 0 へ大幅減少させる = p2 側の試合境界
    _fire_chain(tracker, "p2", chain_score=700, t_sec=15.0)
    tracker.on_state_transition(
        "p2", BoardState.STABLE, BoardState.STABLE, 0, 30.0)
    snap = tracker.get_snapshot(t_sec=31.0)
    assert snap.pending_p2 == 0  # capped 帳簿のリセット (従来挙動)
    assert snap.pending_p2_uncapped == 0  # 並行帳簿も同じ意味論


def test_accounting_uncapped_bounded_by_sanity_max() -> None:
    """並行帳簿は PENDING_UNCAPPED_SANITY_MAX (=2857) だけを防波堤にする。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    per_chain = 2000  # 2000個×2回 = 4000 > 2857
    score = per_chain * OJAMA_RATE_STANDARD  # 140,000 < CHAIN_TOTAL_SANITY_MAX
    _fire_chain(tracker, "p1", chain_score=score, t_sec=5.0)
    _fire_chain(tracker, "p1", chain_score=score, score_before=score, t_sec=15.0)
    snap = tracker.get_snapshot(t_sec=18.0)
    assert snap.pending_p2_uncapped == PENDING_UNCAPPED_SANITY_MAX


# ============================
# _kill_override_chain_completion_inputs の override 意味論
# ============================

def test_completion_inputs_default_none_is_bit_identical() -> None:
    """override 未指定 (None) は従来と完全に同じ値を返す (backwards compat)。"""
    snap = types.SimpleNamespace(pending_p1=10, pending_p2=20)
    got = vao._kill_override_chain_completion_inputs(
        snap, Board(), Board(), 50, 60, 0.0, None, 0.0, None)
    assert got == (50, 60, 10.0, 20.0)


def test_completion_inputs_override_replaces_pending_base() -> None:
    """override 指定時は snap.pending でなく override が基礎 pending になる。"""
    snap = types.SimpleNamespace(pending_p1=10, pending_p2=20)
    got = vao._kill_override_chain_completion_inputs(
        snap, Board(), Board(), 50, 60, 0.0, None, 0.0, None,
        pending_p1_override=203.0, pending_p2_override=0.0)
    assert got == (50, 60, 203.0, 0.0)


# ============================
# 静的配線回帰 (配線先をソーステキストで固定)
# ============================

def test_generate_signature_has_new_flags_default_off() -> None:
    """generate() の新引数は optional かつ既定 False (既存 API 不変)。"""
    sig = inspect.signature(vao.generate)
    assert sig.parameters["enable_kill_override_hysteresis"].default is False
    assert sig.parameters["enable_kill_override_scale_compare"].default is False


def test_argparse_flags_default_off() -> None:
    """CLI フラグは両方とも既定 OFF (store_true)。"""
    assert re.search(
        r'"--kill-override-hysteresis",\s*action="store_true",\s*default=False',
        VAO_SOURCE)
    assert re.search(
        r'"--kill-override-scale-compare",\s*action="store_true",\s*default=False',
        VAO_SOURCE)


def test_main_forwards_new_flags_to_generate() -> None:
    """[配線漏れ検出] main() が argparse の値を generate() へ転送している。"""
    assert ("enable_kill_override_hysteresis=a.enable_kill_override_hysteresis"
            in VAO_SOURCE)
    assert "a.enable_kill_override_scale_compare" in VAO_SOURCE


def test_loop_wires_gate_after_kill_override() -> None:
    """[配線間違い検出] ゲートは per-frame kill_override の直後・EMA の前段で、
    adv_pre_kill_override と t を受けて呼ばれる (別の値を渡す配線間違いを固定)。"""
    assert "kill_gate.apply(adv_pre_kill_override, adv, t)" in VAO_SOURCE
    # ゲート適用はフラグで囲われている (既定 OFF で bit-identical)
    assert re.search(
        r"if enable_kill_override_hysteresis:\s*\n\s*"
        r"adv = kill_gate\.apply\(adv_pre_kill_override, adv, t\)",
        VAO_SOURCE)


def test_loop_wires_sent_tracker_and_uncapped_base() -> None:
    """[配線間違い検出] scale-compare の kill 入力は「並行帳簿 + 未登録送付分」
    で組み立てられている (表示用 pending_p1/p2 を使う配線間違いを固定)。"""
    assert "unregistered_sent_tracker.update(" in VAO_SOURCE
    assert re.search(
        r"float\(snap\.pending_p1_uncapped\) \+ unregistered_extra_p1",
        VAO_SOURCE)
    assert re.search(
        r"float\(snap\.pending_p2_uncapped\) \+ unregistered_extra_p2",
        VAO_SOURCE)


def test_boundary_reset_recreates_gate_and_sent_tracker() -> None:
    """試合境界リセットブロックで両トラッカーが作り直される (前試合持ち越し
    禁止)。init と reset の2箇所に生成があることを固定する。"""
    assert VAO_SOURCE.count("kill_gate = KillOverrideConfidenceGate()") == 2
    assert VAO_SOURCE.count(
        "unregistered_sent_tracker = PostChainUnregisteredSentTracker()") == 2
