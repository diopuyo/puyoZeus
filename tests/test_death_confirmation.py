"""src/death_confirmation.py のユニットテスト (Gate 3R-6 本体、2026-08-25)。

観測対象:
- 純関数3種 (classify_death_candidate_source / is_own_chain_start /
  has_next_key_changed) の遷移判定。
- DeathConfirmTracker (候補→猶予/解除→確定 (ネクスト不動) の状態機械、
  外部wrapper)。
- 実測で確認済みの2場面をシナリオ再現する:
  - 1P・実試合2・t=164.033-164.733 (false candidate、own chain で解除)
  - 2P・実試合2・t=223 (おじゃま満杯による真の敗北、ネクスト不動で確定すべき)

【設計訂正 2026-08-25】確定条件は当初「次の事象 (次のツモが置けた/さらに
おじゃまが降った)」だったが、死亡すると発火しない逆理のため撤回し、
「own chain なしでネクストが `stationary_confirm_sec` 秒動かない」簡易
検出に差し替えた (user 承認済み、底抜け演出検出による根治は後回し)。
"""
from __future__ import annotations

import pytest

from src.death_confirmation import (
    DEATH_SOURCE_OJAMA,
    DEATH_SOURCE_PLACEMENT,
    GAME_IDX_PRE_MATCH,
    NEXT_STATIONARY_CONFIRM_SEC,
    DeathConfirmStats,
    DeathConfirmTracker,
    classify_death_candidate_source,
    has_next_key_changed,
    is_new_tsumo_fall_start,
    is_own_chain_start,
    is_placement_transition,
    is_pre_match_game_idx,
    is_real_new_game_start,
    resolve_boundary_confirmations,
)

STABLE = "STABLE"
TSUMO_FALL = "TSUMO_FALL"
CHAIN = "CHAIN"
OJAMA_FALL = "OJAMA_FALL"
GRAVITY_SETTLE = "GRAVITY_SETTLE"


# ============================
# classify_death_candidate_source
# ============================


def test_candidate_source_placement() -> None:
    assert classify_death_candidate_source(TSUMO_FALL, STABLE, True) == (
        DEATH_SOURCE_PLACEMENT)


def test_candidate_source_ojama() -> None:
    assert classify_death_candidate_source(OJAMA_FALL, STABLE, True) == (
        DEATH_SOURCE_OJAMA)


def test_candidate_source_none_when_not_occupied() -> None:
    assert classify_death_candidate_source(TSUMO_FALL, STABLE, False) is None
    assert classify_death_candidate_source(OJAMA_FALL, STABLE, False) is None


def test_candidate_source_none_when_not_stable() -> None:
    assert classify_death_candidate_source(STABLE, TSUMO_FALL, True) is None
    assert classify_death_candidate_source(TSUMO_FALL, CHAIN, True) is None


def test_candidate_source_none_from_chain_transition() -> None:
    """CHAIN/GRAVITY_SETTLE → STABLE は候補にしない (スコープ外、既知の限界)。"""
    assert classify_death_candidate_source(CHAIN, STABLE, True) is None
    assert classify_death_candidate_source(GRAVITY_SETTLE, STABLE, True) is None


def test_candidate_source_none_on_stable_hold() -> None:
    """STABLE→STABLE (占有継続、遷移なし) は新規候補にしない。"""
    assert classify_death_candidate_source(STABLE, STABLE, True) is None


# ============================
# is_own_chain_start
# ============================


def test_own_chain_start_true_on_new_entry() -> None:
    assert is_own_chain_start(STABLE, CHAIN) is True


def test_own_chain_start_false_when_already_chain() -> None:
    assert is_own_chain_start(CHAIN, CHAIN) is False


def test_own_chain_start_false_otherwise() -> None:
    assert is_own_chain_start(TSUMO_FALL, STABLE) is False
    assert is_own_chain_start(OJAMA_FALL, STABLE) is False


# ============================
# is_new_tsumo_fall_start (2026-08-25 第3版、Codex 承認条件対応)
# ============================


def test_new_tsumo_fall_start_true_on_entry() -> None:
    assert is_new_tsumo_fall_start(STABLE, TSUMO_FALL) is True
    assert is_new_tsumo_fall_start(CHAIN, TSUMO_FALL) is True


def test_new_tsumo_fall_start_false_when_already_tsumo_fall() -> None:
    assert is_new_tsumo_fall_start(TSUMO_FALL, TSUMO_FALL) is False


def test_new_tsumo_fall_start_false_otherwise() -> None:
    assert is_new_tsumo_fall_start(TSUMO_FALL, STABLE) is False
    assert is_new_tsumo_fall_start(STABLE, STABLE) is False


# ============================
# has_next_key_changed
# ============================


def test_next_key_changed_true_on_different_value() -> None:
    assert has_next_key_changed((1, 2), (3, 4)) is True


def test_next_key_changed_false_on_same_value() -> None:
    assert has_next_key_changed((1, 2), (1, 2)) is False


def test_next_key_changed_handles_none() -> None:
    assert has_next_key_changed(None, None) is False
    assert has_next_key_changed(None, (1, 2)) is True


# ============================
# is_pre_match_game_idx
# ============================


def test_is_pre_match_game_idx_true_for_zero() -> None:
    assert is_pre_match_game_idx(GAME_IDX_PRE_MATCH) is True
    assert is_pre_match_game_idx(0) is True


def test_is_pre_match_game_idx_false_for_real_match() -> None:
    assert is_pre_match_game_idx(1) is False
    assert is_pre_match_game_idx(3) is False


def test_is_pre_match_game_idx_false_when_none() -> None:
    """未配線 (既存呼出元、backwards compat) では常に False。"""
    assert is_pre_match_game_idx(None) is False


# ============================
# is_placement_transition (2026-08-25 user指示: content-based な
# 「まだ試合が始まっていない」判定の主判定に使う純関数)
# ============================


def test_is_placement_transition_true_on_tsumo_fall_to_stable() -> None:
    assert is_placement_transition(TSUMO_FALL, STABLE) is True


def test_is_placement_transition_false_otherwise() -> None:
    assert is_placement_transition(OJAMA_FALL, STABLE) is False  # 設置ではない
    assert is_placement_transition(STABLE, STABLE) is False
    assert is_placement_transition(TSUMO_FALL, CHAIN) is False


def test_is_placement_transition_false_on_first_frame() -> None:
    """初回フレーム (prev_state=None) は常に False。"""
    assert is_placement_transition(None, STABLE) is False


# ============================
# is_real_new_game_start (Codex 指摘3: 「本当の新試合開始」の統一判定)
# ============================


def test_is_real_new_game_start_true_when_placement_and_valid_next() -> None:
    assert is_real_new_game_start(TSUMO_FALL, STABLE, (1, 2)) is True


def test_is_real_new_game_start_false_when_next_is_none() -> None:
    """設置完了しても next が無効 (None) なら「本当の新試合開始」ではない
    (単発の認識誤検出への corroboration、Codex 指摘3)。"""
    assert is_real_new_game_start(TSUMO_FALL, STABLE, None) is False


def test_is_real_new_game_start_false_when_not_placement() -> None:
    assert is_real_new_game_start(STABLE, TSUMO_FALL, (1, 2)) is False
    assert is_real_new_game_start(STABLE, STABLE, (1, 2)) is False


def test_is_real_new_game_start_false_on_first_frame() -> None:
    assert is_real_new_game_start(None, STABLE, (1, 2)) is False


# ============================
# DeathConfirmTracker: 基本状態機械
# ============================


def test_tracker_first_frame_no_transition_judgeable() -> None:
    """初回フレームは直前state不明のため何も起きない。"""
    tr = DeathConfirmTracker()
    event, delay = tr.update(STABLE, False, 0.0, next_key=(1, 2))
    assert event is None and delay is None
    assert tr.resolved_is_dead() is False


def test_tracker_candidate_then_chain_release_does_not_confirm() -> None:
    """実測ケース (1P・実試合2・t=164.033-164.733) の再現。

    おじゃま着弾で死亡セル占有 → 0.634秒後に own chain 発火 → 解除。
    resolved_is_dead() は一度も True にならない (ネクストが動いていても
    own chain が先に来れば確定より優先して解除される)。
    """
    tr = DeathConfirmTracker()
    tr.update(TSUMO_FALL, False, 163.9, next_key=(1, 2))  # 初期化
    event, delay = tr.update(OJAMA_FALL, False, 164.0, next_key=(1, 2))
    assert event is None
    assert tr.resolved_is_dead() is False
    event, delay = tr.update(STABLE, True, 164.033, next_key=(1, 2))  # 着弾、占有
    assert event == "candidate_ojama"
    assert tr.resolved_is_dead() is False  # 猶予中はまだ死亡ではない
    # 猶予中、STABLE を維持 (連鎖の実読が来るまで数フレーム、1.5秒未満)
    event, delay = tr.update(STABLE, True, 164.4, next_key=(1, 2))
    assert event is None
    assert tr.resolved_is_dead() is False
    # own chain 発火 (掛け算式実読) → 解除 (ネクスト不動の1.5秒に達する前)
    event, delay = tr.update(CHAIN, True, 164.667, next_key=(1, 2))
    assert event == "released_ojama"
    assert delay is None
    assert tr.resolved_is_dead() is False  # 解除後も死亡確定にはならない
    event, delay = tr.update(STABLE, False, 165.0, next_key=(3, 4))
    assert tr.resolved_is_dead() is False


def test_tracker_confirms_via_next_stationary() -> None:
    """実測ケース (2P・実試合2・t=223、おじゃま満杯の真の敗北) の再現。

    おじゃま着弾で死亡セル占有 → own chain なし → ネクストが
    stationary_confirm_sec 秒動かない → 死亡確定。
    """
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 220.0, next_key=(1, 2))
    tr.update(OJAMA_FALL, False, 221.0, next_key=(1, 2))
    # 着弾の瞬間 (t=223.0) にネクストが最後に動く (以後は不動)。
    event, delay = tr.update(STABLE, True, 223.0, next_key=(2, 3))  # 着弾、占有
    assert event == "candidate_ojama"
    assert tr.resolved_is_dead() is False
    # ネクスト不動が続くが、まだ閾値未満 (1.5秒未満)。
    event, delay = tr.update(STABLE, True, 224.0, next_key=(2, 3))
    assert event is None
    assert tr.resolved_is_dead() is False
    # 閾値 (既定1.5秒) を超えた瞬間に確定する (223.0 + 1.5 = 224.5)。
    event, delay = tr.update(STABLE, True, 224.6, next_key=(2, 3))
    assert event == "confirmed_ojama"
    assert delay is not None and delay > 0
    assert tr.resolved_is_dead() is True
    assert tr.confirmed_source == DEATH_SOURCE_OJAMA
    # 確定後は sticky (以後 STABLE に戻っても True のまま)
    event, delay = tr.update(STABLE, False, 225.0, next_key=(2, 3))
    assert tr.resolved_is_dead() is True


def test_tracker_next_movement_does_not_confirm_or_reset_candidate() -> None:
    """死亡していない場合、ネクストが動けば確定しない。

    【設計の穴の是正 2026-08-25 第3版、Codex 承認条件対応】候補発生後に
    next が変化するのは強い生存証拠であり、`_observe_pending()` が最初の
    変化を検知した時点で即座に「解除」する (以前は単に確定タイマーを
    リセットするだけだったが、Codex 指摘によりそれでは不十分と判断され、
    明示的な解除に強化された)。よって「動き続ける限り確定しない」という
    結果は同じだが、**最初の next 変化の時点で候補自体が解除される**。
    """
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 0.0, next_key=(1, 2))
    tr.update(OJAMA_FALL, False, 0.5, next_key=(1, 2))
    event, _ = tr.update(STABLE, True, 1.0, next_key=(1, 2))  # candidate_ojama
    assert event == "candidate_ojama"
    # 最初の next 変化 (1秒後) で即座に生存証拠として解除される。
    event, _ = tr.update(STABLE, True, 2.0, next_key=(3, 4))
    assert event == "released_survival_ojama"
    assert tr.resolved_is_dead() is False
    assert tr.has_pending_candidate() is False  # 候補は解除済み (もう猶予中ではない)
    # 以後 next がさらに動いても (解除済みのため) 何も起きない。
    for k in [(5, 1), (2, 3), (4, 5)]:
        event, _ = tr.update(STABLE, True, 3.0, next_key=k)
        assert event is None
    assert tr.resolved_is_dead() is False


def test_tracker_stationary_timer_baseline_is_candidate_time_not_pre_history() -> None:
    """【Codex 指摘1・修正確認】不動時間の起点は「候補発生時刻」と「最後に
    有効な next が変化した時刻」の遅い方 (max)。

    当初実装は `t_sec - last_next_change_sec` のみを見ており、候補発生
    より**前から**ネクストが不動だった場合、候補発生の直後 (実測
    0.067秒後) で確定してしまう不具合があった (Codex 独立レビュー NG)。
    本テストは同じ入力パターンで「候補後 0.6 秒では確定しない」ことを
    確認する (回帰防止)。
    """
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 0.0, next_key=(1, 2))
    tr.update(OJAMA_FALL, False, 0.5, next_key=(1, 2))  # ここから next_key 不変
    event, _ = tr.update(STABLE, True, 1.0, next_key=(1, 2))  # candidate_ojama
    assert event == "candidate_ojama"
    # 候補発生 (t=1.0) から 0.6 秒後 (次候補後1.5秒未満) では確定しない
    # (旧実装は next_key が t=0.0 から不変なため誤って確定していた)。
    event, delay = tr.update(STABLE, True, 1.6, next_key=(1, 2))
    assert event is None
    assert tr.resolved_is_dead() is False
    # 候補発生から 1.5 秒経過 (t=2.5) で確定する。
    event, delay = tr.update(STABLE, True, 2.5, next_key=(1, 2))
    assert event == "confirmed_ojama"
    assert delay is not None
    assert delay == pytest.approx(1.5)


def test_tracker_placement_candidate_not_confused_with_ojama() -> None:
    """設置由来とおじゃま由来の発生源が取り違えられないこと。"""
    tr_place = DeathConfirmTracker()
    tr_place.update(STABLE, False, 0.0, next_key=(1, 2))
    tr_place.update(TSUMO_FALL, False, 0.1, next_key=(1, 2))
    tr_place.update(STABLE, True, 0.2, next_key=(1, 2))
    assert tr_place.confirmed_source is None  # まだ確定していない (猶予中)

    tr_ojama = DeathConfirmTracker()
    tr_ojama.update(STABLE, False, 0.0, next_key=(1, 2))
    tr_ojama.update(OJAMA_FALL, False, 0.1, next_key=(1, 2))
    tr_ojama.update(STABLE, True, 0.2, next_key=(1, 2))
    # どちらも猶予中で resolved_is_dead は False、発生源だけ内部的に異なる
    assert tr_place.resolved_is_dead() is False
    assert tr_ojama.resolved_is_dead() is False


def test_tracker_custom_stationary_confirm_sec() -> None:
    """stationary_confirm_sec を明示指定すると既定値 (1.5秒) を上書きできる
    (CLI --death-next-stationary-sec 経由の感度測定用)。"""
    tr = DeathConfirmTracker(stationary_confirm_sec=0.5)
    tr.update(STABLE, False, 0.0, next_key=(1, 2))
    tr.update(OJAMA_FALL, False, 0.1, next_key=(2, 3))  # 着弾直前まで動いていた
    tr.update(STABLE, True, 0.2, next_key=(2, 3))  # candidate_ojama
    event, _ = tr.update(STABLE, True, 0.5, next_key=(2, 3))  # 0.3秒 (未到達)
    assert event is None
    event, _ = tr.update(STABLE, True, 0.71, next_key=(2, 3))  # 0.51秒
    assert event == "confirmed_ojama"


def test_default_stationary_confirm_sec_is_1_5() -> None:
    """既定値が user 指定の暫定値 1.5 秒であること (回帰防止)。"""
    assert NEXT_STATIONARY_CONFIRM_SEC == 1.5


# ============================
# is_match_active ガード (2026-08-25 実測で発見・根治): まちうけ画面誤確定
# ============================


def test_tracker_ignores_candidate_while_match_inactive() -> None:
    """実測で発見した不具合の再現+根治確認 (2P、t=18.067-90.5、72.4秒)。

    まちうけ画面 (試合外) は背景誤検出で死亡セルが占有され、かつネクスト
    表示が固定/無いため、is_match_active ガードが無いと誤って即確定して
    しまう。is_match_active=False の間は候補判定自体を止める。
    """
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 0.0, next_key=None, is_match_active=False)
    # まちうけ画面中に背景誤検出で「占有」に見える遷移が起きても無視。
    event, _ = tr.update(
        OJAMA_FALL, True, 1.0, next_key=None, is_match_active=False)
    assert event is None
    event, _ = tr.update(
        STABLE, True, 1.1, next_key=None, is_match_active=False)
    assert event is None
    # ネクストが不動 (None のまま) でも、試合外の間は確定タイマーが
    # 進まない (何十秒経っても確定しない)。
    event, _ = tr.update(
        STABLE, True, 90.0, next_key=None, is_match_active=False)
    assert event is None
    assert tr.resolved_is_dead() is False


def test_tracker_resumes_clean_after_match_becomes_active() -> None:
    """試合が始まると is_match_active=True 側の通常フローに戻る
    (試合外区間の遷移が試合開始後に誤って「遷移」と解釈されない)。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, True, 0.0, next_key=None, is_match_active=False)
    tr.update(OJAMA_FALL, True, 50.0, next_key=None, is_match_active=False)
    # 試合開始 (is_match_active=True) の最初のフレームは初回同様、
    # 遷移判定不能 (prev_state がリセットされているため)。
    event, _ = tr.update(STABLE, False, 86.9, next_key=(1, 2))
    assert event is None
    # 以後は通常フローで候補が判定できる。
    tr.update(OJAMA_FALL, False, 87.0, next_key=(1, 2))
    event, _ = tr.update(STABLE, True, 87.1, next_key=(1, 2))
    assert event == "candidate_ojama"


def test_tracker_preserves_confirmed_through_match_inactive_period() -> None:
    """試合中に確定済みなら、その後試合外になっても即座には消えない
    (on_game_boundary と同じ「消去待ち」設計。次の本当の試合開始の
    TSUMO_FALL まで保持される)。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 220.0, next_key=(1, 2))
    tr.update(OJAMA_FALL, False, 221.0, next_key=(1, 2))
    tr.update(STABLE, True, 223.0, next_key=(2, 3))  # candidate_ojama
    event, _ = tr.update(STABLE, True, 224.6, next_key=(2, 3))  # confirmed_ojama
    assert event == "confirmed_ojama"
    assert tr.resolved_is_dead() is True

    # 試合外 (決着後の画面) に入っても即座には消えない。
    event, _ = tr.update(STABLE, True, 230.0, next_key=None, is_match_active=False)
    assert event is None
    assert tr.resolved_is_dead() is True


# ============================
# game_idx ガード (2026-08-25 実測で確定): まちうけ画面誤確定の代替対策
# ============================


def test_tracker_ignores_candidate_while_game_idx_is_pre_match() -> None:
    """`is_match_active` が無効だった場合の代替: `game_idx==0` (まちうけ画面
    相当) の間は `is_match_active=True` でも候補判定自体を止める
    (実測: zenchi 動画のまちうけ区間1019行のうち1018行が game_idx==0)。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 0.0, next_key=None, game_idx=0)
    event, _ = tr.update(OJAMA_FALL, True, 1.0, next_key=None, game_idx=0)
    assert event is None
    event, _ = tr.update(STABLE, True, 1.1, next_key=None, game_idx=0)
    assert event is None
    # ネクスト不動 (None のまま) でも、game_idx==0 の間は確定タイマーが
    # 進まない (何十秒経っても確定しない)。
    event, _ = tr.update(STABLE, True, 90.0, next_key=None, game_idx=0)
    assert event is None
    assert tr.resolved_is_dead() is False


def test_tracker_resumes_clean_after_game_idx_leaves_pre_match() -> None:
    """`game_idx` が 1 以上 (実試合) になると通常フローに戻る。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, True, 0.0, next_key=None, game_idx=0)
    tr.update(OJAMA_FALL, True, 50.0, next_key=None, game_idx=0)
    # game_idx==0 を抜けた最初のフレームは初回同様、遷移判定不能。
    event, _ = tr.update(STABLE, False, 86.9, next_key=(1, 2), game_idx=1)
    assert event is None
    tr.update(OJAMA_FALL, False, 87.0, next_key=(1, 2), game_idx=1)
    event, _ = tr.update(STABLE, True, 87.1, next_key=(1, 2), game_idx=1)
    assert event == "candidate_ojama"


def test_tracker_game_idx_none_is_backward_compatible() -> None:
    """`game_idx` 無指定 (既存呼出元) では従来動作と完全一致する
    (backwards compat、`game_idx==0` ガードが誤って発火しないこと)。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 220.0, next_key=(1, 2))
    tr.update(OJAMA_FALL, False, 221.0, next_key=(1, 2))
    event, _ = tr.update(STABLE, True, 223.0, next_key=(2, 3))
    assert event == "candidate_ojama"
    event, _ = tr.update(STABLE, True, 224.6, next_key=(2, 3))
    assert event == "confirmed_ojama"
    assert tr.resolved_is_dead() is True


def test_tracker_game_idx_guard_combines_with_is_match_active() -> None:
    """`is_match_active=True` かつ `game_idx==0` でも凍結される
    (2つのガードは論理和、どちらかが「試合外」と言えば凍結)。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 0.0, next_key=(1, 2),
              is_match_active=True, game_idx=0)
    event, _ = tr.update(OJAMA_FALL, True, 1.0, next_key=(1, 2),
                          is_match_active=True, game_idx=0)
    assert event is None
    event, _ = tr.update(STABLE, True, 1.1, next_key=(1, 2),
                          is_match_active=True, game_idx=0)
    assert event is None
    assert tr.resolved_is_dead() is False


def test_tracker_preserves_confirmed_through_pre_match_game_idx() -> None:
    """試合中に確定済みなら、その後 `game_idx==0` (=決着後にまちうけ画面
    相当の区間へ戻る想定) になっても即座には消えない (on_game_boundary
    と同じ「消去待ち」設計、根治を壊さないことの確認)。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 220.0, next_key=(1, 2), game_idx=2)
    tr.update(OJAMA_FALL, False, 221.0, next_key=(1, 2), game_idx=2)
    tr.update(STABLE, True, 223.0, next_key=(2, 3), game_idx=2)
    event, _ = tr.update(STABLE, True, 224.6, next_key=(2, 3), game_idx=2)
    assert event == "confirmed_ojama"
    assert tr.resolved_is_dead() is True

    event, _ = tr.update(STABLE, True, 230.0, next_key=None, game_idx=0)
    assert event is None
    assert tr.resolved_is_dead() is True


def test_tracker_game_idx_change_to_nonzero_does_not_auto_clear_confirmed() -> None:
    """`game_idx` が 2→3 (別の実試合) へ変わっただけでは `update()` は
    確定フラグを自動的に消さない (設計判断: 消去は `on_game_boundary()` +
    次の本当の TSUMO_FALL という物理事象にのみ委ねる。`update()` 内で
    `game_idx` 変化そのものを消去トリガーにはしない。理由: 実測
    (t=223→232.467) で `game_idx` の切り替わりは決着演出中の settled
    recompute 空白区間の内側で起きており、それを消去トリガーにすると
    空白区間の内側で確定フラグが消えて見逃しに戻ってしまう恐れがある
    ため。呼出元 (`visualize_advantage_overlay.py`) は既に `game_idx`
    を進めるのと同じ `_detect_score_reset` 検知箇所で
    `on_game_boundary()` を呼んでおり、消去は従来どおりそちら経由)。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 220.0, next_key=(1, 2), game_idx=2)
    tr.update(OJAMA_FALL, False, 221.0, next_key=(1, 2), game_idx=2)
    tr.update(STABLE, True, 223.0, next_key=(2, 3), game_idx=2)
    event, _ = tr.update(STABLE, True, 224.6, next_key=(2, 3), game_idx=2)
    assert event == "confirmed_ojama"

    # game_idx が 3 (別の実試合) に変わっただけ (on_game_boundary は
    # 呼出元がここでは呼んでいない想定)。update() 単体では消えない。
    event, _ = tr.update(STABLE, True, 232.5, next_key=(2, 3), game_idx=3)
    assert event is None
    assert tr.resolved_is_dead() is True  # まだ消えない (on_game_boundary待ち)


# ============================
# _has_ever_placed 主判定 (2026-08-25 user指示で再設計): content-based な
# 「まだ試合が始まっていない」判定。game_idx==0/is_match_active=False は
# 「凍結を検討する入口」を開くだけの補助信号に格下げし、主判定は
# 「その side でまだ一度も設置を観測していないか」に置き換えた。
# ============================


def test_tracker_frozen_while_never_placed_even_without_game_idx() -> None:
    """まちうけ相当 (設置を一度も観測していない) では、`game_idx` を渡さず
    `is_match_active=False` だけでも従来通り凍結される (has_ever_placed
    が主判定に回っても、まちうけ画面での凍結そのものは崩れないこと)。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 0.0, next_key=None, is_match_active=False)
    event, _ = tr.update(
        OJAMA_FALL, True, 1.0, next_key=None, is_match_active=False)
    assert event is None
    event, _ = tr.update(
        STABLE, True, 90.0, next_key=None, is_match_active=False)
    assert event is None
    assert tr.resolved_is_dead() is False


def test_tracker_mid_match_start_unfreezes_after_first_real_placement() -> None:
    """【本丸】「まちうけ画面を含まず試合の途中から始まる動画」相当の再現。

    `game_idx` は最初の試合の間ずっと `0` のまま (スコアリセットが一度も
    検知されないため)。旧設計 (`game_idx==0` を主判定) では、この動画の
    最初の試合全体で死亡確定が凍結され続け、本当の敗北を丸ごと見逃す
    (user 指摘の致命的欠陥)。新設計では、動画開始後最初の設置
    (`TSUMO_FALL→STABLE`) が観測された瞬間に凍結が解け、以後は
    `game_idx==0` のままでも通常フローで候補・確定が動く。
    """
    tr = DeathConfirmTracker()
    # 動画が試合の途中 (既にプレイ中) から始まる。game_idx はまだ 0。
    tr.update(STABLE, False, 0.0, next_key=(1, 2), game_idx=0)
    # まだ設置を一度も観測していない = 凍結中。死亡候補が立っても無視。
    event, _ = tr.update(OJAMA_FALL, True, 0.5, next_key=(1, 2), game_idx=0)
    assert event is None
    event, _ = tr.update(STABLE, True, 0.6, next_key=(1, 2), game_idx=0)
    assert event is None  # まだ has_ever_placed=False のため凍結中
    assert tr.resolved_is_dead() is False
    # 生きて対局が続き、最初の本当の設置 (TSUMO_FALL→STABLE) が起きる
    # (死亡セルは空、通常の1手)。この瞬間に凍結が解ける。
    tr.update(TSUMO_FALL, False, 1.0, next_key=(1, 2), game_idx=0)
    event, _ = tr.update(STABLE, False, 1.1, next_key=(1, 2), game_idx=0)
    assert event is None  # 死亡セル非占有、候補ではないが凍結も解けている
    # 以後 (game_idx は依然 0 のまま!) は通常フローで候補が判定できる。
    tr.update(OJAMA_FALL, False, 5.0, next_key=(3, 4), game_idx=0)
    event, _ = tr.update(STABLE, True, 5.1, next_key=(3, 4), game_idx=0)
    assert event == "candidate_ojama"  # 凍結解除後は正常に候補が立つ
    # ネクスト不動が続けば確定にも到達する (game_idx==0 のままでも!)。
    event, delay = tr.update(STABLE, True, 6.7, next_key=(3, 4), game_idx=0)
    assert event == "confirmed_ojama"
    assert delay is not None
    assert tr.resolved_is_dead() is True


def test_tracker_has_ever_placed_persists_across_game_boundary() -> None:
    """`_has_ever_placed` は `on_game_boundary()` でもリセットされない
    (user 指示: 動画を通して一度でも設置を見たら以後は凍結しない、が
    安全側)。誤って `game_idx==0` 相当の値が来ても、`_has_ever_placed`
    の判定自体は依然として「凍結しない」側に倒れることを確認する
    (`is_pre_match_game_idx` ガード単体は無効化される)。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 0.0, next_key=(1, 2), game_idx=1)
    tr.update(TSUMO_FALL, False, 1.0, next_key=(1, 2), game_idx=1)
    tr.update(STABLE, False, 1.1, next_key=(1, 2), game_idx=1)  # 設置観測済み
    tr.on_game_boundary()
    # 【Codex 指摘3】境界後は _has_ever_placed が True でも
    # _post_boundary_armed=False で再アームまで凍結される
    # (2試合目以降の背景誤認を防ぐための独立ゲート)。
    tr.update(OJAMA_FALL, False, 100.0, next_key=(5, 6), game_idx=0)
    event, _ = tr.update(STABLE, True, 100.1, next_key=(5, 6), game_idx=0)
    assert event is None  # まだ再アームされていない (境界後の凍結が効く)
    # 検証済みの実ゲーム開始 (設置完了+有効next) で再アームされる。
    tr.update(TSUMO_FALL, False, 105.0, next_key=(7, 8), game_idx=1)
    event, _ = tr.update(STABLE, False, 105.1, next_key=(7, 8), game_idx=1)
    assert event is None  # 死亡セル非占有、候補ではないが再アームは完了
    # 再アーム後は通常フローに戻る (_has_ever_placed 由来の凍結には
    # そもそも引っかからない、`is_pre_match_game_idx` ガード単体は無効化)。
    tr.update(OJAMA_FALL, False, 110.0, next_key=(9, 1), game_idx=1)
    event, _ = tr.update(STABLE, True, 110.1, next_key=(9, 1), game_idx=1)
    assert event == "candidate_ojama"


def test_tracker_never_placed_has_ever_placed_default_backward_compat() -> None:
    """`game_idx`/`is_match_active` を一切使わない既存呼出元では、
    `_has_ever_placed` が False のままでも凍結は発生しない (=入口の
    `pre_match_entry` 自体が常に False、backwards compat)。"""
    tr = DeathConfirmTracker()
    tr.update(TSUMO_FALL, False, 0.0, next_key=(1, 2))
    event, _ = tr.update(OJAMA_FALL, False, 0.5, next_key=(1, 2))
    assert event is None
    event, _ = tr.update(STABLE, True, 0.6, next_key=(1, 2))
    assert event == "candidate_ojama"  # 凍結されず従来通り即座に候補が立つ


# ============================
# on_game_boundary (2026-08-25 実測で根治): 確定フラグの消去タイミング
# ============================


def test_on_game_boundary_preserves_confirmed_until_real_new_game() -> None:
    """実測で発見した不具合の再現+根治確認 (2P・実試合2・t=223)。

    決着演出〜結果表示中に試合境界検知 (score-reset) が飛んでも、確定済み
    フラグは即座には消えない (旧実装は新規インスタンス差し替えで即座に
    消えて dump に一度も True が現れなかった)。次の
    `is_real_new_game_start()` (設置完了+有効next、Codex 指摘3対応で
    単なる `curr_state==TSUMO_FALL` から強化済み) を観測して初めて消える。
    """
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 220.0, next_key=(1, 2))
    tr.update(OJAMA_FALL, False, 221.0, next_key=(1, 2))
    tr.update(STABLE, True, 223.0, next_key=(1, 2))  # candidate_ojama
    event, _ = tr.update(STABLE, True, 224.5, next_key=(1, 2))  # confirmed_ojama (候補後1.5秒)
    assert event == "confirmed_ojama"
    assert tr.resolved_is_dead() is True

    # 決着演出中、score-reset に基づく試合境界検知が複数回飛んでくる
    # (旧実装はここで確定フラグが消えていた)。
    source, outcome = tr.on_game_boundary()
    assert source is None and outcome == "no_candidate"  # 既に確定済みで猶予中の候補は無い
    assert tr.resolved_is_dead() is True  # まだ消えない (根治の核心)
    tr.on_game_boundary()  # 複数回呼ばれても安全 (idempotent)
    assert tr.resolved_is_dead() is True

    # 結果表示が続く間、STABLE のままでは消えない。
    event, _ = tr.update(STABLE, False, 227.0, next_key=(1, 2))
    assert event is None
    assert tr.resolved_is_dead() is True

    # TSUMO_FALL に入っただけ (設置完了前) ではまだ消えない
    # (Codex 指摘3: 単なる curr_state==TSUMO_FALL は corroboration 不足)。
    event, _ = tr.update(TSUMO_FALL, False, 232.0, next_key=(1, 2))
    assert event is None
    assert tr.resolved_is_dead() is True

    # 設置完了 (TSUMO_FALL→STABLE) + 有効 next で「本当の新試合開始」が
    # 検証され、初めて確定フラグが消える。
    event, _ = tr.update(STABLE, False, 232.5, next_key=(1, 2))
    assert event is None
    assert tr.resolved_is_dead() is False
    assert tr.confirmed_source is None


def test_on_game_boundary_confirms_pending_candidate_as_death() -> None:
    """【設計の穴の是正 2026-08-25 第2版・第3版】試合境界で猶予中
    (未確定・未解除) の候補は「閾値未到達で消滅」ではなく**その場で死亡
    確定する** (Codex 承認条件をすべて満たす場合のみ: 同一 game_idx、
    生存証拠なし、ambiguous でない)。人が本当に詰んだとき試合はその場で
    終わるため、ネクスト不動猶予の条件は真の死亡では原理的に満たせない
    (実測: 2P・実試合2・t=223 の真の窒息が候補後0.37秒で試合終了、
    旧設計では検出できなかった)。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 0.0, next_key=(1, 2), game_idx=3)
    tr.update(OJAMA_FALL, False, 0.5, next_key=(1, 2), game_idx=3)
    event, _ = tr.update(STABLE, True, 1.0, next_key=(1, 2), game_idx=3)  # candidate_ojama
    assert event == "candidate_ojama"
    assert tr.resolved_is_dead() is False

    # 1.5秒 (stationary_confirm_sec) に届く前 (0.37秒後) に試合が終了。
    # 境界検知の game_idx は候補発生時と同じ 3 (同一試合の終わり)。
    source, outcome = tr.on_game_boundary(game_idx=3)
    assert source == DEATH_SOURCE_OJAMA
    assert outcome == "confirmed"
    assert tr.resolved_is_dead() is True  # 境界でその場で死亡確定
    assert tr.confirmed_source == DEATH_SOURCE_OJAMA

    # 確定は sticky (次の本当の新試合開始まで保持、旧根治を壊さない)。
    event, _ = tr.update(STABLE, True, 10.0, next_key=(1, 2))
    assert event is None
    assert tr.resolved_is_dead() is True
    # 別試合の CHAIN 遷移が前試合の猶予解除として誤って処理されない
    # (境界後は候補受付自体が凍結されている)。
    event, _ = tr.update(CHAIN, True, 10.5, next_key=(1, 2))
    assert event is None
    assert tr.resolved_is_dead() is True


def test_on_game_boundary_does_not_confirm_released_candidate() -> None:
    """own chain で既に解除された候補は、境界が来ても死亡確定しない
    (user 伝授「連鎖中は窒息としない」を壊さない、要件2)。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 0.0, next_key=(1, 2))
    tr.update(OJAMA_FALL, False, 0.5, next_key=(1, 2))
    event, _ = tr.update(STABLE, True, 1.0, next_key=(1, 2))  # candidate_ojama
    assert event == "candidate_ojama"
    event, _ = tr.update(CHAIN, True, 1.2, next_key=(1, 2))  # own chain で解除
    assert event == "released_ojama"
    assert tr.resolved_is_dead() is False

    source, outcome = tr.on_game_boundary()
    assert source is None and outcome == "no_candidate"  # 解除済みのため候補は既に無い
    assert tr.resolved_is_dead() is False


def test_on_game_boundary_frozen_pre_match_never_false_confirms() -> None:
    """まちうけ画面 (=試合外、凍結中) では候補自体が発生しないため、境界
    検知が来ても死亡確定しない (まちうけ偽陽性の再発防止、要件3、Codex
    新規回帰#6)。"""
    tr = DeathConfirmTracker()
    # 一度も設置を観測していない = 凍結中。死亡セルが占有されて見えても
    # 候補にならない (背景誤検出のシミュレーション)。
    tr.update(STABLE, False, 0.0, next_key=(1, 2), game_idx=0)
    event, _ = tr.update(OJAMA_FALL, True, 0.5, next_key=None, game_idx=0)
    assert event is None
    event, _ = tr.update(STABLE, True, 0.6, next_key=None, game_idx=0)
    assert event is None
    assert tr.resolved_is_dead() is False

    source, outcome = tr.on_game_boundary(game_idx=0)
    assert source is None and outcome == "no_candidate"  # 凍結中は候補が絶対に立たない
    assert tr.resolved_is_dead() is False


def test_on_game_boundary_rejects_different_game_idx() -> None:
    """【Codex 新規回帰#7】候補発生時と境界の game_idx が異なる場合は
    確定しない (別ゲームへの越境防止、防御的チェック)。通常の呼出し
    経路では凍結ガードにより到達しない想定だが、ガード自体の単体動作を
    直接検証する。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 0.0, next_key=(1, 2), game_idx=1)
    tr.update(OJAMA_FALL, False, 0.5, next_key=(1, 2), game_idx=1)
    event, _ = tr.update(STABLE, True, 1.0, next_key=(1, 2), game_idx=1)
    assert event == "candidate_ojama"  # _pending_game_idx=1 として記録される

    source, outcome = tr.on_game_boundary(game_idx=2)  # 異なる game_idx
    assert source == DEATH_SOURCE_OJAMA
    assert outcome == "rejected_game_idx_mismatch"
    assert tr.resolved_is_dead() is False  # 確定しない


def test_on_game_boundary_ambiguous_when_both_sides_pending() -> None:
    """【Codex 新規回帰#8】両側同時に猶予中の候補が残っている場合、
    勝敗側を一意に決められないため両方とも確定しない
    (`resolve_boundary_confirmations()`)。"""
    tr1, tr2 = DeathConfirmTracker(), DeathConfirmTracker()
    for tr in (tr1, tr2):
        tr.update(STABLE, False, 0.0, next_key=(1, 2), game_idx=1)
        tr.update(OJAMA_FALL, False, 0.5, next_key=(1, 2), game_idx=1)
        event, _ = tr.update(STABLE, True, 1.0, next_key=(1, 2), game_idx=1)
        assert event == "candidate_ojama"

    stats = DeathConfirmStats()
    (s1, o1), (s2, o2) = resolve_boundary_confirmations(tr1, tr2, game_idx=1, stats=stats)
    assert o1 == "suppressed_ambiguous"
    assert o2 == "suppressed_ambiguous"
    assert tr1.resolved_is_dead() is False
    assert tr2.resolved_is_dead() is False
    assert stats.ambiguous_both_pending == 1
    assert stats.boundary_confirmed == 0
    assert stats.pending_at_boundary == 2
    assert stats.total_boundaries == 1


def test_resolve_boundary_confirmations_confirms_single_side() -> None:
    """片側のみ猶予中の候補が残っている場合は ambiguous にならず通常どおり
    その場で死亡確定する (両側チェックが誤って正常系まで抑制しないこと)。"""
    tr1, tr2 = DeathConfirmTracker(), DeathConfirmTracker()
    tr1.update(STABLE, False, 0.0, next_key=(1, 2), game_idx=1)
    tr1.update(OJAMA_FALL, False, 0.5, next_key=(1, 2), game_idx=1)
    event, _ = tr1.update(STABLE, True, 1.0, next_key=(1, 2), game_idx=1)
    assert event == "candidate_ojama"
    tr2.update(STABLE, False, 0.0, next_key=(9, 9), game_idx=1)  # 2P は無関係

    stats = DeathConfirmStats()
    (s1, o1), (s2, o2) = resolve_boundary_confirmations(tr1, tr2, game_idx=1, stats=stats)
    assert o1 == "confirmed"
    assert o2 == "no_candidate"
    assert tr1.resolved_is_dead() is True
    assert tr2.resolved_is_dead() is False
    assert stats.ambiguous_both_pending == 0
    assert stats.boundary_confirmed == 1
    assert stats.pending_at_boundary == 1


def test_on_game_boundary_noop_when_not_confirmed() -> None:
    """未確定 (候補なし) の状態で on_game_boundary を呼んでも何も壊れない。"""
    tr = DeathConfirmTracker()
    assert tr.on_game_boundary() == (None, "no_candidate")
    tr.update(STABLE, False, 0.0, next_key=(1, 2))
    assert tr.on_game_boundary() == (None, "no_candidate")
    event, _ = tr.update(TSUMO_FALL, False, 1.0, next_key=(1, 2))
    assert event is None
    assert tr.resolved_is_dead() is False


# ============================
# DeathConfirmStats
# ============================


def test_stats_record_and_summary_denominator() -> None:
    stats = DeathConfirmStats()
    stats.record(None)
    stats.record("candidate_placement")
    stats.record(None)
    stats.record("released_placement")
    stats.record("candidate_ojama")
    stats.record("confirmed_ojama", delay_sec=0.634)
    assert stats.frames_total == 6
    assert stats.candidate_placement == 1
    assert stats.candidate_ojama == 1
    assert stats.released_placement == 1
    assert stats.confirmed_ojama == 1
    assert stats.confirm_delays_sec == [0.634]
    text = stats.summary()
    assert "候補 2/6行" in text
    assert "確定 1/2" in text


def test_stats_record_expired_at_boundary() -> None:
    """旧メソッド (2026-08-25 第3版で「別ゲームへの越境拒否」の記録先に
    再利用) の構造互換確認。"""
    stats = DeathConfirmStats()
    stats.record("candidate_ojama")
    stats.record_expired_at_boundary(DEATH_SOURCE_OJAMA)
    assert stats.expired_at_boundary_ojama == 1
    assert stats.expired_at_boundary_placement == 0
    text = stats.summary()
    assert "境界で越境拒否 1/1" in text


def test_stats_record_confirmed_at_boundary() -> None:
    """【設計の穴の是正 2026-08-25 第2版】境界で死亡確定した件数を
    「境界で確定」区分として、既存の「境界で越境拒否」と区別して
    母数付きで記録できる (どちらに分類されたか分かる、要件)。"""
    stats = DeathConfirmStats()
    stats.record("candidate_ojama")
    stats.record_confirmed_at_boundary(DEATH_SOURCE_OJAMA)
    assert stats.confirmed_at_boundary_ojama == 1
    assert stats.confirmed_at_boundary_placement == 0
    assert stats.expired_at_boundary_ojama == 0  # 別区分として0のまま
    text = stats.summary()
    assert "境界で確定 1/1" in text
    assert "境界で越境拒否 0/1" in text


def test_stats_record_boundary_outcome_dispatches_all_branches() -> None:
    """【設計の穴の是正 2026-08-25 第3版、Codex 承認条件対応】
    `record_boundary_outcome()` が outcome の種類ごとに正しいカウンタへ
    振り分けることを確認する (母数付き)。"""
    stats = DeathConfirmStats()
    stats.record_boundary_check()
    stats.record_boundary_outcome(None, "no_candidate")
    stats.record_boundary_outcome(DEATH_SOURCE_OJAMA, "confirmed")
    stats.record_boundary_outcome(DEATH_SOURCE_PLACEMENT, "rejected_survival_evidence")
    stats.record_boundary_outcome(DEATH_SOURCE_OJAMA, "rejected_game_idx_mismatch")
    assert stats.total_boundaries == 1
    assert stats.pending_at_boundary == 3  # no_candidate はカウントしない
    assert stats.boundary_confirmed == 1
    assert stats.boundary_rejected_survival_evidence == 1
    assert stats.confirmed_at_boundary_ojama == 1
    assert stats.expired_at_boundary_ojama == 1  # 越境拒否の記録先として再利用


def test_stats_unknown_event_raises() -> None:
    """未知の event 文字列は fail-silent にせず AttributeError で気付ける。"""
    stats = DeathConfirmStats()
    try:
        stats.record("no_such_event")
    except AttributeError:
        pass
    else:
        raise AssertionError("未知イベントで例外が発生しなかった")


def test_stats_summary_zero_denominator_no_crash() -> None:
    """候補が一度も発生しない (0/0)場合でも summary() が例外を出さない。"""
    stats = DeathConfirmStats()
    stats.record(None)
    stats.record(None)
    text = stats.summary()
    assert "候補 0/2行" in text
    assert "確定遅延 n/a" in text


# ============================
# Codex 独立レビュー NG (2026-08-25) 対応: 必須回帰テスト8本
# (指摘1・2・3・5 の修正確認、user 指定の8項目をそのまま実装する)
# ============================


def test_regression1_pre_candidate_stationary_does_not_shortcut_grace() -> None:
    """【回帰1】候補発生前から next が5秒不動でも、候補後1.5秒未満では
    未確定 (Codex 指摘1、修正前は候補後0.033秒で確定していた)。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 0.0, next_key=(1, 2))
    # next が候補発生の遥か前 (5.5秒前) から不動。
    tr.update(OJAMA_FALL, False, 5.5, next_key=(1, 2))
    event, _ = tr.update(STABLE, True, 6.0, next_key=(1, 2))  # candidate_ojama
    assert event == "candidate_ojama"
    # 候補後 0.9 秒 (1.5秒未満) では確定しない。
    event, _ = tr.update(STABLE, True, 6.9, next_key=(1, 2))
    assert event is None
    assert tr.resolved_is_dead() is False
    # 候補後 1.49 秒でもまだ確定しない (境界値確認)。
    event, _ = tr.update(STABLE, True, 7.49, next_key=(1, 2))
    assert event is None
    assert tr.resolved_is_dead() is False


def test_regression2_confirms_after_1_5_sec_from_candidate() -> None:
    """【回帰2】候補後 1.5秒経過で確定する。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 0.0, next_key=(1, 2))
    tr.update(OJAMA_FALL, False, 5.5, next_key=(1, 2))
    event, _ = tr.update(STABLE, True, 6.0, next_key=(1, 2))  # candidate_ojama
    assert event == "candidate_ojama"
    event, delay = tr.update(STABLE, True, 7.5, next_key=(1, 2))  # 候補後1.5秒
    assert event == "confirmed_ojama"
    assert delay is not None and delay == pytest.approx(1.5)
    assert tr.resolved_is_dead() is True


def test_regression3_own_chain_start_releases_during_grace() -> None:
    """【回帰3】猶予中の own chain 開始で解除される (確定しない)。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 0.0, next_key=(1, 2))
    tr.update(OJAMA_FALL, False, 0.5, next_key=(1, 2))
    event, _ = tr.update(STABLE, True, 1.0, next_key=(1, 2))  # candidate_ojama
    assert event == "candidate_ojama"
    event, _ = tr.update(CHAIN, True, 1.2, next_key=(1, 2))  # own chain 開始 (1.5秒未満)
    assert event == "released_ojama"
    assert tr.resolved_is_dead() is False
    # 解除後、ネクストが不動のまま長時間経っても確定しない (候補が消えている)。
    event, _ = tr.update(STABLE, False, 10.0, next_key=(1, 2))
    assert event is None
    assert tr.resolved_is_dead() is False


def test_regression4_next_none_persists_prevents_confirmation() -> None:
    """【回帰4】`next_pair=None` が続く間は未確定 (Codex 指摘2)。
    None は「未検知」であり「動いていない証拠」ではない。

    【2026-08-25 第3版 Codex 承認条件対応で末尾を分離】以前はこのテストの
    末尾で「None の後に異なる next が戻ってきても即座には確定しない
    (測り直し)」まで検証していたが、Codex の新要件「候補後に next 変化
    があれば生存証拠として解除する」により、**異なる next が戻れば
    その場で解除される**のが正しい挙動になった (`test_
    regression3b_next_change_after_none_gap_releases_as_survival`
    参照)。本テストは Codex 指摘2 の核心 (None 単独では確定しない) の
    みを検証する対象に絞る。
    """
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 0.0, next_key=(1, 2))
    tr.update(OJAMA_FALL, False, 0.5, next_key=(1, 2))
    event, _ = tr.update(STABLE, True, 1.0, next_key=(1, 2))  # candidate_ojama
    assert event == "candidate_ojama"
    # next が None のまま何十秒経っても確定しない。
    for t in (2.0, 5.0, 10.0, 30.0):
        event, _ = tr.update(STABLE, True, t, next_key=None)
        assert event is None
    assert tr.resolved_is_dead() is False
    # 有効な next が「候補発生時と同じ値」で戻ってきた場合は生存証拠には
    # ならず (変化していないため)、そこから測り直しになる (1.5秒待つ)。
    event, _ = tr.update(STABLE, True, 30.1, next_key=(1, 2))
    assert event is None
    event, _ = tr.update(STABLE, True, 31.6, next_key=(1, 2))  # 30.1+1.5
    assert event == "confirmed_ojama"


def test_regression3b_next_change_after_none_gap_releases_as_survival() -> None:
    """【2026-08-25 第3版、Codex 承認条件対応】`next_pair=None` が続いた
    後、候補発生時と**異なる** next が観測された場合は生存証拠として
    即座に解除される (regression4 と対になるテスト、次の TSUMO 開始等の
    生存証拠と同じ扱い)。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 0.0, next_key=(1, 2))
    tr.update(OJAMA_FALL, False, 0.5, next_key=(1, 2))
    event, _ = tr.update(STABLE, True, 1.0, next_key=(1, 2))  # candidate_ojama
    assert event == "candidate_ojama"
    for t in (2.0, 5.0, 10.0, 30.0):
        event, _ = tr.update(STABLE, True, t, next_key=None)
        assert event is None
    assert tr.resolved_is_dead() is False
    # 候補発生時 (1,2) と異なる next (3,4) が戻ってきた → 生存証拠。
    event, _ = tr.update(STABLE, True, 30.1, next_key=(3, 4))
    assert event == "released_survival_ojama"
    assert tr.resolved_is_dead() is False
    assert tr.has_pending_candidate() is False


def test_regression5_post_boundary_background_candidate_frozen() -> None:
    """【回帰5】1試合目で設置後、境界後の背景 `OJAMA_FALL→STABLE` ＋
    死亡セル占有＋固定/None next で未確定 (Codex 指摘3)。"""
    tr = DeathConfirmTracker()
    # 1試合目: 設置を観測 (_has_ever_placed=True になる)。
    tr.update(STABLE, False, 0.0, next_key=(1, 2), game_idx=1)
    tr.update(TSUMO_FALL, False, 1.0, next_key=(1, 2), game_idx=1)
    tr.update(STABLE, False, 1.1, next_key=(1, 2), game_idx=1)
    tr.on_game_boundary()
    # 境界後の背景誤認: OJAMA_FALL→STABLE + 死亡セル占有 + next 固定/None。
    tr.update(OJAMA_FALL, True, 100.0, next_key=None, game_idx=0)
    event, _ = tr.update(STABLE, True, 100.1, next_key=None, game_idx=0)
    assert event is None  # 再アームされていないため候補にすらならない
    event, _ = tr.update(STABLE, True, 200.0, next_key=(5, 5), game_idx=0)  # next固定でも
    assert event is None
    assert tr.resolved_is_dead() is False


def test_regression6_mid_video_start_detects_first_real_placement() -> None:
    """【回帰6】動画が試合途中から始まり `game_idx==0` のままでも、
    最初の真正な設置後は検出できる (=丸ごと見逃さない)。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 0.0, next_key=(1, 2), game_idx=0)
    # まだ設置を一度も観測していない = 凍結中。死亡候補が立っても無視。
    event, _ = tr.update(OJAMA_FALL, True, 0.5, next_key=(1, 2), game_idx=0)
    assert event is None
    event, _ = tr.update(STABLE, True, 0.6, next_key=(1, 2), game_idx=0)
    assert event is None
    assert tr.resolved_is_dead() is False
    # 最初の本当の設置 (死亡セルは空、通常の1手)。
    tr.update(TSUMO_FALL, False, 1.0, next_key=(1, 2), game_idx=0)
    event, _ = tr.update(STABLE, False, 1.1, next_key=(1, 2), game_idx=0)
    assert event is None
    # 以後 (game_idx は依然 0 のまま!) は通常フローで候補・確定が動く。
    tr.update(OJAMA_FALL, False, 5.0, next_key=(3, 4), game_idx=0)
    event, _ = tr.update(STABLE, True, 5.1, next_key=(3, 4), game_idx=0)
    assert event == "candidate_ojama"
    event, delay = tr.update(STABLE, True, 6.6, next_key=(3, 4), game_idx=0)  # 5.1+1.5
    assert event == "confirmed_ojama"
    assert delay is not None
    assert tr.resolved_is_dead() is True


def test_regression7_true_death_is_not_missed() -> None:
    """【回帰7】真の2P窒息を見逃さない (実測ケース: 2P・実試合2・t=223、
    おじゃま満杯による真の敗北)。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 220.0, next_key=(1, 2))
    tr.update(OJAMA_FALL, False, 221.0, next_key=(1, 2))
    # おじゃま着弾の瞬間 (t=223.0) にネクストが最後に動く (以後は不動)。
    event, _ = tr.update(STABLE, True, 223.0, next_key=(2, 3))  # candidate_ojama
    assert event == "candidate_ojama"
    assert tr.resolved_is_dead() is False
    # own chain が一度も来ないまま、ネクスト不動が1.5秒続く。
    event, _ = tr.update(STABLE, True, 224.0, next_key=(2, 3))
    assert event is None
    event, delay = tr.update(STABLE, True, 224.5, next_key=(2, 3))  # 223.0+1.5
    assert event == "confirmed_ojama"
    assert delay is not None
    assert tr.resolved_is_dead() is True
    assert tr.confirmed_source == DEATH_SOURCE_OJAMA
    # 確定後は sticky (見逃しに戻らない)。
    event, _ = tr.update(STABLE, False, 230.0, next_key=(2, 3))
    assert tr.resolved_is_dead() is True


def test_regression8_single_frame_fake_tsumo_fall_does_not_clear_confirmed() -> None:
    """【回帰8】偽 `TSUMO_FALL` 一発で前試合の confirmed を消さない
    (Codex 指摘3: 単なる `curr_state==TSUMO_FALL` だけを新試合開始と
    扱わない。認識の単発誤検出は next も同時に乱れることが多く、
    `next_key=None` で corroboration が不成立になる想定)。"""
    tr = DeathConfirmTracker()
    tr.update(STABLE, False, 220.0, next_key=(1, 2))
    tr.update(OJAMA_FALL, False, 221.0, next_key=(1, 2))
    event, _ = tr.update(STABLE, True, 223.0, next_key=(1, 2))  # candidate_ojama
    assert event == "candidate_ojama"
    event, _ = tr.update(STABLE, True, 224.5, next_key=(1, 2))  # confirmed_ojama (候補後1.5秒)
    assert event == "confirmed_ojama"
    assert tr.resolved_is_dead() is True

    tr.on_game_boundary()
    # 認識の単発誤検出: TSUMO_FALL が1フレームだけ現れるが next は None
    # (corroboration 不成立のため「本当の新試合開始」とはみなさない)。
    event, _ = tr.update(TSUMO_FALL, False, 230.0, next_key=None)
    assert event is None
    assert tr.resolved_is_dead() is True  # 消えない
    event, _ = tr.update(STABLE, False, 230.1, next_key=None)
    assert event is None
    assert tr.resolved_is_dead() is True  # まだ消えない (next が無効なまま)

    # 有効な next を伴う本当の設置完了で、初めて消える。
    tr.update(TSUMO_FALL, False, 235.0, next_key=(9, 9))
    event, _ = tr.update(STABLE, False, 235.1, next_key=(9, 9))
    assert event is None
    assert tr.resolved_is_dead() is False
    assert tr.confirmed_source is None


def test_sticky_death_is_excluded_from_following_game_physical_context() -> None:
    """監査用stickyは残しても、前試合の死亡を次試合の勝敗方向へ渡さない。"""
    tr = DeathConfirmTracker()
    tr._pending_source = DEATH_SOURCE_OJAMA
    tr._pending_game_idx = 2

    source, outcome = tr.on_game_boundary(game_idx=2)

    assert source == DEATH_SOURCE_OJAMA
    assert outcome == "confirmed"
    assert tr.resolved_is_dead() is True
    assert tr.resolved_is_dead_for_game(2) is True
    assert tr.resolved_is_dead_for_game(3) is False
