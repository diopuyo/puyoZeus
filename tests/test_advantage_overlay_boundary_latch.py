"""試合境界の「正式受理」ラッチの回帰テスト (P1 是正 2026-08-26)。

背景 (Codex 第26報レビュー NG / P1):
`scripts/visualize_advantage_overlay.py` の `_detect_score_reset()` は、両者の
スコアが `SCORE_NEAR_ZERO_THRESHOLD` (20) 以下の間**毎フレーム True** を返す。
従来は `game_idx += 1` だけが debounce されており、死亡確定の境界処理
`resolve_boundary_confirmations()` は debounce の外にあった。そのため実境界
約6件に対して `total_boundaries=715` 回呼ばれ、`on_game_boundary()` が毎回
`_post_boundary_armed=False` へ戻すことで、**新試合冒頭の再武装や死亡候補を
取りこぼしうる**状態だった。

本ファイルは是正後の配線を検査する:
- `accept_formal_boundary()` / `update_score_reset_latch()` の純関数としての挙動
- 呼出側と同じ順序でフレームを流したときの、境界処理の回数と再武装・候補検出

呼出側ループの再現は `_run_frames()` に閉じ込める。ここが本番の配線
(`scripts/visualize_advantage_overlay.py` の該当ブロック) と同じ順序で
「reset 判定 → 正式受理判定 → 境界処理 → ラッチ更新」を行う。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402
from src.death_confirmation import (  # noqa: E402
    DeathConfirmStats,
    DeathConfirmTracker,
    resolve_boundary_confirmations,
)

STABLE = "STABLE"
TSUMO_FALL = "TSUMO_FALL"
OJAMA_FALL = "OJAMA_FALL"


# ============================
# accept_formal_boundary (純関数)
# ============================


def test_accept_formal_boundary_first_rising_edge_accepted() -> None:
    """最初の立ち上がりは受理する (前回受理が無いので debounce は素通り)。"""
    assert vao.accept_formal_boundary(
        reset_now=True, latched=False, t_sec=100.0, last_formal_t=None) is True


def test_accept_formal_boundary_latched_rejected() -> None:
    """信号が立ち続けている間 (ラッチ中) は受理しない = 毎フレーム再受理しない。"""
    assert vao.accept_formal_boundary(
        reset_now=True, latched=True, t_sec=100.0, last_formal_t=None) is False


def test_accept_formal_boundary_not_reset_rejected() -> None:
    """reset 信号が立っていなければ受理しない。"""
    assert vao.accept_formal_boundary(
        reset_now=False, latched=False, t_sec=100.0, last_formal_t=None) is False


def test_accept_formal_boundary_debounce_blocks_quick_reedge() -> None:
    """立ち上がりでも前回受理から debounce 秒未満なら受理しない
    (OCR ちらつきで信号が一瞬落ちて立ち上がり直す場合)。"""
    assert vao.accept_formal_boundary(
        reset_now=True, latched=False, t_sec=102.0, last_formal_t=100.0) is False


def test_accept_formal_boundary_debounce_boundary_value_accepted() -> None:
    """ちょうど debounce 秒経過なら受理する (境界値、>= で判定)。"""
    t = 100.0 + vao.GAME_BOUNDARY_DEBOUNCE_SEC
    assert vao.accept_formal_boundary(
        reset_now=True, latched=False, t_sec=t, last_formal_t=100.0) is True


def test_accept_formal_boundary_long_low_score_does_not_reaccept() -> None:
    """【Codex 要件3】低得点が debounce 秒より長く続いても、信号が立ち下がって
    いない限り再受理しない (時間ではなく信号の縁で切っているため)。"""
    assert vao.accept_formal_boundary(
        reset_now=True, latched=True, t_sec=200.0, last_formal_t=100.0) is False


# ============================
# update_score_reset_latch (純関数)
# ============================


def test_latch_sets_on_reset() -> None:
    assert vao.update_score_reset_latch(False, True, 0, 0) is True


def test_latch_clears_when_scores_readable_and_no_reset() -> None:
    assert vao.update_score_reset_latch(True, False, 5000, 4000) is False


def test_latch_holds_when_score_unreadable() -> None:
    """score が None (OCR 失敗) は「境界が終わった」ではなく「判定不能」。
    ここで解除すると OCR の瞬断のたびに同じ境界を再受理してしまう。"""
    assert vao.update_score_reset_latch(True, False, None, 4000) is True
    assert vao.update_score_reset_latch(True, False, 5000, None) is True


def test_latch_stays_false_when_nothing_happens() -> None:
    assert vao.update_score_reset_latch(False, False, 5000, 4000) is False


# ============================
# 呼出側と同じ順序でフレームを流す再現ハーネス
# ============================


def _run_frames(
    frames: list[tuple[float, int | None, int | None, str, bool, object]],
    tr1: DeathConfirmTracker, tr2: DeathConfirmTracker,
    stats: DeathConfirmStats,
) -> tuple[list[float], int]:
    """本番の配線と同じ順序でフレームを処理する。

    Returns:
        (正式受理した境界の時刻リスト, 最終 game_idx)。

    frames の各要素は
    `(t_sec, score1, score2, tr2側のstate, tr2側の死亡セル占有, tr2側のnext_key)`。
    1P 側 (tr1) は常に生存側として STABLE / 非占有で流す
    (両側同時 pending = ambiguous を意図せず踏まないため)。
    """
    latched = False
    last_formal_t: float | None = None
    prev1: int | None = None
    prev2: int | None = None
    accepted: list[float] = []
    game_idx = 0
    for (t, s1, s2, state2, occupied2, next2) in frames:
        reset_now = vao._detect_score_reset(s1, s2, prev1, prev2)
        # 【第2版】正式境界の判定は reset の有無に関わらず毎フレーム行い、
        # 受理したイベントの中で game_idx 加算・スナップショット・境界処理を
        # **まとめて1回だけ**行う (本番の配線と同じ順序・同じ条件)。
        formal = vao.accept_formal_boundary(
            reset_now=reset_now, latched=latched, t_sec=t,
            last_formal_t=last_formal_t)
        if formal:
            ending_game_idx = game_idx
            last_formal_t = t
            accepted.append(t)
            game_idx += 1
            resolve_boundary_confirmations(
                tr1, tr2, game_idx=ending_game_idx, stats=stats)
        latched = vao.update_score_reset_latch(latched, reset_now, s1, s2)
        prev1, prev2 = s1, s2
        # 観測は境界処理の後 (本番も pipe.update → 境界 → tracker.update の順)。
        tr1.update(STABLE, False, t, next_key=(9, 9), game_idx=game_idx)
        tr2.update(state2, occupied2, t, next_key=next2, game_idx=game_idx)
    return accepted, game_idx


def test_regression9_long_low_score_yields_single_boundary() -> None:
    """【回帰9・Codex 要件4-1】低得点 reset が多数フレームにわたって続いても、
    境界処理は1回だけで `total_boundaries` も1になる。

    是正前は `_detect_score_reset` が毎フレーム True を返す間ずっと
    `resolve_boundary_confirmations()` が呼ばれ、実境界1件に対して
    フレーム数ぶん (ここでは 300 回) 計上されていた。
    """
    tr1, tr2 = DeathConfirmTracker(), DeathConfirmTracker()
    stats = DeathConfirmStats()
    # 10秒間 (30fps = 300 フレーム) スコアが 0 付近に留まる。
    frames = [
        (100.0 + i / 30.0, 0, 0, STABLE, False, (1, 2)) for i in range(300)
    ]
    accepted, final_game_idx = _run_frames(frames, tr1, tr2, stats)
    assert accepted == [100.0]
    assert stats.total_boundaries == 1, (
        f"境界処理は1回であるべき: {stats.total_boundaries}/{len(frames)}フレーム")


def test_regression10_rearm_during_continuing_reset_signal() -> None:
    """【回帰10・Codex 要件4-2】reset 信号が continue している最中に新試合の
    `TSUMO_FALL → STABLE` が来ても再武装できる。

    是正前は毎フレーム `on_game_boundary()` が呼ばれて
    `_post_boundary_armed=False` に戻されるため、再武装が成立しなかった。
    """
    tr1, tr2 = DeathConfirmTracker(), DeathConfirmTracker()
    stats = DeathConfirmStats()
    frames: list[tuple[float, int | None, int | None, str, bool, object]] = []
    # 前試合で設置を観測させる (スコアは十分大きく reset を立てない)。
    frames.append((90.0, 5000, 4000, TSUMO_FALL, False, (1, 2)))
    frames.append((90.1, 5000, 4000, STABLE, False, (1, 2)))
    # 境界: スコアが 0 付近へ落ち、そのまま 5秒 (150フレーム) 低得点が続く。
    # その途中 (t=101.0 以降) で新試合の設置が始まる。
    for i in range(150):
        t = 100.0 + i / 30.0
        if t < 101.0:
            state, nxt = STABLE, (1, 2)
        elif t < 101.2:
            state, nxt = TSUMO_FALL, (3, 4)
        else:
            state, nxt = STABLE, (3, 4)
        frames.append((t, 0, 0, state, False, nxt))
    accepted, final_game_idx = _run_frames(frames, tr1, tr2, stats)
    assert accepted == [100.0]
    assert stats.total_boundaries == 1
    # 新試合の設置を観測できたので候補受付が再武装されている。
    assert tr2._post_boundary_armed is True, (
        "reset 信号継続中でも新試合の設置で再武装できること")


def test_regression11_detects_candidate_in_new_game_after_reset() -> None:
    """【回帰11・Codex 要件4-3】再武装したその新試合で、後続の死亡候補を
    検出できる (取りこぼさない)。"""
    tr1, tr2 = DeathConfirmTracker(), DeathConfirmTracker()
    stats = DeathConfirmStats()
    frames: list[tuple[float, int | None, int | None, str, bool, object]] = []
    frames.append((90.0, 5000, 4000, TSUMO_FALL, False, (1, 2)))
    frames.append((90.1, 5000, 4000, STABLE, False, (1, 2)))
    # 境界 + 低得点継続中に新試合の設置。
    for i in range(150):
        t = 100.0 + i / 30.0
        if t < 101.0:
            state, nxt = STABLE, (1, 2)
        elif t < 101.2:
            state, nxt = TSUMO_FALL, (3, 4)
        else:
            state, nxt = STABLE, (3, 4)
        frames.append((t, 0, 0, state, False, nxt))
    _run_frames(frames, tr1, tr2, stats)
    assert tr2._post_boundary_armed is True
    # 新試合でスコアが伸びた後、おじゃま着弾で死亡セルが埋まる。
    tr2.update(OJAMA_FALL, False, 120.0, next_key=(5, 6), game_idx=1)
    event, _ = tr2.update(STABLE, True, 120.1, next_key=(5, 6), game_idx=1)
    assert event == "candidate_ojama", f"候補を検出できていない: {event}"


def test_regression12_long_low_score_advances_game_idx_once() -> None:
    """【回帰12・Codex 第27報レビュー 対応2】低得点が debounce 秒より長く
    (ここでは12秒 = 5秒を2回またぐ) 続いても、**正式境界も `game_idx` 加算も
    1回だけ**。

    第1版は `game_idx` の加算を従来の debounce に残していたため、正式境界なしに
    `game_idx` が進み、死亡候補の `_pending_game_idx` と次の `_ending_game_idx`
    が食い違って**真の死亡が `rejected_game_idx_mismatch` になりうる**状態だった。
    第2版で境界処理全体を正式境界イベント1回へ統合したので、それを固定する。
    """
    tr1, tr2 = DeathConfirmTracker(), DeathConfirmTracker()
    stats = DeathConfirmStats()
    # 12秒間 (360フレーム) 低得点が続く = debounce 5秒を2回またぐ。
    frames = [
        (100.0 + i / 30.0, 0, 0, STABLE, False, (1, 2)) for i in range(360)
    ]
    accepted, final_game_idx = _run_frames(frames, tr1, tr2, stats)
    assert accepted == [100.0], f"正式境界が1回でない: {accepted}"
    assert stats.total_boundaries == 1, (
        f"境界処理は1回であるべき: {stats.total_boundaries}/{len(frames)}フレーム")
    assert final_game_idx == 1, (
        f"game_idx も1回だけ進むべき: {final_game_idx} "
        "(正式境界と分離されていると 5秒ごとに進んでしまう)")


def test_regression13_true_death_still_confirms_at_boundary() -> None:
    """【回帰13】統合後も、低得点が長く続く境界で真の死亡が確定する
    (越境拒否にならない)。

    **正直な但し書き**: Codex が懸念した実害「候補の `_pending_game_idx` と
    境界の `_ending_game_idx` が食い違って真の死亡が
    `rejected_game_idx_mismatch` になる」は、**第1版の配線でも再現できなかった**
    (このシナリオを第1版で流すと PASS する)。理由は、候補が立ってから
    `NEXT_STATIONARY_CONFIRM_SEC` (1.5秒) 以内に確定するか解除されるため、
    その間に spurious な `game_idx` 加算 (5秒間隔) が挟まる余地がないこと。
    実データでも真の死亡 (t=223) は候補から 0.37秒で試合終了している。

    したがって本テストは「Codex 指摘の実害の再現」ではなく、
    **統合によって真の死亡の確定経路を壊していないこと**の回帰である。
    実害を捉えているのは `test_regression12`(game_idx の多重加算そのもの)。
    """
    tr1, tr2 = DeathConfirmTracker(), DeathConfirmTracker()
    stats = DeathConfirmStats()
    frames: list[tuple[float, int | None, int | None, str, bool, object]] = []
    # 試合中: 設置を観測し、その後おじゃま着弾で死亡候補が立つ。
    frames.append((90.0, 5000, 4000, TSUMO_FALL, False, (1, 2)))
    frames.append((90.1, 5000, 4000, STABLE, False, (1, 2)))
    frames.append((90.2, 5000, 4000, OJAMA_FALL, False, (1, 2)))
    frames.append((90.3, 5000, 4000, STABLE, True, (1, 2)))   # 死亡候補
    # そのまま決着 → スコアが0付近へ落ち、12秒続く (debounce を2回またぐ)。
    for i in range(360):
        frames.append((100.0 + i / 30.0, 0, 0, STABLE, True, (1, 2)))
    _run_frames(frames, tr1, tr2, stats)
    assert stats.total_boundaries == 1
    # 越境拒否 (`rejected_game_idx_mismatch`) は expired_at_boundary_* に計上される。
    # 0/1 = 「1回の境界で越境拒否は起きなかった」(母数併記)。
    expired = stats.expired_at_boundary_placement + stats.expired_at_boundary_ojama
    assert expired == 0, (
        f"真の死亡が越境拒否された: {expired}/{stats.total_boundaries}")
    assert stats.boundary_confirmed == 1, (
        f"境界で死亡確定されなかった: {stats.boundary_confirmed}/{stats.total_boundaries}")
    assert tr2.resolved_is_dead() is True
