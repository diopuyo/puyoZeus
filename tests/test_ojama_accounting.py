"""src/ojama_accounting.py のユニットテスト (アーキ案A 全面書き換え版)。

テスト対象:
    1. 初期状態 (ゼロ確認)
    2. on_state_transition による連鎖終了一括換算
    3. leftover 繰越 (複数連鎖跨ぎ)
    4. 相殺の正しさ【回帰防止テスト】
       - 小連鎖が大予告を消さない (有利不利反転しない)
       - 大連鎖が大予告を全消し + 余剰を相手に送る
    5. on_tsumo_settled で予告が最大30減る、0なら過剰drainなし
    6. 全消し特別処理が無い (all_clear_pending 常 False)
    7. 試合境界 (score減少/MENU) で forecast/leftover reset
    8. score None 時の遅延確定とタイムアウト
    9. chain_end_triggered が連鎖終了遷移で立つ
    10. get_snapshot のフィールド存在 (後方互換)
    11. reset() で帳簿クリア
    12. 後方互換 API (update_from_score / update_from_boards / update_accounting_with_chain)
    13. forecast フィールドが pending と同値
    14. offboard / pending_capped
    15. net_balance_capped の 0-1 正規化可能性
"""
from __future__ import annotations

import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    Board,
)
from src.board_state_machine import BoardState
from src.ojama_accounting import (
    CHAIN_COALESCE_WINDOW_SEC,
    CHAIN_FIRE_MIN_SCORE,
    CONFIDENCE_SCORE_OCR_ONLY,
    ON_FIELD_CAP,
    PENDING_ABS_CAP,
    PENDING_HARD_CAP,
    SCORE_RESET_THRESHOLD,
    THEORY_DROP_PER_TURN,
    OjamaAccountSnapshot,
    OjamaAccountingTracker,
)
from src.scoring import (
    OJAMA_MAX_DROP_PER_TURN,
    OJAMA_RATE_STANDARD,
    score_to_ojama,
)


# ============================
# ヘルパー
# ============================

def _score_to_ojama_count(score: int, leftover: int = 0) -> tuple[int, int]:
    """(ojama_count, new_leftover) を返す。"""
    r = score_to_ojama(score, prev_leftover=leftover)
    return r.ojama_count, r.leftover_score


def _make_empty_board() -> Board:
    """全 EMPTY の盤面を返す。"""
    data = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    return Board.from_list(data)


def _make_board_with_ojama(ojama_count: int) -> Board:
    """可視領域 (row=1〜) に指定数のおじゃまぷよを配置した盤面を返す。"""
    data = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    placed = 0
    for row in range(BOARD_ROWS - 1, 0, -1):
        for col in range(BOARD_COLS):
            if placed >= ojama_count:
                break
            data[row][col] = COLOR_OJAMA
            placed += 1
        if placed >= ojama_count:
            break
    return Board.from_list(data)


def _settle_score(
    tracker: OjamaAccountingTracker,
    side: str,
    score: int,
    t_start: float,
    dt: float = 0.001,
) -> float:
    """settle 待ちを確定させるため K_SETTLE_FRAMES フレーム分の STABLE→STABLE を流す。

    連鎖終了直後に呼び出すことで finalize を確定させる。
    dt はデフォルト 0.001秒(テスト高速化 + coalesce window への干渉回避)。

    Returns:
        最後のフレームの t_sec。
    """
    from src.ojama_accounting import K_SETTLE_FRAMES

    for i in range(K_SETTLE_FRAMES):
        tracker.on_state_transition(
            side, BoardState.STABLE, BoardState.STABLE,
            score, t_start + (i + 1) * dt,
        )
    return t_start + K_SETTLE_FRAMES * dt


def _fire_chain(
    tracker: OjamaAccountingTracker,
    side: str,
    chain_score: int,
    score_before: int = 0,
    t_sec: float = 5.0,
) -> OjamaAccountSnapshot:
    """連鎖開始 → 連鎖終了 → score settle の状態遷移を一連でシミュレートする。

    score settle 待ち (K_SETTLE_FRAMES 連続不変) が入ったため、
    連鎖終了後に K_SETTLE_FRAMES フレーム分の STABLE→STABLE を同一スコアで流す。
    これにより finalize が確定し、呼出元が即座に forecast を確認できる。

    Returns:
        settle 確定後のスナップショット。
    """
    from src.ojama_accounting import K_SETTLE_FRAMES

    score_after = score_before + chain_score
    # 連鎖開始: STABLE → CHAIN
    tracker.on_state_transition(
        side, BoardState.STABLE, BoardState.CHAIN,
        score_before, t_sec,
    )
    # 連鎖終了: CHAIN → STABLE (settle 待ち開始)
    tracker.on_state_transition(
        side, BoardState.CHAIN, BoardState.STABLE,
        score_after, t_sec + 2.0,
    )
    # settle 確定: K_SETTLE_FRAMES フレーム連続で同一スコアを通知
    # dt=0.001 でテスト高速化 + coalesce window (2.5s) への干渉回避
    _dt = 0.001
    for i in range(K_SETTLE_FRAMES):
        tracker.on_state_transition(
            side, BoardState.STABLE, BoardState.STABLE,
            score_after, t_sec + 2.0 + (i + 1) * _dt,
        )
    return tracker.get_snapshot(t_sec + 2.0 + K_SETTLE_FRAMES * _dt)


# ============================
# 1. 初期状態テスト
# ============================

def test_initial_snapshot_zero() -> None:
    """reset 直後のスナップショットは全帳簿ゼロ。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    snap = tracker.get_snapshot(t_sec=0.0)
    assert snap.pending_p1 == 0
    assert snap.pending_p2 == 0
    assert snap.forecast_p1 == 0
    assert snap.forecast_p2 == 0
    assert snap.total_generated_by_p1 == 0
    assert snap.total_generated_by_p2 == 0
    assert snap.total_offset_by_p1 == 0
    assert snap.total_offset_by_p2 == 0
    assert snap.leftover_p1 == 0
    assert snap.leftover_p2 == 0
    assert not snap.all_clear_pending_p1
    assert not snap.all_clear_pending_p2


# ============================
# 2. 連鎖終了一括換算テスト
# ============================

def test_chain_end_bulk_calculation() -> None:
    """連鎖終了時に chain_total // 70 が生成され相手 forecast に追加される。

    chain_total=350 → G=5(端数50繰越)。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()
    chain_total = 350
    expected_ojama, expected_leftover = _score_to_ojama_count(chain_total)
    assert expected_ojama == 5
    assert expected_leftover == 0  # 350 = 5×70, 余り0

    snap = _fire_chain(tracker, "p1", chain_score=chain_total)

    assert snap.forecast_p2 == expected_ojama, (
        f"forecast_p2={snap.forecast_p2} != expected={expected_ojama}"
    )
    assert snap.forecast_p1 == 0
    assert snap.leftover_p1 == expected_leftover
    assert snap.total_generated_by_p1 == expected_ojama


def test_chain_end_with_leftover() -> None:
    """端数(leftover)が残る連鎖: chain_total=380 → G=5, leftover=30。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    chain_total = 380
    expected_ojama, expected_leftover = _score_to_ojama_count(chain_total)
    assert expected_ojama == 5
    assert expected_leftover == 30  # 380 = 5×70 + 30

    snap = _fire_chain(tracker, "p1", chain_score=chain_total)

    assert snap.forecast_p2 == expected_ojama
    assert snap.leftover_p1 == expected_leftover


def test_mid_chain_no_generation() -> None:
    """連鎖途中(CHAIN状態中)では生成・相殺が起きない。

    STABLE→CHAIN 遷移後、まだ STABLE に戻っていない時点では
    forecast が変化しない。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 連鎖開始のみ通知 (終了は通知しない)
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN,
        score=1000, t_sec=5.0,
    )
    snap_mid = tracker.get_snapshot(t_sec=5.5)

    # まだ連鎖終了していないので forecast は 0
    assert snap_mid.forecast_p2 == 0
    assert snap_mid.total_generated_by_p1 == 0


def test_gravity_settle_to_stable_triggers_chain_end() -> None:
    """GRAVITY_SETTLE → STABLE 遷移でも連鎖終了を正しく検知する。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    chain_total = 2100  # 30 ojama
    expected_ojama, _ = _score_to_ojama_count(chain_total)

    score_before = 0
    score_after = chain_total
    # STABLE → CHAIN (連鎖開始)
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score_before, t_sec=5.0,
    )
    # CHAIN → GRAVITY_SETTLE (連鎖中から重力 settle)
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.GRAVITY_SETTLE, score_after, t_sec=6.0,
    )
    # GRAVITY_SETTLE → STABLE (連鎖終了 → settle 待ち開始)
    tracker.on_state_transition(
        "p1", BoardState.GRAVITY_SETTLE, BoardState.STABLE, score_after, t_sec=7.0,
    )
    # settle 確定: K_SETTLE_FRAMES フレーム流す
    t_end = _settle_score(tracker, "p1", score_after, t_start=7.0)
    snap = tracker.get_snapshot(t_end)

    assert snap.forecast_p2 == expected_ojama, (
        f"GRAVITY_SETTLE経由: forecast_p2={snap.forecast_p2} != {expected_ojama}"
    )


# ============================
# 3. leftover 繰越テスト
# ============================

def test_leftover_carries_over_multiple_chains() -> None:
    """複数連鎖跨ぎで leftover が正しく引き継がれる。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 1連鎖目: 380点 → G=5, leftover=30
    r1 = score_to_ojama(380, prev_leftover=0)
    assert r1.ojama_count == 5
    assert r1.leftover_score == 30

    # 2連鎖目: 380点 + leftover=30 = 410 → G=5, leftover=60
    r2 = score_to_ojama(380, prev_leftover=30)
    assert r2.ojama_count == 5
    assert r2.leftover_score == 60

    # 1連鎖目
    _fire_chain(tracker, "p1", chain_score=380, score_before=0, t_sec=5.0)
    snap1 = tracker.get_snapshot(t_sec=7.0)
    assert snap1.leftover_p1 == 30

    # 2連鎖目
    _fire_chain(tracker, "p1", chain_score=380, score_before=380, t_sec=10.0)
    snap2 = tracker.get_snapshot(t_sec=12.0)
    assert snap2.leftover_p1 == 60
    assert snap2.total_generated_by_p1 == r1.ojama_count + r2.ojama_count


# ============================
# 4. 相殺の正しさ【回帰防止テスト】
# ============================

def test_offset_small_chain_does_not_cancel_large_forecast() -> None:
    """小連鎖(G=1)が大予告(60)を消さない — 有利不利反転バグの回帰防止。

    シナリオ:
        2P が G=60 を 1P に送る → 1P.forecast=60
        1P が G=1 を撃つ → 1P.forecast=59 (1だけ消える)
        2P.forecast は増えない (surplus=0)
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 2P が G=60 を 1P へ送る (60×70 = 4200点相当)
    _fire_chain(tracker, "p2", chain_score=4200, score_before=0, t_sec=5.0)
    snap_after_2p = tracker.get_snapshot(t_sec=7.0)
    expected_from_2p, _ = _score_to_ojama_count(4200)
    assert snap_after_2p.forecast_p1 == expected_from_2p, (
        f"2P連鎖後 forecast_p1={snap_after_2p.forecast_p1} != {expected_from_2p}"
    )

    # 1P が G=1 (70点) を撃つ
    one_ojama_score = OJAMA_RATE_STANDARD  # 70点
    _fire_chain(tracker, "p1", chain_score=one_ojama_score, score_before=0, t_sec=10.0)
    snap = tracker.get_snapshot(t_sec=12.0)

    expected_p1_after = expected_from_2p - 1  # 1だけ相殺
    assert snap.forecast_p1 == expected_p1_after, (
        f"小連鎖(G=1)後 forecast_p1={snap.forecast_p1} "
        f"(expected={expected_p1_after}, 有利不利反転バグなら0になる)"
    )
    # surplus=0 なので 2P の forecast は増えない
    assert snap.forecast_p2 == 0, (
        f"小連鎖後 forecast_p2={snap.forecast_p2} (0になるべき)"
    )


def test_offset_large_chain_cancels_all_and_sends_surplus() -> None:
    """大連鎖(G=80)が大予告(60)を全消し + 余剰(20)を相手に送る。

    シナリオ:
        2P が G=60 を 1P に送る → 1P.forecast=60
        1P が G=80 を撃つ → 1P.forecast=0, 2P.forecast += 20
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 2P が G=60 を送る
    score_for_60 = 60 * OJAMA_RATE_STANDARD  # 4200点
    _fire_chain(tracker, "p2", chain_score=score_for_60, score_before=0, t_sec=5.0)
    snap_mid = tracker.get_snapshot(t_sec=7.0)
    g_from_2p, _ = _score_to_ojama_count(score_for_60)
    assert snap_mid.forecast_p1 == g_from_2p  # =60

    # 1P が G=80 を撃つ
    score_for_80 = 80 * OJAMA_RATE_STANDARD  # 5600点
    _fire_chain(tracker, "p1", chain_score=score_for_80, score_before=0, t_sec=10.0)
    snap = tracker.get_snapshot(t_sec=12.0)

    g_from_1p, _ = _score_to_ojama_count(score_for_80)  # =80
    assert g_from_1p == 80

    expected_p1_forecast = 0  # 全相殺
    expected_p2_forecast = 80 - 60  # surplus=20
    assert snap.forecast_p1 == expected_p1_forecast, (
        f"大連鎖相殺後 forecast_p1={snap.forecast_p1} != 0"
    )
    assert snap.forecast_p2 == expected_p2_forecast, (
        f"大連鎖surplus forecast_p2={snap.forecast_p2} != {expected_p2_forecast}"
    )


def test_offset_exact_cancel() -> None:
    """同量の相互発火で両者 forecast が 0 になる。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    same_score = 2100  # 30 ojama

    # 2P が G=30 を 1P に送る
    _fire_chain(tracker, "p2", chain_score=same_score, score_before=0, t_sec=5.0)
    snap_mid = tracker.get_snapshot(t_sec=7.0)
    assert snap_mid.forecast_p1 == 30

    # 1P が G=30 で相殺
    _fire_chain(tracker, "p1", chain_score=same_score, score_before=0, t_sec=10.0)
    snap = tracker.get_snapshot(t_sec=12.0)

    assert snap.forecast_p1 == 0
    assert snap.forecast_p2 == 0
    assert snap.net_ojama_balance == 0


# ============================
# 5. on_tsumo_settled テスト
# ============================

def test_tsumo_settled_drains_forecast() -> None:
    """on_tsumo_settled で forecast が最大 THEORY_DROP_PER_TURN 減る。

    forecast=30, settled → forecast=0 (全drain)。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()
    chain_score = 2100  # 30 ojama
    expected_ojama, _ = _score_to_ojama_count(chain_score)
    assert expected_ojama == 30

    # 2P が 30 個を 1P へ送る
    _fire_chain(tracker, "p2", chain_score=chain_score, score_before=0, t_sec=5.0)
    snap_before = tracker.get_snapshot(t_sec=7.0)
    assert snap_before.forecast_p1 == 30

    # 1P tsumo 着地
    tracker.on_tsumo_settled("p1", t_sec=8.0)
    snap = tracker.get_snapshot(t_sec=8.0)

    assert snap.forecast_p1 == 0, (
        f"tsumo_settled後 forecast_p1={snap.forecast_p1} should be 0"
    )
    assert snap.total_dropped_to_p1 == expected_ojama


def test_tsumo_settled_partial_drain() -> None:
    """forecast > THEORY_DROP_PER_TURN の場合は最大30だけ drain。

    forecast=60 → settled → forecast=30。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 2P が G=60 を 1P に送る
    score_for_60 = 60 * OJAMA_RATE_STANDARD
    _fire_chain(tracker, "p2", chain_score=score_for_60, score_before=0, t_sec=5.0)
    snap_before = tracker.get_snapshot(t_sec=7.0)
    g_60, _ = _score_to_ojama_count(score_for_60)
    assert snap_before.forecast_p1 == g_60  # 60

    # 1P tsumo 着地
    tracker.on_tsumo_settled("p1", t_sec=8.0)
    snap = tracker.get_snapshot(t_sec=8.0)

    assert snap.forecast_p1 == g_60 - THEORY_DROP_PER_TURN, (
        f"partial drain後 forecast_p1={snap.forecast_p1} "
        f"!= {g_60 - THEORY_DROP_PER_TURN}"
    )


def test_tsumo_settled_no_over_drain_when_zero() -> None:
    """forecast=0 の時に on_tsumo_settled を呼んでも過剰 drain しない。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    tracker.on_tsumo_settled("p1", t_sec=5.0)
    tracker.on_tsumo_settled("p2", t_sec=5.0)
    snap = tracker.get_snapshot(t_sec=5.0)

    assert snap.forecast_p1 == 0
    assert snap.forecast_p2 == 0
    assert snap.total_dropped_to_p1 == 0
    assert snap.total_dropped_to_p2 == 0


# ============================
# 6. 全消し特別処理がない確認
# ============================

def test_all_clear_pending_always_false() -> None:
    """all_clear_pending は常 False (廃止済み)。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    _fire_chain(tracker, "p1", chain_score=2100, t_sec=5.0)
    snap = tracker.get_snapshot(t_sec=7.0)
    assert snap.all_clear_pending_p1 is False
    assert snap.all_clear_pending_p2 is False


def test_no_all_clear_bonus_applied() -> None:
    """全消しボーナス加算なし: 同じ chain_score からは同じ G が生成される。

    旧実装では全消し後に +2100pt ボーナスが乗っていたが、新実装では乗らない。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    chain_score = 1000
    expected_ojama, _ = _score_to_ojama_count(chain_score)
    snap = _fire_chain(tracker, "p1", chain_score=chain_score, t_sec=5.0)

    # ボーナスなし = chain_score のみで換算
    assert snap.total_generated_by_p1 == expected_ojama, (
        f"全消しボーナス非加算確認: generated={snap.total_generated_by_p1} "
        f"!= expected={expected_ojama}"
    )


# ============================
# 7. 試合境界 reset テスト
# ============================

def test_boundary_reset_on_score_decrease() -> None:
    """score 大幅減少(≥ SCORE_RESET_THRESHOLD)で forecast/leftover がリセット。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 2P が G=30 を 1P に送る
    _fire_chain(tracker, "p2", chain_score=2100, score_before=0, t_sec=5.0)
    snap_before = tracker.get_snapshot(t_sec=7.0)
    assert snap_before.forecast_p1 == 30

    # 1P score が大幅減少(試合切り替え)
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE,
        score=31000, t_sec=8.0,
    )
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE,
        score=150, t_sec=9.0,  # 31000→150: SCORE_RESET_THRESHOLD(500)超え
    )
    snap_reset = tracker.get_snapshot(t_sec=9.0)

    assert snap_reset.forecast_p1 == 0, (
        f"境界後 forecast_p1={snap_reset.forecast_p1} should be 0"
    )
    assert snap_reset.leftover_p1 == 0


def test_boundary_reset_on_menu_transition() -> None:
    """MENU 遷移で forecast/leftover がリセット。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 2P が 1P に G=30 送る
    _fire_chain(tracker, "p2", chain_score=2100, score_before=0, t_sec=5.0)
    snap_before = tracker.get_snapshot(t_sec=7.0)
    assert snap_before.forecast_p1 == 30

    # 1P が MENU 遷移
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.MENU, score=5000, t_sec=10.0,
    )
    snap = tracker.get_snapshot(t_sec=10.0)

    assert snap.forecast_p1 == 0, (
        f"MENU遷移後 forecast_p1={snap.forecast_p1} should be 0"
    )


def test_boundary_no_reset_on_small_decrease() -> None:
    """SCORE_RESET_THRESHOLD 未満の score 減少は境界とみなさない。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 2P が G=30 を 1P に送る
    _fire_chain(tracker, "p2", chain_score=2100, score_before=0, t_sec=5.0)
    snap_before = tracker.get_snapshot(t_sec=7.0)
    expected_forecast = snap_before.forecast_p1

    # SCORE_RESET_THRESHOLD - 1 の減少(境界未満)
    high_score = 10000
    small_dec = SCORE_RESET_THRESHOLD - 1
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=high_score, t_sec=8.0,
    )
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE,
        score=high_score - small_dec, t_sec=9.0,
    )
    snap_after = tracker.get_snapshot(t_sec=9.0)

    assert snap_after.forecast_p1 == expected_forecast, (
        f"小幅減少で境界誤検知: forecast_p1={snap_after.forecast_p1} "
        f"!= {expected_forecast}"
    )


# ============================
# 8. score None 遅延確定テスト
# ============================

def test_chain_end_deferred_when_score_none() -> None:
    """CHAIN→STABLE 遷移時 score=None なら待機し、settle 後に確定。

    新仕様: score が来ても即確定せず K_SETTLE_FRAMES 連続不変後に確定。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    chain_total = 2100
    expected_ojama, _ = _score_to_ojama_count(chain_total)

    # 連鎖開始
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=0, t_sec=5.0,
    )
    # 連鎖終了 score=None (OCR 失敗) → settle 待ち開始
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=None, t_sec=7.0,
    )
    snap_pending = tracker.get_snapshot(t_sec=7.0)
    # まだ確定していない
    assert snap_pending.forecast_p2 == 0

    # 次フレームで score が来る → settle 待ちに候補が入るが確定はまだ
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=chain_total, t_sec=7.1,
    )
    snap_one_frame = tracker.get_snapshot(t_sec=7.1)
    # まだ settle 完了していない(1フレームのみ)
    assert snap_one_frame.forecast_p2 == 0

    # K_SETTLE_FRAMES 連続不変で settle 確定
    t_end = _settle_score(tracker, "p1", chain_total, t_start=7.1)
    snap_confirmed = tracker.get_snapshot(t_end)

    assert snap_confirmed.forecast_p2 == expected_ojama, (
        f"settle確定後 forecast_p2={snap_confirmed.forecast_p2} != {expected_ojama}"
    )


def test_chain_end_pending_timeout() -> None:
    """chain_end_pending タイムアウト: 30フレーム後 score None 継続で破棄。"""
    from src.ojama_accounting import CHAIN_END_PENDING_TIMEOUT_FRAMES
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 連鎖開始
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=0, t_sec=5.0,
    )
    # 連鎖終了 score=None
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=None, t_sec=7.0,
    )
    # タイムアウトフレーム分 score=None で通知
    for i in range(CHAIN_END_PENDING_TIMEOUT_FRAMES + 1):
        tracker.on_state_transition(
            "p1", BoardState.STABLE, BoardState.STABLE,
            score=None, t_sec=7.0 + i * 0.033,
        )
    snap = tracker.get_snapshot(t_sec=8.0)

    # タイムアウトで破棄 → forecast は 0 のまま
    assert snap.forecast_p2 == 0, (
        f"タイムアウト後 forecast_p2={snap.forecast_p2} (0になるべき)"
    )


# ============================
# 9. chain_end_triggered テスト
# ============================

def test_chain_end_triggered_set_on_chain_end() -> None:
    """CHAIN → STABLE 遷移で chain_end_triggered が True になる。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 連鎖開始
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=0, t_sec=5.0,
    )
    snap_chain = tracker.get_snapshot(t_sec=5.5)
    # 連鎖中は triggered=False のまま
    assert snap_chain.chain_end_triggered_p1 is False

    # 連鎖終了
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=2100, t_sec=7.0,
    )
    snap_end = tracker.get_snapshot(t_sec=7.0)
    assert snap_end.chain_end_triggered_p1 is True, (
        "CHAIN→STABLE遷移で chain_end_triggered_p1 が True になるべき"
    )


# ============================
# 10. snapshot フィールド存在テスト (後方互換)
# ============================

def test_snapshot_all_fields_present() -> None:
    """必須フィールドおよび新追加フィールドが全て存在する。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    snap = tracker.get_snapshot(t_sec=1.0)
    required_fields = [
        # 既存フィールド (変更禁止)
        "t_sec", "pending_p1", "pending_p2",
        "total_generated_by_p1", "total_generated_by_p2",
        "total_offset_by_p1", "total_offset_by_p2",
        "total_dropped_to_p1", "total_dropped_to_p2",
        "net_ojama_balance",
        "overflow_risk_p1", "overflow_risk_p2",
        "confidence",
        "leftover_p1", "leftover_p2",
        "all_clear_pending_p1", "all_clear_pending_p2",
        "pending_p1_capped", "pending_p2_capped", "net_balance_capped",
        "offboard_p1", "offboard_p2",
        # 新フィールド
        "forecast_p1", "forecast_p2",
        "chain_total_score_p1", "chain_total_score_p2",
        "chain_end_triggered_p1", "chain_end_triggered_p2",
        "score_at_chain_start_p1", "score_at_chain_start_p2",
    ]
    for f in required_fields:
        assert hasattr(snap, f), f"フィールド {f!r} が存在しない"


def test_snapshot_is_frozen() -> None:
    """OjamaAccountSnapshot は frozen dataclass で不変。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    snap = tracker.get_snapshot(t_sec=0.0)
    with pytest.raises((AttributeError, TypeError)):
        snap.pending_p1 = 999  # type: ignore[misc]


def test_forecast_equals_pending() -> None:
    """forecast_p1/p2 が pending_p1/p2 と常に同値。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    _fire_chain(tracker, "p1", chain_score=2100, t_sec=5.0)
    snap = tracker.get_snapshot(t_sec=7.0)
    assert snap.forecast_p1 == snap.pending_p1
    assert snap.forecast_p2 == snap.pending_p2


def test_confidence_fixed() -> None:
    """confidence は常に CONFIDENCE_SCORE_OCR_ONLY を返す。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    snap = tracker.get_snapshot(t_sec=0.0)
    assert snap.confidence == pytest.approx(CONFIDENCE_SCORE_OCR_ONLY)


# ============================
# 11. reset テスト
# ============================

def test_reset_clears_all_ledgers() -> None:
    """reset() で全帳簿がクリアされる。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    _fire_chain(tracker, "p1", chain_score=5000, t_sec=5.0)
    snap_before = tracker.get_snapshot(t_sec=7.0)
    assert snap_before.total_generated_by_p1 > 0

    tracker.reset()
    snap_after = tracker.get_snapshot(t_sec=0.0)
    assert snap_after.pending_p1 == 0
    assert snap_after.pending_p2 == 0
    assert snap_after.forecast_p1 == 0
    assert snap_after.forecast_p2 == 0
    assert snap_after.total_generated_by_p1 == 0
    assert snap_after.total_generated_by_p2 == 0
    assert snap_after.leftover_p1 == 0
    assert snap_after.leftover_p2 == 0


# ============================
# 12. 後方互換 API テスト
# ============================

def test_backward_compat_update_from_score_returns_snapshot() -> None:
    """update_from_score() が OjamaAccountSnapshot を返す(後方互換)。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    snap = tracker.update_from_score(
        score_p1=0, score_p2=0, t_sec=0.0,
    )
    assert isinstance(snap, OjamaAccountSnapshot)


def test_backward_compat_update_from_boards_no_crash() -> None:
    """update_from_boards() がクラッシュしない(後方互換 no-op)。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    empty = _make_empty_board()
    tracker.update_from_boards(board_p1=empty, board_p2=empty)


def test_backward_compat_update_accounting_with_chain_returns_snapshot() -> None:
    """update_accounting_with_chain() が OjamaAccountSnapshot を返す(後方互換)。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    snap = tracker.update_accounting_with_chain(
        t_sec=0.0, chain_p1=False, chain_p2=False,
    )
    assert isinstance(snap, OjamaAccountSnapshot)


# ============================
# 13. offboard / pending_capped テスト
# ============================

def test_offboard_positive_when_forecast_exceeds_on_field_cap() -> None:
    """forecast > ON_FIELD_CAP で offboard > 0、forecast_capped == ON_FIELD_CAP。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 100個相当 (ON_FIELD_CAP=72 を超える)
    score_for_100 = 100 * OJAMA_RATE_STANDARD
    _fire_chain(tracker, "p1", chain_score=score_for_100, t_sec=5.0)
    snap = tracker.get_snapshot(t_sec=7.0)

    assert snap.pending_p2 > ON_FIELD_CAP, (
        f"テスト前提: pending_p2={snap.pending_p2} > ON_FIELD_CAP={ON_FIELD_CAP} が必要"
    )
    assert snap.offboard_p2 > 0
    assert snap.pending_p2_capped == ON_FIELD_CAP
    assert snap.offboard_p2 == snap.pending_p2 - ON_FIELD_CAP


def test_offboard_zero_when_forecast_within_cap() -> None:
    """forecast <= ON_FIELD_CAP では offboard == 0。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    score_for_30 = 30 * OJAMA_RATE_STANDARD
    _fire_chain(tracker, "p1", chain_score=score_for_30, t_sec=5.0)
    snap = tracker.get_snapshot(t_sec=7.0)

    assert snap.offboard_p2 == 0


def test_pending_bounded_by_abs_cap() -> None:
    """forecast は PENDING_ABS_CAP(216) で有界。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    huge_score = PENDING_ABS_CAP * 2 * OJAMA_RATE_STANDARD
    _fire_chain(tracker, "p1", chain_score=huge_score, t_sec=5.0)
    snap = tracker.get_snapshot(t_sec=7.0)
    assert snap.pending_p2 <= PENDING_ABS_CAP


# ============================
# 14. net_balance_capped の 0-1 正規化テスト
# ============================

def test_capped_net_balance_normalization() -> None:
    """net_balance_capped を (x + 72) / 144 で 0-1 正規化できる。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    huge_score = 10000
    _fire_chain(tracker, "p1", chain_score=huge_score, t_sec=5.0)
    snap = tracker.get_snapshot(t_sec=7.0)

    normalized = (snap.net_balance_capped + ON_FIELD_CAP) / (2 * ON_FIELD_CAP)
    assert 0.0 <= normalized <= 1.0


# ============================
# 15. overflow_risk テスト
# ============================

def test_overflow_risk_triggers_above_threshold() -> None:
    """forecast >= threshold で overflow_risk=True。"""
    tracker = OjamaAccountingTracker(overflow_threshold=OJAMA_MAX_DROP_PER_TURN)
    tracker.reset()

    # 60 ojama を 2P に送る (>= 30 threshold)
    score_for_60 = 60 * OJAMA_RATE_STANDARD
    _fire_chain(tracker, "p1", chain_score=score_for_60, t_sec=5.0)
    snap = tracker.get_snapshot(t_sec=7.0)

    assert snap.overflow_risk_p2 is True
    assert snap.overflow_risk_p1 is False


def test_overflow_risk_false_below_threshold() -> None:
    """forecast < threshold では overflow_risk=False。"""
    tracker = OjamaAccountingTracker(overflow_threshold=OJAMA_MAX_DROP_PER_TURN)
    tracker.reset()
    # 10 ojama のみ (< 30 threshold)
    _fire_chain(tracker, "p1", chain_score=700, t_sec=5.0)
    snap = tracker.get_snapshot(t_sec=7.0)
    assert snap.overflow_risk_p2 is False


# ============================
# 16. 定数関係テスト
# ============================

def test_constants_relationship() -> None:
    """定数の関係: ON_FIELD_CAP == PENDING_HARD_CAP < PENDING_ABS_CAP。"""
    assert ON_FIELD_CAP == PENDING_HARD_CAP
    assert PENDING_ABS_CAP > ON_FIELD_CAP
    assert PENDING_ABS_CAP == ON_FIELD_CAP * 3


# ============================
# 17. total_generated / total_offset 累積テスト
# ============================

def test_total_generated_cumulative() -> None:
    """複数連鎖で total_generated が累積される。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 2回連鎖
    _fire_chain(tracker, "p1", chain_score=700, score_before=0, t_sec=5.0)
    _fire_chain(tracker, "p1", chain_score=700, score_before=700, t_sec=10.0)
    snap = tracker.get_snapshot(t_sec=12.0)

    r1 = score_to_ojama(700, prev_leftover=0)
    r2 = score_to_ojama(700, prev_leftover=r1.leftover_score)
    expected_total = r1.ojama_count + r2.ojama_count

    assert snap.total_generated_by_p1 == expected_total, (
        f"total_generated={snap.total_generated_by_p1} != {expected_total}"
    )


def test_total_offset_accumulated_after_cancel() -> None:
    """相殺が発生した分だけ total_offset が記録される。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 2P が G=30 を送る
    _fire_chain(tracker, "p2", chain_score=2100, score_before=0, t_sec=5.0)
    snap_before = tracker.get_snapshot(t_sec=7.0)
    assert snap_before.total_offset_by_p1 == 0

    # 1P が G=30 で相殺
    _fire_chain(tracker, "p1", chain_score=2100, score_before=0, t_sec=10.0)
    snap = tracker.get_snapshot(t_sec=12.0)

    expected_offset = min(30, 30)  # = 30
    assert snap.total_offset_by_p1 == expected_offset, (
        f"total_offset_by_p1={snap.total_offset_by_p1} != {expected_offset}"
    )


# ============================
# 18. score_at_chain_start が snapshot に出る
# ============================

def test_score_at_chain_start_appears_in_snapshot() -> None:
    """on_state_transition 後 score_at_chain_start が snapshot に出る。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 連鎖開始 (スナップ = 1000)
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=1000, t_sec=5.0,
    )
    snap_chain = tracker.get_snapshot(t_sec=5.0)
    assert snap_chain.score_at_chain_start_p1 == 1000, (
        f"score_at_chain_start_p1={snap_chain.score_at_chain_start_p1} != 1000"
    )

    # 連鎖終了後 settle 確定で None に戻る
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=3100, t_sec=7.0,
    )
    # settle 前はまだ score_at_chain_start が残っている(settle 待ち中)
    snap_settle_pending = tracker.get_snapshot(t_sec=7.0)
    assert snap_settle_pending.score_at_chain_start_p1 is not None, (
        "settle待ち中は score_at_chain_start が残っているべき"
    )
    # settle 確定後に None に戻る
    t_end = _settle_score(tracker, "p1", 3100, t_start=7.0)
    snap_end = tracker.get_snapshot(t_end)
    assert snap_end.score_at_chain_start_p1 is None


# ============================
# 19. chain_total_score が snapshot に出る
# ============================

def test_chain_total_score_in_snapshot() -> None:
    """連鎖終了後 chain_total_score_p1 が snapshot に出る。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    chain_total = 4200
    _fire_chain(tracker, "p1", chain_score=chain_total, score_before=0, t_sec=5.0)
    snap = tracker.get_snapshot(t_sec=7.0)

    assert snap.chain_total_score_p1 == chain_total, (
        f"chain_total_score_p1={snap.chain_total_score_p1} != {chain_total}"
    )
    assert snap.chain_total_score_p2 == 0  # 2P は連鎖していない


# ============================
# 20. last_valid_score 回帰テスト (score_at_chain_start None バグ修正)
# ============================

def test_chain_start_uses_last_valid_score_when_frame_score_none() -> None:
    """連鎖開始フレームの score が None でも、直前 STABLE の有効 score を使う。

    シナリオ(実機再現):
        t=5.0: STABLE中 score=800 (有効 → last_valid_score=800 に更新)
        t=5.5: STABLE→CHAIN 遷移, score=None (掛け算式表示開始)
        t=5.6〜7.0: CHAIN中 score=None 継続
        t=7.0: CHAIN→STABLE 遷移, score=None
        t=7.1: STABLE中 score=3500 (score確定 → 遅延確定)
        ⇒ chain_total = 3500 - 800 = 2700
        ⇒ G = 2700 // 70 = 38, leftover = 2700 % 70 = 40
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    expected_chain_total = 2700  # 3500 - 800
    expected_g, expected_leftover = _score_to_ojama_count(expected_chain_total)
    assert expected_g == 38
    assert expected_leftover == 40  # 2700 = 38*70 + 40

    # STABLE 中に有効 score を受信 → last_valid_score が 800 に更新される
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=800, t_sec=5.0,
    )
    # STABLE → CHAIN 遷移, 遷移フレームの score は None (掛け算式)
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.5,
    )
    # 連鎖中ずっと None
    for i in range(5):
        tracker.on_state_transition(
            "p1", BoardState.CHAIN, BoardState.CHAIN, score=None, t_sec=5.6 + i * 0.2,
        )
    # 連鎖終了: CHAIN → STABLE, score=None → 遅延確定待ち
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=None, t_sec=7.0,
    )
    snap_pending = tracker.get_snapshot(t_sec=7.0)
    # 遅延確定前は forecast は 0 のまま
    assert snap_pending.forecast_p2 == 0, (
        f"遅延確定前 forecast_p2={snap_pending.forecast_p2} (まだ 0 であるべき)"
    )

    # score が来て settle 待ちに候補が入る
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=3500, t_sec=7.1,
    )
    # settle 確定: K_SETTLE_FRAMES 連続不変
    t_end = _settle_score(tracker, "p1", 3500, t_start=7.1)
    snap = tracker.get_snapshot(t_end)

    assert snap.forecast_p2 == expected_g, (
        f"forecast_p2={snap.forecast_p2} != {expected_g} "
        f"(chain_total={expected_chain_total})"
    )
    assert snap.leftover_p1 == expected_leftover, (
        f"leftover_p1={snap.leftover_p1} != {expected_leftover}"
    )
    assert snap.chain_total_score_p1 == expected_chain_total, (
        f"chain_total_score_p1={snap.chain_total_score_p1} != {expected_chain_total}"
    )


def test_chain_start_uses_last_valid_score_when_frame_score_present() -> None:
    """遷移フレームの score が非 None でも、last_valid_score(同値)が使われる。

    通常ケース: STABLE 中 score=500、遷移フレームでも score=500 が読める場合。
    ⇒ score_at_chain_start=500 → 連鎖後 score=2500 → chain_total=2000
    ⇒ G = 2000 // 70 = 28, leftover = 2000 % 70 = 40
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    score_before = 500
    chain_total = 2000
    score_after = score_before + chain_total

    expected_g, expected_leftover = _score_to_ojama_count(chain_total)
    assert expected_g == 28
    assert expected_leftover == 40  # 2000 = 28*70 + 40

    # STABLE 中に score=500 受信
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=score_before, t_sec=5.0,
    )
    # STABLE → CHAIN 遷移, 遷移フレームの score=500 が読める
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=score_before, t_sec=5.5,
    )
    # 連鎖終了 → settle 待ち開始
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=score_after, t_sec=7.0,
    )
    # settle 確定
    t_end = _settle_score(tracker, "p1", score_after, t_start=7.0)
    snap = tracker.get_snapshot(t_end)

    assert snap.forecast_p2 == expected_g, (
        f"forecast_p2={snap.forecast_p2} != {expected_g}"
    )
    assert snap.chain_total_score_p1 == chain_total, (
        f"chain_total_score_p1={snap.chain_total_score_p1} != {chain_total}"
    )


def test_chain_start_no_last_valid_score_discards_chain() -> None:
    """last_valid_score も None の場合は過剰計上防止のため連鎖を破棄する。

    試合最初のフレームで即座に連鎖が起きる極端ケース(実機では起こりにくい)。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # reset 直後(last_valid_score=None)で即座に STABLE→CHAIN
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=0.5,
    )
    # 連鎖終了
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=2100, t_sec=2.0,
    )
    snap = tracker.get_snapshot(t_sec=2.0)

    # score_at_chain_start=None → 破棄 → forecast は 0
    assert snap.forecast_p2 == 0, (
        f"last_valid_score=Noneの場合は破棄されるべき: forecast_p2={snap.forecast_p2}"
    )


def test_forecast_equals_chain_total_div_70_no_offset() -> None:
    """forecast増加 == 連鎖合計÷70(相殺なし時)を複数値で検証。

    実機ログで確認された過剰計上ケースを再現:
        chain_total=2100 → G=30
        chain_total=608  → G=8 (端数48繰越)
        chain_total=920  → G=13 (端数10繰越)
    """
    cases = [
        (2100, 30, 0),    # 2100 = 30*70
        (608, 8, 48),     # 608 = 8*70 + 48
        (920, 13, 10),    # 920 = 13*70 + 10
        (880, 12, 40),    # 880 = 12*70 + 40  (実機t=14.6)
        (1407, 20, 7),    # 1407 = 20*70 + 7  (実機t=92.7)
    ]
    for chain_total, expected_g, expected_leftover in cases:
        tracker = OjamaAccountingTracker()
        tracker.reset()
        # STABLE 中 score=0 受信して last_valid_score=0 にセット
        tracker.on_state_transition(
            "p1", BoardState.STABLE, BoardState.STABLE, score=0, t_sec=1.0,
        )
        snap = _fire_chain(
            tracker, "p1", chain_score=chain_total, score_before=0, t_sec=5.0,
        )
        assert snap.forecast_p2 == expected_g, (
            f"chain_total={chain_total}: forecast_p2={snap.forecast_p2} "
            f"!= expected={expected_g}"
        )
        assert snap.leftover_p1 == expected_leftover, (
            f"chain_total={chain_total}: leftover_p1={snap.leftover_p1} "
            f"!= expected={expected_leftover}"
        )
        # chain_total_score_p1 と実際の forecast が整合することを確認
        assert snap.chain_total_score_p1 == chain_total, (
            f"chain_total_score_p1={snap.chain_total_score_p1} != {chain_total}"
        )


def test_forecast_physical_consistency_within_match_score() -> None:
    """送った forecast は「その試合で生成できる量の上限」を超えない物理整合チェック。

    実機ログt=168.5の過剰計上問題:
        修正前: score_at_chain_start=None → chain_total計算不能 → 破棄されるべきが
                旧実装でなぜか過剰計上されていた。
        修正後: last_valid_score=195 → chain_total=608 → G=8 (正常)

    注意: score_to_ojama は elapsed_sec でマージンタイムを適用するため、
    t=160.0 など長時間後は通常より ojama 数が増える。
    このテストは「修正前の None → 破棄」と「修正後の正常計算」を示すため、
    マージンタイムの影響が出ない短い t_sec (< 96秒) を使用する。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # STABLE 中 score=195 (連鎖前の累計) — マージンタイム外(t < 96s)
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=195, t_sec=10.0,
    )
    # STABLE → CHAIN 遷移, score=None (掛け算式)
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=11.0,
    )
    # 連鎖終了 → settle 待ち開始: score=803 (195 + 608)
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=803, t_sec=13.0,
    )
    # settle 確定
    t_end = _settle_score(tracker, "p1", 803, t_start=13.0)
    snap = tracker.get_snapshot(t_end)

    # chain_total = 803 - 195 = 608 → G=8, leftover=48
    expected_g, expected_leftover = _score_to_ojama_count(608)
    assert expected_g == 8
    assert expected_leftover == 48  # 608 = 8*70 + 48

    assert snap.forecast_p2 == expected_g, (
        f"物理整合: forecast_p2={snap.forecast_p2} != {expected_g} "
        f"(修正前は score_at_chain_start=None で破棄 or 過剰計上)"
    )
    assert snap.chain_total_score_p1 == 608, (
        f"chain_total_score_p1={snap.chain_total_score_p1} != 608"
    )
    assert snap.leftover_p1 == expected_leftover, (
        f"leftover_p1={snap.leftover_p1} != {expected_leftover}"
    )


def test_last_valid_score_reset_at_match_boundary() -> None:
    """試合境界(score大幅減少)で last_valid_score もリセットされる。

    前試合の last_valid_score が次試合に引き継がれると連鎖合計が負になりうる。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 前試合: STABLE score=50000
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=50000, t_sec=200.0,
    )
    # 試合境界: score 大幅減少
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=100, t_sec=201.0,
    )
    # 次試合: STABLE → CHAIN, score=None
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=202.0,
    )
    # 連鎖終了: score=500
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=500, t_sec=204.0,
    )
    snap = tracker.get_snapshot(t_sec=204.0)

    # last_valid_score がリセットされていれば score_at_chain_start=None → 破棄
    # (前試合の 50000 が引き継がれていれば chain_total=500-50000=-49500 → chain_total<=0 で破棄)
    # いずれにせよ forecast は 0
    assert snap.forecast_p2 == 0, (
        f"試合境界後の連鎖は過剰計上されないべき: forecast_p2={snap.forecast_p2}"
    )


# ============================
# 21. state 明滅デバウンス (coalesce) 回帰テスト
# ============================

def test_state_flicker_chain_stable_chain_counts_once() -> None:
    """CHAIN→STABLE→CHAIN→STABLE の state 明滅で 1 連鎖 = 1 finalize のみ。

    バグシナリオ:
        1P が score=195 で連鎖開始。
        CHAIN→STABLE (score=803, finalize→G=8) 後、state 明滅で
        直ちに STABLE→CHAIN→STABLE が再発火し、
        2回目 finalize で score_after>803 が使われ G が過剰になる。

    修正後:
        1回目 finalize 後は coalesce window (2.5秒) 内の chain_start は
        score_at_chain_start を上書きしない。
        2回目 CHAIN→STABLE で _finalize_chain_end が呼ばれるが
        score_at_chain_start=None のため破棄される。
        → total_generated_p1 は 8 のみ (過剰なし)。
    """
    from src.ojama_accounting import CHAIN_COALESCE_WINDOW_SEC

    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 連鎖開始前 STABLE で score=195 を受信 (last_valid_score=195)
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=195, t_sec=5.0)

    # 連鎖開始: STABLE→CHAIN, score=None (掛け算式)
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.5)
    snap_start = tracker.get_snapshot(t_sec=5.5)
    assert snap_start.score_at_chain_start_p1 == 195, (
        f"連鎖開始時 score_at_chain_start={snap_start.score_at_chain_start_p1} != 195"
    )

    # 1回目 連鎖終了: CHAIN→STABLE, score=803 → settle 待ち開始
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.STABLE, score=803, t_sec=6.5)
    # settle 確定(dt=0.001 で finalized_at ≒ 6.52)
    t_after_1st = _settle_score(tracker, "p1", 803, t_start=6.5)
    snap_after_1st = tracker.get_snapshot(t_after_1st)
    expected_g1, _ = _score_to_ojama_count(608)
    assert expected_g1 == 8
    assert snap_after_1st.total_generated_by_p1 == 8, (
        f"1回目 settle 確定後 total_generated={snap_after_1st.total_generated_by_p1} != 8"
    )

    # state 明滅: STABLE→CHAIN (coalesce window 内: finalized_at≒6.52, t=6.8 → 差0.28s < 2.5s)
    # → score_at_chain_start は上書きされないはず (coalesce skip)
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=6.8)
    snap_flicker = tracker.get_snapshot(t_sec=6.8)
    # score_at_chain_start は None のまま(coalesce skip で上書き禁止)
    assert snap_flicker.score_at_chain_start_p1 is None, (
        f"coalesce後 score_at_chain_start={snap_flicker.score_at_chain_start_p1} "
        "(None のはず — 上書き禁止)"
    )

    # state 明滅 2回目終了: CHAIN→STABLE, score=803
    # score_at_chain_start=None → settle 後に破棄
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.STABLE, score=803, t_sec=7.0)
    t_after_2nd = _settle_score(tracker, "p1", 803, t_start=7.0)
    snap_after_2nd = tracker.get_snapshot(t_after_2nd)

    # total_generated は 8 のまま (2回目は破棄される)
    assert snap_after_2nd.total_generated_by_p1 == 8, (
        f"state明滅後 total_generated={snap_after_2nd.total_generated_by_p1} "
        f"!= 8 (過剰計上バグ: score_at_chain_start が上書きされた可能性)"
    )
    # forecast_p2 も 8 のまま
    assert snap_after_2nd.forecast_p2 == 8, (
        f"state明滅後 forecast_p2={snap_after_2nd.forecast_p2} != 8"
    )


def test_state_flicker_with_gravity_settle_counts_once() -> None:
    """CHAIN→GRAVITY_SETTLE→STABLE→CHAIN→STABLE の state 明滅でも 1 finalize のみ。

    実機 video_124 t=168 のシナリオ再現(マージンタイム適用外の短い t_sec を使用):
        STABLE→CHAIN (start=195)
        CHAIN→GRAVITY_SETTLE
        GRAVITY_SETTLE→STABLE (score=803, finalize→chain_total=608→G=8)
        ↓ state 明滅
        STABLE→CHAIN (coalesce skip: score_at_chain_start を上書きしてはいけない)
        CHAIN→STABLE (score=2855, もし score_at_chain_start=803 で計算されると
                       chain_total=2052→G=29 過剰計上!
                       修正後: score_at_chain_start=None → 破棄 → G追加0)

    注意: マージンタイム (96秒超で rate 減少) の影響を排除するため
    t_sec=5-7.5 (マージンタイム適用外) を使用する。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 連鎖前 score=195 (マージンタイム適用外: t_sec < 96)
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=195, t_sec=5.0)

    # STABLE→CHAIN (start=195)
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.5)

    # CHAIN→GRAVITY_SETTLE
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.GRAVITY_SETTLE, score=None, t_sec=6.0)

    # GRAVITY_SETTLE→STABLE (score=803) → settle 待ち開始
    tracker.on_state_transition("p1", BoardState.GRAVITY_SETTLE, BoardState.STABLE, score=803, t_sec=6.5)
    # settle 確定(finalized_at ≒ 6.52)
    t_after_1st = _settle_score(tracker, "p1", 803, t_start=6.5)
    snap_1st = tracker.get_snapshot(t_after_1st)
    expected_g, _ = _score_to_ojama_count(608)
    assert expected_g == 8
    assert snap_1st.total_generated_by_p1 == 8, (
        f"1回目 settle 確定後 G={snap_1st.total_generated_by_p1} != 8"
    )

    # state 明滅 (coalesce window 内 < 2.5s): STABLE→CHAIN
    # finalized_at≒6.52, t=7.0 → 差0.48s < 2.5s → coalesce skip
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=7.0)

    # CHAIN→STABLE (score=2855) → settle 待ち → score_at_chain_start=None で破棄
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.STABLE, score=2855, t_sec=7.5)
    t_after_2nd = _settle_score(tracker, "p1", 2855, t_start=7.5)
    snap_2nd = tracker.get_snapshot(t_after_2nd)

    assert snap_2nd.total_generated_by_p1 == 8, (
        f"state明滅後 G={snap_2nd.total_generated_by_p1} != 8 "
        f"(過剰計上: coalesce が効いていない可能性)"
    )


def test_new_chain_after_coalesce_window_fires_normally() -> None:
    """coalesce window 経過後の新規連鎖は正常に finalize される。

    finalize から CHAIN_COALESCE_WINDOW_SEC 秒以上経過した後の
    CHAIN は本物の新規連鎖として処理されるべき。
    """
    from src.ojama_accounting import CHAIN_COALESCE_WINDOW_SEC

    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 1連鎖目
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=0, t_sec=5.0)
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.5)
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.STABLE, score=700, t_sec=6.0)
    # settle 確定(finalized_at ≒ 6.02)
    t_1st_end = _settle_score(tracker, "p1", 700, t_start=6.0)
    snap_1st = tracker.get_snapshot(t_1st_end)
    g1, _ = _score_to_ojama_count(700)
    assert snap_1st.total_generated_by_p1 == g1

    # coalesce window (2.5s) + 余裕 を経過してから 2連鎖目
    # finalized_at ≒ 6.02 + CHAIN_COALESCE_WINDOW_SEC + 0.5 = 9.02 以降で開始
    t_2nd_start = t_1st_end + CHAIN_COALESCE_WINDOW_SEC + 0.5
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=700, t_sec=t_2nd_start)
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=t_2nd_start + 0.5)
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.STABLE, score=1400, t_sec=t_2nd_start + 1.5)
    t_2nd_end = _settle_score(tracker, "p1", 1400, t_start=t_2nd_start + 1.5)
    snap_2nd = tracker.get_snapshot(t_2nd_end)

    # 2連鎖目も正常に加算: chain_total=1400-700=700 → G=g1
    g2, _ = _score_to_ojama_count(700)
    expected_total = g1 + g2
    assert snap_2nd.total_generated_by_p1 == expected_total, (
        f"coalesce window後 total_generated={snap_2nd.total_generated_by_p1} "
        f"!= {expected_total}"
    )


def test_chain_end_pending_during_flicker_coalesces() -> None:
    """chain_end_pending 中に state 明滅 (非CHAIN→CHAIN) が来ても
    score_at_chain_start は上書きされず、遅延確定が正しく機能する。

    シナリオ:
        STABLE→CHAIN (start=195)
        CHAIN→STABLE, score=None → pending 開始
        STABLE→CHAIN (明滅): pending 中 → score_at_chain_start=195 維持
        score=803 が来る → pending finalize: chain_total=608→G=8 (正常)
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=195, t_sec=5.0)
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.5)

    # 連鎖終了 score=None → pending 開始
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.STABLE, score=None, t_sec=6.5)
    snap_pending = tracker.get_snapshot(t_sec=6.5)
    assert snap_pending.total_generated_by_p1 == 0

    # pending 中に state 明滅: STABLE→CHAIN
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=6.6)
    # score=803 が来る → settle 待ちに候補 803 が入る
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.STABLE, score=803, t_sec=7.0)
    # settle 確定
    t_end = _settle_score(tracker, "p1", 803, t_start=7.0)
    snap = tracker.get_snapshot(t_end)

    expected_g, _ = _score_to_ojama_count(608)
    assert expected_g == 8
    assert snap.total_generated_by_p1 == 8, (
        f"pending中state明滅後 total_generated={snap.total_generated_by_p1} != 8"
    )


# ============================
# 22. マージンタイム試合相対 回帰テスト (2026-06-10 修正)
# ============================

def test_margin_time_uses_match_relative_not_clip_relative() -> None:
    """マージンタイムはクリップ相対経過秒ではなく試合相対経過秒で計算される。

    バグシナリオ (video_124_4min t=168s):
        クリップ先頭から168秒経過したところで連鎖(chain_total=608)が発火。
        旧実装: _match_start_sec=None → elapsed=168s → rate=16 → G=38 (過剰)
        修正後: match_start=first_score_time → elapsed≒20s → rate=70 → G=8 (正)

    このテストでは:
        - クリップ先頭 t=0.0 で reset()
        - 試合開始を t=148.0 で score を最初に観測 → _match_start_sec=148.0
        - t=168.0 で chain_total=608 を finalize
        - elapsed = 168.0 - 148.0 = 20.0秒 < 96秒 → マージンタイム非適用 → rate=70
        - G = 608 // 70 = 8 (正)
        - クリップ相対なら elapsed=168秒 → rate=70*(0.75^4)≒22 → G=27 (過剰)
    """
    from src.scoring import MARGIN_TIME_START_SEC, compute_effective_rate

    tracker = OjamaAccountingTracker()
    tracker.reset()  # _match_start_sec=None

    # クリップ先頭 t=0.0 から試合が始まらず、t=148.0 で最初の score を観測
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=195, t_sec=148.0,
    )
    # → _match_start_sec = 148.0 が設定されるはず

    # t=168.0 (クリップ相対168秒、試合相対20秒) で連鎖開始
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=168.0,
    )
    # t=169.0 で連鎖終了 → settle 待ち開始: score=803 (195+608)
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=803, t_sec=169.0,
    )
    # settle 確定(elapsed は settle 確定時刻で計算。dt=0.001 で差は無視できる)
    t_end = _settle_score(tracker, "p1", 803, t_start=169.0)
    snap = tracker.get_snapshot(t_end)

    # chain_total=608、elapsed≒(169-148)=21秒 → マージンタイム非適用 → rate=70 → G=8
    expected_g, expected_leftover = _score_to_ojama_count(608)
    assert expected_g == 8, f"前提確認: score_to_ojama(608)={expected_g} (8であるべき)"
    assert expected_leftover == 48

    assert snap.forecast_p2 == expected_g, (
        f"試合相対elapsed=21s → マージンタイム非適用 → G=8 のはず。"
        f"forecast_p2={snap.forecast_p2} (旧実装なら elapsed=169s→rate減少→G過剰)"
    )
    assert snap.chain_total_score_p1 == 608, (
        f"chain_total_score_p1={snap.chain_total_score_p1} != 608"
    )


def test_margin_time_clip_relative_would_give_wrong_result() -> None:
    """クリップ相対経過秒を使うと過剰計上になることを示す対照テスト。

    elapsed=169秒はマージンタイム域(>=96秒)に入りrateが低下する。
    修正後は elapsed=試合相対なのでこの過剰は起きないことを確認する。
    """
    from src.scoring import MARGIN_TIME_START_SEC, compute_effective_rate

    # elapsed=169秒だとrateが低下する(マージンタイム適用)ことを確認
    clip_elapsed = 169.0
    rate_clip = compute_effective_rate(clip_elapsed)
    assert rate_clip < 70, (
        f"elapsed={clip_elapsed}s ではマージンタイム適用でrate={rate_clip}<70 のはず"
    )

    # 修正後の試合相対elapsed(約21秒)ではrateが70のままであることを確認
    match_elapsed = 21.0
    rate_match = compute_effective_rate(match_elapsed)
    assert rate_match == 70, (
        f"elapsed={match_elapsed}s(試合相対) ではマージンタイム非適用でrate=70 のはず"
        f" (実際: {rate_match})"
    )

    # クリップ相対elapsedで計算したら G が過剰になることを示す
    from src.scoring import score_to_ojama
    r_clip = score_to_ojama(608, prev_leftover=0, elapsed_sec=clip_elapsed)
    r_match = score_to_ojama(608, prev_leftover=0, elapsed_sec=match_elapsed)
    assert r_clip.ojama_count > r_match.ojama_count, (
        f"クリップ相対({clip_elapsed}s)のほうが試合相対({match_elapsed}s)より"
        f"多い予告を出すはず: clip={r_clip.ojama_count}, match={r_match.ojama_count}"
    )
    # 修正後(試合相対)の正解値が8個であることを確認
    assert r_match.ojama_count == 8, (
        f"試合相対elapsed={match_elapsed}sでchain_total=608→G=8 のはず"
        f" (実際: {r_match.ojama_count})"
    )


def test_margin_time_activates_after_96s_within_single_match() -> None:
    """1試合内で96秒を超えた場合はマージンタイムが正しく適用される。

    単一試合で96秒を超えた長試合では、マージンタイムが発動して
    rate が低下し、生成 ojama 数が増加することを確認する。

    シナリオ:
        t=0 でリセット、t=1.0 で最初の score → _match_start_sec=1.0
        t=100.0 (試合相対99秒) で chain_total=608 発火
        elapsed=99秒 > 96秒 → マージンタイム適用 → rate < 70 → G > 8
    """
    from src.scoring import compute_effective_rate

    tracker = OjamaAccountingTracker()
    tracker.reset()

    # t=1.0 で最初の score → _match_start_sec=1.0
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=195, t_sec=1.0,
    )
    # t=100.0 (試合相対99秒) で連鎖開始
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=100.0,
    )
    # t=101.0 で連鎖終了 → settle 待ち開始: score=803 (195+608)
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=803, t_sec=101.0,
    )
    # settle 確定(dt=0.001 で elapsed ≒ 101.02s。マージンタイム区域なので問題なし)
    t_end = _settle_score(tracker, "p1", 803, t_start=101.0)
    snap = tracker.get_snapshot(t_end)

    # elapsed = t_end - 1 ≒ 101.02 - 1 = 100.02秒 > 96秒 → マージンタイム適用
    elapsed = t_end - 1.0
    rate = compute_effective_rate(elapsed)
    assert rate < 70, f"elapsed={elapsed}s でマージンタイム適用後 rate={rate}<70 のはず"

    from src.scoring import score_to_ojama
    expected_result = score_to_ojama(608, prev_leftover=0, elapsed_sec=elapsed)
    assert expected_result.ojama_count > 8, (
        f"elapsed={elapsed}sではマージンタイム適用でG>8のはず: {expected_result.ojama_count}"
    )

    assert snap.forecast_p2 == expected_result.ojama_count, (
        f"長試合(試合相対{elapsed}s)ではマージンタイム適用: "
        f"forecast_p2={snap.forecast_p2} != {expected_result.ojama_count}"
    )


def test_margin_time_resets_at_match_boundary() -> None:
    """試合境界(score大幅減少)で _match_start_sec がリセットされる。

    前試合が長時間続いてマージンタイム域に入っていても、次試合の冒頭では
    マージンタイムが適用されない(elapsed が試合相対でリセットされる)。

    シナリオ:
        前試合: t=0 開始、t=100 (試合相対100秒) で境界
        次試合: t=100 で score リセット → _match_start_sec=100
                t=110 (試合相対10秒) で chain_total=608 発火
                elapsed=10秒 < 96秒 → rate=70 → G=8 (正)
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 前試合開始: t=0.5 で最初の score → _match_start_sec=0.5
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=5000, t_sec=0.5,
    )

    # t=100 で試合境界(score大幅減少、前試合終了)
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=5000, t_sec=99.5,
    )
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=100, t_sec=100.0,
        # 5000→100: SCORE_RESET_THRESHOLD(500)超え → _match_start_sec=100.0 に更新
    )

    # 次試合: t=102 で最初の score → _match_start_sec が設定される
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=195, t_sec=102.0,
    )

    # t=110 (試合相対8秒) で連鎖開始
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=110.0,
    )
    # t=112 で連鎖終了 → settle 待ち開始: score=803 (195+608)
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=803, t_sec=112.0,
    )
    # settle 確定
    t_end = _settle_score(tracker, "p1", 803, t_start=112.0)
    snap = tracker.get_snapshot(t_end)

    # 次試合の elapsed ≒ t_end - 100 ≒ 12.02秒 → マージンタイム非適用 → G=8
    expected_g, _ = _score_to_ojama_count(608)
    assert expected_g == 8

    assert snap.forecast_p2 == expected_g, (
        f"次試合冒頭(試合相対12秒)ではマージンタイム非適用 → G=8 のはず。"
        f"forecast_p2={snap.forecast_p2} (旧実装では前試合継続elapsed=112sで過剰)"
    )


# ============================
# 23. バグ修正 回帰テスト (Phase 1a: 2026-07-03)
# ============================

def test_chain_end_triggered_reset_at_score_boundary() -> None:
    """試合境界(score大幅減少)で chain_end_triggered が False にリセットされる。

    修正前: _reset_side_boundary に chain_end_triggered=False がなかった。
    そのため前試合の triggered=True が次試合に持ち越され、
    overlay で「連鎖終了」が最大 30 秒以上誤表示される。

    シナリオ:
        1P が連鎖を撃つ → chain_end_triggered_p1=True
        試合境界: score 大幅減少
        次試合では chain_end_triggered_p1=False になるべき
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 1P が連鎖を撃つ → triggered=True になる
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=0, t_sec=5.0,
    )
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=2100, t_sec=7.0,
    )
    snap_after_chain = tracker.get_snapshot(t_sec=7.0)
    assert snap_after_chain.chain_end_triggered_p1 is True, "前提: 連鎖後 triggered=True"

    # 試合境界: score 大幅減少
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=2100, t_sec=8.0,
    )
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=100, t_sec=9.0,
        # 2100→100: SCORE_RESET_THRESHOLD(500)超え → _reset_side_boundary
    )
    snap_after_boundary = tracker.get_snapshot(t_sec=9.0)

    assert snap_after_boundary.chain_end_triggered_p1 is False, (
        f"試合境界後 chain_end_triggered_p1={snap_after_boundary.chain_end_triggered_p1} "
        "should be False (修正前は True が持ち越されていた)"
    )


def test_chain_end_triggered_reset_at_menu_transition() -> None:
    """MENU 遷移で chain_end_triggered が False にリセットされる。

    MENU 遷移の _reset_side_boundary でも同様に chain_end_triggered をリセットする。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 1P が連鎖 → triggered=True
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=0, t_sec=5.0,
    )
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=2100, t_sec=7.0,
    )
    snap_chain = tracker.get_snapshot(t_sec=7.0)
    assert snap_chain.chain_end_triggered_p1 is True

    # MENU 遷移 → _reset_side_boundary が呼ばれる
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.MENU, score=2100, t_sec=10.0,
    )
    snap_menu = tracker.get_snapshot(t_sec=10.0)

    assert snap_menu.chain_end_triggered_p1 is False, (
        f"MENU遷移後 chain_end_triggered_p1={snap_menu.chain_end_triggered_p1} "
        "should be False"
    )


def test_chain_total_min_score_guard_discards_tiny_chain() -> None:
    """chain_total < CHAIN_TOTAL_MIN_SCORE (=40) の極小連鎖は破棄される。

    score OCR の端数誤読(例: score差=1~39)が幻のお邪魔を生成しないよう
    下限ガードで破棄し、leftover の誤累積も防ぐ。
    """
    from src.ojama_accounting import CHAIN_TOTAL_MIN_SCORE
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # STABLE で score=1000 を受信 (last_valid_score=1000)
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=1000, t_sec=5.0,
    )
    # STABLE→CHAIN (start=1000)
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.5,
    )
    # 連鎖終了: score=1001 → chain_total=1 < CHAIN_TOTAL_MIN_SCORE(=40) → 破棄
    chain_total_tiny = 1
    assert chain_total_tiny < CHAIN_TOTAL_MIN_SCORE, "前提確認"
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=1000 + chain_total_tiny, t_sec=7.0,
    )
    snap = tracker.get_snapshot(t_sec=7.0)

    assert snap.forecast_p2 == 0, (
        f"極小chain_total={chain_total_tiny}は破棄: forecast_p2={snap.forecast_p2} (0になるべき)"
    )
    # leftover も汚染されていないこと
    assert snap.leftover_p1 == 0, (
        f"極小chain破棄でleftover汚染なし: leftover_p1={snap.leftover_p1} (0になるべき)"
    )
    # total_generated も増えていないこと
    assert snap.total_generated_by_p1 == 0, (
        f"極小chain破棄でtotal_generated汚染なし: {snap.total_generated_by_p1}"
    )


def test_chain_total_min_score_guard_allows_small_valid_chain() -> None:
    """chain_total >= CHAIN_TOTAL_MIN_SCORE の正当な小連鎖は破棄されない。

    1連鎖最小スコア = CHAIN_TOTAL_MIN_SCORE(=40) 以上は正常に処理される。
    """
    from src.ojama_accounting import CHAIN_TOTAL_MIN_SCORE
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 正当な小連鎖: chain_total = CHAIN_TOTAL_MIN_SCORE(=40) ちょうど
    chain_total = CHAIN_TOTAL_MIN_SCORE
    expected_g, expected_leftover = _score_to_ojama_count(chain_total)

    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=0, t_sec=5.0,
    )
    snap = _fire_chain(tracker, "p1", chain_score=chain_total, score_before=0, t_sec=5.5)

    # chain_total=40 → score_to_ojama(40) = ojama=0, leftover=40
    assert expected_g == 0  # 40点はお邪魔1個(70点)未満なのでお邪魔生成なし
    assert expected_leftover == 40

    # 破棄されず leftover に正しく積まれる (お邪魔は生成されないが leftover は増える)
    assert snap.forecast_p2 == 0, (
        f"chain_total=40: forecast_p2={snap.forecast_p2} (0が正しい: 40<70でお邪魔なし)"
    )
    assert snap.leftover_p1 == expected_leftover, (
        f"chain_total=40: leftover_p1={snap.leftover_p1} != {expected_leftover} (正当な小連鎖は破棄されない)"
    )


def test_tiny_score_diff_leftover_no_contamination() -> None:
    """score差=1 の極小連鎖が leftover を汚染しないことを確認。

    score OCR 端数誤読の典型ケース(差=1)で leftover が 0 のまま維持されるか。
    その後の正常連鎖の計算が正しいことも確認。
    """
    from src.ojama_accounting import CHAIN_TOTAL_MIN_SCORE
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 極小連鎖1回目 (score差=1, settle後に破棄される)
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=500, t_sec=5.0,
    )
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.5,
    )
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=501, t_sec=7.0,
    )
    # settle 確定(chain_total=1 < CHAIN_TOTAL_MIN_SCORE=40 → 破棄)
    t_tiny_end = _settle_score(tracker, "p1", 501, t_start=7.0)
    snap_tiny = tracker.get_snapshot(t_tiny_end)
    assert snap_tiny.leftover_p1 == 0, "極小連鎖後の leftover は 0 のまま"
    assert snap_tiny.forecast_p2 == 0, "極小連鎖後の forecast は 0 のまま"

    # 正常連鎖 (chain_total=700, 10個)
    expected_g, expected_leftover = _score_to_ojama_count(700)
    assert expected_g == 10
    assert expected_leftover == 0

    # coalesce window を超えるため十分時間を空けて発火
    # t_tiny_end ≒ 7.02。coalesce window(2.5s) + α = 10.0s で安全
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=501, t_sec=10.0,
    )
    snap = _fire_chain(tracker, "p1", chain_score=700, score_before=501, t_sec=10.5)

    assert snap.forecast_p2 == expected_g, (
        f"正常連鎖後 forecast_p2={snap.forecast_p2} != {expected_g} "
        "(leftover汚染があると +1 余分になる可能性)"
    )
    assert snap.leftover_p1 == expected_leftover, (
        f"正常連鎖後 leftover_p1={snap.leftover_p1} != {expected_leftover}"
    )


# ============================
# 24. 連鎖中フリッカー score_at_chain_start 上書きバグ 回帰テスト (2026-07-06 修正)
# ============================

def test_mid_chain_gravity_settle_to_chain_does_not_overwrite_score_at_start() -> None:
    """連鎖途中の GRAVITY_SETTLE→CHAIN フリッカーで score_at_chain_start が上書きされない。

    診断済み実例(t=142連鎖, 1P):
        スコア段階上昇: 465→505→825→1465→3565
        STABLE→CHAIN(start=465) → 連鎖中 GRAVITY_SETTLE→CHAIN フリッカー(score=1465)
        修正前: score_at_chain_start=1465 に上書き → chain_total=3565-1465=2100(30個) 過少
        修正後: score_at_chain_start=465 維持 → chain_total=3565-465=3100(44個) 正確
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 連鎖前 score=465 (last_valid_score=465)
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=465, t_sec=5.0,
    )
    # 連鎖開始: STABLE→CHAIN (score_at_chain_start=465 に設定)
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.5,
    )
    snap_start = tracker.get_snapshot(t_sec=5.5)
    assert snap_start.score_at_chain_start_p1 == 465, (
        f"前提: score_at_chain_start={snap_start.score_at_chain_start_p1} (465 であるべき)"
    )

    # 連鎖途中: CHAIN→GRAVITY_SETTLE
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.GRAVITY_SETTLE, score=1465, t_sec=6.0,
    )
    # 連鎖途中フリッカー: GRAVITY_SETTLE→CHAIN (score=1465)
    # 修正前: score_at_chain_start=1465 に上書きされていた
    # 修正後: score_at_chain_start=465 のまま維持されるべき
    tracker.on_state_transition(
        "p1", BoardState.GRAVITY_SETTLE, BoardState.CHAIN, score=1465, t_sec=6.2,
    )
    snap_mid = tracker.get_snapshot(t_sec=6.2)
    assert snap_mid.score_at_chain_start_p1 == 465, (
        f"連鎖中GRAVITY_SETTLE→CHAINフリッカー後 score_at_chain_start="
        f"{snap_mid.score_at_chain_start_p1} (465 のまま維持されるべき、修正前は1465に上書き)"
    )

    # 連鎖終了: CHAIN→STABLE (score=3565) → settle 待ち開始
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=3565, t_sec=7.0,
    )
    # settle 確定
    t_end = _settle_score(tracker, "p1", 3565, t_start=7.0)
    snap = tracker.get_snapshot(t_end)

    # chain_total = 3565 - 465 = 3100 → G=44, leftover=20
    expected_chain_total = 3100
    expected_g, expected_leftover = _score_to_ojama_count(expected_chain_total)
    assert expected_g == 44
    assert expected_leftover == 20  # 3100 = 44*70 + 20

    assert snap.chain_total_score_p1 == expected_chain_total, (
        f"chain_total_score_p1={snap.chain_total_score_p1} != {expected_chain_total} "
        f"(修正前は1465で上書きされ chain_total=2100になっていた)"
    )
    assert snap.forecast_p2 == expected_g, (
        f"forecast_p2={snap.forecast_p2} != {expected_g} "
        f"(修正前はG=30、修正後はG=44)"
    )


def test_mid_chain_flicker_new_chain_after_finalize_resets_normally() -> None:
    """finalize 後に score_at_chain_start=None に戻り、次の本物の連鎖で再設定される。

    finalize → score_at_chain_start=None → coalesce window 経過 → 次連鎖で再設定。
    エッジケース: 今回の修正が「次連鎖の開始スコア設定」を妨げないことを確認。
    """
    from src.ojama_accounting import CHAIN_COALESCE_WINDOW_SEC

    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 1連鎖目: score_before=200, chain_total=700
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=200, t_sec=5.0)
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.5)
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.STABLE, score=900, t_sec=6.5)
    # settle 確定後は score_at_chain_start=None に戻る
    t_1st_end = _settle_score(tracker, "p1", 900, t_start=6.5)
    snap_1st = tracker.get_snapshot(t_1st_end)

    # finalize(settle確定)後は score_at_chain_start=None に戻る
    assert snap_1st.score_at_chain_start_p1 is None, (
        f"finalize後 score_at_chain_start={snap_1st.score_at_chain_start_p1} (None であるべき)"
    )

    # coalesce window (2.5s) + 余裕 を経過させてから 2連鎖目
    # t_1st_end ≒ 6.52。coalesce window(2.5s) + 0.5 = t_2nd ≒ 9.52
    t_2nd = t_1st_end + CHAIN_COALESCE_WINDOW_SEC + 0.5
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=900, t_sec=t_2nd)
    # 2連鎖目開始: score_at_chain_start=900 が設定されるべき
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=t_2nd + 0.5)
    snap_2nd_start = tracker.get_snapshot(t_sec=t_2nd + 0.5)

    assert snap_2nd_start.score_at_chain_start_p1 == 900, (
        f"2連鎖目開始時 score_at_chain_start={snap_2nd_start.score_at_chain_start_p1} "
        f"(900 に設定されるべき — 今回修正が次連鎖を妨げないことを確認)"
    )

    # 2連鎖目終了
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.STABLE, score=1600, t_sec=t_2nd + 2.0)
    t_2nd_end = _settle_score(tracker, "p1", 1600, t_start=t_2nd + 2.0)
    snap_2nd = tracker.get_snapshot(t_2nd_end)

    # chain_total = 1600 - 900 = 700 → 1連鎖目と同じG
    g1, _ = _score_to_ojama_count(700)
    g2, _ = _score_to_ojama_count(700)
    assert snap_2nd.total_generated_by_p1 == g1 + g2, (
        f"total_generated={snap_2nd.total_generated_by_p1} != {g1 + g2}"
    )


def test_mid_chain_ojama_fall_to_chain_does_not_overwrite_score_at_start() -> None:
    """連鎖途中の OJAMA_FALL→CHAIN フリッカーでも score_at_chain_start が上書きされない。

    GRAVITY_SETTLE 以外の非CHAIN state からの CHAIN 遷移フリッカーもケアされることを確認。
    OJAMA_FALL は _chain_states に含まれないため _is_chain_start=True になるが、
    _already_started=True でスキップされるべき。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 連鎖前 score=300
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=300, t_sec=5.0,
    )
    # 連鎖開始 (score_at_chain_start=300)
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.5,
    )
    snap_start = tracker.get_snapshot(t_sec=5.5)
    assert snap_start.score_at_chain_start_p1 == 300

    # 連鎖途中: 任意の非CHAIN state (OJAMA_FALL に相当するシミュレーション)
    # BoardState.OJAMA_FALL が存在すれば使うが、ここでは GRAVITY_SETTLE で代用
    # 重要なのは「非CHAIN→CHAIN の _is_chain_start=True になるケース全般」
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.GRAVITY_SETTLE, score=800, t_sec=6.0,
    )
    tracker.on_state_transition(
        "p1", BoardState.GRAVITY_SETTLE, BoardState.CHAIN, score=800, t_sec=6.1,
    )
    snap_mid = tracker.get_snapshot(t_sec=6.1)

    # score_at_chain_start は 300 のまま維持されるべき
    assert snap_mid.score_at_chain_start_p1 == 300, (
        f"GRAVITY_SETTLE→CHAINフリッカー後 score_at_chain_start="
        f"{snap_mid.score_at_chain_start_p1} (300 維持されるべき)"
    )

    # 連鎖終了 (score=2100) → settle 待ち開始
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=2100, t_sec=7.0,
    )
    # settle 確定
    t_end = _settle_score(tracker, "p1", 2100, t_start=7.0)
    snap = tracker.get_snapshot(t_end)

    # chain_total = 2100 - 300 = 1800 → G=25, leftover=50
    expected_chain_total = 1800
    expected_g, expected_leftover = _score_to_ojama_count(expected_chain_total)
    assert expected_g == 25
    assert expected_leftover == 50  # 1800 = 25*70 + 50

    assert snap.chain_total_score_p1 == expected_chain_total, (
        f"chain_total_score_p1={snap.chain_total_score_p1} != {expected_chain_total}"
    )
    assert snap.forecast_p2 == expected_g, (
        f"forecast_p2={snap.forecast_p2} != {expected_g}"
    )


# ============================
# 25. MENU 多重発火防止 回帰テスト (2026-07-06 エッジトリガ化修正)
# ============================

def test_menu_edge_trigger_resets_only_once() -> None:
    """MENU 継続中は _reset_side_boundary が 1 回だけ呼ばれる (多重発火防止)。

    修正前: curr_state==MENU の毎フレームでリセットが発火 (video_124 で 22 回)。
    修正後: prev_state != MENU のエッジでのみ 1 回だけリセット。

    検証: MENU に入る前に forecast を積み、MENU 継続フレームを複数流した後も
    forecast は 0 に 1 回だけリセットされ、それ以降はゼロ維持のまま。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 2P が G=30 を 1P へ送る (forecast_p1=30)
    _fire_chain(tracker, "p2", chain_score=2100, score_before=0, t_sec=5.0)
    snap_before = tracker.get_snapshot(t_sec=7.0)
    assert snap_before.forecast_p1 == 30, "前提: forecast_p1=30"

    # MENU 入場エッジ (prev=STABLE, curr=MENU) → 1 回目リセット
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.MENU, score=5000, t_sec=10.0,
    )
    snap_edge = tracker.get_snapshot(t_sec=10.0)
    assert snap_edge.forecast_p1 == 0, (
        f"MENU入場エッジでリセット: forecast_p1={snap_edge.forecast_p1} (0になるべき)"
    )

    # MENU 継続 (prev=MENU, curr=MENU) × 30 フレーム: 追加リセット発火なし
    # (リセット済みのため 0 維持が期待値)
    for i in range(30):
        tracker.on_state_transition(
            "p1", BoardState.MENU, BoardState.MENU, score=None, t_sec=10.1 + i * 0.033,
        )
    snap_after_menu = tracker.get_snapshot(t_sec=11.1)
    assert snap_after_menu.forecast_p1 == 0, (
        f"MENU継続後も forecast_p1=0 維持: {snap_after_menu.forecast_p1}"
    )


def test_menu_edge_trigger_fires_on_entry_not_continuation() -> None:
    """MENU→MENU (継続) はリセットを発火しない。非MENU→MENU (入場) のみ発火。

    2試合: 試合1でforecast積む → MENU → MENU継続 → 次試合開始
    次試合冒頭でforecastが正しく0であることを確認 (リセット1回で十分)。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 試合1: 2P が G=10 を 1P へ送る
    _fire_chain(tracker, "p2", chain_score=700, score_before=0, t_sec=5.0)
    snap_trial1 = tracker.get_snapshot(t_sec=7.0)
    g10, _ = _score_to_ojama_count(700)
    assert snap_trial1.forecast_p1 == g10, f"前提: forecast_p1={g10}"

    # MENU 入場 (非MENU→MENU)
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.MENU, score=None, t_sec=20.0,
    )
    # MENU 継続 5 フレーム
    for i in range(5):
        tracker.on_state_transition(
            "p1", BoardState.MENU, BoardState.MENU, score=None, t_sec=20.1 + i * 0.033,
        )

    # 試合2 開始 (MENU→STABLE)
    tracker.on_state_transition(
        "p1", BoardState.MENU, BoardState.STABLE, score=0, t_sec=22.0,
    )
    snap_trial2 = tracker.get_snapshot(t_sec=22.0)

    # MENU 入場時に 1 回リセットされているので forecast=0
    assert snap_trial2.forecast_p1 == 0, (
        f"試合2冒頭 forecast_p1={snap_trial2.forecast_p1} (0になるべき, 前試合forecast漏れなし)"
    )
    # leftover もリセット済み
    assert snap_trial2.leftover_p1 == 0, (
        f"試合2冒頭 leftover_p1={snap_trial2.leftover_p1} (0になるべき)"
    )


def test_menu_forecast_zero_at_next_match_start() -> None:
    """MENU 経由で試合境界をまたいでも前試合の forecast が次試合に漏れない。

    video_124 で観測された多重発火ケースの縮約再現:
        試合終了 → MENU (30フレーム) → 次試合開始
        次試合冒頭で forecast = 0 (前試合予告の漏れなし)
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 前試合: 1P に G=50 相当の予告が溜まっている
    _fire_chain(tracker, "p2", chain_score=3500, score_before=0, t_sec=10.0)
    snap_pre = tracker.get_snapshot(t_sec=12.0)
    expected_g, _ = _score_to_ojama_count(3500)
    assert snap_pre.forecast_p1 == expected_g, f"前提: forecast_p1={expected_g}"

    # 試合境界 → MENU 入場
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.MENU, score=10000, t_sec=50.0,
    )
    # MENU 継続 30 フレーム (約 1 秒 @ 30fps)
    for i in range(30):
        tracker.on_state_transition(
            "p1", BoardState.MENU, BoardState.MENU, score=None, t_sec=50.1 + i * 0.033,
        )

    # 次試合 STABLE 開始
    tracker.on_state_transition(
        "p1", BoardState.MENU, BoardState.STABLE, score=0, t_sec=52.0,
    )
    snap_next = tracker.get_snapshot(t_sec=52.0)

    assert snap_next.forecast_p1 == 0, (
        f"次試合冒頭 forecast_p1={snap_next.forecast_p1} (前試合予告漏れなし, 0になるべき)"
    )
    assert snap_next.leftover_p1 == 0, (
        f"次試合冒頭 leftover_p1={snap_next.leftover_p1} (0になるべき)"
    )


# ============================
# 26. last_stable_score 差分方式 回帰テスト (2026-07-06 新方針)
# ============================

def test_last_stable_score_used_as_chain_total_base() -> None:
    """last_stable_score (連鎖間の落ち着いた STABLE score) を基準に chain_total が算出される。

    新方針:
        - 連鎖間の STABLE フレームで score を読んだとき last_stable_score を更新する。
        - finalize 時は last_stable_score を score_start として使用する。
        - これにより score_at_chain_start の取り違えによる計算誤差を排除する。

    シナリオ:
        t=5.0: STABLE score=100 → last_stable_score=100
        t=5.5: STABLE→CHAIN (score_at_chain_start=100)
        t=7.0: CHAIN→STABLE score=3100 → chain_total=3100-100=3000 → G=42, leftover=60
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # STABLE フレームで score=100 → last_stable_score=100 に設定
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=100, t_sec=5.0,
    )
    # 連鎖開始
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.5,
    )
    # 連鎖終了 → settle 待ち開始
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=3100, t_sec=7.0,
    )
    # settle 確定
    t_end = _settle_score(tracker, "p1", 3100, t_start=7.0)
    snap = tracker.get_snapshot(t_end)

    expected_chain_total = 3000  # 3100 - 100
    expected_g, expected_leftover = _score_to_ojama_count(expected_chain_total)
    assert expected_g == 42
    assert expected_leftover == 60  # 3000 = 42*70 + 60

    assert snap.chain_total_score_p1 == expected_chain_total, (
        f"chain_total_score_p1={snap.chain_total_score_p1} != {expected_chain_total}"
    )
    assert snap.forecast_p2 == expected_g, (
        f"forecast_p2={snap.forecast_p2} != {expected_g}"
    )
    assert snap.leftover_p1 == expected_leftover, (
        f"leftover_p1={snap.leftover_p1} != {expected_leftover}"
    )


def test_last_stable_score_not_updated_during_chain() -> None:
    """連鎖中 (chain_active=True) は last_stable_score が更新されない。

    連鎖中に score が読めても last_stable_score を汚さないことを確認する。
    GRAVITY_SETTLE→STABLE で score が来たときに last_stable_score が
    連鎖前の基準値のまま維持されることが重要。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 連鎖前 STABLE: score=200 → last_stable_score=200
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=200, t_sec=5.0,
    )
    # 連鎖開始
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.5,
    )
    # 連鎖途中: CHAIN→GRAVITY_SETTLE (score=800) → 連鎖中なので last_stable_score は更新されない
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.GRAVITY_SETTLE, score=800, t_sec=6.0,
    )
    # 連鎖終了: GRAVITY_SETTLE→STABLE (score=2200) → settle 待ち開始
    # chain_total = 2200 - 200(last_stable_score) = 2000 となるべき
    # もし last_stable_score が 800 に上書きされていたら chain_total=1400 になる
    tracker.on_state_transition(
        "p1", BoardState.GRAVITY_SETTLE, BoardState.STABLE, score=2200, t_sec=7.0,
    )
    # settle 確定
    t_end = _settle_score(tracker, "p1", 2200, t_start=7.0)
    snap = tracker.get_snapshot(t_end)

    expected_chain_total = 2000  # 2200 - 200 (not 800)
    expected_g, expected_leftover = _score_to_ojama_count(expected_chain_total)
    assert expected_g == 28
    assert expected_leftover == 40  # 2000 = 28*70 + 40

    assert snap.chain_total_score_p1 == expected_chain_total, (
        f"chain_total_score_p1={snap.chain_total_score_p1} != {expected_chain_total} "
        f"(連鎖中 score=800 で last_stable_score が汚染された可能性)"
    )
    assert snap.forecast_p2 == expected_g, (
        f"forecast_p2={snap.forecast_p2} != {expected_g}"
    )


def test_last_stable_score_updated_after_finalize() -> None:
    """finalize 後に last_stable_score=score_after に更新され次連鎖の基準になる。

    2連鎖シナリオ:
        1連鎖目: score=0→2100 → finalize → last_stable_score=2100
        2連鎖目: score=2100→5600 → chain_total=5600-2100=3500
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 試合開始: score=0 (last_stable_score=0)
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=0, t_sec=1.0,
    )

    # 1連鎖目
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.0,
    )
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=2100, t_sec=7.0,
    )
    # settle 確定(finalized_at ≒ 7.02)
    t_1st_end = _settle_score(tracker, "p1", 2100, t_start=7.0)
    snap_1st = tracker.get_snapshot(t_1st_end)
    g1, _ = _score_to_ojama_count(2100)
    assert snap_1st.chain_total_score_p1 == 2100, (
        f"1連鎖目 chain_total={snap_1st.chain_total_score_p1} != 2100"
    )

    # finalize 後 last_stable_score=2100 が設定済み。
    # 2連鎖目: coalesce window (2.5s) + α を超えてから発火
    t_2nd = t_1st_end + CHAIN_COALESCE_WINDOW_SEC + 0.5
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=t_2nd,
    )
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=5600, t_sec=t_2nd + 2.0,
    )
    t_2nd_end = _settle_score(tracker, "p1", 5600, t_start=t_2nd + 2.0)
    snap_2nd = tracker.get_snapshot(t_2nd_end)

    expected_chain_total_2nd = 3500  # 5600 - 2100 (last_stable_score=finalize後の2100)
    expected_g2, _ = _score_to_ojama_count(expected_chain_total_2nd)
    assert expected_g2 == 50

    assert snap_2nd.chain_total_score_p1 == expected_chain_total_2nd, (
        f"2連鎖目 chain_total={snap_2nd.chain_total_score_p1} != {expected_chain_total_2nd} "
        f"(last_stable_score がfinalize後のscore_afterに更新されていることを確認)"
    )
    assert snap_2nd.total_generated_by_p1 == g1 + expected_g2, (
        f"total_generated={snap_2nd.total_generated_by_p1} != {g1 + expected_g2}"
    )


def test_last_stable_score_cleared_at_match_boundary() -> None:
    """試合境界(score大幅減少)で last_stable_score がリセットされる。

    前試合の last_stable_score が次試合の基準に使われると誤計算が起きる。
    試合境界で None にクリアされ、フォールバック(score_at_chain_start)が使われることを確認。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 前試合: score=50000 まで到達
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=50000, t_sec=200.0,
    )
    # 試合境界: score 大幅減少 → last_stable_score リセット
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=100, t_sec=201.0,
    )
    # 次試合: score=100 で STABLE を受信 → last_stable_score=100 に設定
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=100, t_sec=202.0,
    )
    # 連鎖開始 (score_at_chain_start=100)
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=203.0,
    )
    # 連鎖終了 → settle 待ち開始: score=2200 → chain_total=2200-100=2100
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=2200, t_sec=205.0,
    )
    # settle 確定
    t_end = _settle_score(tracker, "p1", 2200, t_start=205.0)
    snap = tracker.get_snapshot(t_end)

    expected_chain_total = 2100  # 2200 - 100 (last_stable_scoreが100にリセット済み)
    expected_g, _ = _score_to_ojama_count(expected_chain_total)
    assert expected_g == 30

    assert snap.chain_total_score_p1 == expected_chain_total, (
        f"次試合chain_total={snap.chain_total_score_p1} != {expected_chain_total} "
        f"(前試合のlast_stable_score=50000が漏れていたらchain_totalが負になる)"
    )
    assert snap.forecast_p2 == expected_g, (
        f"forecast_p2={snap.forecast_p2} != {expected_g}"
    )


def test_last_stable_score_fallback_to_score_at_chain_start() -> None:
    """last_stable_score が None のとき score_at_chain_start にフォールバックする。

    試合開始直後 (last_stable_score 未設定) で連鎖が起きた場合、
    score_at_chain_start (last_valid_score から設定) を使って正常に計算できることを確認。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # STABLE 中 score=500 を受信 (last_valid_score=500, last_stable_score=500 も更新)
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=500, t_sec=5.0,
    )
    # 連鎖開始: STABLE→CHAIN, score=None
    # この時点で last_stable_score=500 (STABLE フレームで更新済み)
    # → chain_total = score_after - 500 となることを確認
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.5,
    )
    # 連鎖終了 → settle 待ち開始
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=2800, t_sec=7.0,
    )
    # settle 確定
    t_end = _settle_score(tracker, "p1", 2800, t_start=7.0)
    snap = tracker.get_snapshot(t_end)

    expected_chain_total = 2300  # 2800 - 500
    expected_g, expected_leftover = _score_to_ojama_count(expected_chain_total)
    assert expected_g == 32
    assert expected_leftover == 60  # 2300 = 32*70 + 60

    assert snap.chain_total_score_p1 == expected_chain_total, (
        f"chain_total={snap.chain_total_score_p1} != {expected_chain_total}"
    )
    assert snap.forecast_p2 == expected_g, (
        f"forecast_p2={snap.forecast_p2} != {expected_g}"
    )


def test_last_stable_score_not_updated_while_chain_end_pending() -> None:
    """chain_end_pending 中 (score None 遅延確定待ち) は last_stable_score が更新されない。

    CHAIN→STABLE 後 score=None で pending になった後、
    STABLE 状態で score が来ても pending 中は last_stable_score を汚さない。
    遅延確定時に正しい chain_total が計算されることを確認する。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 連鎖前 STABLE: score=300 → last_stable_score=300
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=300, t_sec=5.0,
    )
    # 連鎖開始
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.5,
    )
    # 連鎖終了 score=None → pending 開始
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=None, t_sec=7.0,
    )
    # settle 待ち中: STABLE で score=999 が来る
    # → settle 候補=999。last_stable_score は score_settle_pending 中なので更新されない。
    # settle 確定後: chain_total = 999 - last_stable_score(300) = 699 → G=9, leftover=69
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=999, t_sec=7.1,
    )
    # settle 確定(K_SETTLE_FRAMES 連続不変)
    t_end = _settle_score(tracker, "p1", 999, t_start=7.1)
    snap = tracker.get_snapshot(t_end)

    expected_chain_total = 699  # 999 - 300
    expected_g, expected_leftover = _score_to_ojama_count(expected_chain_total)
    assert expected_g == 9
    assert expected_leftover == 69  # 699 = 9*70 + 69

    assert snap.chain_total_score_p1 == expected_chain_total, (
        f"settle確定 chain_total={snap.chain_total_score_p1} != {expected_chain_total}"
    )
    assert snap.forecast_p2 == expected_g, (
        f"settle確定後 forecast_p2={snap.forecast_p2} != {expected_g}"
    )


# ============================
# 27. score settle 待ち 回帰テスト (2026-07-07 新機能)
# ============================

def test_score_settle_not_finalized_before_k_settle_frames() -> None:
    """K_SETTLE_FRAMES フレーム未満では finalize されない(途中スコア確定バグ回帰防止)。

    問題の根本: 旧実装では CHAIN→STABLE で即 finalize していた。
    このため大連鎖の途中スコアで確定し、残りの連鎖が取りこぼされた。
    新実装: K_SETTLE_FRAMES-1 フレーム連続不変でも finalize しない。
    """
    from src.ojama_accounting import K_SETTLE_FRAMES

    tracker = OjamaAccountingTracker()
    tracker.reset()

    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=0, t_sec=1.0)
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.0)
    # 連鎖終了: settle 待ち開始(score=2100)
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.STABLE, score=2100, t_sec=7.0)

    # K_SETTLE_FRAMES-1 フレームだけ同一スコアを流す → まだ finalize しない
    _dt = 0.001
    for i in range(K_SETTLE_FRAMES - 1):
        tracker.on_state_transition(
            "p1", BoardState.STABLE, BoardState.STABLE,
            score=2100, t_sec=7.0 + (i + 1) * _dt,
        )
    snap_early = tracker.get_snapshot(t_sec=7.0 + (K_SETTLE_FRAMES - 1) * _dt)
    assert snap_early.forecast_p2 == 0, (
        f"K_SETTLE_FRAMES-1 フレームでは finalize しない: forecast_p2={snap_early.forecast_p2} "
        f"(0 であるべき、途中確定バグなら非ゼロ)"
    )

    # K_SETTLE_FRAMES フレーム目で finalize
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE,
        score=2100, t_sec=7.0 + K_SETTLE_FRAMES * _dt,
    )
    snap_settled = tracker.get_snapshot(t_sec=7.0 + K_SETTLE_FRAMES * _dt)
    expected_g, _ = _score_to_ojama_count(2100)
    assert expected_g == 30
    assert snap_settled.forecast_p2 == expected_g, (
        f"K_SETTLE_FRAMES フレームで finalize: forecast_p2={snap_settled.forecast_p2} != {expected_g}"
    )


def test_score_settle_resets_on_score_rise() -> None:
    """settle 待ち中にスコアが上昇したらカウンタリセット(連鎖継続認識)。

    大連鎖のスコア段階上昇シナリオ:
        CHAIN→STABLE で settle 待ち(候補=500)
        K_SETTLE_FRAMES/2 フレーム後: score=800(上昇) → カウントリセット
        その後 K_SETTLE_FRAMES フレーム score=800 不変 → finalize(chain_total=800-0=800)
    """
    from src.ojama_accounting import K_SETTLE_FRAMES

    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 連鎖前 score=0
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=0, t_sec=1.0)
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.0)
    # 連鎖終了: settle 待ち開始(候補=500)
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.STABLE, score=500, t_sec=7.0)

    _dt = 0.001
    # K_SETTLE_FRAMES/2 フレーム: score=500 不変
    half = K_SETTLE_FRAMES // 2
    for i in range(half):
        tracker.on_state_transition(
            "p1", BoardState.STABLE, BoardState.STABLE,
            score=500, t_sec=7.0 + (i + 1) * _dt,
        )
    # スコア上昇: score=800 → カウントリセット
    t_rise = 7.0 + (half + 1) * _dt
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=800, t_sec=t_rise)
    snap_after_rise = tracker.get_snapshot(t_rise)
    # まだ finalize していない
    assert snap_after_rise.forecast_p2 == 0, (
        f"スコア上昇後はまだ未確定: forecast_p2={snap_after_rise.forecast_p2}"
    )

    # K_SETTLE_FRAMES フレーム: score=800 不変 → finalize
    t_end = _settle_score(tracker, "p1", 800, t_start=t_rise)
    snap = tracker.get_snapshot(t_end)
    expected_g, _ = _score_to_ojama_count(800)  # 800-0=800
    assert snap.forecast_p2 == expected_g, (
        f"スコア上昇後 settle: forecast_p2={snap.forecast_p2} != {expected_g}"
    )
    assert snap.chain_total_score_p1 == 800, (
        f"chain_total={snap.chain_total_score_p1} != 800 (上昇後のスコアで確定)"
    )


def test_score_settle_aborted_at_match_boundary() -> None:
    """settle 待ち中に試合境界(score大幅減少)が来たら settle を破棄する。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=0, t_sec=1.0)
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.0)
    # 連鎖終了: settle 待ち開始
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.STABLE, score=2100, t_sec=7.0)

    # 試合境界: score 大幅減少 → settle 破棄 + forecast=0 にリセット
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=2100, t_sec=7.5)
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=50, t_sec=8.0)
    snap = tracker.get_snapshot(t_sec=8.0)

    assert snap.forecast_p2 == 0, (
        f"試合境界後 forecast_p2={snap.forecast_p2} (settle破棄で0になるべき)"
    )
    assert snap.chain_total_score_p1 == 0, (
        f"試合境界後 chain_total={snap.chain_total_score_p1} (破棄で0のまま)"
    )


def test_score_settle_with_none_frames_in_between() -> None:
    """settle 待ち中に score=None フレームが挟まっても K_SETTLE_FRAMES カウントは止まる。

    score=None は「まだ掛け算式中かもしれない」とみなしカウントしない。
    None フレームの後に同じスコアが来てもカウントは継続。
    """
    from src.ojama_accounting import K_SETTLE_FRAMES

    tracker = OjamaAccountingTracker()
    tracker.reset()

    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=0, t_sec=1.0)
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.0)
    # 連鎖終了: settle 待ち(候補=2100)
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.STABLE, score=2100, t_sec=7.0)

    _dt = 0.001
    # K_SETTLE_FRAMES/2 フレーム: score=2100 不変でカウント進む
    half = K_SETTLE_FRAMES // 2
    for i in range(half):
        tracker.on_state_transition(
            "p1", BoardState.STABLE, BoardState.STABLE,
            score=2100, t_sec=7.0 + (i + 1) * _dt,
        )
    # score=None フレーム挿入: カウントは進まない
    t_none = 7.0 + (half + 1) * _dt
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=None, t_sec=t_none)
    snap_none = tracker.get_snapshot(t_none)
    assert snap_none.forecast_p2 == 0, "score=None フレームでは finalize しない"

    # score=2100 再開: カウント継続(None で止まっていた分)
    # 残り (K_SETTLE_FRAMES - half) フレームで settle 確定
    remaining = K_SETTLE_FRAMES - half
    t_resume = t_none + _dt
    for i in range(remaining):
        tracker.on_state_transition(
            "p1", BoardState.STABLE, BoardState.STABLE,
            score=2100, t_sec=t_resume + i * _dt,
        )
    snap = tracker.get_snapshot(t_resume + (remaining - 1) * _dt)

    expected_g, _ = _score_to_ojama_count(2100)
    assert expected_g == 30
    assert snap.forecast_p2 == expected_g, (
        f"None挿入後も settle 確定: forecast_p2={snap.forecast_p2} != {expected_g}"
    )


def test_score_settle_video124_scenario() -> None:
    """video_124 t=142 の実例シナリオ: 2P 連鎖が settle で正しく計上される。

    問題:
        旧実装: (CHAIN|GRAVITY_SETTLE)→STABLE の最初の遷移で即 finalize。
                2P score が 246→286(+40) で確定。gen=0 → 相殺なし → forecast_p2=45(過多)。
    期待:
        新実装: score settle (K_SETTLE_FRAMES 連続不変) まで待つ。
                2P score が最終値(287→389+)で確定 → gen>0 → 相殺成立 → forecast_p2 減少。
    """
    from src.ojama_accounting import K_SETTLE_FRAMES

    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 2P 連鎖開始前スコア: 246
    tracker.on_state_transition("p2", BoardState.STABLE, BoardState.STABLE, score=246, t_sec=136.0)
    tracker.on_state_transition("p2", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=137.0)

    # 連鎖途中: 旧実装はここで即確定していた(score=286)
    tracker.on_state_transition("p2", BoardState.CHAIN, BoardState.STABLE, score=286, t_sec=138.0)
    snap_mid = tracker.get_snapshot(t_sec=138.0)
    # 新実装: settle 待ち中なのでまだ finalize していない
    assert snap_mid.total_generated_by_p2 == 0, (
        f"settle待ち中(score=286)では finalize しない: gen={snap_mid.total_generated_by_p2}"
    )

    # スコアが継続上昇(連鎖継続)
    tracker.on_state_transition("p2", BoardState.STABLE, BoardState.STABLE, score=350, t_sec=138.5)
    tracker.on_state_transition("p2", BoardState.STABLE, BoardState.STABLE, score=389, t_sec=139.0)

    # score=389 が K_SETTLE_FRAMES フレーム連続不変 → finalize
    t_end = _settle_score(tracker, "p2", 389, t_start=139.0)
    snap = tracker.get_snapshot(t_end)

    # chain_total = 389 - 246 = 143 → G=2, leftover=3
    expected_chain_total = 143  # 389 - 246
    expected_g, expected_leftover = _score_to_ojama_count(expected_chain_total)
    assert expected_g == 2  # 143 // 70 = 2
    assert snap.total_generated_by_p2 == expected_g, (
        f"settle後 gen={snap.total_generated_by_p2} != {expected_g} "
        f"(旧実装は286-246=40<70でgen=0。新実装は389-246=143でgen=2)"
    )
    assert snap.chain_total_score_p2 == expected_chain_total, (
        f"chain_total_p2={snap.chain_total_score_p2} != {expected_chain_total}"
    )


# ============================
# 28. coalesce window 内の本物の2本目連鎖 回帰テスト (2026-07-09 修正)
# ============================

def test_real_second_chain_within_coalesce_window_is_counted() -> None:
    """coalesce window(2.5s)内でもスコアが増加していれば本物の2本目連鎖として計上される。

    修正前バグ: 短時間に2連鎖撃つと2本目が coalesce window で skip され
                score_at_chain_start=None のまま _finalize_chain_end で破棄されていた。
                例: video_124 t=142 2P: chain1(246→286) → chain2(287→391)
                    2本目が未計上 → 相殺なし → forecast_p2 過多。

    修正後: coalesce window 内でも last_valid_score > last_stable_score なら
            本物の2本目連鎖として score_at_chain_start を設定し、計上する。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 1本目連鎖: score 246→286 (chain_total=40<70, gen=0, leftover=40)
    tracker.on_state_transition("p2", BoardState.STABLE, BoardState.STABLE, score=246, t_sec=136.0)
    tracker.on_state_transition("p2", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=137.0)
    tracker.on_state_transition("p2", BoardState.CHAIN, BoardState.STABLE, score=286, t_sec=138.0)
    # settle 確定 (finalized_at ≒ 138.02, last_stable_score=286)
    t_1st_end = _settle_score(tracker, "p2", 286, t_start=138.0)
    snap_1st = tracker.get_snapshot(t_1st_end)
    # chain_total=40 < 70 → gen=0, leftover=40
    assert snap_1st.chain_total_score_p2 == 40, (
        f"1本目 chain_total={snap_1st.chain_total_score_p2} (40 であるべき)"
    )
    assert snap_1st.total_generated_by_p2 == 0, (
        f"1本目 gen={snap_1st.total_generated_by_p2} (40<70 で gen=0)"
    )
    assert snap_1st.leftover_p2 == 40, (
        f"1本目 leftover={snap_1st.leftover_p2} (40 であるべき)"
    )

    # 2本目連鎖: coalesce window(2.5s)内に開始 (t=139.33 ≒ 1.3s後 < 2.5s)
    # → last_stable_score=286, last_valid_score=287 > 286 → score_rose=True → 計上
    t_2nd_chain_start = t_1st_end + 1.3  # ≒ 139.32: coalesce window 内
    assert t_2nd_chain_start - t_1st_end < CHAIN_COALESCE_WINDOW_SEC, (
        "前提: 2本目の連鎖開始は coalesce window(2.5s)内"
    )
    tracker.on_state_transition("p2", BoardState.STABLE, BoardState.STABLE, score=287, t_sec=t_2nd_chain_start - 0.1)
    tracker.on_state_transition("p2", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=t_2nd_chain_start)
    # 連鎖終了: score=391
    tracker.on_state_transition("p2", BoardState.CHAIN, BoardState.STABLE, score=391, t_sec=t_2nd_chain_start + 2.0)
    # settle 確定
    t_2nd_end = _settle_score(tracker, "p2", 391, t_start=t_2nd_chain_start + 2.0)
    snap_2nd = tracker.get_snapshot(t_2nd_end)

    # 2本目連鎖の基準スコアは以下のとおり:
    #   - score_at_chain_start = 287 (last_valid_score。直前 STABLE score=287 で設定)
    #   - last_stable_score = 287 (STABLE score=287 のフレームで更新済み)
    #   - finalize時 score_start = last_stable_score = 287
    #   - chain_total = 391 - 287 = 104
    #   - leftover=40 引き継ぎ: score_to_ojama(104, prev_leftover=40) → (144//70=2, 144%70=4)
    r_2nd = score_to_ojama(104, prev_leftover=40)
    expected_gen_2nd = r_2nd.ojama_count  # (104+40)=144, 144//70=2
    assert expected_gen_2nd == 2, f"前提確認: 2本目 gen={expected_gen_2nd} (2 であるべき)"
    expected_leftover_2nd = r_2nd.leftover_score  # 144 % 70 = 4

    assert snap_2nd.chain_total_score_p2 == 104, (
        f"2本目 chain_total={snap_2nd.chain_total_score_p2} != 104 "
        "(修正前は score_at_chain_start=None で破棄)"
    )
    assert snap_2nd.total_generated_by_p2 == expected_gen_2nd, (
        f"2本目 total_gen={snap_2nd.total_generated_by_p2} != {expected_gen_2nd} "
        "(修正前は skip で 0 のまま)"
    )
    assert snap_2nd.leftover_p2 == expected_leftover_2nd, (
        f"2本目 leftover={snap_2nd.leftover_p2} != {expected_leftover_2nd}"
    )
    # 送り先 (1P) に gen が届いている
    assert snap_2nd.forecast_p1 == expected_gen_2nd, (
        f"forecast_p1={snap_2nd.forecast_p1} != {expected_gen_2nd} "
        "(2本目の surplus が 1P に届くべき)"
    )


def test_real_second_chain_within_coalesce_score_not_rose_skipped() -> None:
    """coalesce window 内でもスコアが増加していない場合は明滅として skip される。

    state 明滅: 同一連鎖が CHAIN→STABLE→CHAIN と明滅するだけ。
    スコアは finalize 後の last_stable_score と同じ → _score_rose=False → skip。
    これが既存の「二重 finalize 防止」テストと同じ動作であることを確認。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 連鎖: score 0→2100
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=0, t_sec=5.0)
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.5)
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.STABLE, score=2100, t_sec=6.5)
    # settle 確定 (finalized_at ≒ 6.52, last_stable_score=2100)
    t_1st_end = _settle_score(tracker, "p1", 2100, t_start=6.5)
    snap_1st = tracker.get_snapshot(t_1st_end)
    g1, _ = _score_to_ojama_count(2100)
    assert snap_1st.total_generated_by_p1 == g1

    # state 明滅: coalesce window 内 (t=6.8: 差0.28s < 2.5s)
    # last_valid_score=2100 (明滅なのでスコア変化なし)
    # → last_valid_score(2100) > last_stable_score(2100) は False → skip
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=6.8)
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.STABLE, score=2100, t_sec=7.0)
    t_2nd_end = _settle_score(tracker, "p1", 2100, t_start=7.0)
    snap_2nd = tracker.get_snapshot(t_2nd_end)

    # 明滅なので total_generated は変わらない
    assert snap_2nd.total_generated_by_p1 == g1, (
        f"明滅(score不変)後 total_gen={snap_2nd.total_generated_by_p1} != {g1} "
        "(明滅は skip されるべき)"
    )
    assert snap_2nd.forecast_p2 == g1, (
        f"明滅後 forecast_p2={snap_2nd.forecast_p2} != {g1}"
    )


def test_real_second_chain_coalesce_window_offset_cancels_properly() -> None:
    """coalesce window 内の本物の2本目連鎖の相殺が正しく計算される。

    video_124 t=142 のシナリオ縮約:
        1P が forecast_p2=44 を持っている。
        2P が2本目連鎖(score 287→391, chain_total=391-287=104)を撃つ
        → leftover=40引継ぎ: score_to_ojama(104,40)=gen=2
        → surplus=2 → 1P forecast が 44→42 に減少。

    注: 修正前は2本目が計上されず gen=0 → canceled=0 → forecast_p1=44 のまま。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # まず 2P に forecast_incoming=44 を積む。
    # 1P が G=44 を 2P に送る: t=128 前後 (マージンタイム非適用: elapsed < 96s)。
    # _match_start_sec を試合相対短時間に保つため 1P 連鎖を t=128 台に配置する。
    # これにより 2P 連鎖(t=136)の elapsed = 136-128 = 8s < 96s → rate=70 → マージンタイム非適用。
    score_for_44 = 44 * OJAMA_RATE_STANDARD  # 3080点
    _fire_chain(tracker, "p1", chain_score=score_for_44, score_before=0, t_sec=128.0)
    snap_before = tracker.get_snapshot(t_sec=130.0)
    assert snap_before.forecast_p2 == 44, (
        f"前提: forecast_p2={snap_before.forecast_p2} (44 であるべき)"
    )

    # 2P の1本目連鎖: score 246→286 (chain_total=40, gen=0, leftover=40)
    tracker.on_state_transition("p2", BoardState.STABLE, BoardState.STABLE, score=246, t_sec=136.0)
    tracker.on_state_transition("p2", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=137.0)
    tracker.on_state_transition("p2", BoardState.CHAIN, BoardState.STABLE, score=286, t_sec=138.0)
    t_1st_end = _settle_score(tracker, "p2", 286, t_start=138.0)
    snap_after_1st = tracker.get_snapshot(t_1st_end)
    # 1本目: gen=0 → 1P側の forecast_incoming(=forecast_p1)は変化なし=0のまま。
    # 1Pが送った予告 (forecast_p2=2Pへの予告) は44のまま変わらない。
    assert snap_after_1st.forecast_p2 == 44, (
        f"1本目連鎖後 forecast_p2={snap_after_1st.forecast_p2} (gen=0で変化なし, 44のまま)"
    )
    assert snap_after_1st.forecast_p1 == 0, (
        f"1本目連鎖後 forecast_p1={snap_after_1st.forecast_p1} (2Pからの予告はまだ0)"
    )

    # 2P の2本目連鎖: coalesce window 内 (1.3s後), score 287→391
    t_2nd = t_1st_end + 1.3
    tracker.on_state_transition("p2", BoardState.STABLE, BoardState.STABLE, score=287, t_sec=t_2nd - 0.1)
    tracker.on_state_transition("p2", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=t_2nd)
    tracker.on_state_transition("p2", BoardState.CHAIN, BoardState.STABLE, score=391, t_sec=t_2nd + 2.0)
    t_2nd_end = _settle_score(tracker, "p2", 391, t_start=t_2nd + 2.0)
    snap_2nd = tracker.get_snapshot(t_2nd_end)

    # 2本目 chain_total=391-287=104(last_stable_score=287), leftover=40引継ぎ
    # score_to_ojama(104,40): (104+40)=144, 144//70=2, leftover=4
    r_2nd = score_to_ojama(104, prev_leftover=40)
    gen_2nd = r_2nd.ojama_count  # 2
    assert gen_2nd == 2

    # 2P が gen=2 を撃つ:
    #   - 2P 自身の forecast_incoming (=p2が受け取る予告, 1Pからの44個) から相殺
    #   - canceled = min(gen=2, p2.forecast_incoming=44) = 2
    #   - p2.forecast_incoming = 44 - 2 = 42 → forecast_p2 = 42
    #   - surplus = 2 - 2 = 0 → 1P の forecast_incoming は変化なし=0
    expected_forecast_p2 = 44 - gen_2nd  # 42
    expected_forecast_p1 = 0  # surplus=0 なので 1P 側に追加なし
    assert snap_2nd.forecast_p2 == expected_forecast_p2, (
        f"2本目連鎖後 forecast_p2={snap_2nd.forecast_p2} != {expected_forecast_p2} "
        f"(修正前は gen=0 で相殺なし → forecast_p2=44 のまま)"
    )
    assert snap_2nd.forecast_p1 == expected_forecast_p1, (
        f"2本目連鎖後 forecast_p1={snap_2nd.forecast_p1} != {expected_forecast_p1} "
        f"(surplus=0 なので 1P 側への追加なし)"
    )


# ============================
# 29. 物理イベント基準 finalize (TSUMO_FALL/OJAMA_FALL トリガー) 回帰テスト (2026-07-09)
# ============================

def test_tsumo_fall_triggers_settle_during_chain() -> None:
    """chain_active 中に TSUMO_FALL 遷移が来たとき settle 待ちが開始される。

    新基準: 「次ツモ出現 (TSUMO_FALL) = 連鎖の得点計算完了の可能性」として
    settle 待ちを開始し、K_SETTLE_FRAMES 連続不変で finalize する。

    注: TSUMO_FALL→STABLE でも既存の補完トリガーが動くため、
    ここでは TSUMO_FALL → TSUMO_FALL (継続) → 不変スコアで settle を確認する。
    """
    from src.ojama_accounting import K_SETTLE_FRAMES

    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 連鎖開始前 score=465
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=465, t_sec=5.0)
    # 連鎖開始: STABLE→CHAIN
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.5)
    # 連鎖中 → TSUMO_FALL (新トリガー: settle 待ち開始)
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.TSUMO_FALL, score=3565, t_sec=6.5)
    snap_tsumo = tracker.get_snapshot(t_sec=6.5)
    # settle 待ち開始直後はまだ finalize していない (consec=1 < K_SETTLE_FRAMES)
    assert snap_tsumo.forecast_p2 == 0, (
        f"TSUMO_FALL直後: settle待ち中 forecast_p2={snap_tsumo.forecast_p2} (0 であるべき)"
    )
    # K_SETTLE_FRAMES 連続不変 → finalize
    t_end = _settle_score(tracker, "p1", 3565, t_start=6.5)
    snap = tracker.get_snapshot(t_end)

    # chain_total = 3565 - 465 = 3100 → G=44, leftover=20
    expected_chain_total = 3100
    expected_g, expected_leftover = _score_to_ojama_count(expected_chain_total)
    assert expected_g == 44
    assert expected_leftover == 20  # 3100 = 44*70 + 20

    assert snap.chain_total_score_p1 == expected_chain_total, (
        f"TSUMO_FALLトリガー: chain_total={snap.chain_total_score_p1} != {expected_chain_total}"
    )
    assert snap.forecast_p2 == expected_g, (
        f"TSUMO_FALLトリガー: forecast_p2={snap.forecast_p2} != {expected_g}"
    )
    assert snap.leftover_p1 == expected_leftover, (
        f"TSUMO_FALLトリガー: leftover_p1={snap.leftover_p1} != {expected_leftover}"
    )


def test_ojama_fall_triggers_settle_during_chain() -> None:
    """chain_active 中に OJAMA_FALL 遷移が来たとき settle 待ちが開始される。

    OJAMA_FALL 基準: お邪魔落下 = 連鎖の得点計算が完了したことを示す物理イベント。
    settle 待ちを開始し、score 上昇停止後に finalize する。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 連鎖開始前 score=200
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=200, t_sec=5.0)
    # 連鎖開始: STABLE→CHAIN
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.5)
    # 連鎖中スコア上昇
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.CHAIN, score=1200, t_sec=6.0)
    # OJAMA_FALL に遷移 (新トリガー: settle 待ち開始)
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.OJAMA_FALL, score=2100, t_sec=6.5)
    snap_oj = tracker.get_snapshot(t_sec=6.5)
    # settle 待ち中はまだ finalize していない
    assert snap_oj.forecast_p2 == 0, (
        f"OJAMA_FALL直後: settle待ち中 forecast_p2={snap_oj.forecast_p2} (0 であるべき)"
    )
    # K_SETTLE_FRAMES 連続不変 → finalize
    t_end = _settle_score(tracker, "p1", 2100, t_start=6.5)
    snap = tracker.get_snapshot(t_end)

    # chain_total = 2100 - 200 = 1900 → G=27, leftover=10
    expected_chain_total = 1900
    expected_g, expected_leftover = _score_to_ojama_count(expected_chain_total)
    assert expected_g == 27
    assert expected_leftover == 10  # 1900 = 27*70 + 10

    assert snap.chain_total_score_p1 == expected_chain_total, (
        f"OJAMA_FALLトリガー: chain_total={snap.chain_total_score_p1} != {expected_chain_total}"
    )
    assert snap.forecast_p2 == expected_g, (
        f"OJAMA_FALLトリガー: forecast_p2={snap.forecast_p2} != {expected_g}"
    )


def test_tsumo_fall_score_still_rising_does_not_finalize() -> None:
    """TSUMO_FALL が来ても score が上昇中(得点計算中)は finalize しない。

    1P の大連鎖シナリオ (受け入れ基準B):
        score が 465 → 825 と上昇中に OJAMA_FALL が来る。
        settle 待ちは開始されるが score 上昇中なのでカウントがリセットされ、
        最終スコア(3571)が確定するまで finalize しない。
    """
    from src.ojama_accounting import K_SETTLE_FRAMES

    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 連鎖前 score=465
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.STABLE, score=465, t_sec=5.0)
    # 連鎖開始
    tracker.on_state_transition("p1", BoardState.STABLE, BoardState.CHAIN, score=None, t_sec=5.5)
    # 連鎖途中: score 上昇
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.CHAIN, score=825, t_sec=6.0)
    # OJAMA_FALL に遷移(settle 待ち開始: 候補=825)
    tracker.on_state_transition("p1", BoardState.CHAIN, BoardState.OJAMA_FALL, score=825, t_sec=6.3)
    snap_oj = tracker.get_snapshot(t_sec=6.3)
    # まだ finalize していない
    assert snap_oj.forecast_p2 == 0

    # score が 825 → 1465 → 3571 と継続上昇 (settle カウントがリセットされ続ける)
    tracker.on_state_transition("p1", BoardState.OJAMA_FALL, BoardState.OJAMA_FALL, score=1465, t_sec=6.6)
    tracker.on_state_transition("p1", BoardState.OJAMA_FALL, BoardState.OJAMA_FALL, score=3571, t_sec=7.0)
    # この時点でカウント=1 → まだ finalize せず
    snap_rising = tracker.get_snapshot(t_sec=7.0)
    assert snap_rising.forecast_p2 == 0, (
        f"score上昇中(3571): まだ finalize しない: forecast_p2={snap_rising.forecast_p2}"
    )

    # score=3571 が K_SETTLE_FRAMES 連続不変 → finalize
    t_end = _settle_score(tracker, "p1", 3571, t_start=7.0)
    snap = tracker.get_snapshot(t_end)

    # chain_total = 3571 - 465 = 3106 → G=44, leftover=26
    expected_chain_total = 3106
    expected_g, expected_leftover = _score_to_ojama_count(expected_chain_total)
    assert expected_g == 44
    assert expected_leftover == 26  # 3106 = 44*70 + 26

    assert snap.chain_total_score_p1 == expected_chain_total, (
        f"score上昇停止後 finalize: chain_total={snap.chain_total_score_p1} != {expected_chain_total} "
        f"(途中 825 or 1465 で誤分割されていたら chain_total が過少になる)"
    )
    assert snap.forecast_p2 == expected_g, (
        f"forecast_p2={snap.forecast_p2} != {expected_g} "
        f"(1P大連鎖 score上昇中に OJAMA_FALL が来ても誤分割しない)"
    )
