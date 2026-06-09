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


def _fire_chain(
    tracker: OjamaAccountingTracker,
    side: str,
    chain_score: int,
    score_before: int = 0,
    t_sec: float = 5.0,
) -> OjamaAccountSnapshot:
    """連鎖開始 → 連鎖終了 の状態遷移を一連でシミュレートする。

    Returns:
        連鎖終了直後のスナップショット。
    """
    score_after = score_before + chain_score
    # 連鎖開始: STABLE → CHAIN
    tracker.on_state_transition(
        side, BoardState.STABLE, BoardState.CHAIN,
        score_before, t_sec,
    )
    # 連鎖終了: CHAIN → STABLE
    tracker.on_state_transition(
        side, BoardState.CHAIN, BoardState.STABLE,
        score_after, t_sec + 2.0,
    )
    return tracker.get_snapshot(t_sec + 2.0)


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
    # GRAVITY_SETTLE → STABLE (連鎖終了)
    tracker.on_state_transition(
        "p1", BoardState.GRAVITY_SETTLE, BoardState.STABLE, score_after, t_sec=7.0,
    )
    snap = tracker.get_snapshot(t_sec=7.0)

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
    """CHAIN→STABLE 遷移時 score=None なら待機し、次フレームで確定。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    chain_total = 2100
    expected_ojama, _ = _score_to_ojama_count(chain_total)

    # 連鎖開始
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.CHAIN, score=0, t_sec=5.0,
    )
    # 連鎖終了 score=None (OCR 失敗)
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=None, t_sec=7.0,
    )
    snap_pending = tracker.get_snapshot(t_sec=7.0)
    # まだ確定していない
    assert snap_pending.forecast_p2 == 0

    # 次フレームで score が来る
    tracker.on_state_transition(
        "p1", BoardState.STABLE, BoardState.STABLE, score=chain_total, t_sec=7.1,
    )
    snap_confirmed = tracker.get_snapshot(t_sec=7.1)

    assert snap_confirmed.forecast_p2 == expected_ojama, (
        f"遅延確定後 forecast_p2={snap_confirmed.forecast_p2} != {expected_ojama}"
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

    # 連鎖終了後は None に戻る
    tracker.on_state_transition(
        "p1", BoardState.CHAIN, BoardState.STABLE, score=3100, t_sec=7.0,
    )
    snap_end = tracker.get_snapshot(t_sec=7.0)
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
