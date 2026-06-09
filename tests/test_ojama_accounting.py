"""src/ojama_accounting.py のユニットテスト。

3 局面で理論値突合:
    1. 単純生成: 1P が連鎖発火 → 2P pending が理論値と一致
    2. 相殺: 両者ほぼ同時発火 → net_balance が片側 pending 残と一致
    3. 全消し: 全消し後の次連鎖で 2100pt ボーナスが正しく載る
    4. score_to_ojama の rate/leftover carry-over 整合
    5. 信頼度: score OCR のみ / visual 一致 / visual 不一致
    6. reset() で帳簿クリア
    7. get_snapshot() はイベントなしでも現在状態を返す
    8. overflow_risk の補助 bool
    9. net_ojama_balance の符号
    10. total_generated / total_offset の累積整合
    --- Step1.5 追加テスト ---
    11. ① 落下 total_dropped: 盤面内おじゃま増分で帳簿一致 (生成 = 相殺 + 落下 + 残 pending)
    12. ① 帳簿一致テスト (生成 = 相殺 + 落下 + 残 pending)
    13. ① 落下 sanity clamp: 増分が DROP_SANITY_CLAMP を超えたら clamp
    14. ② 全消し自動検出: confirmed_board が全 EMPTY で all_clear_pending がセットされる
    15. ② 全消し自動検出: 色ぷよあり (全消しでない) はセットされない
    16. ③ 相殺タイミング: update_accounting_with_chain() で chain 確定後に相殺
    17. ③ update_accounting_with_chain() は連続呼出でエッジ検出 (連続 True は二重相殺しない)
    18. ④ クロスチェック乖離: update_from_boards 後 update_accounting_with_chain で confidence 低下
    --- STABLE ガードテスト (連鎖中誤検知防止) ---
    19. 連鎖中に盤面内おじゃまが一時変動しても pending が誤減算されない
    20. 連鎖中→STABLE 復帰時に正味落下のみが 1 回計上される
    21. 1P/2P のガードは独立 (片方 STABLE なら計上, もう片方 chain なら skip)
    22. is_chain=False (STABLE) の場合は従来通り落下を計上する (後退テスト)
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
from src.ojama_accounting import (
    CHAIN_FIRE_MIN_SCORE,
    CONFIDENCE_SCORE_OCR_ONLY,
    CONFIDENCE_VISUAL_AGREE,
    CONFIDENCE_VISUAL_MISMATCH_PENALTY,
    DROP_SANITY_CLAMP,
    ON_FIELD_CAP,
    PENDING_ABS_CAP,
    PENDING_HARD_CAP,
    SCORE_RESET_THRESHOLD,
    THEORY_DROP_PER_TURN,
    VISIBLE_OJAMA_MISMATCH_THRESHOLD,
    OjamaAccountSnapshot,
    OjamaAccountingTracker,
)
from src.scoring import (
    ALL_CLEAR_BONUS,
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
    """可視領域 (row=1〜) に指定数のおじゃまぷよを配置した盤面を返す。

    下段 (row=12) から順に col=0〜5 で埋める。
    """
    data = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    placed = 0
    for row in range(BOARD_ROWS - 1, 0, -1):  # row 12→1 (隠し段 row=0 は除外)
        for col in range(BOARD_COLS):
            if placed >= ojama_count:
                break
            data[row][col] = COLOR_OJAMA
            placed += 1
        if placed >= ojama_count:
            break
    return Board.from_list(data)


def _make_board_with_color_puyo(color: int = 1, count: int = 4) -> Board:
    """色ぷよを指定数配置した盤面を返す (全消し非該当用)。"""
    data = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    placed = 0
    for row in range(BOARD_ROWS - 1, 0, -1):
        for col in range(BOARD_COLS):
            if placed >= count:
                break
            data[row][col] = color
            placed += 1
        if placed >= count:
            break
    return Board.from_list(data)


# ============================
# 初期状態テスト
# ============================

def test_initial_snapshot_zero() -> None:
    """reset 直後のスナップショットは全帳簿ゼロ。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    snap = tracker.get_snapshot(t_sec=0.0)
    assert snap.pending_p1 == 0
    assert snap.pending_p2 == 0
    assert snap.total_generated_by_p1 == 0
    assert snap.total_generated_by_p2 == 0
    assert snap.total_offset_by_p1 == 0
    assert snap.total_offset_by_p2 == 0
    assert snap.net_ojama_balance == 0
    assert snap.leftover_p1 == 0
    assert snap.leftover_p2 == 0
    assert not snap.all_clear_pending_p1
    assert not snap.all_clear_pending_p2


# ============================
# 1. 単純生成テスト
# ============================

def test_simple_generation_p1_fires() -> None:
    """1P が score を増やすと 2P に ojama が送られる (相殺なし)。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 5連鎖相当の典型的 score: 2000 点 → OJAMA_RATE_STANDARD=70 で 28 個
    chain_score = 2000
    expected_ojama, expected_leftover = _score_to_ojama_count(chain_score)

    # 1P score を 0 → chain_score に増加させる (差分が発火条件)
    # まず prev を設定するため 1 フレーム空打ち
    tracker.update_from_score(
        score_p1=0, score_p2=0, t_sec=5.0,
    )
    snap = tracker.update_from_score(
        score_p1=chain_score, score_p2=0, t_sec=6.0,
    )

    assert snap.total_generated_by_p1 == expected_ojama
    assert snap.pending_p2 == expected_ojama, (
        f"2P pending={snap.pending_p2} != expected={expected_ojama}"
    )
    assert snap.pending_p1 == 0
    assert snap.leftover_p1 == expected_leftover
    assert snap.net_ojama_balance == expected_ojama  # pending_p2 - pending_p1


def test_simple_generation_p2_fires() -> None:
    """2P が score を増やすと 1P に ojama が送られる。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    chain_score = 1400  # 20 ojama + leftover
    expected_ojama, _ = _score_to_ojama_count(chain_score)

    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=5.0)
    snap = tracker.update_from_score(score_p1=0, score_p2=chain_score, t_sec=6.0)

    assert snap.total_generated_by_p2 == expected_ojama
    assert snap.pending_p1 == expected_ojama
    assert snap.pending_p2 == 0
    assert snap.net_ojama_balance == -expected_ojama  # 1P 不利


def test_score_below_threshold_is_ignored() -> None:
    """CHAIN_FIRE_MIN_SCORE 未満の差分は ojama を生成しない。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=0.0)
    small_delta = CHAIN_FIRE_MIN_SCORE - 1
    snap = tracker.update_from_score(
        score_p1=small_delta, score_p2=0, t_sec=1.0,
    )
    assert snap.pending_p2 == 0
    assert snap.total_generated_by_p1 == 0


# ============================
# 2. 相殺テスト
# ============================

def test_offset_partial_p1_fires_larger() -> None:
    """1P が 2P より大きい連鎖 → 1P pending を全消し、余剰を 2P pending に積む。

    シナリオ:
        フレーム1: 2P が先に連鎖 → 1P に pending X
        フレーム2: 1P が連鎖完了 (chain_p1=True) → 相殺 → 余剰は 2P へ
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    score_2p = 2800   # 40 ojama → 1P pending=40
    score_1p_chain = 4200  # 60 ojama (1P の連鎖分)
    expected_2p_gen, _ = _score_to_ojama_count(score_2p)
    expected_1p_gen, _ = _score_to_ojama_count(score_1p_chain)

    # フレーム1: 2P 発火、1P は静止
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=5.0)
    tracker.update_from_score(
        score_p1=0, score_p2=score_2p, t_sec=6.0,
    )
    # この時点で p1.pending == expected_2p_gen
    snap_before = tracker.get_snapshot(t_sec=6.0)
    assert snap_before.pending_p1 == expected_2p_gen

    # フレーム2: 1P が score 増加 (生成) + chain_p1=True (相殺トリガー)
    # chain は False→True で相殺を実行する
    tracker.update_from_score(
        score_p1=score_1p_chain, score_p2=score_2p, t_sec=7.0,
        chain_p1=True, chain_p2=False,
    )
    snap = tracker.update_from_score(
        score_p1=score_1p_chain, score_p2=score_2p, t_sec=7.1,
        chain_p1=False, chain_p2=False,
    )
    # 1P が送った分 expected_1p_gen が、自分に向かう pending expected_2p_gen を
    # 相殺 → 余剰 = expected_1p_gen - expected_2p_gen が 2P.pending に入る
    expected_surplus = expected_1p_gen - expected_2p_gen
    if expected_surplus >= 0:
        assert snap.pending_p1 == 0
        assert snap.pending_p2 == expected_surplus, (
            f"pending_p2={snap.pending_p2} != expected_surplus={expected_surplus}"
        )
    else:
        # 1P が逆に足りなかった場合は pending_p1 が残る
        assert snap.pending_p2 == 0


def test_offset_exact_cancel() -> None:
    """同量の相互発火で両者 pending が 0 になる。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    same_score = 2100  # 30 ojama (= rock 1 個)
    expected_ojama, _ = _score_to_ojama_count(same_score)

    # step1: 2P 発火 → 1P pending
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(
        score_p1=0, score_p2=same_score, t_sec=2.0,
    )
    # step2: 1P 発火 + chain_p1 完了
    tracker.update_from_score(
        score_p1=same_score, score_p2=same_score, t_sec=3.0,
        chain_p1=True,
    )
    snap = tracker.update_from_score(
        score_p1=same_score, score_p2=same_score, t_sec=3.1,
        chain_p1=False,
    )
    # 両者 expected_ojama ずつ生成 & 相殺 → net=0
    assert snap.net_ojama_balance == 0, (
        f"net_balance={snap.net_ojama_balance} should be 0"
    )


# ============================
# 3. 全消しテスト
# ============================

def test_all_clear_bonus_applied_on_next_chain() -> None:
    """all_clear_pending_p1=True 状態で 1P が次の連鎖を発火すると 2100pt ボーナスが載る。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 全消し持越しを手動でセット (内部 _p1 へアクセス)
    tracker._p1.all_clear_pending = True  # type: ignore[attr-defined]

    base_score = 700  # 10 ojama (700 // 70)
    # 全消しボーナス込みの合計 score
    total_effective = base_score + ALL_CLEAR_BONUS
    expected_ojama, _ = _score_to_ojama_count(total_effective)

    # 発火
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=5.0)
    snap = tracker.update_from_score(
        score_p1=base_score, score_p2=0, t_sec=6.0,
    )

    assert snap.total_generated_by_p1 == expected_ojama, (
        f"generated={snap.total_generated_by_p1} != expected={expected_ojama} "
        f"(all_clear_bonus included)"
    )
    assert snap.pending_p2 == expected_ojama
    # 全消しフラグが消費された
    assert not snap.all_clear_pending_p1


def test_all_clear_bonus_not_applied_if_not_set() -> None:
    """全消し持越しがなければボーナスは加算されない。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    base_score = 700
    expected_ojama, _ = _score_to_ojama_count(base_score)

    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=5.0)
    snap = tracker.update_from_score(
        score_p1=base_score, score_p2=0, t_sec=6.0,
    )
    assert snap.total_generated_by_p1 == expected_ojama
    assert snap.pending_p2 == expected_ojama


# ============================
# 4. score_to_ojama leftover carry-over 整合テスト
# ============================

def test_leftover_carries_over_multiple_fires() -> None:
    """複数回連鎖でも leftover が正しく引き継がれる。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 1 回目: 100 点 → 1 ojama + leftover 30
    score1 = 100
    r1 = score_to_ojama(score1, prev_leftover=0)
    # 2 回目: 100 点 + leftover 30 = 130 → 1 ojama + leftover 60
    score2 = 100
    r2 = score_to_ojama(score2, prev_leftover=r1.leftover_score)

    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=0.0)
    tracker.update_from_score(score_p1=score1, score_p2=0, t_sec=1.0)
    snap = tracker.update_from_score(score_p1=score1 + score2, score_p2=0, t_sec=2.0)

    expected_total = r1.ojama_count + r2.ojama_count
    assert snap.total_generated_by_p1 == expected_total, (
        f"total_generated={snap.total_generated_by_p1} != {expected_total}"
    )
    assert snap.leftover_p1 == r2.leftover_score


# ============================
# 5. confidence テスト
# ============================

def test_confidence_score_ocr_only_no_visual() -> None:
    """visible_ojama なしのときは CONFIDENCE_SCORE_OCR_ONLY。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=0.0)
    snap = tracker.update_from_score(
        score_p1=0, score_p2=0, t_sec=1.0,
    )
    assert snap.confidence == pytest.approx(CONFIDENCE_SCORE_OCR_ONLY)


def test_confidence_increases_when_visual_agrees() -> None:
    """visible_ojama が pending と一致するときは CONFIDENCE_VISUAL_AGREE。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=0.0)
    # p1 pending=0 に対して visible=0 → 一致
    snap = tracker.update_from_score(
        score_p1=0, score_p2=0, t_sec=1.0,
        visible_ojama_p1=0, visible_ojama_p2=0,
    )
    assert snap.confidence == pytest.approx(CONFIDENCE_VISUAL_AGREE)


def test_confidence_drops_on_visual_mismatch() -> None:
    """visible_ojama が pending と大きく乖離するときは confidence が低下する。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=0.0)
    # pending_p1=0 に対して visible=100 → 大きな乖離
    snap = tracker.update_from_score(
        score_p1=0, score_p2=0, t_sec=1.0,
        visible_ojama_p1=VISIBLE_OJAMA_MISMATCH_THRESHOLD + 10,
    )
    expected_conf = max(
        0.0,
        CONFIDENCE_SCORE_OCR_ONLY - CONFIDENCE_VISUAL_MISMATCH_PENALTY,
    )
    assert snap.confidence == pytest.approx(expected_conf)


# ============================
# 6. reset テスト
# ============================

def test_reset_clears_all_ledgers() -> None:
    """reset() で全帳簿がクリアされる。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=0.0)
    tracker.update_from_score(score_p1=5000, score_p2=3000, t_sec=1.0)
    snap_before = tracker.get_snapshot(t_sec=1.0)
    assert snap_before.total_generated_by_p1 > 0

    tracker.reset()
    snap_after = tracker.get_snapshot(t_sec=0.0)
    assert snap_after.pending_p1 == 0
    assert snap_after.pending_p2 == 0
    assert snap_after.total_generated_by_p1 == 0
    assert snap_after.total_generated_by_p2 == 0
    assert snap_after.leftover_p1 == 0
    assert snap_after.leftover_p2 == 0


# ============================
# 7. get_snapshot テスト
# ============================

def test_get_snapshot_returns_current_state() -> None:
    """get_snapshot() はイベントなしでも現在の状態を返す。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=0.0)
    tracker.update_from_score(score_p1=2100, score_p2=0, t_sec=1.0)
    snap_event = tracker.get_snapshot(t_sec=1.5)
    snap_no_event = tracker.get_snapshot(t_sec=2.0)
    # どちらも同じ pending_p2
    assert snap_event.pending_p2 == snap_no_event.pending_p2


# ============================
# 8. overflow_risk テスト
# ============================

def test_overflow_risk_triggers_above_threshold() -> None:
    """pending が OJAMA_MAX_DROP_PER_TURN 以上で overflow_risk=True。"""
    tracker = OjamaAccountingTracker(
        overflow_threshold=OJAMA_MAX_DROP_PER_TURN,
    )
    tracker.reset()

    # rock 1 個 = 30 ojama を 2 回送る
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=0.0)
    tracker.update_from_score(score_p1=2100, score_p2=0, t_sec=1.0)
    snap1 = tracker.update_from_score(score_p1=4200, score_p2=0, t_sec=2.0)
    # 30 + 30 = 60 >= 30 → overflow_risk_p2=True
    assert snap1.overflow_risk_p2 is True
    assert snap1.overflow_risk_p1 is False


def test_overflow_risk_false_below_threshold() -> None:
    """pending が threshold 未満なら overflow_risk=False。"""
    tracker = OjamaAccountingTracker(
        overflow_threshold=OJAMA_MAX_DROP_PER_TURN,
    )
    tracker.reset()
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=0.0)
    # 10 ojama のみ (閾値 30 未満)
    snap = tracker.update_from_score(score_p1=700, score_p2=0, t_sec=1.0)
    assert snap.overflow_risk_p2 is False


# ============================
# 9. net_ojama_balance 符号テスト
# ============================

def test_net_balance_positive_means_p1_advantage() -> None:
    """net_balance = pending_p2 - pending_p1 が正なら 1P 有利。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=0.0)
    snap = tracker.update_from_score(score_p1=4200, score_p2=0, t_sec=1.0)
    expected_ojama, _ = _score_to_ojama_count(4200)
    assert snap.net_ojama_balance == expected_ojama  # pending_p2 が大きい = 1P 有利


def test_net_balance_negative_means_p2_advantage() -> None:
    """net_balance が負なら 2P 有利 (1P に多くの ojama が向かう)。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=0.0)
    snap = tracker.update_from_score(score_p1=0, score_p2=4200, t_sec=1.0)
    expected_ojama, _ = _score_to_ojama_count(4200)
    assert snap.net_ojama_balance == -expected_ojama


# ============================
# 10. total_generated / total_offset 累積整合テスト
# ============================

def test_total_generated_cumulative() -> None:
    """複数発火のたびに total_generated が累積される。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=0.0)
    tracker.update_from_score(score_p1=700, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=1400, score_p2=0, t_sec=2.0)
    snap = tracker.update_from_score(score_p1=2100, score_p2=0, t_sec=3.0)

    # 3 回とも 700 差分
    r1 = score_to_ojama(700, prev_leftover=0)
    r2 = score_to_ojama(700, prev_leftover=r1.leftover_score)
    r3 = score_to_ojama(700, prev_leftover=r2.leftover_score)
    expected_total = r1.ojama_count + r2.ojama_count + r3.ojama_count

    assert snap.total_generated_by_p1 == expected_total


def test_total_offset_accumulated() -> None:
    """相殺が発生した分だけ total_offset が記録される。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 2P が先に 30 ojama 送る
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=0.0)
    tracker.update_from_score(score_p1=0, score_p2=2100, t_sec=1.0)
    # 1P が 30 ojama 送り chain 完了 → 相殺
    tracker.update_from_score(
        score_p1=2100, score_p2=2100, t_sec=2.0,
        chain_p1=True,
    )
    snap = tracker.update_from_score(
        score_p1=2100, score_p2=2100, t_sec=2.1,
        chain_p1=False,
    )

    expected_2p_gen, _ = _score_to_ojama_count(2100)
    expected_1p_gen, _ = _score_to_ojama_count(2100)
    # 相殺量 = min(p1_sent, p1.pending_before_offset)
    # p1 の pending は 2P が送った expected_2p_gen
    # p1 が送った pending (= 1P が生成) は expected_1p_gen
    # 相殺 = min(expected_1p_gen, expected_2p_gen)
    expected_offset = min(expected_1p_gen, expected_2p_gen)
    assert snap.total_offset_by_p1 == expected_offset


# ============================
# OjamaAccountSnapshot の型テスト
# ============================

def test_snapshot_is_frozen_dataclass() -> None:
    """OjamaAccountSnapshot は frozen dataclass で不変。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    snap = tracker.get_snapshot(t_sec=0.0)
    assert isinstance(snap, OjamaAccountSnapshot)
    with pytest.raises((AttributeError, TypeError)):
        snap.pending_p1 = 999  # type: ignore[misc]


def test_snapshot_all_fields_present() -> None:
    """必要なフィールドがすべて存在する。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    snap = tracker.get_snapshot(t_sec=1.0)
    # 全フィールドが属性として存在するか確認
    required_fields = [
        "t_sec", "pending_p1", "pending_p2",
        "total_generated_by_p1", "total_generated_by_p2",
        "total_offset_by_p1", "total_offset_by_p2",
        "total_dropped_to_p1", "total_dropped_to_p2",
        "net_ojama_balance",
        "overflow_risk_p1", "overflow_risk_p2",
        "confidence",
        "leftover_p1", "leftover_p2",
        "all_clear_pending_p1", "all_clear_pending_p2",
    ]
    for field in required_fields:
        assert hasattr(snap, field), f"フィールド {field!r} が存在しない"


# ============================
# Step1.5 ① 落下 total_dropped テスト
# ============================

def test_drop_total_dropped_increases_on_ojama_increase() -> None:
    """① 盤面内おじゃま増分 = 落下量として total_dropped_to_p2 が増える。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 2P へ 10 ojama を送る
    chain_score = 700  # 10 ojama
    expected_ojama, _ = _score_to_ojama_count(chain_score)

    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=chain_score, score_p2=0, t_sec=2.0)
    snap_before = tracker.get_snapshot(t_sec=2.0)
    assert snap_before.pending_p2 == expected_ojama

    # 2P 盤面に expected_ojama 個のおじゃまが落下した状況をシミュレート
    empty_board = _make_empty_board()
    board_with_ojama = _make_board_with_ojama(expected_ojama)
    tracker.update_from_boards(
        board_p1=empty_board,
        board_p2=board_with_ojama,
    )

    snap_after = tracker.get_snapshot(t_sec=3.0)
    assert snap_after.total_dropped_to_p2 == expected_ojama, (
        f"total_dropped_to_p2={snap_after.total_dropped_to_p2} "
        f"!= expected={expected_ojama}"
    )


def test_drop_pending_reduced_on_ojama_increase() -> None:
    """① 落下後は pending が減少する (落下分 pending から消える)。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    chain_score = 1400  # ~20 ojama
    expected_ojama, _ = _score_to_ojama_count(chain_score)

    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=chain_score, score_p2=0, t_sec=2.0)

    # 半分が落下
    drop_count = expected_ojama // 2
    empty_board = _make_empty_board()
    board_half_ojama = _make_board_with_ojama(drop_count)
    tracker.update_from_boards(
        board_p1=empty_board,
        board_p2=board_half_ojama,
    )

    snap = tracker.get_snapshot(t_sec=3.0)
    expected_remaining = expected_ojama - drop_count
    assert snap.pending_p2 == expected_remaining, (
        f"pending_p2={snap.pending_p2} != expected_remaining={expected_remaining}"
    )
    assert snap.total_dropped_to_p2 == drop_count


def test_ledger_balance_generation_equals_offset_plus_dropped_plus_pending() -> None:
    """① 帳簿一致: 生成 = 相殺 + 落下 + 残 pending。

    シナリオ:
        1P が 30 ojama 生成 → 2P へ送信
        2P 盤面に 10 個落下 (visible 増)
        相殺なし
        → 生成 30 = 相殺 0 + 落下 10 + 残 pending 20
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    chain_score = 2100  # = 30 ojama
    expected_ojama, _ = _score_to_ojama_count(chain_score)

    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=chain_score, score_p2=0, t_sec=2.0)

    # 10 個が落下
    drop_count = 10
    empty_board = _make_empty_board()
    board_p2 = _make_board_with_ojama(drop_count)
    tracker.update_from_boards(
        board_p1=empty_board,
        board_p2=board_p2,
    )

    snap = tracker.get_snapshot(t_sec=3.0)
    generated = snap.total_generated_by_p1
    offset = snap.total_offset_by_p1
    dropped = snap.total_dropped_to_p2
    remaining = snap.pending_p2

    assert generated == expected_ojama
    assert generated == offset + dropped + remaining, (
        f"帳簿不一致: 生成={generated} != 相殺={offset} + 落下={dropped} + 残={remaining}"
    )


def test_drop_sanity_clamp_caps_large_increase() -> None:
    """① 異常に大きいおじゃま増分は DROP_SANITY_CLAMP で clamp される。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # pending を大量に積む (DROP_SANITY_CLAMP * 3 = 90 ojama)
    large_score = DROP_SANITY_CLAMP * 3 * OJAMA_RATE_STANDARD  # = 6300
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=large_score, score_p2=0, t_sec=2.0)

    # DROP_SANITY_CLAMP * 2 + 5 個が一度に落下した (異常値)
    abnormal_drop = DROP_SANITY_CLAMP * 2 + 5
    empty_board = _make_empty_board()
    board_p2 = _make_board_with_ojama(abnormal_drop)
    tracker.update_from_boards(
        board_p1=empty_board,
        board_p2=board_p2,
    )

    snap = tracker.get_snapshot(t_sec=3.0)
    # clamp により total_dropped <= DROP_SANITY_CLAMP
    assert snap.total_dropped_to_p2 <= DROP_SANITY_CLAMP, (
        f"total_dropped_to_p2={snap.total_dropped_to_p2} > clamp={DROP_SANITY_CLAMP}"
    )


def test_drop_ojama_decrease_does_not_add_dropped() -> None:
    """① おじゃまが減った (連鎖消去) 場合は total_dropped は増えない。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 先に 10 個落下させた状態にする
    chain_score = 700
    expected_ojama, _ = _score_to_ojama_count(chain_score)
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=chain_score, score_p2=0, t_sec=2.0)
    empty_board = _make_empty_board()
    board_p2 = _make_board_with_ojama(expected_ojama)
    tracker.update_from_boards(board_p1=empty_board, board_p2=board_p2)
    snap_after_drop = tracker.get_snapshot(t_sec=3.0)
    dropped_before = snap_after_drop.total_dropped_to_p2

    # おじゃまが減った (連鎖で消えた)
    empty_board_p2 = _make_empty_board()
    tracker.update_from_boards(board_p1=empty_board, board_p2=empty_board_p2)
    snap_after_clear = tracker.get_snapshot(t_sec=4.0)

    # total_dropped は増えてはいけない
    assert snap_after_clear.total_dropped_to_p2 == dropped_before, (
        f"total_dropped should not increase on ojama decrease: "
        f"{snap_after_clear.total_dropped_to_p2} != {dropped_before}"
    )


# ============================
# Step1.5 ② 全消し自動検出テスト
# ============================

def test_all_clear_auto_detected_from_empty_board() -> None:
    """② 全 EMPTY 盤面 + score > 0 で all_clear_pending が自動セットされる。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 先に score を積んでおく (score > 0 が全消し判定の条件)
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=700, score_p2=0, t_sec=2.0)

    # 1P 盤面が全 EMPTY (全消し状態) + score > 0
    empty_board = _make_empty_board()
    tracker.update_from_boards(
        board_p1=empty_board,
        board_p2=empty_board,
        score_p1=700,
        score_p2=0,
    )

    snap = tracker.get_snapshot(t_sec=3.0)
    assert snap.all_clear_pending_p1 is True, (
        "全 EMPTY 盤面 + score > 0 で all_clear_pending_p1 がセットされるべき"
    )


def test_all_clear_not_set_when_color_puyo_present() -> None:
    """② 色ぷよがあれば全消しとみなされない。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=700, score_p2=0, t_sec=2.0)

    board_with_color = _make_board_with_color_puyo(color=1, count=4)
    empty_board = _make_empty_board()
    tracker.update_from_boards(
        board_p1=board_with_color,
        board_p2=empty_board,
        score_p1=700,
        score_p2=0,
    )

    snap = tracker.get_snapshot(t_sec=3.0)
    assert snap.all_clear_pending_p1 is False, (
        "色ぷよあり盤面では all_clear_pending_p1 はセットされないべき"
    )


def test_all_clear_not_set_when_score_zero() -> None:
    """② score=0 (試合冒頭) では全消しとみなされない。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)

    empty_board = _make_empty_board()
    tracker.update_from_boards(
        board_p1=empty_board,
        board_p2=empty_board,
        score_p1=0,  # score=0 なので全消しでない
        score_p2=0,
    )

    snap = tracker.get_snapshot(t_sec=2.0)
    assert snap.all_clear_pending_p1 is False
    assert snap.all_clear_pending_p2 is False


def test_all_clear_bonus_applied_after_auto_detection() -> None:
    """② 自動検出した all_clear_pending が次の連鎖発火でボーナスとして加算される。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # step1: score 加算
    base_score = 700
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=base_score, score_p2=0, t_sec=2.0)

    # step2: 全 EMPTY 盤面で全消し自動検出
    empty_board = _make_empty_board()
    tracker.update_from_boards(
        board_p1=empty_board,
        board_p2=empty_board,
        score_p1=base_score,
        score_p2=0,
    )
    snap_mid = tracker.get_snapshot(t_sec=2.5)
    assert snap_mid.all_clear_pending_p1 is True

    # step3: 次の連鎖で全消しボーナスが乗る
    next_chain_score = 700  # base 700 + AC 2100 = 2800 effective
    total_effective = next_chain_score + ALL_CLEAR_BONUS
    expected_ojama, _ = _score_to_ojama_count(total_effective)

    # prev score を base_score に設定済なので、base_score → base_score+700 で差分=700
    tracker.update_from_score(
        score_p1=base_score + next_chain_score, score_p2=0, t_sec=3.0,
    )
    snap = tracker.get_snapshot(t_sec=3.0)
    assert snap.total_generated_by_p1 >= expected_ojama - 1, (
        f"全消しボーナス込み生成量が期待値未満: {snap.total_generated_by_p1} < {expected_ojama}"
    )
    # 全消しフラグが消費された
    assert snap.all_clear_pending_p1 is False


# ============================
# Step1.5 ③ 相殺タイミングテスト
# ============================

def test_offset_via_update_accounting_with_chain() -> None:
    """③ update_accounting_with_chain() で chain 確定後に相殺が実行される。

    update_from_score() では chain=False で呼んで相殺を起こさず、
    update_accounting_with_chain() で初めて相殺を実行するフローをテスト。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 2P が先に 30 ojama 送る (1P に pending)
    chain_score_2p = 2100  # 30 ojama
    expected_2p_gen, _ = _score_to_ojama_count(chain_score_2p)

    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=0, score_p2=chain_score_2p, t_sec=2.0)
    snap_before_offset = tracker.get_snapshot(t_sec=2.0)
    assert snap_before_offset.pending_p1 == expected_2p_gen

    # 1P が score を増やす (chain=False で相殺なし)
    chain_score_1p = 2100  # 30 ojama
    tracker.update_from_score(
        score_p1=chain_score_1p, score_p2=chain_score_2p, t_sec=3.0,
        chain_p1=False,  # 相殺エッジ発火させない
    )
    snap_before_chain = tracker.get_snapshot(t_sec=3.0)
    # まだ相殺されていない
    assert snap_before_chain.pending_p1 > 0

    # chain 確定後に update_accounting_with_chain() で相殺
    snap_after_chain = tracker.update_accounting_with_chain(
        t_sec=3.1,
        chain_p1=True,
        chain_p2=False,
    )
    # 相殺後 pending_p1 は減少 (全消しなら 0)
    assert snap_after_chain.pending_p1 < snap_before_chain.pending_p1, (
        f"相殺後に pending_p1 が減少しているべき: "
        f"before={snap_before_chain.pending_p1} after={snap_after_chain.pending_p1}"
    )


def test_update_accounting_with_chain_no_double_offset_on_continuous_true() -> None:
    """③ update_accounting_with_chain() の連続 True 呼出でエッジ検出 (二重相殺なし)。

    chain_p1=True を 2 回連続で呼んでも、2 回目は False→True エッジなので相殺不発。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 2P が 30 ojama 送る
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=0, score_p2=2100, t_sec=2.0)
    # 1P が 30 ojama 生成
    tracker.update_from_score(score_p1=2100, score_p2=2100, t_sec=3.0)

    # 1回目: chain_p1=True で相殺 (False→True エッジ)
    snap1 = tracker.update_accounting_with_chain(
        t_sec=3.1, chain_p1=True, chain_p2=False,
    )
    pending_after_first = snap1.pending_p1

    # 2回目: chain_p1=True のまま (True→True なのでエッジなし = 相殺不発)
    snap2 = tracker.update_accounting_with_chain(
        t_sec=3.2, chain_p1=True, chain_p2=False,
    )
    assert snap2.pending_p1 == pending_after_first, (
        f"二重相殺が発生: pending_p1 が変化 {pending_after_first} -> {snap2.pending_p1}"
    )


# ============================
# Step1.5 ④ クロスチェック乖離検知テスト
# ============================

def test_crosscheck_mismatch_detected_via_update_from_boards() -> None:
    """④ update_from_boards で visible_ojama が保存され、
    update_accounting_with_chain の confidence に乖離が反映される。

    pending と visible_ojama の乖離 > VISIBLE_OJAMA_MISMATCH_THRESHOLD で
    confidence が CONFIDENCE_SCORE_OCR_ONLY - CONFIDENCE_VISUAL_MISMATCH_PENALTY になる。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 1P に向かう pending をゼロのまま維持
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)

    # 盤面に大量のおじゃまを置く (pending=0 と大きく乖離)
    empty_board = _make_empty_board()
    # VISIBLE_OJAMA_MISMATCH_THRESHOLD + 10 個: 明らかな乖離
    large_ojama = VISIBLE_OJAMA_MISMATCH_THRESHOLD + 10
    board_with_ojama = _make_board_with_ojama(large_ojama)
    tracker.update_from_boards(
        board_p1=board_with_ojama,  # p1 に大量おじゃま
        board_p2=empty_board,
    )

    # update_accounting_with_chain 経由で confidence を確認
    snap = tracker.update_accounting_with_chain(
        t_sec=2.0, chain_p1=False, chain_p2=False,
    )
    expected_conf = max(
        0.0,
        CONFIDENCE_SCORE_OCR_ONLY - CONFIDENCE_VISUAL_MISMATCH_PENALTY,
    )
    assert snap.confidence == pytest.approx(expected_conf), (
        f"乖離時の confidence={snap.confidence} != expected={expected_conf}"
    )


def test_crosscheck_confidence_agrees_when_boards_match_pending() -> None:
    """④ visible_ojama と pending が一致 (乖離小) のとき confidence が CONFIDENCE_VISUAL_AGREE。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 1P に向かう pending = 0 (2P は何も生成しない)
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)

    # 盤面もおじゃまゼロ (pending=0 と一致)
    empty_board = _make_empty_board()
    tracker.update_from_boards(
        board_p1=empty_board,
        board_p2=empty_board,
    )

    snap = tracker.update_accounting_with_chain(
        t_sec=2.0, chain_p1=False, chain_p2=False,
    )
    assert snap.confidence == pytest.approx(CONFIDENCE_VISUAL_AGREE), (
        f"一致時 confidence={snap.confidence} != {CONFIDENCE_VISUAL_AGREE}"
    )


# ============================
# Step1.5 STABLE ガードテスト (連鎖中誤検知防止)
# ============================

def test_chain_guard_prevents_drop_during_chain() -> None:
    """STABLE ガード: 連鎖中 (is_chain=True) に盤面内おじゃまが増えても pending が減算されない。

    シナリオ:
        1P が 30 ojama を 2P へ送信 (pending_p2=30)
        2P 盤面で連鎖エフェクト中に一時的におじゃまが 5 個表れる (ノイズ増分)
        is_chain_p2=True でガードするため pending_p2 は変化しない
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 1P が 30 ojama 送る
    chain_score = 2100  # = 30 ojama
    expected_ojama, _ = _score_to_ojama_count(chain_score)
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=chain_score, score_p2=0, t_sec=2.0)
    snap_before = tracker.get_snapshot(t_sec=2.0)
    assert snap_before.pending_p2 == expected_ojama

    # 連鎖中に 2P 盤面でおじゃまが一時的に 5 個出現 (認識ノイズ想定)
    empty_board = _make_empty_board()
    noise_board = _make_board_with_ojama(5)
    tracker.update_from_boards(
        board_p1=empty_board,
        board_p2=noise_board,
        is_chain_p1=False,
        is_chain_p2=True,   # 2P は連鎖中 → ガード ON
    )

    snap_after = tracker.get_snapshot(t_sec=3.0)
    assert snap_after.pending_p2 == expected_ojama, (
        f"連鎖中に pending_p2 が誤減算: before={expected_ojama} after={snap_after.pending_p2}"
    )
    assert snap_after.total_dropped_to_p2 == 0, (
        f"連鎖中に total_dropped_to_p2 が誤計上: {snap_after.total_dropped_to_p2}"
    )


def test_chain_guard_stable_resume_counts_net_drop() -> None:
    """STABLE ガード: 連鎖中→STABLE 復帰時に正味落下のみが 1 回計上される。

    シナリオ:
        pending_p2 = 30
        連鎖中フレーム: 盤面内おじゃまが 0→5→10→5 と変動 (ノイズ)
            → prev_visible_ojama = 0 のまま保持 (ガードにより更新なし)
        STABLE 復帰フレーム: 盤面内おじゃまが 12 個に確定
            → delta = 12 - 0 = 12 が正味落下として計上
        期待値: total_dropped_to_p2 = 12, pending_p2 = 30 - 12 = 18
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 30 ojama を 2P へ送る
    chain_score = 2100  # = 30 ojama
    expected_ojama, _ = _score_to_ojama_count(chain_score)
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=chain_score, score_p2=0, t_sec=2.0)

    empty_board = _make_empty_board()

    # 連鎖中フレーム①: 盤面内 5 個 (ノイズ) → ガードでスキップ
    tracker.update_from_boards(
        board_p1=empty_board,
        board_p2=_make_board_with_ojama(5),
        is_chain_p2=True,
    )
    # 連鎖中フレーム②: 盤面内 10 個 (ノイズ) → ガードでスキップ
    tracker.update_from_boards(
        board_p1=empty_board,
        board_p2=_make_board_with_ojama(10),
        is_chain_p2=True,
    )
    # 連鎖中フレーム③: 盤面内 5 個に戻る (ノイズ) → ガードでスキップ
    tracker.update_from_boards(
        board_p1=empty_board,
        board_p2=_make_board_with_ojama(5),
        is_chain_p2=True,
    )

    # STABLE 復帰: 落下が確定し盤面内 12 個
    # prev_visible_ojama = 0 のまま → delta = 12 - 0 = 12 が 1 回だけ計上
    net_drop = 12
    tracker.update_from_boards(
        board_p1=empty_board,
        board_p2=_make_board_with_ojama(net_drop),
        is_chain_p2=False,   # STABLE 復帰
    )

    snap = tracker.get_snapshot(t_sec=5.0)
    assert snap.total_dropped_to_p2 == net_drop, (
        f"STABLE 復帰後の total_dropped_to_p2={snap.total_dropped_to_p2} != {net_drop}"
    )
    expected_remaining = expected_ojama - net_drop
    assert snap.pending_p2 == expected_remaining, (
        f"STABLE 復帰後の pending_p2={snap.pending_p2} != {expected_remaining}"
    )


def test_chain_guard_p1_p2_independent() -> None:
    """STABLE ガード: 1P と 2P のガードは独立している。

    2P が連鎖中でもガードされない 1P 側の落下は正常に計上される。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 2P が 1P へ 10 ojama 送る
    chain_score = 700  # 10 ojama
    expected_ojama_p1, _ = _score_to_ojama_count(chain_score)
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=0, score_p2=chain_score, t_sec=2.0)

    # 1P に落下: is_chain_p1=False (STABLE) → 計上される
    # 2P は連鎖中: is_chain_p2=True → 2P 側はスキップ
    drop_p1 = expected_ojama_p1  # 全落下
    tracker.update_from_boards(
        board_p1=_make_board_with_ojama(drop_p1),
        board_p2=_make_board_with_ojama(10),  # 2P ノイズ (ガードでスキップ)
        is_chain_p1=False,
        is_chain_p2=True,
    )

    snap = tracker.get_snapshot(t_sec=3.0)
    # 1P 側: 正常に計上
    assert snap.total_dropped_to_p1 == drop_p1, (
        f"1P 落下計上: total_dropped_to_p1={snap.total_dropped_to_p1} != {drop_p1}"
    )
    # 2P 側: ガードで計上なし
    assert snap.total_dropped_to_p2 == 0, (
        f"2P 連鎖中は落下計上されないべき: total_dropped_to_p2={snap.total_dropped_to_p2}"
    )


def test_chain_guard_no_effect_when_stable() -> None:
    """STABLE ガード: is_chain=False の場合は従来通り落下を計上する (後退テスト)。

    ガード追加で STABLE 時の従来挙動が変わらないことを確認。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    chain_score = 700  # 10 ojama
    expected_ojama, _ = _score_to_ojama_count(chain_score)
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=chain_score, score_p2=0, t_sec=2.0)

    # STABLE フレーム (is_chain=False): 落下計上が行われる
    empty_board = _make_empty_board()
    board_with_ojama = _make_board_with_ojama(expected_ojama)
    tracker.update_from_boards(
        board_p1=empty_board,
        board_p2=board_with_ojama,
        is_chain_p1=False,
        is_chain_p2=False,  # default と同じ
    )

    snap = tracker.get_snapshot(t_sec=3.0)
    assert snap.total_dropped_to_p2 == expected_ojama, (
        f"STABLE 時の落下計上が変化: {snap.total_dropped_to_p2} != {expected_ojama}"
    )
    assert snap.pending_p2 == 0, (
        f"STABLE 時の pending 減算が変化: {snap.pending_p2} != 0"
    )


# ============================
# 修正A: 試合境界 reset テスト
# ============================

def test_boundary_reset_on_score_decrease() -> None:
    """修正A: score 大幅減少で pending/leftover/all_clear が 0 にリセットされる。

    シナリオ:
        1P が 2P に 30 ojama 送る (pending_p2=30)
        次フレームで 2P score が 31000 → 150 に激減 (試合切り替え)
        → pending_p2 が 0 にリセットされる (境界検知)
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 1P が 30 ojama を 2P へ送る
    chain_score = 2100  # = 30 ojama
    expected_ojama, _ = _score_to_ojama_count(chain_score)
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=chain_score, score_p2=0, t_sec=2.0)
    snap_before = tracker.get_snapshot(t_sec=2.0)
    assert snap_before.pending_p2 == expected_ojama

    # 2P score が急減 (試合切り替え)
    snap_after = tracker.update_from_score(
        score_p1=chain_score,
        score_p2=150,       # 0 < prev(0) は境界でないが、p1 score を下げてテスト
        t_sec=3.0,
    )
    # pending_p2 はまだ変化なし (2P側の境界未発生)
    # 代わりに 1P score を激減させて 1P 側境界をテストする
    tracker.reset()
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    # 2P が 1P に 30 ojama 送る
    tracker.update_from_score(score_p1=0, score_p2=chain_score, t_sec=2.0)
    snap_mid = tracker.get_snapshot(t_sec=2.0)
    assert snap_mid.pending_p1 == expected_ojama, (
        f"境界前 pending_p1={snap_mid.pending_p1} != {expected_ojama}"
    )

    # 1P score が大幅に下がる (試合境界)
    high_score = 31000
    tracker.update_from_score(score_p1=high_score, score_p2=chain_score, t_sec=3.0)
    snap_high = tracker.get_snapshot(t_sec=3.0)
    # まだリセット未発生 (増加しただけ)
    # 次フレームで激減
    snap_reset = tracker.update_from_score(
        score_p1=150,           # 31000 → 150 = SCORE_RESET_THRESHOLD(500) 超え
        score_p2=chain_score,
        t_sec=4.0,
    )
    assert snap_reset.pending_p1 == 0, (
        f"境界後 pending_p1={snap_reset.pending_p1} should be 0"
    )


def test_boundary_reset_prev_score_updated_to_new() -> None:
    """修正A: 境界後の prev_score が新スコアに更新され、次フレームの増分が正しく計上される。

    設計: score 減少で境界検知した際、prev_score を新スコアに更新する。
    これにより次フレームの差分が「戻りジャンプ全体」ではなく「新試合の増分のみ」になる。

    シナリオ:
        フレーム1: 1P score = 0 (初期化)
        フレーム2: 1P score = 31000 (連鎖生成)
        フレーム3: 1P score = 150 (試合切り替え。31000→150 で境界検知)
                   → _p1.pending=0 (1P 側受取リセット)
                   → prev_score_p1 = 150 に更新
        フレーム4: 1P score = 800 (差分 = 800 - 150 = 650)
                   → 31000→800 の差分でなく 150→800 の差分 650 が生成計上される
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=31000, score_p2=0, t_sec=2.0)
    snap_after_big = tracker.get_snapshot(t_sec=2.0)
    generated_before_boundary = snap_after_big.total_generated_by_p1

    # 1P score が激減 → 境界リセット
    tracker.update_from_score(score_p1=150, score_p2=0, t_sec=3.0)

    # 次フレーム: 150 → 800 (差分 650)
    # 戻りジャンプ (31000→800 差分=−30200) ではなく 650 が計上される
    chain_delta = 800 - 150  # = 650
    expected_from_650, _ = _score_to_ojama_count(chain_delta)
    snap_next = tracker.update_from_score(score_p1=800, score_p2=0, t_sec=4.0)
    # 境界後の新規生成量
    new_generated = snap_next.total_generated_by_p1 - generated_before_boundary
    assert new_generated == expected_from_650, (
        f"境界後増分の計上: new_generated={new_generated} "
        f"!= expected_from_650={expected_from_650} "
        f"(戻りジャンプ分が計上されていないことを確認)"
    )


def test_boundary_no_reset_on_small_decrease() -> None:
    """修正A: SCORE_RESET_THRESHOLD 未満の減少は境界とみなさない。

    OCR ノイズで小さく下がっても pending はリセットされない。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    chain_score = 2100
    expected_ojama, _ = _score_to_ojama_count(chain_score)
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=0, score_p2=chain_score, t_sec=2.0)
    snap_before = tracker.get_snapshot(t_sec=2.0)
    assert snap_before.pending_p1 == expected_ojama

    # SCORE_RESET_THRESHOLD - 1 の減少 (境界未満)
    small_dec = SCORE_RESET_THRESHOLD - 1
    snap_after = tracker.update_from_score(
        score_p1=0,
        score_p2=chain_score - small_dec,
        t_sec=3.0,
    )
    # pending_p1 は変化しない (境界判定されない)
    assert snap_after.pending_p1 == expected_ojama, (
        f"小幅減少で境界誤検知: pending_p1={snap_after.pending_p1} != {expected_ojama}"
    )


# ============================
# 修正B: 理論落下 drain テスト
# ============================

def test_theory_drop_drains_pending_on_stable_resume() -> None:
    """修正B: tsumo_settled=True で pending から THEORY_DROP_PER_TURN 分が drain される。

    シナリオ:
        2P に 30 ojama 送信 (pending_p2=30)
        tsumo_settled_p2=True を渡す
        → pending_p2 = 30 - min(30, 30) = 0
        → total_dropped_to_p2 = 30
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    chain_score = 2100  # = 30 ojama
    expected_ojama, _ = _score_to_ojama_count(chain_score)
    assert expected_ojama == 30

    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=chain_score, score_p2=0, t_sec=2.0)

    # stable 復帰ターン
    snap = tracker.update_from_score(
        score_p1=chain_score,
        score_p2=0,
        t_sec=3.0,
        tsumo_settled_p2=True,
    )
    assert snap.pending_p2 == 0, (
        f"理論落下後 pending_p2={snap.pending_p2} should be 0"
    )
    assert snap.total_dropped_to_p2 == expected_ojama, (
        f"理論落下 total_dropped_to_p2={snap.total_dropped_to_p2} != {expected_ojama}"
    )


def test_theory_drop_partial_when_pending_less_than_cap() -> None:
    """修正B: pending が THEORY_DROP_PER_TURN より少ない場合は pending 分だけ drain する。

    pending=10 で tsumo_settled=True → drain=10 (30 ではなく)。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 700 点 = 10 ojama
    chain_score = 700
    expected_ojama, _ = _score_to_ojama_count(chain_score)
    assert expected_ojama == 10

    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=chain_score, score_p2=0, t_sec=2.0)

    snap = tracker.update_from_score(
        score_p1=chain_score,
        score_p2=0,
        t_sec=3.0,
        tsumo_settled_p2=True,
    )
    assert snap.pending_p2 == 0, (
        f"partial drain 後 pending_p2={snap.pending_p2} should be 0"
    )
    assert snap.total_dropped_to_p2 == expected_ojama, (
        f"partial drain total_dropped_to_p2={snap.total_dropped_to_p2} != {expected_ojama}"
    )


def test_theory_drop_no_over_drain_when_pending_zero() -> None:
    """修正B: pending=0 の時に tsumo_settled=True を渡しても過剰 drain しない (下限 0)。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    snap = tracker.update_from_score(
        score_p1=0,
        score_p2=0,
        t_sec=2.0,
        tsumo_settled_p1=True,
        tsumo_settled_p2=True,
    )
    assert snap.pending_p1 == 0
    assert snap.pending_p2 == 0
    assert snap.total_dropped_to_p1 == 0
    assert snap.total_dropped_to_p2 == 0


def test_theory_drop_multiple_turns_drain_incrementally() -> None:
    """修正B: 複数ターン連続で drain されるごとに pending が 30 ずつ減る。

    シナリオ:
        90 ojama を 2P へ送る (pending_p2=90 → PENDING_ABS_CAP(216) 未満なのでそのまま蓄積)
        3 ターン tsumo_settled_p2=True → 90 - 30 = 60 → 60 - 30 = 30 → 30 - 30 = 0
        ※ ON_FIELD_CAP(72) を超えた分は offboard として表現されるが drain は pending 全体に適用
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 90 ojama 相当: 2100*3 = 6300 点 (PENDING_ABS_CAP=216 未満なのでそのまま蓄積)
    large_score = 2100 * 3  # 90 ojama
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=large_score, score_p2=0, t_sec=2.0)

    snap0 = tracker.get_snapshot(t_sec=2.0)
    # pending は PENDING_ABS_CAP 以内 (90 < 216)
    assert snap0.pending_p2 <= PENDING_ABS_CAP, (
        f"abs_cap 超過: pending_p2={snap0.pending_p2} > PENDING_ABS_CAP={PENDING_ABS_CAP}"
    )

    pending_after_cap = snap0.pending_p2  # この時点の pending 値

    # ターン1
    snap1 = tracker.update_from_score(
        score_p1=large_score, score_p2=0, t_sec=3.0, tsumo_settled_p2=True,
    )
    expected1 = max(0, pending_after_cap - THEORY_DROP_PER_TURN)
    assert snap1.pending_p2 == expected1, (
        f"ターン1後 pending_p2={snap1.pending_p2} != {expected1}"
    )

    # ターン2
    snap2 = tracker.update_from_score(
        score_p1=large_score, score_p2=0, t_sec=4.0, tsumo_settled_p2=True,
    )
    expected2 = max(0, expected1 - THEORY_DROP_PER_TURN)
    assert snap2.pending_p2 == expected2, (
        f"ターン2後 pending_p2={snap2.pending_p2} != {expected2}"
    )

    # ターン3
    snap3 = tracker.update_from_score(
        score_p1=large_score, score_p2=0, t_sec=5.0, tsumo_settled_p2=True,
    )
    expected3 = max(0, expected2 - THEORY_DROP_PER_TURN)
    assert snap3.pending_p2 == expected3, (
        f"ターン3後 pending_p2={snap3.pending_p2} != {expected3}"
    )


# ============================
# 修正C: pending hard cap テスト
# ============================

def test_pending_hard_cap_prevents_overflow() -> None:
    """修正C: pending は PENDING_ABS_CAP (216) を超えない (絶対サニティ上限)。

    ON_FIELD_CAP(72) は超えてよい (offboard として表現される)。
    PENDING_ABS_CAP(216) = 絶対上限であり、これを超えたらclampされる。
    pending_capped = min(pending, ON_FIELD_CAP) は有界 (指標用)。
    offboard = max(0, pending - ON_FIELD_CAP) が正になることを確認。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # ON_FIELD_CAP を超えるが PENDING_ABS_CAP 未満の量: 100個相当
    score_for_100 = 100 * OJAMA_RATE_STANDARD  # = 7000
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    snap = tracker.update_from_score(score_p1=score_for_100, score_p2=0, t_sec=2.0)

    expected_ojama, _ = _score_to_ojama_count(score_for_100)
    # pending は ON_FIELD_CAP を超えられる
    assert snap.pending_p2 > ON_FIELD_CAP, (
        f"100個分の pending={snap.pending_p2} が ON_FIELD_CAP={ON_FIELD_CAP} を超えるべき"
    )
    # pending は PENDING_ABS_CAP 以下に収まる
    assert snap.pending_p2 <= PENDING_ABS_CAP, (
        f"abs_cap 超過: pending_p2={snap.pending_p2} > PENDING_ABS_CAP={PENDING_ABS_CAP}"
    )
    # pending_capped は ON_FIELD_CAP で有界
    assert snap.pending_p2_capped <= PENDING_HARD_CAP, (
        f"capped 超過: pending_p2_capped={snap.pending_p2_capped} > cap={PENDING_HARD_CAP}"
    )
    assert snap.pending_p2_capped == ON_FIELD_CAP, (
        f"capped 値={snap.pending_p2_capped} は ON_FIELD_CAP={ON_FIELD_CAP} になるべき"
    )
    # offboard は pending - ON_FIELD_CAP の超過分
    expected_offboard = snap.pending_p2 - ON_FIELD_CAP
    assert snap.offboard_p2 == expected_offboard, (
        f"offboard_p2={snap.offboard_p2} != expected={expected_offboard}"
    )


def test_pending_hard_cap_total_generated_preserved() -> None:
    """修正C: pending は PENDING_ABS_CAP で抑えられるが total_generated は元の値を保持する。

    ON_FIELD_CAP(72) を超えた分は offboard として表現され、pending 自体は蓄積される。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # ON_FIELD_CAP を確実に超える量 (10000点 / 70 ≈ 142 個)
    huge_score = 10000
    expected_generated, _ = _score_to_ojama_count(huge_score)
    assert expected_generated > ON_FIELD_CAP, (
        f"テスト前提: expected_generated={expected_generated} > ON_FIELD_CAP={ON_FIELD_CAP} が必要"
    )

    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    snap = tracker.update_from_score(score_p1=huge_score, score_p2=0, t_sec=2.0)

    # pending は PENDING_ABS_CAP 以内 (絶対上限)
    assert snap.pending_p2 <= PENDING_ABS_CAP
    # pending_capped は ON_FIELD_CAP で有界 (指標用)
    assert snap.pending_p2_capped <= ON_FIELD_CAP
    # total_generated は cap されない (元値保持)
    assert snap.total_generated_by_p1 == expected_generated, (
        f"total_generated が cap で削られている: {snap.total_generated_by_p1} != {expected_generated}"
    )


def test_capped_fields_in_snapshot() -> None:
    """修正C: OjamaAccountSnapshot に pending_p*_capped / net_balance_capped が存在する。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    snap = tracker.get_snapshot(t_sec=0.0)
    # フィールド存在確認
    assert hasattr(snap, "pending_p1_capped")
    assert hasattr(snap, "pending_p2_capped")
    assert hasattr(snap, "net_balance_capped")
    # 初期値はゼロ
    assert snap.pending_p1_capped == 0
    assert snap.pending_p2_capped == 0
    assert snap.net_balance_capped == 0


def test_capped_fields_bounded_range() -> None:
    """修正C: capped フィールドは -72..+72 の範囲に収まる (ON_FIELD_CAP 有界)。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 大量 ojama を 2P に送る
    huge_score = 15000
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    snap = tracker.update_from_score(score_p1=huge_score, score_p2=0, t_sec=2.0)

    assert 0 <= snap.pending_p2_capped <= ON_FIELD_CAP
    assert -ON_FIELD_CAP <= snap.net_balance_capped <= ON_FIELD_CAP


def test_capped_net_balance_normalization() -> None:
    """修正C: net_balance_capped を (x + 72) / 144 で 0-1 正規化できる。

    0-1 の範囲に収まることを確認する。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    huge_score = 10000
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    snap = tracker.update_from_score(score_p1=huge_score, score_p2=0, t_sec=2.0)

    normalized = (snap.net_balance_capped + ON_FIELD_CAP) / (2 * ON_FIELD_CAP)
    assert 0.0 <= normalized <= 1.0, (
        f"正規化値が範囲外: {normalized} (net_balance_capped={snap.net_balance_capped})"
    )


# ============================
# 正当大連鎖は境界誤判定しないテスト
# ============================

def test_large_chain_score_not_mistaken_for_boundary() -> None:
    """正当大連鎖 (+14850 相当) は試合境界と誤判定されない。

    score が 0 → 14850 に増加した場合、境界ではなく正当な生成として計上される。
    増分はスコア増加なので SCORE_RESET_THRESHOLD (500) による境界検知は発動しない。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    large_chain_score = 14850  # 実測: 大連鎖の正当スコア
    expected_ojama, _ = _score_to_ojama_count(large_chain_score)

    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    # 1P が大連鎖 (CHAIN 状態中)
    snap = tracker.update_from_score(
        score_p1=large_chain_score, score_p2=0, t_sec=2.0,
        chain_p1=True,  # 連鎖中
    )
    # pending は 0 にリセットされず、正当な生成量が計上されている
    assert snap.total_generated_by_p1 == expected_ojama, (
        f"大連鎖が境界誤判定: generated={snap.total_generated_by_p1} "
        f"(expected={expected_ojama})"
    )
    # pending_p2 >= 0 (正常に積まれている)
    assert snap.pending_p2 >= 0


def test_large_chain_9090_not_boundary() -> None:
    """正当大連鎖 (+9090) も境界誤判定しない。score 減少でないので検知されない。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    large_chain_score = 9090
    expected_ojama, _ = _score_to_ojama_count(large_chain_score)

    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    snap = tracker.update_from_score(
        score_p1=large_chain_score, score_p2=0, t_sec=2.0,
    )
    assert snap.total_generated_by_p1 == expected_ojama
    assert snap.pending_p2 > 0, (
        f"9090点大連鎖で pending_p2={snap.pending_p2} (0 になってはいけない)"
    )


# ============================
# snapshot 全フィールド存在 (後退テスト: 新フィールド含む)
# ============================

def test_snapshot_all_fields_present_with_new_fields() -> None:
    """OjamaAccountSnapshot に追加フィールド含む全フィールドが存在する。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()
    snap = tracker.get_snapshot(t_sec=1.0)
    required_fields = [
        "t_sec", "pending_p1", "pending_p2",
        "total_generated_by_p1", "total_generated_by_p2",
        "total_offset_by_p1", "total_offset_by_p2",
        "total_dropped_to_p1", "total_dropped_to_p2",
        "net_ojama_balance",
        "overflow_risk_p1", "overflow_risk_p2",
        "confidence",
        "leftover_p1", "leftover_p2",
        "all_clear_pending_p1", "all_clear_pending_p2",
        # 修正C 追加
        "pending_p1_capped", "pending_p2_capped", "net_balance_capped",
        # offboard フィールド追加
        "offboard_p1", "offboard_p2",
    ]
    for field in required_fields:
        assert hasattr(snap, field), f"フィールド {field!r} が存在しない"


# ============================
# (a) offboard: pending > ON_FIELD_CAP で offboard > 0、pending_capped == ON_FIELD_CAP
# ============================

def test_offboard_positive_when_pending_exceeds_on_field_cap() -> None:
    """(a) pending > ON_FIELD_CAP で offboard_p2 > 0 かつ pending_capped == ON_FIELD_CAP。

    ON_FIELD_CAP(72) を超える pending は offboard として表現される。
    pending_capped は有界(≤ ON_FIELD_CAP)、offboard = pending - ON_FIELD_CAP。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 100個相当 (ON_FIELD_CAP=72 を超える)
    score_for_100 = 100 * OJAMA_RATE_STANDARD
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    snap = tracker.update_from_score(score_p1=score_for_100, score_p2=0, t_sec=2.0)

    assert snap.pending_p2 > ON_FIELD_CAP, (
        f"テスト前提: pending_p2={snap.pending_p2} が ON_FIELD_CAP={ON_FIELD_CAP} を超えるべき"
    )
    assert snap.offboard_p2 > 0, (
        f"offboard_p2={snap.offboard_p2} は pending({snap.pending_p2}) > "
        f"ON_FIELD_CAP({ON_FIELD_CAP}) なので正になるべき"
    )
    assert snap.pending_p2_capped == ON_FIELD_CAP, (
        f"pending_p2_capped={snap.pending_p2_capped} == ON_FIELD_CAP={ON_FIELD_CAP} になるべき"
    )
    # offboard = pending - ON_FIELD_CAP
    assert snap.offboard_p2 == snap.pending_p2 - ON_FIELD_CAP, (
        f"offboard_p2={snap.offboard_p2} != pending({snap.pending_p2}) - "
        f"ON_FIELD_CAP({ON_FIELD_CAP})"
    )
    # 1P の offboard は 0 (何も送られていない)
    assert snap.offboard_p1 == 0


def test_offboard_zero_when_pending_within_on_field_cap() -> None:
    """(a) pending <= ON_FIELD_CAP では offboard_p2 == 0。"""
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 30個相当 (ON_FIELD_CAP=72 以内)
    score_for_30 = 30 * OJAMA_RATE_STANDARD
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    snap = tracker.update_from_score(score_p1=score_for_30, score_p2=0, t_sec=2.0)

    assert snap.pending_p2 <= ON_FIELD_CAP, (
        f"テスト前提: pending_p2={snap.pending_p2} <= ON_FIELD_CAP={ON_FIELD_CAP} のはず"
    )
    assert snap.offboard_p2 == 0, (
        f"offboard_p2={snap.offboard_p2} は pending <= ON_FIELD_CAP なので 0 になるべき"
    )


# ============================
# (b) offboard が理論落下 drain で減る
# ============================

def test_offboard_reduces_with_theory_drain() -> None:
    """(b) offboard が tsumo_settled=True の理論落下 drain で減少する。

    pending 派生のため、pending が drain されれば offboard も連動して減少する。
    シナリオ: 100個送信 → pending=100, offboard=28
              1ターン drain(30) → pending=70, offboard=max(0, 70-72)=0
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    score_for_100 = 100 * OJAMA_RATE_STANDARD
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    snap_before = tracker.update_from_score(
        score_p1=score_for_100, score_p2=0, t_sec=2.0,
    )
    assert snap_before.offboard_p2 > 0, "テスト前提: offboard > 0 が必要"

    # 1ターン理論落下 drain (THEORY_DROP_PER_TURN=30)
    snap_after = tracker.update_from_score(
        score_p1=score_for_100, score_p2=0, t_sec=3.0,
        tsumo_settled_p2=True,
    )
    # pending が 30 drain されれば offboard も連動して変わる (または 0 になる)
    expected_offboard = max(0, snap_before.pending_p2 - THEORY_DROP_PER_TURN - ON_FIELD_CAP)
    assert snap_after.offboard_p2 == expected_offboard, (
        f"drain後 offboard_p2={snap_after.offboard_p2} != expected={expected_offboard} "
        f"(pending_before={snap_before.pending_p2})"
    )


# ============================
# (c) 試合境界 (score 減少) で pending/offboard が 0 にreset (回帰防止テスト)
# ============================

def test_offboard_resets_to_zero_on_match_boundary() -> None:
    """(c) 試合境界 (score 大幅減少) で pending/offboard が 0 にreset される。

    ユーザー問題の回帰防止: 「試合終わっても O+60 がリセットされず続く」。
    pending が reset されれば offboard (pending 派生) も自動的に 0 になる。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # ON_FIELD_CAP を超える量を 2P に送る
    score_for_100 = 100 * OJAMA_RATE_STANDARD
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    snap_before = tracker.update_from_score(
        score_p1=score_for_100, score_p2=0, t_sec=2.0,
    )
    assert snap_before.pending_p2 > ON_FIELD_CAP, "テスト前提: pending > ON_FIELD_CAP が必要"
    assert snap_before.offboard_p2 > 0, "テスト前提: offboard > 0 が必要"

    # 2P score が大幅減少 (試合切り替え: 高スコア → 0 相当)
    # まず 2P に score を積む
    tracker.update_from_score(
        score_p1=score_for_100, score_p2=20000, t_sec=3.0,
    )
    # 次フレームで 2P score が激減 (試合境界)
    snap_reset = tracker.update_from_score(
        score_p1=score_for_100,
        score_p2=100,  # 20000 → 100 = SCORE_RESET_THRESHOLD(500) 超え
        t_sec=4.0,
    )
    # 2P 側の pending が 0 にリセットされる
    assert snap_reset.pending_p2 == 0, (
        f"試合境界後 pending_p2={snap_reset.pending_p2} should be 0 "
        f"(ユーザー問題: O+N がリセットされない)"
    )
    # offboard も 0 になる (pending 派生)
    assert snap_reset.offboard_p2 == 0, (
        f"試合境界後 offboard_p2={snap_reset.offboard_p2} should be 0 "
        f"(ユーザー問題の回帰防止)"
    )


# ============================
# (d) pending は PENDING_ABS_CAP で有界
# ============================

def test_pending_bounded_by_pending_abs_cap() -> None:
    """(d) pending は PENDING_ABS_CAP(216) で有界。

    PENDING_ABS_CAP を大きく超えようとしても clamp される。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # PENDING_ABS_CAP * 2 = 432 個相当の score
    huge_score = PENDING_ABS_CAP * 2 * OJAMA_RATE_STANDARD
    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    snap = tracker.update_from_score(score_p1=huge_score, score_p2=0, t_sec=2.0)

    assert snap.pending_p2 <= PENDING_ABS_CAP, (
        f"PENDING_ABS_CAP 超過: pending_p2={snap.pending_p2} > {PENDING_ABS_CAP}"
    )


def test_pending_abs_cap_greater_than_on_field_cap() -> None:
    """(d) 定数の関係: ON_FIELD_CAP == PENDING_HARD_CAP < PENDING_ABS_CAP。"""
    assert ON_FIELD_CAP == PENDING_HARD_CAP, (
        f"ON_FIELD_CAP={ON_FIELD_CAP} == PENDING_HARD_CAP={PENDING_HARD_CAP} が前提"
    )
    assert PENDING_ABS_CAP > ON_FIELD_CAP, (
        f"PENDING_ABS_CAP={PENDING_ABS_CAP} > ON_FIELD_CAP={ON_FIELD_CAP} が前提"
    )
    assert PENDING_ABS_CAP == ON_FIELD_CAP * 3, (
        f"PENDING_ABS_CAP={PENDING_ABS_CAP} == ON_FIELD_CAP*3={ON_FIELD_CAP * 3} が前提"
    )


# ============================
# (e) total_offset がスナップショットに出る
# ============================

def test_offset_appears_in_snapshot() -> None:
    """(e) 相殺発生後に total_offset_by_p* がスナップショットに反映される。

    1P が相殺すると total_offset_by_p1 が増加する。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()

    # 2P が先に 30 ojama を 1P に送る
    chain_score = 2100  # 30 ojama
    expected_ojama, _ = _score_to_ojama_count(chain_score)

    tracker.update_from_score(score_p1=0, score_p2=0, t_sec=1.0)
    tracker.update_from_score(score_p1=0, score_p2=chain_score, t_sec=2.0)
    snap_before_offset = tracker.get_snapshot(t_sec=2.0)
    assert snap_before_offset.total_offset_by_p1 == 0, "相殺前は offset=0"

    # 1P が同量生成して chain 完了 → 相殺発生
    tracker.update_from_score(
        score_p1=chain_score, score_p2=chain_score, t_sec=3.0,
        chain_p1=True,
    )
    snap_after_offset = tracker.update_from_score(
        score_p1=chain_score, score_p2=chain_score, t_sec=3.1,
        chain_p1=False,
    )
    # total_offset_by_p1 が増加している
    assert snap_after_offset.total_offset_by_p1 > 0, (
        f"相殺後 total_offset_by_p1={snap_after_offset.total_offset_by_p1} > 0 になるべき"
    )
    expected_offset_val = min(expected_ojama, expected_ojama)  # 同量なら全相殺
    assert snap_after_offset.total_offset_by_p1 == expected_offset_val, (
        f"total_offset_by_p1={snap_after_offset.total_offset_by_p1} "
        f"!= expected={expected_offset_val}"
    )
    # offboard_p1 は 0 (pending が 0 になっているはず)
    assert snap_after_offset.offboard_p1 == 0, (
        f"相殺後 offboard_p1={snap_after_offset.offboard_p1} should be 0"
    )
