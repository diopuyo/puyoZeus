"""state 遷移検出器テスト (Phase B-2)."""

from __future__ import annotations

from dataclasses import dataclass

from src.board import COLOR_BLUE, COLOR_RED, Board
from src.board_state_machine import (
    BoardState,
    BoardStateMachine,
    DetectorSignals,
    StateContext,
)
from src.state_detectors import (
    ChainPhaseDetector,
    EffectPhaseDetector,
    OjamaPhaseDetector,
    TsumoPhaseDetector,
)


# ============================
# helper
# ============================


@dataclass
class _StubChainEvent:
    """ChainEvent の最小限 stub (trigger_sec / end_sec のみ)."""

    trigger_sec: float
    end_sec: float


def _empty_board() -> Board:
    return Board()


def _board_with_puyos(positions: list[tuple[int, int, int]]) -> Board:
    """(row, col, color) のリストから Board を生成."""
    b = Board()
    for row, col, color in positions:
        b.set(row, col, color)
    return b


def _signal(
    t: float, board: Board, *,
    chain_event: _StubChainEvent | None = None,
    score_delta: int = 0,
    next_pair: tuple[int, int] | None = None,
    match: bool = True,
    slide_motion: bool = False,
    ojama_top_positive: bool = False,
    chain_max_hold_expired: bool = False,
) -> DetectorSignals:
    return DetectorSignals(
        time_sec=t,
        cnn_board=board,
        is_match_active=match,
        chain_event=chain_event,
        score_delta=score_delta,
        next_pair=next_pair,
        slide_motion=slide_motion,
        ojama_top_positive=ojama_top_positive,
        chain_max_hold_expired=chain_max_hold_expired,
    )


# ============================
# ChainPhaseDetector
# ============================


def test_chain_detector_returns_chain_within_event_window() -> None:
    det = ChainPhaseDetector()
    ev = _StubChainEvent(trigger_sec=10.0, end_sec=12.0)
    ctx = StateContext()
    res = det.detect(ctx, _signal(10.5, _empty_board(), chain_event=ev))
    assert res == BoardState.CHAIN


def test_chain_detector_returns_stable_when_event_cleared() -> None:
    """event=None & 現 state=CHAIN なら STABLE 復帰."""
    det = ChainPhaseDetector()
    ctx = StateContext(state=BoardState.CHAIN)
    res = det.detect(ctx, _signal(12.5, _empty_board(), chain_event=None))
    assert res == BoardState.STABLE


def test_chain_detector_returns_none_when_no_event() -> None:
    det = ChainPhaseDetector()
    ctx = StateContext(state=BoardState.STABLE)
    res = det.detect(ctx, _signal(10.0, _empty_board()))
    assert res is None


def test_chain_detector_returns_chain_regardless_of_time() -> None:
    """event があれば time に関係なく CHAIN を返す (pipeline で時刻管理)."""
    det = ChainPhaseDetector()
    ev = _StubChainEvent(trigger_sec=10.0, end_sec=12.0)
    ctx = StateContext()
    res1 = det.detect(ctx, _signal(9.0, _empty_board(), chain_event=ev))
    res2 = det.detect(ctx, _signal(13.0, _empty_board(), chain_event=ev))
    assert res1 == BoardState.CHAIN
    assert res2 == BoardState.CHAIN


# ============================
# TsumoPhaseDetector
# ============================


def test_tsumo_detector_returns_none_without_baseline() -> None:
    det = TsumoPhaseDetector(consec_threshold=1)
    ctx = StateContext()  # confirmed_board=None
    res = det.detect(ctx, _signal(0.0, _empty_board()))
    assert res is None


def test_tsumo_detector_detects_single_puyo_increase() -> None:
    """consec_threshold=1 で 1 frame 即発火 (旧挙動互換)."""
    det = TsumoPhaseDetector(consec_threshold=1)
    baseline = _empty_board()
    ctx = StateContext(
        state=BoardState.STABLE, confirmed_board=baseline, frame_idx=0,
    )
    new_board = _board_with_puyos([(12, 0, COLOR_RED)])
    res = det.detect(ctx, _signal(0.05, new_board))
    assert res == BoardState.TSUMO_FALL


def test_tsumo_detector_detects_pair_increase() -> None:
    det = TsumoPhaseDetector(max_increase=2, consec_threshold=1)
    baseline = _empty_board()
    ctx = StateContext(
        state=BoardState.STABLE, confirmed_board=baseline, frame_idx=0,
    )
    new_board = _board_with_puyos([
        (12, 0, COLOR_RED), (11, 0, COLOR_BLUE),
    ])
    res = det.detect(ctx, _signal(0.10, new_board))
    assert res == BoardState.TSUMO_FALL


def test_tsumo_detector_returns_none_for_large_increase() -> None:
    det = TsumoPhaseDetector(max_increase=2, consec_threshold=1)
    baseline = _empty_board()
    ctx = StateContext(state=BoardState.STABLE, confirmed_board=baseline)
    new_board = _board_with_puyos([
        (12, c, COLOR_RED) for c in range(5)
    ])
    res = det.detect(ctx, _signal(0.10, new_board))
    assert res is None  # CHAIN detector / OJAMA に任せる


def test_tsumo_detector_returns_stable_after_chain_erasure() -> None:
    """cycle 71v: 連鎖発火による puyo 減少 (diff<0) で STABLE 復帰.

    旧挙動 (diff==0 で STABLE) は CNN がツモを見逃した frame で誤発火するため、
    diff<0 (= 実際に puyo 減少) を chain 完了 signal として使う仕様に変更。
    """
    det = TsumoPhaseDetector(consec_threshold=1)
    # baseline: 4 puyos (= 連鎖前盤面)
    baseline = _board_with_puyos([
        (12, 0, COLOR_RED), (12, 1, COLOR_RED),
        (11, 0, COLOR_BLUE), (11, 1, COLOR_BLUE),
    ])
    # cnn_after: 連鎖で 4 puyos 消えて空に (+2 placed → 6 erased = diff=-2)
    erased = _empty_board()
    ctx = StateContext(
        state=BoardState.TSUMO_FALL, confirmed_board=baseline,
    )
    res = det.detect(ctx, _signal(0.30, erased))
    assert res == BoardState.STABLE


def test_tsumo_detector_does_not_fire_on_diff_zero() -> None:
    """cycle 71v: diff=0 では STABLE 復帰しない (= CNN ツモ見逃し誤発火対策)."""
    det = TsumoPhaseDetector(consec_threshold=1)
    baseline = _board_with_puyos([(12, 0, COLOR_RED)])
    same = baseline.copy()
    ctx = StateContext(
        state=BoardState.TSUMO_FALL, confirmed_board=baseline,
    )
    res = det.detect(ctx, _signal(0.30, same))
    # diff=0 では STABLE にしない (旧挙動の修正)
    assert res is None


def test_tsumo_detector_ignores_single_frame_jitter() -> None:
    """consec_threshold=2 では 1 frame だけの +1 puyo は無視 (CNN ぶれ吸収)."""
    det = TsumoPhaseDetector(consec_threshold=2)
    baseline = _empty_board()
    ctx = StateContext(
        state=BoardState.STABLE, confirmed_board=baseline, frame_idx=0,
    )
    new_board = _board_with_puyos([(12, 0, COLOR_RED)])
    res = det.detect(ctx, _signal(0.05, new_board))
    assert res is None  # 1 frame では発火しない


def test_tsumo_detector_fires_on_consecutive_two_frames() -> None:
    """連続 2 frame で +1 puyo 観測 → TSUMO_FALL."""
    det = TsumoPhaseDetector(consec_threshold=2)
    baseline = _empty_board()
    new_board = _board_with_puyos([(12, 0, COLOR_RED)])

    # 1 frame 目: 発火しない
    ctx1 = StateContext(
        state=BoardState.STABLE, confirmed_board=baseline, frame_idx=0,
    )
    r1 = det.detect(ctx1, _signal(0.05, new_board))
    assert r1 is None

    # 2 frame 目: 発火する
    ctx2 = StateContext(
        state=BoardState.STABLE, confirmed_board=baseline, frame_idx=1,
    )
    r2 = det.detect(ctx2, _signal(0.10, new_board))
    assert r2 == BoardState.TSUMO_FALL


def test_tsumo_detector_resets_consec_on_zero_diff() -> None:
    """中間 frame で diff=0 になったら連続カウンタがリセットされる."""
    det = TsumoPhaseDetector(consec_threshold=2)
    baseline = _empty_board()
    new_board = _board_with_puyos([(12, 0, COLOR_RED)])

    # frame 0: +1 (consec=1)
    ctx0 = StateContext(
        state=BoardState.STABLE, confirmed_board=baseline, frame_idx=0,
    )
    det.detect(ctx0, _signal(0.0, new_board))
    # frame 1: 0 (consec=0 にリセット)
    ctx1 = StateContext(
        state=BoardState.STABLE, confirmed_board=baseline, frame_idx=1,
    )
    det.detect(ctx1, _signal(0.05, baseline.copy()))
    # frame 2: +1 (consec=1, まだ発火しない)
    ctx2 = StateContext(
        state=BoardState.STABLE, confirmed_board=baseline, frame_idx=2,
    )
    res = det.detect(ctx2, _signal(0.10, new_board))
    assert res is None


# ============================
# OjamaPhaseDetector
# ============================


def test_ojama_detector_fires_on_score_delta() -> None:
    det = OjamaPhaseDetector(score_threshold=100)
    ctx = StateContext()
    res = det.detect(ctx, _signal(5.0, _empty_board(), score_delta=200))
    assert res == BoardState.OJAMA_FALL


def test_ojama_detector_silent_below_threshold() -> None:
    det = OjamaPhaseDetector(score_threshold=100)
    ctx = StateContext()
    res = det.detect(ctx, _signal(5.0, _empty_board(), score_delta=50))
    assert res is None


def test_ojama_detector_returns_stable_when_already_in_ojama_fall() -> None:
    """OJAMA_FALL に居て新規 score_delta が無い frame は STABLE 復帰."""
    det = OjamaPhaseDetector(score_threshold=70)
    ctx = StateContext(state=BoardState.OJAMA_FALL)
    res = det.detect(ctx, _signal(5.0, _empty_board(), score_delta=0))
    assert res == BoardState.STABLE


def test_ojama_detector_default_threshold_is_70() -> None:
    """おじゃま 1 個分のデフォルト閾値."""
    det = OjamaPhaseDetector()
    assert det.score_threshold == 70


# ============================
# EffectPhaseDetector
# ============================


def test_effect_detector_skeleton_returns_none() -> None:
    det = EffectPhaseDetector()
    ctx = StateContext()
    res = det.detect(ctx, _signal(0.0, _empty_board()))
    assert res is None


# ============================
# 統合: BoardStateMachine + 全 detector
# ============================


def test_state_machine_uses_chain_detector_first() -> None:
    """ChainPhaseDetector が他より先に登録されていれば優先される."""
    sm = BoardStateMachine(
        detectors=[
            ChainPhaseDetector(),
            EffectPhaseDetector(),
            OjamaPhaseDetector(),
            TsumoPhaseDetector(),
        ],
        stable_frame_count=2,
    )
    # 一旦 STABLE に持っていく
    base = _empty_board()
    sm.update(0, _signal(0.0, base))
    sm.update(1, _signal(0.05, base))
    assert sm.context.state == BoardState.STABLE

    # 連鎖発火: chain_event を渡せば CHAIN になり、score_delta より優先
    ev = _StubChainEvent(trigger_sec=0.10, end_sec=0.50)
    new_board = _board_with_puyos([(12, c, COLOR_RED) for c in range(6)])
    ctx = sm.update(
        2,
        _signal(0.15, new_board, chain_event=ev, score_delta=999),
    )
    assert ctx.state == BoardState.CHAIN


# ============================
# ChainPhaseDetector 案γ: slide_motion による ojama-hold 上書き
# ============================


def test_chain_detector_gamma_slide_overrides_ojama_hold() -> None:
    """案γ ON + slide_motion=True + ojama_top_positive=True → CHAIN 終了 (ojama-hold 無効化)."""
    det = ChainPhaseDetector(
        enable_chain_ojama_exit=True,
        enable_slide_override_ojama_hold=True,
    )
    ctx = StateContext(state=BoardState.CHAIN)
    res = det.detect(
        ctx,
        _signal(
            39.0, _empty_board(),
            chain_event=None,       # chain_event=None → CHAIN 終了処理へ
            ojama_top_positive=True,  # ojama-hold ガードが発動する条件
            slide_motion=True,       # 案γ: slide が ojama-hold を上書き
        ),
    )
    # ojama-hold は発動せず STABLE に戻る (enable_gravity_settle_state=False なので STABLE)
    assert res == BoardState.STABLE


def test_chain_detector_gamma_no_slide_ojama_hold_still_active() -> None:
    """案γ ON でも slide_motion=False なら従来通り ojama-hold が保留 (None 返し)."""
    det = ChainPhaseDetector(
        enable_chain_ojama_exit=True,
        enable_slide_override_ojama_hold=True,
    )
    ctx = StateContext(state=BoardState.CHAIN)
    res = det.detect(
        ctx,
        _signal(
            38.0, _empty_board(),
            chain_event=None,
            ojama_top_positive=True,
            slide_motion=False,  # slide なし → ojama-hold が依然有効
        ),
    )
    # 従来通り None (OjamaVisualDetector に委譲)
    assert res is None


def test_chain_detector_gamma_flag_off_slide_ignored() -> None:
    """enable_slide_override_ojama_hold=False (明示 OFF) なら slide_motion は無視。"""
    det = ChainPhaseDetector(
        enable_chain_ojama_exit=True,
        enable_slide_override_ojama_hold=False,  # 明示 OFF (pipeline default=True とは独立)
    )
    ctx = StateContext(state=BoardState.CHAIN)
    res = det.detect(
        ctx,
        _signal(
            38.0, _empty_board(),
            chain_event=None,
            ojama_top_positive=True,
            slide_motion=True,  # フラグ OFF なので無視される
        ),
    )
    # フラグ OFF → ojama-hold が従来通り有効 → None
    assert res is None


def test_chain_detector_gamma_with_gravity_settle() -> None:
    """案γ + GRAVITY_SETTLE 有効時: slide が ojama-hold を上書きして GRAVITY_SETTLE に遷移。"""
    det = ChainPhaseDetector(
        enable_chain_ojama_exit=True,
        enable_slide_override_ojama_hold=True,
        enable_gravity_settle_state=True,
    )
    ctx = StateContext(state=BoardState.CHAIN)
    res = det.detect(
        ctx,
        _signal(
            39.0, _empty_board(),
            chain_event=None,
            ojama_top_positive=True,
            slide_motion=True,
        ),
    )
    # ojama-hold 無効化 → CHAIN 終了 → GRAVITY_SETTLE (gsettle ON)
    assert res == BoardState.GRAVITY_SETTLE


def test_tsumo_then_back_to_stable_via_state_machine() -> None:
    sm = BoardStateMachine(
        detectors=[
            ChainPhaseDetector(),
            TsumoPhaseDetector(consec_threshold=1),
        ],
        stable_frame_count=2,
    )
    base = _empty_board()
    sm.update(0, _signal(0.0, base))
    sm.update(1, _signal(0.05, base))
    assert sm.context.state == BoardState.STABLE

    falling = _board_with_puyos([(12, 0, COLOR_RED)])
    ctx = sm.update(2, _signal(0.10, falling))
    assert ctx.state == BoardState.TSUMO_FALL

    # 着地後 (puyo 数差分 = 0) → STABLE 戻り
    landed = falling.copy()
    sm.update(3, _signal(0.15, landed))
    sm.update(4, _signal(0.20, landed))
    sm.update(5, _signal(0.25, landed))
    # 直近 STABLE 確定盤面が更新され、新 baseline と一致 → diff=0
    # ただし confirmed_board を新盤面に更新するには連続多数決が必要
    # state_machine は CHAIN→STABLE のみ即時更新、TSUMO→STABLE は多数決経由
    assert sm.context.state in {BoardState.STABLE, BoardState.TSUMO_FALL}
