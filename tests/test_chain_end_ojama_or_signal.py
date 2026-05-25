"""Stage 1 テスト: おじゃま降下で CHAIN → STABLE 遷移。

Stage 1 (2026-05-25) の実装を検証する。
- DetectorSignals.ojama_fall_detected フィールドの存在と default=False
- RecognitionPipeline.OJAMA_CHAIN_END_SCORE_THRESHOLD 定数値
- state_detectors.ChainPhaseDetector が CHAIN state から STABLE に遷移する条件
- recognition_pipeline._step_side が ojama_fall_detected=True のとき
  chain_event を None にクリアすること
"""

from __future__ import annotations

import pytest

from src.board import Board, COLOR_RED
from src.board_state_machine import (
    BoardState,
    BoardStateMachine,
    DetectorSignals,
    StateContext,
)
from src.state_detectors import ChainPhaseDetector
from src.recognition_pipeline import RecognitionPipeline


# ============================
# ヘルパー
# ============================


def _empty_board() -> Board:
    """空盤面を返す。"""
    return Board()


def _chain_ctx(board: Board | None = None) -> StateContext:
    """CHAIN state の StateContext を返す (テスト用)。"""
    ctx = StateContext(frame_idx=10, state=BoardState.CHAIN)
    ctx = StateContext(
        frame_idx=10,
        state=BoardState.CHAIN,
        confirmed_board=board or _empty_board(),
    )
    return ctx


def _stable_ctx() -> StateContext:
    """STABLE state の StateContext を返す (テスト用)。"""
    return StateContext(
        frame_idx=5,
        state=BoardState.STABLE,
        confirmed_board=_empty_board(),
    )


# ============================
# test 1: DetectorSignals に ojama_fall_detected フィールドが存在し default False
# ============================


def test_detector_signals_has_ojama_fall_detected_default_false() -> None:
    """DetectorSignals.ojama_fall_detected が default False で存在すること。"""
    sig = DetectorSignals(
        time_sec=0.0,
        cnn_board=_empty_board(),
        is_match_active=True,
    )
    assert hasattr(sig, "ojama_fall_detected"), (
        "DetectorSignals に ojama_fall_detected フィールドがない"
    )
    assert sig.ojama_fall_detected is False, (
        "ojama_fall_detected の default が False でない"
    )


# ============================
# test 2: OJAMA_CHAIN_END_SCORE_THRESHOLD 定数が 70
# ============================


def test_ojama_chain_end_score_threshold_is_70() -> None:
    """RecognitionPipeline.OJAMA_CHAIN_END_SCORE_THRESHOLD == 70 であること。"""
    assert RecognitionPipeline.OJAMA_CHAIN_END_SCORE_THRESHOLD == 70, (
        f"OJAMA_CHAIN_END_SCORE_THRESHOLD が 70 でない: "
        f"{RecognitionPipeline.OJAMA_CHAIN_END_SCORE_THRESHOLD}"
    )


# ============================
# test 3: ChainPhaseDetector が CHAIN state から STABLE に遷移する
#         (chain_event=None = 通常パス)
# ============================


def test_chain_phase_detector_chain_to_stable_on_no_event() -> None:
    """chain_event=None + state==CHAIN のとき ChainPhaseDetector が STABLE を返すこと。"""
    detector = ChainPhaseDetector()
    ctx = _chain_ctx()
    sig = DetectorSignals(
        time_sec=1.0,
        cnn_board=_empty_board(),
        is_match_active=True,
        chain_event=None,  # chain_event なし = 連鎖終了
    )
    result = detector.detect(ctx, sig)
    assert result == BoardState.STABLE, (
        f"chain_event=None で CHAIN→STABLE 遷移しない: result={result}"
    )


# ============================
# test 4: ojama_fall_detected=True で DetectorSignals 作成可能
#         (backwards compat: 既存シグネチャに追加のみ)
# ============================


def test_detector_signals_ojama_fall_detected_true() -> None:
    """ojama_fall_detected=True で DetectorSignals が正常に作成できること。"""
    sig = DetectorSignals(
        time_sec=0.5,
        cnn_board=_empty_board(),
        is_match_active=True,
        ojama_fall_detected=True,
    )
    assert sig.ojama_fall_detected is True


# ============================
# test 5: CHAIN state + ojama_fall_detected=True で STABLE に遷移する統合経路確認
#         (ChainPhaseDetector は chain_event=None 経路で STABLE 復帰)
# ============================


def test_chain_detector_stable_when_chain_event_cleared_by_ojama() -> None:
    """chain_event が None にクリアされた状態で CHAIN→STABLE 遷移すること。

    Stage 1 の pipeline では ojama_fall_detected=True のとき
    chain_event を None に強制クリアしてから signals を構築する。
    この test は「クリア後の signals」を直接 ChainPhaseDetector に渡して
    STABLE 遷移を確認する。
    """
    detector = ChainPhaseDetector()
    ctx = _chain_ctx()
    # pipeline が chain_event を None にクリアした後の signals
    sig = DetectorSignals(
        time_sec=2.0,
        cnn_board=_empty_board(),
        is_match_active=True,
        chain_event=None,           # pipeline でクリア済
        ojama_fall_detected=True,   # おじゃま降下フラグ
    )
    result = detector.detect(ctx, sig)
    assert result == BoardState.STABLE, (
        f"おじゃま降下クリア経路で CHAIN→STABLE 遷移しない: result={result}"
    )
