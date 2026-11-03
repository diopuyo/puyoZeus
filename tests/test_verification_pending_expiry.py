"""答え合わせ pending の連鎖失効フラグのテスト (段1: 単体)。

q第14試合1P の状況 (連鎖前3枚+連鎖後2枚の履歴で古い盤面が多数決に勝つ) を再現し、
既定OFFで従来と同一、ONで幻おじゃまを書かず、単発連鎖の通常補正は残ることを確認する。
"""
from __future__ import annotations

import pytest

from src import recognition_pipeline as rp
from src.board import Board, COLOR_OJAMA
from src.board_state_machine import BoardState
from tests.test_recognition_pipeline import _StubImageReader, _StubMatchDetector

STABLE, CHAIN, SETTLE = BoardState.STABLE, BoardState.CHAIN, BoardState.GRAVITY_SETTLE
PRE_FRAMES = 3   # 連鎖前 (古い盤面) の履歴枚数
POST_FRAMES = 2  # 連鎖後の履歴枚数 (CHAIN_VERIFY_FRAMES=5 の残り)


def _pipe(**kw) -> "rp.RecognitionPipeline":
    return rp.RecognitionPipeline(
        image_reader=_StubImageReader(Board(), Board()),  # type: ignore[arg-type]
        match_state_detector=_StubMatchDetector(True),  # type: ignore[arg-type]
        score_ocr=None, chain_tracker_1p=None, chain_tracker_2p=None,
        stable_frame_count=1, **kw,
    )


def _ojama_board() -> Board:
    b = Board()
    # 不一致セル数が閾値 CHAIN_VERIFY_MISMATCH_CELLS を超える古い盤面
    for r in range(12, 12 - (rp.RecognitionPipeline.CHAIN_VERIFY_MISMATCH_CELLS + 1), -1):
        b.set(r, 5, COLOR_OJAMA)
    return b


def _arm(pipe) -> None:
    """1回目の連鎖後 pending を作る (期待盤面 = 空)。"""
    pipe._chain_verify_pending_1p = {"expected": Board(), "cnn_history": []}


def _feed(pipe, states_boards) -> list:
    return [pipe._update_chain_estimate_verification("1P", s, b)
            for s, b in states_boards]


def _two_chain_sequence() -> list:
    """連鎖前3枚 (幻に見える古い盤面) → 2回目の連鎖 → 連鎖後2枚 (空)。"""
    seq = [(STABLE, _ojama_board())] * PRE_FRAMES
    seq += [(CHAIN, Board()), (SETTLE, Board())]
    seq += [(STABLE, Board())] * POST_FRAMES
    return seq


def test_flag_default_off() -> None:
    assert _pipe()._enable_verification_pending_chain_expiry is False


def test_off_reproduces_phantom_write() -> None:
    """OFF: 従来どおり古い盤面が多数決に勝ち、幻おじゃまで補正する (退行防止の基準)。"""
    pipe = _pipe()
    _arm(pipe)
    results = _feed(pipe, _two_chain_sequence())
    corrected = [r for r in results if r[0] == "verified_mismatch_corrected"]
    assert len(corrected) == 1
    assert corrected[0][1].get(12, 5) == COLOR_OJAMA
    assert pipe.verification_pending_expired_count == 0


def test_on_expires_stale_pending_no_phantom() -> None:
    """ON: 2回目の連鎖開始で pending 失効 → 幻おじゃまを書かない。"""
    pipe = _pipe(enable_verification_pending_chain_expiry=True)
    _arm(pipe)
    results = _feed(pipe, _two_chain_sequence())
    assert all(r == (None, None) for r in results)
    assert pipe._chain_verify_pending_1p is None
    assert pipe.verification_pending_expired_count == 1


@pytest.mark.parametrize("enable", [False, True])
def test_single_chain_still_corrects(enable: bool) -> None:
    """単発連鎖 (連鎖後 STABLE のみ5枚) の通常補正は ON でも従来どおり。"""
    pipe = _pipe(enable_verification_pending_chain_expiry=enable)
    _arm(pipe)
    results = _feed(pipe, [(STABLE, _ojama_board())] * 5)
    assert results[-1][0] == "verified_mismatch_corrected"
    assert results[-1][1].get(12, 5) == COLOR_OJAMA
    assert pipe.verification_pending_expired_count == 0


def test_on_non_chain_nonstable_keeps_pending() -> None:
    """ON でも TSUMO_FALL 等の連鎖以外の非STABLEでは失効させない (狭い条件)。"""
    pipe = _pipe(enable_verification_pending_chain_expiry=True)
    _arm(pipe)
    _feed(pipe, [(BoardState.TSUMO_FALL, Board()), (BoardState.OJAMA_FALL, Board())])
    assert pipe._chain_verify_pending_1p is not None


def test_load_default_plumbing() -> None:
    import inspect
    p = inspect.signature(rp.RecognitionPipeline.load_default).parameters
    assert p["enable_verification_pending_chain_expiry"].default is False
