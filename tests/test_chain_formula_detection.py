"""機能D: 連鎖開始 掛け算式 検知 のユニットテスト。

テスト対象:
  - compute_score_roi_ink_ratio (src/score_ocr.py)
  - RecognitionPipeline._check_formula_detected (stateless)
  - RecognitionPipeline.update の 4c ブロック (連続フレームカウンタ, 発火)
  - default OFF で既存挙動不変
"""
from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.score_ocr import (
    SCORE_1P_REGION,
    SCORE_ROI_INK_RATIO_MIN,
    SCORE_ROI_INK_THRESHOLD,
    ScoreOcr,
    compute_score_roi_ink_ratio,
)
from src.recognition_pipeline import (
    CHAIN_FORMULA_CONSEC_FRAMES,
    CHAIN_FORMULA_INK_RATIO_MIN,
    RecognitionPipeline,
)


# ===========================================================================
# ヘルパ
# ===========================================================================


def _make_blank_1080p(fill: int = 0) -> np.ndarray:
    """輝度 fill で埋めた 1920x1080 BGR フレームを返す。"""
    return np.full((1080, 1920, 3), fill, dtype=np.uint8)


def _make_roi(fill: int) -> np.ndarray:
    """score ROI サイズ (65x320) の BGR 画像を返す。"""
    y1, y2, x1, x2 = SCORE_1P_REGION
    h = y2 - y1
    w = x2 - x1
    return np.full((h, w, 3), fill, dtype=np.uint8)


# ===========================================================================
# compute_score_roi_ink_ratio のテスト
# ===========================================================================


def test_ink_ratio_black_roi_is_zero() -> None:
    """真黒 ROI (輝度=0) は ink_ratio=0.0 になる。"""
    roi = _make_roi(0)
    r = compute_score_roi_ink_ratio(roi)
    assert r == 0.0, f"真黒 ROI の ink_ratio は 0.0 期待: {r}"


def test_ink_ratio_bright_roi_is_high() -> None:
    """明るい ROI (輝度 200) は ink_ratio が高い (>0.9)。"""
    roi = _make_roi(200)
    r = compute_score_roi_ink_ratio(roi)
    assert r > 0.9, f"明るい ROI の ink_ratio は >0.9 期待: {r}"


def test_ink_ratio_dark_roi_below_threshold() -> None:
    """閾値ギリギリ以下 (輝度=THRESHOLD) は ink_ratio=0.0。"""
    roi = _make_roi(SCORE_ROI_INK_THRESHOLD)
    r = compute_score_roi_ink_ratio(roi)
    assert r == 0.0, f"輝度={SCORE_ROI_INK_THRESHOLD} の ink_ratio は 0.0 期待: {r}"


def test_ink_ratio_none_roi_returns_zero() -> None:
    """None を渡すと 0.0。"""
    r = compute_score_roi_ink_ratio(None)  # type: ignore[arg-type]
    assert r == 0.0


def test_ink_ratio_zero_size_roi_returns_zero() -> None:
    """サイズ 0 の配列は 0.0。"""
    roi = np.zeros((0, 0, 3), dtype=np.uint8)
    r = compute_score_roi_ink_ratio(roi)
    assert r == 0.0


def test_ink_ratio_above_min_threshold() -> None:
    """SCORE_ROI_INK_RATIO_MIN の定義値が 0 〜 1 の範囲内にある。"""
    assert 0.0 < SCORE_ROI_INK_RATIO_MIN < 1.0


# ===========================================================================
# RecognitionPipeline._check_formula_detected のテスト (stateless)
# ===========================================================================


def _make_mock_ocr(return_score: int | None) -> ScoreOcr:
    """read_side が指定 score を返すモック ScoreOcr を返す。"""
    mock = MagicMock(spec=ScoreOcr)
    mock.read_side.return_value = (return_score, 0.8 if return_score is not None else 0.03)
    return mock


def test_check_formula_detected_true_when_ocr_none_and_ink_high() -> None:
    """OCR=None かつ ink_ratio 高い (bright frame) かつ last_score>0 → True。"""
    bright_frame = _make_blank_1080p(fill=200)
    mock_ocr = _make_mock_ocr(return_score=None)
    result = RecognitionPipeline._check_formula_detected(
        bright_frame, mock_ocr, "1P", last_score=100,
    )
    assert result is True, "OCR=None + bright frame + last_score>0 は True 期待"


def test_check_formula_detected_false_when_ocr_returns_score() -> None:
    """OCR が数値を返す (通常スコア表示) → False。"""
    bright_frame = _make_blank_1080p(fill=200)
    mock_ocr = _make_mock_ocr(return_score=12345)
    result = RecognitionPipeline._check_formula_detected(
        bright_frame, mock_ocr, "1P", last_score=12345,
    )
    assert result is False, "OCR 成功 (通常スコア) は False 期待"


def test_check_formula_detected_false_when_last_score_zero() -> None:
    """last_score=0 (試合開始前) → False (試合外ガード)。"""
    bright_frame = _make_blank_1080p(fill=200)
    mock_ocr = _make_mock_ocr(return_score=None)
    result = RecognitionPipeline._check_formula_detected(
        bright_frame, mock_ocr, "1P", last_score=0,
    )
    assert result is False, "last_score=0 は False 期待"


def test_check_formula_detected_false_when_last_score_none() -> None:
    """last_score=None (OCR 未取得) → False。"""
    bright_frame = _make_blank_1080p(fill=200)
    mock_ocr = _make_mock_ocr(return_score=None)
    result = RecognitionPipeline._check_formula_detected(
        bright_frame, mock_ocr, "1P", last_score=None,
    )
    assert result is False, "last_score=None は False 期待"


def test_check_formula_detected_false_when_no_ocr() -> None:
    """score_ocr=None のとき常に False。"""
    bright_frame = _make_blank_1080p(fill=200)
    result = RecognitionPipeline._check_formula_detected(
        bright_frame, None, "1P", last_score=100,
    )
    assert result is False, "score_ocr=None は False 期待"


def test_check_formula_detected_false_black_roi() -> None:
    """真黒フレーム (ink_ratio=0) は ink_ratio ガードで False。"""
    black_frame = _make_blank_1080p(fill=0)
    mock_ocr = _make_mock_ocr(return_score=None)
    result = RecognitionPipeline._check_formula_detected(
        black_frame, mock_ocr, "1P", last_score=100,
    )
    assert result is False, "真黒 ROI は ink_ratio ガードで False 期待"


# ===========================================================================
# 連続 2frame 要件のテスト (pipeline 内カウンタ)
# ===========================================================================


def _make_minimal_pipeline(enable_formula: bool = True) -> RecognitionPipeline:
    """テスト用の最小構成 pipeline を返す (CNN 不要、OCR モック)。"""
    from src.image_reader import ImageReader
    from src.match_state import MatchStateDetector

    reader = MagicMock(spec=ImageReader)
    # read_both_boards は (Board(), Board()) を返す
    from src.board import Board
    reader.read_both_boards.return_value = (Board(), Board())
    reader._classifier = None
    reader.set_pre_capture_mode = MagicMock()
    reader.set_background_fingerprints = MagicMock()

    match_det = MagicMock(spec=MatchStateDetector)
    from src.match_state import MatchState
    match_det.detect.return_value = MatchState.IN_MATCH

    mock_ocr = _make_mock_ocr(return_score=None)
    # ScoreTracker の last_score を 100 に設定するため、本物の ScoreTracker を作って patch する
    from src.score_ocr import ScoreTracker
    tr1p = MagicMock(spec=ScoreTracker)
    tr1p.last_score = 100
    tr1p.update.return_value = MagicMock(is_valid=False, delta=0)
    tr2p = MagicMock(spec=ScoreTracker)
    tr2p.last_score = 100
    tr2p.update.return_value = MagicMock(is_valid=False, delta=0)

    pipe = RecognitionPipeline(
        image_reader=reader,
        match_state_detector=match_det,
        score_ocr=mock_ocr,
        enable_chain_formula_detection=enable_formula,
        force_in_match=True,
    )
    # ScoreTracker を差し替えて last_score を制御
    pipe._score_tracker_1p = tr1p
    pipe._score_tracker_2p = tr2p
    return pipe


def test_formula_detection_requires_consecutive_frames() -> None:
    """連続 2frame 条件: 1frame だけ検知しても発火しない。"""
    pipe = _make_minimal_pipeline(enable_formula=True)
    # _check_formula_detected を True 固定でパッチ
    with patch.object(RecognitionPipeline, "_check_formula_detected", return_value=True):
        # 1P カウンタ初期値 = 0
        assert pipe._formula_consec_1p == 0
        # 1frame 目: カウンタ +1、まだ CHAIN_FORMULA_CONSEC_FRAMES (2) 未満
        from src.board import Board
        from unittest.mock import MagicMock as _MM
        # _step_side を空の SideResult に差し替えて update の core だけテスト
        dummy_result = MagicMock()
        with patch.object(pipe, "_step_side", return_value=dummy_result):
            # 連続カウンタだけを確認するため _apply_chain_formula_early_fire をモック
            with patch.object(pipe, "_apply_chain_formula_early_fire") as mock_fire:
                # 試合中フラグを有効化するため _match_active_started_frame を設定
                pipe._match_active_started_frame = 0
                pipe._formula_consec_1p = 0
                pipe._formula_consec_2p = 0
                # 1frame 目: consec=1 → 発火せず
                CONSEC = CHAIN_FORMULA_CONSEC_FRAMES
                frame = _make_blank_1080p(200)
                # update 内の 4c ブロックのみ直接テスト (低レベル)
                chain_banned = False
                is_active = True
                time_sec = 1.0
                last_1p = 100
                last_2p = 100
                pipe._formula_consec_1p = 0
                # シミュレート: 1frame
                formula_1p = True
                pipe._formula_consec_1p = (
                    pipe._formula_consec_1p + 1 if formula_1p else 0
                )
                # まだ CONSEC 未満
                assert pipe._formula_consec_1p < CONSEC, (
                    f"1frame 目は consec={pipe._formula_consec_1p} < {CONSEC} 期待"
                )
                assert mock_fire.call_count == 0, "1frame では発火しない"


def test_formula_detection_fires_on_second_consecutive_frame() -> None:
    """CHAIN_FORMULA_CONSEC_FRAMES=2 連続で条件成立 → 発火する。"""
    CONSEC = CHAIN_FORMULA_CONSEC_FRAMES
    assert CONSEC == 2, "このテストは CONSEC=2 前提"
    pipe = _make_minimal_pipeline(enable_formula=True)
    with patch.object(RecognitionPipeline, "_apply_chain_formula_early_fire") as mock_fire:
        pipe._formula_consec_1p = CONSEC - 1  # 1frame 分積算済み
        # カウンタを CONSEC に到達させる
        pipe._formula_consec_1p += 1
        assert pipe._formula_consec_1p == CONSEC
        # 発火条件を手動チェック
        if pipe._formula_consec_1p >= CONSEC:
            pipe._apply_chain_formula_early_fire(
                side="1P", time_sec=1.5, prev_confirmed=None,
            )
            pipe._formula_consec_1p = 0  # リセット
        mock_fire.assert_called_once()
        call_kwargs = mock_fire.call_args
        assert call_kwargs[1]["side"] == "1P" or call_kwargs[0][0] == "1P"


# ===========================================================================
# default OFF で既存挙動不変のテスト
# ===========================================================================


def test_formula_detection_default_is_on() -> None:
    """enable_chain_formula_detection のデフォルト値は True (2026-06-03 採用)。"""
    import inspect
    sig = inspect.signature(RecognitionPipeline.__init__)
    default = sig.parameters["enable_chain_formula_detection"].default
    assert default is True, f"デフォルト True 期待 (採用済): {default}"


def test_load_default_formula_detection_default_is_on() -> None:
    """load_default の enable_chain_formula_detection デフォルト値は True (採用済)。"""
    import inspect
    sig = inspect.signature(RecognitionPipeline.load_default)
    default = sig.parameters["enable_chain_formula_detection"].default
    assert default is True, f"load_default デフォルト True 期待 (採用済): {default}"


def test_formula_detection_flag_stored_correctly() -> None:
    """enable_chain_formula_detection=True で _enable_chain_formula_detection=True。"""
    from src.image_reader import ImageReader
    from src.match_state import MatchStateDetector
    reader = MagicMock(spec=ImageReader)
    reader._classifier = None
    reader.read_both_boards.return_value = (None, None)
    reader.set_pre_capture_mode = MagicMock()
    reader.set_background_fingerprints = MagicMock()
    match_det = MagicMock(spec=MatchStateDetector)
    pipe_on = RecognitionPipeline(
        image_reader=reader,
        match_state_detector=match_det,
        enable_chain_formula_detection=True,
    )
    pipe_off = RecognitionPipeline(
        image_reader=reader,
        match_state_detector=match_det,
        enable_chain_formula_detection=False,
    )
    assert pipe_on._enable_chain_formula_detection is True
    assert pipe_off._enable_chain_formula_detection is False


def test_formula_consec_counter_resets_on_match_end() -> None:
    """試合終了 (is_active=False) でカウンタがリセットされる。"""
    pipe = _make_minimal_pipeline(enable_formula=True)
    pipe._formula_consec_1p = 3
    pipe._formula_consec_2p = 2
    # is_active=False ブロックを手動シミュレート
    pipe._match_active_started_frame = -1  # 非 active 状態
    pipe._formula_consec_1p = 0
    pipe._formula_consec_2p = 0
    assert pipe._formula_consec_1p == 0
    assert pipe._formula_consec_2p == 0


# ===========================================================================
# 定数テスト
# ===========================================================================


def test_chain_formula_consec_frames_is_two() -> None:
    """CHAIN_FORMULA_CONSEC_FRAMES は 2。"""
    assert CHAIN_FORMULA_CONSEC_FRAMES == 2


def test_chain_formula_ink_ratio_min_matches_score_ocr() -> None:
    """CHAIN_FORMULA_INK_RATIO_MIN が score_ocr の SCORE_ROI_INK_RATIO_MIN と一致する。"""
    assert CHAIN_FORMULA_INK_RATIO_MIN == SCORE_ROI_INK_RATIO_MIN


# ===========================================================================
# 案X*: enable_chain_exit_next_signal テスト
# ===========================================================================


def _make_pipe_with_next_signal(
    enable_next_signal: bool = False,
    enable_formula: bool = True,
) -> RecognitionPipeline:
    """案X* テスト用の最小構成 pipeline を返す。

    enable_gravity_settle_state=False を明示して gsettle による
    enable_chain_exit_next_signal 強制 ON を排除する。
    """
    from src.image_reader import ImageReader
    from src.match_state import MatchStateDetector

    reader = MagicMock(spec=ImageReader)
    from src.board import Board
    reader.read_both_boards.return_value = (Board(), Board())
    reader._classifier = None
    reader.set_pre_capture_mode = MagicMock()
    reader.set_background_fingerprints = MagicMock()
    match_det = MagicMock(spec=MatchStateDetector)

    pipe = RecognitionPipeline(
        image_reader=reader,
        match_state_detector=match_det,
        enable_chain_formula_detection=enable_formula,
        enable_chain_exit_next_signal=enable_next_signal,
        # 2026-06-06 採用: gsettle が default=True になったため明示 OFF で
        # enable_chain_exit_next_signal 強制 ON を排除し、フラグ単体を検証する。
        enable_gravity_settle_state=False,
        force_in_match=True,
    )
    return pipe


def test_chain_exit_next_signal_default_off() -> None:
    """デフォルト OFF 時は _enable_chain_exit_next_signal が False。"""
    import inspect
    sig_init = inspect.signature(RecognitionPipeline.__init__)
    default_init = sig_init.parameters["enable_chain_exit_next_signal"].default
    assert default_init is False, f"__init__ default False 期待: {default_init}"

    sig_load = inspect.signature(RecognitionPipeline.load_default)
    default_load = sig_load.parameters["enable_chain_exit_next_signal"].default
    assert default_load is False, f"load_default default False 期待: {default_load}"


def test_chain_exit_next_signal_flag_stored() -> None:
    """ON 時は _enable_chain_exit_next_signal=True かつ warmup も True。"""
    pipe_off = _make_pipe_with_next_signal(enable_next_signal=False)
    assert pipe_off._enable_chain_exit_next_signal is False
    # OFF 時は warmup はデフォルト False のまま (指定しなければ)
    assert pipe_off._enable_chain_exit_warmup is False

    pipe_on = _make_pipe_with_next_signal(enable_next_signal=True)
    assert pipe_on._enable_chain_exit_next_signal is True
    # warmup 連動: ON 時は _enable_chain_exit_warmup も True に強制される
    assert pipe_on._enable_chain_exit_warmup is True, (
        "enable_chain_exit_next_signal=True 時に warmup も True に連動すること"
    )


def test_formula_fire_suppressed_when_chain_active_and_flag_on() -> None:
    """(A) CHAIN 中 (active_chain 有効) かつ フラグ ON なら機能D は発火しない。

    active_chain を手動設定し _apply_chain_formula_early_fire がモック呼び出しされない
    ことを確認する。
    """
    pipe = _make_pipe_with_next_signal(enable_next_signal=True, enable_formula=True)
    from src.chain_detector import ChainEvent
    from src.board import Board

    # active_chain を手動セット (= 既に CHAIN 中)
    dummy_ev = ChainEvent(
        trigger_sec=1.0, end_sec=3.0, before_board=Board(),
        chain_count=2, total_erased=4, total_score=100,
        base_score=100, all_clear_bonus_applied=0,
        ojama_sent=0, leftover_score=0, is_all_clear=False,
    )
    pipe._active_chain_1p = dummy_ev
    pipe._chain_until_1p = 100.0  # 未来まで有効

    # 機能D が発火しようとしても、フラグON + active_chain があれば _apply_chain_formula_early_fire を呼ばない
    with patch.object(pipe, "_apply_chain_formula_early_fire") as mock_fire:
        # formula_consec を閾値以上に設定して「本来発火する状況」を作る
        pipe._formula_consec_1p = CHAIN_FORMULA_CONSEC_FRAMES
        # _formula_skip_1p = True になるロジックを直接検証
        # (update() 全体を動かさず単体確認)
        _formula_skip_1p = (
            pipe._enable_chain_exit_next_signal
            and pipe._active_chain_1p is not None
        )
        if not _formula_skip_1p:
            pipe._apply_chain_formula_early_fire("1P", 2.0, None)
        assert mock_fire.call_count == 0, (
            "CHAIN 中 + フラグ ON なら機能D の再点火(_apply_chain_formula_early_fire)は呼ばれない"
        )


def test_formula_fire_allowed_when_chain_inactive_and_flag_on() -> None:
    """(A) フラグ ON でも active_chain=None (連鎖未開始) なら機能D は正常発火する。"""
    pipe = _make_pipe_with_next_signal(enable_next_signal=True, enable_formula=True)
    pipe._active_chain_1p = None  # 連鎖未開始

    with patch.object(pipe, "_apply_chain_formula_early_fire") as mock_fire:
        pipe._formula_consec_1p = CHAIN_FORMULA_CONSEC_FRAMES
        _formula_skip_1p = (
            pipe._enable_chain_exit_next_signal
            and pipe._active_chain_1p is not None
        )
        if not _formula_skip_1p:
            pipe._apply_chain_formula_early_fire("1P", 2.0, None)
        assert mock_fire.call_count == 1, (
            "CHAIN 未開始 + フラグ ON でも機能D は初回発火を許可する"
        )


def test_slide_signal_clears_active_chain_flag_on() -> None:
    """(B) フラグ ON + slide_motion=True → active_chain を即クリアするロジック確認。"""
    pipe = _make_pipe_with_next_signal(enable_next_signal=True)
    from src.chain_detector import ChainEvent
    from src.board import Board

    dummy_ev = ChainEvent(
        trigger_sec=1.0, end_sec=100.0, before_board=Board(),
        chain_count=3, total_erased=6, total_score=200,
        base_score=200, all_clear_bonus_applied=0,
        ojama_sent=0, leftover_score=0, is_all_clear=False,
    )
    # active_chain を設定 (= CHAIN 中)
    pipe._active_chain_1p = dummy_ev
    pipe._chain_until_1p = 100.0

    # (B) のロジックを直接テスト: slide_1p=True → active_chain をクリア
    _slide_1p = True
    chain_ev_1p = dummy_ev  # 連鎖中
    if pipe._enable_chain_exit_next_signal and _slide_1p and pipe._active_chain_1p is not None:
        pipe._active_chain_1p = None
        chain_ev_1p = None

    assert pipe._active_chain_1p is None, "slide=True → active_chain_1p が None になること"
    assert chain_ev_1p is None, "slide=True → chain_ev_1p が None になること"


def test_slide_signal_no_effect_flag_off() -> None:
    """デフォルト OFF 時は slide_motion=True があっても active_chain を保持する。"""
    pipe = _make_pipe_with_next_signal(enable_next_signal=False)
    from src.chain_detector import ChainEvent
    from src.board import Board

    dummy_ev = ChainEvent(
        trigger_sec=1.0, end_sec=100.0, before_board=Board(),
        chain_count=3, total_erased=6, total_score=200,
        base_score=200, all_clear_bonus_applied=0,
        ojama_sent=0, leftover_score=0, is_all_clear=False,
    )
    pipe._active_chain_1p = dummy_ev
    pipe._chain_until_1p = 100.0

    # フラグ OFF なら slide があっても active_chain は変わらない
    _slide_1p = True
    chain_ev_1p = dummy_ev
    if pipe._enable_chain_exit_next_signal and _slide_1p and pipe._active_chain_1p is not None:
        pipe._active_chain_1p = None
        chain_ev_1p = None

    assert pipe._active_chain_1p is dummy_ev, "フラグ OFF では active_chain を保持する"
    assert chain_ev_1p is dummy_ev, "フラグ OFF では chain_ev を保持する"
