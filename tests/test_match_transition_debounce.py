"""長時間劣化修正 A' (`enable_match_transition_debounce`) のテスト (2026-08-06)。

docs/LONGRUN_DEGRADATION_INVESTIGATION_2026-08-06.md §4追補 (第4機構)。

構成:
    1. `_resolve_match_active_debounce` (stateless純関数) の遷移表
       (フリッカー無視/N秒超で確定/確定時刻の意味論)
    2. フラグ既定値・reset()クリア
    3. backwards compat (フラグOFFでbit-identical)
    4. 統合: 実 pipeline.update() でのフリッカー吸収確認
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.board import Board
from src.recognition_pipeline import (
    MatchActiveDebounceState,
    RecognitionPipeline,
    _resolve_match_active_debounce,
)

# =============================================================================
# stub helpers (tests/test_recognition_pipeline.py の設計を踏襲、独立定義)
# =============================================================================


@dataclass
class _StubMatchResult:
    state: object


class _StubMatchDetector:
    """`in_match` を可変にできる MatchStateDetector スタブ (フリッカー再現用)。"""

    def __init__(self, in_match: bool = True) -> None:
        self.in_match = in_match

    def detect(self, frame: "np.ndarray") -> _StubMatchResult:
        from src.match_state import MatchState
        return _StubMatchResult(
            state=MatchState.IN_MATCH if self.in_match else MatchState.NOT_IN_MATCH,
        )


@dataclass
class _StubScoreZeroResult:
    both_zero: bool


class _StubScoreZeroDetector:
    """`both_zero` を可変にできる ScoreZeroDetector スタブ。

    hard_match_off (= raw_active/recent_active/sm_active を全て上書きする
    確定 inactive シグナル) を確実に起こすために使う。盤面が完全空
    (count_puyos()==0) の間は「既に puyo 観測」救済ロジックが働かないため、
    both_zero=True にすると is_active を確実に False にできる。
    """

    def __init__(self, both_zero: bool = False) -> None:
        self.both_zero = both_zero

    def detect(self, frame: "np.ndarray") -> _StubScoreZeroResult:
        return _StubScoreZeroResult(both_zero=self.both_zero)


class _StubImageReader:
    """固定 empty board を返すスタブ。"""

    def __init__(self, p1: Board, p2: Board) -> None:
        self._p1 = p1
        self._p2 = p2

    def read_both_boards(
        self, frame: "np.ndarray",
        p1_roi_offset: tuple[float, float] = (0.0, 0.0),
        p2_roi_offset: tuple[float, float] = (0.0, 0.0),
        skip_tier1_1p: bool = False, skip_tier1_2p: bool = False,
        telop_result: object | None = None,
    ) -> tuple[Board, Board]:
        return self._p1.copy(), self._p2.copy()


def _dummy_frame() -> "np.ndarray":
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _make_pipe(
    detector: "_StubMatchDetector", enable_debounce: bool = False,
    score_zero_detector: "_StubScoreZeroDetector | None" = None,
) -> RecognitionPipeline:
    """テスト用の最小構成 pipeline を組み立てる。"""
    reader = _StubImageReader(Board(), Board())
    return RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_zero_detector=score_zero_detector,  # type: ignore[arg-type]
        score_ocr=None, chain_tracker_1p=None, chain_tracker_2p=None,
        stable_frame_count=2,
        # force_in_match=True (既定) だと match_state_detector の結果が
        # is_active に一切反映されない (raw_active が常にTrue) ため、
        # detector.in_match のトグルでフリッカーを再現するには False が必須。
        force_in_match=False,
        enable_match_transition_debounce=enable_debounce,
    )


# =============================================================================
# 1. _resolve_match_active_debounce 遷移表
# =============================================================================


def test_debounce_raw_matches_confirmed_true_noop() -> None:
    """raw と confirmed が共に True で一致 → 変化なし。"""
    state = MatchActiveDebounceState(confirmed_active=True)
    new_state, transitioned, fi, ts = _resolve_match_active_debounce(
        state, True, 10, 1.0, debounce_sec=1.0,
    )
    assert transitioned is False
    assert new_state.confirmed_active is True
    assert new_state.pending_active is None


def test_debounce_raw_matches_confirmed_false_noop() -> None:
    """raw と confirmed が共に False で一致 → 変化なし。"""
    state = MatchActiveDebounceState(confirmed_active=False)
    new_state, transitioned, fi, ts = _resolve_match_active_debounce(
        state, False, 10, 1.0, debounce_sec=1.0,
    )
    assert transitioned is False
    assert new_state.confirmed_active is False


def test_debounce_new_change_starts_pending() -> None:
    """confirmed=False → raw=True (新規変化) は即確定せずpendingを開始する。"""
    state = MatchActiveDebounceState(confirmed_active=False)
    new_state, transitioned, fi, ts = _resolve_match_active_debounce(
        state, True, 100, 5.0, debounce_sec=1.0,
    )
    assert transitioned is False
    assert new_state.confirmed_active is False  # まだ確定しない
    assert new_state.pending_active is True
    assert new_state.pending_since_frame == 100
    assert new_state.pending_since_time == 5.0


def test_debounce_pending_continues_under_threshold_not_confirmed() -> None:
    """pending継続中、debounce_sec未満の経過では確定しない (state不変)。"""
    state = MatchActiveDebounceState(
        confirmed_active=False, pending_active=True,
        pending_since_frame=100, pending_since_time=5.0,
    )
    new_state, transitioned, fi, ts = _resolve_match_active_debounce(
        state, True, 130, 5.9, debounce_sec=1.0,  # 経過0.9s < 1.0s
    )
    assert transitioned is False
    assert new_state == state  # pending は一切変化しない


def test_debounce_pending_confirms_exactly_at_threshold() -> None:
    """pending継続中、経過がdebounce_sec以上で確定する (境界値=以上で確定)。"""
    state = MatchActiveDebounceState(
        confirmed_active=False, pending_active=True,
        pending_since_frame=100, pending_since_time=5.0,
    )
    new_state, transitioned, fi, ts = _resolve_match_active_debounce(
        state, True, 160, 6.0, debounce_sec=1.0,  # 経過ちょうど1.0s
    )
    assert transitioned is True
    assert new_state.confirmed_active is True
    assert new_state.pending_active is None


def test_debounce_confirmed_time_semantics_uses_true_transition_instant() -> None:
    """確定時刻の意味論: 確定frame_idx/time_secは「raw値が最初に変化した瞬間」

    (pending_since) であり、確認が完了した現在フレームの値ではない。
    """
    state = MatchActiveDebounceState(
        confirmed_active=False, pending_active=True,
        pending_since_frame=100, pending_since_time=5.0,
    )
    # 確認完了フレームは frame_idx=200, time_sec=6.5 (pending開始より後)。
    new_state, transitioned, confirmed_fi, confirmed_ts = _resolve_match_active_debounce(
        state, True, 200, 6.5, debounce_sec=1.0,
    )
    assert transitioned is True
    # 確定値は「真の遷移瞬間」(pending_since) であるべき、確認完了瞬間ではない。
    assert confirmed_fi == 100
    assert confirmed_ts == 5.0
    assert confirmed_fi != 200
    assert confirmed_ts != 6.5


def test_debounce_flicker_reverts_before_threshold_is_ignored() -> None:
    """フリッカー: N秒未満でraw値が元に戻ると pending がクリアされ確定しない。"""
    state = MatchActiveDebounceState(confirmed_active=True)
    # 1frame目: raw=False (フリッカー開始)
    state, transitioned1, _, _ = _resolve_match_active_debounce(
        state, False, 100, 5.0, debounce_sec=1.0,
    )
    assert transitioned1 is False
    assert state.pending_active is False
    # 2frame目 (0.3s後): raw=True (元の confirmed に復帰、フリッカー終了)
    state, transitioned2, _, _ = _resolve_match_active_debounce(
        state, True, 118, 5.3, debounce_sec=1.0,
    )
    assert transitioned2 is False
    assert state.confirmed_active is True  # 一切変化していない
    assert state.pending_active is None  # pending はクリアされた


def test_debounce_direction_switch_restarts_pending_clock() -> None:
    """pending中に反対方向へさらに変化した場合、pending時計がリスタートする。"""
    state = MatchActiveDebounceState(confirmed_active=True)
    # False方向のpending開始
    state, _, _, _ = _resolve_match_active_debounce(
        state, False, 100, 5.0, debounce_sec=1.0,
    )
    # 0.5s後に True 方向へ切り替わる (=raw が confirmed と同じ True に戻る)
    state, _, _, _ = _resolve_match_active_debounce(
        state, True, 130, 5.5, debounce_sec=1.0,
    )
    # さらに 0.5s後に再度 False へ (直前は pending=None だったので新規pending)
    state, transitioned, _, _ = _resolve_match_active_debounce(
        state, False, 160, 6.0, debounce_sec=1.0,
    )
    assert transitioned is False
    assert state.pending_since_time == 6.0  # 新規pendingとして再カウント


# =============================================================================
# 2. フラグ既定値・reset()クリア
# =============================================================================


def test_enable_match_transition_debounce_default_false() -> None:
    """enable_match_transition_debounce 未指定時は既定 False (backwards compat)。"""
    pipe = _make_pipe(_StubMatchDetector(in_match=True))
    assert pipe._enable_match_transition_debounce is False


def test_match_active_debounce_state_inits_default() -> None:
    """新規インスタンスの _match_active_debounce_state は既定値で初期化される。"""
    pipe = _make_pipe(_StubMatchDetector(in_match=True))
    assert pipe._match_active_debounce_state == MatchActiveDebounceState()


def test_reset_clears_match_active_debounce_state() -> None:
    """reset() で _match_active_debounce_state がクリアされる (試合境界残留防止)。"""
    pipe = _make_pipe(_StubMatchDetector(in_match=True), enable_debounce=True)
    pipe._match_active_debounce_state = MatchActiveDebounceState(
        confirmed_active=True, pending_active=False,
        pending_since_frame=5, pending_since_time=1.0,
    )
    pipe.reset()
    assert pipe._match_active_debounce_state == MatchActiveDebounceState()


# =============================================================================
# 3. backwards compat (フラグOFFでbit-identical)
# =============================================================================


def test_match_transition_debounce_explicit_false_restores_legacy_output() -> None:
    """enable_match_transition_debounce=False を明示しても、未指定時と完全に同じ結果になる。"""
    pipe_default = _make_pipe(_StubMatchDetector(in_match=True))
    pipe_explicit = _make_pipe(_StubMatchDetector(in_match=True), enable_debounce=False)
    frame = _dummy_frame()
    for i in range(6):
        r1 = pipe_default.update(i, 0.1 * i, frame)
        r2 = pipe_explicit.update(i, 0.1 * i, frame)
        assert r1.p1.state == r2.p1.state
        assert r1.p2.state == r2.p2.state
        assert pipe_default._match_active_started_frame == pipe_explicit._match_active_started_frame
        assert pipe_default._match_active_started_time == pipe_explicit._match_active_started_time


# =============================================================================
# 4. 統合: 実 pipeline.update() でのフリッカー吸収確認
# =============================================================================


def test_pipeline_flicker_shorter_than_debounce_does_not_reset_started_time(
) -> None:
    """flag=True で、debounce_sec未満の短いis_active flickerでは

    _match_active_started_frame/_time が変化しない (第4機構の直接対策)。

    is_active は raw_active/recent_active/sm_active の OR に加え、
    hard_match_off (score_zero_both 等) が立つと確定的に False になる
    (§ hard_match_off は hysteresis を上書きする確定シグナル)。静的な空盤面
    では sm_active が常時Trueのため、この確定シグナル経由でのみ genuine な
    is_active=False を再現できる (盤面は完全空=count_puyos()==0 のため
    「既に puyo 観測」救済ロジックは働かず score_zero_both がそのまま通る)。
    """
    detector = _StubMatchDetector(in_match=True)
    score_zero = _StubScoreZeroDetector(both_zero=False)
    pipe = _make_pipe(detector, enable_debounce=True, score_zero_detector=score_zero)
    frame = _dummy_frame()
    fps = 30.0
    # 試合中を2秒分進め、開始デバウンス確定 (>=1.0s) を確実に済ませてから
    # _match_active_started_time を確定させる。
    for i in range(60):
        pipe.update(i, i / fps, frame)
    started_time_before = pipe._match_active_started_time
    started_frame_before = pipe._match_active_started_frame
    assert started_time_before >= 0, "事前に試合開始が記録されているはず"
    # 0.3秒分 (debounce_sec=1.0未満) だけ hard_match_off が立つフリッカーを起こす。
    score_zero.both_zero = True
    for i in range(60, 69):
        pipe.update(i, i / fps, frame)
    score_zero.both_zero = False
    for i in range(69, 75):
        pipe.update(i, i / fps, frame)
    # フリッカーが吸収され、試合開始時刻が全く変化していないはず。
    assert pipe._match_active_started_time == started_time_before
    assert pipe._match_active_started_frame == started_frame_before


def test_pipeline_inactive_longer_than_debounce_does_reset_started_time() -> None:
    """flag=True でも、debounce_sec以上続く真の非active化では

    _match_active_started_frame/_time が正しく-1/-1.0にリセットされる
    (フリッカー無視が「無防備化」ではないことの確認)。
    """
    detector = _StubMatchDetector(in_match=True)
    score_zero = _StubScoreZeroDetector(both_zero=False)
    pipe = _make_pipe(detector, enable_debounce=True, score_zero_detector=score_zero)
    frame = _dummy_frame()
    fps = 30.0
    for i in range(60):
        pipe.update(i, i / fps, frame)
    assert pipe._match_active_started_time >= 0
    # debounce_sec (1.0s) を明確に超える 1.5秒間、hard_match_off で非active にする。
    score_zero.both_zero = True
    for i in range(60, 105):
        pipe.update(i, i / fps, frame)
    assert pipe._match_active_started_frame == -1
    assert pipe._match_active_started_time == -1.0
