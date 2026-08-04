"""バーストガード再設計 Stage0/Stage1 のテスト (2026-08-05)。

docs/BURST_GUARD_DESIGN_2026-08-05.md 準拠。案B (enable_effect_gate +
enable_effect_visual_gate) の3敗因 (ChainEvent見逃し・persist逆転・守備範囲
固定) を Schmitt trigger 視覚トリガー + ハード凍結で構造的に解消する新方式
(enable_burst_guard_v2) を検証する。

構成:
    1. `_update_burst_visual_gate` (stateless純関数) の遷移表網羅
    2. `_resolve_burst_gate_state` (frame_bgr None時の安全弁) の分岐網羅
    3. `effect_gate_hard_freeze` の board_state_machine 凍結挙動
    4. RecognitionPipeline 統合 (フラグ受理・reset()・no-op警告)
    5. backwards compat (新フラグ全て False で既存挙動 bit-identical)
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.board import BOARD_ROWS, COLOR_BLUE, COLOR_EMPTY, COLOR_RED, Board
from src.board_state_machine import (
    BoardState,
    BoardStateMachine,
    DetectorSignals,
    EFFECT_GATE_TOP_ROWS,
    _update_burst_visual_gate,
)
from src.image_reader import DEFAULT_P1_REGION
from src.match_state import MatchState
from src.recognition_pipeline import (
    RecognitionPipeline,
    _resolve_burst_gate_state,
)

# テスト用の閾値 (docs/BURST_GUARD_DESIGN_2026-08-05.md §4 の本番値と別に、
# 遷移表を高速に検証するための値を明示的に渡す。関数はデフォルト値を
# 持たない設計のため、呼び出し側で必ず指定する)。
_OPEN_TH = 0.97
_CLOSE_TH = 0.97
_MIN_WINDOW = 0.2
_MAX_WINDOW = 30.0
_QUIESCENCE = 0.25


def _step(is_open, opened_at, quiet_since, score, t, **overrides):
    kwargs = dict(
        open_threshold=_OPEN_TH, close_threshold=_CLOSE_TH,
        min_window_sec=_MIN_WINDOW, max_window_sec=_MAX_WINDOW,
        quiescence_min_sec=_QUIESCENCE,
    )
    kwargs.update(overrides)
    return _update_burst_visual_gate(is_open, opened_at, quiet_since, score, t, **kwargs)


# ============================
# 1. _update_burst_visual_gate 遷移表
# ============================


def test_closed_stays_closed_below_open_threshold() -> None:
    """closed かつ score<open_threshold → closed 維持。"""
    assert _step(False, None, None, 0.5, 1.0) == (False, None, None)


def test_closed_opens_when_score_reaches_threshold() -> None:
    """closed かつ score>=open_threshold → 即時 open (opened_at=今time_sec)。"""
    new_open, opened_at, quiet = _step(False, None, None, 0.98, 5.0)
    assert new_open is True
    assert opened_at == 5.0
    assert quiet is None


def test_open_stays_open_and_resets_quiet_when_score_high() -> None:
    """open 中に score>=close_threshold に戻ると quiet_since がリセットされる。"""
    new_open, opened_at, quiet = _step(True, 5.0, 5.05, 0.99, 5.1)
    assert new_open is True
    assert opened_at == 5.0
    assert quiet is None


def test_open_quiet_since_starts_on_first_drop() -> None:
    """open 中に初めて score<close_threshold になった frame で quiet_since が今time_secに設定される。"""
    new_open, opened_at, quiet = _step(True, 0.0, None, 0.1, 0.1)
    assert new_open is True
    assert opened_at == 0.0
    assert quiet == 0.1


def test_min_window_blocks_early_close_even_if_quiescence_satisfied() -> None:
    """min_window_sec 未達なら、quiescence条件だけ満たしても close しない。"""
    # opened_at=0.0, quiet_since=0.05, quiescence_min_sec=0.05。
    # t=0.15: quiescent=0.10>=0.05 (満たす) だが elapsed_open=0.15<0.2 (未達)。
    new_open, opened_at, quiet = _step(
        True, 0.0, 0.05, 0.1, 0.15, quiescence_min_sec=0.05,
    )
    assert new_open is True
    assert opened_at == 0.0
    assert quiet == 0.05


def test_closes_after_min_window_and_quiescence_both_satisfied() -> None:
    """min_window_sec と quiescence_min_sec の両方を満たしたら close する。"""
    new_open, opened_at, quiet = _step(
        True, 0.0, 0.05, 0.1, 0.25, quiescence_min_sec=0.05,
    )
    assert (new_open, opened_at, quiet) == (False, None, None)


def test_max_window_forces_close_even_if_score_still_high() -> None:
    """安全弁: max_window_sec 超過で score に関係なく強制close。"""
    new_open, opened_at, quiet = _step(True, 0.0, None, 0.99, 30.0)
    assert (new_open, opened_at, quiet) == (False, None, None)


def test_force_close_overrides_open_state() -> None:
    """force_close=True は他条件を無視して即時close (openから)。"""
    new_open, opened_at, quiet = _step(
        True, 5.0, 5.1, 0.99, 5.2, force_close=True,
    )
    assert (new_open, opened_at, quiet) == (False, None, None)


def test_force_close_overrides_closed_state() -> None:
    """force_close=True は closed 状態でも安全側 (False,None,None) を返す。"""
    new_open, opened_at, quiet = _step(
        False, None, None, 0.99, 1.0, force_close=True,
    )
    assert (new_open, opened_at, quiet) == (False, None, None)


# ============================
# 2. _resolve_burst_gate_state (frame_bgr None 安全弁)
# ============================


def _black_frame() -> "np.ndarray":
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _bright_top_row_frame() -> "np.ndarray":
    frame = _black_frame()
    row = next(iter(EFFECT_GATE_TOP_ROWS))
    x1, y1, x2, y2 = DEFAULT_P1_REGION.cell_sample_rect(row, 0)
    frame[y1:y2, x1:x2] = (255, 255, 255)
    return frame


def test_resolve_burst_gate_frame_none_keeps_prev_state_unchanged() -> None:
    """frame_bgr=None (画像取得不能) は無情報を「静穏」と誤認せず直前状態を維持する。"""
    new_open, opened_at, quiet = _resolve_burst_gate_state(
        None, DEFAULT_P1_REGION, EFFECT_GATE_TOP_ROWS,
        True, 10.0, 10.1, time_sec=10.2, force_close=False,
    )
    assert (new_open, opened_at, quiet) == (True, 10.0, 10.1)


def test_resolve_burst_gate_frame_none_still_applies_max_window() -> None:
    """frame_bgr=None でも max_window_sec 安全弁は time_sec 経過で効く。"""
    new_open, opened_at, quiet = _resolve_burst_gate_state(
        None, DEFAULT_P1_REGION, EFFECT_GATE_TOP_ROWS,
        True, 0.0, None, time_sec=30.0, force_close=False,
    )
    assert (new_open, opened_at, quiet) == (False, None, None)


def test_resolve_burst_gate_frame_none_still_applies_force_close() -> None:
    """frame_bgr=None でも force_close 条件 (own_chain_active等) は効く。"""
    new_open, opened_at, quiet = _resolve_burst_gate_state(
        None, DEFAULT_P1_REGION, EFFECT_GATE_TOP_ROWS,
        True, 0.0, None, time_sec=0.1, force_close=True,
    )
    assert (new_open, opened_at, quiet) == (False, None, None)


def test_resolve_burst_gate_with_frame_delegates_to_schmitt_trigger() -> None:
    """frame_bgr がある場合は視覚スコアを計算して通常の Schmitt trigger を通す。"""
    new_open, _opened_at, _quiet = _resolve_burst_gate_state(
        _bright_top_row_frame(), DEFAULT_P1_REGION, EFFECT_GATE_TOP_ROWS,
        False, None, None, time_sec=1.0, force_close=False,
    )
    assert new_open is True  # 高輝度検出 → open

    new_open2, opened_at2, quiet2 = _resolve_burst_gate_state(
        _black_frame(), DEFAULT_P1_REGION, EFFECT_GATE_TOP_ROWS,
        False, None, None, time_sec=1.0, force_close=False,
    )
    assert (new_open2, opened_at2, quiet2) == (False, None, None)


# ============================
# 3. effect_gate_hard_freeze (board_state_machine 凍結挙動)
# ============================


def _empty_board() -> Board:
    return Board()


def _stacked_board(target_row: int, col: int, target_color: int) -> Board:
    b = Board()
    for r in range(target_row + 1, BOARD_ROWS):
        b.set(r, col, COLOR_BLUE)
    if target_color != COLOR_EMPTY:
        b.set(target_row, col, target_color)
    return b


def _make_burst_guard_sm(
    *, effect_gate_hard_freeze: bool, recovery_min_frames: int = 2,
    confirmed: "Board | None" = None,
) -> BoardStateMachine:
    sm = BoardStateMachine(
        enable_stable_recovery_gate=True,
        recovery_min_frames=recovery_min_frames,
        enable_effect_gate=True,
        effect_gate_hard_freeze=effect_gate_hard_freeze,
    )
    sm._ctx.state = BoardState.STABLE
    sm._ctx.confirmed_board = confirmed if confirmed is not None else _empty_board()
    return sm


def _gated_signal(t: float, cnn: Board, hsv: Board, *, window_active: bool) -> DetectorSignals:
    return DetectorSignals(
        time_sec=t, cnn_board=cnn, is_match_active=True, hsv_board=hsv,
        effect_gate_window_active=window_active,
    )


def test_hard_freeze_never_commits_while_window_open() -> None:
    """effect_gate_hard_freeze=True: window ON がどれだけ続いても持続確認せず発火しない。"""
    row = next(iter(EFFECT_GATE_TOP_ROWS))
    confirmed = _stacked_board(row, 0, COLOR_EMPTY)
    sm = _make_burst_guard_sm(effect_gate_hard_freeze=True, confirmed=confirmed)
    cnn = _stacked_board(row, 0, COLOR_RED)
    hsv = _stacked_board(row, 0, COLOR_RED)
    for i in range(20):  # EFFECT_PERSIST_SEC (0.4s) を大幅に超える長さでも確定しない
        sm.update(i, _gated_signal(i * 0.1, cnn, hsv, window_active=True))
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(row, 0) == COLOR_EMPTY
    # hold dict (persist方式) は一切使われない。カウンタも pop 済み。
    assert sm.context.effect_gate_hold == {}
    assert (row, 0) not in sm.context.stable_recovery_counters


def test_hard_freeze_false_uses_legacy_persist_hold_dict() -> None:
    """effect_gate_hard_freeze=False (既定): 従来通り effect_gate_hold が使われる (backwards compat)。"""
    row = next(iter(EFFECT_GATE_TOP_ROWS))
    confirmed = _stacked_board(row, 0, COLOR_EMPTY)
    sm = _make_burst_guard_sm(effect_gate_hard_freeze=False, confirmed=confirmed)
    cnn = _stacked_board(row, 0, COLOR_RED)
    hsv = _stacked_board(row, 0, COLOR_RED)
    sm.update(0, _gated_signal(0.0, cnn, hsv, window_active=True))
    assert (row, 0) in sm.context.effect_gate_hold


def test_hard_freeze_window_close_reenters_from_zero() -> None:
    """window close 後は recovery_counters がゼロから再エントリーする (追加実装不要の副産物を確認)。"""
    row = next(iter(EFFECT_GATE_TOP_ROWS))
    min_frames = 3
    confirmed = _stacked_board(row, 0, COLOR_EMPTY)
    sm = _make_burst_guard_sm(
        effect_gate_hard_freeze=True, recovery_min_frames=min_frames, confirmed=confirmed,
    )
    cnn = _stacked_board(row, 0, COLOR_RED)
    hsv = _stacked_board(row, 0, COLOR_RED)
    for i in range(5):  # window ON 中は凍結
        sm.update(i, _gated_signal(i * 0.1, cnn, hsv, window_active=True))
    assert sm.context.confirmed_board.get(row, 0) == COLOR_EMPTY

    t = 0.5
    for j in range(min_frames - 1):  # window close 後、min_frames 未達の間は未確定
        sm.update(5 + j, _gated_signal(t, cnn, hsv, window_active=False))
        t += 0.1
    assert sm.context.confirmed_board.get(row, 0) == COLOR_EMPTY

    sm.update(5 + min_frames - 1, _gated_signal(t, cnn, hsv, window_active=False))
    assert sm.context.confirmed_board.get(row, 0) == COLOR_RED  # min_frames目で確定


# ============================
# 4. RecognitionPipeline 統合
# ============================


class _StubMatchDetectorForBurst:
    def detect(self, frame: "np.ndarray") -> object:
        class _R:
            state = MatchState.IN_MATCH
            bg_value = 100.0
            bg_saturation = 50.0
            samples = 1
        return _R()


class _StubImageReaderForBurst:
    def read_both_boards(self, frame: "np.ndarray", **_kwargs: object) -> tuple[Board, Board]:
        return _empty_board(), _empty_board()


def _make_burst_pipe(**kwargs: object) -> RecognitionPipeline:
    return RecognitionPipeline(
        image_reader=_StubImageReaderForBurst(),  # type: ignore[arg-type]
        match_state_detector=_StubMatchDetectorForBurst(),  # type: ignore[arg-type]
        **kwargs,
    )


def test_enable_burst_guard_v2_default_is_false() -> None:
    """enable_burst_guard_v2 未指定時は既定 False (backwards compat)。"""
    pipe = _make_burst_pipe()
    assert pipe._enable_burst_guard_v2 is False


def test_enable_burst_guard_v2_true_propagates_to_state_machines() -> None:
    """enable_burst_guard_v2=True は effect_gate_hard_freeze として両 side に伝播する。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = _make_burst_pipe(enable_effect_gate=True, enable_burst_guard_v2=True)
    assert pipe._enable_burst_guard_v2 is True
    assert pipe._sm_1p._effect_gate_hard_freeze is True  # noqa: SLF001
    assert pipe._sm_2p._effect_gate_hard_freeze is True  # noqa: SLF001


def test_reset_clears_burst_gate_state() -> None:
    """reset() で Schmitt trigger 状態 6 属性が全てクリアされる (試合境界残留防止)。"""
    pipe = _make_burst_pipe()
    pipe._burst_gate_open_1p = True
    pipe._burst_gate_open_2p = True
    pipe._burst_gate_opened_at_1p = 1.0
    pipe._burst_gate_opened_at_2p = 2.0
    pipe._burst_gate_quiet_since_1p = 3.0
    pipe._burst_gate_quiet_since_2p = 4.0
    pipe.reset()
    assert pipe._burst_gate_open_1p is False
    assert pipe._burst_gate_open_2p is False
    assert pipe._burst_gate_opened_at_1p is None
    assert pipe._burst_gate_opened_at_2p is None
    assert pipe._burst_gate_quiet_since_1p is None
    assert pipe._burst_gate_quiet_since_2p is None


def test_enable_burst_guard_v2_without_effect_gate_warns_no_op() -> None:
    """enable_burst_guard_v2=True かつ enable_effect_gate=False は no-op 警告を出す。"""
    with pytest.warns(UserWarning, match="no-op"):
        _make_burst_pipe(enable_burst_guard_v2=True, enable_effect_gate=False)


def test_enable_burst_guard_v2_with_effect_gate_no_warning() -> None:
    """enable_burst_guard_v2=True かつ enable_effect_gate=True では警告しない。"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _make_burst_pipe(enable_burst_guard_v2=True, enable_effect_gate=True)


# ============================
# 5. backwards compat (新フラグ全て False で既存挙動 bit-identical)
# ============================


def _dummy_frame() -> "np.ndarray":
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def test_burst_guard_v2_explicit_false_restores_legacy_pipeline_output() -> None:
    """enable_burst_guard_v2=False を明示しても、未指定時と完全に同じ結果になる。"""
    pipe_default = _make_burst_pipe(stable_frame_count=2)
    pipe_explicit = _make_burst_pipe(stable_frame_count=2, enable_burst_guard_v2=False)
    for i in range(3):
        r1 = pipe_default.update(i, 0.05 * i, _dummy_frame())
        r2 = pipe_explicit.update(i, 0.05 * i, _dummy_frame())
        assert r1.p1.state == r2.p1.state
        assert r1.p2.state == r2.p2.state
        assert r1.p1.confirmed_board == r2.p1.confirmed_board
        assert r1.p2.confirmed_board == r2.p2.confirmed_board
    # burst-gate 属性は一切書き込まれない (経路自体に入らない、backwards compat)
    assert pipe_explicit._burst_gate_open_1p is False


# ============================
# 6. Stage1.5: _filter_transition_new_cnn_for_burst_guard (遷移merge時フィルタ)
# ============================


def _board_all(color: int) -> Board:
    """全セルを color で埋めた盤面 (テスト用、対象セル以外は同一値で差分ゼロにする)。"""
    b = Board()
    for r in range(BOARD_ROWS):
        for c in range(6):
            b.set(r, c, color)
    return b


def test_filter_transition_noop_for_out_of_scope_from_states() -> None:
    """【最優先】GRAVITY_SETTLE/CHAIN/EFFECT では完全no-op (恒等、過剰介入防止の安全網)。"""
    from src.board_state_machine import _filter_transition_new_cnn_for_burst_guard

    baseline = _board_all(COLOR_EMPTY)
    new_cnn = _board_all(COLOR_RED)  # 大量差分 (連鎖後の正当な変化を模す)
    for state in (BoardState.GRAVITY_SETTLE, BoardState.CHAIN, BoardState.EFFECT):
        result = _filter_transition_new_cnn_for_burst_guard(baseline, new_cnn, state)
        assert result is new_cnn  # 恒等 (コピーもしない)


def test_filter_transition_noop_when_baseline_is_none() -> None:
    """baseline is None (初回STABLE確定) は対象 from_state でも完全no-op。"""
    from src.board_state_machine import _filter_transition_new_cnn_for_burst_guard

    new_cnn = _board_all(COLOR_RED)
    for state in (BoardState.TSUMO_FALL, BoardState.OJAMA_FALL):
        result = _filter_transition_new_cnn_for_burst_guard(None, new_cnn, state)
        assert result is new_cnn


# c18の10セルパターン (§10.1): from_state別に「物理的に説明可能な diff」
# 以外は全て COLOR_UNKNOWN に棄却されることを表形式で網羅する。
# (from_state, base_v, cnn_v, expected_v, 説明)
_TRANSITION_FILTER_CASES: "tuple[tuple[BoardState, int, int, int, str], ...]" = (
    # TSUMO_FALL: 空→色(1-5)のみ説明可能。
    (BoardState.TSUMO_FALL, 0, 4, 4, "空→黄、正当な設置"),
    (BoardState.TSUMO_FALL, 0, 1, 1, "空→赤、正当な設置"),
    (BoardState.TSUMO_FALL, 1, 4, 10, "既存赤がバーストで黄に化ける (c18型)"),
    (BoardState.TSUMO_FALL, 5, 9, 10, "既存紫がバーストで9に化ける (c18本命パターン)"),
    (BoardState.TSUMO_FALL, 0, 9, 10, "空セルに9出現 (TSUMO_FALL中はおじゃま落下しない)"),
    (BoardState.TSUMO_FALL, 2, 2, 2, "差分なし (base==cnn、そもそも棄却対象外)"),
    # OJAMA_FALL: 空→9(おじゃま)のみ説明可能。
    (BoardState.OJAMA_FALL, 0, 9, 9, "空→9、正当なおじゃま着弾"),
    (BoardState.OJAMA_FALL, 1, 4, 10, "既存赤がバーストで黄に化ける"),
    (BoardState.OJAMA_FALL, 1, 9, 10, "既存赤がバーストで9に化ける"),
    (BoardState.OJAMA_FALL, 5, 9, 10, "既存紫がバーストで9に化ける (c18型)"),
    (BoardState.OJAMA_FALL, 0, 4, 10, "空セルに色出現 (OJAMA_FALL中は色puyoが降らない)"),
)


@pytest.mark.parametrize(
    "from_state,base_v,cnn_v,expected_v,_desc", _TRANSITION_FILTER_CASES,
)
def test_filter_transition_c18_pattern_table(
    from_state: BoardState, base_v: int, cnn_v: int, expected_v: int, _desc: str,
) -> None:
    """c18の10セルパターンを表形式で網羅し、説明不可能な diff が全て棄却されることを確認する。"""
    from src.board_state_machine import _filter_transition_new_cnn_for_burst_guard

    target = (0, 0)
    baseline = _board_all(COLOR_EMPTY)
    baseline.set(*target, base_v)
    new_cnn = _board_all(COLOR_EMPTY)
    new_cnn.set(*target, cnn_v)
    result = _filter_transition_new_cnn_for_burst_guard(baseline, new_cnn, from_state)
    assert result.get(*target) == expected_v, _desc


def _gated_transition_signal(cnn: Board, window_active: bool) -> DetectorSignals:
    """遷移merge検証用の signals (hsv_board=None、_apply_transition のみ関与)。"""
    return DetectorSignals(
        time_sec=0.0, cnn_board=cnn, is_match_active=True,
        effect_gate_window_active=window_active,
    )


def _make_transition_guard_sm(
    *, enable_transition_merge_guard: bool, confirmed: Board,
) -> BoardStateMachine:
    sm = BoardStateMachine(enable_transition_merge_guard=enable_transition_merge_guard)
    sm._ctx.state = BoardState.TSUMO_FALL
    sm._ctx.confirmed_board = confirmed
    return sm


_BOTTOM_ROW: int = BOARD_ROWS - 1  # 最下段 (gravity filterの浮きぷよ判定を避けるため使用)


def test_transition_guard_rejects_unexplainable_diff_when_window_active() -> None:
    """enable_transition_merge_guard=True かつ window活性中: 説明不可能な diff は棄却される。"""
    baseline = _board_all(COLOR_EMPTY)
    baseline.set(_BOTTOM_ROW, 0, COLOR_RED)  # 既存の赤puyo (最下段=支えあり)
    sm = _make_transition_guard_sm(
        enable_transition_merge_guard=True, confirmed=baseline.copy(),
    )
    new_cnn = _board_all(COLOR_EMPTY)
    new_cnn.set(_BOTTOM_ROW, 0, COLOR_BLUE)  # バースト誤読: 赤→青に化ける (説明不可能)
    sm._apply_transition(BoardState.STABLE, _gated_transition_signal(new_cnn, True))
    assert sm.context.confirmed_board.get(_BOTTOM_ROW, 0) == COLOR_RED  # baseline維持


def test_transition_guard_is_identity_when_window_inactive() -> None:
    """window非活性中は恒等 (フィルタ未適用、従来通り cnn 値がそのまま書き込まれる)。"""
    baseline = _board_all(COLOR_EMPTY)
    baseline.set(_BOTTOM_ROW, 0, COLOR_RED)
    sm = _make_transition_guard_sm(
        enable_transition_merge_guard=True, confirmed=baseline.copy(),
    )
    new_cnn = _board_all(COLOR_EMPTY)
    new_cnn.set(_BOTTOM_ROW, 0, COLOR_BLUE)
    sm._apply_transition(BoardState.STABLE, _gated_transition_signal(new_cnn, False))
    assert sm.context.confirmed_board.get(_BOTTOM_ROW, 0) == COLOR_BLUE  # フィルタ未適用、素通り


def test_transition_guard_disabled_is_bit_identical_to_legacy() -> None:
    """enable_transition_merge_guard=False (既定): window活性中でも従来通り素通りする。"""
    baseline = _board_all(COLOR_EMPTY)
    baseline.set(_BOTTOM_ROW, 0, COLOR_RED)
    sm = _make_transition_guard_sm(
        enable_transition_merge_guard=False, confirmed=baseline.copy(),
    )
    new_cnn = _board_all(COLOR_EMPTY)
    new_cnn.set(_BOTTOM_ROW, 0, COLOR_BLUE)
    sm._apply_transition(BoardState.STABLE, _gated_transition_signal(new_cnn, True))
    assert sm.context.confirmed_board.get(_BOTTOM_ROW, 0) == COLOR_BLUE  # 従来挙動 (bit-identical)


def test_transition_guard_flag_stored_default_false() -> None:
    """enable_transition_merge_guard 未指定時は既定 False (backwards compat)。"""
    sm = BoardStateMachine()
    assert sm._enable_transition_merge_guard is False


# ============================
# 7. RecognitionPipeline 統合: enable_transition_merge_guard / burst_gate_open_threshold
# ============================


def test_enable_transition_merge_guard_default_false_and_propagates() -> None:
    """既定False + 明示True時に BoardStateMachine 側へ正しく伝播する。"""
    pipe = _make_burst_pipe()
    assert pipe._enable_transition_merge_guard is False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe2 = _make_burst_pipe(
            enable_burst_guard_v2=True, enable_transition_merge_guard=True,
        )
    assert pipe2._enable_transition_merge_guard is True
    assert pipe2._sm_1p._enable_transition_merge_guard is True  # noqa: SLF001
    assert pipe2._sm_2p._enable_transition_merge_guard is True  # noqa: SLF001


def test_enable_transition_merge_guard_without_burst_guard_v2_warns() -> None:
    """enable_transition_merge_guard=True かつ enable_burst_guard_v2=False は no-op 警告。"""
    with pytest.warns(UserWarning, match="no-op"):
        _make_burst_pipe(
            enable_transition_merge_guard=True, enable_burst_guard_v2=False,
        )


def test_burst_gate_open_threshold_default_is_module_constant() -> None:
    """burst_gate_open_threshold 未指定時は既存定数 (0.97) を使う (bit-identical)。"""
    from src.recognition_pipeline import BURST_GATE_OPEN_THRESHOLD
    pipe = _make_burst_pipe()
    assert pipe._burst_gate_open_threshold == BURST_GATE_OPEN_THRESHOLD


def test_burst_gate_open_threshold_explicit_value_stored() -> None:
    """明示指定した閾値が格納される。"""
    pipe = _make_burst_pipe(burst_gate_open_threshold=0.85)
    assert pipe._burst_gate_open_threshold == 0.85


def test_burst_gate_open_threshold_passed_to_schmitt_trigger() -> None:
    """_resolve_burst_gate_state に閾値が実際に渡り、開窓判定が変わることを確認する。"""
    frame = _bright_top_row_frame()
    # 実測スコアは1.0 (全白パッチ)。閾値0.999でも開くが、比較のため極端な
    # 高閾値 (1.5、スコアの取り得る範囲外) では開かないことを確認する。
    new_open_default, _, _ = _resolve_burst_gate_state(
        frame, DEFAULT_P1_REGION, EFFECT_GATE_TOP_ROWS,
        False, None, None, time_sec=1.0, force_close=False,
    )
    new_open_high_threshold, _, _ = _resolve_burst_gate_state(
        frame, DEFAULT_P1_REGION, EFFECT_GATE_TOP_ROWS,
        False, None, None, time_sec=1.0, force_close=False,
        open_threshold=1.5, close_threshold=1.5,
    )
    assert new_open_default is True
    assert new_open_high_threshold is False
