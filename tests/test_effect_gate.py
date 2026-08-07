"""エフェクト時間ゲート (enable_effect_gate, 2026-08-03) のテスト。

満杯盤面 47 セル誤りの真因確定 (memory
`project_full_board_error_taxonomy_2026-08-02`)。相手の連鎖 1 リンクごとに
約 0.2 秒発生する「予告おじゃま送付エフェクト」+ お邪魔落下時の煙が、
自盤面上段 (row1-3) の色→空/空→色/色→色ちらつきとして confirmed_board に
混入するのを防ぐ「領域限定 + 持続確認 (実秒ベース)」ゲートを検証する。

4 方向 + α:
    (a) 相手連鎖中の上段 0.2 秒フリッカが確定盤面に入らない
    (b) 同じ窓中の本物の設置 (0.4 秒以上持続) は遅延して確定する
    (c) 窓外 (相手連鎖していない / 上段以外) は従来レイテンシ
    (d) フラグ OFF で旧挙動 bit 一致
    + 低レベルヘルパー (_update_effect_gate_hold) の単体テスト
"""

from __future__ import annotations

import numpy as np

from src.board import BOARD_ROWS, COLOR_BLUE, COLOR_EMPTY, COLOR_RED, Board
from src.board_state_machine import (
    BoardState,
    BoardStateMachine,
    DetectorSignals,
    EFFECT_GATE_TOP_ROWS,
    _update_effect_gate_hold,
)
from src.image_reader import DEFAULT_P1_REGION
from src.match_state import MatchState
from src.recognition_pipeline import (
    RecognitionPipeline,
    _compute_effect_gate_window_active,
    _update_all_clear_pending,
)

# テスト用の短い持続秒数 (本番既定 EFFECT_PERSIST_SEC=0.4 相当の挙動を
# 0.05s フレーム間隔で高速に検証するための値)。
_PERSIST_SEC = 0.2
_STEP_SEC = 0.05


def _empty_board() -> Board:
    return Board()


def _stacked_board(target_row: int, col: int, target_color: int) -> Board:
    """target_row より下 (row+1..最終行) を BLUE で埋めた「積み上がった」盤面を作る。

    復旧ゲートの安全弁C (列単位の重力整合チェック) は「候補セルの下が
    confirmed で空」だと浮きぷよとして却下する。上段 (EFFECT_GATE_TOP_ROWS)
    での 空→色 遷移を検証するには、下の列が既に積まれている必要がある
    (テスト内実際の対戦盤面で上段に置く=既に下が埋まっている状況を模す)。
    """
    b = Board()
    for r in range(target_row + 1, BOARD_ROWS):
        b.set(r, col, COLOR_BLUE)
    if target_color != COLOR_EMPTY:
        b.set(target_row, col, target_color)
    return b


def _make_effect_gate_sm(
    *, enable_effect_gate: bool, recovery_min_frames: int = 2,
    confirmed: "Board | None" = None,
) -> BoardStateMachine:
    """エフェクト時間ゲート検証用の STABLE state machine を生成する。"""
    sm = BoardStateMachine(
        enable_stable_recovery_gate=True,
        recovery_min_frames=recovery_min_frames,
        enable_effect_gate=enable_effect_gate,
        effect_gate_persist_sec=_PERSIST_SEC,
    )
    sm._ctx.state = BoardState.STABLE
    sm._ctx.confirmed_board = confirmed if confirmed is not None else _empty_board()
    return sm


def _gated_signal(
    t: float, cnn: Board, hsv: Board, *, window_active: bool,
) -> DetectorSignals:
    return DetectorSignals(
        time_sec=t,
        cnn_board=cnn,
        is_match_active=True,
        hsv_board=hsv,
        effect_gate_window_active=window_active,
    )


# ============================
# (a) 相手連鎖中の上段 0.2 秒フリッカは確定盤面に入らない
# ============================


def test_effect_gate_rejects_short_flicker_in_top_row() -> None:
    """上段セルの一時的な CNN==HSV 一致 (_PERSIST_SEC 未満) は確定させない。"""
    row = next(iter(EFFECT_GATE_TOP_ROWS))
    confirmed = _stacked_board(row, 0, COLOR_EMPTY)  # 下は積まれている、対象セルは空
    sm = _make_effect_gate_sm(enable_effect_gate=True, confirmed=confirmed)

    cnn = _stacked_board(row, 0, COLOR_RED)
    hsv = _stacked_board(row, 0, COLOR_RED)
    # _PERSIST_SEC (0.2s) に満たない 2 フレーム (0.0s, 0.05s) だけ観測させ、
    # 3 フレーム目で元の EMPTY (背景) に戻す (= エフェクト痕の消滅を模擬)。
    sm.update(0, _gated_signal(0.0, cnn, hsv, window_active=True))
    sm.update(1, _gated_signal(_STEP_SEC, cnn, hsv, window_active=True))
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(row, 0) == COLOR_EMPTY  # まだ空

    reverted = _stacked_board(row, 0, COLOR_EMPTY)
    sm.update(
        2, _gated_signal(2 * _STEP_SEC, reverted, reverted, window_active=True),
    )
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(row, 0) == COLOR_EMPTY  # 消えて元の空のまま


# ============================
# (b) 同じ窓中の本物の設置 (0.4 秒以上持続) は遅延して確定する
# ============================


def test_effect_gate_commits_after_sustained_persistence() -> None:
    """_PERSIST_SEC 秒以上継続観測されれば、窓中でも遅延して確定する。"""
    row = next(iter(EFFECT_GATE_TOP_ROWS))
    confirmed = _stacked_board(row, 0, COLOR_EMPTY)
    sm = _make_effect_gate_sm(enable_effect_gate=True, confirmed=confirmed)

    cnn = _stacked_board(row, 0, COLOR_RED)
    hsv = _stacked_board(row, 0, COLOR_RED)
    t = 0.0
    # _PERSIST_SEC (0.2s) 未満の間は未確定であることを再確認しつつ持続させる。
    for i in range(3):
        sm.update(i, _gated_signal(t, cnn, hsv, window_active=True))
        t += _STEP_SEC
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(row, 0) == COLOR_EMPTY  # 0.10s 時点でまだ未確定

    # 継続して _PERSIST_SEC を超過させる (0.05s刻みで合計 0.25s 経過)。
    for i in range(3, 6):
        sm.update(i, _gated_signal(t, cnn, hsv, window_active=True))
        t += _STEP_SEC
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(row, 0) == COLOR_RED  # 持続確認を経て確定


# ============================
# (c) 窓外 (相手連鎖していない / 上段以外) は従来レイテンシ
# ============================


def test_effect_gate_window_inactive_uses_normal_fast_path() -> None:
    """window_active=False の間は EFFECT_GATE_TOP_ROWS でも通常の frame カウントで確定する。"""
    row = next(iter(EFFECT_GATE_TOP_ROWS))
    min_frames = 2
    confirmed = _stacked_board(row, 0, COLOR_EMPTY)
    sm = _make_effect_gate_sm(
        enable_effect_gate=True, recovery_min_frames=min_frames, confirmed=confirmed,
    )

    cnn = _stacked_board(row, 0, COLOR_RED)
    hsv = _stacked_board(row, 0, COLOR_RED)
    # window_active=False (相手連鎖していない/お邪魔着弾直後でもない) なら
    # 通常の recovery_min_frames (=2) で確定する (_PERSIST_SEC を待たない)。
    sm.update(0, _gated_signal(0.0, cnn, hsv, window_active=False))
    sm.update(1, _gated_signal(_STEP_SEC, cnn, hsv, window_active=False))
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(row, 0) == COLOR_RED  # 通常速度で確定済み


def test_effect_gate_ignores_rows_outside_top_rows() -> None:
    """window_active=True でも EFFECT_GATE_TOP_ROWS 外の行は通常速度で確定する。"""
    non_top_row = max(EFFECT_GATE_TOP_ROWS) + 1  # 上段ゲート対象外の行
    assert non_top_row not in EFFECT_GATE_TOP_ROWS
    min_frames = 2
    confirmed = _stacked_board(non_top_row, 0, COLOR_EMPTY)
    sm = _make_effect_gate_sm(
        enable_effect_gate=True, recovery_min_frames=min_frames, confirmed=confirmed,
    )

    cnn = _stacked_board(non_top_row, 0, COLOR_RED)
    hsv = _stacked_board(non_top_row, 0, COLOR_RED)
    sm.update(0, _gated_signal(0.0, cnn, hsv, window_active=True))
    sm.update(1, _gated_signal(_STEP_SEC, cnn, hsv, window_active=True))
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(non_top_row, 0) == COLOR_RED  # 上段外は対象外


# ============================
# (d) フラグ OFF で旧挙動 bit 一致
# ============================


def test_effect_gate_disabled_is_bit_identical_to_legacy() -> None:
    """enable_effect_gate=False なら signals.effect_gate_window_active=True でも無視される。

    復旧ゲート単体の既存挙動 (recovery_min_frames で即確定) と完全一致することを
    確認する (= 新規パラメータが未指定時と同じ結果になる backwards compat 保証)。
    """
    row = next(iter(EFFECT_GATE_TOP_ROWS))
    min_frames = 2
    confirmed = _stacked_board(row, 0, COLOR_EMPTY)
    sm_off = _make_effect_gate_sm(
        enable_effect_gate=False, recovery_min_frames=min_frames,
        confirmed=confirmed.copy(),
    )
    sm_legacy = BoardStateMachine(
        enable_stable_recovery_gate=True, recovery_min_frames=min_frames,
    )
    sm_legacy._ctx.state = BoardState.STABLE
    sm_legacy._ctx.confirmed_board = confirmed.copy()

    cnn = _stacked_board(row, 0, COLOR_RED)
    hsv = _stacked_board(row, 0, COLOR_RED)
    for i in range(min_frames):
        t = i * _STEP_SEC
        # sm_off には (誤って呼び出し元が渡したケースを模して)
        # window_active=True を渡すが、enable_effect_gate=False なので無視される。
        sm_off.update(i, _gated_signal(t, cnn, hsv, window_active=True))
        sm_legacy.update(
            i,
            DetectorSignals(
                time_sec=t, cnn_board=cnn, is_match_active=True, hsv_board=hsv,
            ),
        )

    assert sm_off.context.confirmed_board == sm_legacy.context.confirmed_board
    assert sm_off.context.confirmed_board is not None
    assert sm_off.context.confirmed_board.get(row, 0) == COLOR_RED


# ============================
# 低レベルヘルパー単体テスト
# ============================


def test_update_effect_gate_hold_resets_on_color_change() -> None:
    """候補色が変わると持続計測がリセットされる (別色の再持続を要求)。"""
    hold: dict[tuple[int, int], tuple[int, float]] = {}
    cell = (1, 0)
    assert _update_effect_gate_hold(hold, cell, 1, 0.0, 0.2) is False
    assert _update_effect_gate_hold(hold, cell, 1, 0.1, 0.2) is False
    # 候補色が変わる (1 → 2) と計測がリセットされる
    assert _update_effect_gate_hold(hold, cell, 2, 0.15, 0.2) is False
    assert _update_effect_gate_hold(hold, cell, 2, 0.36, 0.2) is True  # 0.15→0.36 で0.2s超過


def test_update_effect_gate_hold_fires_after_persist_sec() -> None:
    """同一候補色が persist_sec 秒以上継続すれば True を返す。"""
    hold: dict[tuple[int, int], tuple[int, float]] = {}
    cell = (2, 3)
    assert _update_effect_gate_hold(hold, cell, 4, 10.0, 0.4) is False
    assert _update_effect_gate_hold(hold, cell, 4, 10.39, 0.4) is False  # 僅かに未達
    assert _update_effect_gate_hold(hold, cell, 4, 10.40, 0.4) is True  # ちょうど到達


# ============================
# 案B (2026-08-04): _update_all_clear_pending 単体テスト
# ============================


def test_update_all_clear_pending_chain_fired_forces_false() -> None:
    """chain_fired=True なら prev_pending の値に関わらず必ず False にクリアする。"""
    empty_all_clear_board = _empty_board()  # 完全に空 = 全消し形状
    assert _update_all_clear_pending(
        True, empty_all_clear_board, 1000, chain_fired=True,
    ) is False
    assert _update_all_clear_pending(
        False, empty_all_clear_board, 1000, chain_fired=True,
    ) is False


def test_update_all_clear_pending_keeps_prev_when_not_confirmed() -> None:
    """confirmed_board=None (STABLE 以外) では直前ラッチ状態を維持する。"""
    assert _update_all_clear_pending(
        True, None, 1000, chain_fired=False,
    ) is True
    assert _update_all_clear_pending(
        False, None, 1000, chain_fired=False,
    ) is False


def test_update_all_clear_pending_true_on_all_clear_board() -> None:
    """STABLE 確定 + 全消し盤面 (完全に空 + score>0) では True にセットする。"""
    empty_board = _empty_board()
    assert _update_all_clear_pending(
        False, empty_board, 1000, chain_fired=False,
    ) is True


def test_update_all_clear_pending_false_on_non_all_clear_board() -> None:
    """STABLE 確定でも全消し形状でなければ False。"""
    non_all_clear = _board_with_puyo(6, 0, COLOR_RED)
    assert _update_all_clear_pending(
        True, non_all_clear, 1000, chain_fired=False,
    ) is False


def _board_with_puyo(row: int, col: int, color: int) -> Board:
    b = Board()
    b.set(row, col, color)
    return b


# ============================
# 案B (2026-08-04): _compute_effect_gate_window_active 単体テスト
# ============================


def _black_frame() -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _bright_top_row_frame() -> np.ndarray:
    """DEFAULT_P1_REGION の EFFECT_GATE_TOP_ROWS 内セルを高輝度で塗った合成フレーム。"""
    frame = _black_frame()
    row = next(iter(EFFECT_GATE_TOP_ROWS))
    x1, y1, x2, y2 = DEFAULT_P1_REGION.cell_sample_rect(row, 0)
    frame[y1:y2, x1:x2] = (255, 255, 255)
    return frame


def test_compute_effect_gate_window_active_disabled_passes_through() -> None:
    """enable_visual_gate=False なら time_window_active をそのまま返す (両値)。"""
    for tw in (True, False):
        assert _compute_effect_gate_window_active(
            time_window_active=tw,
            own_chain_active=True,  # 他条件が False 側でも無視されるはず
            all_clear_pending=True,
            frame_bgr=None,
            region=DEFAULT_P1_REGION,
            enable_visual_gate=False,
        ) is tw


def test_compute_effect_gate_window_active_own_chain_active_blocks() -> None:
    """own_chain_active=True (自連鎖中) なら他条件が全部クリアでも False。"""
    assert _compute_effect_gate_window_active(
        time_window_active=True,
        own_chain_active=True,
        all_clear_pending=False,
        frame_bgr=_bright_top_row_frame(),
        region=DEFAULT_P1_REGION,
        enable_visual_gate=True,
    ) is False


def test_compute_effect_gate_window_active_all_clear_pending_blocks() -> None:
    """all_clear_pending=True (全消しラッチ) なら他条件が全部クリアでも False。"""
    assert _compute_effect_gate_window_active(
        time_window_active=True,
        own_chain_active=False,
        all_clear_pending=True,
        frame_bgr=_bright_top_row_frame(),
        region=DEFAULT_P1_REGION,
        enable_visual_gate=True,
    ) is False


def test_compute_effect_gate_window_active_frame_none_blocks() -> None:
    """frame_bgr=None (画像なし) は安全弁として False。"""
    assert _compute_effect_gate_window_active(
        time_window_active=True,
        own_chain_active=False,
        all_clear_pending=False,
        frame_bgr=None,
        region=DEFAULT_P1_REGION,
        enable_visual_gate=True,
    ) is False


def test_compute_effect_gate_window_active_time_window_inactive_blocks() -> None:
    """time_window_active=False (既存時間窓が不発) なら他条件に関わらず False。"""
    assert _compute_effect_gate_window_active(
        time_window_active=False,
        own_chain_active=False,
        all_clear_pending=False,
        frame_bgr=_bright_top_row_frame(),
        region=DEFAULT_P1_REGION,
        enable_visual_gate=True,
    ) is False


def test_compute_effect_gate_window_active_all_clear_conditions_true() -> None:
    """4条件すべてクリア (時間窓True・自連鎖なし・全消しなし・高輝度検出) で True。"""
    assert _compute_effect_gate_window_active(
        time_window_active=True,
        own_chain_active=False,
        all_clear_pending=False,
        frame_bgr=_bright_top_row_frame(),
        region=DEFAULT_P1_REGION,
        enable_visual_gate=True,
    ) is True


def test_compute_effect_gate_window_active_no_visual_glow_is_false() -> None:
    """3条件クリアでも視覚グロー未検出 (黒フレーム) なら False。"""
    assert _compute_effect_gate_window_active(
        time_window_active=True,
        own_chain_active=False,
        all_clear_pending=False,
        frame_bgr=_black_frame(),
        region=DEFAULT_P1_REGION,
        enable_visual_gate=True,
    ) is False


# ============================
# 案B (2026-08-04): RecognitionPipeline 統合レベル
# ============================


class _StubMatchDetectorForGate:
    """常に IN_MATCH を返す MatchStateDetector スタブ (統合テスト専用最小版)。"""

    def detect(self, frame: np.ndarray) -> object:
        class _R:
            state = MatchState.IN_MATCH
            bg_value = 100.0
            bg_saturation = 50.0
            samples = 1
        return _R()


class _StubImageReaderForGate:
    """固定の空盤面を返す ImageReader スタブ (統合テスト専用最小版)。"""

    def read_both_boards(
        self, frame: np.ndarray, **_kwargs: object,
    ) -> tuple[Board, Board]:
        return _empty_board(), _empty_board()


def _make_gate_pipe(**kwargs: object) -> RecognitionPipeline:
    return RecognitionPipeline(
        image_reader=_StubImageReaderForGate(),  # type: ignore[arg-type]
        match_state_detector=_StubMatchDetectorForGate(),  # type: ignore[arg-type]
        **kwargs,
    )


def test_enable_effect_visual_gate_default_is_false() -> None:
    """enable_effect_visual_gate 未指定時は既定 False (backwards compat)。"""
    pipe = _make_gate_pipe()
    assert pipe._enable_effect_visual_gate is False


def test_enable_effect_visual_gate_flag_is_accepted_and_stored() -> None:
    """enable_effect_visual_gate=True を渡すと self._enable_effect_visual_gate に反映される。"""
    pipe = _make_gate_pipe(enable_effect_visual_gate=True)
    assert pipe._enable_effect_visual_gate is True


def test_reset_clears_all_clear_pending_latches() -> None:
    """reset() で _all_clear_pending_1p/_2p が False に戻る (試合切替時クリア)。"""
    pipe = _make_gate_pipe()
    pipe._all_clear_pending_1p = True
    pipe._all_clear_pending_2p = True
    pipe.reset()
    assert pipe._all_clear_pending_1p is False
    assert pipe._all_clear_pending_2p is False
