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

from src.board import BOARD_ROWS, COLOR_BLUE, COLOR_EMPTY, COLOR_RED, Board
from src.board_state_machine import (
    BoardState,
    BoardStateMachine,
    DetectorSignals,
    EFFECT_GATE_TOP_ROWS,
    _update_effect_gate_hold,
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
