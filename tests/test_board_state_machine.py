"""BoardStateMachine 骨組みテスト (Phase B-1).

state 遷移ロジックのみ検証する。各 detector の中身は B-2 で別途テスト。
"""

from __future__ import annotations

import pytest

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, COLOR_RED, Board
from src.board_state_machine import (
    BoardState,
    BoardStateMachine,
    DetectorSignals,
    NON_STABLE_STATES,
    NullDetector,
    STABLE_RECOVERY_MIN_FRAMES,
    StateContext,
    StateTransitionDetector,
)


# ============================
# helper
# ============================


def _empty_board() -> Board:
    return Board()


def _board_with_red(row: int, col: int) -> Board:
    b = Board()
    b.set(row, col, COLOR_RED)
    return b


def _signal(
    t: float, board: Board, *, match: bool = True,
    next_pair: tuple[int, int] | None = None,
    hsv_board: "Board | None" = None,
) -> DetectorSignals:
    return DetectorSignals(
        time_sec=t,
        cnn_board=board,
        is_match_active=match,
        next_pair=next_pair,
        hsv_board=hsv_board,
    )


class _ForceState:
    """指定 frame で指定 state に強制遷移する detector (テスト用)."""

    def __init__(
        self, target_state: BoardState, fire_at_frame: int,
    ) -> None:
        self._target = target_state
        self._fire = fire_at_frame

    def detect(
        self, ctx: StateContext, signals: DetectorSignals,
    ) -> BoardState | None:
        if ctx.frame_idx == self._fire:
            return self._target
        return None


# ============================
# 基本状態
# ============================


def test_initial_state_is_menu() -> None:
    sm = BoardStateMachine()
    assert sm.context.state == BoardState.MENU
    assert sm.context.confirmed_board is None


def test_inactive_match_keeps_menu() -> None:
    sm = BoardStateMachine(detectors=[NullDetector()])
    ctx = sm.update(0, _signal(0.0, _empty_board(), match=False))
    assert ctx.state == BoardState.MENU
    assert ctx.confirmed_board is None


# ============================
# STABLE 連続多数決
# ============================


def test_stable_confirmed_after_n_consecutive_frames() -> None:
    """同一盤面が N 回連続で観測されたら STABLE 確定。"""
    n = 6
    sm = BoardStateMachine(stable_frame_count=n)
    board = _board_with_red(12, 0)
    last_ctx: StateContext | None = None
    for i in range(n):
        last_ctx = sm.update(i, _signal(0.05 * i, board.copy()))
    assert last_ctx is not None
    assert last_ctx.state == BoardState.STABLE
    assert last_ctx.confirmed_board is not None
    assert last_ctx.confirmed_board == board


def test_match_just_started_forces_empty_initial_confirmed() -> None:
    """cycle 71v: 試合開始直後の初回 STABLE 確定で confirmed=空 Board() を強制.

    CNN が背景を puyo 誤認しても物理ルール (= 試合開始時は空フィールド) で上書き。
    """
    n = 3
    sm = BoardStateMachine(stable_frame_count=n)
    # CNN は誤って (12, 0) と (12, 1) に puyo を見ている (= 背景誤認 simulation)
    corrupted = Board()
    corrupted.set(12, 0, COLOR_RED)
    corrupted.set(12, 1, COLOR_RED)
    last_ctx: StateContext | None = None
    for i in range(n):
        sig = DetectorSignals(
            time_sec=0.05 * i,
            cnn_board=corrupted.copy(),
            is_match_active=True,
            match_just_started=True,  # ← 試合開始 window 内
        )
        last_ctx = sm.update(i, sig)
    assert last_ctx is not None
    assert last_ctx.state == BoardState.STABLE
    # confirmed は空 Board() に強制されている (CNN の誤認を採用しない)
    assert last_ctx.confirmed_board is not None
    assert last_ctx.confirmed_board == Board()


def test_match_just_started_false_uses_cnn_observation() -> None:
    """cycle 71v: window 外なら従来通り CNN 多数決を採用 (回帰防止)."""
    n = 3
    sm = BoardStateMachine(stable_frame_count=n)
    board = _board_with_red(12, 0)
    last_ctx: StateContext | None = None
    for i in range(n):
        sig = DetectorSignals(
            time_sec=0.05 * i,
            cnn_board=board.copy(),
            is_match_active=True,
            match_just_started=False,  # ← window 外
        )
        last_ctx = sm.update(i, sig)
    assert last_ctx is not None
    assert last_ctx.state == BoardState.STABLE
    assert last_ctx.confirmed_board == board


def test_stable_not_confirmed_when_board_keeps_changing() -> None:
    """毎 frame 異なる盤面 → STABLE 確定しない。"""
    sm = BoardStateMachine(stable_frame_count=3)
    boards = [_board_with_red(12, c) for c in range(BOARD_COLS)]
    last_ctx: StateContext | None = None
    for i, b in enumerate(boards):
        last_ctx = sm.update(i, _signal(0.05 * i, b))
    assert last_ctx is not None
    assert last_ctx.state != BoardState.STABLE
    assert last_ctx.confirmed_board is None


def test_pending_count_resets_on_change() -> None:
    sm = BoardStateMachine(stable_frame_count=4)
    a = _empty_board()
    b = _board_with_red(12, 0)
    sm.update(0, _signal(0.0, a))
    sm.update(1, _signal(0.05, a))
    ctx = sm.update(2, _signal(0.10, b))
    assert ctx.pending_count == 1
    assert ctx.state != BoardState.STABLE


# ============================
# 強制遷移
# ============================


def test_detector_forces_chain_state() -> None:
    sm = BoardStateMachine(
        detectors=[_ForceState(BoardState.CHAIN, fire_at_frame=2)],
        stable_frame_count=2,
    )
    sm.update(0, _signal(0.0, _empty_board()))
    sm.update(1, _signal(0.05, _empty_board()))
    ctx = sm.update(2, _signal(0.10, _board_with_red(12, 0)))
    assert ctx.state == BoardState.CHAIN
    assert ctx.is_action()
    assert not ctx.is_stable()


def test_action_states_skip_pending_update() -> None:
    """NON-STABLE 中は pending_count を増やさない (= 認識を盤面確定に使わない)."""
    sm = BoardStateMachine(
        detectors=[_ForceState(BoardState.CHAIN, fire_at_frame=0)],
        stable_frame_count=3,
    )
    board = _board_with_red(12, 0)
    for i in range(5):
        ctx = sm.update(i, _signal(0.05 * i, board.copy()))
    # CHAIN のまま、pending_count は 0
    assert ctx.state == BoardState.CHAIN
    assert ctx.pending_count == 0
    assert ctx.confirmed_board is None


def test_stable_returns_after_chain_resyncs_with_cnn() -> None:
    """CHAIN → STABLE 復帰時に CNN 盤面で confirmed を即時更新する."""
    detectors = [
        _ForceState(BoardState.CHAIN, fire_at_frame=1),
        _ForceState(BoardState.STABLE, fire_at_frame=5),
    ]
    sm = BoardStateMachine(detectors=detectors, stable_frame_count=10)
    sm.update(0, _signal(0.0, _empty_board()))
    sm.update(1, _signal(0.05, _empty_board()))  # → CHAIN
    sm.update(2, _signal(0.10, _empty_board()))
    sm.update(3, _signal(0.15, _empty_board()))
    sm.update(4, _signal(0.20, _empty_board()))
    confirmed = _board_with_red(12, 3)
    ctx = sm.update(5, _signal(0.25, confirmed.copy()))  # → STABLE
    assert ctx.state == BoardState.STABLE
    assert ctx.confirmed_board == confirmed
    assert ctx.last_stable_idx == 5


# ============================
# next_queue
# ============================


def test_next_queue_appends_unique_pairs() -> None:
    sm = BoardStateMachine()
    sm.update(0, _signal(0.0, _empty_board(), next_pair=(1, 2)))
    sm.update(1, _signal(0.05, _empty_board(), next_pair=(1, 2)))  # 重複
    sm.update(2, _signal(0.10, _empty_board(), next_pair=(3, 4)))
    assert sm.context.next_queue == [(1, 2), (3, 4)]


def test_next_queue_caps_at_8() -> None:
    sm = BoardStateMachine()
    for i in range(12):
        sm.update(i, _signal(0.05 * i, _empty_board(), next_pair=(i, i + 1)))
    assert len(sm.context.next_queue) == 8
    # 末尾は最新ペア
    assert sm.context.next_queue[-1] == (11, 12)


# ============================
# reset
# ============================


def test_reset_clears_state() -> None:
    sm = BoardStateMachine(stable_frame_count=2)
    board = _board_with_red(12, 0)
    sm.update(0, _signal(0.0, board))
    sm.update(1, _signal(0.05, board))
    assert sm.context.state == BoardState.STABLE
    sm.reset()
    assert sm.context.state == BoardState.MENU
    assert sm.context.confirmed_board is None
    assert sm.context.next_queue == []


def test_reset_keep_match_state_preserves_stable() -> None:
    sm = BoardStateMachine(stable_frame_count=2)
    sm.update(0, _signal(0.0, _empty_board()))
    sm.update(1, _signal(0.05, _empty_board()))
    assert sm.context.state == BoardState.STABLE
    sm.reset(keep_match_state=True)
    assert sm.context.state == BoardState.STABLE
    # ただし confirmed は破棄される
    assert sm.context.confirmed_board is None


# ============================
# state 集合の整合性
# ============================


@pytest.mark.parametrize(
    "state",
    [BoardState.TSUMO_FALL, BoardState.CHAIN,
     BoardState.OJAMA_FALL, BoardState.EFFECT],
)
def test_non_stable_states_membership(state: BoardState) -> None:
    assert state in NON_STABLE_STATES


def test_stable_and_menu_not_in_non_stable(  # noqa: D103
) -> None:
    assert BoardState.STABLE not in NON_STABLE_STATES
    assert BoardState.MENU not in NON_STABLE_STATES


# ============================
# StateTransitionDetector Protocol 整合
# ============================


def test_null_detector_is_protocol() -> None:
    det: StateTransitionDetector = NullDetector()
    ctx = StateContext()
    sig = _signal(0.0, _empty_board())
    assert det.detect(ctx, sig) is None


# ============================
# F (cycle 56) STABLE 復帰ゲート
# ============================


def test_vote_majority_board_majority_color_adopted() -> None:
    """history 5 frame 中 3 frame で同色 → 多数決で採用."""
    from src.board_state_machine import _vote_majority_board

    history: list[Board] = []
    for _ in range(3):
        history.append(_board_with_red(12, 0))
    for _ in range(2):
        history.append(_empty_board())
    result = _vote_majority_board(history, min_votes=3)
    assert result.get(12, 0) == COLOR_RED


def test_vote_majority_board_minority_color_rejected() -> None:
    """history 5 frame 中 2 frame しか同色観測なし → EMPTY 維持."""
    from src.board import COLOR_EMPTY
    from src.board_state_machine import _vote_majority_board

    history: list[Board] = []
    for _ in range(2):
        history.append(_board_with_red(12, 0))
    for _ in range(3):
        history.append(_empty_board())
    result = _vote_majority_board(history, min_votes=3)
    assert result.get(12, 0) == COLOR_EMPTY


def test_vote_majority_board_empty_history_returns_empty_board() -> None:
    """空 history なら EMPTY 盤面を返す (= ガード無効)."""
    from src.board_state_machine import _vote_majority_board

    result = _vote_majority_board([], min_votes=3)
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            assert result.get(r, c) == 0


def test_merge_diff_empty_to_color_guard_blocks_minority() -> None:
    """F: EMPTY → 色 遷移 cell について、 多数決で同色が確認できない場合は baseline 維持."""
    from src.board_state_machine import _merge_diff_only

    baseline = _empty_board()
    cnn = _board_with_red(12, 0)
    # guard は EMPTY のみ (= 多数決で赤が確認できなかった)
    guard = _empty_board()
    merged = _merge_diff_only(
        baseline, cnn, empty_to_color_guard=guard,
    )
    # F により採用が抑制され baseline (= EMPTY) 維持
    assert merged.get(12, 0) == 0


def test_merge_diff_empty_to_color_guard_allows_majority() -> None:
    """F: guard で同色が確認できる cell は採用."""
    from src.board_state_machine import _merge_diff_only

    baseline = _empty_board()
    cnn = _board_with_red(12, 0)
    # guard でも赤 (= 多数決で赤が観測されていた)
    guard = _board_with_red(12, 0)
    merged = _merge_diff_only(
        baseline, cnn, empty_to_color_guard=guard,
    )
    assert merged.get(12, 0) == COLOR_RED


def test_non_stable_history_accumulates_in_chain_state() -> None:
    """F: CHAIN state 中、 cnn_board 履歴が context に蓄積される."""
    sm = BoardStateMachine(
        detectors=[_ForceState(BoardState.CHAIN, fire_at_frame=0)],
    )
    # frame 0: CHAIN に遷移
    sm.update(0, _signal(0.0, _empty_board()))
    # frame 1-3: CHAIN 中、 cnn_board を渡し続ける
    for i in range(1, 4):
        sm.update(i, _signal(0.05 * i, _board_with_red(12, 0)))
    # CHAIN 中 history に蓄積されている
    assert len(sm.context.non_stable_cnn_history) >= 3


def test_stable_resume_gate_blocks_2sec_residual() -> None:
    """F 統合: NON-STABLE 中 5 frame ずっと EMPTY cell を観測、
    STABLE 復帰 frame でのみ「色」 が CNN に出る → 多数決ガードで阻止される.

    = ユーザー指摘「2 秒残る」 パターンの再演防止テスト.
    """
    sm = BoardStateMachine(
        detectors=[
            _ForceState(BoardState.CHAIN, fire_at_frame=0),
            _ForceState(BoardState.STABLE, fire_at_frame=6),
        ],
    )
    # 初回 STABLE 確定のため confirmed_board に空盤面を準備
    # frame 0: CHAIN 遷移、 confirmed_board 未確定
    sm.update(0, _signal(0.0, _empty_board()))
    sm.context.confirmed_board = _empty_board()  # 明示的に空盤面を確定
    # frame 1-5: CHAIN 中、 全 EMPTY を観測 (= 背景認識安定)
    for i in range(1, 6):
        sm.update(i, _signal(0.05 * i, _empty_board()))
    # frame 6: STABLE 復帰、 CNN が突然「(12, 0) に赤」 と誤認
    sm.update(6, _signal(0.30, _board_with_red(12, 0)))
    # F ガードで誤認 cell は採用されず baseline (= EMPTY) 維持
    assert sm.context.state == BoardState.STABLE
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(12, 0) == 0


def test_stable_resume_gate_disabled_via_constructor() -> None:
    """F: enable_stable_resume_gate=False で従来挙動 (= ガードなし)."""
    sm = BoardStateMachine(
        detectors=[
            _ForceState(BoardState.CHAIN, fire_at_frame=0),
            _ForceState(BoardState.STABLE, fire_at_frame=6),
        ],
        enable_stable_resume_gate=False,
    )
    sm.update(0, _signal(0.0, _empty_board()))
    sm.context.confirmed_board = _empty_board()
    for i in range(1, 6):
        sm.update(i, _signal(0.05 * i, _empty_board()))
    sm.update(6, _signal(0.30, _board_with_red(12, 0)))
    # ガード無効 → 従来挙動 = 赤が confirmed に乗る (= ただし gravity filter で 浮きぷよ check は通る)
    # (12, 0) は最下段なので gravity filter は通過、 赤が記録される
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(12, 0) == COLOR_RED


# ============================
# 設計C 事後復旧ゲート テスト (2026-06-02)
# ============================


def _make_recovery_sm(min_frames: int = 3) -> BoardStateMachine:
    """復旧ゲート有効 state machine を生成するヘルパー。
    テスト用に min_frames を小さく設定できる。
    """
    return BoardStateMachine(
        enable_stable_recovery_gate=True,
        recovery_min_frames=min_frames,
    )


def _stable_signal_with_hsv(
    t: float,
    cnn: Board,
    hsv: Board,
) -> DetectorSignals:
    """STABLE 用に CNN/HSV 両盤面を乗せたシグナルを生成する。"""
    return DetectorSignals(
        time_sec=t,
        cnn_board=cnn,
        is_match_active=True,
        hsv_board=hsv,
    )


def test_recovery_gate_fires_after_n_frames() -> None:
    """設計C ②: N フレーム連続 CNN==HSV==有効色 かつ confirmed==EMPTY → 復旧される."""
    n = 3
    sm = _make_recovery_sm(min_frames=n)
    # STABLE 確定 (確定盤面は空)
    sm._ctx.state = BoardState.STABLE
    sm._ctx.confirmed_board = _empty_board()

    # N-1 フレーム: まだ発火しない
    cnn = _board_with_red(12, 0)
    hsv = _board_with_red(12, 0)
    for i in range(n - 1):
        sm.update(
            i, _stable_signal_with_hsv(0.05 * i, cnn, hsv),
        )
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(12, 0) == COLOR_EMPTY  # まだ空

    # N フレーム目: 発火して復旧
    sm.update(n - 1, _stable_signal_with_hsv(0.05 * (n - 1), cnn, hsv))
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(12, 0) == COLOR_RED  # 復旧された


def test_recovery_gate_does_not_fire_before_n_frames() -> None:
    """設計C ①: N-1 フレームでは発火しない."""
    n = 4
    sm = _make_recovery_sm(min_frames=n)
    sm._ctx.state = BoardState.STABLE
    sm._ctx.confirmed_board = _empty_board()

    cnn = _board_with_red(12, 0)
    hsv = _board_with_red(12, 0)
    for i in range(n - 1):
        sm.update(i, _stable_signal_with_hsv(0.05 * i, cnn, hsv))

    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(12, 0) == COLOR_EMPTY  # まだ空


def test_recovery_gate_no_fire_when_cnn_hsv_differ() -> None:
    """設計C ③: CNN≠HSV の場合は発火しない (安全弁A)."""
    from src.board import COLOR_BLUE
    n = 3
    sm = _make_recovery_sm(min_frames=n)
    sm._ctx.state = BoardState.STABLE
    sm._ctx.confirmed_board = _empty_board()

    cnn = _board_with_red(12, 0)
    hsv_diff = Board()
    hsv_diff.set(12, 0, COLOR_BLUE)  # CNN と異なる色

    for i in range(n + 2):
        sm.update(i, _stable_signal_with_hsv(0.05 * i, cnn, hsv_diff))

    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(12, 0) == COLOR_EMPTY  # 発火しない


def test_recovery_gate_no_fire_when_floating_puyo() -> None:
    """設計C ④: 下が空の浮きぷよになる場合は発火しない (安全弁C)."""
    n = 3
    sm = _make_recovery_sm(min_frames=n)
    sm._ctx.state = BoardState.STABLE
    # row=10, col=0 は confirmed==EMPTY
    # row=11, col=0 も confirmed==EMPTY (= 下が空 → 浮きぷよ)
    # row=12, col=0 は confirmed==EMPTY
    sm._ctx.confirmed_board = _empty_board()

    # (row=10, col=0) に赤を CNN/HSV で継続観測するが、
    # 下の (row=11, col=0) が EMPTY なので浮きぷよ → 発火しない
    cnn = Board()
    cnn.set(10, 0, COLOR_RED)
    hsv = Board()
    hsv.set(10, 0, COLOR_RED)

    for i in range(n + 2):
        sm.update(i, _stable_signal_with_hsv(0.05 * i, cnn, hsv))

    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(10, 0) == COLOR_EMPTY  # 浮きぷよで却下


def test_recovery_gate_resets_on_non_stable_transition() -> None:
    """設計C ⑤: NON-STABLE 遷移でカウンタと recovery_cells がクリアされる."""
    n = 2
    sm = _make_recovery_sm(min_frames=n)
    sm._ctx.state = BoardState.STABLE
    sm._ctx.confirmed_board = _empty_board()

    cnn = _board_with_red(12, 0)
    hsv = _board_with_red(12, 0)
    # N-1 フレーム蓄積 (カウンタに 1 が入る)
    sm.update(0, _stable_signal_with_hsv(0.0, cnn, hsv))
    assert sm.context.stable_recovery_counters.get((12, 0), 0) >= 1

    # CHAIN に強制遷移
    det = _ForceState(BoardState.CHAIN, fire_at_frame=1)
    sm._detectors.append(det)
    sm.update(1, _stable_signal_with_hsv(0.05, cnn, hsv))

    # カウンタと recovery_cells がクリアされている
    assert sm.context.stable_recovery_counters == {}
    assert sm.context.recovery_cells == set()


def test_recovery_gate_off_explicit() -> None:
    """設計C ⑥: フラグ OFF を明示すると従来挙動と完全同一 (発火しない) 回帰防止。

    2026-06-02: BoardStateMachine のデフォルトは enable_stable_recovery_gate=True に変更。
    このテストは False を明示して OFF 時の挙動が維持されることを回帰防止する。
    """
    sm = BoardStateMachine(enable_stable_recovery_gate=False)
    sm._ctx.state = BoardState.STABLE
    sm._ctx.confirmed_board = _empty_board()

    cnn = _board_with_red(12, 0)
    hsv = _board_with_red(12, 0)
    # STABLE_RECOVERY_MIN_FRAMES + 余裕 分 observations
    for i in range(STABLE_RECOVERY_MIN_FRAMES + 5):
        sm.update(i, _stable_signal_with_hsv(0.05 * i, cnn, hsv))

    # フラグ OFF → confirmed は EMPTY のまま
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(12, 0) == COLOR_EMPTY


# ============================
# 設計C 双方向拡張テスト (2026-06-02)
# ============================


def _board_with_color(row: int, col: int, color: int) -> Board:
    """指定 row/col に任意の色をセットした Board を返す。"""
    b = Board()
    b.set(row, col, color)
    return b


def test_recovery_gate_color_to_empty_ghost_removal() -> None:
    """双方向①: confirmed=色 だが CNN==HSV==EMPTY が N 連続 → 空に修正 (幽霊除去)。

    col5 お邪魔幽霊シナリオ: confirmed=赤 だが CNN/HSV は空継続。
    """
    n = 3
    sm = _make_recovery_sm(min_frames=n)
    sm._ctx.state = BoardState.STABLE
    # confirmed に赤が焼き付いている (幽霊)
    confirmed_init = Board()
    confirmed_init.set(12, 5, COLOR_RED)
    sm._ctx.confirmed_board = confirmed_init

    # CNN==HSV==EMPTY を N 連続
    cnn = _empty_board()
    hsv = _empty_board()
    for i in range(n):
        sm.update(i, _stable_signal_with_hsv(0.05 * i, cnn, hsv))

    # 幽霊が除去されて EMPTY になっている
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(12, 5) == COLOR_EMPTY


def test_recovery_gate_color_to_different_color_correction() -> None:
    """双方向②: confirmed=色A だが CNN==HSV==色B が N 連続 → 色B に訂正 (誤色修正)。

    黄→赤 誤色固定シナリオ: confirmed=赤 だが CNN/HSV は黄継続。
    """
    from src.board import COLOR_YELLOW
    n = 3
    sm = _make_recovery_sm(min_frames=n)
    sm._ctx.state = BoardState.STABLE
    # confirmed に赤が焼き付いている (誤色)
    confirmed_init = Board()
    confirmed_init.set(12, 4, COLOR_RED)
    sm._ctx.confirmed_board = confirmed_init

    # CNN==HSV==黄 を N 連続
    cnn = _board_with_color(12, 4, COLOR_YELLOW)
    hsv = _board_with_color(12, 4, COLOR_YELLOW)
    for i in range(n):
        sm.update(i, _stable_signal_with_hsv(0.05 * i, cnn, hsv))

    # 誤色が訂正されて黄になっている
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(12, 4) == COLOR_YELLOW


def test_recovery_gate_ojama_ghost_removal() -> None:
    """双方向③: confirmed=お邪魔(9) だが CNN==HSV==EMPTY が N 連続 → 空に除去。

    col5 お邪魔幽霊シナリオ: 通過済みお邪魔が confirmed に残留。
    """
    n = 3
    sm = _make_recovery_sm(min_frames=n)
    sm._ctx.state = BoardState.STABLE
    # confirmed にお邪魔幽霊
    confirmed_init = Board()
    confirmed_init.set(12, 5, COLOR_OJAMA)
    sm._ctx.confirmed_board = confirmed_init

    # CNN==HSV==EMPTY を N 連続
    cnn = _empty_board()
    hsv = _empty_board()
    for i in range(n):
        sm.update(i, _stable_signal_with_hsv(0.05 * i, cnn, hsv))

    # お邪魔幽霊が除去されて EMPTY になっている
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(12, 5) == COLOR_EMPTY


def test_recovery_gate_no_fire_when_cnn_hsv_disagree_bidirectional() -> None:
    """双方向④: CNN≠HSV の場合は方向2/3でも発火しない (安全弁A)。"""
    from src.board import COLOR_YELLOW
    n = 3
    sm = _make_recovery_sm(min_frames=n)
    sm._ctx.state = BoardState.STABLE
    # confirmed に赤が焼き付いている
    confirmed_init = Board()
    confirmed_init.set(12, 0, COLOR_RED)
    sm._ctx.confirmed_board = confirmed_init

    # CNN=EMPTY, HSV=黄 (不一致)
    cnn = _empty_board()
    hsv = _board_with_color(12, 0, COLOR_YELLOW)

    for i in range(n + 2):
        sm.update(i, _stable_signal_with_hsv(0.05 * i, cnn, hsv))

    # CNN≠HSV → 発火しない → confirmed=赤のまま
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(12, 0) == COLOR_RED


def test_recovery_gate_no_fire_at_n_minus_1_bidirectional() -> None:
    """双方向⑤: N-1 フレームでは方向2も発火しない。"""
    n = 5
    sm = _make_recovery_sm(min_frames=n)
    sm._ctx.state = BoardState.STABLE
    # confirmed に赤
    confirmed_init = Board()
    confirmed_init.set(12, 0, COLOR_RED)
    sm._ctx.confirmed_board = confirmed_init

    # CNN==HSV==EMPTY を N-1 連続
    cnn = _empty_board()
    hsv = _empty_board()
    for i in range(n - 1):
        sm.update(i, _stable_signal_with_hsv(0.05 * i, cnn, hsv))

    # N-1 では発火しない → まだ赤のまま
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(12, 0) == COLOR_RED


def test_recovery_gate_no_fire_when_hsv_board_none_bidirectional() -> None:
    """双方向⑥: hsv_board=None の場合は方向2でも発火しない (安全弁A)。"""
    n = 3
    sm = _make_recovery_sm(min_frames=n)
    sm._ctx.state = BoardState.STABLE
    # confirmed に赤が焼き付いている
    confirmed_init = Board()
    confirmed_init.set(12, 0, COLOR_RED)
    sm._ctx.confirmed_board = confirmed_init

    # hsv_board なし
    cnn = _empty_board()
    for i in range(n + 2):
        sig = DetectorSignals(
            time_sec=0.05 * i,
            cnn_board=cnn,
            is_match_active=True,
            hsv_board=None,  # None → 発火しない
        )
        sm.update(i, sig)

    # hsv_board=None → confirmed=赤のまま
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(12, 0) == COLOR_RED


def test_recovery_gate_off_does_not_fix_ghost() -> None:
    """双方向⑦: フラグ OFF を明示すると方向2も従来挙動と完全同一 (発火しない) 回帰防止。

    2026-06-02: デフォルト True 変更後も enable_stable_recovery_gate=False 明示で
    OFF 時の従来挙動が維持されることを保証する。
    """
    sm = BoardStateMachine(enable_stable_recovery_gate=False)
    sm._ctx.state = BoardState.STABLE
    # confirmed に赤が焼き付いている
    confirmed_init = Board()
    confirmed_init.set(12, 0, COLOR_RED)
    sm._ctx.confirmed_board = confirmed_init

    cnn = _empty_board()
    hsv = _empty_board()
    for i in range(STABLE_RECOVERY_MIN_FRAMES + 5):
        sm.update(i, _stable_signal_with_hsv(0.05 * i, cnn, hsv))

    # フラグ OFF → confirmed=赤のまま
    assert sm.context.confirmed_board is not None
    assert sm.context.confirmed_board.get(12, 0) == COLOR_RED
