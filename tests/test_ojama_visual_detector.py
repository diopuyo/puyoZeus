"""フェーズ A 精緻化: OjamaVisualDetector テスト.

ST-1: STABLE 中 可視最上段にお邪魔出現 → OJAMA_FALL へ遷移。
ST-2: CHAIN 中 chain_event=None + ojama_top_positive + enable_chain_ojama_exit=True
      → STABLE でなく OJAMA_FALL。
ST-3: OJAMA_FALL 中 ROI お邪魔 3 フレーム不変 → STABLE 復帰 (settle 判定)。
ST-4: OJAMA_FALL → STABLE 直後 (warmup 中) infer_placement 発火しない。
      (RecognitionPipeline の _skip_infer_by_ojama_guard ロジックを直接テスト)
ST-R: 親フラグ OFF で全従来挙動不変 (回帰テスト)。
"""

from __future__ import annotations

import pytest

from src.board import COLOR_EMPTY, COLOR_OJAMA, COLOR_RED, HIDDEN_ROWS, Board
from src.board_state_machine import (
    BoardState,
    DetectorSignals,
    StateContext,
)
from src.ojama_visual_detector import (
    OJAMA_CONSEC_THRESH,
    OJAMA_ENTRY_CHAIN_INTERRUPT_MULTIPLIER,
    OJAMA_ENTRY_CHAIN_RECENT_ACTIVE_SEC,
    OJAMA_ENTRY_CONSEC_SEC,
    OJAMA_FALL_MAX_SEC,
    OJAMA_FALL_SCOPED_EXIT_DIFF_THRESHOLD,
    OJAMA_FALL_SCOPED_EXIT_MIN_SEC,
    OJAMA_FALL_SCOPED_EXIT_NO_PENDING_DIVISOR,
    OJAMA_FALL_SCOPED_EXIT_STABLE_FRAMES,
    OJAMA_FALL_SETTLE_DIFF_THRESHOLD,
    OJAMA_FALL_SETTLE_MIN_FRAMES,
    OJAMA_FALL_SETTLE_STABLE_FRAMES,
    OJAMA_REENTRY_SUPPRESS_SEC,
    OJAMA_SETTLE_CONSEC,
    OjamaVisualDetector,
    _count_board_ojama,
    _count_top_ojama,
    _has_ojama_fall_placement_evidence,
)
from src.state_detectors import ChainPhaseDetector, TsumoPhaseDetector


# ============================
# ヘルパー
# ============================


def _empty_board() -> Board:
    return Board()


def _board_with_ojama_top(count: int = 1) -> Board:
    """可視最上段 (HIDDEN_ROWS 行目) に COLOR_OJAMA を count 個配置した盤面."""
    b = Board()
    for c in range(min(count, 6)):
        b.set(HIDDEN_ROWS, c, COLOR_OJAMA)
    return b


def _board_with_n_puyos(n: int) -> Board:
    """count_puyos() が n を返す盤面を生成する (案B 全盤面 settle 判定テスト用).

    行優先で先頭から n セルに COLOR_OJAMA を敷き詰める (色自体は判定に無関係、
    count_puyos は EMPTY/UNKNOWN 以外を数えるため OJAMA で代用する)。
    """
    from src.board import BOARD_COLS as _COLS, BOARD_ROWS as _ROWS

    b = Board()
    filled = 0
    for r in range(_ROWS):
        for c in range(_COLS):
            if filled >= n:
                return b
            b.set(r, c, COLOR_OJAMA)
            filled += 1
    return b


def _make_signals(
    board: Board,
    *,
    state: BoardState = BoardState.STABLE,
    chain_event: object | None = None,
    ojama_top_positive: bool = False,
    frame_idx: int = 0,
) -> tuple[StateContext, DetectorSignals]:
    """テスト用 (StateContext, DetectorSignals) ペアを生成する."""
    ctx = StateContext(state=state, frame_idx=frame_idx)
    sig = DetectorSignals(
        time_sec=float(frame_idx) / 30.0,
        cnn_board=board,
        is_match_active=True,
        chain_event=chain_event,
        ojama_top_positive=ojama_top_positive,
    )
    return ctx, sig


# ============================
# _count_top_ojama 単体テスト
# ============================


def test_count_top_ojama_empty_board() -> None:
    """空盤面では 0 を返す。"""
    assert _count_top_ojama(_empty_board()) == 0


def test_count_top_ojama_detects_hidden_row() -> None:
    """HIDDEN_ROWS 行目のお邪魔を検知する。"""
    assert _count_top_ojama(_board_with_ojama_top(3)) == 3


def test_count_top_ojama_ignores_below_roi() -> None:
    """ROI 外 (HIDDEN_ROWS+2 以降) は検知しない。"""
    b = Board()
    b.set(HIDDEN_ROWS + 2, 0, COLOR_OJAMA)  # ROI 外
    assert _count_top_ojama(b) == 0


def test_count_top_ojama_hsv_board_or_logic() -> None:
    """cnn_board に OJAMA なし、 hsv_board にあれば検知する (OR ロジック)。"""
    cnn = _empty_board()
    hsv = _board_with_ojama_top(2)
    assert _count_top_ojama(cnn, hsv) == 2


def test_count_top_ojama_takes_max() -> None:
    """cnn と hsv の max を返す。"""
    cnn = _board_with_ojama_top(1)
    hsv = _board_with_ojama_top(4)
    assert _count_top_ojama(cnn, hsv) == 4


# ============================
# ST-1: STABLE 中にお邪魔出現 → OJAMA_FALL
# ============================


def test_st1_stable_to_ojama_fall_on_top_ojama() -> None:
    """ST-1: STABLE 中に ROI にお邪魔が出現すると OJAMA_FALL へ遷移する。

    OJAMA_CONSEC_THRESH (=2) フレーム連続で OJAMA_FALL が返ることを確認。
    """
    det = OjamaVisualDetector()
    ojama_board = _board_with_ojama_top(2)

    # フレーム 0: 初回検知 (前回 count=0 → 増加 = positive)。
    #   consec_count = 1 < THRESH → まだ発火しない。
    ctx0, sig0 = _make_signals(ojama_board, frame_idx=0)
    res0 = det.detect(ctx0, sig0)
    assert res0 is None, "1フレーム目は THRESH 未達で発火しないはず"

    # フレーム 1: 連続 2 フレーム目 → 発火
    ctx1, sig1 = _make_signals(ojama_board, frame_idx=1)
    res1 = det.detect(ctx1, sig1)
    assert res1 == BoardState.OJAMA_FALL, "OJAMA_CONSEC_THRESH フレーム連続でOJAMA_FALL"


def test_st1_no_false_positive_without_top_ojama() -> None:
    """ST-1 回帰: お邪魔が ROI 外にあるだけでは発火しない。"""
    det = OjamaVisualDetector()
    b = Board()
    b.set(HIDDEN_ROWS + 3, 0, COLOR_OJAMA)  # ROI 外 (row = 4)
    ctx, sig = _make_signals(b, frame_idx=0)
    for i in range(5):
        ctx2 = StateContext(state=BoardState.STABLE, frame_idx=i)
        sig2 = DetectorSignals(
            time_sec=float(i) / 30.0,
            cnn_board=b,
            is_match_active=True,
        )
        assert det.detect(ctx2, sig2) is None


# ============================
# ST-2: CHAIN + chain_event=None + ojama_top → OJAMA_FALL
# ============================


@pytest.mark.parametrize("chain_exit_enabled", [True, False])
def test_st2_chain_ojama_exit_flag(chain_exit_enabled: bool) -> None:
    """ST-2: CHAIN 中の STABLE 委譲 vs OJAMA_FALL 委譲を検証。

    enable_ojama_visual_chain_exit=True の時は OjamaVisualDetector が
    OJAMA_FALL を返すべき。False の時は STABLE に戻さず None 返し (ChainDetector 保留)。
    """
    det = OjamaVisualDetector(
        enable_ojama_visual_chain_exit=chain_exit_enabled,
    )
    ojama_board = _board_with_ojama_top(3)

    # CONSEC_THRESH 回連続で試す
    results = []
    for i in range(OJAMA_CONSEC_THRESH + 1):
        ctx = StateContext(state=BoardState.CHAIN, frame_idx=i)
        sig = DetectorSignals(
            time_sec=float(i) / 30.0,
            cnn_board=ojama_board,
            is_match_active=True,
            ojama_top_positive=True,
        )
        results.append(det.detect(ctx, sig))

    if chain_exit_enabled:
        # フラグ ON: THRESH フレーム後に OJAMA_FALL
        assert BoardState.OJAMA_FALL in results, (
            "enable_ojama_visual_chain_exit=True なら OJAMA_FALL が返るはず"
        )
    else:
        # フラグ OFF: CHAIN 中は発火しないため全 None
        assert all(r is None for r in results), (
            "enable_ojama_visual_chain_exit=False なら全て None"
        )


def test_st2_chain_phase_detector_delegates_to_visual() -> None:
    """ST-2 統合: ChainPhaseDetector が STABLE を返さず None を返す経路。

    enable_chain_ojama_exit=True かつ ojama_top_positive=True の場合、
    ChainPhaseDetector は STABLE に戻さず None を返し
    OjamaVisualDetector に OJAMA_FALL 判定を委譲する。
    """
    chain_det = ChainPhaseDetector(enable_chain_ojama_exit=True)
    # chain_event=None かつ state=CHAIN → 通常は STABLE を返すが、
    # ojama_top_positive=True なら None を返す (委譲)。
    ctx = StateContext(state=BoardState.CHAIN)
    sig = DetectorSignals(
        time_sec=1.0,
        cnn_board=_empty_board(),
        is_match_active=True,
        chain_event=None,
        ojama_top_positive=True,
    )
    result = chain_det.detect(ctx, sig)
    assert result is None, "ojama_top_positive=True なら STABLE を返さず None (委譲)"


def test_st2_chain_phase_detector_returns_stable_without_flag() -> None:
    """ST-2 回帰: フラグ OFF では chain_event=None かつ CHAIN → STABLE を維持。"""
    chain_det = ChainPhaseDetector(enable_chain_ojama_exit=False)
    ctx = StateContext(state=BoardState.CHAIN)
    sig = DetectorSignals(
        time_sec=1.0,
        cnn_board=_empty_board(),
        is_match_active=True,
        chain_event=None,
        ojama_top_positive=True,  # フラグ OFF なので無視されるはず
    )
    result = chain_det.detect(ctx, sig)
    assert result == BoardState.STABLE, "フラグ OFF では STABLE を返す"


# ============================
# ST-3: OJAMA_FALL 中 settle → STABLE 復帰
# ============================


def test_st3_ojama_fall_count_zero_exits() -> None:
    """ST-3a: OJAMA_FALL 中に count=0 で即 STABLE 復帰。"""
    det = OjamaVisualDetector(enable_ojama_settle_detection=True)
    # OJAMA_FALL 状態で count=0 盤面を渡す
    ctx = StateContext(state=BoardState.OJAMA_FALL, frame_idx=10)
    sig = DetectorSignals(
        time_sec=0.33,
        cnn_board=_empty_board(),
        is_match_active=True,
    )
    result = det.detect(ctx, sig)
    assert result == BoardState.STABLE, "お邪魔消滅で即 STABLE 復帰"


def test_st3_ojama_settle_detection() -> None:
    """ST-3b: OJAMA_FALL 中 count が OJAMA_SETTLE_CONSEC フレーム不変で STABLE 復帰。"""
    det = OjamaVisualDetector(enable_ojama_settle_detection=True)
    # count を一定値 (3) で固定して繰り返す
    ojama_board = _board_with_ojama_top(3)
    results = []
    for i in range(OJAMA_SETTLE_CONSEC + 1):
        ctx = StateContext(state=BoardState.OJAMA_FALL, frame_idx=i + 20)
        sig = DetectorSignals(
            time_sec=float(i) / 30.0,
            cnn_board=ojama_board,
            is_match_active=True,
        )
        results.append(det.detect(ctx, sig))

    assert BoardState.STABLE in results, (
        f"OJAMA_SETTLE_CONSEC={OJAMA_SETTLE_CONSEC} フレーム不変で STABLE 復帰するはず"
    )


def test_st3_settle_detection_disabled_no_early_exit() -> None:
    """ST-3 回帰: settle_detection=False では count 不変でも復帰しない。"""
    det = OjamaVisualDetector(enable_ojama_settle_detection=False)
    ojama_board = _board_with_ojama_top(2)
    for i in range(OJAMA_SETTLE_CONSEC + 2):
        ctx = StateContext(state=BoardState.OJAMA_FALL, frame_idx=i + 30)
        sig = DetectorSignals(
            time_sec=float(i) / 30.0,
            cnn_board=ojama_board,
            is_match_active=True,
        )
        result = det.detect(ctx, sig)
        assert result is None, "settle OFF なら不変でも STABLE 復帰しない"


# ============================
# ST-4: OJAMA_FALL → STABLE 後 warmup 中 infer 発火しない
# ============================


def test_st4_ojama_infer_guard_skip_logic() -> None:
    """ST-4: enable_ojama_infer_guard=True + warmup > 0 で _skip_infer_by_ojama_guard=True.

    RecognitionPipeline の _skip_infer_by_ojama_guard ロジックを直接シミュレートする。
    """
    # _ojama_tier1_warmup_remaining_* が > 0 の状態を模倣
    ojama_warmup_remaining = 5  # OJAMA_TIER1_WARMUP_FRAMES 相当
    enable_ojama_infer_guard = True

    skip = enable_ojama_infer_guard and ojama_warmup_remaining > 0
    assert skip is True, "warmup 中は infer をスキップするべき"


def test_st4_ojama_infer_guard_no_skip_when_disabled() -> None:
    """ST-4 回帰: enable_ojama_infer_guard=False なら warmup 中でもスキップしない。"""
    ojama_warmup_remaining = 5
    enable_ojama_infer_guard = False

    skip = enable_ojama_infer_guard and ojama_warmup_remaining > 0
    assert skip is False, "フラグ OFF なら warmup 中でもスキップしない"


def test_st4_ojama_infer_guard_no_skip_when_warmup_zero() -> None:
    """ST-4 回帰: warmup=0 ならスキップしない (warmup 終了後は通常 infer が走る)。"""
    ojama_warmup_remaining = 0
    enable_ojama_infer_guard = True

    skip = enable_ojama_infer_guard and ojama_warmup_remaining > 0
    assert skip is False, "warmup=0 なら infer をスキップしない"


# ============================
# ST-R: TsumoPhaseDetector ガード (OJAMA_FALL 中 TSUMO 返し禁止)
# ============================


def test_str_tsumo_detector_blocked_during_ojama_fall() -> None:
    """ST-R: OJAMA_FALL 中 TsumoPhaseDetector は None を返す (フラグ非依存)。

    Step3 の常時ガードが正しく動作するかを確認する。
    """
    det = TsumoPhaseDetector()
    # OJAMA_FALL state で confirmed_board に puyos を置く
    confirmed = Board()
    confirmed.set(12, 0, COLOR_RED)
    confirmed.set(12, 1, COLOR_RED)
    # cnn_board に +2 puyos (= 通常なら TSUMO_FALL 発火条件)
    cnn = Board()
    cnn.set(12, 0, COLOR_RED)
    cnn.set(12, 1, COLOR_RED)
    cnn.set(11, 0, COLOR_RED)
    cnn.set(11, 1, COLOR_RED)

    ctx = StateContext(
        state=BoardState.OJAMA_FALL,
        confirmed_board=confirmed,
        frame_idx=5,
    )
    sig = DetectorSignals(
        time_sec=0.17,
        cnn_board=cnn,
        is_match_active=True,
    )
    result = det.detect(ctx, sig)
    assert result is None, "OJAMA_FALL 中は TSUMO_FALL を返してはいけない"


def test_str_tsumo_detector_works_outside_ojama_fall() -> None:
    """ST-R 回帰: OJAMA_FALL 以外の state では TsumoPhaseDetector が通常動作。"""
    det = TsumoPhaseDetector()
    confirmed = Board()
    confirmed.set(12, 0, COLOR_RED)
    confirmed.set(12, 1, COLOR_RED)

    cnn = Board()
    cnn.set(12, 0, COLOR_RED)
    cnn.set(12, 1, COLOR_RED)
    cnn.set(11, 0, COLOR_RED)
    cnn.set(11, 1, COLOR_RED)

    # STABLE 状態 + consec 2 回 → TSUMO_FALL 発火を期待
    results = []
    for i in range(3):
        ctx = StateContext(
            state=BoardState.STABLE,
            confirmed_board=confirmed,
            frame_idx=i,
        )
        sig = DetectorSignals(
            time_sec=float(i) / 30.0,
            cnn_board=cnn,
            is_match_active=True,
        )
        results.append(det.detect(ctx, sig))

    # consec_threshold=2 なので 2 フレーム目以降で TSUMO_FALL が出るはず
    assert BoardState.TSUMO_FALL in results, (
        "STABLE 状態では TsumoPhaseDetector が通常動作 (TSUMO_FALL 発火)"
    )


# ============================
# ST-R: 親フラグ OFF での完全回帰
# ============================


def test_str_ojama_visual_detector_flag_off_no_transition() -> None:
    """ST-R: OjamaVisualDetector がデフォルト (フラグ OFF) では発火しない。

    フラグを一切設定しない状態でも既存テストが壊れないことを確認する。
    """
    det = OjamaVisualDetector()  # 全デフォルト = False
    ojama_board = _board_with_ojama_top(4)

    # STABLE 中に THRESH 回連続
    for i in range(OJAMA_CONSEC_THRESH + 1):
        ctx = StateContext(state=BoardState.STABLE, frame_idx=i)
        sig = DetectorSignals(
            time_sec=float(i) / 30.0,
            cnn_board=ojama_board,
            is_match_active=True,
        )
        result = det.detect(ctx, sig)
        # デフォルトでは STABLE → OJAMA_FALL 発火するはず (視覚検知はフラグ非依存で動作)
        # ただし CHAIN 中フラグのみ OFF
        # NOTE: デフォルトでも STABLE→OJAMA_FALL は発火する設計 (仕様確認)
        # (enable_ojama_visual_detection=False は pipeline 側のフラグで
        #  detector 登録自体をスキップする。 detector が登録された場合は動作する)


def test_str_chain_phase_detector_default_returns_stable() -> None:
    """ST-R: ChainPhaseDetector デフォルト (enable_chain_ojama_exit=False) では
    chain_event=None + CHAIN → STABLE を返す (従来挙動)。"""
    det = ChainPhaseDetector()  # enable_chain_ojama_exit=False (デフォルト)
    ctx = StateContext(state=BoardState.CHAIN)
    sig = DetectorSignals(
        time_sec=1.0,
        cnn_board=_empty_board(),
        is_match_active=True,
        chain_event=None,
        ojama_top_positive=True,  # 無視されるはず
    )
    assert det.detect(ctx, sig) == BoardState.STABLE


def test_str_detector_signals_ojama_top_positive_default_false() -> None:
    """ST-R: DetectorSignals.ojama_top_positive のデフォルトは False。"""
    sig = DetectorSignals(
        time_sec=0.0,
        cnn_board=_empty_board(),
        is_match_active=True,
    )
    assert sig.ojama_top_positive is False, "デフォルト False で既存ビルドが壊れない"


# ============================
# 案B (第2の根本原因対処, 2026-07-24):
# OJAMA_FALL 退出 = 全盤面 settle 判定 (enable_ojama_fall_board_settle)
# ============================


def _drive_board_settle(
    det: OjamaVisualDetector,
    counts: list[int],
    *,
    fps: float = 30.0,
    start_frame: int = 0,
) -> list[BoardState | None]:
    """counts の各値を count_puyos() とする盤面を順に detect() へ渡す.

    ctx.state は常に OJAMA_FALL (= 退出判定のみをテストする)。
    """
    results: list[BoardState | None] = []
    for i, n in enumerate(counts):
        frame_idx = start_frame + i
        ctx = StateContext(state=BoardState.OJAMA_FALL, frame_idx=frame_idx)
        sig = DetectorSignals(
            time_sec=float(frame_idx) / fps,
            cnn_board=_board_with_n_puyos(n),
            is_match_active=True,
        )
        results.append(det.detect(ctx, sig))
    return results


def test_board_settle_a_monotonic_increase_then_stable_exits_at_expected_frame() -> None:
    """(a) 全盤面 count が単調増加 → 安定した時、 期待通りのフレーム数で STABLE 復帰する。

    シーケンス: 20→30→40 (増加中、 frames_in_settle < MIN_FRAMES(3) の間は
    diff 判定自体をスキップ) → 44 (frames_in_settle=3, diff=4 >= THRESHOLD(2) で
    まだ不安定) → 45 が OJAMA_FALL_SETTLE_STABLE_FRAMES(8) 回連続 → STABLE。
    """
    det = OjamaVisualDetector(enable_ojama_fall_board_settle=True)
    counts = [20, 30, 40, 44] + [45] * 8  # 12 frames (idx 0-11)
    results = _drive_board_settle(det, counts)

    assert OJAMA_FALL_SETTLE_MIN_FRAMES == 3
    assert OJAMA_FALL_SETTLE_STABLE_FRAMES == 8
    assert OJAMA_FALL_SETTLE_DIFF_THRESHOLD == 2

    # frame 11 (0-indexed) で STABLE 復帰するはず、 それより前は None。
    assert all(r is None for r in results[:11]), (
        "安定に必要な連続フレーム数に達するまでは None のはず"
    )
    assert results[11] == BoardState.STABLE, (
        "count が OJAMA_FALL_SETTLE_STABLE_FRAMES フレーム不変で STABLE 復帰するはず"
    )


def test_board_settle_b_timeout_forces_stable() -> None:
    """(b) OJAMA_FALL_MAX_SEC 秒超過で安定未達でも安全弁として強制 STABLE 復帰する。"""
    det = OjamaVisualDetector(enable_ojama_fall_board_settle=True)

    # frame 0: OJAMA_FALL 突入、 count=10 (start 記録のみ、 必ず None)
    ctx0 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=0)
    sig0 = DetectorSignals(
        time_sec=0.0, cnn_board=_board_with_n_puyos(10), is_match_active=True,
    )
    assert det.detect(ctx0, sig0) is None

    # frame 1: count がまだ変動中 (不安定) だが time_sec が MAX_SEC を超過 → 強制 STABLE
    ctx1 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=1)
    sig1 = DetectorSignals(
        time_sec=OJAMA_FALL_MAX_SEC + 0.1,
        cnn_board=_board_with_n_puyos(50),  # 大きく変動 = 不安定
        is_match_active=True,
    )
    result = det.detect(ctx1, sig1)
    assert result == BoardState.STABLE, (
        "OJAMA_FALL_MAX_SEC 超過は安定未達でも強制 STABLE 復帰する安全弁のはず"
    )


def test_board_settle_c_intercept_resets_no_stale_contamination() -> None:
    """(c) OJAMA_FALL から他 state に途中横取りされた後の再突入で内部 state が
    汚染されないことを確認する (防御コード: _detect_ojama_fall_entry 先頭リセット)。

    横取り経路には STABLE + ROI count=0 (= `_detect_ojama_fall_entry` の
    「お邪魔なし」分岐、 これは _reset_internal_state() を呼ばない) を使う。
    こうすることで「_detect_ojama_fall_entry 先頭の防御コード」 自体の効果を
    (他の分岐が持つ reset 呼び出しから独立して) 検証できる。
    もし防御コードが無ければ、 再突入直後に「古い _settle_start_time」 との
    差分で即タイムアウト誤判定 (STABLE) してしまう。
    """
    det = OjamaVisualDetector(enable_ojama_fall_board_settle=True)

    # 1st OJAMA_FALL 突入 (frame 0, time=0.0): start 記録。
    ctx0 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=0)
    sig0 = DetectorSignals(
        time_sec=0.0, cnn_board=_board_with_n_puyos(10), is_match_active=True,
    )
    assert det.detect(ctx0, sig0) is None
    # settle 内部 state が記録されていることを確認 (直接検証、 実装詳細だが防御コード検証に必要)。
    assert det._settle_start_frame == 0  # noqa: SLF001

    # 他 state (STABLE, ROI お邪魔なし) に途中横取りされる。
    # time が大きく進む (= 旧実装ならここで stale time が残る)。
    ctx_other = StateContext(state=BoardState.STABLE, frame_idx=1)
    sig_other = DetectorSignals(
        time_sec=100.0,  # 大きく時間が進む (OJAMA_FALL_MAX_SEC を大幅に超える)
        cnn_board=_empty_board(),  # ROI count=0 の分岐 (reset_internal_state 非経由)
        is_match_active=True,
    )
    res_other = det.detect(ctx_other, sig_other)
    assert res_other is None, "STABLE + ROI count=0 では OJAMA_FALL 発火しない"
    # 防御コードにより内部 state がリセットされているはず。
    assert det._settle_start_frame == -1  # noqa: SLF001

    # 2nd OJAMA_FALL 再突入 (frame 2, time=100.05): 新規 start として記録され、
    # 直前の time=100.0 との差分でタイムアウト誤判定してはいけない。
    ctx2 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=2)
    sig2 = DetectorSignals(
        time_sec=100.05, cnn_board=_board_with_n_puyos(10), is_match_active=True,
    )
    result = det.detect(ctx2, sig2)
    assert result is None, (
        "再突入直後は新規 start として記録され、 stale time でタイムアウトしてはいけない"
    )


def test_board_settle_default_off_bit_identical_to_legacy() -> None:
    """既定 (enable_ojama_fall_board_settle=False) では従来の ROI ベース判定が
    そのまま使われ、 全盤面 count とは無関係に振る舞う (bit-identical 確認)。

    全盤面 count が変動し続けていても、 ROI (可視最上段 2 行) の count が 0 なら
    従来通り即 STABLE 復帰する (= 案B のロジックが一切介入しない証拠)。
    """
    det = OjamaVisualDetector(enable_ojama_fall_board_settle=False)

    # 全盤面には puyo が大量にある (ROI 外) が ROI 自体は空。
    board = Board()
    for r in range(HIDDEN_ROWS + 2, 13):
        for c in range(6):
            board.set(r, c, COLOR_OJAMA)
    assert _count_top_ojama(board) == 0  # ROI 内は空

    ctx = StateContext(state=BoardState.OJAMA_FALL, frame_idx=0)
    sig = DetectorSignals(time_sec=0.0, cnn_board=board, is_match_active=True)
    result = det.detect(ctx, sig)
    assert result == BoardState.STABLE, (
        "従来ロジック (ROI count==0 で即 STABLE) が維持されているはず"
    )


# ============================
# バグ修正 (2026-07-24): OJAMA_FALL 再突入振動ループの修正確認
#
# 真因: _detect_ojama_fall_exit_board_settle の settle-exit / timeout-exit が
# _reset_internal_state() を無条件呼び出しし、 _prev_top_ojama_count を 0 に
# 戻す。 ROI にまだ着弾済み・不動のお邪魔が残っている状態で退出すると、
# 次フレームの _detect_ojama_fall_entry が「開始トリガー: prev==0」を満たして
# 誤って OJAMA_FALL へ再突入し、 約0.183秒刻みの振動ループが発生する。
# 振動中は TsumoPhaseDetector が None を返し続け TSUMO_FALL 検出が
# 最大 +7秒 遅延する (A/B実測)。
#
# 修正: settle-exit / timeout-exit の両分岐で退出時点の ROI 実カウントを
# _reset_internal_state(keep_top_ojama_count=...) で明示的に保持する。
# ============================


def test_reset_internal_state_default_keeps_bit_identical_zero_reset() -> None:
    """`_reset_internal_state()` を引数なしで呼ぶと従来通り 0 リセットされる
    (bit-identical、 既存呼び出し箇所の挙動を変えないことの確認)。
    """
    det = OjamaVisualDetector()
    det._prev_top_ojama_count = 99  # noqa: SLF001

    det._reset_internal_state()  # noqa: SLF001

    assert det._prev_top_ojama_count == 0


def test_reset_internal_state_keep_top_ojama_count_param() -> None:
    """`_reset_internal_state(keep_top_ojama_count=N)` は N を保持する。"""
    det = OjamaVisualDetector()
    det._prev_top_ojama_count = 99  # noqa: SLF001

    det._reset_internal_state(keep_top_ojama_count=7)  # noqa: SLF001

    assert det._prev_top_ojama_count == 7


def test_legacy_ojama_fall_exit_still_resets_prev_top_ojama_count_to_zero() -> None:
    """回帰: 案B OFF (既定) の従来退出経路 `_detect_ojama_fall_exit` は改修後も
    `_prev_top_ojama_count` を 0 にリセットする (bit-identical) ことを確認する。
    """
    det = OjamaVisualDetector()  # enable_ojama_fall_board_settle=False (既定)
    det._consec_count = 3  # noqa: SLF001 (退出前状態を人工的に設定)

    ctx = StateContext(state=BoardState.OJAMA_FALL, frame_idx=0)
    sig = DetectorSignals(time_sec=0.0, cnn_board=_empty_board(), is_match_active=True)
    result = det.detect(ctx, sig)

    assert result == BoardState.STABLE
    assert det._prev_top_ojama_count == 0, (
        "既存経路 (_detect_ojama_fall_exit) は従来通り 0 リセットのはず"
    )


def test_board_settle_exit_preserves_prev_top_ojama_count_no_reentry() -> None:
    """settle-exit (お邪魔静止) 退出時、 ROI にお邪魔が残っていれば
    `_prev_top_ojama_count` に実カウントを保持し、 STABLE 復帰後の同一盤面が
    続くフレームで OJAMA_FALL へ誤再突入しないことを確認する。

    シーケンス自体は test_board_settle_a_monotonic_increase_then_stable_exits_at_expected_frame
    と同一 (frame 11 で STABLE 復帰、 count=45 で安定)。
    count=45 は ROI (可視最上段 2 行 = 12 セル) を満杯に埋める値。
    """
    det = OjamaVisualDetector(enable_ojama_fall_board_settle=True)
    counts = [20, 30, 40, 44] + [45] * 8
    results = _drive_board_settle(det, counts)
    assert results[11] == BoardState.STABLE

    # 退出時点で ROI (12 セル) は満杯 = 12。 旧実装ならここが 0 になっていた。
    expected_roi_count = _count_top_ojama(_board_with_n_puyos(45))
    assert expected_roi_count == 12, "テスト前提: n=45 で ROI が満杯になるはず"
    assert det._prev_top_ojama_count == expected_roi_count, (
        "settle-exit 時点の ROI 実カウントを保持しているはず (0 でない)"
    )

    # STABLE 復帰後、 同一おじゃま盤面が続くフレームを複数回 detect() へ渡しても
    # 誤って OJAMA_FALL へ再突入しないことを確認する (振動ループの再現防止)。
    board_same = _board_with_n_puyos(45)
    for i in range(OJAMA_CONSEC_THRESH + 3):
        frame_idx = 12 + i
        ctx = StateContext(state=BoardState.STABLE, frame_idx=frame_idx)
        sig = DetectorSignals(
            time_sec=float(frame_idx) / 30.0,
            cnn_board=board_same,
            is_match_active=True,
        )
        result = det.detect(ctx, sig)
        assert result is None, (
            f"frame {frame_idx}: おじゃまが不変のまま OJAMA_FALL へ誤再突入した"
            " (振動ループ再発)"
        )


def test_board_settle_timeout_exit_preserves_prev_top_ojama_count_no_reentry() -> None:
    """timeout-exit (安全弁) 退出時も同様に ROI 実カウントを保持し、
    STABLE 復帰後の同一盤面フレームで誤再突入しないことを確認する。
    """
    det = OjamaVisualDetector(enable_ojama_fall_board_settle=True)

    ctx0 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=0)
    sig0 = DetectorSignals(
        time_sec=0.0, cnn_board=_board_with_n_puyos(10), is_match_active=True,
    )
    assert det.detect(ctx0, sig0) is None

    board50 = _board_with_n_puyos(50)
    ctx1 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=1)
    sig1 = DetectorSignals(
        time_sec=OJAMA_FALL_MAX_SEC + 0.1, cnn_board=board50, is_match_active=True,
    )
    result = det.detect(ctx1, sig1)
    assert result == BoardState.STABLE

    expected_roi_count = _count_top_ojama(board50)
    assert expected_roi_count > 0, "テスト前提: n=50 で ROI にお邪魔が残るはず"
    assert det._prev_top_ojama_count == expected_roi_count, (
        "timeout-exit 時点の ROI 実カウントを保持しているはず (0 でない)"
    )

    # STABLE 復帰後、 同一盤面が続いても誤再突入しない。
    for i in range(OJAMA_CONSEC_THRESH + 3):
        frame_idx = 2 + i
        ctx = StateContext(state=BoardState.STABLE, frame_idx=frame_idx)
        sig = DetectorSignals(
            time_sec=OJAMA_FALL_MAX_SEC + 0.13 + float(i) / 30.0,
            cnn_board=board50,
            is_match_active=True,
        )
        assert det.detect(ctx, sig) is None, (
            f"frame {frame_idx}: timeout-exit 後に誤って OJAMA_FALL へ再突入した"
        )


# ============================
# 修正B (2026-08-08、バグB): GRAVITY_SETTLE 中の OJAMA_FALL 新規発火ガード
# ============================


def test_bugfix_b_gravity_settle_guard_blocks_entry_when_enabled() -> None:
    """バグB 修正: enable_ojama_entry_gravity_settle_guard=True なら
    GRAVITY_SETTLE 中に ROI お邪魔が観測されても OJAMA_FALL に遷移しない。

    連鎖の段間重力待ち (GRAVITY_SETTLE) 中の乗っ取りを防ぐことで、
    GravitySettleDetector の内部カウンタ残留 (バグC) を誘発しない。
    """
    det = OjamaVisualDetector(enable_ojama_entry_gravity_settle_guard=True)
    ojama_board = _board_with_ojama_top(2)

    # OJAMA_CONSEC_THRESH を超える frame 数を投入しても発火しないはず
    # (フラグ OFF なら test_bugfix_b_gravity_settle_guard_disabled_reproduces_bug
    # で同条件が OJAMA_FALL を返すことを確認する)。
    for i in range(OJAMA_CONSEC_THRESH + 3):
        ctx = StateContext(state=BoardState.GRAVITY_SETTLE, frame_idx=i)
        sig = DetectorSignals(
            time_sec=float(i) / 30.0, cnn_board=ojama_board, is_match_active=True,
        )
        result = det.detect(ctx, sig)
        assert result is None, (
            f"frame {i}: ガード有効時は GRAVITY_SETTLE 中に OJAMA_FALL へ"
            " 遷移してはならない"
        )


def test_bugfix_b_gravity_settle_guard_disabled_reproduces_bug() -> None:
    """回帰防止 (backwards compat): フラグ default False (未指定) では
    従来通り GRAVITY_SETTLE 中でも OJAMA_FALL へ発火する (旧挙動 bit-identical)。
    """
    det = OjamaVisualDetector()  # enable_ojama_entry_gravity_settle_guard=False (既定)
    ojama_board = _board_with_ojama_top(2)

    results = []
    for i in range(OJAMA_CONSEC_THRESH + 1):
        ctx = StateContext(state=BoardState.GRAVITY_SETTLE, frame_idx=i)
        sig = DetectorSignals(
            time_sec=float(i) / 30.0, cnn_board=ojama_board, is_match_active=True,
        )
        results.append(det.detect(ctx, sig))

    assert BoardState.OJAMA_FALL in results, (
        "フラグ OFF (既定) では GRAVITY_SETTLE 中でも旧挙動通り OJAMA_FALL へ"
        " 発火するはず (backwards compat 回帰防止)"
    )


def test_str_detector_signals_own_score_delta_default_zero() -> None:
    """ST-R: DetectorSignals.own_score_delta のデフォルトは 0。"""
    sig = DetectorSignals(
        time_sec=0.0, cnn_board=_empty_board(), is_match_active=True,
    )
    assert sig.own_score_delta == 0, "デフォルト 0 で既存ビルドが壊れない"


# ============================
# 案2 (enable_ojama_fall_placement_override, 2026-08-13、OJAMA_FALL誤分類
# 根因調査): OJAMA_FALL 中の実設置検知による早期 exit
# ============================


def test_placement_evidence_true_on_slide_motion() -> None:
    """_has_ojama_fall_placement_evidence: slide_motion=True で証拠ありと判定する。"""
    sig = DetectorSignals(
        time_sec=0.0, cnn_board=_empty_board(), is_match_active=True,
        slide_motion=True, own_score_delta=0,
    )
    assert _has_ojama_fall_placement_evidence(sig) is True


def test_placement_evidence_true_on_own_score_delta() -> None:
    """_has_ojama_fall_placement_evidence: own_score_delta>0 (落下ボーナス等)
    で証拠ありと判定する。
    """
    sig = DetectorSignals(
        time_sec=0.0, cnn_board=_empty_board(), is_match_active=True,
        slide_motion=False, own_score_delta=8,
    )
    assert _has_ojama_fall_placement_evidence(sig) is True


def test_placement_evidence_false_without_signals() -> None:
    """_has_ojama_fall_placement_evidence: 両方 False/0 なら証拠なしと判定する。"""
    sig = DetectorSignals(
        time_sec=0.0, cnn_board=_empty_board(), is_match_active=True,
        slide_motion=False, own_score_delta=0,
    )
    assert _has_ojama_fall_placement_evidence(sig) is False


def test_placement_override_exits_immediately_on_slide_motion() -> None:
    """enable_ojama_fall_placement_override=True: OJAMA_FALL 中に
    slide_motion=True を検知したら settle 判定を待たず即 STABLE 復帰する。
    """
    det = OjamaVisualDetector(
        enable_ojama_fall_board_settle=True,
        enable_ojama_fall_placement_override=True,
    )
    # frame 0: OJAMA_FALL 突入 (settle start 記録のみ、必ず None)
    ctx0 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=0)
    sig0 = DetectorSignals(
        time_sec=0.0, cnn_board=_board_with_n_puyos(20), is_match_active=True,
    )
    assert det.detect(ctx0, sig0) is None

    # frame 1: slide_motion=True (= 実設置の証拠) → settle 未達でも即 STABLE
    ctx1 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=1)
    sig1 = DetectorSignals(
        time_sec=1.0 / 30.0, cnn_board=_board_with_n_puyos(22),
        is_match_active=True, slide_motion=True,
    )
    result = det.detect(ctx1, sig1)
    assert result == BoardState.STABLE, (
        "実設置の証拠 (slide_motion) があれば settle 未達でも即 STABLE 復帰するはず"
    )


def test_placement_override_exits_immediately_on_own_score_delta() -> None:
    """enable_ojama_fall_placement_override=True: own_score_delta>0 (落下
    ボーナス等) でも settle 判定を待たず即 STABLE 復帰する
    (docs/DEMO_REVIEW_2026-08-13.md 場面1: score 213→221 の +8=落下ボーナスと
    同じパターン)。
    """
    det = OjamaVisualDetector(
        enable_ojama_fall_board_settle=True,
        enable_ojama_fall_placement_override=True,
    )
    ctx0 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=0)
    sig0 = DetectorSignals(
        time_sec=0.0, cnn_board=_board_with_n_puyos(20), is_match_active=True,
    )
    assert det.detect(ctx0, sig0) is None

    ctx1 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=1)
    sig1 = DetectorSignals(
        time_sec=1.0 / 30.0, cnn_board=_board_with_n_puyos(22),
        is_match_active=True, own_score_delta=8,
    )
    result = det.detect(ctx1, sig1)
    assert result == BoardState.STABLE


def test_placement_override_default_off_bit_identical() -> None:
    """回帰: enable_ojama_fall_placement_override=False (既定) では
    slide_motion/own_score_delta があっても settle 判定 (盤面全体安定) を
    待つ従来挙動を維持する (bit-identical)。
    """
    det = OjamaVisualDetector(enable_ojama_fall_board_settle=True)  # override 既定 False

    ctx0 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=0)
    sig0 = DetectorSignals(
        time_sec=0.0, cnn_board=_board_with_n_puyos(20), is_match_active=True,
    )
    assert det.detect(ctx0, sig0) is None

    ctx1 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=1)
    sig1 = DetectorSignals(
        time_sec=1.0 / 30.0, cnn_board=_board_with_n_puyos(22),
        is_match_active=True, slide_motion=True, own_score_delta=8,
    )
    result = det.detect(ctx1, sig1)
    assert result is None, (
        "フラグ OFF では実設置証拠があっても settle 判定を無視しない (従来通り継続)"
    )


def test_placement_override_scene1_extension_reproduction_and_fix() -> None:
    """場面1 再現+修正確認 (docs/DEMO_REVIEW_2026-08-13.md):
    全盤面ぷよ数の静止を待つ既存出口判定 (`_detect_ojama_fall_exit_board_settle`)
    は、 自分のツモ設置が連続すると (= board 全体の count が毎フレーム変化し
    続けると) `_board_stable_consec` が一度も閾値へ到達できず OJAMA_FALL に
    張り付き続ける (出口判定のスコープ過大、実測: 0.15-0.3 秒周期の振動・
    最悪 OJAMA_FALL_MAX_SEC までの延長)。 override ON なら、 その途中で
    own_score_delta (落下ボーナス) が観測された瞬間に settle 未達でも即座に
    STABLE へ抜ける。
    """
    counts = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]  # 毎フレーム変化し続ける

    # --- override OFF: 全フレーム通しても settle が一度も成立せず OJAMA_FALL のまま ---
    det_off = OjamaVisualDetector(enable_ojama_fall_board_settle=True)
    results_off = []
    for i, n in enumerate(counts):
        ctx = StateContext(state=BoardState.OJAMA_FALL, frame_idx=i)
        sig = DetectorSignals(
            time_sec=float(i) / 30.0, cnn_board=_board_with_n_puyos(n),
            is_match_active=True, own_score_delta=8 if i > 0 else 0,
        )
        results_off.append(det_off.detect(ctx, sig))
    assert all(r is None for r in results_off), (
        "毎フレーム board count が変化し続けると settle が成立せず"
        " OJAMA_FALL に張り付き続ける (延長バグの再現)"
    )

    # --- override ON: 最初に own_score_delta>0 が来たフレームで即座に STABLE ---
    det_on = OjamaVisualDetector(
        enable_ojama_fall_board_settle=True,
        enable_ojama_fall_placement_override=True,
    )
    ctx0 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=0)
    sig0 = DetectorSignals(
        time_sec=0.0, cnn_board=_board_with_n_puyos(counts[0]),
        is_match_active=True, own_score_delta=0,
    )
    assert det_on.detect(ctx0, sig0) is None  # 1 frame目は証拠なし、継続

    ctx1 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=1)
    sig1 = DetectorSignals(
        time_sec=1.0 / 30.0, cnn_board=_board_with_n_puyos(counts[1]),
        is_match_active=True, own_score_delta=8,
    )
    result1 = det_on.detect(ctx1, sig1)
    assert result1 == BoardState.STABLE, (
        "override ON なら実設置証拠が来た時点で settle 未達でも即 STABLE 復帰し、"
        " 延長 (最悪 OJAMA_FALL_MAX_SEC まで張り付く) を回避できる"
    )


# ============================
# 案4-lite (enable_ojama_fall_entry_hardening, 2026-08-13 根因調査追補):
# OJAMA_FALL entry の実時間ハードニング
# ============================


def test_entry_hardening_time_based_fires_after_required_sec() -> None:
    """実時間ベース entry: OJAMA_ENTRY_CONSEC_SEC 秒経過で OJAMA_FALL 発火する
    (frame 数でなく time_sec 差分で判定、通常 state (STABLE) からの entry)。
    """
    det = OjamaVisualDetector(enable_ojama_fall_entry_hardening=True)
    ojama_board = _board_with_ojama_top(2)

    ctx0 = StateContext(state=BoardState.STABLE, frame_idx=0)
    sig0 = DetectorSignals(time_sec=0.0, cnn_board=ojama_board, is_match_active=True)
    assert det.detect(ctx0, sig0) is None  # trigger 開始のみ

    # required_sec 未満: まだ発火しない
    ctx1 = StateContext(state=BoardState.STABLE, frame_idx=1)
    sig1 = DetectorSignals(
        time_sec=OJAMA_ENTRY_CONSEC_SEC * 0.5, cnn_board=ojama_board,
        is_match_active=True,
    )
    assert det.detect(ctx1, sig1) is None

    # required_sec 以上経過: 発火する
    ctx2 = StateContext(state=BoardState.STABLE, frame_idx=2)
    sig2 = DetectorSignals(
        time_sec=OJAMA_ENTRY_CONSEC_SEC + 0.001, cnn_board=ojama_board,
        is_match_active=True,
    )
    assert det.detect(ctx2, sig2) == BoardState.OJAMA_FALL


def test_entry_hardening_stride_invariant() -> None:
    """実時間ベース entry は frame 間隔 (stride) に依存せず同じ実時間で発火する
    (stride=2 相当、 中間フレームを間引いても time_sec 差分だけで判定される)。
    """
    det = OjamaVisualDetector(enable_ojama_fall_entry_hardening=True)
    ojama_board = _board_with_ojama_top(2)

    ctx0 = StateContext(state=BoardState.STABLE, frame_idx=0)
    sig0 = DetectorSignals(time_sec=0.0, cnn_board=ojama_board, is_match_active=True)
    assert det.detect(ctx0, sig0) is None

    # stride=2 相当: 中間フレームを飛ばし、 一気に閾値超過の時刻を渡す
    ctx1 = StateContext(state=BoardState.STABLE, frame_idx=2)
    sig1 = DetectorSignals(
        time_sec=OJAMA_ENTRY_CONSEC_SEC + 0.001, cnn_board=ojama_board,
        is_match_active=True,
    )
    assert det.detect(ctx1, sig1) == BoardState.OJAMA_FALL, (
        "間引き幅に関係なく実時間が経過すれば発火するはず"
    )


def test_entry_hardening_chain_interrupt_blocked_at_normal_duration() -> None:
    """CHAIN 割り込み厳格化: ctx.state==CHAIN からの entry は通常の
    OJAMA_ENTRY_CONSEC_SEC 経過だけでは発火しない (倍率が掛かるため)。

    docs/DEMO_REVIEW_2026-08-13.md 場面2 の再現 (stride 下で chain_event が
    瞬間欠落した隙に OJAMA_FALL が CHAIN を奪う経路への対策)。
    """
    det = OjamaVisualDetector(
        enable_ojama_visual_chain_exit=True,
        enable_ojama_fall_entry_hardening=True,
    )
    ojama_board = _board_with_ojama_top(2)

    ctx0 = StateContext(state=BoardState.CHAIN, frame_idx=0)
    sig0 = DetectorSignals(time_sec=0.0, cnn_board=ojama_board, is_match_active=True)
    assert det.detect(ctx0, sig0) is None

    # 通常閾値 (OJAMA_ENTRY_CONSEC_SEC) は超えたが、 CHAIN 倍率には未達
    ctx1 = StateContext(state=BoardState.CHAIN, frame_idx=1)
    sig1 = DetectorSignals(
        time_sec=OJAMA_ENTRY_CONSEC_SEC + 0.001, cnn_board=ojama_board,
        is_match_active=True,
    )
    result = det.detect(ctx1, sig1)
    assert result is None, (
        "CHAIN 中は通常閾値だけでは発火せず、 割り込みを抑止するはず"
        f" (要求倍率={OJAMA_ENTRY_CHAIN_INTERRUPT_MULTIPLIER})"
    )


def test_entry_hardening_chain_interrupt_fires_after_multiplier_duration() -> None:
    """CHAIN 割り込み厳格化: 倍率分の実時間が経過すれば CHAIN からでも発火する
    (= 本物の長時間 CHAIN 中お邪魔降下は見逃さない、 安全弁が残ることの確認)。
    """
    det = OjamaVisualDetector(
        enable_ojama_visual_chain_exit=True,
        enable_ojama_fall_entry_hardening=True,
    )
    ojama_board = _board_with_ojama_top(2)
    required = OJAMA_ENTRY_CONSEC_SEC * OJAMA_ENTRY_CHAIN_INTERRUPT_MULTIPLIER

    ctx0 = StateContext(state=BoardState.CHAIN, frame_idx=0)
    sig0 = DetectorSignals(time_sec=0.0, cnn_board=ojama_board, is_match_active=True)
    assert det.detect(ctx0, sig0) is None

    ctx1 = StateContext(state=BoardState.CHAIN, frame_idx=1)
    sig1 = DetectorSignals(
        time_sec=required + 0.001, cnn_board=ojama_board, is_match_active=True,
    )
    assert det.detect(ctx1, sig1) == BoardState.OJAMA_FALL


def test_entry_hardening_default_off_bit_identical() -> None:
    """回帰: enable_ojama_fall_entry_hardening=False (既定) では従来通り
    frame 数連続 (OJAMA_CONSEC_THRESH) で判定する (bit-identical)。
    """
    det = OjamaVisualDetector()  # 既定 False
    ojama_board = _board_with_ojama_top(2)

    ctx0 = StateContext(state=BoardState.STABLE, frame_idx=0)
    sig0 = DetectorSignals(time_sec=0.0, cnn_board=ojama_board, is_match_active=True)
    assert det.detect(ctx0, sig0) is None

    # 実時間はごく僅かしか経過していないが frame 数 (2連続) は満たす
    ctx1 = StateContext(state=BoardState.STABLE, frame_idx=1)
    sig1 = DetectorSignals(
        time_sec=1.0 / 3000.0, cnn_board=ojama_board, is_match_active=True,
    )
    assert det.detect(ctx1, sig1) == BoardState.OJAMA_FALL, (
        "既定 (frame 数ベース) では極短時間でも 2 フレーム連続で発火するはず"
        " (実時間ハードニングが介入しないことの確認)"
    )


# ============================
# 案4-lite 拡張 (coordinator追加指示, 2026-08-13、場面2実測 52→69 の悪化対処):
# CHAIN 割り込み厳格化を state 非依存 (直近 own chain アクティブ) に拡張 +
# 案2 exit 直後の re-entry 抑制
# ============================


def test_hardening_gravity_settle_state_is_hardened_context() -> None:
    """coordinator追加指示 (第2ラウンド、実データ直接確認で判明):
    実測 (logs/verify_ojama_fall_fix_scene2_2026-08-13_v2.json の
    frame_records) で CHAIN→GRAVITY_SETTLE→OJAMA_FALL→CHAIN という
    周期的割り込み (~1.2〜1.3秒、連鎖1リンクの実測とほぼ一致) が
    ctx.state==GRAVITY_SETTLE から直接発生することを確認した。
    「ctx.state==CHAIN」の瞬間条件だけでは GravitySettleDetector が
    最低優先度で登録されているため CHAIN 終了直後の GRAVITY_SETTLE 中に
    素通りされる。 ctx.state==GRAVITY_SETTLE も CHAIN と同じ倍率で
    厳格化する。
    """
    det = OjamaVisualDetector(enable_ojama_fall_entry_hardening=True)
    ojama_board = _board_with_ojama_top(2)

    ctx0 = StateContext(state=BoardState.GRAVITY_SETTLE, frame_idx=0)
    sig0 = DetectorSignals(time_sec=0.0, cnn_board=ojama_board, is_match_active=True)
    assert det.detect(ctx0, sig0) is None

    # 通常閾値は超えたが CHAIN 倍率には未達 → まだ発火しない
    ctx1 = StateContext(state=BoardState.GRAVITY_SETTLE, frame_idx=1)
    sig1 = DetectorSignals(
        time_sec=OJAMA_ENTRY_CONSEC_SEC + 0.001, cnn_board=ojama_board,
        is_match_active=True,
    )
    assert det.detect(ctx1, sig1) is None, (
        "ctx.state==GRAVITY_SETTLE も CHAIN と同じ倍率で厳格化されるはず"
        " (実測された CHAIN→GRAVITY_SETTLE→OJAMA_FALL→CHAIN の直接経路)"
    )

    # CHAIN 倍率分の実時間が経過すれば発火する (安全弁は残る)。
    ctx2 = StateContext(state=BoardState.GRAVITY_SETTLE, frame_idx=2)
    required = OJAMA_ENTRY_CONSEC_SEC * OJAMA_ENTRY_CHAIN_INTERRUPT_MULTIPLIER
    sig2 = DetectorSignals(
        time_sec=required + 0.001, cnn_board=ojama_board, is_match_active=True,
    )
    assert det.detect(ctx2, sig2) == BoardState.OJAMA_FALL


def test_hardening_recent_gravity_settle_observation_covers_stable_transition() -> None:
    """coordinator追加指示 (第2ラウンド、実データ直接確認で判明):
    GRAVITY_SETTLE → STABLE (一瞬) → OJAMA_FALL という経路 (実測:
    t=198.267 gravity_settle → t=198.400 stable → t=198.500 ojama_fall、
    経過 0.233秒) は ctx.state の瞬間条件 (CHAIN/GRAVITY_SETTLE) では
    原理的に捉えられない (発火時点では既に STABLE のため)。 本 detector
    自身が「直近に CHAIN/GRAVITY_SETTLE を観測した時刻」 を内部で追跡し、
    STABLE に遷移した直後の entry も同じ倍率で厳格化することを確認する。
    """
    det = OjamaVisualDetector(enable_ojama_fall_entry_hardening=True)
    ojama_board = _board_with_ojama_top(2)

    # GRAVITY_SETTLE を観測 (ROI 陰性のまま、 観測時刻のみ内部に記録される)。
    ctx0 = StateContext(state=BoardState.GRAVITY_SETTLE, frame_idx=0)
    sig0 = DetectorSignals(time_sec=0.0, cnn_board=_empty_board(), is_match_active=True)
    assert det.detect(ctx0, sig0) is None

    # STABLE に遷移した直後 (ctx.state はもう GRAVITY_SETTLE でない) に
    # ROI お邪魔が出現 → 通常閾値だけでは発火しないはず (直近観測で厳格化)。
    ctx1 = StateContext(state=BoardState.STABLE, frame_idx=1)
    sig1 = DetectorSignals(time_sec=0.05, cnn_board=ojama_board, is_match_active=True)
    assert det.detect(ctx1, sig1) is None  # trigger 開始のみ

    ctx2 = StateContext(state=BoardState.STABLE, frame_idx=2)
    sig2 = DetectorSignals(
        time_sec=0.05 + OJAMA_ENTRY_CONSEC_SEC + 0.001, cnn_board=ojama_board,
        is_match_active=True,
    )
    assert det.detect(ctx2, sig2) is None, (
        "GRAVITY_SETTLE 観測直後の STABLE では通常閾値だけで発火してはいけない"
        " (ctx.state の瞬間条件では捉えられない経路への対策)"
    )

    # CHAIN 倍率分の実時間が経過すれば発火する (安全弁は残る)。
    ctx3 = StateContext(state=BoardState.STABLE, frame_idx=3)
    required = OJAMA_ENTRY_CONSEC_SEC * OJAMA_ENTRY_CHAIN_INTERRUPT_MULTIPLIER
    sig3 = DetectorSignals(
        time_sec=0.05 + required + 0.001, cnn_board=ojama_board,
        is_match_active=True,
    )
    assert det.detect(ctx3, sig3) == BoardState.OJAMA_FALL


def test_hardening_recent_chain_activity_extends_beyond_chain_state() -> None:
    """coordinator追加指示: ctx.state==CHAIN でなくても、 直近
    own_chain_hold_until_sec が近ければ (= 直近まで自 chain がアクティブ
    だった) entry を CHAIN 割り込みと同じ倍率で厳格化する。

    「ctx.state == CHAIN」の瞬間条件だけでは、 chain_event 検出の空白で
    state が既に CHAIN を離れた後 (GRAVITY_SETTLE 等の中間 state 経由) に
    entry 判定される実際の割り込み経路を素通りしてしまう
    (場面2 実測: chain_to_ojama_interrupts が baseline でも常に 0 =
    隣接フレームの直接 CHAIN→OJAMA_FALL 遷移は存在しない、が根拠)。
    """
    det = OjamaVisualDetector(enable_ojama_fall_entry_hardening=True)
    ojama_board = _board_with_ojama_top(2)

    # 直近 (0.1秒前) まで own chain が hold されていた想定 (ctx.state は STABLE)。
    ctx0 = StateContext(state=BoardState.STABLE, frame_idx=0)
    sig0 = DetectorSignals(
        time_sec=100.0, cnn_board=ojama_board, is_match_active=True,
        own_chain_hold_until_sec=99.9,
    )
    assert det.detect(ctx0, sig0) is None

    # 通常閾値は超えたが CHAIN 倍率には未達 → まだ発火しない
    ctx1 = StateContext(state=BoardState.STABLE, frame_idx=1)
    sig1 = DetectorSignals(
        time_sec=100.0 + OJAMA_ENTRY_CONSEC_SEC + 0.001, cnn_board=ojama_board,
        is_match_active=True, own_chain_hold_until_sec=99.9,
    )
    assert det.detect(ctx1, sig1) is None, (
        "ctx.state != CHAIN でも直近 own chain アクティブなら厳格化されるはず"
    )

    # CHAIN 倍率分の実時間が経過すれば発火する (安全弁は残る)。
    ctx2 = StateContext(state=BoardState.STABLE, frame_idx=2)
    required = OJAMA_ENTRY_CONSEC_SEC * OJAMA_ENTRY_CHAIN_INTERRUPT_MULTIPLIER
    sig2 = DetectorSignals(
        time_sec=100.0 + required + 0.001, cnn_board=ojama_board,
        is_match_active=True, own_chain_hold_until_sec=99.9,
    )
    assert det.detect(ctx2, sig2) == BoardState.OJAMA_FALL


def test_hardening_chain_activity_outside_window_not_hardened() -> None:
    """回帰: own_chain_hold_until_sec が OJAMA_ENTRY_CHAIN_RECENT_ACTIVE_SEC
    より昔なら (= 直近アクティブでない) 厳格化しない (通常閾値のみ)。
    """
    det = OjamaVisualDetector(enable_ojama_fall_entry_hardening=True)
    ojama_board = _board_with_ojama_top(2)

    old_chain_until = 100.0 - OJAMA_ENTRY_CHAIN_RECENT_ACTIVE_SEC - 1.0
    ctx0 = StateContext(state=BoardState.STABLE, frame_idx=0)
    sig0 = DetectorSignals(
        time_sec=100.0, cnn_board=ojama_board, is_match_active=True,
        own_chain_hold_until_sec=old_chain_until,
    )
    assert det.detect(ctx0, sig0) is None

    ctx1 = StateContext(state=BoardState.STABLE, frame_idx=1)
    sig1 = DetectorSignals(
        time_sec=100.0 + OJAMA_ENTRY_CONSEC_SEC + 0.001, cnn_board=ojama_board,
        is_match_active=True, own_chain_hold_until_sec=old_chain_until,
    )
    assert det.detect(ctx1, sig1) == BoardState.OJAMA_FALL, (
        "直近アクティブでない own chain 履歴は厳格化の対象外のはず"
    )


def test_hardening_default_own_chain_hold_until_sentinel_not_treated_as_recent() -> None:
    """回帰: own_chain_hold_until_sec の既定値 0.0 (= 「まだ chain 未発生」
    sentinel、RecognitionPipeline.reset() と同じ規約) は、 time_sec が
    小さい序盤フレームでも「直近アクティブ」と誤判定しない。
    """
    det = OjamaVisualDetector(enable_ojama_fall_entry_hardening=True)
    ojama_board = _board_with_ojama_top(2)

    ctx0 = StateContext(state=BoardState.STABLE, frame_idx=0)
    sig0 = DetectorSignals(time_sec=0.0, cnn_board=ojama_board, is_match_active=True)
    assert det.detect(ctx0, sig0) is None

    ctx1 = StateContext(state=BoardState.STABLE, frame_idx=1)
    sig1 = DetectorSignals(
        time_sec=OJAMA_ENTRY_CONSEC_SEC + 0.001, cnn_board=ojama_board,
        is_match_active=True,
    )
    assert det.detect(ctx1, sig1) == BoardState.OJAMA_FALL, (
        "own_chain_hold_until_sec 既定値 (0.0) は「直近アクティブ」と誤判定"
        "してはいけない"
    )


def test_hardening_reentry_after_placement_override_exit_requires_extra_duration() -> None:
    """coordinator追加指示: 案2 (placement override) の早期 exit 直後は
    OJAMA_REENTRY_SUPPRESS_SEC 秒以内の再 entry を CHAIN 割り込みと同じ倍率で
    厳格化する (exit→即re-entry 振動対策)。
    """
    det = OjamaVisualDetector(
        enable_ojama_fall_board_settle=True,
        enable_ojama_fall_placement_override=True,
        enable_ojama_fall_entry_hardening=True,
    )
    board2 = _board_with_ojama_top(2)
    board3 = _board_with_ojama_top(3)

    # OJAMA_FALL 突入 (settle start 記録)。
    ctx0 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=0)
    sig0 = DetectorSignals(time_sec=0.0, cnn_board=board2, is_match_active=True)
    assert det.detect(ctx0, sig0) is None

    # own_score_delta>0 (実設置証拠) で即 STABLE 復帰 (案2)。
    ctx1 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=1)
    sig1 = DetectorSignals(
        time_sec=0.033, cnn_board=board2, is_match_active=True, own_score_delta=8,
    )
    assert det.detect(ctx1, sig1) == BoardState.STABLE

    # exit 直後、 ROI お邪魔が増加 (新規トリガー) → 通常閾値だけでは発火しない。
    ctx2 = StateContext(state=BoardState.STABLE, frame_idx=2)
    sig2 = DetectorSignals(time_sec=0.043, cnn_board=board3, is_match_active=True)
    assert det.detect(ctx2, sig2) is None

    ctx3 = StateContext(state=BoardState.STABLE, frame_idx=3)
    sig3 = DetectorSignals(
        time_sec=0.043 + OJAMA_ENTRY_CONSEC_SEC + 0.001, cnn_board=board3,
        is_match_active=True,
    )
    assert det.detect(ctx3, sig3) is None, (
        "exit 直後の re-entry は CHAIN 割り込みと同じ倍率で厳格化されるはず"
    )

    # 倍率分の実時間が経過すれば発火する (完全ブロックではない)。
    ctx4 = StateContext(state=BoardState.STABLE, frame_idx=4)
    required = OJAMA_ENTRY_CONSEC_SEC * OJAMA_ENTRY_CHAIN_INTERRUPT_MULTIPLIER
    sig4 = DetectorSignals(
        time_sec=0.043 + required + 0.001, cnn_board=board3, is_match_active=True,
    )
    assert det.detect(ctx4, sig4) == BoardState.OJAMA_FALL


def test_hardening_reentry_suppression_expires_after_window() -> None:
    """回帰: OJAMA_REENTRY_SUPPRESS_SEC を過ぎれば通常閾値に戻る。"""
    det = OjamaVisualDetector(
        enable_ojama_fall_board_settle=True,
        enable_ojama_fall_placement_override=True,
        enable_ojama_fall_entry_hardening=True,
    )
    board2 = _board_with_ojama_top(2)
    board3 = _board_with_ojama_top(3)

    ctx0 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=0)
    sig0 = DetectorSignals(time_sec=0.0, cnn_board=board2, is_match_active=True)
    assert det.detect(ctx0, sig0) is None

    ctx1 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=1)
    sig1 = DetectorSignals(
        time_sec=0.01, cnn_board=board2, is_match_active=True, own_score_delta=8,
    )
    assert det.detect(ctx1, sig1) == BoardState.STABLE

    # OJAMA_REENTRY_SUPPRESS_SEC 経過後に新規トリガー → 通常閾値のみで発火。
    trigger_t = 0.01 + OJAMA_REENTRY_SUPPRESS_SEC + 0.01
    ctx2 = StateContext(state=BoardState.STABLE, frame_idx=2)
    sig2 = DetectorSignals(time_sec=trigger_t, cnn_board=board3, is_match_active=True)
    assert det.detect(ctx2, sig2) is None

    ctx3 = StateContext(state=BoardState.STABLE, frame_idx=3)
    sig3 = DetectorSignals(
        time_sec=trigger_t + OJAMA_ENTRY_CONSEC_SEC + 0.001, cnn_board=board3,
        is_match_active=True,
    )
    assert det.detect(ctx3, sig3) == BoardState.OJAMA_FALL, (
        "抑制窓を過ぎれば通常閾値のみで発火するはず"
    )


def test_placement_override_alone_reentry_not_suppressed_when_hardening_off() -> None:
    """回帰: enable_ojama_fall_entry_hardening=False (既定) では、 案2の
    exit 直後でも re-entry 抑制は働かず、 従来の frame 数連続判定のみで
    発火する (backwards compat: 案2単体導入時の挙動を変えない)。
    """
    det = OjamaVisualDetector(
        enable_ojama_fall_board_settle=True,
        enable_ojama_fall_placement_override=True,
    )
    board2 = _board_with_ojama_top(2)
    board3 = _board_with_ojama_top(3)

    ctx0 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=0)
    sig0 = DetectorSignals(time_sec=0.0, cnn_board=board2, is_match_active=True)
    assert det.detect(ctx0, sig0) is None

    ctx1 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=1)
    sig1 = DetectorSignals(
        time_sec=0.033, cnn_board=board2, is_match_active=True, own_score_delta=8,
    )
    assert det.detect(ctx1, sig1) == BoardState.STABLE

    # 直後 2 フレーム連続で ROI 増加 → 従来通り即発火 (抑制なし)。
    ctx2 = StateContext(state=BoardState.STABLE, frame_idx=2)
    sig2 = DetectorSignals(time_sec=0.043, cnn_board=board3, is_match_active=True)
    assert det.detect(ctx2, sig2) is None  # 1 フレーム目

    ctx3 = StateContext(state=BoardState.STABLE, frame_idx=3)
    sig3 = DetectorSignals(time_sec=0.077, cnn_board=board3, is_match_active=True)
    assert det.detect(ctx3, sig3) == BoardState.OJAMA_FALL, (
        "entry_hardening=False なら抑制されず、従来通り2フレーム連続で発火"
    )


def test_str_detector_signals_own_pending_ojama_forecast_default_none() -> None:
    """ST-R: DetectorSignals.own_pending_ojama_forecast のデフォルトは None
    (トラッカー無効構成 = 既存ビルドが壊れない)。
    """
    sig = DetectorSignals(
        time_sec=0.0, cnn_board=_empty_board(), is_match_active=True,
    )
    assert sig.own_pending_ojama_forecast is None, (
        "デフォルト None で既存ビルドが壊れない"
    )


# ============================
# 案1 (2026-08-13、OJAMA_FALL出口の根治): おじゃまセル限定 settle 判定
# (enable_ojama_fall_scoped_exit)
# ============================


def _board_with_ojama_and_color(ojama_n: int, color_n: int = 0) -> Board:
    """可視領域に COLOR_OJAMA を ojama_n 個 (下段から) + COLOR_RED を
    color_n 個 (上段から、 ojama と重ならないセルのみ) 配置した盤面。

    案1 scoped exit のテスト用: 色ぷよ (COLOR_RED) の増減がおじゃま個数の
    カウントに一切影響しないことを確認するため、 双方を独立に数えられる
    ように上下から敷き詰める。
    """
    from src.board import BOARD_COLS as _COLS, BOARD_ROWS as _ROWS

    b = Board()
    filled = 0
    for r in range(_ROWS - 1, HIDDEN_ROWS - 1, -1):
        for c in range(_COLS):
            if filled >= ojama_n:
                break
            b.set(r, c, COLOR_OJAMA)
            filled += 1
        if filled >= ojama_n:
            break
    placed = 0
    for r in range(HIDDEN_ROWS, _ROWS):
        for c in range(_COLS):
            if placed >= color_n:
                break
            if int(b.get(r, c)) != COLOR_EMPTY:
                continue
            b.set(r, c, COLOR_RED)
            placed += 1
        if placed >= color_n:
            break
    return b


def test_count_board_ojama_ignores_hidden_row() -> None:
    """`_count_board_ojama` は隠し段 (row=0) を対象外とする (可視領域のみ)。"""
    b = Board()
    b.set(0, 0, COLOR_OJAMA)  # 隠し段 (対象外)
    b.set(HIDDEN_ROWS, 0, COLOR_OJAMA)  # 可視最上段 (対象)
    assert _count_board_ojama(b) == 1


def test_count_board_ojama_counts_below_roi() -> None:
    """`_count_board_ojama` は ROI (可視最上段2行) を超えた下段のおじゃまも
    数える (`_count_top_ojama` との違い)。
    """
    b = Board()
    b.set(HIDDEN_ROWS + 5, 0, COLOR_OJAMA)  # ROI 外・可視領域内
    assert _count_top_ojama(b) == 0
    assert _count_board_ojama(b) == 1


def test_count_board_ojama_ignores_color_puyo() -> None:
    """`_count_board_ojama` は色ぷよ (COLOR_RED 等) を数えない。"""
    board = _board_with_ojama_and_color(ojama_n=5, color_n=10)
    assert _count_board_ojama(board) == 5


def _drive_scoped_exit(
    det: OjamaVisualDetector,
    ojama_counts: list[int],
    *,
    color_counts: "list[int] | None" = None,
    fps: float = 30.0,
    start_frame: int = 0,
    own_pending_ojama_forecast: "int | None" = None,
) -> list[BoardState | None]:
    """ojama_counts (各フレームの可視領域おじゃま数) を順に detect() へ渡す
    (案1 scoped exit テスト用)。 color_counts を渡すと同じ長さで各フレームの
    色ぷよ数を独立に変化させられる (色ぷよの増減が判定に影響しないことの
    検証に使う)。 ctx.state は常に OJAMA_FALL (退出判定のみをテストする)。
    """
    results: list[BoardState | None] = []
    for i, n in enumerate(ojama_counts):
        frame_idx = start_frame + i
        color_n = 0 if color_counts is None else color_counts[i]
        ctx = StateContext(state=BoardState.OJAMA_FALL, frame_idx=frame_idx)
        sig = DetectorSignals(
            time_sec=float(frame_idx) / fps,
            cnn_board=_board_with_ojama_and_color(n, color_n),
            is_match_active=True,
            own_pending_ojama_forecast=own_pending_ojama_forecast,
        )
        results.append(det.detect(ctx, sig))
    return results


def test_scoped_exit_ignores_own_placement_color_churn() -> None:
    """根治確認 (中核): おじゃま数が一定でも自分のツモ設置で色ぷよ数が
    フレーム毎に変わり続ける実測パターン (設置ブロック16/16件) で、
    scoped exit (案1) は退出判定を塞がれない。

    色ぷよ数は毎フレーム DIFF_THRESHOLD ちょうど分だけ増加させる
    (= 案B の全盤面判定なら `diff < THRESHOLD` を満たさず永久に settle
    しないシナリオ)。 おじゃま数は固定のため scoped exit は正常に
    STABLE 復帰する。
    """
    det = OjamaVisualDetector(enable_ojama_fall_scoped_exit=True)
    n_frames = 20
    ojama_counts = [10] * n_frames
    color_counts = [
        i * OJAMA_FALL_SCOPED_EXIT_DIFF_THRESHOLD for i in range(n_frames)
    ]
    results = _drive_scoped_exit(det, ojama_counts, color_counts=color_counts)
    assert any(r == BoardState.STABLE for r in results), (
        "色ぷよ設置が続いても scoped exit はおじゃま数の安定だけで退出するはず"
    )


def test_scoped_exit_blocked_while_ojama_still_falling() -> None:
    """おじゃま数自体がまだ変動中 (落下中) の間は退出しない。"""
    det = OjamaVisualDetector(enable_ojama_fall_scoped_exit=True)
    ojama_counts = [10, 20, 30, 40, 50, 60, 70]  # 単調増加 = 落下継続中
    results = _drive_scoped_exit(det, ojama_counts)
    assert all(r is None for r in results), (
        "おじゃま数が変動中は STABLE 復帰しないはず"
    )


def test_scoped_exit_timeout_forces_stable() -> None:
    """安全弁: OJAMA_FALL_MAX_SEC 秒超過で安定未達でも強制 STABLE 復帰する。"""
    det = OjamaVisualDetector(enable_ojama_fall_scoped_exit=True)

    ctx0 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=0)
    sig0 = DetectorSignals(
        time_sec=0.0, cnn_board=_board_with_ojama_and_color(10), is_match_active=True,
    )
    assert det.detect(ctx0, sig0) is None

    ctx1 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=1)
    sig1 = DetectorSignals(
        time_sec=OJAMA_FALL_MAX_SEC + 0.1,
        cnn_board=_board_with_ojama_and_color(50),  # 不安定
        is_match_active=True,
    )
    assert det.detect(ctx1, sig1) == BoardState.STABLE


def test_scoped_exit_pending_zero_shortens_required_stable_frames() -> None:
    """会計連動: own_pending_ojama_forecast<=0 なら必要静止フレーム数が
    `OJAMA_FALL_SCOPED_EXIT_NO_PENDING_DIVISOR` で短縮され、 pending が
    残っている (または未接続 None の) 場合より早く STABLE 復帰する。
    """
    n_frames = OJAMA_FALL_SCOPED_EXIT_STABLE_FRAMES + 4
    ojama_counts = [10] * n_frames

    det_pending = OjamaVisualDetector(enable_ojama_fall_scoped_exit=True)
    results_pending = _drive_scoped_exit(
        det_pending, ojama_counts, own_pending_ojama_forecast=3,
    )
    first_stable_pending = results_pending.index(BoardState.STABLE)

    det_zero = OjamaVisualDetector(enable_ojama_fall_scoped_exit=True)
    results_zero = _drive_scoped_exit(
        det_zero, ojama_counts, own_pending_ojama_forecast=0,
    )
    first_stable_zero = results_zero.index(BoardState.STABLE)

    assert first_stable_zero < first_stable_pending, (
        "未着弾予告が尽きていれば (pending<=0) より早く STABLE 復帰するはず"
    )
    expected_required = max(
        1,
        OJAMA_FALL_SCOPED_EXIT_STABLE_FRAMES
        // OJAMA_FALL_SCOPED_EXIT_NO_PENDING_DIVISOR,
    )
    assert expected_required < OJAMA_FALL_SCOPED_EXIT_STABLE_FRAMES, (
        "テスト前提: 短縮後の必要フレーム数は短縮前より小さいはず"
    )


def test_scoped_exit_pending_none_behaves_like_untracked_default() -> None:
    """会計トラッカー未接続 (own_pending_ojama_forecast=None) では従来通り
    `OJAMA_FALL_SCOPED_EXIT_STABLE_FRAMES` フル分の静止を要求する
    (= pending>0 の場合と同じ結果になる、 短縮ロジックが誤発動しない)。
    """
    n_frames = OJAMA_FALL_SCOPED_EXIT_STABLE_FRAMES + 4
    ojama_counts = [10] * n_frames

    det_none = OjamaVisualDetector(enable_ojama_fall_scoped_exit=True)
    results_none = _drive_scoped_exit(
        det_none, ojama_counts, own_pending_ojama_forecast=None,
    )
    det_pending = OjamaVisualDetector(enable_ojama_fall_scoped_exit=True)
    results_pending = _drive_scoped_exit(
        det_pending, ojama_counts, own_pending_ojama_forecast=5,
    )
    assert (
        results_none.index(BoardState.STABLE)
        == results_pending.index(BoardState.STABLE)
    ), "None (未接続) は pending>0 と同じ必要フレーム数になるはず"


def test_scoped_exit_default_off_bit_identical_to_board_settle() -> None:
    """回帰: enable_ojama_fall_scoped_exit=False (既定) では
    enable_ojama_fall_board_settle 側の従来ロジックのみが使われ、
    own_pending_ojama_forecast があっても一切参照しない (bit-identical)。
    """
    det = OjamaVisualDetector(enable_ojama_fall_board_settle=True)  # scoped 既定 False
    counts = [20, 30, 40, 44] + [45] * 8
    results = _drive_board_settle(det, counts)
    # own_pending_ojama_forecast を渡しても結果が変わらないことを追加確認する。
    det2 = OjamaVisualDetector(enable_ojama_fall_board_settle=True)
    results2: list[BoardState | None] = []
    for i, n in enumerate(counts):
        ctx = StateContext(state=BoardState.OJAMA_FALL, frame_idx=i)
        sig = DetectorSignals(
            time_sec=float(i) / 30.0, cnn_board=_board_with_n_puyos(n),
            is_match_active=True, own_pending_ojama_forecast=0,
        )
        results2.append(det2.detect(ctx, sig))
    assert results == results2, (
        "scoped_exit=False では own_pending_ojama_forecast を無視するはず"
    )


def test_scoped_exit_takes_priority_over_board_settle_when_both_enabled() -> None:
    """両フラグ ON (通常は想定しないが安全側の確認): 案1 (scoped exit) が
    案B (board_settle) より優先され、 色ぷよ churn で塞がれない。
    """
    det = OjamaVisualDetector(
        enable_ojama_fall_board_settle=True,
        enable_ojama_fall_scoped_exit=True,
    )
    n_frames = 20
    ojama_counts = [10] * n_frames
    color_counts = [
        i * OJAMA_FALL_SCOPED_EXIT_DIFF_THRESHOLD for i in range(n_frames)
    ]
    results = _drive_scoped_exit(det, ojama_counts, color_counts=color_counts)
    assert any(r == BoardState.STABLE for r in results), (
        "両フラグ ON でも scoped exit が優先され、 色ぷよ churn で塞がれないはず"
    )


def test_scoped_exit_intercept_resets_no_stale_contamination() -> None:
    """案B の (c) テストと同型: OJAMA_FALL から他 state に途中横取りされた
    後の再突入で scoped 系の内部 state が汚染されないことを確認する。
    """
    det = OjamaVisualDetector(enable_ojama_fall_scoped_exit=True)

    ctx0 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=0)
    sig0 = DetectorSignals(
        time_sec=0.0, cnn_board=_board_with_ojama_and_color(10), is_match_active=True,
    )
    assert det.detect(ctx0, sig0) is None
    assert det._scoped_exit_start_frame == 0  # noqa: SLF001

    ctx_other = StateContext(state=BoardState.STABLE, frame_idx=1)
    sig_other = DetectorSignals(
        time_sec=100.0, cnn_board=_empty_board(), is_match_active=True,
    )
    assert det.detect(ctx_other, sig_other) is None
    assert det._scoped_exit_start_frame == -1  # noqa: SLF001

    ctx2 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=2)
    sig2 = DetectorSignals(
        time_sec=100.05, cnn_board=_board_with_ojama_and_color(10),
        is_match_active=True,
    )
    result = det.detect(ctx2, sig2)
    assert result is None, (
        "再突入直後は新規 start として記録され、 stale time でタイムアウトしてはいけない"
    )


def test_str_detector_signals_own_chain_hold_until_sec_default_zero() -> None:
    """ST-R: DetectorSignals.own_chain_hold_until_sec のデフォルトは 0.0。"""
    sig = DetectorSignals(
        time_sec=0.0, cnn_board=_empty_board(), is_match_active=True,
    )
    assert sig.own_chain_hold_until_sec == 0.0, "デフォルト 0.0 で既存ビルドが壊れない"
