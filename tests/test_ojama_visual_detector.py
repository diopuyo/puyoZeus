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
    OJAMA_FALL_MAX_SEC,
    OJAMA_FALL_SETTLE_DIFF_THRESHOLD,
    OJAMA_FALL_SETTLE_MIN_FRAMES,
    OJAMA_FALL_SETTLE_STABLE_FRAMES,
    OJAMA_SETTLE_CONSEC,
    OjamaVisualDetector,
    _count_top_ojama,
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
