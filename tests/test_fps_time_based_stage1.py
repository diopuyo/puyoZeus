"""フレーム定数→時間定数化 Stage1 の回帰テスト (2026-07-25).

frame_idx 差分で比較していた各種ゲートを time_sec 差分に置換した際、
60fps 動画では旧フレーム基準の判定と完全 bit-identical になり、
30fps 動画では実秒基準になる (= 旧実装の「体感 2 倍遅延」が解消される)
ことを確認する。

対象:
    - GravitySettleDetector (src/state_detectors.py) の
      GRAVITY_SETTLE_PHYSICS_CLEAR_MIN(_SEC) ゲート。
    - OjamaVisualDetector._detect_ojama_fall_exit_board_settle
      (src/ojama_visual_detector.py) の OJAMA_FALL_SETTLE_MIN(_SEC) ゲート。
    - RecognitionPipeline.update() (src/recognition_pipeline.py) の
      MATCH_ACTIVE_HOLD_(FRAMES|SEC) ゲート (recent_active)。
"""

from __future__ import annotations

from src.board import BOARD_COLS, BOARD_ROWS, Board
from src.board_state_machine import (
    BoardState,
    DetectorSignals,
    GRAVITY_SETTLE_PHYSICS_CLEAR_MIN,
    StateContext,
)
from src.ojama_visual_detector import (
    OJAMA_FALL_SETTLE_MIN_FRAMES,
    OjamaVisualDetector,
)
from src.state_detectors import GravitySettleDetector


def _board_with_n_puyos(n: int) -> Board:
    """count_puyos() が n を返す盤面を生成する (色は判定に無関係)."""
    from src.board import COLOR_OJAMA

    b = Board()
    filled = 0
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if filled >= n:
                return b
            b.set(r, c, COLOR_OJAMA)
            filled += 1
    return b


# ============================
# GravitySettleDetector: GRAVITY_SETTLE_PHYSICS_CLEAR_MIN(_SEC) ゲート
# ============================


def _drive_gravity_settle_gate(fps: float) -> int:
    """一定盤面 (diff=0) を fps 相当の時刻で投入し、

    `_stable_consec` が 0→1 に切り替わる (= physics_clear_min ゲートが
    開いた) 最初の frame_idx を返す。

    frame 0 は GRAVITY_SETTLE 突入の記録フレーム (必ず None・stable_consec
    更新なし) なので、 呼び出しは frame_idx=1 から開始する。
    """
    det = GravitySettleDetector()
    board = Board()
    ctx0 = StateContext(state=BoardState.GRAVITY_SETTLE, frame_idx=0)
    sig0 = DetectorSignals(time_sec=0.0, cnn_board=board, is_match_active=True)
    assert det.detect(ctx0, sig0) is None  # entry frame

    frame_idx = 0
    while det._stable_consec == 0:  # noqa: SLF001 (内部 state を直接観測)
        frame_idx += 1
        ctx = StateContext(state=BoardState.GRAVITY_SETTLE, frame_idx=frame_idx)
        sig = DetectorSignals(
            time_sec=frame_idx / fps, cnn_board=board, is_match_active=True,
        )
        det.detect(ctx, sig)
        assert frame_idx < 1000, "gate が開かない (無限ループ防止の安全弁)"
    return frame_idx


def test_gravity_settle_physics_clear_gate_bit_identical_at_60fps() -> None:
    """60fps (time_sec = frame_idx/60) では旧フレーム基準ゲートと同一 frame で開く。

    旧実装: `frames_in_settle = frame_idx - start >= GRAVITY_SETTLE_PHYSICS_CLEAR_MIN`
    で最初に True になる frame_idx は厳密に GRAVITY_SETTLE_PHYSICS_CLEAR_MIN。
    """
    gate_frame = _drive_gravity_settle_gate(fps=60.0)
    assert gate_frame == GRAVITY_SETTLE_PHYSICS_CLEAR_MIN, (
        "60fps では旧フレーム基準ゲートと bit-identical であるべき"
    )


def test_gravity_settle_physics_clear_gate_faster_real_time_at_30fps() -> None:
    """30fps では実秒基準になり、旧フレーム基準より早くゲートが開く
    (= 体感 2 倍遅延の解消を確認)。

    旧実装 (frame_idx 差分) なら fps に関わらず GRAVITY_SETTLE_PHYSICS_CLEAR_MIN
    frame 必要 = 30fps では実時間 2 倍かかっていた。新実装は実秒基準なので
    30fps でもより少ない frame 数でゲートが開く。
    """
    gate_frame_30fps = _drive_gravity_settle_gate(fps=30.0)
    assert gate_frame_30fps < GRAVITY_SETTLE_PHYSICS_CLEAR_MIN, (
        "30fps では旧フレーム基準 (frame 固定) より早くゲートが開くべき"
        " (実秒基準化の効果)"
    )
    # 実秒換算で旧実装 (frame 基準 3 frame = 0.1s@30fps) の半分近くに短縮される。
    gate_time_sec = gate_frame_30fps / 30.0
    old_frame_based_time_sec = GRAVITY_SETTLE_PHYSICS_CLEAR_MIN / 30.0
    assert gate_time_sec < old_frame_based_time_sec


# ============================
# OjamaVisualDetector: OJAMA_FALL_SETTLE_MIN(_SEC) ゲート (案B board settle)
# ============================


def _drive_ojama_board_settle_gate(fps: float) -> int:
    """全盤面 count 不変 (diff=0) を fps 相当の時刻で投入し、

    `_board_stable_consec` が 0→1 に切り替わる最初の frame_idx を返す。
    """
    det = OjamaVisualDetector(enable_ojama_fall_board_settle=True)
    board = _board_with_n_puyos(20)
    ctx0 = StateContext(state=BoardState.OJAMA_FALL, frame_idx=0)
    sig0 = DetectorSignals(time_sec=0.0, cnn_board=board, is_match_active=True)
    assert det.detect(ctx0, sig0) is None  # entry frame

    frame_idx = 0
    while det._board_stable_consec == 0:  # noqa: SLF001
        frame_idx += 1
        ctx = StateContext(state=BoardState.OJAMA_FALL, frame_idx=frame_idx)
        sig = DetectorSignals(
            time_sec=frame_idx / fps, cnn_board=board, is_match_active=True,
        )
        det.detect(ctx, sig)
        assert frame_idx < 1000, "gate が開かない (無限ループ防止の安全弁)"
    return frame_idx


def test_ojama_board_settle_gate_bit_identical_at_60fps() -> None:
    """60fps では旧フレーム基準ゲート (OJAMA_FALL_SETTLE_MIN_FRAMES) と同一 frame。"""
    gate_frame = _drive_ojama_board_settle_gate(fps=60.0)
    assert gate_frame == OJAMA_FALL_SETTLE_MIN_FRAMES


def test_ojama_board_settle_gate_faster_real_time_at_30fps() -> None:
    """30fps では実秒基準になり、旧フレーム基準より早くゲートが開く。"""
    gate_frame_30fps = _drive_ojama_board_settle_gate(fps=30.0)
    assert gate_frame_30fps < OJAMA_FALL_SETTLE_MIN_FRAMES


# ============================
# RecognitionPipeline.update(): MATCH_ACTIVE_HOLD_(FRAMES|SEC) (recent_active)
# ============================


def _make_recent_active_pipe():
    """MATCH_ACTIVE_HOLD 境界テスト用の最小 RecognitionPipeline を返す。

    stable_frame_count を大きくして STABLE に到達させず、
    sm_active (state machine 由来の hysteresis) を混入させずに
    recent_active 単体の挙動を観測できるようにする。
    """
    from src.recognition_pipeline import RecognitionPipeline
    from tests.test_recognition_pipeline import (
        _StubImageReader,
        _StubMatchDetector,
        _empty_board,
    )

    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    return RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        stable_frame_count=1000,
    )


def test_match_active_hold_recent_active_bit_identical_at_60fps() -> None:
    """60fps では MATCH_ACTIVE_HOLD_FRAMES と MATCH_ACTIVE_HOLD_SEC の境界が一致する。

    is_match_active を一度 True にした後 in_match=False に切り替え、
    MATCH_ACTIVE_HOLD_SEC 境界のちょうど内側で is_match_active の
    hysteresis (recent_active) が旧フレーム基準と同じ境界で継続することを
    確認する。
    """
    from tests.test_recognition_pipeline import _dummy_frame, _StubMatchDetector

    pipe = _make_recent_active_pipe()
    frame = _dummy_frame()

    # frame 0: in_match=True で active 観測を記録 (t=0.0)。
    pipe.update(0, 0.0, frame)
    assert pipe._last_active_frame_time == 0.0  # noqa: SLF001

    # in_match=False に切替 (以降 raw_active=False)。
    pipe._match_detector = _StubMatchDetector(in_match=False)  # type: ignore[assignment]

    hold_sec = pipe.MATCH_ACTIVE_HOLD_SEC
    # 境界のちょうど内側 (60fps: hold_sec 秒未満) では recent_active で継続。
    frame_idx_inside = round(hold_sec * 60) - 1
    t_inside = frame_idx_inside / 60.0
    res_inside = pipe.update(frame_idx_inside, t_inside, frame)
    assert res_inside.is_match_active is True, (
        "MATCH_ACTIVE_HOLD_SEC 境界内側では試合中継続のはず"
    )


def test_match_active_hold_recent_active_faster_expiry_at_30fps() -> None:
    """30fps では実秒基準になり、旧フレーム基準 (frame 固定) より早く

    (=より少ない frame 数で) hysteresis が切れることを確認する
    (体感 2 倍遅延の解消)。
    """
    from tests.test_recognition_pipeline import _dummy_frame, _StubMatchDetector

    pipe = _make_recent_active_pipe()
    frame = _dummy_frame()

    pipe.update(0, 0.0, frame)
    pipe._match_detector = _StubMatchDetector(in_match=False)  # type: ignore[assignment]

    hold_sec = pipe.MATCH_ACTIVE_HOLD_SEC
    hold_frames_legacy = pipe.MATCH_ACTIVE_HOLD_FRAMES

    # 30fps で「旧フレーム基準なら継続していたはずの frame 数」を投入すると、
    # 実秒ベースでは既に hold_sec を超えているため非 active になる。
    frame_idx_30fps = hold_frames_legacy  # 旧実装ならまだ hold 内 (<=)
    t_30fps = frame_idx_30fps / 30.0
    assert t_30fps > hold_sec, "テスト前提: 30fps ではこの frame で hold_sec を超過する"
    res = pipe.update(frame_idx_30fps, t_30fps, frame)
    assert res.is_match_active is False, (
        "30fps では実秒基準になり、旧フレーム基準より早く (体感 2 倍遅延なく)"
        " 非 active に切り替わるべき"
    )
