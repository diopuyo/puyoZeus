"""長時間劣化 仕切り直し計装: c22フル走行での reset()発火有無の直接確定 (2026-08-06)。

修正A+B (`enable_online_hsv_refresh`) をフル適用しても c22 f188154 の12セル
破損が同一再現した (乖離は試合43開始直後、最初の共通frameで既に18セル差)。
新疑い: 試合43の境界で `RecognitionPipeline.reset()` 自体が発火していない
(なら修正A+Bは作動機会ゼロ)。本スクリプトは c22 を t=0〜3150 まで1回だけ
フル走行し、reset() 呼び出し・試合境界検知・drift resync 発火/抑止・
12セル破損の焼き付き onset を計装する。

## 判定したいこと
    (a) reset()が試合43境界で発火していない → 境界検知の欠落が真因
    (b) reset()は発火したが破損する → A+B以外の持ち越し状態 or 試合内の機構
    (c) drift resyncの抑止/発火状況 → 実験Aの結果の実条件での再評価

## 計装方式 (src/ 本番コード変更禁止、読み取りとラップのみ)
    1. `RecognitionPipeline.reset` — 呼び出し毎に time_sec + 前後の
       2P confirmed_board 概要 (残渣確認) + online_hsv較正状態を記録。
    2. `DriftDetector.reset` — 呼び出し毎に time_sec + side を記録
       (drift resyncが実際に発火した証拠)。
    3. `_is_score_reset_boundary` — 呼び出し毎の score1/score2/prev1/prev2
       と判定結果を記録 (試合境界検知そのものの動作確認)。
    4. 毎frame (詳細窓 t∈[DETAIL_WINDOW_START, DETAIL_WINDOW_END] のみ)
       is_match_active・_match_start_boundary_latched・
       drift resync 抑止カウンタ・12セル値を記録。

Usage (フル走行、~1-2時間想定):
    PYTHONPATH=. ./venv/bin/python -m scripts._verify_c22_reset_trace_2026-08-06
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.recognition_pipeline as rp  # noqa: E402
from src.drift_detector import DriftDetector  # noqa: E402
from src.fps_normalize import resolve_normalize_fps_30_stride  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# =============================================================================
# 定数
# =============================================================================

VIDEO_DIR: Path = Path("/home/ryouj/frames")
OUT_DIR: Path = Path("data/verify/c22_reset_trace_2026-08-06")
VIDEO_STEM: str = "c22"
TARGET_SIDE: str = "2P"

FULL_RUN_START_SEC: float = 0.0
FULL_RUN_END_SEC: float = 3150.0

TARGET_W: int = 1920
TARGET_H: int = 1080

BURST_GATE_OPEN_THRESHOLD_V4: float = 0.954

# 詳細frame記録窓 (試合43開始 t≈3110.7 の前後、メモリ節約のため全編は記録しない)。
DETAIL_WINDOW_START: float = 3090.0
DETAIL_WINDOW_END: float = 3150.0

# 進捗ログ間隔 (秒、フル走行の生存確認用)。
PROGRESS_LOG_INTERVAL_SEC: float = 60.0

# col3 rows3-11 (9) + col4 rows5-7 (3) = 12セル (row, col, correct_value)。
# 2026-08-06 診断で確定した正解値 (baseline npz 直接diff、row10c3=3 に修正済み)。
TARGET_CELLS: "tuple[tuple[int, int, int], ...]" = (
    (3, 3, 1), (4, 3, 4), (5, 3, 3), (6, 3, 3), (7, 3, 4),
    (8, 3, 4), (9, 3, 4), (10, 3, 3), (11, 3, 3),
    (5, 4, 1), (6, 4, 4), (7, 4, 3),
)


@dataclass
class ResetCallEvent:
    """`RecognitionPipeline.reset()` 1呼び出し分の観測。"""

    call_idx: int
    t_sec: float
    confirmed_2p_before_nonempty: int
    confirmed_2p_after_is_none: bool
    online_hsv_injected_before: bool
    online_hsv_injected_colors_before: str
    online_hsv_injected_after: bool
    online_hsv_injected_colors_after: str


@dataclass
class DriftResetEvent:
    """`DriftDetector.reset()` 1呼び出し分の観測 (resync実発火の証拠)。"""

    t_sec: float
    side: str


@dataclass
class BoundaryCheckEvent:
    """`_is_score_reset_boundary` 1呼び出し分の観測。"""

    t_sec: float
    score1: "int | None"
    score2: "int | None"
    prev1: "int | None"
    prev2: "int | None"
    boundary_candidate: bool


@dataclass
class DetailFrameRecord:
    """詳細窓内 1frame分の観測 (試合境界+ドリフト抑止+12セル)。"""

    frame_idx: int
    t_sec: float
    is_match_active: bool
    match_start_boundary_latched: bool
    score_reset_boundary_streak: int
    score_1p: "int | None"
    score_2p: "int | None"
    online_hsv_injected: bool
    online_hsv_injected_colors: str
    drift_resync_hsv_gate_suppressed_2p: int
    drift_resync_start_guard_suppressed_2p: int
    state_2p: str
    cell_values: str  # "r3c3=1,r4c3=4,..." 形式 (12セル一括)


@dataclass
class _ProbeState:
    """計装の可変状態。"""

    current_t_sec: float = -1.0
    current_frame_idx: int = -1
    pipeline_ref: "RecognitionPipeline | None" = None
    drift_side_map: dict[int, str] = field(default_factory=dict)
    reset_call_idx: int = 0
    reset_events: list[ResetCallEvent] = field(default_factory=list)
    drift_reset_events: list[DriftResetEvent] = field(default_factory=list)
    boundary_events: list[BoundaryCheckEvent] = field(default_factory=list)
    detail_records: list[DetailFrameRecord] = field(default_factory=list)


_STATE = _ProbeState()


# =============================================================================
# monkeypatch 1: RecognitionPipeline.reset (発火有無の直接証拠)
# =============================================================================

_ORIG_RESET: Callable = RecognitionPipeline.reset


def _summarize_confirmed(board: "object | None") -> int:
    """confirmed_board の非空セル数を返す (残渣確認用の軽量サマリ)。"""
    if board is None:
        return -1
    from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY
    n = 0
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if int(board.get(r, c)) != COLOR_EMPTY:
                n += 1
    return n


def _wrapped_reset(self: RecognitionPipeline) -> None:
    """`RecognitionPipeline.reset()` のラップ (発火の直接記録、前後の状態比較)。"""
    before_board = self._sm_2p.context.confirmed_board
    before_n = _summarize_confirmed(before_board)
    before_injected = self._online_hsv_injected
    before_colors = sorted(self._online_hsv_injected_colors)
    _ORIG_RESET(self)
    after_board = self._sm_2p.context.confirmed_board
    _STATE.reset_call_idx += 1
    _STATE.reset_events.append(ResetCallEvent(
        call_idx=_STATE.reset_call_idx, t_sec=_STATE.current_t_sec,
        confirmed_2p_before_nonempty=before_n,
        confirmed_2p_after_is_none=(after_board is None),
        online_hsv_injected_before=before_injected,
        online_hsv_injected_colors_before=str(before_colors),
        online_hsv_injected_after=self._online_hsv_injected,
        online_hsv_injected_colors_after=str(sorted(self._online_hsv_injected_colors)),
    ))
    print(
        f"[reset_trace] reset() #{_STATE.reset_call_idx} at t={_STATE.current_t_sec:.2f} "
        f"confirmed_2p_before_nonempty={before_n}",
        flush=True,
    )


# =============================================================================
# monkeypatch 2: DriftDetector.reset (resync実発火の直接証拠)
# =============================================================================

_ORIG_DRIFT_RESET: Callable = DriftDetector.reset


def _wrapped_drift_reset(self: DriftDetector) -> None:
    """`DriftDetector.reset()` のラップ (side別のresync発火を記録)。"""
    side = _STATE.drift_side_map.get(id(self), "?")
    _STATE.drift_reset_events.append(DriftResetEvent(
        t_sec=_STATE.current_t_sec, side=side,
    ))
    _ORIG_DRIFT_RESET(self)


# =============================================================================
# monkeypatch 3: _is_score_reset_boundary (試合境界検知そのものの動作確認)
# =============================================================================

_ORIG_IS_BOUNDARY: Callable = rp._is_score_reset_boundary


def _wrapped_is_score_reset_boundary(
    score1: "int | None", score2: "int | None", prev1: "int | None",
    prev2: "int | None", strict: bool = False,
) -> bool:
    """`_is_score_reset_boundary` のラップ (詳細窓内のみ記録、判定は不変)。"""
    result = _ORIG_IS_BOUNDARY(score1, score2, prev1, prev2, strict)
    if DETAIL_WINDOW_START <= _STATE.current_t_sec <= DETAIL_WINDOW_END:
        _STATE.boundary_events.append(BoundaryCheckEvent(
            t_sec=_STATE.current_t_sec, score1=score1, score2=score2,
            prev1=prev1, prev2=prev2, boundary_candidate=result,
        ))
    return result


def install_probes() -> None:
    """全3 monkeypatch をインストールする (src/ 本番ファイルは書き換えない)。"""
    RecognitionPipeline.reset = _wrapped_reset
    DriftDetector.reset = _wrapped_drift_reset
    rp._is_score_reset_boundary = _wrapped_is_score_reset_boundary


def uninstall_probes() -> None:
    """monkeypatch を復元する。"""
    RecognitionPipeline.reset = _ORIG_RESET
    DriftDetector.reset = _ORIG_DRIFT_RESET
    rp._is_score_reset_boundary = _ORIG_IS_BOUNDARY


# =============================================================================
# pipeline構築 (v4本命構成 + enable_online_hsv_refresh=True)
# =============================================================================


def _build_pipeline() -> RecognitionPipeline:
    """本番規格 + v4フラグ + 長時間劣化修正A+B (enable_online_hsv_refresh)。"""
    return RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
        enable_effect_gate=True, enable_burst_guard_v2=True,
        enable_transition_merge_guard=True,
        burst_gate_open_threshold=BURST_GATE_OPEN_THRESHOLD_V4,
        enable_hidden_row_burst_guard=True,
        enable_online_hsv_refresh=True,
    )


# =============================================================================
# 1frame分の詳細記録 (詳細窓内のみ)
# =============================================================================


def _format_cells(board: "object | None") -> str:
    """対象12セルの値を "r{r}c{c}={v}" 形式でまとめる (board=None なら空文字)。"""
    if board is None:
        return ""
    return ",".join(
        f"r{r}c{c}={int(board.get(r, c))}" for r, c, _ in TARGET_CELLS
    )


def _record_detail_frame(
    pipeline: RecognitionPipeline, fi: int, t_sec: float, result: "object",
) -> None:
    """詳細窓内の1frame分を記録する。"""
    p2 = result.p2 if TARGET_SIDE == "2P" else result.p1
    _STATE.detail_records.append(DetailFrameRecord(
        frame_idx=fi, t_sec=t_sec, is_match_active=result.is_match_active,
        match_start_boundary_latched=pipeline._match_start_boundary_latched,
        score_reset_boundary_streak=pipeline._score_reset_boundary_streak,
        score_1p=result.p1.score, score_2p=result.p2.score,
        online_hsv_injected=pipeline._online_hsv_injected,
        online_hsv_injected_colors=str(sorted(pipeline._online_hsv_injected_colors)),
        drift_resync_hsv_gate_suppressed_2p=pipeline._drift_resync_hsv_gate_suppressed_2p,
        drift_resync_start_guard_suppressed_2p=pipeline._drift_resync_start_guard_suppressed_2p,
        state_2p=p2.state.name, cell_values=_format_cells(p2.confirmed_board),
    ))


# =============================================================================
# メイン走行
# =============================================================================


def run() -> None:
    """c22 を t=0〜FULL_RUN_END_SEC までフル走行する (1回のみ)。"""
    video_path = VIDEO_DIR / f"video_{VIDEO_STEM}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 動画を開けません: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(FULL_RUN_START_SEC * fps)
    n_frames = int((FULL_RUN_END_SEC - FULL_RUN_START_SEC) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    pipeline = _build_pipeline()
    pipeline.set_video_id(f"video_{VIDEO_STEM}")
    _STATE.pipeline_ref = pipeline
    _STATE.drift_side_map = {
        id(pipeline._drift_1p): "1P", id(pipeline._drift_2p): "2P",
    }
    stride = resolve_normalize_fps_30_stride(fps)
    print(
        f"[{VIDEO_STEM}] full run: start_frame={start_frame} fps={fps:.2f} "
        f"stride={stride} n_frames={n_frames} (推定処理frame数={n_frames // stride})",
        flush=True,
    )

    wall_start = time.monotonic()
    next_progress_log_sec = 0.0
    for local_i in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if local_i % stride != 0:
            continue
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        fi = start_frame + local_i
        t_sec = fi / fps
        _STATE.current_frame_idx = fi
        _STATE.current_t_sec = t_sec
        result = pipeline.update(fi, t_sec, frame)
        if DETAIL_WINDOW_START <= t_sec <= DETAIL_WINDOW_END:
            _record_detail_frame(pipeline, fi, t_sec, result)
        if t_sec >= next_progress_log_sec:
            elapsed = time.monotonic() - wall_start
            print(
                f"[progress] t_sec={t_sec:.1f}/{FULL_RUN_END_SEC:.1f} "
                f"wall_elapsed={elapsed:.1f}s reset_calls={_STATE.reset_call_idx}",
                flush=True,
            )
            next_progress_log_sec += PROGRESS_LOG_INTERVAL_SEC
    cap.release()
    _write_outputs()
    _report()


# =============================================================================
# 出力
# =============================================================================


def _write_records_csv(path: Path, records: list) -> None:
    """dataclass のリストを1枚のCSVに書き出す共通ヘルパー。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].__dataclass_fields__))
        writer.writeheader()
        for r in records:
            writer.writerow(r.__dict__)


def _write_outputs() -> None:
    """4種のCSVを出力する。"""
    _write_records_csv(OUT_DIR / f"{VIDEO_STEM}_reset_events.csv", _STATE.reset_events)
    _write_records_csv(OUT_DIR / f"{VIDEO_STEM}_drift_reset_events.csv", _STATE.drift_reset_events)
    _write_records_csv(OUT_DIR / f"{VIDEO_STEM}_boundary_events.csv", _STATE.boundary_events)
    _write_records_csv(OUT_DIR / f"{VIDEO_STEM}_detail_frames.csv", _STATE.detail_records)
    print(f"[{VIDEO_STEM}] CSV出力: 4ファイル ({OUT_DIR})", flush=True)


# =============================================================================
# レポート
# =============================================================================


def _report_reset_near_boundary() -> None:
    """試合43開始 (t≈3106-3115) 近傍で reset() が発火したかを報告する。"""
    near = [e for e in _STATE.reset_events if 3100.0 <= e.t_sec <= 3120.0]
    print(f"  t=3100-3120 の reset()呼び出し: {len(near)}件")
    for e in near:
        print(
            f"    #{e.call_idx} t={e.t_sec:.3f} "
            f"confirmed_2p_before_nonempty={e.confirmed_2p_before_nonempty} "
            f"confirmed_2p_after_is_none={e.confirmed_2p_after_is_none} "
            f"online_hsv: injected {e.online_hsv_injected_before}->{e.online_hsv_injected_after} "
            f"colors {e.online_hsv_injected_colors_before}->{e.online_hsv_injected_colors_after}"
        )
    if not near:
        print("  [(a)の疑い] 試合43境界での reset() 呼び出しが0件")


def _report_drift_resync_near_boundary() -> None:
    """試合43近傍でのdrift resync発火 (DriftDetector.reset) を報告する。"""
    near = [
        e for e in _STATE.drift_reset_events
        if e.side == "2P" and 3100.0 <= e.t_sec <= 3150.0
    ]
    print(f"  t=3100-3150 の 2P drift.reset()呼び出し: {len(near)}件")
    for e in near[:20]:
        print(f"    t={e.t_sec:.3f}")


def _report_onset() -> None:
    """12セル破損の焼き付きonsetを詳細窓レコードから確定する。"""
    for rec in _STATE.detail_records:
        cells = dict(
            item.split("=") for item in rec.cell_values.split(",") if item
        )
        wrong = [
            (r, c, correct, cells.get(f"r{r}c{c}"))
            for r, c, correct in TARGET_CELLS
            if cells.get(f"r{r}c{c}") is not None and int(cells[f"r{r}c{c}"]) != correct
        ]
        if len(wrong) >= 6:
            print(
                f"  [onset候補] t={rec.t_sec:.3f} frame_idx={rec.frame_idx} "
                f"state_2p={rec.state_2p} 誤りセル数={len(wrong)} "
                f"online_hsv_injected={rec.online_hsv_injected} "
                f"colors={rec.online_hsv_injected_colors} "
                f"drift_hsv_gate_suppressed={rec.drift_resync_hsv_gate_suppressed_2p} "
                f"drift_start_guard_suppressed={rec.drift_resync_start_guard_suppressed_2p}"
            )
            return
    print("  [onset候補] 詳細窓内 (t=3090-3150) で6セル以上の誤りは検出されなかった")


def _report() -> None:
    """(a)/(b)/(c) の判定材料を出力する。"""
    print(f"\n=== {VIDEO_STEM} フル走行 reset()発火トレース 判定 ===")
    print("\n[1] reset() 発火有無 (試合43境界近傍)")
    _report_reset_near_boundary()
    print(f"\n[1b] reset() 全呼び出し総数: {len(_STATE.reset_events)}件")
    print("\n[2] drift resync 実発火 (試合43近傍)")
    _report_drift_resync_near_boundary()
    print(f"\n[3] _is_score_reset_boundary 詳細窓内呼び出し総数: {len(_STATE.boundary_events)}件")
    true_events = [e for e in _STATE.boundary_events if e.boundary_candidate]
    print(f"    boundary_candidate=True: {len(true_events)}件")
    for e in true_events[:10]:
        print(f"    t={e.t_sec:.3f} score1={e.score1} score2={e.score2} prev1={e.prev1} prev2={e.prev2}")
    print("\n[4] 12セル破損の焼き付きonset")
    _report_onset()


def main() -> None:
    install_probes()
    try:
        run()
    finally:
        uninstall_probes()


if __name__ == "__main__":
    main()
