"""境界マルチシグナル Step 0診断 (docs/BOUNDARY_MULTISIGNAL_DESIGN_2026-08-17.md A-1)。

video_c21 f57548 (t≈959.133秒、2P「ばたんきゅー」/1P「やった!」結果パネル) で
is_match_active=True のまま STABLE スナップショットが採取された不発理由を特定する。

本体コード (src/) は変更しない。RecognitionPipeline / MatchEndDetector /
ScoreZeroDetector を外部から直接呼び計装するだけ。

## 手順 (A-1 の3点)
1. MatchEndDetector.detect(frame) を該当フレーム列 (演出開始前まで遡って) に
   直接適用し、score/template_name/detected を全フレームログ。
2. 本番相当の連続処理で同区間を流し、hard_match_off 系の全分岐値を
   フレームごとに再構築してログ (production の PipelineResult.match_end_locked
   を正とし、score_zero_both/chain_in_progress/score_actively_moving/
   hard_match_off/effective_hard_off は pipeline の private state を直接読んで
   再構成する — ロジックの再実装ではなく同一インスタンスの内部状態読み出し)。
   **重要な訂正**: scripts/collect_boards_lean.py (本番収集経路) は
   force_in_match=True を配線上ハードコードしている
   (collect_boards_lean.py:1327)。本診断は force_in_match=True (本番実際値) と
   False (発注時想定値) の両方で走らせ、どちらが実測構成と一致するかを明示する。
3. 30秒チャンク再起動方式 (scripts/_collect_yardstick_v2_bc_2026-08-15.py の
   CHUNK_SEC=30) が MatchEndDetector._last_detected_t をリセットしていないか
   確認する。ただし c21_g11.npz (f57548 のアンカー起源) は
   scripts/_jobs_general_yardstick_F_2026-08-17.txt により
   `--start-sec 858.0 --max-sec 132.0` の単一連続プロセスで収集されている
   (チャンク方式ではない) — この事実をログに明記する。

## 使い方 (WSL)
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_match_end_miss_2026-08-17
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board_state_machine import BoardState  # noqa: E402
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402
from src.match_end_detector import (  # noqa: E402
    DEFAULT_NCC_THRESHOLD,
    MatchEndDetector,
)
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# =============================================================================
# 定数
# =============================================================================

VIDEO_DIR: Path = Path.home() / "frames"
VIDEO: str = "c21"
# labels.tsv 030 行: c21 2P f57548 t=959.133 (skip=ばたんきゅー)
ANCHOR_FRAME_IDX: int = 57548
ANCHOR_T_SEC: float = 959.133
# 演出開始前まで遡る (design doc A-1-1)。既存計装 (diag_general_chain_
# contamination) の密ログで OJAMA_FALL/STABLE/CHAIN の実ゲーム挙動が
# 955.6-959.1 に見えているため、955.0 から開始する。
WINDOW_START_SEC: float = 950.0
# lockdown_sec=5.0 の1サイクル以上 + 次試合開始境界の様子まで見る
WINDOW_END_SEC: float = 972.0
# pipeline 内部 state (score tracker 等) を汚さないための warmup 開始点
# (yardstick 収集と同じ game_start - 2.0 秒マージン、anchor_plan.tsv 準拠)
GAME_START_SEC: float = 858.0
WARMUP_MARGIN_SEC: float = 2.0

OUT_DIR: Path = _ROOT / "data" / "verify" / "diag_match_end_miss_2026-08-17"
FRAMES_DIR: Path = OUT_DIR / "frames"


# =============================================================================
# 1. MatchEndDetector 直接計装 (ステートレス detect())
# =============================================================================


@dataclass
class DetectRow:
    frame_idx: int
    t_sec: float
    dt_from_anchor: float
    detected: bool
    template_name: "str | None"
    score: float


def run_detect_only(cap: cv2.VideoCapture, fps: float) -> "list[DetectRow]":
    """MatchEndDetector.detect() を毎フレーム独立に適用 (状態なし)。"""
    detector = MatchEndDetector.load_default()
    start_frame = int(WINDOW_START_SEC * fps)
    end_frame = int(WINDOW_END_SEC * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    rows: "list[DetectRow]" = []
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t_sec = frame_idx / fps
        res = detector.detect(frame)
        rows.append(DetectRow(
            frame_idx=frame_idx, t_sec=round(t_sec, 4),
            dt_from_anchor=round(t_sec - ANCHOR_T_SEC, 4),
            detected=res.detected, template_name=res.template_name,
            score=round(res.score, 4),
        ))
        frame_idx += 1
    return rows


# =============================================================================
# 2. 本番相当連続処理 (force_in_match True/False 両方)
# =============================================================================


def build_pipeline(force_in_match: bool) -> RecognitionPipeline:
    """構成F相当 (jobs_general_yardstick_F_2026-08-17.txt と同一フラグ集合)。

    collect_boards_lean.py:1327 は force_in_match=True をハードコードしている
    (本番実態)。design doc 発注文の force_in_match=False は想定値であり、
    両方を比較する。
    """
    return RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        load_next_detector=True,
        temporal_smoothing=1,
        force_in_match=force_in_match,
        enable_effect_gate=True,
        enable_burst_guard_v2=True,
        enable_transition_merge_guard=True,
        burst_gate_open_threshold=0.954,
        enable_hidden_row_burst_guard=True,
        enable_match_transition_debounce=True,
        enable_ojama_fall_placement_override=True,
        enable_patch_fp_hsv_guard=True,
        enable_floating_gap_restore=True,
        enable_landing_color_guard=True,
        enable_override_color_guard=True,
        enable_ojama_column_stack_fix=True,
        enable_next_history_starvation_fix=True,
        enable_chain_tracker=True,
    )


@dataclass
class BranchRow:
    frame_idx: int
    t_sec: float
    dt_from_anchor: float
    is_match_active: bool
    match_end_locked: bool  # PipelineResult 公開値 (正)
    match_end_last_detected_t: "float | None"  # 内部タイマー (診断用)
    p1_state: str
    p2_state: str
    chain_in_progress: bool
    score_zero_both: bool
    score_actively_moving: bool
    hard_match_off: bool
    effective_hard_off: bool
    score_1p: "int | None"
    score_2p: "int | None"


def run_production_equivalent(
    cap: cv2.VideoCapture, fps: float, force_in_match: bool,
) -> "list[BranchRow]":
    """本番相当パイプラインを連続処理し、hard_match_off 系分岐値を再構成する。

    branch値の再構成は pipeline 内部 private attribute を **読み出すだけ**
    (ロジック再実装ではない) — self._sm_1p/2p.context.state,
    self._recent_scores_1p/2p, self._is_score_actively_moving(),
    self._score_zero_detector.detect(frame) (同一インスタンス)。
    """
    pipeline = build_pipeline(force_in_match)
    start_sec = max(0.0, GAME_START_SEC - WARMUP_MARGIN_SEC)
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    rows: "list[BranchRow]" = []
    frame_idx = start_frame
    end_frame = int(WINDOW_END_SEC * fps)
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t_sec = frame_idx / fps
        res = pipeline.update(frame_idx, t_sec, frame)

        if t_sec >= WINDOW_START_SEC:
            p1_state = res.p1.state
            p2_state = res.p2.state
            chain_in_progress = (
                p1_state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE)
                or p2_state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE)
            )
            score_actively_moving = (
                pipeline._is_score_actively_moving(pipeline._recent_scores_1p)
                or pipeline._is_score_actively_moving(pipeline._recent_scores_2p)
            )
            score_zero_both = False
            if pipeline._score_zero_detector is not None:
                try:
                    sz = pipeline._score_zero_detector.detect(frame)
                    score_zero_both = bool(sz.both_zero)
                except Exception:
                    pass
            hard_match_off = score_zero_both or res.match_end_locked
            effective_hard_off = (
                hard_match_off and not score_actively_moving
                and not chain_in_progress
            )
            rows.append(BranchRow(
                frame_idx=frame_idx, t_sec=round(t_sec, 4),
                dt_from_anchor=round(t_sec - ANCHOR_T_SEC, 4),
                is_match_active=bool(res.is_match_active),
                match_end_locked=bool(res.match_end_locked),
                match_end_last_detected_t=(
                    pipeline._match_end_detector.last_detected_t
                    if pipeline._match_end_detector is not None else None
                ),
                p1_state=getattr(p1_state, "name", str(p1_state)),
                p2_state=getattr(p2_state, "name", str(p2_state)),
                chain_in_progress=chain_in_progress,
                score_zero_both=score_zero_both,
                score_actively_moving=score_actively_moving,
                hard_match_off=hard_match_off,
                effective_hard_off=effective_hard_off,
                score_1p=pipeline._recent_scores_1p[-1] if pipeline._recent_scores_1p else None,
                score_2p=pipeline._recent_scores_2p[-1] if pipeline._recent_scores_2p else None,
            ))
        frame_idx += 1
    return rows


# =============================================================================
# 3. 証拠フレーム保存
# =============================================================================


def save_evidence_frames(cap: cv2.VideoCapture, fps: float) -> None:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    # 演出開始直前〜lockdown想定終了まで 1 秒おき
    for t in np.arange(WINDOW_START_SEC, WINDOW_END_SEC + 0.001, 0.5):
        f_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        vis = frame.copy()
        for region, color in ((DEFAULT_P1_REGION, (0, 255, 0)), (DEFAULT_P2_REGION, (0, 255, 255))):
            cv2.rectangle(
                vis, (region.x, region.y),
                (region.x + region.width, region.y + region.height), color, 2,
            )
        from src.match_end_detector import SEARCH_P1, SEARCH_P2
        for region, color in ((SEARCH_P1, (255, 0, 0)), (SEARCH_P2, (0, 0, 255))):
            sx, sy, sw, sh = region
            cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), color, 2)
        fname = f"f{f_idx}_t{t:.1f}_dt{t - ANCHOR_T_SEC:+.2f}.png"
        cv2.imwrite(str(FRAMES_DIR / fname), vis)


# =============================================================================
# main
# =============================================================================


def main() -> None:
    video_path = VIDEO_DIR / f"video_{VIDEO}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[{VIDEO}] fps={fps:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] MatchEndDetector.detect() 単独計装...")
    detect_rows = run_detect_only(cap, fps)
    (OUT_DIR / "detect_only.json").write_text(
        json.dumps([asdict(r) for r in detect_rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    max_score = max((r.score for r in detect_rows), default=-1.0)
    n_detected = sum(1 for r in detect_rows if r.detected)
    print(
        f"  score最大={max_score:.4f} (閾値{DEFAULT_NCC_THRESHOLD}) "
        f"検出フレーム数={n_detected}/{len(detect_rows)}"
    )

    print("[2/4] 本番相当連続処理 (force_in_match=True、実際の本番設定)...")
    rows_true = run_production_equivalent(cap, fps, force_in_match=True)
    (OUT_DIR / "branches_force_in_match_true.json").write_text(
        json.dumps([asdict(r) for r in rows_true], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[3/4] 本番相当連続処理 (force_in_match=False、発注時想定値)...")
    rows_false = run_production_equivalent(cap, fps, force_in_match=False)
    (OUT_DIR / "branches_force_in_match_false.json").write_text(
        json.dumps([asdict(r) for r in rows_false], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[4/4] 証拠フレーム保存...")
    save_evidence_frames(cap, fps)
    cap.release()

    # --- 簡易サマリ (どちらの構成でも is_active が False になった最初の時刻) ---
    def first_false(rows: "list[BranchRow]") -> "BranchRow | None":
        for r in rows:
            if not r.is_match_active:
                return r
        return None

    ff_true = first_false(rows_true)
    ff_false = first_false(rows_false)
    summary = {
        "video": VIDEO, "fps": fps,
        "anchor_frame_idx": ANCHOR_FRAME_IDX, "anchor_t_sec": ANCHOR_T_SEC,
        "detect_only_max_score": max_score,
        "detect_only_threshold": DEFAULT_NCC_THRESHOLD,
        "detect_only_n_detected": n_detected,
        "detect_only_n_frames": len(detect_rows),
        "force_in_match_true_first_is_active_false": (
            asdict(ff_true) if ff_true else None
        ),
        "force_in_match_false_first_is_active_false": (
            asdict(ff_false) if ff_false else None
        ),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
