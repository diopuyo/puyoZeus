"""一般分布35盤面 STABLEスナップショットの「連鎖中混入」計測 (2026-08-17)。

## 目的
`data/verify/board_labels_general_2026-08-17/labels.tsv` の35アンカーが、
STABLE確定盤面として記録された瞬間に「実際には連鎖/おじゃま落下/試合外の
最中だったか」を、user目視に頼らず機械シグナルで全数判定する。

本体コード (worktree src/) は変更しない。RecognitionPipeline.load_default
の内部 state / staticmethod / ScoreOcr を外部から呼び計装するだけ。

## 重要な制約 (計測の妥当性への影響、必ずレポートに明記)
worktree HEAD (a55b919, 2026-08-11) は本ラベル収集時点の本番構成 F
(2026-08-17収集、labels.tsv base_config列) より古く、
enable_ojama_fall_placement_override / enable_patch_fp_hsv_guard /
enable_floating_gap_restore / enable_landing_color_guard /
enable_override_color_guard / enable_ojama_column_stack_fix /
enable_next_history_starvation_fix 等の後発フラグを RecognitionPipeline.
load_default が受け付けない (worktree はコーダが編集中の main src/ を
import してはならない制約のため、この差は解消できない)。
そのため本計装が出す BoardState (STABLE/CHAIN/OJAMA_FALL 等) は
「構成Fが実際に何と判定したか」の再現ではなく **参考・二次シグナル**
に留める。判定の一次根拠は本体コードの状態機械に依存しない独立シグナル
(生ピクセル差分 / score欄の掛け算式表示 / score値の増加) とする。

## 使い方 (WSL、venvはメイン側)
    PYTHONPATH=. /mnt/.../venv/bin/python -m \
        scripts._diag_general_chain_contamination_2026-08-17 --all
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from src.score_ocr import ScoreOcr, _crop_score_roi, compute_score_roi_ink_ratio  # noqa: E402

# =============================================================================
# 定数
# =============================================================================

MAIN_ROOT: Path = Path("/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer")
LABELS_DIR: Path = MAIN_ROOT / "data" / "verify" / "board_labels_general_2026-08-17"
LABELS_TSV: Path = LABELS_DIR / "labels.tsv"
ANCHOR_PLAN_TSV: Path = LABELS_DIR / "anchor_plan.tsv"
VIDEO_DIR: Path = Path.home() / "frames"
OUT_DIR: Path = MAIN_ROOT / "data" / "verify" / "diag_general_chain_contamination_2026-08-17"
FRAMES_DIR: Path = OUT_DIR / "frames"

DENSE_WINDOW_SEC: float = 3.5  # アンカー ± この秒数を密ログ
LOOKBACK_MARGIN_SEC: float = 2.0  # game_start からの安全マージン (state machine 用)
TAIL_MARGIN_SEC: float = 4.0  # 最終アンカー後の余白

CHAIN_FORMULA_INK_RATIO_MIN: float = 0.1  # src.score_ocr.SCORE_ROI_INK_RATIO_MIN と同値


@dataclass(frozen=True)
class LabelAnchor:
    sheet: str
    video: str
    side: str
    frame_idx: int
    t_sec: float
    user_note: str  # wrong_cells 列の生値 (skip / ok / rXcY=Z,... など)


@dataclass
class VideoJob:
    video: str
    game_start: float
    game_end: float
    anchors: "list[LabelAnchor]" = field(default_factory=list)


# =============================================================================
# 1. labels.tsv / anchor_plan.tsv 読込
# =============================================================================


def load_anchors() -> "list[LabelAnchor]":
    lines = [
        l for l in LABELS_TSV.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#")
    ]
    rows = list(csv.DictReader(lines, delimiter="\t"))
    return [
        LabelAnchor(
            sheet=r["sheet"], video=r["video"], side=r["side"],
            frame_idx=int(r["frame_idx"]), t_sec=float(r["t_sec"]),
            user_note=r["wrong_cells"],
        )
        for r in rows
    ]


def load_game_bounds() -> "dict[str, tuple[float, float]]":
    """video -> (game_start, game_end) (anchor_plan.tsv、1動画1試合のみ収録)。"""
    rows = list(csv.DictReader(ANCHOR_PLAN_TSV.open(encoding="utf-8"), delimiter="\t"))
    out: "dict[str, tuple[float, float]]" = {}
    for r in rows:
        out[r["video"]] = (float(r["game_start"]), float(r["game_end"]))
    return out


def build_jobs() -> "list[VideoJob]":
    anchors = load_anchors()
    bounds = load_game_bounds()
    by_video: "dict[str, VideoJob]" = {}
    for a in anchors:
        if a.video not in by_video:
            gs, ge = bounds[a.video]
            by_video[a.video] = VideoJob(video=a.video, game_start=gs, game_end=ge)
        by_video[a.video].anchors.append(a)
    return sorted(by_video.values(), key=lambda j: j.video)


# =============================================================================
# 2. pipeline構築 (worktree が対応するフラグのみ、docstring 参照)
# =============================================================================


def build_pipeline() -> RecognitionPipeline:
    """worktree HEAD が対応する範囲で構成Fに最も近い設定 (二次シグナル用)。"""
    return RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        load_next_detector=True,
        temporal_smoothing=1,
        force_in_match=False,
        enable_effect_gate=True,
        enable_burst_guard_v2=True,
        enable_transition_merge_guard=True,
        burst_gate_open_threshold=0.954,
        enable_hidden_row_burst_guard=True,
        enable_match_transition_debounce=True,
        enable_chain_tracker=True,
    )


# =============================================================================
# 3. 独立シグナル計測 (生ピクセル差分 / score掛け算式)
# =============================================================================


def _region_for(side: str):
    return DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION


def _board_roi_gray(frame: np.ndarray, side: str) -> np.ndarray:
    r = _region_for(side)
    crop = frame[r.y: r.y + r.height, r.x: r.x + r.width]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def _column_diffs(prev_gray: np.ndarray, cur_gray: np.ndarray, n_cols: int = 6) -> "list[float]":
    """列ごとの平均絶対差分 (盤面幅を n_cols 等分)。"""
    h, w = cur_gray.shape
    col_w = w / n_cols
    diff = np.abs(cur_gray.astype(np.int16) - prev_gray.astype(np.int16))
    out = []
    for i in range(n_cols):
        x1, x2 = int(i * col_w), int((i + 1) * col_w)
        out.append(float(diff[:, x1:x2].mean()))
    return out


def _formula_detected(
    frame: np.ndarray, score_ocr: ScoreOcr, side: str, last_score: "int | None",
) -> "tuple[bool, float, int | None]":
    """機能D判定を stateless に再現する (src.recognition_pipeline._check_formula_
    detected と同一ロジック、外部から独立呼出しのため複製ではなく再利用可能な
    部分のみ ScoreOcr / compute_score_roi_ink_ratio から直接構築)。

    Returns:
        (formula_detected, ink_ratio, score_val)
    """
    score_val, _conf = score_ocr.read_side(frame, side)  # type: ignore[arg-type]
    roi = _crop_score_roi(frame, side)  # type: ignore[arg-type]
    ir = compute_score_roi_ink_ratio(roi) if roi is not None else 0.0
    if score_val is not None:
        return False, ir, score_val
    if last_score is None or last_score <= 0:
        return False, ir, score_val
    return ir > CHAIN_FORMULA_INK_RATIO_MIN, ir, score_val


# =============================================================================
# 4. 1動画分の通し処理
# =============================================================================


def process_video(job: VideoJob, save_frames: bool) -> dict:
    video_path = VIDEO_DIR / f"video_{job.video}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_sec = max(0.0, job.game_start - LOOKBACK_MARGIN_SEC)
    end_sec = max(a.t_sec for a in job.anchors) + TAIL_MARGIN_SEC
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print(f"[{job.video}] fps={fps:.3f} start={start_sec:.1f} end={end_sec:.1f} "
          f"anchors={len(job.anchors)}")

    pipeline = build_pipeline()
    score_ocr = ScoreOcr.load_default()  # 独立インスタンス (pipeline内部と別)
    prev_gray = {"1P": None, "2P": None}
    prev_score_indep = {"1P": None, "2P": None}  # 独立ScoreOcrの直前値
    dense: "dict[str, list[dict]]" = {a.sheet: [] for a in job.anchors}

    frame_idx = start_frame
    t_sec = start_sec
    n = 0
    while t_sec <= end_sec:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        res = pipeline.update(frame_idx, t_sec, frame)

        for side in ("1P", "2P"):
            gray = _board_roi_gray(frame, side)
            col_diffs = (
                _column_diffs(prev_gray[side], gray) if prev_gray[side] is not None else None
            )
            prev_gray[side] = gray
            formula, ink, score_indep = _formula_detected(
                frame, score_ocr, side, prev_score_indep[side],
            )
            score_delta_indep = (
                (score_indep - prev_score_indep[side])
                if (score_indep is not None and prev_score_indep[side] is not None)
                else None
            )
            if score_indep is not None:
                prev_score_indep[side] = score_indep

            side_res = res.p1 if side == "1P" else res.p2
            state_name = getattr(side_res.state, "name", str(side_res.state))

            for a in job.anchors:
                if a.side != side:
                    continue
                if abs(t_sec - a.t_sec) > DENSE_WINDOW_SEC:
                    continue
                dense[a.sheet].append({
                    "frame_idx": frame_idx, "t_sec": round(t_sec, 4),
                    "dt_from_anchor": round(t_sec - a.t_sec, 4),
                    "state": state_name,
                    "is_match_active": bool(res.is_match_active),
                    "col_diffs": [round(x, 3) for x in col_diffs] if col_diffs else None,
                    "diff_mean": round(float(np.mean(col_diffs)), 3) if col_diffs else None,
                    "diff_max_col": round(float(np.max(col_diffs)), 3) if col_diffs else None,
                    "n_cols_changed_gt5": (
                        int(sum(1 for x in col_diffs if x > 5.0)) if col_diffs else None
                    ),
                    "formula_detected": formula,
                    "score_roi_ink_ratio": round(ink, 4),
                    "score_indep": score_indep,
                    "score_delta_indep": score_delta_indep,
                })

        n += 1
        frame_idx += 1
        t_sec = frame_idx / fps

    cap.release()
    print(f"[{job.video}] 処理完了 n_frames={n}")

    if save_frames:
        _save_evidence_frames(video_path, fps, job)

    return {
        "video": job.video, "fps": fps, "start_sec": start_sec, "end_sec": end_sec,
        "n_frames": n,
        "anchors": [
            {
                "sheet": a.sheet, "side": a.side, "frame_idx": a.frame_idx,
                "t_sec": a.t_sec, "user_note": a.user_note,
                "dense": dense[a.sheet],
            }
            for a in job.anchors
        ],
    }


def _save_evidence_frames(video_path: Path, fps: float, job: VideoJob) -> None:
    """各アンカーの中心フレーム±3フレームを枠付きPNGとして保存する。"""
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    cap2 = cv2.VideoCapture(str(video_path))
    for a in job.anchors:
        region = _region_for(a.side)
        center_idx = int(round(a.t_sec * fps))
        for delta in range(-3, 4):
            f_idx = center_idx + delta
            if f_idx < 0:
                continue
            cap2.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = cap2.read()
            if not ok:
                continue
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            vis = frame.copy()
            cv2.rectangle(
                vis, (region.x, region.y),
                (region.x + region.width, region.y + region.height), (0, 255, 255), 2,
            )
            fname = f"{a.sheet.replace('.png', '')}_f{f_idx}_d{delta:+d}.png"
            cv2.imwrite(str(FRAMES_DIR / fname), vis)
    cap2.release()


# =============================================================================
# 5. main
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", help="単一動画のみ処理 (例: c10)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-frames", action="store_true")
    args = ap.parse_args()
    jobs = build_jobs()
    if args.video:
        jobs = [j for j in jobs if j.video == args.video]
    elif not args.all:
        ap.print_help()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        result = process_video(job, save_frames=not args.no_frames)
        out_path = OUT_DIR / f"{job.video}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{job.video}] -> {out_path}")


if __name__ == "__main__":
    main()
