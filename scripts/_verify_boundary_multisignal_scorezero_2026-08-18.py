"""(b-2) 解除信号 score_zero_both 持続方式の実写検証 (2026-08-18、アーキ発注)。

docs/BOUNDARY_MULTISIGNAL_DESIGN_2026-08-17.md §3(b-2) の解除信号置換後、
③試合外5件 (022/023_c18, 024/025_c20, 030_c21) の窓で、latch状態+
score_zero_both + is_match_active を毎フレームCSV計装し、対戦カード紹介
区間で latch=True (is_match_active=False) が維持されることを確認する。

本体コード (src/) は変更しない。RecognitionPipeline を外部から直接
呼び計装するだけ。

## 使い方 (WSL)
    PYTHONPATH=. ./venv/bin/python -m \
        scripts._verify_boundary_multisignal_scorezero_2026-08-18
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO_DIR: Path = Path.home() / "frames"
OUT_DIR: Path = Path(
    "/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
    "/data/verify/boundary_impl_verify_2026-08-18"
)
WARMUP_MARGIN_SEC: float = 2.0
TAIL_MARGIN_SEC: float = 5.0


@dataclass(frozen=True)
class AnchorCase:
    sheet: str
    video: str
    side: str
    anchor_t_sec: float
    game_start_sec: float


CASES: "list[AnchorCase]" = [
    AnchorCase("022_c18_1P_f56752", "c18", "1P", 1891.733, 1850.0),
    AnchorCase("023_c18_2P_f56752", "c18", "2P", 1891.733, 1850.0),
    AnchorCase("024_c20_1P_f49838", "c20", "1P", 830.633, 782.0),
    AnchorCase("025_c20_2P_f49836", "c20", "2P", 830.6, 782.0),
    AnchorCase("030_c21_2P_f57548", "c21", "2P", 959.133, 858.0),
]


def build_pipeline() -> RecognitionPipeline:
    return RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        load_next_detector=True,
        temporal_smoothing=1,
        force_in_match=True,
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
        enable_match_end_persist_override=True,
        enable_post_match_lockdown_latch=True,
    )


def run_case(case: AnchorCase) -> dict:
    video_path = VIDEO_DIR / f"video_{case.video}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_sec = max(0.0, case.game_start_sec - WARMUP_MARGIN_SEC)
    end_sec = case.anchor_t_sec + TAIL_MARGIN_SEC
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pipeline = build_pipeline()
    frame_idx = start_frame
    anchor_frame = int(round(case.anchor_t_sec * fps))
    rows: "list[dict]" = []
    result_at_anchor: dict = {}
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t_sec = frame_idx / fps
        res = pipeline.update(frame_idx, t_sec, frame)
        if t_sec >= case.anchor_t_sec - WARMUP_MARGIN_SEC - 10.0:
            sz = getattr(pipeline, "_score_zero_detector", None)
            score_zero_both = None
            if sz is not None:
                try:
                    score_zero_both = bool(sz.detect(frame).both_zero)
                except Exception:
                    score_zero_both = None
            row = {
                "sheet": case.sheet, "video": case.video, "frame_idx": frame_idx,
                "t_sec": round(t_sec, 4),
                "dt_from_anchor": round(t_sec - case.anchor_t_sec, 4),
                "is_match_active": bool(res.is_match_active),
                "match_end_locked": bool(res.match_end_locked),
                "post_match_lockdown_active": bool(
                    getattr(pipeline, "_post_match_lockdown_active", False),
                ),
                "score_zero_both": score_zero_both,
            }
            rows.append(row)
            if abs(frame_idx - anchor_frame) <= 1:
                result_at_anchor = row
        frame_idx += 1
    cap.release()
    return {"sheet": case.sheet, "rows": rows, "result_at_anchor": result_at_anchor}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: "list[dict]" = []
    summaries: "list[dict]" = []
    excluded_count = 0
    for case in CASES:
        print(f"[{case.sheet}] 処理中...")
        result = run_case(case)
        all_rows.extend(result["rows"])
        anchor_row = result["result_at_anchor"]
        excluded = bool(anchor_row) and not anchor_row.get("is_match_active", True)
        if excluded:
            excluded_count += 1
        summaries.append({
            "sheet": case.sheet, "anchor_row": anchor_row, "excluded": excluded,
        })
        print(json.dumps(summaries[-1], ensure_ascii=False, indent=2))

    if all_rows:
        out_csv = OUT_DIR / "scorezero_latch_verify.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[out] {out_csv} ({len(all_rows)} 行)")

    out_json = OUT_DIR / "scorezero_latch_verify_summary.json"
    out_json.write_text(
        json.dumps(
            {"excluded_count": excluded_count, "total": len(CASES),
             "summaries": summaries},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[result] 遮断件数: {excluded_count}/{len(CASES)}")


if __name__ == "__main__":
    main()
