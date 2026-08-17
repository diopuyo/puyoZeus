"""境界マルチシグナル (b-1)+(b-2) 実装検証・③試合外4件 (2026-08-18)。

docs/BOUNDARY_MULTISIGNAL_DESIGN_2026-08-17.md §3 の実装で、一般分布35盤面の
③試合外カテゴリ (D判定、data/verify/diag_general_chain_contamination_2026-08-17/
report.md) のうち c21 (030) 以外の4件 (022/023_c18, 024/025_c20) が
is_match_active=False として正しく除外されるようになったかを実写確認する。

本体コード (src/) は変更しない。RecognitionPipeline を外部から直接
呼び計装するだけ。

## 使い方 (WSL)
    PYTHONPATH=. ./venv/bin/python -m \
        scripts._verify_boundary_multisignal_dcategory_2026-08-18
"""
from __future__ import annotations

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
TAIL_MARGIN_SEC: float = 3.0


@dataclass(frozen=True)
class DCase:
    sheet: str
    video: str
    side: str
    anchor_t_sec: float
    game_start_sec: float


# report.md / classification_corrected_2026-08-17.json より (030=c21 は
# 既に別スクリプトで確認済のため対象外)。
CASES: "list[DCase]" = [
    DCase("022_c18_1P_f56752", "c18", "1P", 1891.733, 1850.0),
    DCase("023_c18_2P_f56752", "c18", "2P", 1891.733, 1850.0),
    DCase("024_c20_1P_f49838", "c20", "1P", 830.633, 782.0),
    DCase("025_c20_2P_f49836", "c20", "2P", 830.6, 782.0),
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


def run_case(case: DCase) -> dict:
    video_path = VIDEO_DIR / f"video_{case.video}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_sec = max(0.0, case.game_start_sec - WARMUP_MARGIN_SEC)
    end_sec = case.anchor_t_sec + TAIL_MARGIN_SEC
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pipeline = build_pipeline()
    frame_idx = start_frame
    end_frame = int(end_sec * fps)
    anchor_frame = int(round(case.anchor_t_sec * fps))
    result_at_anchor: dict = {}
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t_sec = frame_idx / fps
        res = pipeline.update(frame_idx, t_sec, frame)
        if abs(frame_idx - anchor_frame) <= 1:
            result_at_anchor = {
                "frame_idx": frame_idx, "t_sec": round(t_sec, 4),
                "is_match_active": bool(res.is_match_active),
                "match_end_locked": bool(res.match_end_locked),
                "confirmed_board_is_none": (
                    (res.p1 if case.side == "1P" else res.p2).confirmed_board
                    is None
                ),
            }
    cap.release()
    return {
        "sheet": case.sheet, "video": case.video, "side": case.side,
        "anchor_t_sec": case.anchor_t_sec,
        "result_near_anchor": result_at_anchor,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for case in CASES:
        print(f"[{case.sheet}] 処理中...")
        r = run_case(case)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    (OUT_DIR / "dcategory_verify.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
