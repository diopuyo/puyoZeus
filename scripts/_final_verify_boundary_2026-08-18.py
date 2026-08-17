"""境界実装の仕上げ 最終検証 (2026-08-18、アーキ発注)。

1. (b-1)+(b-2)+enable_result_screen_hardening ON で③試合外5件
   (022/023_c18, 024/025_c20, 030_c21) の RT側遮断数を実測する。
2. 実測した match_end_locked/post_match_lockdown_active 値から合成npzを
   作り、build_labeled_win_from_npz.py --exclude-match-end-locked
   フィルタが実際に5/5全て除外することを実証する。

本体コード (src/) は変更しない。RecognitionPipeline / build_labeled_win_
from_npz.convert_one_npz を外部から直接呼び計装するだけ。

## 使い方 (WSL)
    PYTHONPATH=. ./venv/bin/python -m scripts._final_verify_boundary_2026-08-18
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

import scripts.build_labeled_win_from_npz as blwn  # noqa: E402
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
    """(b-1)+(b-2)+enable_result_screen_hardening 全ON構成。"""
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
        enable_result_screen_hardening=True,
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
                "post_match_lockdown_active": bool(
                    getattr(pipeline, "_post_match_lockdown_active", False),
                ),
            }
        frame_idx += 1
    cap.release()
    return {
        "sheet": case.sheet, "video": case.video, "side": case.side,
        "anchor_t_sec": case.anchor_t_sec,
        "result_at_anchor": result_at_anchor,
    }


def build_synthetic_npz_and_filter(results: "list[dict]") -> "tuple[Path, int]":
    """実測した match_end_locked/post_match_lockdown_active 値から合成npzを
    作り、--exclude-match-end-locked フィルタ相当を適用して除外数を返す。
    """
    from src.board import BOARD_COLS, BOARD_ROWS

    n = len(results)
    grids = np.zeros((n, BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    grids[:, -1, :] = 1  # 各盤面に1色ぷよを置く (count_puyos()>=1 を満たす)
    video_id = np.array([r["video"] for r in results])
    side = np.array(["1P", "2P"] * (n // 2 + 1))[:n]
    t_sec = np.array(
        [r["result_at_anchor"].get("t_sec", 0.0) for r in results], dtype=np.float32,
    )
    game_idx = np.arange(n, dtype=np.int32)
    frame_idx = np.array(
        [r["result_at_anchor"].get("frame_idx", 0) for r in results], dtype=np.int32,
    )
    won = np.full(n, np.nan, dtype=np.float32)
    score = np.full(n, -1, dtype=np.int32)
    match_end_locked = np.array(
        [int(r["result_at_anchor"].get("match_end_locked", False)) for r in results],
        dtype=np.int8,
    )
    post_match_lockdown_active = np.array(
        [
            int(r["result_at_anchor"].get("post_match_lockdown_active", False))
            for r in results
        ],
        dtype=np.int8,
    )
    npz_path = OUT_DIR / "final_verify_synthetic_5anchors.npz"
    np.savez_compressed(
        str(npz_path), grids=grids, video_id=video_id, side=side, t_sec=t_sec,
        game_idx=game_idx, frame_idx=frame_idx, won=won, score=score,
        match_end_locked=match_end_locked,
        post_match_lockdown_active=post_match_lockdown_active,
    )
    registry = blwn._resolve_indicator_registry("light")
    rows_off = blwn.convert_one_npz(npz_path, registry, exclude_match_end_locked=False)
    rows_on = blwn.convert_one_npz(npz_path, registry, exclude_match_end_locked=True)
    assert len(rows_off) == n, "フィルタ無効時は全件残るはず"
    n_excluded = n - len(rows_on)
    return npz_path, n_excluded


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for case in CASES:
        print(f"[{case.sheet}] RT処理中 (全フラグON)...")
        r = run_case(case)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=2))

    rt_blocked = sum(
        1 for r in results
        if r["result_at_anchor"] and not r["result_at_anchor"]["is_match_active"]
    )

    npz_path, n_excluded_by_filter = build_synthetic_npz_and_filter(results)

    summary = {
        "rt_blocked_count": rt_blocked,
        "rt_total": len(CASES),
        "column_filter_excluded_count": n_excluded_by_filter,
        "column_filter_total": len(CASES),
        "per_case": results,
        "synthetic_npz": str(npz_path),
    }
    out_json = OUT_DIR / "final_verify_summary.json"
    out_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[out] {out_json}")


if __name__ == "__main__":
    main()
