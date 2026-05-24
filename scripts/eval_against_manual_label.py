"""手動 label と各 viz 設定の認識結果を比較して accuracy 算出.

使い方:
    python scripts/eval_against_manual_label.py \
        --video data/evaluation_videos/v89_match3_95s.mp4 \
        --frame 660 \
        --gt data/evaluation_videos/v89_compare/manual_label_v89_22s.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.recognition_pipeline import RecognitionPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--cnn-model", type=Path,
                    default=Path("models/cnn_phase_b_finetuned.pt"))
    return p.parse_args()


def load_gt(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def evaluate_setting(
    label: str, video: Path, frame_idx: int, cnn_model: Path,
    hsv_state: Path | None, gt: dict,
) -> dict:
    pipe = RecognitionPipeline.load_default(
        cnn_model_path=cnn_model, force_in_match=True,
    )
    if hsv_state is not None:
        with hsv_state.open() as f:
            state = json.load(f)
        ranges = {
            int(k): tuple(int(x) for x in v)
            for k, v in state["per_video_ranges"].items()
        }
        from src.hybrid_classifier import HybridClassifier
        hc = pipe._reader._classifier
        if isinstance(hc, HybridClassifier) and ranges:
            hc._hsv.set_color_ranges_from_simple(ranges)
            if pipe._online_hsv is not None:
                pipe._online_hsv_injected = True
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, fr = cap.read(); cap.release()
    if not ok:
        return {"error": "frame read failed"}
    if fr.shape[:2] != (1080, 1920):
        fr = cv2.resize(fr, (1920, 1080))
    # state machine 走らせて confirmed_board 取得 (= 60 frame 助走 + 対象 frame)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 60))
    for fi in range(max(0, frame_idx - 60), frame_idx + 1):
        ok, fr = cap.read()
        if not ok: break
        if fr.shape[:2] != (1080, 1920):
            fr = cv2.resize(fr, (1920, 1080))
        result = pipe.update(fi, fi / fps, fr)
    cap.release()
    pred_1p = result.p1.confirmed_board
    pred_2p = result.p2.confirmed_board
    if pred_1p is None or pred_2p is None:
        return {"error": "confirmed_board None"}
    # accuracy
    correct_1p = correct_2p = total_1p = total_2p = 0
    for r in range(13):
        for c in range(6):
            gt_v_1p = gt["1P"][r][c]
            gt_v_2p = gt["2P"][r][c]
            pred_v_1p = int(pred_1p.get(r, c))
            pred_v_2p = int(pred_2p.get(r, c))
            if pred_v_1p == gt_v_1p:
                correct_1p += 1
            total_1p += 1
            if pred_v_2p == gt_v_2p:
                correct_2p += 1
            total_2p += 1
    return {
        "label": label,
        "1P_acc": correct_1p / total_1p,
        "2P_acc": correct_2p / total_2p,
        "total_acc": (correct_1p + correct_2p) / (total_1p + total_2p),
        "correct_total": correct_1p + correct_2p,
        "total_cells": total_1p + total_2p,
    }


def main() -> None:
    args = parse_args()
    gt = load_gt(args.gt)
    settings = [
        ("default (no DB)", None),
        ("v89 DB", Path("data/per_video_hsv_ranges/v89.json")),
        ("merged_default", Path("data/per_video_hsv_ranges/_merged_default.json")),
    ]
    print(f"=== Manual label evaluation @ frame {args.frame} ({args.gt}) ===")
    print(f"{'setting':>20} | {'1P acc':>7} | {'2P acc':>7} | {'total':>7}")
    print("-" * 60)
    for label, hsv_state in settings:
        r = evaluate_setting(
            label, args.video, args.frame, args.cnn_model, hsv_state, gt,
        )
        if "error" in r:
            print(f"{label:>20} | ERROR: {r['error']}")
        else:
            print(
                f"{r['label']:>20} | {r['1P_acc']:.3f} | {r['2P_acc']:.3f} | "
                f"{r['total_acc']:.3f} ({r['correct_total']}/{r['total_cells']})",
            )


if __name__ == "__main__":
    main()
