"""W8-D: NextDetector 専用 CNN 訓練ラベル収集。

NextDetector は現状、汎用の CnnPatchClassifier (8x8 board cell 用) を使用。
Next pair パッチは 75x75 で大きく、背景・ぷよ形状もボードセルと異なるため
専用 CNN の方が有利のはず。

ラベル収集戦略 (高信頼自動ラベリング):
  1. StableNextDetector で 3 連続同色一致のみ採用
  2. P1/P2 両側で next_pair および dnext_pair が一致 (= 同じツモを見ている)
  3. 上記 2 条件を満たすフレームから、4 ROI × 2 sides = 8 パッチを採取
  4. 採取した patch (75x75 BGR uint8) に StableDetector の合議色をラベル付与

入力:
    - data/frames/video_{01..19}.mp4
    - models/cnn_phase_u_v10.pt (label 用 base classifier)

出力:
    - data/training_phase_u/next_pair_labels.npz  (patches, labels, side, slot)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2
import numpy as np

from src.next_detector import (
    NextDetector,
    ROI_1P_NEXT_TOP, ROI_1P_NEXT_BOT,
    ROI_1P_DNEXT_TOP, ROI_1P_DNEXT_BOT,
    ROI_2P_NEXT_TOP, ROI_2P_NEXT_BOT,
    ROI_2P_DNEXT_TOP, ROI_2P_DNEXT_BOT,
)
from src.patch_classifier import CnnPatchClassifier
from src.patch_classifier_v2 import CnnPatchClassifierV2
from src.stable_next_detector import StableNextDetector

PATCH_SIZE = 32  # next pair 用 (board cell の 8/16 より大きく)
STABILITY_WINDOW = 3
SAMPLE_INTERVAL_SEC = 0.6  # 0.6 秒ごとに 1 サンプル候補


def _load_classifier(model_path: Path):
    """v10 (16x16) なら CnnPatchClassifierV2、それ以外は CnnPatchClassifier。"""
    import torch
    state = torch.load(str(model_path), map_location="cpu", weights_only=True)
    keys = list(state.keys())
    is_v10 = any("conv1.weight" in k for k in keys)
    if is_v10:
        cls = CnnPatchClassifierV2()
        cls.load(model_path)
        return cls
    return CnnPatchClassifier.load(model_path)


def _extract_patch(
    frame: np.ndarray, roi: tuple[int, int, int, int],
) -> np.ndarray:
    y1, y2, x1, x2 = roi
    h, w = frame.shape[:2]
    y1, y2 = max(0, y1), min(h, y2)
    x1, x2 = max(0, x1), min(w, x2)
    p = frame[y1:y2, x1:x2]
    if p.size == 0:
        return np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    return cv2.resize(
        p, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA,
    )


def collect_one_video(
    video_path: Path,
    cnn_model_path: Path,
    sample_interval: float,
    max_samples: int | None = None,
) -> tuple[list[np.ndarray], list[int], list[str], list[str]]:
    """1 動画から (patch, label, side, slot) を収集。"""
    cnn = _load_classifier(cnn_model_path)
    detector = NextDetector(classifier=cnn)
    stable = StableNextDetector(detector, stability_window=STABILITY_WINDOW)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], [], [], []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    patches: list[np.ndarray] = []
    labels: list[int] = []
    sides: list[str] = []
    slots: list[str] = []

    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            t += sample_interval
            continue
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        try:
            sd = stable.detect_both(frame)
        except Exception:
            t += sample_interval
            continue

        # 採用条件: 1P/2P 両側に stable がある AND next_pair と dnext_pair が一致
        if (
            sd.p1_next is None or sd.p2_next is None
            or sd.p1_dnext is None or sd.p2_dnext is None
        ):
            t += sample_interval
            continue
        if sd.p1_next != sd.p2_next:
            t += sample_interval
            continue
        if sd.p1_dnext != sd.p2_dnext:
            t += sample_interval
            continue

        # 採用色は通常 5 色のみ (RED/BLUE/GREEN/YELLOW/PURPLE)
        # next pair に EMPTY/OJAMA が出る場合は試合間/メニュー画面の誤検出
        valid_colors = {1, 2, 3, 4, 5}
        next_colors = {*sd.p1_next, *sd.p2_next, *sd.p1_dnext, *sd.p2_dnext}
        if not next_colors.issubset(valid_colors):
            t += sample_interval
            continue

        # 8 ROI を採取 (1P/2P × 4 slots)
        rois = [
            ("1P", "next_top",  ROI_1P_NEXT_TOP,  sd.p1_next[0]),
            ("1P", "next_bot",  ROI_1P_NEXT_BOT,  sd.p1_next[1]),
            ("1P", "dnext_top", ROI_1P_DNEXT_TOP, sd.p1_dnext[0]),
            ("1P", "dnext_bot", ROI_1P_DNEXT_BOT, sd.p1_dnext[1]),
            ("2P", "next_top",  ROI_2P_NEXT_TOP,  sd.p2_next[0]),
            ("2P", "next_bot",  ROI_2P_NEXT_BOT,  sd.p2_next[1]),
            ("2P", "dnext_top", ROI_2P_DNEXT_TOP, sd.p2_dnext[0]),
            ("2P", "dnext_bot", ROI_2P_DNEXT_BOT, sd.p2_dnext[1]),
        ]
        for side, slot, roi, code in rois:
            patch = _extract_patch(frame, roi)
            patches.append(patch)
            labels.append(int(code))
            sides.append(side)
            slots.append(slot)

        if max_samples and len(patches) >= max_samples:
            break
        t += sample_interval

    cap.release()
    return patches, labels, sides, slots


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames-dir", default="data/frames",
    )
    parser.add_argument(
        "--cnn-model", default="models/cnn_phase_u_v10.pt",
    )
    parser.add_argument(
        "--out-npz",
        default="data/training_phase_u/next_pair_labels.npz",
    )
    parser.add_argument(
        "--videos", nargs="*",
        help="絞り込み (例: video_01 video_05); 未指定なら全 video_*.mp4",
    )
    parser.add_argument(
        "--sample-interval", type=float, default=SAMPLE_INTERVAL_SEC,
    )
    parser.add_argument(
        "--max-per-video", type=int, default=2000,
        help="各動画で採取する patch 上限",
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    if args.videos:
        video_paths = [
            frames_dir / f"{v}.mp4" if not v.endswith(".mp4")
            else frames_dir / v
            for v in args.videos
        ]
    else:
        video_paths = sorted(frames_dir.glob("video_[0-9][0-9].mp4"))

    print(f"target videos: {len(video_paths)}")

    all_patches: list[np.ndarray] = []
    all_labels: list[int] = []
    all_sides: list[str] = []
    all_slots: list[str] = []
    all_videos: list[str] = []

    for vp in video_paths:
        if not vp.exists():
            print(f"skip (missing): {vp.name}")
            continue
        print(f"\n--- {vp.name} ---")
        patches, labels, sides, slots = collect_one_video(
            vp, Path(args.cnn_model),
            args.sample_interval, args.max_per_video,
        )
        print(f"  collected: {len(patches)} patches")
        if patches:
            all_patches.extend(patches)
            all_labels.extend(labels)
            all_sides.extend(sides)
            all_slots.extend(slots)
            all_videos.extend([vp.stem] * len(patches))

    if not all_patches:
        print("no samples collected")
        return 1

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        patches=np.array(all_patches, dtype=np.uint8),
        labels=np.array(all_labels, dtype=np.int32),
        sides=np.array(all_sides),
        slots=np.array(all_slots),
        videos=np.array(all_videos),
    )
    print(f"\nsaved: {to_windows_path(out_path)} ({len(all_patches)} samples)")
    unique, counts = np.unique(np.array(all_labels), return_counts=True)
    for c, n in zip(unique, counts):
        print(f"  code={c}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
