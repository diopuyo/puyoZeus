"""
第2プレイリスト (S級リーグ) を並行処理。
index=1 (KmRmSHe7DPg) と deleted を除外。
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.calibration import CalibratedConfig
from src.patch_extraction import PatchExtractor, PatchDataset, balance_dataset
from src.board import COLOR_EMPTY

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLsjREVssD8bZlPdMhq0kyaVKpSK-Dl7Jk"
EXCLUDED_IDS = {"KmRmSHe7DPg"}  # user 指定除外
WORK_DIR = Path("data/frames/bulk2")
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path("data/training")
SAMPLE_INTERVAL_SEC = 8.0


def has_eyes(patch: np.ndarray) -> bool:
    h, w = patch.shape[:2]
    mh, mw = int(h*0.15), int(w*0.15)
    c = patch[mh:h-mh, mw:w-mw]
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    d = (g < 70).astype(np.uint8) * 255
    n, _, s, _ = cv2.connectedComponentsWithStats(d, connectivity=4)
    ta = c.shape[0] * c.shape[1]
    return sum(
        1 for i in range(1, n)
        if 2 <= s[i, cv2.CC_STAT_AREA] <= ta*0.12
    ) >= 2


def playlist_videos() -> list[tuple[int, str]]:
    result = subprocess.run([
        "yt-dlp", "--flat-playlist", "--print",
        "%(playlist_index)s\t%(id)s\t%(duration)s",
        PLAYLIST_URL,
    ], capture_output=True, text=True, check=True)
    items = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        idx, vid_id, dur = parts[0], parts[1], parts[2]
        if vid_id in EXCLUDED_IDS:
            continue
        if dur == "NA" or not dur.isdigit():
            continue
        if int(dur) < 500:  # short clips skip
            continue
        items.append((int(idx), f"https://www.youtube.com/watch?v={vid_id}"))
    return items


def download_video(url: str, out_path: Path) -> bool:
    if out_path.exists() and out_path.stat().st_size > 10_000_000:
        return True
    cmd = [
        "yt-dlp",
        "-f",
        "bestvideo[ext=mp4][vcodec^=avc1][height<=720]/"
        "bestvideo[ext=mp4][height<=720]",
        "-o", str(out_path),
        "--no-playlist", "--quiet",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and out_path.exists()


def extract(extractor: PatchExtractor, video_path: Path) -> PatchDataset:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return PatchDataset(
            patches=np.zeros((0, 32, 32, 3), dtype=np.uint8),
            labels=np.zeros(0, dtype=np.int64),
        )
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * SAMPLE_INTERVAL_SEC)))
    patches, labels = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if frame.shape[0] != 1080:
                frame = cv2.resize(frame, (1920, 1080))
            ps, ls = extractor.extract_from_frame(frame)
            patches.extend(ps)
            labels.extend(ls)
        idx += 1
    cap.release()
    if not patches:
        return PatchDataset(
            patches=np.zeros((0, 32, 32, 3), dtype=np.uint8),
            labels=np.zeros(0, dtype=np.int64),
        )
    return PatchDataset(
        patches=np.stack(patches),
        labels=np.array(labels, dtype=np.int64),
    )


def save_incremental(
    all_patches: list[np.ndarray], all_labels: list[int], idx: int,
) -> None:
    if not all_patches:
        return
    patches = np.concatenate(all_patches)
    labels = np.array(all_labels, dtype=np.int64)
    keep = np.zeros(len(labels), dtype=bool)
    for i in range(len(labels)):
        e = has_eyes(patches[i])
        keep[i] = (not e) if labels[i] == COLOR_EMPTY else e
    ds = PatchDataset(patches=patches[keep], labels=labels[keep])
    ds.stats.patches_total = len(ds.labels)
    unique, counts = np.unique(ds.labels, return_counts=True)
    ds.stats.per_class_count = {int(k): int(v) for k, v in zip(unique, counts)}
    balanced = balance_dataset(ds, empty_ratio_cap=0.35)
    out = OUT_DIR / f"bulk2_patches_balanced_through_v{idx:02d}.npz"
    balanced.save(out)
    print(f"  保存: {out.name} ({balanced.stats.patches_total})")


def main() -> None:
    config = CalibratedConfig.load("models/calibration_video01.json")
    extractor = PatchExtractor(config=config)
    videos = playlist_videos()
    print(f"第2プレイリスト対象: {len(videos)} 動画")

    all_patches: list[np.ndarray] = []
    all_labels: list[int] = []
    processed = 0
    for idx, url in videos:
        video_path = WORK_DIR / f"video_{idx:02d}.mp4"
        start = time.time()
        print(f"[{idx}] DL: {url}")
        if not download_video(url, video_path):
            print("  失敗")
            continue
        print(f"  DL完了: {video_path.stat().st_size // 1024 // 1024}MB "
              f"({time.time()-start:.0f}s)")
        start = time.time()
        ds = extract(extractor, video_path)
        print(f"  抽出: {len(ds.labels)} ({time.time()-start:.0f}s)")
        if len(ds.labels) > 0:
            all_patches.append(ds.patches)
            all_labels.extend(ds.labels.tolist())
        video_path.unlink(missing_ok=True)
        processed += 1
        if processed % 3 == 0:
            save_incremental(all_patches, all_labels, idx)
    save_incremental(all_patches, all_labels, videos[-1][0] if videos else 0)
    print(f"完了: {processed} 動画")


if __name__ == "__main__":
    main()
