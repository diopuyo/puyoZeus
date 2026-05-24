"""
プレイリスト全動画を順次処理して統合データセットを作成。
ディスク節約のため streaming 式: DL→抽出→削除。
結果は data/training/bulk_patches.npz に永続化。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.calibration import CalibratedConfig
from src.patch_extraction import PatchExtractor, PatchDataset, balance_dataset
from src.board import (
    COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
    COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA,
)

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLsjREVssD8baPrsapHGszFzhObLANmd6B"
WORK_DIR = Path("data/frames/bulk")
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path("data/training")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_INTERVAL_SEC = 8.0  # より長い間隔で時間短縮

NAMES_JA = {
    COLOR_EMPTY: "空", COLOR_RED: "赤", COLOR_BLUE: "青", COLOR_GREEN: "緑",
    COLOR_YELLOW: "黄", COLOR_PURPLE: "紫", COLOR_OJAMA: "お邪魔",
}


def has_eyes(patch: np.ndarray, min_eyes: int = 2) -> bool:
    h, w = patch.shape[:2]
    mh, mw = int(h * 0.15), int(w * 0.15)
    c = patch[mh:h-mh, mw:w-mw]
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    d = (g < 70).astype(np.uint8) * 255
    n, _, s, _ = cv2.connectedComponentsWithStats(d, connectivity=4)
    ta = c.shape[0] * c.shape[1]
    return sum(
        1 for i in range(1, n)
        if 2 <= s[i, cv2.CC_STAT_AREA] <= ta * 0.12
    ) >= min_eyes


def get_playlist_urls() -> list[tuple[int, str]]:
    result = subprocess.run([
        "yt-dlp", "--flat-playlist", "--print",
        "%(playlist_index)s\t%(id)s",
        PLAYLIST_URL,
    ], capture_output=True, text=True, check=True)
    urls = []
    for line in result.stdout.strip().split("\n"):
        idx_str, vid_id = line.split("\t")
        urls.append((int(idx_str), f"https://www.youtube.com/watch?v={vid_id}"))
    return urls


def download_video(url: str, out_path: Path) -> bool:
    if out_path.exists() and out_path.stat().st_size > 10_000_000:
        return True
    cmd = [
        "yt-dlp",
        "-f",
        "bestvideo[ext=mp4][vcodec^=avc1][height<=720]/"
        "bestvideo[ext=mp4][height<=720]",
        "-o", str(out_path),
        "--no-playlist",
        "--quiet",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and out_path.exists()


def extract_from_video(
    extractor: PatchExtractor, video_path: Path,
    sample_sec: float = SAMPLE_INTERVAL_SEC,
) -> PatchDataset:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return PatchDataset(
            patches=np.zeros((0, 32, 32, 3), dtype=np.uint8),
            labels=np.zeros(0, dtype=np.int64),
        )
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * sample_sec)))
    patches, labels = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            # 720p -> 1080p scale up
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


def main(start_index: int = 1, max_videos: int | None = None) -> None:
    config = CalibratedConfig.load("models/calibration_video01.json")
    extractor = PatchExtractor(config=config)

    urls = get_playlist_urls()
    print(f"プレイリスト: {len(urls)} 動画")

    all_patches: list[np.ndarray] = []
    all_labels: list[int] = []

    # 既存の multi3 データがあれば基盤として使う
    existing = Path("data/training/multi3_patches_balanced.npz")
    if existing.exists():
        ds = PatchDataset.load(existing)
        all_patches.append(ds.patches)
        all_labels.extend(ds.labels.tolist())
        print(f"既存 multi3 パッチを継承: {len(ds.labels)}")

    processed = 0
    for idx, url in urls:
        if idx < start_index:
            continue
        if max_videos is not None and processed >= max_videos:
            break
        video_path = WORK_DIR / f"video_{idx:02d}.mp4"
        start = time.time()
        print(f"\n[{idx}/{len(urls)}] DL: {url}")
        if not download_video(url, video_path):
            print(f"  失敗 - スキップ")
            continue
        print(f"  DL完了: {video_path.stat().st_size // 1024 // 1024}MB ({time.time()-start:.0f}s)")

        start = time.time()
        ds = extract_from_video(extractor, video_path)
        print(f"  抽出: {len(ds.labels)} パッチ ({time.time()-start:.0f}s)")

        if len(ds.labels) > 0:
            all_patches.append(ds.patches)
            all_labels.extend(ds.labels.tolist())

        # 動画削除 (容量節約)
        video_path.unlink(missing_ok=True)
        processed += 1

        # 定期的に中間保存
        if processed % 5 == 0:
            print("  中間保存...")
            save_intermediate(all_patches, all_labels, idx)

    print(f"\n全処理完了: {processed} 動画")
    save_intermediate(all_patches, all_labels, idx)


def save_intermediate(
    patches_list: list[np.ndarray], labels: list[int], idx: int,
) -> None:
    if not patches_list:
        return
    patches = np.concatenate(patches_list)
    labels_arr = np.array(labels, dtype=np.int64)

    # 目フィルタ適用
    keep = np.zeros(len(labels_arr), dtype=bool)
    for i in range(len(labels_arr)):
        e = has_eyes(patches[i])
        keep[i] = (not e) if labels_arr[i] == COLOR_EMPTY else e
    filt_p = patches[keep]
    filt_l = labels_arr[keep]

    ds = PatchDataset(patches=filt_p, labels=filt_l)
    ds.stats.patches_total = len(filt_l)
    unique, counts = np.unique(filt_l, return_counts=True)
    ds.stats.per_class_count = {int(k): int(v) for k, v in zip(unique, counts)}

    balanced = balance_dataset(ds, empty_ratio_cap=0.35)
    out_raw = OUT_DIR / f"bulk_patches_through_v{idx:02d}.npz"
    out_bal = OUT_DIR / f"bulk_patches_balanced_through_v{idx:02d}.npz"
    ds.save(out_raw)
    balanced.save(out_bal)
    print(f"  保存: {out_bal.name} ({balanced.stats.patches_total} パッチ)")
    for k in sorted(NAMES_JA.keys()):
        print(f"    {NAMES_JA[k]}: {balanced.stats.per_class_count.get(k, 0)}")


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    maxv = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(start_index=start, max_videos=maxv)
