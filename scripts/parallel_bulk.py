"""
16コア CPU を使い倒す並列 bulk 処理。
- 4 concurrent downloads (bandwidth shared)
- 8 concurrent extractions (CPU)
- 2 playlists を統合処理
- 各動画: DL → extract → delete → save patches
"""
from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import numpy as np

from src.calibration import CalibratedConfig
from src.patch_extraction import PatchExtractor, PatchDataset
from src.board import COLOR_EMPTY

PLAYLISTS = [
    ("pl1", "https://www.youtube.com/playlist?list=PLsjREVssD8baPrsapHGszFzhObLANmd6B"),
    ("pl2", "https://www.youtube.com/playlist?list=PLsjREVssD8bZlPdMhq0kyaVKpSK-Dl7Jk"),
]
EXCLUDED_IDS = {"KmRmSHe7DPg"}
START_INDEX = {"pl1": 4, "pl2": 2}  # pl1 は 1-3 既処理, pl2 は 1 除外

N_DOWNLOAD_WORKERS = 4
N_EXTRACT_WORKERS = 8
SAMPLE_SEC = 8.0

WORK = Path("data/frames/parallel")
WORK.mkdir(parents=True, exist_ok=True)
OUT = Path("data/training/parallel")
OUT.mkdir(parents=True, exist_ok=True)


def fetch_playlist(url: str, start_idx: int) -> list[tuple[str, int, str]]:
    """(pl_tag, index, video_url) リスト取得"""
    result = subprocess.run([
        "yt-dlp", "--flat-playlist", "--print",
        "%(playlist_index)s\t%(id)s\t%(duration)s", url,
    ], capture_output=True, text=True, check=True)
    items = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        idx_s, vid_id, dur = parts
        if not idx_s.isdigit():
            continue
        idx = int(idx_s)
        if idx < start_idx or vid_id in EXCLUDED_IDS:
            continue
        if dur == "NA" or not dur.isdigit() or int(dur) < 500:
            continue
        items.append((idx, vid_id))
    return items


def download_one(tag: str, idx: int, vid_id: str) -> Path | None:
    out = WORK / f"{tag}_video_{idx:02d}.mp4"
    if out.exists() and out.stat().st_size > 10_000_000:
        return out
    cmd = [
        "yt-dlp",
        "-f",
        "bestvideo[ext=mp4][vcodec^=avc1][height<=720]/"
        "bestvideo[ext=mp4][height<=720]",
        "-o", str(out),
        "--no-playlist", "--quiet",
        f"https://www.youtube.com/watch?v={vid_id}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and out.exists() and out.stat().st_size > 10_000_000:
        return out
    return None


def extract_one(args: tuple[str, Path]) -> tuple[str, np.ndarray, np.ndarray]:
    """worker: 1 動画から patches+labels を返し、動画削除"""
    tag, video_path = args
    config = CalibratedConfig.load("models/calibration_video01.json")
    extractor = PatchExtractor(config=config)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return tag, np.zeros((0, 32, 32, 3), dtype=np.uint8), np.zeros(0, dtype=np.int64)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * SAMPLE_SEC)))
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
    # 動画削除 (ディスク節約)
    video_path.unlink(missing_ok=True)
    if not patches:
        return tag, np.zeros((0, 32, 32, 3), dtype=np.uint8), np.zeros(0, dtype=np.int64)
    return (
        tag,
        np.stack(patches),
        np.array(labels, dtype=np.int64),
    )


def save_video_patches(tag: str, idx: int, patches: np.ndarray, labels: np.ndarray) -> None:
    """1動画分のパッチを個別ファイルに保存 (後で merge)"""
    if len(labels) == 0:
        return
    out = OUT / f"{tag}_v{idx:02d}.npz"
    np.savez_compressed(out, patches=patches, labels=labels)


def main() -> None:
    # 全動画リスト構築
    all_videos: list[tuple[str, int, str]] = []
    for tag, url in PLAYLISTS:
        items = fetch_playlist(url, START_INDEX[tag])
        for idx, vid_id in items:
            all_videos.append((tag, idx, vid_id))
    print(f"合計対象動画: {len(all_videos)}")

    # 既に処理済みのものはスキップ
    done = set()
    for f in OUT.glob("*.npz"):
        name = f.stem  # pl1_v05 等
        parts = name.rsplit("_v", 1)
        if len(parts) == 2:
            t, i = parts
            if i.isdigit():
                done.add((t, int(i)))
    todo = [v for v in all_videos if (v[0], v[1]) not in done]
    print(f"処理済: {len(done)}  処理予定: {len(todo)}")

    # DL queue + Extract queue で生産-消費
    dl_queue: Queue = Queue(maxsize=N_EXTRACT_WORKERS * 2)
    print_lock = mp.Lock()

    def log(msg: str) -> None:
        with print_lock:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def dl_worker_task(vid: tuple[str, int, str]) -> tuple[str, int, Path | None]:
        tag, idx, vid_id = vid
        start = time.time()
        path = download_one(tag, idx, vid_id)
        if path:
            log(f"DL OK {tag} #{idx}: {path.stat().st_size//1024//1024}MB ({time.time()-start:.0f}s)")
        else:
            log(f"DL FAIL {tag} #{idx}")
        return tag, idx, path

    # ThreadPool で DL (I/O bound)
    # ProcessPool で extract (CPU bound)
    extract_pool = ProcessPoolExecutor(max_workers=N_EXTRACT_WORKERS)
    dl_pool = ThreadPoolExecutor(max_workers=N_DOWNLOAD_WORKERS)

    # DL 全部投げる
    dl_futures = [dl_pool.submit(dl_worker_task, v) for v in todo]

    # DL完了 → extract 投げる
    extract_futures = []
    for dl_fut in as_completed(dl_futures):
        tag, idx, path = dl_fut.result()
        if path is None:
            continue
        fut = extract_pool.submit(extract_one, (tag, path))
        extract_futures.append((tag, idx, fut))

    # Extract 結果収集
    completed = 0
    for tag, idx, fut in extract_futures:
        start = time.time()
        try:
            _, patches, labels = fut.result()
        except Exception as e:
            log(f"EXTRACT FAIL {tag} #{idx}: {e}")
            continue
        save_video_patches(tag, idx, patches, labels)
        completed += 1
        log(f"EXTRACT OK {tag} #{idx}: {len(labels)} patches (+{completed}/{len(todo)})")

    extract_pool.shutdown()
    dl_pool.shutdown()
    log(f"全完了: {completed}/{len(todo)} 動画処理")


if __name__ == "__main__":
    main()
