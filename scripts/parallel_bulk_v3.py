"""
並列 bulk 処理 v3 — シンプル＆確実版。
- DL: subprocess 直接呼び出し（1本ずつ、ただし高速）
- 抽出: マルチプロセス (ProcessPoolExecutor) でワーカー内保存
- mp.Lock 不使用、ThreadPool 不使用 → デッドロック回避
- DL済み動画は即座に抽出キューへ投入
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from src.calibration import CalibratedConfig
from src.patch_extraction import PatchExtractor
from src.board import COLOR_EMPTY

# ============================
# パス解決
# ============================
_VENV_BIN = Path(sys.executable).parent
YT_DLP = str(_VENV_BIN / "yt-dlp") if (_VENV_BIN / "yt-dlp").exists() else "yt-dlp"


def _get_ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


# ============================
# 設定
# ============================
PLAYLISTS = [
    ("pl1", "https://www.youtube.com/playlist?list=PLsjREVssD8baPrsapHGszFzhObLANmd6B"),
    ("pl2", "https://www.youtube.com/playlist?list=PLsjREVssD8bZlPdMhq0kyaVKpSK-Dl7Jk"),
]
EXCLUDED_IDS = {"KmRmSHe7DPg"}
START_INDEX = {"pl1": 4, "pl2": 2}

N_EXTRACT_WORKERS = 14  # 16コア中14を抽出に使う
SAMPLE_SEC = 8.0

WORK = Path("data/frames/parallel")
WORK.mkdir(parents=True, exist_ok=True)
OUT = Path("data/training/parallel")
OUT.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================
# プレイリスト取得
# ============================
def fetch_playlist(tag: str, url: str) -> list[tuple[int, str]]:
    result = subprocess.run(
        [YT_DLP, "--flat-playlist", "--print",
         "%(playlist_index)s\t%(id)s\t%(duration)s", url],
        capture_output=True, text=True, check=True,
    )
    items = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        idx_s, vid_id, dur = parts
        if not idx_s.isdigit():
            continue
        idx = int(idx_s)
        if idx < START_INDEX[tag] or vid_id in EXCLUDED_IDS:
            continue
        if dur == "NA" or not dur.isdigit() or int(dur) < 500:
            continue
        items.append((idx, vid_id))
    return items


# ============================
# ダウンロード (メインスレッドで逐次)
# ============================
def download(tag: str, idx: int, vid_id: str) -> Path | None:
    out = WORK / f"{tag}_video_{idx:02d}.mp4"
    if out.exists() and out.stat().st_size > 10_000_000:
        return out
    # .part 削除
    part = Path(str(out) + ".part")
    part.unlink(missing_ok=True)
    cmd = [
        YT_DLP, "-f",
        "bestvideo[ext=mp4][vcodec^=avc1][height<=720]/"
        "bestvideo[ext=mp4][height<=720]",
        "-o", str(out), "--no-playlist", "--quiet",
        f"https://www.youtube.com/watch?v={vid_id}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode == 0 and out.exists() and out.stat().st_size > 10_000_000:
        return out
    return None


# ============================
# 抽出 (ProcessPool ワーカー内で保存)
# ============================
def extract_and_save(tag: str, idx: int, video_path_str: str) -> tuple[str, int, int]:
    """ワーカー内で ffmpeg 抽出→パッチ保存→動画削除。パッチ数だけ返す。"""
    video_path = Path(video_path_str)
    config = CalibratedConfig.load("models/calibration_video01.json")
    extractor = PatchExtractor(config=config)
    ffmpeg = _get_ffmpeg()

    patches, labels = [], []
    with tempfile.TemporaryDirectory(prefix="puyo_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        out_pattern = str(tmpdir_path / "f_%06d.png")
        try:
            subprocess.run(
                [ffmpeg, "-i", str(video_path),
                 "-vf", f"fps=1/{SAMPLE_SEC}", "-vsync", "vfr",
                 "-q:v", "2", out_pattern, "-loglevel", "error"],
                capture_output=True, text=True, check=True, timeout=600,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return tag, idx, 0

        for fp in sorted(tmpdir_path.glob("f_*.png")):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue
            if frame.shape[0] != 1080:
                frame = cv2.resize(frame, (1920, 1080))
            ps, ls = extractor.extract_from_frame(frame)
            patches.extend(ps)
            labels.extend(ls)

    # 動画削除
    video_path.unlink(missing_ok=True)

    if patches:
        np.savez_compressed(
            OUT / f"{tag}_v{idx:02d}.npz",
            patches=np.stack(patches),
            labels=np.array(labels, dtype=np.int64),
        )
    return tag, idx, len(labels)


# ============================
# メイン: DL→抽出パイプライン
# ============================
def main() -> None:
    # .part 掃除
    for f in WORK.glob("*.part"):
        f.unlink(missing_ok=True)

    # 動画リスト
    all_videos: list[tuple[str, int, str]] = []
    for tag, url in PLAYLISTS:
        items = fetch_playlist(tag, url)
        for idx, vid_id in items:
            all_videos.append((tag, idx, vid_id))
    log(f"対象動画: {len(all_videos)}")

    # 処理済みスキップ
    done = set()
    for f in OUT.glob("*.npz"):
        parts = f.stem.rsplit("_v", 1)
        if len(parts) == 2 and parts[1].isdigit():
            done.add((parts[0], int(parts[1])))
    todo = [v for v in all_videos if (v[0], v[1]) not in done]
    log(f"処理済: {len(done)}  残り: {len(todo)}")

    if not todo:
        log("全完了済み")
        return

    pool = ProcessPoolExecutor(max_workers=N_EXTRACT_WORKERS)
    pending: dict[str, tuple[str, int]] = {}  # future_key -> (tag, idx)
    extract_count = 0
    total = len(todo)

    def collect_done() -> None:
        """完了済み future を回収してログ出力。"""
        nonlocal extract_count
        newly_done = []
        for key, (t, i) in list(pending.items()):
            fut = futures_map.get(key)
            if fut is not None and fut.done():
                newly_done.append(key)
                try:
                    _, _, n = fut.result()
                    extract_count += 1
                    log(f"EXTRACT OK {t} #{i}: {n} patches  [{extract_count}/{total}]")
                except Exception as e:
                    extract_count += 1
                    log(f"EXTRACT FAIL {t} #{i}: {e}  [{extract_count}/{total}]")
        for key in newly_done:
            del pending[key]
            del futures_map[key]

    futures_map: dict[str, object] = {}

    for vi, (tag, idx, vid_id) in enumerate(todo):
        # DL済みファイルがあれば即座に抽出投入
        video_path = WORK / f"{tag}_video_{idx:02d}.mp4"
        if video_path.exists() and video_path.stat().st_size > 10_000_000:
            log(f"DL SKIP {tag} #{idx} (既存)")
            path = video_path
        else:
            t0 = time.time()
            path = download(tag, idx, vid_id)
            if path:
                log(f"DL OK {tag} #{idx}: {path.stat().st_size//1024//1024}MB ({time.time()-t0:.0f}s)")
            else:
                log(f"DL FAIL {tag} #{idx}")
                extract_count += 1
                continue

        # 抽出ジョブ投入
        key = f"{tag}_{idx}"
        fut = pool.submit(extract_and_save, tag, idx, str(path))
        futures_map[key] = fut
        pending[key] = (tag, idx)

        # 完了分を回収 (メモリ・プロセス管理)
        collect_done()

    # 残りの抽出完了を待つ
    log(f"全DL完了。抽出待ち: {len(pending)}")
    while pending:
        time.sleep(2)
        collect_done()

    pool.shutdown()
    # 掃除
    for f in WORK.glob("*.part"):
        f.unlink(missing_ok=True)
    log(f"全完了: {extract_count}/{total}")


if __name__ == "__main__":
    main()
