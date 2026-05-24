"""
最適化版 並列 bulk 処理 v2。
- ffmpeg でフレーム抽出 (cv2.VideoCapture の全フレームデコードを回避)
- torch は CPU 強制
- 処理完了後に動画ファイルと .part ファイルを削除
- ThreadPool (DL) + ProcessPool (抽出) の並列アーキテクチャ維持
"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from src.calibration import CalibratedConfig
from src.patch_extraction import PatchExtractor, PatchDataset
from src.board import COLOR_EMPTY

# ============================
# 実行環境のバイナリパス解決
# ============================

# venv 内の yt-dlp バイナリを使う (システム PATH に無い場合の対策)
_VENV_BIN = Path(sys.executable).parent
YT_DLP = str(_VENV_BIN / "yt-dlp") if (_VENV_BIN / "yt-dlp").exists() else "yt-dlp"


def _get_ffmpeg_path() -> str:
    """imageio_ffmpeg からバンドル ffmpeg バイナリのパスを取得する。"""
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


# ============================
# 定数定義
# ============================

PLAYLISTS = [
    ("pl1", "https://www.youtube.com/playlist?list=PLsjREVssD8baPrsapHGszFzhObLANmd6B"),
    ("pl2", "https://www.youtube.com/playlist?list=PLsjREVssD8bZlPdMhq0kyaVKpSK-Dl7Jk"),
]
EXCLUDED_IDS = {"KmRmSHe7DPg"}
START_INDEX = {"pl1": 4, "pl2": 2}  # pl1 は 1-3 既処理, pl2 は 1 除外

N_DOWNLOAD_WORKERS = 8
N_EXTRACT_WORKERS = 14
SAMPLE_SEC = 8.0  # フレーム抽出間隔 (秒)

WORK = Path("data/frames/parallel")
WORK.mkdir(parents=True, exist_ok=True)
OUT = Path("data/training/parallel")
OUT.mkdir(parents=True, exist_ok=True)


def fetch_playlist(url: str, start_idx: int) -> list[tuple[int, str]]:
    """(index, video_id) リストを取得する。"""
    result = subprocess.run([
        YT_DLP, "--flat-playlist", "--print",
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
        # 短すぎる動画はスキップ (500秒未満)
        if dur == "NA" or not dur.isdigit() or int(dur) < 500:
            continue
        items.append((idx, vid_id))
    return items


def download_one(tag: str, idx: int, vid_id: str) -> Path | None:
    """1動画をダウンロードする。既存ファイルがあればスキップ。"""
    out = WORK / f"{tag}_video_{idx:02d}.mp4"
    if out.exists() and out.stat().st_size > 10_000_000:
        return out
    cmd = [
        YT_DLP,
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


def _extract_frames_ffmpeg(video_path: Path, output_dir: Path) -> list[Path]:
    """
    ffmpeg で SAMPLE_SEC 間隔のフレームだけを抽出する。

    cv2.VideoCapture で全フレームをデコードする代わりに、
    ffmpeg -vf fps=1/SAMPLE_SEC を使って必要なフレームだけ出力する。
    これにより 240 フレーム中 1 フレームだけ使うような無駄なデコードを回避する。
    """
    ffmpeg = _get_ffmpeg_path()
    output_pattern = str(output_dir / "frame_%06d.png")

    cmd = [
        ffmpeg,
        "-i", str(video_path),
        "-vf", f"fps=1/{SAMPLE_SEC}",       # 8秒に1フレーム
        "-vsync", "vfr",                      # 可変フレームレート
        "-q:v", "2",                          # 高品質 PNG
        output_pattern,
        "-loglevel", "error",
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    # 出力されたフレームファイルをソートして返す
    frames = sorted(output_dir.glob("frame_*.png"))
    return frames


def extract_and_save(args: tuple[str, int, Path]) -> tuple[str, int, int]:
    """
    worker: 1動画から ffmpeg でフレーム抽出 → パッチ保存 → 動画削除。

    デッドロック対策: 巨大な ndarray をメインプロセスに返さず、
    ワーカー内で直接 npz に保存し、パッチ数だけ返す。
    """
    tag, idx, video_path = args
    config = CalibratedConfig.load("models/calibration_video01.json")
    extractor = PatchExtractor(config=config)

    patches, labels = [], []

    # 一時ディレクトリにフレームを抽出
    with tempfile.TemporaryDirectory(prefix="puyo_frames_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        try:
            frames = _extract_frames_ffmpeg(video_path, tmpdir_path)
        except subprocess.CalledProcessError:
            # ffmpeg エラー時は 0 パッチで返す
            return tag, idx, 0

        # 各フレームからパッチを抽出
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            if frame.shape[0] != 1080:
                frame = cv2.resize(frame, (1920, 1080))
            ps, ls = extractor.extract_from_frame(frame)
            patches.extend(ps)
            labels.extend(ls)

    # 動画削除 (ディスク節約)
    video_path.unlink(missing_ok=True)

    # ワーカー内で直接保存 (メモリ転送回避)
    if patches:
        out_path = OUT / f"{tag}_v{idx:02d}.npz"
        np.savez_compressed(
            out_path,
            patches=np.stack(patches),
            labels=np.array(labels, dtype=np.int64),
        )
    return tag, idx, len(labels)


def save_video_patches(tag: str, idx: int, patches: np.ndarray, labels: np.ndarray) -> None:
    """1動画分のパッチを個別ファイルに保存 (後で merge する)。(互換用)"""
    if len(labels) == 0:
        return
    out = OUT / f"{tag}_v{idx:02d}.npz"
    np.savez_compressed(out, patches=patches, labels=labels)


def cleanup_part_files() -> int:
    """不完全ダウンロード (.part) ファイルを削除する。"""
    count = 0
    for f in WORK.glob("*.part"):
        f.unlink(missing_ok=True)
        count += 1
    return count


def main() -> None:
    # .part ファイルを削除 (中断されたダウンロード)
    removed = cleanup_part_files()
    if removed:
        print(f".part ファイル {removed} 件を削除")

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

    if not todo:
        print("処理対象なし。全て完了済み。")
        return

    # ログ用ロック
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

    # DL完了 → extract_and_save 投げる (ワーカー内で保存、巨大データ転送なし)
    extract_futures = []
    for dl_fut in as_completed(dl_futures):
        tag, idx, path = dl_fut.result()
        if path is None:
            continue
        fut = extract_pool.submit(extract_and_save, (tag, idx, path))
        extract_futures.append((tag, idx, fut))

    # Extract 結果収集 (パッチ数のみ受け取る)
    completed = 0
    for tag, idx, fut in extract_futures:
        try:
            _, _, n_patches = fut.result()
        except Exception as e:
            log(f"EXTRACT FAIL {tag} #{idx}: {e}")
            continue
        completed += 1
        log(f"EXTRACT OK {tag} #{idx}: {n_patches} patches  [{completed}/{len(extract_futures)}]")

    extract_pool.shutdown()
    dl_pool.shutdown()

    # 残りの .part ファイルがあれば掃除
    cleanup_part_files()

    log(f"全完了: {completed}/{len(todo)} 動画処理")


if __name__ == "__main__":
    main()
