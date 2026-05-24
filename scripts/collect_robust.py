"""
堅牢なデータ収集スクリプト (collect_robust.py)

特徴:
1. 複数のダウンロード戦略 (Cookie, JS-Runtime) を試行
2. ffmpeg による高速並列抽出
3. 既存の .npz をスキップして pl2 を中心に未完了分を補完
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# ============================
# パス・設定
# ============================
_PROJECT_ROOT = Path(__file__).parent.parent
_VENV_BIN = _PROJECT_ROOT / "venv" / "bin"
YT_DLP = str(_VENV_BIN / "yt-dlp") if (_VENV_BIN / "yt-dlp").exists() else "yt-dlp"

# Node.js パス (以前のログから推定)
NODE_PATH = "/home/ryouj/.nvm/versions/node/v20.20.1/bin/node"

PLAYLISTS = [
    ("pl1", "https://www.youtube.com/playlist?list=PLsjREVssD8baPrsapHGszFzhObLANmd6B"),
    ("pl2", "https://www.youtube.com/playlist?list=PLsjREVssD8bZlPdMhq0kyaVKpSK-Dl7Jk"),
]
START_INDEX = {"pl1": 4, "pl2": 2}

WORK_DIR = _PROJECT_ROOT / "data" / "frames" / "parallel"
OUT_DIR = _PROJECT_ROOT / "data" / "training" / "parallel"

N_WORKERS = 14
SAMPLE_SEC = 8.0

# 必要なモジュールをパスに追加
sys.path.append(str(_PROJECT_ROOT))
from src.calibration import CalibratedConfig
from src.patch_extraction import PatchExtractor

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================
# ダウンロード戦略
# ============================
def try_download(vid_id: str, out_path: Path) -> bool:
    base_cmd = [
        YT_DLP, "-f",
        "bestvideo[ext=mp4][vcodec^=avc1][height<=720]/bestvideo[ext=mp4][height<=720]",
        "-o", str(out_path), "--no-playlist", "--quiet",
        f"https://www.youtube.com/watch?v={vid_id}",
    ]
    
    # 戦略リスト
    strategies = [
        # 1. 最小構成
        base_cmd,
        # 2. Node.js ランタイム使用
        base_cmd[:1] + ["--js-runtimes", f"node:{NODE_PATH}"] + base_cmd[1:],
        # 3. ブラウザクッキー使用 (Chrome)
        base_cmd[:1] + ["--cookies-from-browser", "chrome"] + base_cmd[1:],
        # 4. ブラウザクッキー使用 (Firefox/Edgeなど)
        base_cmd[:1] + ["--cookies-from-browser", "firefox"] + base_cmd[1:],
    ]

    for i, cmd in enumerate(strategies):
        try:
            log(f"  DL試行 (戦略 {i+1})...")
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 5_000_000:
                return True
        except Exception as e:
            log(f"  戦略 {i+1} でエラー: {e}")
        # .part ファイルの掃除
        Path(str(out_path) + ".part").unlink(missing_ok=True)
    return False

# ============================
# 並列抽出
# ============================
def extract_worker(tag: str, idx: int, video_path_str: str) -> int:
    video_path = Path(video_path_str)
    config_path = _PROJECT_ROOT / "models" / "calibration_video01.json"
    if not config_path.exists():
        return 0
        
    config = CalibratedConfig.load(str(config_path))
    extractor = PatchExtractor(config=config)
    
    # WSL環境を想定して ffmpeg を取得
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    patches, labels = [], []
    with tempfile.TemporaryDirectory(prefix="puyo_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        out_pattern = str(tmpdir_path / "f_%06d.png")
        try:
            subprocess.run(
                [ffmpeg, "-i", str(video_path),
                 "-vf", f"fps=1/{SAMPLE_SEC}", "-vsync", "vfr",
                 "-q:v", "2", out_pattern, "-loglevel", "error"],
                capture_output=True, check=True, timeout=600,
            )
        except Exception:
            return 0

        for fp in sorted(tmpdir_path.glob("f_*.png")):
            frame = cv2.imread(str(fp))
            if frame is None: continue
            if frame.shape[0] != 1080:
                frame = cv2.resize(frame, (1920, 1080))
            ps, ls = extractor.extract_from_frame(frame)
            patches.extend(ps)
            labels.extend(ls)

    video_path.unlink(missing_ok=True)
    if patches:
        np.savez_compressed(
            OUT_DIR / f"{tag}_v{idx:02d}.npz",
            patches=np.stack(patches),
            labels=np.array(labels, dtype=np.int64),
        )
    return len(labels)

# ============================
# メインループ
# ============================
def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done = {f.stem for f in OUT_DIR.glob("*.npz")}
    
    pool = ProcessPoolExecutor(max_workers=N_WORKERS)
    
    for tag, url in PLAYLISTS:
        log(f"プレイリスト処理開始: {tag}")
        try:
            # プレイリスト内の一覧取得
            r = subprocess.run(
                [YT_DLP, "--flat-playlist", "--print", "%(playlist_index)s\t%(id)s", url],
                capture_output=True, text=True, check=True, timeout=120
            )
        except Exception as e:
            log(f"プレイリスト取得失敗: {e}")
            continue

        for line in r.stdout.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) < 2: continue
            idx, vid_id = int(parts[0]), parts[1]
            
            if idx < START_INDEX.get(tag, 0): continue
            if f"{tag}_v{idx:02d}" in done: continue
            
            log(f"処理対象: {tag} #{idx} ({vid_id})")
            video_path = WORK_DIR / f"{tag}_v{idx:02d}.mp4"
            
            if try_download(vid_id, video_path):
                log(f"  DL成功。抽出キューへ投入します。")
                pool.submit(extract_worker, tag, idx, str(video_path))
            else:
                log(f"  DL失敗。スキップします。")

    log("すべてのダウンロード指示が完了しました。抽出の終了を待ちます。")
    pool.shutdown(wait=True)
    log("全工程完了。")

if __name__ == "__main__":
    main()
