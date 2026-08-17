# -*- coding: utf-8 -*-
"""DL先行モード (2026-08-12 user特別指示): 残りの動画DLだけを先に終わらせる。

- 走行中の _regen148_orchestrator と共存する設計:
  - マニフェストの「逆順」から処理 (本体は先頭から進むため正面衝突しない)
  - 出力ファイル or .part が既に存在する動画はスキップ (二重DL防止)
  - 収集はしない。DLのみ。削除もしない (本体が収集後に削除する)
- 3並列 (帯域律速のため)。ディスク安全弁 40GB。
"""
from __future__ import annotations
import subprocess
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "orch", str(PROJECT_ROOT / "scripts" / "_regen148_orchestrator_2026-08-11.py"))
_orch = importlib.util.module_from_spec(_spec)
sys.modules["orch"] = _orch
_spec.loader.exec_module(_orch)
FRAMES_DIR = _orch.FRAMES_DIR
FFMPEG_LOCATION = _orch.FFMPEG_LOCATION
YT_DLP_FORMAT = _orch.YT_DLP_FORMAT
load_manifest = _orch.load_manifest

JS_RUNTIME = "node:/home/ryouj/.nvm/versions/node/v20.20.1/bin/node"
MIN_FREE_GB = 40.0
PARALLEL = 3


def _free_gb() -> float:
    return shutil.disk_usage(str(FRAMES_DIR)).free / 2**30


def _dl_one(t) -> str:
    out = FRAMES_DIR / t.video_filename
    if out.exists():
        return f"[skip] {t.target_id} 既存"
    if list(FRAMES_DIR.glob(f"{t.video_filename}*part")):
        return f"[skip] {t.target_id} 本体がDL中"
    while _free_gb() < MIN_FREE_GB:
        time.sleep(120)
    url = f"https://www.youtube.com/watch?v={t.video_id}"
    cmd = [
        str(PROJECT_ROOT / "venv" / "bin" / "python"), "-m", "yt_dlp",
        "--ffmpeg-location", str(FFMPEG_LOCATION),
        "--js-runtimes", JS_RUNTIME,
        "-f", YT_DLP_FORMAT,
        "--remux-video", "mp4", "--no-playlist", "--no-progress",
        "-o", str(out), url,
    ]
    t0 = time.monotonic()
    subprocess.run(cmd, capture_output=True, text=True)
    if out.exists() and out.stat().st_size > 0:
        return f"[dl-ok] {t.target_id} ({time.monotonic()-t0:.0f}s)"
    for p in FRAMES_DIR.glob(f"{t.video_filename}*part"):
        p.unlink(missing_ok=True)
    return f"[dl-fail] {t.target_id}"


def main() -> int:
    npz_dir = PROJECT_ROOT / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-11"
    targets = load_manifest()
    todo = [
        t for t in reversed(targets)
        if t.video_id and t.origin == "download"
        and not (npz_dir / f"{t.target_id}.npz").exists()
    ]
    print(f"[dl-first] 対象 {len(todo)} 本 (逆順・{PARALLEL}並列)", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        for msg in pool.map(_dl_one, todo):
            done += 1
            print(f"{msg}  ({done}/{len(todo)})", flush=True)
    print("[dl-first] 完了", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
