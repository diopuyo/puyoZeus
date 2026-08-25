"""50本収集 (2026-08-19、新方式: 1手区切り+物理制約+色スワップ拒否)。

2026-08-18 に収集方式を作り替えた後の最初の本収集。
`scripts/_regen148_orchestrator_2026-08-11.py` の実績あるパイプライン
(DL直列 + 収集並列 + 再開可能 status.tsv) を、パス定数の上書きだけで流用する
(`_regen148_recollect_2026-08-18.py` と同じパターン、ロジックは複製しない)。

新方式の中身は `src/production_config.py` の `collect_flags()` が単一情報源:
- `--enable-move-segmented-recording` 1手ごとに1枚 (NEXT繰り上がり or tsumo_count増分、猶予15フレームで最短採用)
- `--enable-physics-persistence-filter` 物理制約違反の持続を棄却
- `--enable-ojama-fall-color-swap-guard` おじゃま落下中の色→別色誤読を拒否 (W26)
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_ORCH_PATH = PROJECT_ROOT / "scripts" / "_regen148_orchestrator_2026-08-11.py"
_spec = importlib.util.spec_from_file_location("_regen148_orchestrator_impl", _ORCH_PATH)
orch = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = orch
_spec.loader.exec_module(orch)

_NODE24_PATH = "/home/ryouj/.nvm/versions/node/v24.19.0/bin/node"
_DL_RETRY_COUNT = 5
_DL_RETRY_SLEEP_SEC = 45.0

# yt-dlp は --cookies に渡したファイルへセッションを書き戻し、その際に
# YouTube以外のCookieを削除してログイン情報を壊す (2026-08-18 実測)。
# マスターは触らせず毎回使い捨てコピーを渡す。
_COOKIES_MASTER = Path(
    "/mnt/c/Users/ryouj/AppData/Local/Temp/claude/"
    "C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer/"
    "22abd085-8e57-4d2a-857e-8516be642774/scratchpad/yt_cookies_master.txt"
)


def _cookie_args(target_id: str) -> list:
    if not (_COOKIES_MASTER.exists() and _COOKIES_MASTER.stat().st_size > 0):
        return []
    work = _COOKIES_MASTER.parent / f"yt_cookies_work_r50_{target_id}.txt"
    try:
        shutil.copyfile(_COOKIES_MASTER, work)
        work.chmod(0o644)
    except OSError:
        return []
    return ["--cookies", str(work)]


def _download_video(t):  # noqa: ANN001, ANN201
    out_path = orch.FRAMES_DIR / t.video_filename
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, 0.0
    url = f"https://www.youtube.com/watch?v={t.video_id}"
    start = time.monotonic()
    for attempt in range(1 + _DL_RETRY_COUNT):
        cmd = [
            str(PROJECT_ROOT / "venv" / "bin" / "python"), "-m", "yt_dlp",
            "--ffmpeg-location", str(orch.FFMPEG_LOCATION),
            "--js-runtimes", f"node:{_NODE24_PATH}",
            *_cookie_args(t.target_id),
            "-f", orch.YT_DLP_FORMAT,
            "--remux-video", "mp4", "--no-playlist", "--no-progress",
            "-o", str(out_path), url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if out_path.exists() and out_path.stat().st_size > 0:
            return True, time.monotonic() - start
        for p in orch.FRAMES_DIR.glob(f"{t.video_filename}*part"):
            p.unlink(missing_ok=True)
        print(f"[dl][retry={attempt}] {t.target_id}: {proc.stderr[-200:]}", flush=True)
        if attempt < _DL_RETRY_COUNT:
            time.sleep(_DL_RETRY_SLEEP_SEC)
    return False, time.monotonic() - start


if __name__ == "__main__":
    base = PROJECT_ROOT / "data" / "verify" / "regen_2026-08-19_subset50"
    orch.MANIFEST_TSV = base / "manifest.tsv"
    orch.STATUS_TSV = base / "status.tsv"
    orch.NEW_NPZ_DIR = PROJECT_ROOT / "data" / "indicators_v2" / "boards_lean_subset50_2026-08-19"
    orch.LOG_DIR = PROJECT_ROOT / "logs" / "regen50_2026-08-19_per_video"
    orch.download_video = _download_video
    orch.MAX_COLLECT_PARALLEL = 14
    orch.TOTAL_HOLD_SLOTS = orch.MAX_COLLECT_PARALLEL + orch.DOWNLOAD_QUEUE_SIZE
    orch._HOLD_SEMAPHORE = threading.Semaphore(orch.TOTAL_HOLD_SLOTS)

    if "--smoke-check" in sys.argv:
        print("[smoke] MANIFEST =", orch.MANIFEST_TSV)
        print("[smoke] STATUS   =", orch.STATUS_TSV)
        print("[smoke] NPZ_DIR  =", orch.NEW_NPZ_DIR)
        print("[smoke] 並列     =", orch.MAX_COLLECT_PARALLEL)
        from src.production_config import collect_flags
        cf = collect_flags()
        for n in ("--enable-move-segmented-recording",
                  "--enable-physics-persistence-filter",
                  "--enable-ojama-fall-color-swap-guard"):
            print(f"[smoke] {'OK ' if n in cf else 'NG '}{n}")
        raise SystemExit(0)

    raise SystemExit(orch.main())
