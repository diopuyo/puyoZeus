"""ラッチ解除修正 + 新試合証拠ゲートの再収集検証 (2026-08-19)。

手元にある subset50 動画を、subset50 と同一の採用フラグ構成
(src.production_config.collect_flags、単一情報源) + 修正フラグ3本
(--enable-lockdown-score-numeric-release / --enable-lockdown-score-moving-release /
--enable-boundary-newmatch-evidence) で再収集し、
locked比率・won欠損率・試合数の before/after を測る。

⚠️ 再DL動画 (c13/c113/c111/c135/c11) は同一IDでも同一内容でない可能性がある
(memory feedback_redownload_content_drift_2026-08-14)。動画単位の集計比較は
可能だが、行単位の突合は元から手元にあった9本 (29/31/32/33/34/35/37/c109/
c132) のみで行うこと。

出力: data/indicators_v2/boards_lean_lockfix_2026-08-19/<vid>.npz
"""
from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.production_config import collect_flags  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "indicators_v2" / "boards_lean_lockfix_2026-08-19"
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
LOG_DIR = PROJECT_ROOT / "logs" / "recollect_lockfix_2026-08-19"

EXTRA_FLAGS = [
    "--enable-lockdown-score-numeric-release",
    "--enable-lockdown-score-moving-release",
    "--enable-boundary-newmatch-evidence",
]
# subset50 収集 (orchestrator) と同一の追加引数
BASE_ARGS = ["--with-next", "--enable-phantom-board-guard",
             "--max-sec", "0", "--sample-interval", "0"]
PARALLEL = 12


def run_one(stem: str) -> tuple[str, bool]:
    video = FRAMES_DIR / f"video_{stem}.mp4"
    out_npz = OUT_DIR / f"{stem}.npz"
    if out_npz.exists():
        return stem, True
    cmd = (
        [str(PROJECT_ROOT / "venv" / "bin" / "python"), "-u", "-m",
         "scripts._collect_lean_1t",
         "--video", str(video), "--out-npz", str(out_npz)]
        + collect_flags().split() + EXTRA_FLAGS + BASE_ARGS
    )
    log = LOG_DIR / f"{stem}.log"
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    ok = out_npz.exists()
    print(f"[recollect] {stem}: {'OK' if ok else 'FAIL rc=' + str(proc.returncode)}",
          flush=True)
    return stem, ok


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stems = sys.argv[1:]
    if not stems:
        raise SystemExit("usage: _recollect_lockfix_2026-08-19.py stem [stem...]")
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(run_one, s) for s in stems]
        for f in as_completed(futs):
            f.result()
    print("[recollect] all done", flush=True)


if __name__ == "__main__":
    main()
