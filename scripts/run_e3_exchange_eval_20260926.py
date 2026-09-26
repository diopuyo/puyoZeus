"""E3の事前登録済みOFF/ONレンダを動画×モード単位で再開する。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from src.production_config import advantage_overlay_flags

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/e3"
ASSETS = Path("/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer")
SOURCES = ("q_7gc4TgFig", "fcXG83vInDY", "mia8KCjr52g")
VIDEO_ROOT = Path("/mnt/d/puyo_analyzer/videos/source")
MODES = ("off", "on")
MAX_SECONDS = 900
WORKERS = 3
SEED = 20260926
SMOKE_SECONDS = 2


def save_json(path: Path, value: object) -> None:
    """途中で切れても完成状態を誤認しないよう原子的に保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def digest(path: Path) -> str:
    """固定した入力の内容を記録する。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare() -> None:
    """欠けた本番資産だけを指定元から複製する。探索はしない。"""
    files = ("models/calibration_video01.json",
             "models/cnn_phase_b_large_v2.pt", "models/cnn_global_best.pt",
             "models/cnn_best.pt",
             "data/verify/retrain148_2026-08-14/model_full148_full_features.joblib",
             "data/verify/retrain148_2026-08-14/feature_cols_full.json")
    for name in files:
        target = ROOT / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ASSETS / name, target)
    templates = ROOT / "models/ui_templates"
    if not templates.exists():
        shutil.copytree(ASSETS / "models/ui_templates", templates)
    prereg = ROOT / "docs/agent_coordination/E3_PREREGISTRATION_2026-09-26.md"
    frozen = OUT / "PREREGISTRATION.md"
    if frozen.exists() and digest(frozen) != digest(prereg):
        raise ValueError("事前登録の変更を検知。既存実行へ上書きしない")
    if not frozen.exists():
        shutil.copy2(prereg, frozen)
    save_json(OUT / "assets.json", {name: digest(ROOT / name) for name in files})


def command(source: str, mode: str, dest: Path, seconds: int) -> list[str]:
    """本番フラグは単一情報源を実行時に読む。"""
    flags = shlex.split(advantage_overlay_flags())
    if "--exchange-event-update" in flags:
        raise ValueError("OFF基準にexchange-event-updateが含まれている")
    args = [sys.executable, "-m", "scripts.run_e3_exchange_eval_20260926",
            "--worker", "--video", str(VIDEO_ROOT / f"{source}_first_0_900_20260925_v1.mp4"),
            "--out", str(dest / "overlay.mp4"), "--max-sec", str(seconds),
            "--dump-timeline", str(dest / "settled.npz"),
            "--dump-display-timeline", str(dest / "display.npz"), *flags]
    if mode == "on":
        args += ["--exchange-event-update", "--dump-exchange-events", str(dest / "events.jsonl")]
    return args


def validate(source: str, dest: Path, seconds: int) -> dict:
    """全処理フレームのdumpと動画の完了を確認する。"""
    import cv2
    import numpy as np
    from src.fps_normalize import resolve_normalize_fps_30_stride
    cap = cv2.VideoCapture(str(VIDEO_ROOT / f"{source}_first_0_900_20260925_v1.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(seconds * fps))
    cap.release()
    stride = resolve_normalize_fps_30_stride(fps)
    expected = np.arange(0, frames, stride) / fps
    with np.load(dest / "display.npz") as dump:
        if not np.array_equal(dump["t_sec"], expected):
            raise ValueError("固定した全フレーム集合とdumpが不一致")
        count = len(dump["t_sec"])
    cap = cv2.VideoCapture(str(dest / "overlay.mp4"))
    rendered = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if rendered != count:
        raise ValueError(f"動画/dump不一致: {rendered}/{count}")
    return dict(frames=count, fps=fps, stride=stride)


def run_one(source: str, mode: str, smoke: bool = False) -> dict:
    """成功済み単位はスキップし、失敗単位は再実行できる。"""
    dest = OUT / ("smoke" if smoke else "renders") / source / mode
    dest.mkdir(parents=True, exist_ok=True)
    status_path = dest / "status.json"
    seconds = SMOKE_SECONDS if smoke else MAX_SECONDS
    args = command(source, mode, dest, seconds)
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("state") == "completed" and previous.get("command") == args:
            validate(source, dest, seconds)
            return previous
    state = dict(source=source, mode=mode, state="running", command=args,
                 start_utc=datetime.now(timezone.utc).isoformat())
    start = time.monotonic()
    with (dest / "render.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        state["pid"] = process.pid
        save_json(status_path, state)
        state["returncode"] = process.wait()
    state["elapsed_seconds"] = time.monotonic() - start
    try:
        if state["returncode"]:
            raise RuntimeError(f"render終了コード={state['returncode']}")
        state.update(validate(source, dest, seconds))
        state["state"] = "completed"
    except Exception as error:
        state.update(state="failed", error=str(error))
    save_json(status_path, state)
    print(json.dumps(state, ensure_ascii=False), flush=True)
    return state


def worker() -> None:
    """両モードの乱数初期状態を合わせて既存CLIをそのまま呼ぶ。"""
    import random
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    sys.argv.remove("--worker")
    from scripts.visualize_advantage_overlay import main as render_main
    render_main()


def main() -> None:
    """多重起動を防ぎ、独立した3動画を並列実行する。"""
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.nice(max(0, 19 - os.nice(0)))
    prepare()
    save_json(OUT / "runner.json", dict(pid=os.getpid(), nice=os.nice(0),
              start_utc=datetime.now(timezone.utc).isoformat(), smoke=options.smoke,
              preregistration_sha256=digest(OUT / "PREREGISTRATION.md")))
    if options.smoke:
        results = [run_one(SOURCES[0], mode, True) for mode in MODES]
    else:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(run_one, source, mode)
                       for mode in MODES for source in SOURCES]
            results = [future.result() for future in futures]
    save_json(OUT / ("smoke_summary.json" if options.smoke else "render_summary.json"), results)
    if any(row["state"] != "completed" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    worker() if "--worker" in sys.argv else main()
