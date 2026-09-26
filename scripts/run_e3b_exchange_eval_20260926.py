"""E3b: OFFの実動画不変確認後、失敗したfcX ONだけを再実行する。"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.run_e3_exchange_eval_20260926 import (
    ROOT, OUT, command, digest, run_one, save_json, worker,
)

CHECK = OUT / "e3b"
CHECK_SOURCE = "q_7gc4TgFig"
CHECK_START = 140
CHECK_SECONDS = 25
CHECK_FPS = 30


def baseline_modules() -> None:
    """修正前の二モジュールを凍結コピーから読み込む。OFFの依存も実物を通す。"""
    for name in ("exchange_event_tracker", "exchange_event_overlay"):
        qualified = "src." + name
        spec = importlib.util.spec_from_file_location(
            qualified, CHECK / "baseline_source" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)


def short_off(label: str) -> dict:
    """同じ25秒区間を旧コード/修正コードでレンダし、成果物をハッシュ化する。"""
    import numpy as np
    dest = CHECK / label
    dest.mkdir(parents=True, exist_ok=True)
    args = command(CHECK_SOURCE, "off", dest, CHECK_SECONDS)
    args[2] = "scripts.run_e3b_exchange_eval_20260926"
    args += ["--start-sec", str(CHECK_START)]
    env = dict(os.environ, E3B_BASELINE="1" if label == "off_before" else "0")
    with (dest / "render.log").open("w", encoding="utf-8") as log:
        subprocess.run(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with np.load(dest / "display.npz") as saved:
        expected = np.arange(CHECK_START * CHECK_FPS,
                             (CHECK_START + CHECK_SECONDS) * CHECK_FPS) / CHECK_FPS
        np.testing.assert_array_equal(saved["t_sec"], expected)
    return {name: digest(dest / name) for name in ("overlay.mp4", "display.npz", "settled.npz")}


def main() -> None:
    """OFF確認を通過した後に、既存の成功5本を触らず再開する。"""
    import fcntl
    lock = (OUT / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.nice(max(0, 19 - os.nice(0)))
    save_json(CHECK / "runner.json", dict(pid=os.getpid(), nice=os.nice(0)))
    before, after = short_off("off_before"), short_off("off_after")
    result = dict(before=before, after=after, identical=before == after,
                  source=CHECK_SOURCE, start_sec=CHECK_START, duration_sec=CHECK_SECONDS,
                  frames=CHECK_SECONDS * CHECK_FPS)
    save_json(CHECK / "off_sha256.json", result)
    if not result["identical"]:
        raise RuntimeError("OFFの25秒SHA-256が不一致。fcX ONは未起動")
    failed = OUT / "renders/fcXG83vInDY/on"
    for name in ("render.log", "status.json"):
        shutil.copy2(failed / name, CHECK / ("failed_fcx_on_" + name))
    completed = run_one("fcXG83vInDY", "on")
    save_json(CHECK / "rerender_result.json", completed)
    if completed["state"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        if os.environ.get("E3B_BASELINE") == "1":
            baseline_modules()
        worker()
    else:
        main()
