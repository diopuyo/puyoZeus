"""E4: 25秒OFF互換照合後、既存OFFを再用してON三本だけ再レンダする。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess

from scripts import run_e3_exchange_eval_20260926 as e3

OUT = e3.ROOT / "logs/e4"
CHECK_SOURCE = e3.SOURCES[0]
CHECK_START = 140
CHECK_SECONDS = 25
CHECK_FPS = 30
ARTIFACTS = ("overlay.mp4", "display.npz", "settled.npz")


def check_off() -> None:
    """E3b修正済みOFFの同一25秒成果物と、現コードの出力を比較する。"""
    import numpy as np
    before = e3.ROOT / "logs/e3/e3b/off_after"
    after = OUT / "off_check"
    after.mkdir(parents=True, exist_ok=True)
    args = e3.command(CHECK_SOURCE, "off", after, CHECK_SECONDS)
    args += ["--start-sec", str(CHECK_START)]
    with (after / "render.log").open("w", encoding="utf-8") as log:
        subprocess.run(args, cwd=e3.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    with np.load(after / "display.npz") as saved:
        np.testing.assert_array_equal(saved["t_sec"],
            np.arange(CHECK_START * CHECK_FPS, (CHECK_START + CHECK_SECONDS) * CHECK_FPS) / CHECK_FPS)
    hashes = {mode: {name: e3.digest(root / name) for name in ARTIFACTS}
              for mode, root in (("before", before), ("after", after))}
    identical = hashes["before"] == hashes["after"]
    e3.save_json(OUT / "off_sha256.json", dict(hashes, identical=identical,
        source=CHECK_SOURCE, start_sec=CHECK_START, duration_sec=CHECK_SECONDS))
    if not identical:
        raise RuntimeError("25秒OFFのSHA-256が不一致")


def main() -> None:
    """最大三並列・nice 19を固定し、セッション切断後も再開できる。"""
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.nice(max(0, 19 - os.nice(0)))
    e3.save_json(OUT / "runner.json", dict(pid=os.getpid(), nice=os.nice(0)))
    check_off()
    e3.OUT = OUT
    with ThreadPoolExecutor(max_workers=e3.WORKERS) as pool:
        futures = [pool.submit(e3.run_one, source, "on") for source in e3.SOURCES]
        results = [future.result() for future in futures]
    e3.save_json(OUT / "render_summary.json", results)
    if any(row["state"] != "completed" for row in results):
        raise SystemExit(1)
    from scripts.aggregate_e4_exchange_eval_20260926 import main as aggregate
    aggregate()


if __name__ == "__main__":
    main()
