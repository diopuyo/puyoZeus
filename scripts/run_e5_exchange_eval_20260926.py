"""E5の記録ONレンダ・OFF SHA・三本再生とE4集計を一括実行する。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time

from scripts import run_e3_exchange_eval_20260926 as e3
from scripts import run_e4_exchange_eval_20260926 as e4
from scripts.replay_exchange_event_20260926 import compare, replay

OUT = e3.ROOT / "logs/e5"
BASELINE = e3.ROOT / "logs/e4"
ORIGINAL_COMMAND = e3.command


def record_command(source: str, mode: str, dest: Path, seconds: int) -> list[str]:
    """ONだけ入力記録を追加し、既存固定条件を維持する。"""
    args = ORIGINAL_COMMAND(source, mode, dest, seconds)
    if mode == "on":
        args += ["--exchange-event-record", str(dest / "inputs.jsonl.gz")]
    return args


def replay_all() -> None:
    """三本を直列再生して実時間を測り、出力全列・イベント全体を照合する。"""
    from scripts import aggregate_e4_exchange_eval_20260926 as aggregate
    start, results = time.perf_counter(), []
    for source in e3.SOURCES:
        rendered = OUT / "renders" / source / "on"
        output = OUT / "replay/renders" / source / "on"
        result = replay(rendered / "inputs.jsonl.gz", output)
        result.update(source=source, equivalence=compare(rendered, output))
        result["recording_invariance"] = compare(BASELINE / "renders" / source / "on", rendered)
        before_sha = e3.digest(BASELINE / "renders" / source / "on/overlay.mp4")
        after_sha = e3.digest(rendered / "overlay.mp4")
        if before_sha != after_sha:
            raise AssertionError(f"記録有無の動画SHAが不一致: {source}")
        result["recording_invariance"]["video_sha256"] = after_sha
        results.append(result)
        e3.save_json(OUT / "replay_progress.json", results)
    e3.save_json(OUT / "replay_summary.json", dict(videos=results,
        elapsed_seconds=time.perf_counter() - start, execution="sequential"))
    aggregate.OUT = OUT / "replay"
    aggregate.main()


def main() -> None:
    """最大三並列・nice 19・排他ロックで長時間処理を継続する。"""
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.nice(max(0, 19 - os.nice(0)))
    e3.save_json(OUT / "runner.json", dict(pid=os.getpid(), nice=os.nice(0)))
    e4.OUT = OUT
    e4.check_off()
    e3.OUT, e3.command = OUT, record_command
    with ThreadPoolExecutor(max_workers=e3.WORKERS) as pool:
        results = list(pool.map(lambda source: e3.run_one(source, "on"), e3.SOURCES))
    e3.save_json(OUT / "render_summary.json", results)
    if any(row["state"] != "completed" for row in results):
        raise RuntimeError("記録レンダ失敗")
    replay_all()


if __name__ == "__main__":
    main()
