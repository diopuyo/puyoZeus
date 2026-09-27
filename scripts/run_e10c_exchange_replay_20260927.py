"""固定E8入力からE10cを再生し、同一定義の全指標を出力する。"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import subprocess
import sys

from scripts import run_e9_exchange_replay_20260927 as runner
from scripts.replay_exchange_event_20260926 import replay
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json
from src.exchange_event_tracker import ExchangeEventTracker

OUT = Path("logs/e10c")
MAX_REPLAY_WORKERS = 2  # 別レーンのzenchiと合わせて最大3プロセス。


def worker(source: str) -> None:
    """動画ごとに独立プロセスで計装状態を隔離する。"""
    runner.OUT = OUT
    reach = runner.reach
    reach.instrument(ExchangeEventTracker)
    reach.STATS = defaultdict(reach.empty_stats)
    reach.FRAMES.clear()
    out = OUT / "renders" / source / "on"
    status = replay(Path("logs/e8/renders") / source / "on/inputs.jsonl.gz", out)
    rows = reach.event_rows(out, source)
    save_json(OUT / "v2" / source / "reach.json", dict(rows=rows,
        summary=dict(source=source, frames=dict(reach.FRAMES), **reach.summary(rows))))
    print(source, status, flush=True)


def launch(source: str) -> None:
    """標準出力と異常終了を動画別に保存する。"""
    with (OUT / f"{source}.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "scripts.run_e10c_exchange_replay_20260927",
                        "--source", source], stdout=log, stderr=subprocess.STDOUT, check=True)


def aggregate() -> None:
    """既存の集計関数と固定分母を使い、E10b比較を保存する。"""
    runner.OUT, runner.metrics.OUT = OUT, OUT
    summaries, rows, events, reaches, videos = [], [], [], [], []
    for source in SOURCES:
        summary, video_rows, video_events = runner.summarize(source)
        summaries.append(summary)
        rows.extend(video_rows)
        events.extend(video_events)
        reached = json.loads((OUT / "v2" / source / "reach.json").read_text())
        reaches.extend(reached["rows"])
        videos.append(reached["summary"])
    save_json(OUT / "reach_events.json", reaches)
    save_json(OUT / "reach_summary.json", dict(videos=videos, pooled=runner.reach.summary(reaches)))
    data = runner.metrics.pooled(summaries, rows, events)
    data["gates_off"] = runner.metrics.gates(data, summaries)
    save_json(OUT / "v2/summary.json", runner.metrics.e3.finite_json(summaries))
    save_json(OUT / "v2/pooled.json", runner.metrics.e3.finite_json(data))


def main() -> None:
    """独立入力を最大2プロセスで再生してから集計する。"""
    runner.metrics.FLIP_RATIO = 1.2
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=SOURCES)
    options = parser.parse_args()
    if options.source:
        worker(options.source)
        return
    with ThreadPoolExecutor(max_workers=MAX_REPLAY_WORKERS) as pool:
        list(pool.map(launch, SOURCES))
    aggregate()


if __name__ == "__main__":
    main()
