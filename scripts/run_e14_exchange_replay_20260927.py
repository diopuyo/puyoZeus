"""E14の固定入力再生と事前登録ゲートを実行する。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from scripts import run_e10c_exchange_replay_20260927 as runner
from scripts import report_e13_20260927 as report_base
from scripts.replay_exchange_event_20260926 import replay
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json
from src.exchange_event_tracker import ExchangeEventTracker

OUT = Path("logs/e14")
MODEL = Path("models/exchange_event_v2")


def worker(source: str) -> None:
    """E13と同じ保存入力にv2だけを指定する。"""
    reach = runner.runner.reach
    reach.instrument(ExchangeEventTracker)
    reach.STATS = defaultdict(reach.empty_stats)
    reach.FRAMES.clear()
    record = (Path("logs/review_zenchi_part3/on_e10c/inputs.jsonl.gz") if source == "zenchi"
              else Path("logs/e8/renders") / source / "on/inputs.jsonl.gz")
    dest = OUT / ("zenchi" if source == "zenchi" else f"renders/{source}/on")
    print(replay(record, dest, MODEL), flush=True)
    if source != "zenchi":
        rows = reach.event_rows(dest, source)
        save_json(OUT / "v2" / source / "reach.json", dict(rows=rows,
            summary=dict(source=source, frames=dict(reach.FRAMES), **reach.summary(rows))))


def launch(source: str) -> None:
    """最大2本で再生し、描画用に1枠を残す。"""
    with (OUT / f"{source}.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "scripts.run_e14_exchange_replay_20260927",
                        "--source", source], stdout=log, stderr=subprocess.STDOUT, check=True)


def report() -> dict:
    """統括固定の5基準を同じ分母で比較する。"""
    runner.OUT = OUT
    runner.aggregate()
    helpers = report_base.helpers
    roots = dict(v1=Path("logs/e13/final"), v2=OUT)
    pooled = {v: helpers.read(p / "v2/pooled.json") for v, p in roots.items()}
    games = helpers.read(helpers.ZENCHI / "official_games.json")
    agreement = {v: helpers.agreement(p / "zenchi/display.npz", games) for v, p in roots.items()}
    spikes = {v: report_base.spikes(p) for v, p in roots.items()}
    deaths = {}
    for version, root in roots.items():
        report_base.death_metrics.OUT = root
        rows = report_base.death_metrics.deaths()
        labelled = [d for d in rows if d["winner"] is not None]
        false = sum(d["false_positive"] for d in labelled)
        deaths[version] = dict(total=len(rows), false=false, unlabelled=len(rows)-len(labelled),
                               fraction=false/max(1, len(labelled)))
    a, b = pooled["v1"], pooled["v2"]
    gates = dict(q_log_loss=b["M3_q"]["on"]["groups"]["all"]["log_loss"] <=
        a["M3_q"]["on"]["groups"]["all"]["log_loss"] + .005,
        zenchi=agreement["v2"]["agreement"] >= agreement["v1"]["agreement"]-.005,
        deaths=deaths["v2"]["unlabelled"] == 0 and deaths["v2"]["fraction"] <= .05,
        flips=b["M4"]["on"]["flips"] <= a["M4"]["on"]["flips"]*1.1,
        spikes=spikes["v2"]["per_minute"] <= spikes["v1"]["per_minute"])
    metrics = [dict(metric=n, v1=x, v2=y) for (n, x), (_, y) in
               zip(report_base.metric_rows(a), report_base.metric_rows(b))]
    result = dict(gates=gates, metrics=metrics, zenchi=agreement, spikes=spikes, deaths=deaths)
    save_json(OUT / "report.json", result)
    return result


def main() -> None:
    """再生完了後に全指標を保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["zenchi", *SOURCES])
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.source:
        worker(args.source)
        return
    save_json(OUT / "preregistered.json", dict(q_loss_tolerance=.005,
        zenchi_tolerance=.005, max_false_death=.05, flip_ratio=1.1, spike_ratio=1.0))
    if not args.report_only:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(launch, [*SOURCES, "zenchi"]))
    print(json.dumps(report(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
