"""E18: E15＋②の固定入力で打ち返し仮想着弾を事前登録基準に照合する。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import numpy as np
from scripts import run_e17_ablation_20260928 as prior
from scripts.replay_exchange_event_20260926 import replay, compare
from scripts.run_e3_exchange_eval_20260926 import save_json
from scripts.run_e15_exchange_replay_20260928 import Trace
from src.exchange_event_tracker import ExchangeEventTracker

OUT = Path("logs/e18")
LOSS_MAX, AGREEMENT_MIN = .5293 + .002, .8040 - .003
SCENE_ROWS, HALF_ROWS = 62, 31
CSV_FIELDS = ("t_sec", "game_idx", "source", "p1", "current_gfe_p1", "gfe_no_response_p1",
    "gfe_response_p1", "response_selected", "response_layer", "incoming", "hands",
    "response_send", "response_incoming", "response_surplus", "response_board_sec")


class CounterTrace(Trace):
    """時系列CSVと指定場面の採点に同じ更新直後の値を使う。"""
    def __init__(self, stream: Any, source: str) -> None:
        super().__init__()
        self.source = source
        self.writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        self.writer.writeheader()

    def __call__(self, overlay: Any, inputs: tuple) -> None:
        if self.source == "review":
            super().__call__(overlay, inputs)
        landing, tracker = overlay._landing_projection, overlay.tracker
        record = tracker.current or landing.death_record
        projection = landing.last if record and landing.identity == (inputs[4], record.exchange_id) else None
        row = {k: (projection or {}).get(k) for k in CSV_FIELDS}
        row.update(t_sec=inputs[3], game_idx=inputs[4], source=tracker.source,
                   p1=tracker.probability, current_gfe_p1=tracker._static_probability)
        self.writer.writerow(row)


def worker(source: str, enabled: bool = True) -> None:
    """認識とモデルを固定し、OFFではE17組合せへのバイト一致も要求する。"""
    prior.OUT = OUT
    variant = "on" if enabled else "off"
    dest = prior.directory(variant, source)
    dest.mkdir(parents=True, exist_ok=True)
    reach = prior.runner.runner.reach
    reach.instrument(ExchangeEventTracker)
    reach.STATS = defaultdict(reach.empty_stats)
    reach.FRAMES.clear()
    with (dest/"counter_response.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        trace = CounterTrace(stream, source)
        status = replay(Path("logs/e16/records")/f"{source}.jsonl.gz", dest,
            Path("models/exchange_event_v3"), True, trace, death_guard=True,
            landing_counter_response=enabled)
    if source == "review":
        save_json(dest/"count_trace.json", trace.rows)
    if not enabled:
        save_json(dest/"equivalence.json", compare(Path("logs/e17/combined/review"), dest))
    if source in prior.SOURCES:
        rows = reach.event_rows(dest, source)
        save_json(OUT/variant/"v2"/source/"reach.json", dict(rows=rows,
            summary=dict(source=source, frames=dict(reach.FRAMES), **reach.summary(rows))))
    save_json(dest/"DONE.json", status)


def launch(source: str) -> None:
    """各再生を独立プロセスへ渡す。"""
    dest = prior.directory("on", source)
    if (dest/"DONE.json").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/"runner.log").open("w") as stream:
        subprocess.run([sys.executable, "-m", "scripts.run_e18_counter_response_20260928",
            "--source", source], stdout=stream, stderr=subprocess.STDOUT, check=True)


def halves(path: Path) -> dict:
    """事前指定の62行を時刻順に31行ずつ比較する。"""
    rows = json.loads(path.read_text())
    values = [r["p1"] for r in rows if prior.PREFIRE[0] <= r["t_sec"] <= prior.PREFIRE[1]]
    assert len(values) == SCENE_ROWS
    return dict(first_half_mean=float(np.mean(values[:HALF_ROWS])),
                last_half_mean=float(np.mean(values[HALF_ROWS:])), half_frames=HALF_ROWS)


def report() -> None:
    """三つの全体指標と、平均・前後半の場面指標を全て合否に使う。"""
    prior.OUT, prior.LOSS_MAX, prior.AGREEMENT_MIN = OUT, LOSS_MAX, AGREEMENT_MIN
    result = prior.report("on")
    baseline = json.loads(Path("logs/e17/combined/METRICS.json").read_text())
    for value, root in ((result, OUT/"on"), (baseline, Path("logs/e17/combined"))):
        value["scenes"].update(halves(root/"review/count_trace.json"))
    scene = result["scenes"]
    result["gates"].update(scene_mean=scene["prefire_mean"] > baseline["scenes"]["prefire_mean"],
        scene_rise=scene["first_half_mean"] < scene["last_half_mean"])
    result["candidate"] = all(result["gates"].values())
    save_json(OUT/"on/METRICS.json", result)
    save_json(OUT/"METRICS.json", dict(baseline=baseline, counter_response=result))


def main() -> None:
    """採点前に基準を保存してから固定3動画とzenchi・場面記録を再生する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=prior.ALL_SOURCES)
    parser.add_argument("--off", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    prior.OUT = OUT
    if args.source:
        variant = "off" if args.off else "on"
        if not (prior.directory(variant, args.source)/"DONE.json").exists():
            worker(args.source, not args.off)
    elif args.report_only:
        report()
    else:
        save_json(OUT/"PROTOCOL.json", dict(baseline="E15+② / E17 combined / 75b0cbd",
            q_loss_max=LOSS_MAX, zenchi_agreement_min=AGREEMENT_MIN, false_max=1,
            death_denominator=28, scene=prior.PREFIRE, frames=SCENE_ROWS, half_frames=HALF_ROWS,
            scene_mean="E17 combinedの未丸め平均より大", scene_rise="前半31行平均 < 後半31行平均",
            selection="応手量>=純受け量の場合だけ応手後G_feを既存S3合成へ渡す",
            model="models/exchange_event_v3", inputs="logs/e16/records", default=False))
        with ThreadPoolExecutor(max_workers=prior.WORKERS) as pool:
            list(pool.map(launch, prior.ALL_SOURCES))
        report()


if __name__ == "__main__":
    main()
