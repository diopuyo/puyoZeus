"""E16を固定入力で再生し、事前登録の全基準と母数を保存する。"""
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
from scripts import run_e10c_exchange_replay_20260927 as runner
from scripts import report_e13_20260927 as report_base
from scripts.replay_exchange_event_20260926 import replay
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json
from scripts.run_e15_exchange_replay_20260928 import Trace
from src.exchange_event_tracker import ExchangeEventTracker

OUT = Path("logs/e16/on")
MODEL = Path("models/exchange_event_v4")
ALL_SOURCES = (*SOURCES, "zenchi", "review")
PREFIRE_START, PREFIRE_END, TERMINAL_START, TERMINAL_END = 2612., 2614.08, 2699.5, 2701.
LOSS_MAX, AGREEMENT_MIN, TERMINAL_MIN = .5518, .7955, .6
MAX_FALSE_DEATH, REGISTERED_DEATHS = 1, 28
MAX_REPLAY_WORKERS = 2


class LayerTrace(Trace):
    """モデルへ渡したcountと現在/予測層、同期の採用時刻を同時に記録する。"""
    def __init__(self) -> None:
        super().__init__()
        self.layers: list[dict] = []

    def __call__(self, overlay: Any, inputs: tuple) -> None:
        super().__call__(overlay, inputs)
        row = dict(overlay.tracker.layer_eval)
        row["sync_reasons"] = "|".join(row["sync_reasons"])
        row["sync_sec"] = json.dumps(row["sync_sec"])
        row["confirmed_dead_sides"] = "|".join(row["confirmed_dead_sides"])
        self.layers.append(row)


def directory(source: str) -> Path:
    """既存の集計器が読むディレクトリ構成を維持する。"""
    return OUT/source if source in ("zenchi", "review") else OUT/"renders"/source/"on"


def worker(source: str) -> None:
    """各プロセスで計装を独立させ、同じ補完済み記録だけを再生する。"""
    reach = runner.runner.reach
    reach.instrument(ExchangeEventTracker)
    reach.STATS, trace = defaultdict(reach.empty_stats), LayerTrace()
    reach.FRAMES.clear()
    dest = directory(source)
    status = replay(Path("logs/e16/records")/f"{source}.jsonl.gz", dest, MODEL, True, trace, e16=True)
    save_json(dest / "count_trace.json", trace.rows)
    with (dest / "review_layers.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace.layers[0]))
        writer.writeheader()
        writer.writerows(trace.layers)
    if source in SOURCES:
        rows = reach.event_rows(dest, source)
        save_json(OUT/"v2"/source/"reach.json", dict(rows=rows,
            summary=dict(source=source, frames=dict(reach.FRAMES), **reach.summary(rows))))
    print(json.dumps(status), flush=True)


def launch(source: str) -> None:
    """完了した入力は再実行せず、失敗単位だけ再開する。"""
    if (directory(source)/"review_layers.csv").exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/f"{source}.log").open("w") as stream:
        subprocess.run([sys.executable, "-m", "scripts.run_e16_replay_20260928", "--source", source],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def scenes() -> dict:
    """指定窓の平均・符号反転・終盤最小値を、同じ時刻のE15と比較する。"""
    baseline = json.loads(Path("logs/e15/on/review/count_trace.json").read_text())
    after = json.loads((OUT/"review/count_trace.json").read_text())
    first = [r for r in after if PREFIRE_START <= r["t_sec"] <= PREFIRE_END]
    old = {r["t_sec"]: r for r in baseline}
    assert len(first) == sum(PREFIRE_START <= r["t_sec"] <= PREFIRE_END for r in baseline)
    means = dict(E15=float(np.mean([old[r["t_sec"]]["p1"] for r in first])),
                 E16=float(np.mean([r["p1"] for r in first])))
    signs = np.sign([r["counts"][0][-1] for r in first])
    nonzero = signs[signs != 0]
    flips = int(np.count_nonzero(nonzero[1:] != nonzero[:-1]))
    transitions = [dict(t_sec=b["t_sec"], before=float(np.sign(a["counts"][0][-1])
        *np.expm1(abs(a["counts"][0][-1]))), after=float(np.sign(b["counts"][0][-1])
        *np.expm1(abs(b["counts"][0][-1])))) for a, b in zip(first, first[1:])
        if np.sign(a["counts"][0][-1])*np.sign(b["counts"][0][-1]) < 0]
    display = np.load(OUT/"review/display.npz")
    late = display["display_p1"][(display["t_sec"] >= TERMINAL_START)
                                  & (display["t_sec"] <= TERMINAL_END)]
    result = dict(prefire=dict(frames=len(first), means=means, sign_flips=flips, transitions=transitions),
        terminal=dict(frames=len(late), minimum=float(late.min()), mean=float(late.mean())))
    result["checks"] = dict(prefire_mean=means["E16"] > means["E15"], counter_sign_flips=flips == 0,
                             terminal_probability=result["terminal"]["minimum"] >= TERMINAL_MIN)
    return result


def layer_summary() -> dict:
    """理由別の現在層復帰回数と、死亡後の発火拒否件数を集計する。"""
    reasons, frames, rejected, fallback, combined = defaultdict(int), 0, 0, 0, 0
    for source in ALL_SOURCES:
        with (directory(source)/"review_layers.csv").open() as stream:
            for row in csv.DictReader(stream):
                frames += 1
                fallback += bool(row["prediction_reasons"])
                combined += bool(row["p1_prediction"]) and not row["prediction_reasons"]
                for reason in filter(None, row["prediction_reasons"].split("|")):
                    reasons[reason] += 1
        diag = json.loads((directory(source)/"events.diagnostics.json").read_text())
        rejected += diag["counts"].get("E16_fire_after_observed_death", 0)
    return dict(frames=frames, fallback_frames=fallback, combined_frames=combined,
                fallback_reasons=dict(reasons), rejected_after_death=rejected)


def report() -> None:
    """基準は固定し、未達を合格扱いせず同じMETRICSに出力する。"""
    runner.OUT = OUT
    runner.aggregate()
    helpers = report_base.helpers
    pooled = helpers.read(OUT/"v2/pooled.json")
    q = pooled["M3_q"]["on"]["groups"]["all"]
    agreement = helpers.agreement(OUT/"zenchi/display.npz", helpers.read(helpers.ZENCHI/"official_games.json"))
    report_base.death_metrics.OUT = OUT
    deaths = report_base.death_metrics.deaths()
    false = sum(r["false_positive"] is True for r in deaths)
    missing = sum(r["winner"] is None for r in deaths)
    fraction = false/max(1, len(deaths)-missing)
    cv = helpers.read(Path("logs/e16/train/METRICS.json"))
    scene = scenes()
    gates = dict(q_log_loss=q["log_loss"] <= LOSS_MAX, zenchi_agreement=agreement["agreement"] >= AGREEMENT_MIN,
        false_death=(false <= MAX_FALSE_DEATH and missing == 0
                     and fraction <= MAX_FALSE_DEATH/REGISTERED_DEATHS), cv=cv["passed"], **scene["checks"])
    metrics = dict(protocol=helpers.read(Path("logs/e16/PROTOCOL.json")), q=q, zenchi=agreement,
        deaths=dict(false=false, total=len(deaths), fraction=fraction, unlabelled=missing,
                    registered_denominator=REGISTERED_DEATHS, cases=deaths),
        scenes=scene, cv=cv, layers=layer_summary(), gates=gates, status="PASS" if all(gates.values()) else "FAIL")
    save_json(Path("logs/e16/METRICS.json"), metrics)


def main() -> None:
    """比較再生は独立2プロセスまでとし、完了単位ごとに出力を残す。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=ALL_SOURCES)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.source:
        worker(args.source)
        return
    if not args.report:
        with ThreadPoolExecutor(max_workers=MAX_REPLAY_WORKERS) as pool:
            list(pool.map(launch, ALL_SOURCES))
    report()


if __name__ == "__main__":
    main()
