"""E15の固定入力をOFF/ON再生し、基準b〜dを同じMETRICSへ保存する。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from scripts import run_e10c_exchange_replay_20260927 as runner
from scripts import report_e13_20260927 as report_base
from scripts.replay_exchange_event_20260926 import replay, compare
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json
from src.exchange_event_tracker import ExchangeEventTracker

OUT = Path("logs/e15")
MODEL = Path("models/exchange_event_v3")
REVIEW = Path("logs/review_zenchi_g41_43_e14")
PREFIRE_SEC, FIRE_SEC, LATE_SEC = 2613.9, 2614.08, 2694.3
PREFIRE_WINDOW = 2.
NF_MIN, JUMP_MAX, LATE_PROBABILITY_MIN = 700, .66, .566
LOSS_MAX, AGREEMENT_MIN, DEATH_DENOMINATOR = .5708, .7955, 28
TIME_TOLERANCE = 1e-6


class Trace:
    """表示でなくモデルに渡ったcountと、各側の確定入力変更を記録する。"""
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.keys: tuple | None = None

    def __call__(self, overlay: Any, inputs: tuple) -> None:
        tracker = overlay.tracker
        if tracker.current is None or getattr(tracker, "count_sides", None) is None:
            return
        latest = [h[-1] for h in overlay._history]
        keys = tuple((s.board._grid.tobytes(), s.queue.tobytes()) for s in latest)
        changed = self.keys is not None and self.keys != keys
        observation = tracker.firing.count_observation
        self.rows.append(dict(t_sec=inputs[3], game=inputs[4], exchange=tracker.current.exchange_id,
            p1=tracker.probability, source=tracker.source, changed=changed,
            observed_sec=observation.elapsed_sec + overlay._start,
            counts=tracker.count_sides.tolist(), confirmed_sec=[s.t_sec for s in latest]))
        self.keys = keys


def worker(source: str, output: Path, model: Path, live: bool) -> None:
    """既存3動画・zenchi全編・E14レビューを同じ記録再生器へ渡す。"""
    reach = runner.runner.reach
    reach.instrument(ExchangeEventTracker)
    reach.STATS = defaultdict(reach.empty_stats)
    reach.FRAMES.clear()
    if source == "review":
        record, dest = REVIEW / "inputs.jsonl.gz", output / "review"
    elif source == "zenchi":
        record = Path("logs/review_zenchi_part3/on_e10c/inputs.jsonl.gz")
        dest = output / "zenchi"
    else:
        record = Path("logs/e8/renders") / source / "on/inputs.jsonl.gz"
        dest = output / "renders" / source / "on"
    trace = Trace()
    print(replay(record, dest, model, live, trace), flush=True)
    save_json(dest / "count_trace.json", trace.rows)
    if source in SOURCES:
        rows = reach.event_rows(dest, source)
        save_json(output / "v2" / source / "reach.json", dict(rows=rows,
            summary=dict(source=source, frames=dict(reach.FRAMES), **reach.summary(rows))))
    if source == "review" and not live and model.name == "exchange_event_v2":
        save_json(dest / "off_equivalence.json", compare(REVIEW, dest))


def scene(rows: list[dict]) -> dict:
    """指定時刻の直前観測を使い、発火前の上昇と発火ジャンプを独立に出す。"""
    def before(stamp: float) -> dict:
        return next(r for r in reversed(rows) if r["t_sec"] <= stamp)
    pre, late = before(PREFIRE_SEC), before(LATE_SEC)
    after = next(r for r in rows if r["t_sec"] >= FIRE_SEC)
    previous = next(r for r in reversed(rows) if r["t_sec"] < FIRE_SEC)
    window = [r for r in rows if PREFIRE_SEC-PREFIRE_WINDOW <= r["t_sec"] <= PREFIRE_SEC]
    steps = [r for i, r in enumerate(window) if i == 0 or r["p1"] != window[i-1]["p1"]]
    increases = sum(b["p1"] > a["p1"] for a, b in zip(steps, steps[1:]))
    values = dict(nf_1p=float(np.expm1(pre["counts"][0][0])), prefire_p1=pre["p1"],
        prefire_start_p1=window[0]["p1"], prefire_window_sec=PREFIRE_WINDOW,
        prefire_steps=[dict(t_sec=r["t_sec"], p1=r["p1"]) for r in steps],
        prefire_increases=increases, prefire_frames=len(window),
        fire_jump=after["p1"]-previous["p1"], fire_sec=after["t_sec"],
        late_p1=late["p1"], late_sec=late["t_sec"], frames=len(rows))
    values["checks"] = dict(nf=values["nf_1p"] >= NF_MIN,
        prefire_rise=increases >= 2 and window[-1]["p1"] > window[0]["p1"],
        fire_jump=values["fire_jump"] < JUMP_MAX, late_probability=late["p1"] > LATE_PROBABILITY_MIN)
    values["passed"] = all(values["checks"].values())
    return values


def freshness(off: list[dict], on: list[dict]) -> dict:
    """同時刻同区間の全count列を比較し、STABLE変更後の追従も別に計数する。"""
    baseline = {(r["t_sec"], r["game"], r["exchange"]): r for r in off}
    equal, total, changed, fresh, nf_equal = 0, 0, 0, 0, 0
    for row in on:
        old = baseline.get((row["t_sec"], row["game"], row["exchange"]))
        if old is None:
            continue
        total += 1
        equal += int(np.array_equal(row["counts"], old["counts"], equal_nan=True))
        nf_equal += int(np.array_equal(np.array(row["counts"])[:, :-1],
                                      np.array(old["counts"])[:, :-1], equal_nan=True))
        if row["changed"]:
            changed += 1
            fresh += int(abs(row["observed_sec"] - row["t_sec"]) < TIME_TOLERANCE)
    return dict(frames=total, E14_OFF=dict(equal=total, rate=1. if total else None),
        E15_ON=dict(equal=equal, rate=equal/total if total else None,
                    nf_equal=nf_equal, nf_rate=nf_equal/total if total else None),
        stable_updates=changed, refreshed_updates=fresh,
        passed=bool(total and equal < total and changed and fresh == changed))


def report(output: Path) -> dict:
    """基準b〜dの全分母と採否をMETRICSに追記する。"""
    runner.OUT = output
    runner.aggregate()
    helpers = report_base.helpers
    pooled = helpers.read(output / "v2/pooled.json")
    q = pooled["M3_q"]["on"]["groups"]["all"]
    agreement = helpers.agreement(output / "zenchi/display.npz", helpers.read(helpers.ZENCHI / "official_games.json"))
    report_base.death_metrics.OUT = output
    deaths = report_base.death_metrics.deaths()
    unlabelled = sum(d["winner"] is None for d in deaths)
    false = sum(d["false_positive"] is True for d in deaths)
    labelled = len(deaths) - unlabelled
    false_fraction = false / max(1, labelled)
    checks = dict(q_log_loss=q["log_loss"] <= LOSS_MAX,
        zenchi=agreement["agreement"] >= AGREEMENT_MIN,
        deaths=false_fraction <= 1 / DEATH_DENOMINATOR and labelled > 0 and not unlabelled)
    metrics = helpers.read(OUT / "METRICS.json")
    metrics["b"] = dict(q=q, zenchi=agreement, deaths=dict(false=false, total=len(deaths),
        unlabelled=unlabelled, fraction=false_fraction, registered_denominator=DEATH_DENOMINATOR),
        checks=checks, passed=all(checks.values()))
    on = helpers.read(output / "review/count_trace.json")
    off = helpers.read(OUT / "off/review/count_trace.json")
    metrics["c"] = scene(on)
    metrics["d"] = freshness(off, on)
    metrics["status"] = "PASS" if all(metrics[k]["passed"] for k in "abcd") else "FAIL"
    save_json(OUT / "METRICS.json", metrics)
    return metrics


def launch(source: str, output: Path, model: Path) -> None:
    """再生は独立プロセスで起動し、完了入力は再利用する。"""
    directory = output / source if source in ("review", "zenchi") else output / "renders" / source / "on"
    if (directory / "count_trace.json").exists() and (directory / "status.json").exists():
        status = json.loads((directory / "status.json").read_text())
        if status.get("live_count") and status.get("model_dir") == str(model):
            return
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{source}.log").open("w") as stream:
        subprocess.run([sys.executable, "-m", "scripts.run_e15_exchange_replay_20260928",
            "--source", source, "--output", str(output), "--model", str(model), "--live-count"],
            stdout=stream, stderr=subprocess.STDOUT, check=True)


def run_all(output: Path, model: Path) -> None:
    """オラクル超過時は再生を開始せず、通常時だけ固定5入力を完了する。"""
    metrics = json.loads((OUT / "METRICS.json").read_text())
    if metrics["a"]["leak_suspected"]:
        raise RuntimeError("AUCがオラクル上限を超過したため、指定どおり後続評価を停止")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda source: launch(source, output, model), [*SOURCES, "zenchi", "review"]))


def main() -> None:
    """各入力の再生を独立起動し、未完了評価を合格にしない。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=[*SOURCES, "zenchi", "review"])
    parser.add_argument("--output", type=Path, default=OUT / "on")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--live-count", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.source:
        worker(args.source, args.output, args.model, args.live_count)
    else:
        if args.all:
            run_all(args.output, args.model)
        print(json.dumps(report(args.output), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
