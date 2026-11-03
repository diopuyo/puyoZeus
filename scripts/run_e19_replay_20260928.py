"""E19: 確率応手と死亡確定表示を独立の既定OFF条件で採点する。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from scripts import run_e17_ablation_20260928 as prior
from scripts.run_e18_counter_response_20260928 import halves
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path("logs/e19")
VARIANTS = {"hold": dict(death_guard=True, confirmed_death_hold=True),
            "prob": dict(death_guard=True, landing_counter_prob=True)}


def launch(task: tuple[str, str]) -> None:
    """完了済み入力は再利用し、プロセスごとの評価計装を隔離する。"""
    variant, source = task
    dest = prior.directory(variant, source)
    if (dest/"DONE.json").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/"runner.log").open("w") as stream:
        subprocess.run([sys.executable, "-m", "scripts.run_e19_replay_20260928",
            "--variant", variant, "--source", source], stdout=stream, stderr=subprocess.STDOUT, check=True)


def report(variant: str) -> dict:
    """指定した固定母数と、事前登録した条件だけで採否を決定する。"""
    baseline = json.loads(Path("logs/e18/METRICS.json").read_text())["baseline"]
    prior.LOSS_MAX = baseline["q"]["log_loss"] if variant == "hold" else .5313
    prior.AGREEMENT_MIN = baseline["zenchi"]["agreement"] if variant == "hold" else .801
    result = prior.report(variant)
    result["scenes"].update(halves(prior.directory(variant, "review")/"count_trace.json"))
    if variant == "prob":
        scene = result["scenes"]
        result["gates"].update(scene_mean=scene["prefire_mean"] > .270,
            scene_rise=scene["first_half_mean"] < scene["last_half_mean"])
    result["candidate"] = all(result["gates"].values())
    save_json(OUT/variant/"METRICS.json", result)
    return result


def main() -> None:
    """分類を完了してから独立条件ごとの固定5入力を実行する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--source", choices=prior.ALL_SOURCES)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    assert (OUT/"CLASSIFICATION.json").exists() and (OUT/"PROTOCOL.json").exists()
    prior.OUT = OUT
    if args.source:
        prior.worker(args.variant, args.source, VARIANTS[args.variant])
    elif args.report_only:
        report(args.variant)
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(launch, [(args.variant, s) for s in prior.ALL_SOURCES]))
        report(args.variant)


if __name__ == "__main__":
    main()
