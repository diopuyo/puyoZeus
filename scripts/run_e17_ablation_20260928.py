"""E17: E15固定入力・固定採否でE16の各変更を独立再生する。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import numpy as np
from scripts import run_e10c_exchange_replay_20260927 as runner
from scripts import report_e13_20260927 as report_base
from scripts.replay_exchange_event_20260926 import replay, compare
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json
from scripts.run_e15_exchange_replay_20260928 import Trace
from src.exchange_event_tracker import ExchangeEventTracker

OUT = Path("logs/e17")
VARIANTS = dict(baseline={}, sync=dict(count_sync=True), death=dict(death_guard=True),
    layers=dict(evaluation_layers=True), all=dict(e16=True), fixed=dict(completion_check=True))
ALL_SOURCES = (*SOURCES, "zenchi", "review")
PREFIRE = (2612., 2614.08)
TERMINAL = (2699.5, 2701.)
LOSS_MAX, AGREEMENT_MIN, DEATHS = .5518 + .002, .7913 - .003, 28
WORKERS = 2


def directory(variant: str, source: str) -> Path:
    """既存集計器と同じ出力構造を用いる。"""
    root = OUT / variant
    return root/source if source in ("zenchi", "review") else root/"renders"/source/"on"


class AuditTrace(Trace):
    """盤面不一致の初回判定時点と、比較した側・時刻を保存する。"""
    def __init__(self, source: str) -> None:
        super().__init__()
        self.source = source
        self.reasons: Counter = Counter()
        self.comparisons: list[dict] = []
        self.seen: set[tuple] = set()
        self.frames = 0

    def __call__(self, overlay: Any, inputs: tuple) -> None:
        self.frames += 1
        if self.source == "review":
            super().__call__(overlay, inputs)
        layers = overlay._e16
        if layers is None or not layers.layer_enabled:
            return
        self.reasons.update(filter(None, overlay.tracker.layer_eval["prediction_reasons"].split("|")))
        record = overlay.tracker.current or overlay._landing_projection.death_record
        for chain in record.chains if record else ():
            sample = layers.completion.samples.get(chain.chain_id) if layers.completion_fix else None
            checked = sample.checked if sample else chain.chain_id in layers.checked
            key = (overlay._game, chain.chain_id, sample.end if sample else None)
            if not checked or key in self.seen:
                continue
            self.seen.add(key)
            idx = int(chain.side == "2P")
            actual = sample.board if sample else overlay._history[idx][-1].board._grid
            self.comparisons.append(dict(game=overlay._game, chain=chain.chain_id, side=chain.side,
                t_sec=inputs[3], end=chain.end_signal_sec, end_confirmed=chain.end_confirmed,
                board_sec=sample.stamp if sample else overlay._history[idx][-1].t_sec,
                post_end_drop_sec=chain.post_end_drop_sec, mismatch=chain.chain_id in layers.mismatches,
                actual=np.asarray(actual).tolist(), predicted=chain.predicted_final_board))


def worker(variant: str, source: str, options: dict) -> None:
    """独立プロセスで再生し、全条件に同じ補完済み入力を渡す。"""
    if variant in ("baseline", "all") and source != "review":
        reuse_reference(variant, source, options)
        return
    reach = runner.runner.reach
    reach.instrument(ExchangeEventTracker)
    reach.STATS, trace = defaultdict(reach.empty_stats), AuditTrace(source)
    reach.FRAMES.clear()
    model = "v4" if options.get("count_sync") or options.get("e16") else "v3"
    dest = directory(variant, source)
    result = replay(Path("logs/e16/records")/f"{source}.jsonl.gz", dest,
                    Path("models/exchange_event_"+model), True, trace, **options)
    if source == "review":
        save_json(dest/"count_trace.json", trace.rows)
        if variant in ("baseline", "all"):
            reference = Path("logs/e15/on" if variant == "baseline" else "logs/e16/on")/source
            save_json(dest/"equivalence.json", compare(reference, dest))
    save_json(dest/"audit.json", dict(frames=trace.frames, reasons=dict(trace.reasons),
                                    comparisons=trace.comparisons))
    if source in SOURCES:
        rows = reach.event_rows(dest, source)
        save_json(OUT/variant/"v2"/source/"reach.json", dict(rows=rows,
            summary=dict(source=source, frames=dict(reach.FRAMES), **reach.summary(rows))))
    save_json(dest/"DONE.json", dict(options=options, model=model, **result))
    print(variant, source, result, flush=True)


def reuse_reference(variant: str, source: str, options: dict) -> None:
    """不変のE15/E16条件は既存固定入力出力を再用し、ハッシュを残す。"""
    root = Path("logs/e15/on" if variant == "baseline" else "logs/e16/on")
    relative = Path(source) if source == "zenchi" else Path("renders")/source/"on"
    before, dest = root/relative, directory(variant, source)
    dest.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ("display.npz", "events.jsonl", "events.diagnostics.json", "status.json"):
        shutil.copyfile(before/name, dest/name)
        hashes[name] = hashlib.sha256((before/name).read_bytes()).hexdigest()
    if source in SOURCES:
        reach = OUT/variant/"v2"/source/"reach.json"
        reach.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/"v2"/source/"reach.json", reach)
    save_json(dest/"DONE.json", dict(reused_from=str(before), sha256=hashes, options=options))


def launch(task: tuple[str, str, dict]) -> None:
    """完了単位を保存して再開できるようにする。"""
    variant, source, options = task
    dest = directory(variant, source)
    if (dest/"DONE.json").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/"runner.log").open("w") as stream:
        subprocess.run([sys.executable, "-m", "scripts.run_e17_ablation_20260928",
            "--variant", variant, "--source", source, "--options", json.dumps(options)],
            stdout=stream, stderr=subprocess.STDOUT, check=True)


def locked_worker(variant: str, source: str, options: dict) -> None:
    """WSLの再開・補助実行が重なっても、同じ出力を同時に書かない。"""
    import fcntl
    dest = directory(variant, source)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/"RUN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (dest/"DONE.json").exists():
            assert json.loads((dest/"DONE.json").read_text())["options"] == options
            return
        worker(variant, source, options)


def scenes(variant: str) -> dict:
    """発火前62行と終盤45行を全条件で同じ時間窓から抽出する。"""
    root = directory(variant, "review")
    rows = json.loads((root/"count_trace.json").read_text())
    first = [r for r in rows if PREFIRE[0] <= r["t_sec"] <= PREFIRE[1]]
    signs = np.sign([r["counts"][0][-1] for r in first])
    nonzero = signs[signs != 0]
    data = np.load(root/"display.npz")
    late = data["display_p1"][(data["t_sec"] >= TERMINAL[0]) & (data["t_sec"] <= TERMINAL[1])]
    return dict(prefire_frames=len(first), prefire_mean=float(np.mean([r["p1"] for r in first])),
        margin_sign_flips=int(np.count_nonzero(nonzero[1:] != nonzero[:-1])),
        terminal_frames=len(late), terminal_min=float(late.min()))


def report(variant: str) -> dict:
    """事前登録の三条件だけで採用候補を決め、場面指標は別列に置く。"""
    runner.OUT = OUT/variant
    runner.aggregate()
    helpers = report_base.helpers
    pooled = helpers.read(OUT/variant/"v2/pooled.json")
    q = pooled["M3_q"]["on"]["groups"]["all"]
    agreement = helpers.agreement(directory(variant, "zenchi")/"display.npz",
                                  helpers.read(helpers.ZENCHI/"official_games.json"))
    report_base.death_metrics.OUT = OUT/variant
    deaths = report_base.death_metrics.deaths()
    false = sum(d["false_positive"] is True for d in deaths)
    missing = sum(d["winner"] is None for d in deaths)
    gates = dict(q=q["log_loss"] <= LOSS_MAX, zenchi=agreement["agreement"] >= AGREEMENT_MIN,
                 deaths=false <= 1 and missing == 0 and false/max(1, len(deaths)) <= 1/DEATHS)
    result = dict(variant=variant, q=q, zenchi=agreement,
        deaths=dict(false=false, total=len(deaths), unlabelled=missing),
        scenes=scenes(variant), gates=gates, candidate=all(gates.values()))
    save_json(OUT/variant/"METRICS.json", result)
    return result


def run() -> None:
    """単独条件の採否確定後に限り、採用候補だけの組合せを再生する。"""
    save_json(OUT/"PROTOCOL.json", dict(variants=VARIANTS, q_max=LOSS_MAX,
        zenchi_min=AGREEMENT_MIN, max_false=1, death_denominator=DEATHS,
        prefire=PREFIRE, terminal=TERMINAL, cv="同期countのみ既存同一行15foldを照合再集計",
        combination="①②③③′の合格条件のみ。③と③′が共に合格なら③′を優先"))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(launch, [(v, s, o) for v, o in VARIANTS.items() for s in ALL_SOURCES]))
    results = {v: report(v) for v in VARIANTS}
    selected = [v for v in ("sync", "death", "layers", "fixed") if results[v]["candidate"]]
    if "fixed" in selected and "layers" in selected:
        selected.remove("layers")
    options = {k: value for v in selected for k, value in VARIANTS[v].items()}
    save_json(OUT/"SELECTION.json", dict(selected=selected, options=options))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(launch, [("combined", s, options) for s in ALL_SOURCES]))
    results["combined"] = report("combined")
    save_json(OUT/"METRICS.json", dict(results=results, selected=selected))


def main() -> None:
    """再生子プロセスと全条件実行を切り替える。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant")
    parser.add_argument("--source", choices=ALL_SOURCES)
    parser.add_argument("--options")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.source:
        locked_worker(args.variant, args.source, json.loads(args.options) if args.options else VARIANTS[args.variant])
    elif args.report_only:
        print(json.dumps(report(args.variant), ensure_ascii=False))
    else:
        run()


if __name__ == "__main__":
    main()
