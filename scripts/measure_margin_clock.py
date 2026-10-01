"""保存済み5記録を本番CLIで再生し、既存E36b採点器へ渡す。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

from scripts import replay_exchange_event_20260926 as replay_module
from scripts import run_e17_ablation_20260928 as prior
from scripts.run_e3_exchange_eval_20260926 import save_json
from src.margin_clock import MarginClock
from src.exchange_event_record import FIRST_PLACEMENT_ARGUMENT_INDEX
from src.scoring import compute_effective_rate

ROOT = Path("logs/margin_clock")
RECORDS = Path("logs/pending_expiry/full/records")
EXPECTED_Q = .507567
EXPECTED_ZENCHI_HITS = 7671
EXPECTED_DEATHS = 37
EXPECTED_SCENE = 2760.38
Q_TOLERANCE = .0000005
SCENE_TOLERANCE = .005


class ClockTrace:
    """実際に評価器へ渡った時計と、同時点で復元した設置起点を記録する。"""

    def __init__(self, source: str) -> None:
        self.trace = prior.AuditTrace(source)
        self.clock = MarginClock()
        self.games: dict[int, dict] = {}

    def __call__(self, overlay: Any, inputs: tuple) -> None:
        result, _, _, stamp, game, *_ = inputs
        self.trace(overlay, inputs)
        times = inputs[FIRST_PLACEMENT_ARGUMENT_INDEX] if len(inputs) > FIRST_PLACEMENT_ARGUMENT_INDEX else None
        self.clock.observe((result.p1, result.p2), stamp, game, times)
        row = self.games.setdefault(game, dict(game=game, frames=0, old_origin=None,
            first_placement=None, rate_difference_frames=0, applied_difference_frames=0))
        row["frames"] += 1
        row["old_origin"], row["first_placement"] = overlay._start, self.clock.origin
        old = 0. if overlay._start is None else max(0., stamp-overlay._start)
        candidate = self.clock.elapsed(stamp, overlay._start)
        row["rate_difference_frames"] += int(compute_effective_rate(old) != compute_effective_rate(candidate))
        actual = overlay._margin_elapsed(stamp)
        row["applied_difference_frames"] += int(compute_effective_rate(old) != compute_effective_rate(actual))


def worker(variant: str, source: str) -> None:
    """本番パーサーを通し、評価入力を変更しない監査だけを追加する。"""
    prior.OUT = ROOT / variant
    dest = prior.directory("on", source)
    dest.mkdir(parents=True, exist_ok=True)
    reach = prior.runner.runner.reach
    reach.instrument(prior.ExchangeEventTracker)
    reach.STATS, trace = defaultdict(reach.empty_stats), ClockTrace(source)
    reach.FRAMES.clear()
    original = replay_module.replay

    def measured(*args: Any, **kwargs: Any) -> dict:
        """旧表示の基準再現と現本番を分離し、解析用観測を保存する。"""
        kwargs["observer"] = trace
        if variant == "legacy_off":
            kwargs["switch_smoothing"] = False
        result = original(*args, **kwargs)
        save_json(dest / "margin_origins.json", list(trace.games.values()))
        if source == "review":
            save_json(dest / "count_trace.json", trace.trace.rows)
        if source in prior.SOURCES:
            rows = reach.event_rows(dest, source)
            save_json(prior.OUT / "on/v2" / source / "reach.json", dict(
                rows=rows, summary=dict(source=source, frames=dict(reach.FRAMES),
                                       **reach.summary(rows))))
        return result

    replay_module.replay = measured
    sys.argv = ["replay", str(RECORDS / f"{source}.jsonl.gz"), "--out", str(dest),
                "--production-exchange-event"]
    if variant == "on":
        sys.argv.append("--margin-origin-first-placement")
    replay_module.main()


def report(variant: str) -> None:
    """既存採点器の計算と母数をそのまま使う。"""
    from scripts import report_e36, report_e36b, report_e35
    report_e36b.OUT = ROOT / variant
    report_e36.RECORDS = RECORDS
    summary = report_e36b.report()
    report_e35.scene()
    if variant == "legacy_off":
        checks = dict(q=abs(summary["q"]["log_loss"] - EXPECTED_Q) <= Q_TOLERANCE,
            zenchi=summary["zenchi"]["hits"] == EXPECTED_ZENCHI_HITS,
            deaths=summary["deaths"]["false"] == 0 and summary["deaths"]["total"] == EXPECTED_DEATHS,
            scene=abs(summary["scene_first_sec"] - EXPECTED_SCENE) <= SCENE_TOLERANCE)
        save_json(ROOT / "BASELINE_REPRODUCED.json", checks)
        assert all(checks.values()), checks


def locked_worker(variant: str, source: str) -> None:
    """途中再開や個別実行でも同じ記録への二重書込を防ぐ。"""
    import fcntl
    root = ROOT / variant
    root.mkdir(parents=True, exist_ok=True)
    done = root / f"{source}.complete.json"
    with (root / f"{source}.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if not done.exists():
            worker(variant, source)
            save_json(done, dict(variant=variant, source=source, completed=True))


def main() -> None:
    """再生は1呼出し1動画とし、呼出元で最大並列数を管理する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("legacy_off", "off", "on"), required=True)
    parser.add_argument("--source", choices=prior.ALL_SOURCES)
    args = parser.parse_args()
    if args.source:
        locked_worker(args.variant, args.source)
    else:
        report(args.variant)


if __name__ == "__main__":
    main()
