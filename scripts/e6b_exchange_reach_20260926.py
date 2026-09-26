"""E5/E6/E6bの撃ち合い到達率・実表示滞在・阻害条件を同じ再生で測る。"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np

from scripts import replay_exchange_event_20260926 as replay_module
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json
from src.exchange_event_tracker import S3_END_QUIET_SEC, TIME_EPSILON_SEC
from src.ojama_accounting import CHAIN_TOTAL_MIN_SCORE

FPS = 30
END_SECONDS = 900.0
REACH_LIMIT = .8
DWELL_TOLERANCE_SEC = 2.0
COMPLETE_REASONS = ("confirmed_after_score", "activity_timeout")
STATS: dict = {}
FRAMES: Counter = Counter()


def load_e6() -> type:
    """差戻し時のコードを固定し、修正中も同じE6を再現する。"""
    import scripts.visualize_advantage_overlay as vao
    modules = {}
    for name in ("tracker", "overlay"):
        module = ModuleType("e6b_baseline_" + name)
        sys.modules[module.__name__] = module
        path = Path("logs/e6b/baseline") / (name + ".py")
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), vars(module))
        modules[name] = module
    modules["overlay"].ExchangeEventTracker = modules["tracker"].ExchangeEventTracker
    replay_module.ExchangeEventOverlay = modules["overlay"].ExchangeEventOverlay
    source = Path("logs/e6b/baseline/visualize.py").read_text(encoding="utf-8")
    names = ("_ExchangeEventPhysicalSensor", "_ExchangeEventEndSignals")
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name in names]
    namespace = vars(vao).copy()
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "E6_signals", "exec"), namespace)
    vao._ExchangeEventEndSignals = namespace["_ExchangeEventEndSignals"]
    return modules["tracker"].ExchangeEventTracker


def gate_state(tracker: Any, t_sec: float) -> str:
    """finish_frameの判定順で、実際に満たされていない条件を記録する。"""
    record = tracker.current
    if not record or not record.chains:
        return "no_exchange"
    if any(c.end_signal_sec is None for c in record.chains):
        return "missing_end"
    if any(c.score_ready_sec is None for c in record.chains):
        return "missing_score_ready"
    if any(t_sec - c.end_signal_sec + TIME_EPSILON_SEC < S3_END_QUIET_SEC for c in record.chains):
        return "end_quiet"
    ready = max(c.score_ready_sec for c in record.chains)
    if tracker._s3_sec is not None and ready <= tracker._s3_sec:
        return "already_evaluated"
    return "eligible"


def instrument(tracker_type: type) -> None:
    """表示も状態遷移も変えず、フレームを実際の撃ち合いIDへ割り当てる。"""
    original_finish = tracker_type.finish_frame
    original_score = tracker_type.observe_score
    original_update = replay_module.ExchangeEventOverlay.update

    def finish(self: Any, t_sec: float) -> None:
        self._reach_frame_id = self.current.exchange_id if self.current else None
        if self.current:
            row = STATS[self.current.exchange_id]
            row["last_gate"] = gate_state(self, t_sec)
            row["block_frames"][row["last_gate"]] += 1
        original_finish(self, t_sec)

    def score(self: Any, side: str, t: float, value: Any, *args: Any, **kwargs: Any) -> None:
        chain = self.latest_chain(side)
        previous = chain.display_score if chain else None
        small = (value is not None and previous is not None
                 and 0 < value - previous < CHAIN_TOTAL_MIN_SCORE)
        ready, source = (chain.score_ready_sec if chain else None), self.source
        last_activity = self._last_activity_sec
        row = STATS[self.current.exchange_id] if self.current else None
        original_score(self, side, t, value, *args, **kwargs)
        if small and row is not None:
            row["small_score_updates"] += 1
            row["small_score_timer_resets"] += int(self._last_activity_sec > last_activity)
            row["small_score_ready_resets"] += int(ready is not None and chain.score_ready_sec is None)
            row["small_score_s3_to_s1"] += int(source == "S3" and self.source == "S1")

    def update(self: Any, *args: Any, **kwargs: Any) -> None:
        original_update(self, *args, **kwargs)
        source = self.tracker.source
        FRAMES[source] += 1
        identity = getattr(self.tracker, "_reach_frame_id", None)
        if identity is not None:
            row = STATS[identity]
            row["source_frames"][source] += 1
            if source == "S3" and row["first_display_s3"] is None:
                row["first_display_s3"] = args[3]

    tracker_type.finish_frame, tracker_type.observe_score = finish, score
    replay_module.ExchangeEventOverlay.update = update


def empty_stats() -> dict:
    return dict(source_frames=Counter(), block_frames=Counter(), last_gate=None,
                first_display_s3=None, small_score_updates=0, small_score_timer_resets=0,
                small_score_ready_resets=0, small_score_s3_to_s1=0)


def event_rows(directory: Path, source: str) -> list[dict]:
    """終了理由別の全母数を残し、試合境界・区間末は打切りとして分ける。"""
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    rows = []
    for event in events:
        stats = STATS[event["exchange_id"]]
        s3 = [v["t_sec"] for v in event["values"] if v["source"] == "S3"]
        ends = [c["end_signal_sec"] for c in event["chains"]]
        all_ended = bool(ends) and all(t is not None for t in ends)
        reason = event["close_reason"] or "stream_end"
        rows.append(dict(source=source, exchange_id=event["exchange_id"], game_idx=event["game_idx"],
            trigger_sec=event["trigger_sec"], first_s3=min(s3) if s3 else None,
            closed_sec=event["closed_sec"], close_reason=reason, complete=reason in COMPLETE_REASONS,
            s1_seconds=stats["source_frames"]["S1"] / FPS,
            firing_to_last_end=max(ends) - event["trigger_sec"] if all_ended else None,
            chains=len(event["chains"]), missing_end=sum(t is None for t in ends),
            missing_score_ready=sum(c["score_ready_sec"] is None for c in event["chains"]), **stats))
    return rows


def summary(rows: list[dict]) -> dict:
    """正常終了と安全弁終了を完了母数とし、分子から静止直行を隠さない。"""
    complete = [r for r in rows if r["complete"]]
    reached = sum(r["first_s3"] is not None for r in complete)
    visible = sum(r["first_display_s3"] is not None for r in complete)
    durations = [r["firing_to_last_end"] for r in complete if r["firing_to_last_end"] is not None]
    dwell = float(np.median([r["s1_seconds"] for r in complete])) if complete else None
    ending = float(np.median(durations)) if durations else None
    fraction = reached / len(complete) if complete else 0.0
    visible_fraction = visible / len(complete) if complete else 0.0
    return dict(exchanges=len(rows), completed=len(complete), reached=reached,
        reach_fraction=fraction, display_reached=visible, display_reach_fraction=visible_fraction,
        s1_median_seconds=dwell, all_s1_median_seconds=float(np.median([r["s1_seconds"] for r in rows])),
        firing_to_last_end_median=ending, end_observed_count=len(durations),
        close_reasons=dict(Counter(r["close_reason"] for r in rows)),
        unreached_last_gate=dict(Counter(r["last_gate"] for r in complete if r["first_s3"] is None)),
        small_score={key: sum(r[key] for r in rows) for key in (
            "small_score_updates", "small_score_timer_resets", "small_score_ready_resets", "small_score_s3_to_s1")},
        reach_pass=fraction >= REACH_LIMIT, display_reach_pass=visible_fraction >= REACH_LIMIT,
        dwell_pass=bool(ending is not None and dwell <= ending + DWELL_TOLERANCE_SEC))


def existing_gates(out: Path, reach: dict) -> None:
    """到達・滞在未達時には既存ゲートだけで合格を出さない。"""
    from scripts import aggregate_e4_exchange_eval_20260926 as aggregate
    from scripts.run_e6_exchange_eval_20260926 import gates
    aggregate.OUT = out
    aggregate.main()
    checks = gates(out)
    checks.pop("all_pass")
    checks.update(reach=reach["reach_pass"], display_reach=reach["display_reach_pass"],
                  dwell=reach["dwell_pass"])
    checks["existing_gate_qualified"] = checks["reach"] and checks["dwell"]
    checks["all_pass"] = all(checks.values())
    save_json(out / "gates.json", checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("e5", "e6", "current"), default="current")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from src.exchange_event_tracker import ExchangeEventTracker
    tracker_type = ExchangeEventTracker
    if args.stage == "e5":
        from scripts import diagnose_e6_exchange_20260926 as e5
        e5.load_baseline()
        tracker_type = e5.ExchangeEventTracker
    elif args.stage == "e6":
        tracker_type = load_e6()
    instrument(tracker_type)
    all_rows, videos = [], []
    global STATS
    for source in SOURCES:
        STATS, _ = defaultdict(empty_stats), FRAMES.clear()
        out = args.out / "renders" / source / "on"
        status = replay_module.replay(Path("logs/e5/renders") / source / "on/inputs.jsonl.gz", out)
        if args.stage != "current":
            baseline = Path("logs/e5/replay" if args.stage == "e5" else "logs/e6/final")
            replay_module.compare(baseline / "renders" / source / "on", out)
        rows = event_rows(out, source)
        save_json(out / "reach_events.json", rows)
        all_rows.extend(rows)
        videos.append(dict(source=source, frames=dict(FRAMES), **summary(rows)))
        print(source, status, videos[-1], flush=True)
    result = dict(stage=args.stage, videos=videos, pooled=summary(all_rows))
    save_json(args.out / "reach_events.json", all_rows)
    save_json(args.out / "reach_summary.json", result)
    if args.stage == "current":
        existing_gates(args.out, result["pooled"])
    print(result["pooled"], flush=True)


if __name__ == "__main__":
    main()
