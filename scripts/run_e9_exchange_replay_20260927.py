"""E8記録からE9を再生し、同じv2物差しと到達率を保存する。"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from scripts import e6b_exchange_reach_20260926 as reach
from scripts import report_e7_exchange_20260927 as metrics
from scripts.replay_exchange_event_20260926 import replay
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json
from src.exchange_event_tracker import ExchangeEventTracker

OUT = Path("logs/e9")
BEFORE = Path("logs/e8")


def summarize(source: str) -> tuple[dict, list, list]:
    """E8の固定データと同じラベル・v2定義を用いる。"""
    root = OUT / "renders" / source / "on"
    displays = dict(off=metrics.load_display(metrics.OFF / source / "off/display.npz"),
                    on=metrics.load_display(root / "display.npz"))
    for field in ("t_sec", "game_idx"):
        np.testing.assert_array_equal(displays["off"][field], displays["on"][field])
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    rows = metrics.m2_rows(events, displays)
    windows, provenance = metrics.e3.outcomes(source)
    first, _ = metrics.m1_events(events)
    result = dict(source=source, M1=first, M2=metrics.m2_summary(rows),
        M3={m: metrics.e3.m3_scores(d, windows) for m, d in displays.items()},
        M4={m: metrics.m4(d) for m, d in displays.items()},
        freshness={m: metrics.evaluation_freshness(d, m, metrics.e3.FPS)
                   for m, d in displays.items()}, outcome_provenance=provenance)
    save_json(OUT / "v2" / source / "events.json", metrics.e3.finite_json(rows))
    save_json(OUT / "v2" / source / "summary.json", metrics.e3.finite_json(result))
    return result, rows, events


def main() -> None:
    """全編描画なしで3動画を順に再生する。"""
    reach.instrument(ExchangeEventTracker)
    all_rows, videos = [], []
    for source in SOURCES:
        reach.STATS = defaultdict(reach.empty_stats)
        reach.FRAMES.clear()
        out = OUT / "renders" / source / "on"
        status = replay(BEFORE / "renders" / source / "on/inputs.jsonl.gz", out)
        rows = reach.event_rows(out, source)
        all_rows.extend(rows)
        videos.append(dict(source=source, frames=dict(reach.FRAMES), **reach.summary(rows)))
        print(source, status, flush=True)
    save_json(OUT / "reach_events.json", all_rows)
    save_json(OUT / "reach_summary.json", dict(videos=videos, pooled=reach.summary(all_rows)))
    summaries, rows, events = [], [], []
    for source in SOURCES:
        summary, video_rows, video_events = summarize(source)
        summaries.append(summary)
        rows.extend(video_rows)
        events.extend(video_events)
    metrics.OUT = OUT
    data = metrics.pooled(summaries, rows, events)
    before = json.loads((BEFORE / "v2/pooled.json").read_text())
    data["gates_off"] = metrics.gates(data, summaries)
    checks = dict(q_log_loss=data["M3_q"]["on"]["groups"]["all"]["log_loss"] <=
        before["M3_q"]["on"]["groups"]["all"]["log_loss"] + metrics.LOSS_TOLERANCE,
        flips=data["M4"]["on"]["flips"] <= before["M4"]["on"]["flips"] * metrics.FLIP_RATIO,
        early_s3=data["M1"]["revoked_fraction"] <= metrics.M1_LIMIT,
        freshness=data["gates_off"]["freshness"])
    data["regression_gates"] = dict(checks, all_pass=all(checks.values()))
    save_json(OUT / "v2/summary.json", metrics.e3.finite_json(summaries))
    save_json(OUT / "v2/pooled.json", metrics.e3.finite_json(data))
    print(data["regression_gates"], flush=True)


if __name__ == "__main__":
    main()
