"""E4の鮮度ゲートと撤回型M1を、E3の固定母集団・M2〜M4へ追加する。"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from scripts import aggregate_e3_exchange_eval_20260926 as e3
from scripts.run_e3_exchange_eval_20260926 import ROOT, SOURCES, save_json
from src.display_freshness import evaluation_freshness

OUT = ROOT / "logs/e4"
E3_OUT = ROOT / "logs/e3"
FRESHNESS_TOLERANCE = .05


def m1_events(events: list[dict]) -> tuple[dict, list[dict]]:
    """S3が根拠にした終了合図の撤回と、その後の追加参加を分離する。"""
    summary, rows = e3.m1_events(events)
    for event, row in zip(events, rows):
        s3 = [v["t_sec"] for v in event["values"] if v["source"] == "S3"]
        revoked = [signal["revoked_sec"] for chain in event["chains"]
                   for signal in chain["end_signals"] if "revoked_sec" in signal
                   and any(signal["t_sec"] <= t < signal["revoked_sec"] for t in s3)]
        row.update(revoked_sec=revoked, revoked=bool(revoked),
                   continued=bool(row["continuation_sec"]))
    count = summary["s3_exchanges"]
    for key in ("revoked", "continued"):
        summary[key] = sum(row[key] for row in rows)
        summary[key + "_fraction"] = summary[key] / count if count else None
    summary["pass_threshold"] = summary["revoked_fraction"] <= e3.EARLY_LIMIT if count else None
    return summary, rows


def freshness_gate(freshness: dict) -> bool:
    """両方の鮮度条件を満たすまでM3/M4の合否を出さない。"""
    off, on = freshness["off"], freshness["on"]
    return bool(on["equal_fraction"] <= off["equal_fraction"] + FRESHNESS_TOLERANCE
                and on["longest_equal_seconds"] <= off["longest_equal_seconds"])


def m2_targets(events: list[dict], display: dict) -> list[dict]:
    """早期に閉じた区間も、密な表示列から着地後最初のG_feを拾う。"""
    result = deepcopy(events)
    for event in result:
        landing = [r["t_sec"] for r in event["landings"] if r["t_sec"] is not None]
        cutoff = min([e["trigger_sec"] for e in events if e["game_idx"] == event["game_idx"]
                      and e["exchange_id"] > event["exchange_id"]] + [e3.MAX_SECONDS])
        start = min(landing) if landing else event["closed_sec"]
        event["values"] = [v for v in event["values"] if v["source"] != "G_fe"]
        if start is None:
            continue
        eligible = np.flatnonzero((display["t_sec"] >= start) & (display["t_sec"] < cutoff)
            & (display["game_idx"] == event["game_idx"]) & (display["source"] == "G_fe"))
        if len(eligible):
            idx = eligible[0]
            event["values"].append(dict(source="G_fe", t_sec=float(display["t_sec"][idx]),
                                        p1=float(display["display_p1"][idx])))
    return result


def summarize(source: str) -> dict:
    """ON新規・OFF既存の全27000フレームを照合して集計する。"""
    displays = {}
    for mode, root in (("off", E3_OUT), ("on", OUT)):
        directory = root / "renders" / source / mode
        assert e3.read_json(directory / "status.json")["state"] == "completed"
        with np.load(directory / "display.npz") as saved:
            displays[mode] = {key: saved[key].copy() for key in saved.files}
        np.testing.assert_array_equal(displays[mode]["t_sec"], np.arange(e3.EXPECTED_FRAMES) / e3.FPS)
    np.testing.assert_array_equal(displays["on"]["game_idx"], displays["off"]["game_idx"])
    directory = OUT / "renders" / source / "on"
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    windows, provenance = e3.outcomes(source)
    m1, rows1 = m1_events(events)
    m2, rows2 = e3.m2_events(m2_targets(events, displays["on"]), displays)
    m3 = {mode: e3.m3_scores(display, windows) for mode, display in displays.items()}
    m4 = {mode: e3.m4_stability(display) for mode, display in displays.items()}
    fresh = {mode: evaluation_freshness(display, mode, e3.FPS) for mode, display in displays.items()}
    gate = freshness_gate(fresh)
    thresholds = dict(F4=gate, M1=m1["pass_threshold"], M2=m2["pass_threshold"],
        M3=(m3["on"]["groups"]["all"]["log_loss"] <= m3["off"]["groups"]["all"]["log_loss"]
            + e3.LOSS_TOLERANCE if source == SOURCES[0] else "reference") if gate else None,
        M4=m4["on"]["flips_per_minute"] <= m4["off"]["flips_per_minute"] * e3.FLIP_RATIO_LIMIT if gate else None)
    diagnostics = e3.read_json(directory / "events.diagnostics.json")
    result = dict(source=source, frames=e3.EXPECTED_FRAMES,
        event_matches=len({event["game_idx"] for event in events}), M1=m1, M2=m2, M3=m3,
        M4=m4, F4=fresh, thresholds=thresholds, outcome_provenance=provenance,
        unknown_firing_side=diagnostics["counts"].get("unknown_firing_side", 0),
        sources={mode: {str(k): int(v) for k, v in zip(*np.unique(display["source"], return_counts=True))}
                 for mode, display in displays.items()})
    save_json(OUT / "metrics" / source / "event_metrics.json", e3.finite_json(dict(M1=rows1, M2=rows2)))
    save_json(OUT / "metrics" / source / "summary.json", e3.finite_json(result))
    return result


def pooled_summary(summaries: list[dict]) -> dict:
    """動画境界を隣接対として混ぜず、合算の分子分母を維持する。"""
    events, delays = [], []
    for data in summaries:
        source = data["source"]
        events.extend(json.loads(line) for line in
            (OUT / "renders" / source / "on/events.jsonl").read_text().splitlines())
        rows = e3.read_json(OUT / "metrics" / source / "event_metrics.json")["M2"]
        for row in rows:
            if "delays" in row:
                row["delays"] = {mode: float(value) for mode, value in row["delays"].items()}
        delays.extend(rows)
    m1, _ = m1_events(events)
    m2, m4, fresh = e3.m2_summary(delays), {}, {}
    for mode in ("off", "on"):
        fields = ("frames", "matches", "minutes", "saturated_frames", "flips")
        row = {key: sum(data["M4"][mode][key] for data in summaries) for key in fields}
        row.update(saturated_fraction=row["saturated_frames"] / row["frames"],
                   flips_per_minute=row["flips"] / row["minutes"])
        m4[mode] = row
        fields = ("frames", "adjacent_pairs", "equal_pairs", "updates", "minutes", "missing_frames")
        row = {key: sum(data["F4"][mode][key] for data in summaries) for key in fields}
        row.update(equal_fraction=row["equal_pairs"] / row["adjacent_pairs"],
            longest_equal_seconds=max(data["F4"][mode]["longest_equal_seconds"] for data in summaries),
            updates_per_minute=row["updates"] / row["minutes"])
        fresh[mode] = row
        row["column"] = summaries[0]["F4"][mode]["column"]
    gate = all(data["thresholds"]["F4"] for data in summaries)
    return dict(source="all", frames=sum(d["frames"] for d in summaries),
        event_matches=sum(d["event_matches"] for d in summaries), M1=m1, M2=m2, M4=m4, F4=fresh,
        thresholds=dict(F4=gate, M1=m1["pass_threshold"], M2=m2["pass_threshold"],
            M3=summaries[0]["thresholds"]["M3"] if gate else None,
            M4=m4["on"]["flips_per_minute"] <= m4["off"]["flips_per_minute"] * e3.FLIP_RATIO_LIMIT if gate else None))


def main() -> None:
    """結果JSONと全指標の検収表を保存する。"""
    summaries = [summarize(source) for source in SOURCES]
    pooled = pooled_summary(summaries)
    save_json(OUT / "metrics/summary.json", e3.finite_json(summaries))
    save_json(OUT / "metrics/pooled.json", e3.finite_json(pooled))
    from scripts import report_e4_exchange_eval_20260926 as reporting
    reporting.OUT = OUT
    reporting.report(e3.finite_json(summaries), e3.finite_json(pooled))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--off-root", type=Path, default=E3_OUT)
    options = parser.parse_args()
    OUT, E3_OUT = options.out, options.off_root
    main()
