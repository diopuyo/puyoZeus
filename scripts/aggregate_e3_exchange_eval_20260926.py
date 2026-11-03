"""事前登録E3の同一フレーム比較とイベント時刻指標を集計する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from scripts.run_e3_exchange_eval_20260926 import MAX_SECONDS, OUT, SOURCES, digest, save_json

VERIFY = Path("/mnt/d/puyo_analyzer/verify")
FPS = 30
EXPECTED_FRAMES = MAX_SECONDS * FPS
SECONDS_PER_MINUTE = 60
SATURATION_SCORE = 99
PROBABILITY_EPSILON = 1e-7
EARLY_LIMIT = .2
LOSS_TOLERANCE = .005
FLIP_RATIO_LIMIT = 1.2
TARGET_FRACTION = .5
PHASE_COUNT = 3


def read_json(path: Path) -> Any:
    """日本語を含む既存成果物を読む。"""
    return json.loads(path.read_text(encoding="utf-8"))


def outcomes(source: str) -> tuple[list[dict], dict]:
    """既存の独立ラベルまたは認識由来ラベルを区間に固定する。"""
    base = VERIFY / f"judgment_unseen_{source}_2026-09-25_v1"
    if source == SOURCES[0]:
        paths = [base / "outcome_receipt_v1/RESULT.json",
                 base / "counter_binding_v1/RESULT.json", base / "counter_binding_v1/ASSIGNMENTS.json"]
        receipt = {r["range_id"]: r for r in read_json(paths[0])["rows"]}
        assignments = read_json(paths[2])
        windows = []
        for row in read_json(paths[1])["intervals"]:
            winner = receipt[row["range_id"]]["independently_observed_winner"]
            assigned = [r for r in assignments if r["range_id"] == row["range_id"]]
            assert assigned and all(r["conditional_winner"] == winner for r in assigned)
            windows.append(dict(start=row["start"], end=row["end_exclusive"], winner=winner))
    elif source == "fcXG83vInDY":
        paths = [base / "winner_endpoint_7290_v2/BINDING.json",
                 base / "first_game_5290_7218_v1/PLAN.json"]
        winner = read_json(paths[0])["new_existing_median_winner"]
        start, end = read_json(paths[1])["frames"]
        windows = [dict(start=start, end=end + 1, winner=winner)]
    else:
        paths = [base / "first_game_recovered_v1/RESULT.json",
                 base / "first_game_4020_5351_v2/PLAN.json"]
        winners = {r["conditional_posthoc_winner"] for r in read_json(paths[0])["rows"]}
        assert len(winners) == 1
        start, end = read_json(paths[1])["frames"]
        windows = [dict(start=start, end=end + 1, winner=winners.pop())]
    provenance = dict(kind="independent" if source == SOURCES[0] else "recognition_reference",
                      files={str(path): digest(path) for path in paths}, windows=windows)
    return windows, provenance


def m1_events(events: list[dict]) -> tuple[dict, list[dict]]:
    """最初のS3を固定し、後続の撤回・追加参加も元の撃ち合いへ数える。"""
    rows = []
    for event in events:
        s3 = [v["t_sec"] for v in event["values"] if v["source"] == "S3"]
        landing = [r["t_sec"] for r in event["landings"] if r["t_sec"] is not None]
        first = min(s3) if s3 else None
        revoked = [s["revoked_sec"] for c in event["chains"] for s in c["end_signals"]
                   if s.get("revoked_sec") is not None and first is not None and s["revoked_sec"] > first]
        continued = [c["observed_sec"] for c in event["chains"]
                     if first is not None and c["observed_sec"] > first]
        delta = first - min(landing) if first is not None and landing else None
        rows.append(dict(exchange_id=event["exchange_id"], game_idx=event["game_idx"],
                         s3_sec=first, first_landing_sec=min(landing) if landing else None,
                         delta_sec=delta, early=bool(revoked or continued),
                         revoked_sec=revoked, continuation_sec=continued))
    paired = [r for r in rows if r["delta_sec"] is not None]
    with_s3 = [r for r in rows if r["s3_sec"] is not None]
    early = sum(r["early"] for r in with_s3)
    fraction = early / len(with_s3) if with_s3 else None
    return dict(exchanges=len(events), s3_exchanges=len(with_s3), paired=len(paired),
                missing_s3=len(events) - len(with_s3), missing_pair=len(events) - len(paired),
                delta_median_sec=float(np.median([r["delta_sec"] for r in paired])) if paired else None,
                before_landing=sum(r["delta_sec"] < 0 for r in paired),
                before_landing_fraction=sum(r["delta_sec"] < 0 for r in paired) / len(paired) if paired else None,
                early=early, early_fraction=fraction,
                pass_threshold=fraction <= EARLY_LIMIT if fraction is not None else None), rows


def response_delay(display: dict, end: float, cutoff: float, target: float) -> float:
    """終了直前値から目標までの半分に達するまでを片方向で測る。"""
    stamps, probabilities = display["t_sec"], display["display_p1"]
    before = np.searchsorted(stamps, end, side="left") - 1
    if before < 0:
        return float("inf")
    base = probabilities[before]
    change = target - base
    if change == 0:
        return 0.0
    indices = np.flatnonzero((stamps >= end) & (stamps < cutoff))
    midpoint = base + TARGET_FRACTION * change
    reached = indices[(probabilities[indices] - midpoint) * np.sign(change) >= 0]
    return float(stamps[reached[0]] - end) if reached.size else float("inf")


def m2_events(events: list[dict], displays: dict) -> tuple[dict, list[dict]]:
    """同じ終了時刻・G_fe目標・探索区間で両モードを比較する。"""
    rows = []
    stamps, games = displays["on"]["t_sec"], displays["on"]["game_idx"]
    for event in events:
        row = dict(exchange_id=event["exchange_id"], game_idx=event["game_idx"])
        times = [c["score_finalize_sec"] for c in event["chains"]]
        targets = [v for v in event["values"] if v["source"] == "G_fe"]
        if not times or any(t is None for t in times):
            rows.append(dict(row, missing="score_finalize"))
            continue
        if not targets:
            rows.append(dict(row, missing="post_exchange_G_fe"))
            continue
        end = max(times)
        same_game = stamps[games == event["game_idx"]]
        next_events = [e["trigger_sec"] for e in events if e["game_idx"] == event["game_idx"]
                       and e["exchange_id"] > event["exchange_id"]]
        cutoff = min(next_events + [float(same_game[-1] + 1 / FPS)])
        target = min(targets, key=lambda v: v["t_sec"])
        delays = {mode: response_delay(display, end, cutoff, target["p1"])
                  for mode, display in displays.items()}
        rows.append(dict(row, end_sec=end, target_p1=target["p1"], target_sec=target["t_sec"],
                         cutoff_sec=cutoff, delays=delays))
    return m2_summary(rows), rows


def m2_summary(rows: list[dict]) -> dict:
    """未到達も含む個別遅延から中央値を取る。動画別中央値の平均は使わない。"""
    modes = ("off", "on")
    eligible = [r for r in rows if "delays" in r]
    medians = {mode: float(np.median([r["delays"][mode] for r in eligible])) if eligible else None
               for mode in modes}
    return dict(exchanges=len(rows), eligible=len(eligible),
                missing_finalize=sum(r.get("missing") == "score_finalize" for r in rows),
                missing_target=sum(r.get("missing") == "post_exchange_G_fe" for r in rows),
                unreached={mode: sum(np.isinf(r["delays"][mode]) for r in eligible) for mode in modes},
                median_seconds=medians,
                pass_threshold=medians["on"] < medians["off"] if eligible else None)


def m3_scores(display: dict, windows: list[dict]) -> dict:
    """区間内全フレームを採点し、無ラベルと単一クラスを区別する。"""
    frames = np.rint(display["t_sec"] * FPS).astype(int)
    labels = np.full(len(frames), np.nan)
    phases, game = np.full(len(frames), -1), np.full(len(frames), -1)
    for idx, window in enumerate(windows):
        mask = (frames >= window["start"]) & (frames < window["end"])
        labels[mask] = int(window["winner"] == "1P")
        phases[mask] = np.floor(PHASE_COUNT * (frames[mask] - window["start"]) /
                                (window["end"] - window["start"])).astype(int)
        game[mask] = idx
    valid = np.isfinite(labels)
    result = dict(all_frames=len(frames), unlabeled_frames=int((~valid).sum()), groups={})
    for name, mask in [("all", valid)] + [(f"P_time_{p + 1}", valid & (phases == p)) for p in range(PHASE_COUNT)]:
        y = labels[mask]
        probabilities = np.clip(display["display_p1"][mask], PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
        if not np.isfinite(probabilities).all():
            raise ValueError("表示勝率に非有限値があるためM3未成立")
        result["groups"][name] = dict(frames=int(mask.sum()), matches=len(np.unique(game[mask])),
                                    log_loss=float(log_loss(y, probabilities, labels=[0, 1])) if len(y) else None,
                                    auc=float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None)
    return result


def m4_stability(display: dict) -> dict:
    """全フレーム分母を保ち、試合内の0跨ぎも符号反転として数える。"""
    values, games = display["display_adv"], display["game_idx"]
    if not np.isfinite(values).all():
        raise ValueError("表示スコアの欠測でM4未成立")
    changes = np.flatnonzero(np.r_[True, games[1:] != games[:-1], True])
    flips = 0
    for start, end in zip(changes[:-1], changes[1:]):
        signs = np.sign(values[start:end])
        signs = signs[signs != 0]
        flips += int(np.count_nonzero(signs[1:] != signs[:-1]))
    saturated = int((np.abs(values) >= SATURATION_SCORE).sum())
    minutes = len(values) / FPS / SECONDS_PER_MINUTE
    return dict(frames=len(values), matches=len(np.unique(games)), minutes=minutes,
                saturated_frames=saturated, saturated_fraction=saturated / len(values),
                flips=flips, flips_per_minute=flips / minutes)


def finite_json(value: Any) -> Any:
    """未到達の無限大はJSON標準を保って文字列で記録する。"""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else None
    return value


def summarize(source: str) -> dict:
    """一動画を保存し、他動画の再計算と切り離す。"""
    root = OUT / "renders" / source
    displays = {}
    for mode in ("off", "on"):
        assert read_json(root / mode / "status.json")["state"] == "completed"
        with np.load(root / mode / "display.npz") as saved:
            displays[mode] = {key: saved[key].copy() for key in saved.files}
        np.testing.assert_array_equal(displays[mode]["t_sec"], np.arange(EXPECTED_FRAMES) / FPS)
    np.testing.assert_array_equal(displays["off"]["game_idx"], displays["on"]["game_idx"])
    events = [json.loads(line) for line in (root / "on/events.jsonl").read_text().splitlines()]
    windows, provenance = outcomes(source)
    first, first_rows = m1_events(events)
    second, second_rows = m2_events(events, displays)
    third = {mode: m3_scores(display, windows) for mode, display in displays.items()}
    fourth = {mode: m4_stability(display) for mode, display in displays.items()}
    diagnostic = root / "on/events.diagnostics.json"
    unknown = read_json(diagnostic)["counts"].get("unknown_firing_side", 0) if diagnostic.exists() else 0
    loss_off, loss_on = (third[mode]["groups"]["all"]["log_loss"] for mode in ("off", "on"))
    result = dict(source=source, frames=EXPECTED_FRAMES, event_matches=len({e["game_idx"] for e in events}),
                  M1=first, M2=second, M3=third, M4=fourth, outcome_provenance=provenance,
                  unknown_firing_side=unknown,
                  early_examples=sorted((r for r in first_rows if r["early"]), key=lambda r: r["s3_sec"])[:3],
                  thresholds=dict(M1=first["pass_threshold"], M2=second["pass_threshold"],
                      M3=loss_on <= loss_off + LOSS_TOLERANCE if source == SOURCES[0] else "reference",
                      M4=fourth["on"]["flips_per_minute"] <= fourth["off"]["flips_per_minute"] * FLIP_RATIO_LIMIT))
    destination = OUT / "metrics" / source
    save_json(destination / "event_metrics.json", finite_json(dict(M1=first_rows, M2=second_rows)))
    save_json(destination / "summary.json", finite_json(result))
    return result


def pooled_summary(summaries: list[dict]) -> dict:
    """独立勝敗のないM3を混ぜず、M1/M2/M4を全動画の分母で合算する。"""
    events, delays = [], []
    for data in summaries:
        source = data["source"]
        events.extend(json.loads(line) for line in
                      (OUT / "renders" / source / "on/events.jsonl").read_text().splitlines())
        rows = read_json(OUT / "metrics" / source / "event_metrics.json")["M2"]
        for row in rows:
            if "delays" in row:
                row["delays"] = {mode: float(value) for mode, value in row["delays"].items()}
        delays.extend(rows)
    first, _ = m1_events(events)
    second, fourth = m2_summary(delays), {}
    for mode in ("off", "on"):
        fields = ("frames", "matches", "minutes", "saturated_frames", "flips")
        row = {key: sum(data["M4"][mode][key] for data in summaries) for key in fields}
        row.update(saturated_fraction=row["saturated_frames"] / row["frames"],
                   flips_per_minute=row["flips"] / row["minutes"])
        fourth[mode] = row
    return dict(source="all", frames=sum(data["frames"] for data in summaries),
                event_matches=sum(data["event_matches"] for data in summaries),
                M1=first, M2=second, M4=fourth,
                thresholds=dict(M1=first["pass_threshold"], M2=second["pass_threshold"],
                    M4=fourth["on"]["flips_per_minute"] <= fourth["off"]["flips_per_minute"] * FLIP_RATIO_LIMIT))


def main() -> None:
    """全動画または指定動画だけを集計する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args()
    summaries = [summarize(source) for source in args.sources]
    save_json(OUT / "metrics/summary.json", finite_json(summaries))
    if tuple(args.sources) == SOURCES:
        statuses = [read_json(OUT / "renders" / source / mode / "status.json")
                    for mode in ("off", "on") for source in SOURCES]
        save_json(OUT / "render_summary.json", statuses)
        save_json(OUT / "metrics/pooled.json", finite_json(pooled_summary(summaries)))
    for summary in summaries:
        print(json.dumps(finite_json(summary), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
