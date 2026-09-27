"""E10b登録基準と全区間の固定・誤固定を既存ラベルで照合する。"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

from scripts.aggregate_e3_exchange_eval_20260926 import outcomes, FPS
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json
from scripts.report_e9_exchange_20260927 import metric_rows

OUT = Path("logs/e10b")
ZENCHI = Path("logs/review_zenchi_part3")
CHECK_TIMES = (2694.916666666667, 2695.95, 2696.05, 2696.15, 2696.1833333333334, 2696.45, 2696.85, 2697.0, 2693.1833333333334, 2695.016666666667, 2695.516666666667,
               2696.016666666667, 2806.2166666666667, 2807.616666666667,
               2808.15, 2809.616666666667, 2810.616666666667, 2812.016666666667)
A_CONFIRMED_SEC = 2695.0
A_MAX_DELAY_SEC = .5
B_DOUBLE_SEC = 2807.616666666667
A_MAX_P2 = .35
B_MAX_P1 = .10
FALSE_POSITIVE_LIMIT = .05
A_HOLD_START_SEC = 2694.9
A_LANDING_SEC = 2696.1833333333334
FIXED_DEATH_PROBABILITY = .02


def read(path: Path) -> object:
    """保存済みの検証結果を読む。"""
    return json.loads(path.read_text(encoding="utf-8"))


def agreement(path: Path, games: list) -> dict:
    """E10と同じ終盤1/3・表示符号・フレーム加重で算出する。"""
    data = np.load(path)
    hits, frames = 0, 0
    for game in games:
        begin = game["start"] + (game["end"] - game["start"]) * 2 / 3
        mask = (data["t_sec"] >= begin) & (data["t_sec"] < game["end"])
        sign = 1 if game["winner"] == "1P" else -1
        hits += int(np.count_nonzero(data["display_adv"][mask] * sign > 0))
        frames += int(mask.sum())
    return dict(hits=hits, frames=frames, agreement=hits/frames)


def death_rows(directory: Path, source: str, windows: list) -> list:
    """固定した側と試合勝者を照合し、ラベルのない固定も件数から除外しない。"""
    rows = []
    for line in (directory / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        for side in ("1P", "2P"):
            values = [v for v in event["values"] if v["source"] == "unavoidable_death"
                      and side in v["dead_sides"]]
            if not values:
                continue
            t = values[0]["t_sec"]
            window = next((w for w in windows if w["start"] <= t < w["end"]), None)
            winner = window["winner"] if window else None
            rows.append(dict(source=source, exchange_id=event["exchange_id"],
                game_idx=event["game_idx"], side=side, first_sec=t, last_sec=values[-1]["t_sec"],
                winner=winner, false_positive=None if winner is None else winner == side,
                first=values[0]))
    return rows


def timeline() -> list:
    """A/Bの観測時刻を同一フレームで比較する。"""
    data = {"E10": np.load(Path("logs/e10/zenchi/display.npz")), "E10b": np.load(OUT / "zenchi/display.npz")}
    rows = []
    for t in CHECK_TIMES:
        row = dict(t_sec=t)
        for version, display in data.items():
            idx = int(np.argmin(np.abs(display["t_sec"] - t)))
            row[version] = dict(p1=float(display["display_p1"][idx]),
                adv=float(display["display_adv"][idx]), source=str(display["source"][idx]))
        rows.append(row)
    return rows


def detailed_timeline() -> None:
    """±5秒の入力診断へ修正後表示と実際に用いた受け量・応手量を付ける。"""
    rows = read(OUT / "diagnosis_e9.json")
    display = np.load(OUT / "zenchi/display.npz")
    events = [json.loads(s) for s in (OUT / "zenchi/events.jsonl").read_text().splitlines()]
    for row in rows:
        idx = int(np.argmin(np.abs(display["t_sec"]-row["t"])))
        row["E10b"] = dict(source=str(display["source"][idx]), p1=float(display["display_p1"][idx]),
                          display_adv=float(display["display_adv"][idx]))
        event = next((e for e in reversed(events) if e["trigger_sec"] <= row["t"]
                      and e["game_idx"] == row["game"]), None)
        if event is not None:
            row["projection"] = next((v for v in reversed(event["values"])
                if v["t_sec"] <= row["t"] and "incoming" in v), None)
    save_json(OUT / "AB_timeline.json", rows)


def hold_checks(display: object) -> dict:
    """Aの着地直前までとBの決着までを全表示フレームで照合する。"""
    a_mask = (display["t_sec"] >= A_HOLD_START_SEC) & (display["t_sec"] < A_LANDING_SEC)
    game = display["game_idx"][np.searchsorted(display["t_sec"], B_DOUBLE_SEC)]
    b_mask = (display["t_sec"] >= B_DOUBLE_SEC) & (display["game_idx"] == game)
    checks = dict(A_hold=bool(np.all(display["display_p1"][a_mask] == 1-FIXED_DEATH_PROBABILITY)),
                  B_hold=bool(np.all(display["display_p1"][b_mask] == FIXED_DEATH_PROBABILITY)))
    selected = ((display["t_sec"] >= A_HOLD_START_SEC) & (display["t_sec"] <= 2697)) | b_mask
    rows = [dict(t_sec=float(display["t_sec"][i]), p1=float(display["display_p1"][i]),
                 adv=float(display["display_adv"][i]), source=str(display["source"][i]))
            for i in np.flatnonzero(selected)]
    save_json(OUT / "AB_frames.json", dict(checks=checks, A_frames=int(a_mask.sum()),
        B_frames=int(b_mask.sum()), frames=rows))
    return checks


def main() -> None:
    """全指標と判定対象数をJSONへ保存する。"""
    games = read(ZENCHI / "official_games.json")
    before = agreement(Path("logs/e10/zenchi/display.npz"), games)
    after = agreement(OUT / "zenchi/display.npz", games)
    deaths = death_rows(OUT / "zenchi", "zenchi", games)
    panel = read(OUT / "panel_outcomes.json")
    for source in SOURCES:
        windows, _ = outcomes(source)
        seconds = [dict(start=w["start"]/FPS, end=w["end"]/FPS, winner=w["winner"]) for w in windows]
        deaths.extend(death_rows(OUT / "renders" / source / "on", source, seconds))
    for row in deaths:
        if row["winner"] is not None:
            continue
        label = next((r for r in panel if r["source"] == row["source"]
                      and r["game_idx"] == row["game_idx"]), None)
        if label is not None and label["winner"] is not None:
            row["winner"] = label["winner"]
            row["false_positive"] = label["winner"] == row["side"]
            row["provenance"] = "panel_outcomes.json"
    summary = dict(firings=len(deaths), false_positives=sum(r["false_positive"] is True for r in deaths),
                   unlabelled=sum(r["winner"] is None for r in deaths))
    summary["false_fraction"] = summary["false_positives"] / max(1, summary["firings"]-summary["unlabelled"])
    left, right = (read(Path(f"logs/{v}/v2/pooled.json")) for v in ("e10", "e10b"))
    metrics = [dict(metric=a, E10=b, E10b=d) for (a, b), (_, d) in zip(metric_rows(left), metric_rows(right))]
    display = np.load(OUT / "zenchi/display.npz")
    a_mask = (display["t_sec"] >= A_CONFIRMED_SEC) & (display["t_sec"] <= A_CONFIRMED_SEC+A_MAX_DELAY_SEC)
    b_game = display["game_idx"][np.searchsorted(display["t_sec"], B_DOUBLE_SEC)]
    b_mask = (display["t_sec"] >= B_DOUBLE_SEC) & (display["game_idx"] == b_game)
    gates = dict(A=bool(np.any(display["display_p1"][a_mask] >= 1-A_MAX_P2)),
        B=bool(np.all(display["display_p1"][b_mask] <= B_MAX_P1)),
        zenchi=after["agreement"] >= before["agreement"],
        deaths=summary["unlabelled"] == 0 and summary["false_fraction"] <= FALSE_POSITIVE_LIMIT,
        q_log_loss=right["M3_q"]["on"]["groups"]["all"]["log_loss"] <= left["M3_q"]["on"]["groups"]["all"]["log_loss"] + .005,
        flips=right["M4"]["on"]["flips"] <= left["M4"]["on"]["flips"] * 1.2)
    gates.update(hold_checks(display))
    report = dict(gates=gates, zenchi=dict(E10=before, E10b=after),
                  death_summary=summary, deaths=deaths, timeline=timeline(), metrics=metrics)
    save_json(OUT / "report.json", report)
    detailed_timeline()
    print(json.dumps({k: v for k, v in report.items() if k not in ("deaths", "timeline")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zenchi-only", action="store_true")
    options = parser.parse_args()
    if options.zenchi_only:
        print(json.dumps(dict(timeline=timeline(), agreement=agreement(
            OUT / "zenchi/display.npz", read(ZENCHI / "official_games.json"))), indent=2))
    else:
        main()
