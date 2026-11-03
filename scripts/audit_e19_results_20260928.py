"""E19の確定表示区間と、分類・学習・固定再生の成果物を集約する。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path("logs/e19")
SCENE_DEATH_SEC = 2698.98
PROJECTION_FIELDS = ("t_sec", "p1", "base_p1", "gfe_no_response_p1", "gfe_response_p1",
    "gfe_weighted_p1", "counter_probability", "counter_probability_features",
    "counter_probability_reasons", "incoming", "hands", "response_send", "response_incoming")


def terminal_span() -> dict:
    """最初の死亡確定表示から次の試合境界までを全行検査する。"""
    with np.load(OUT/"hold/review/display.npz") as held:
        mask = (held["t_sec"] >= SCENE_DEATH_SEC) & (held["source"] == "confirmed_death")
        first = int(np.flatnonzero(mask)[0])
        game = held["game_idx"][first]
        ids = np.flatnonzero((held["game_idx"] == game) & (held["t_sec"] >= held["t_sec"][first]))
        assert np.all(held["display_p1"][ids] == 1.)
        assert np.all(held["display_adv"][ids] == 100.)
        later = np.flatnonzero((held["t_sec"] > held["t_sec"][first]) & (held["game_idx"] != game))
        boundary = float(held["t_sec"][later[0]]) if len(later) else None
        return dict(first_sec=float(held["t_sec"][first]), last_sec=float(held["t_sec"][ids[-1]]),
            frames=len(ids), p1_min=1., next_boundary_sec=boundary)


def export_predictions() -> None:
    """レビューの予測更新DTOを両仮定・確率付きのCSVへ書き出す。"""
    root = OUT/"prob/review"
    events = [json.loads(line) for line in (root/"events.jsonl").read_text().splitlines()]
    fields = ("game_idx", "exchange_id", *PROJECTION_FIELDS)
    with (root/"counter_probability.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for event in events:
            for value in event["values"]:
                if "gfe_weighted_p1" in value:
                    row = {key: value.get(key) for key in PROJECTION_FIELDS}
                    writer.writerow(dict(game_idx=event["game_idx"], exchange_id=event["exchange_id"], **row))


def main() -> None:
    """基準値を変更せず、独立した実験結果をまとめる。"""
    baseline = json.loads(Path("logs/e18/METRICS.json").read_text())["baseline"]
    result = dict(baseline=baseline, terminal_span=terminal_span())
    for key, path in (("prob", "prob"), ("hold", "hold"), ("training", "train")):
        result[key] = json.loads((OUT/path/"METRICS.json").read_text())
    export_predictions()
    result["label_limitation"] = "148原票に実着弾時刻なし。識別可能な増加区間の代理ラベルであり全母集団の応手率ではない"
    save_json(OUT/"METRICS.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
