"""E10目視指摘区間の入力・E9表示・近未来火力を保存する。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.exchange_event_record import read_records
from src.indicators_v2 import near_future_fire_power
from src.scoring import score_to_ojama
from src.ojama_accounting import ON_FIELD_CAP

ROOT = Path("logs/review_zenchi_part3/on_e9")
WINDOWS = ((2690.0, 2701.0), (2805.6, 2815.6))
OUT = Path("logs/e10")


def confirmed_input(events: list, t: float, game: int, elapsed: float) -> dict:
    """将来確定する得点を使わず、その時刻までにE9へ渡った段累積量を示す。"""
    event = next((e for e in reversed(events) if e["game_idx"] == game
                  and e["trigger_sec"] <= t and (e["closed_sec"] is None or e["closed_sec"] >= t)), None)
    if event is None:
        return dict(cumulative_scores=None, confirmed_incoming=None)
    scores = next((v["score_totals"] for v in reversed(event["values"])
                   if v["t_sec"] <= t and "score_totals" in v), [0, 0])
    generated = [int(score_to_ojama(s, elapsed_sec=elapsed).ojama_count) for s in scores]
    return dict(cumulative_scores=scores,
                confirmed_incoming=[max(0, generated[1]-generated[0]), max(0, generated[0]-generated[1])])


def main() -> None:
    """変化フレームを取り出し、同じ確定盤面の探索を再用する。"""
    display = np.load(ROOT / "display.npz")
    events = [json.loads(s) for s in (ROOT / "events.jsonl").read_text().splitlines()]
    boards, queues, cache, rows = [None, None], [None, None], {}, []
    previous, start, game = None, 0.0, None
    for item in read_records(ROOT / "inputs.jsonl.gz"):
        if item["kind"] != "update":
            continue
        result, snap, final, t, idx, formulas, scores, visible = item["args"]
        if game != idx:
            boards, queues, game, start = [None, None], [None, None], idx, t
        sides = (result.p1, result.p2)
        for i, side in enumerate(sides):
            if side.state.name == "STABLE" and side.confirmed_board is not None:
                boards[i], queues[i] = side.confirmed_board, (side.next_pair, side.dnext_pair)
        if not any(a <= t <= b for a, b in WINDOWS):
            continue
        j = min(int(np.searchsorted(display["t_sec"], t)), len(display["t_sec"]) - 1)
        row = dict(t=t, game=idx, source=str(display["source"][j]),
            p1=float(display["display_p1"][j]), state=[s.state.name for s in sides],
            forecast=vars(snap), formulas=formulas, scores=scores,
            chains=[vars(s.chain_event) if s.chain_event else None for s in sides])
        key = json.dumps({k: v for k, v in row.items() if k != "t"}, sort_keys=True)
        if key == previous:
            continue
        previous = key
        fire = []
        for board, queue in zip(boards, queues):
            if board is None:
                fire.append(None)
                continue
            identity = (board._grid.tobytes(), str(queue))
            if identity not in cache:
                value = near_future_fire_power(board, *queue, elapsed_sec=t - start)
                cache[identity] = {k: v.raw for k, v in value.values.items()}
            fire.append(cache[identity])
        row["near_future"] = fire
        row["generated"] = [score_to_ojama(s or 0, elapsed_sec=t-start).ojama_count for s in formulas]
        row.update(confirmed_input(events, t, idx, t-start))
        row["forecast_p2_capped"] = min(snap.forecast_p1, ON_FIELD_CAP) + snap.net_balance_capped
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnosis_e9.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row in rows:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
