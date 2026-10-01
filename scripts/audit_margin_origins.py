"""保存入力の試合ごとに3時計の観測差と欠測を記録する（再認識なし）。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.board_state_machine import BoardState
from src.exchange_event_record import read_records, FIRST_PLACEMENT_ARGUMENT_INDEX
from src.margin_clock import MarginClock
from src.ojama_accounting import OjamaAccountingTracker, logger

ROOT = Path("logs/margin_clock")
RECORDS = Path("logs/pending_expiry/full/records")
QUANTILES = (0., .25, .5, .75, .95, 1.)
DIFFERENCE_COLUMNS = ("overlay_delta", "accounting_delta",
                      "overlay_applied_delta", "accounting_applied_delta")


def new_row(source: str, game: int, stamp: float) -> dict:
    """欠測をゼロ差に混ぜず、分母を試合単位に固定する。"""
    return dict(source=source, game=game, first_frame=stamp, frames=0,
                overlay_off=None, accounting_off=None, first_placement=None,
                overlay_delta=None, accounting_delta=None)


def audit(path: Path) -> list[dict]:
    """時計を観測だけで再構成する。保存snapshot・学習データは変更しない。"""
    rows: dict[int, dict] = {}
    clock, accounting = MarginClock(), OjamaAccountingTracker()
    previous = (None, None)
    for item in read_records(path):
        if item["kind"] != "update":
            continue
        inputs = item["args"]
        result, _, _, stamp, game, *_ = inputs
        sides = (result.p1, result.p2)
        row = rows.setdefault(game, new_row(path.name.split(".")[0], game, stamp))
        row["frames"] += 1
        times = inputs[FIRST_PLACEMENT_ARGUMENT_INDEX] if len(inputs) > FIRST_PLACEMENT_ARGUMENT_INDEX else None
        clock.observe(sides, stamp, game, times)
        for idx, side in enumerate(sides):
            accounting.on_state_transition("p1" if idx == 0 else "p2", previous[idx],
                                           side.state, side.score, stamp)
            if row["overlay_off"] is None and side.state == BoardState.STABLE and side.confirmed_board is not None:
                row["overlay_off"] = stamp
        previous = tuple(side.state for side in sides)
        if row["accounting_off"] is None:
            row["accounting_off"] = accounting._match_start_sec
        if clock.origin is not None and row["first_placement"] is None:
            row["first_placement"] = clock.origin
            row["accounting_at_placement"] = accounting._match_start_sec
    for row in rows.values():
        for prefix in ("overlay", "accounting"):
            if row["first_placement"] is not None and row[f"{prefix}_off"] is not None:
                row[f"{prefix}_delta"] = row["first_placement"] - row[f"{prefix}_off"]
            row[f"{prefix}_on"] = (row["first_placement"] if row["first_placement"] is not None
                                      else row[f"{prefix}_off"])
            old, new = row[f"{prefix}_off"], row[f"{prefix}_on"]
            row[f"{prefix}_applied_delta"] = None if old is None or new is None else new-old
    return list(rows.values())


def distribution(rows: list[dict], name: str) -> dict:
    """秒の符号はON起点−OFF起点。欠測数を併記する。"""
    values = [r[name] for r in rows if r[name] is not None]
    return dict(games=len(rows), measured=len(values), missing=len(rows)-len(values),
                quantiles=dict(zip(map(str, QUANTILES), np.quantile(values, QUANTILES).tolist())) if values else {},
                mean=float(np.mean(values)) if values else None)


def main() -> None:
    """原票をCSV、分布をJSONへ保存する。"""
    logger.disabled = True
    rows = [row for path in sorted(RECORDS.glob("*.jsonl.gz")) for row in audit(path)]
    summary = {name: distribution(rows, name) for name in DIFFERENCE_COLUMNS}
    summary["sources"] = {source: {name: distribution([r for r in rows if r["source"] == source], name)
        for name in DIFFERENCE_COLUMNS} for source in sorted({r["source"] for r in rows})}
    (ROOT / "origins.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=2), encoding="utf-8")
    with (ROOT / "origins.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
