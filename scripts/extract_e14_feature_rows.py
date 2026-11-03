"""E13保存記録から発火前行と実得点行を固定fixtureへ抽出する。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from src.board_state_machine import BoardState
from src.exchange_event_record import read_records

RECORD = Path("logs/review_zenchi_g41_43_e13/inputs.jsonl.gz")
OUTPUT = Path("tests/fixtures/e14_prefire_rows.json")
MAX_ROWS = 6
POST_DELAY_SEC = 5.


def extract() -> list[dict]:
    """最新STABLEを側別に保存し、triggerより厳密に前の行だけ選ぶ。"""
    history, rows, seen = [[], []], [], set()
    start, game = None, None
    pending = None
    for number, row in enumerate(read_records(RECORD)):
        if row["kind"] != "update":
            continue
        result, _, _, sec, current, _, scores, _ = row["args"]
        if current != game:
            history, start, game, pending = [[], []], None, current, None
        if pending and sec >= pending["trigger"] + POST_DELAY_SEC:
            pending["after"] = [None if v is None else float(v) for v in scores]
            pending["post_sec"] = sec
            rows.append(pending)
            pending = None
            if len(rows) == MAX_ROWS:
                return rows
        sides = (result.p1, result.p2)
        triggers = [s.chain_event.trigger_sec if s.chain_event else None for s in sides]
        fresh = [t for idx, t in enumerate(triggers) if t is not None and (game, idx, t) not in seen]
        seen.update((game, idx, t) for idx, t in enumerate(triggers) if t is not None)
        if fresh and pending is None and start is not None:
            first = min(fresh)
            selected = [next((v for v in reversed(h) if v["sec"] < first), None) for h in history]
            if all(selected):
                pending = dict(record=str(RECORD), update_row=number, game=game, start=start,
                    trigger=first, pre=selected, firing=[t == first for t in triggers])
        for idx, side in enumerate(sides):
            if side.state == BoardState.STABLE and side.confirmed_board is not None:
                if start is None:
                    start = sec
                history[idx].append(dict(sec=sec, grid=side.confirmed_board._grid.tolist(),
                    queue=[*(side.next_pair or (0, 0)), *(side.dnext_pair or (0, 0))],
                    score=side.score))
    return rows


def main() -> None:
    """JSONだけで再実行できる少数の実盤面を保存する。"""
    rows = extract()
    assert len(rows) == MAX_ROWS
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"E13記録から{len(rows)}区間を抽出", flush=True)


if __name__ == "__main__":
    main()
