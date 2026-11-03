"""誤発火と登録済みA/Bの確定盤面・火力・余裕を照合する。"""
from __future__ import annotations

import json
from pathlib import Path

from src.chain import ChainSimulator
from src.board import DEATH_COL
from src.exchange_event_record import read_records
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.scoring import calculate_chain_score


def inspect(record: Path, targets: list[float]) -> None:
    """入力記録の該当時刻で最新STABLEと連鎖解消後を比較する。"""
    latest, done = [None, None], set()
    sim = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    for row in read_records(record):
        if row["kind"] != "update":
            continue
        result, snapshot, _, t, *rest = row["args"]
        for i, side in enumerate((result.p1, result.p2)):
            if side.state.name == "STABLE" and side.confirmed_board is not None:
                latest[i] = (t, side.confirmed_board, side.next_pair, side.dnext_pair)
        for target in targets:
            if t < target-.001 or target in done:
                continue
            done.add(target)
            boards = []
            for timestamp, board, first, second in latest:
                resolved = sim.simulate(board)
                boards.append(dict(t=timestamp, height=board.height_of(DEATH_COL),
                    dead=board.is_dead(), grid=board._grid.tolist(), queue=[first, second],
                    resolved_height=resolved.final_board.height_of(DEATH_COL),
                    resolved_dead=resolved.final_board.is_dead(),
                    score=calculate_chain_score(resolved).total_score))
            print(json.dumps(dict(t=t, states=[s.state.name for s in (result.p1, result.p2)],
                                  boards=boards, snapshot=vars(snapshot))))


def main() -> None:
    """既知の6件とA/Bの対照点だけ出力する。"""
    report = json.loads(Path("logs/e10b/report.json").read_text())
    targets = [r["first_sec"] for r in report["deaths"] if r["false_positive"]]
    inspect(Path("logs/e8/renders/mia8KCjr52g/on/inputs.jsonl.gz"), targets)
    inspect(Path("logs/review_zenchi_part3/on_e9/inputs.jsonl.gz"),
            [2693.1833, 2694.9166, 2696.0166, 2696.1833, 2807.6166, 2807.7833,
             2808.15, 2809.6166, 2810.6166])


if __name__ == "__main__":
    main()
