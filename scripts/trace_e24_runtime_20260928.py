"""E23と完全同一の再生に読み取り診断を挿入し、時点別の入力を保存する。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts.diagnose_e23_landing_20260928 import capture
from scripts.inspect_e24_saved_20260928 import OUT, WINDOWS, event_path, save_rows
from scripts.replay_exchange_event_20260926 import replay, compare
from scripts.run_e3_exchange_eval_20260926 import save_json
from src.exchange_event_landing import ExchangeLandingProjection

OPTIONS = dict(death_guard=True, confirmed_death_hold=True,
               death_formula_guard=True, multi_landing_death=True)


def details(projection: Any, overlay: Any, latest: tuple, snapshot: Any) -> dict:
    """当時の台帳とSTABLE時刻を複製し、終了後に上書きされた値と区別する。"""
    tracker = overlay.tracker
    record = tracker.current or projection.death_record
    chains = [dict(**asdict(chain), provisional_score=chain.provisional_score)
              for chain in record.chains]
    return dict(chains=chains, busy=projection.busy, counts=list(projection.counts),
        drops_baseline=projection.drops, snapshot=vars(snapshot).copy(),
        stable=[dict(t_sec=s.t_sec, board=s.board._grid.tolist(), queue=s.queue.tolist())
                for s in latest])


def main() -> None:
    """全入力を再生し、診断によって判定が変わっていないことを全出力で検査する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=list(WINDOWS))
    source = parser.parse_args().source
    begin, end = WINDOWS[source]
    original, rows = ExchangeLandingProjection._evaluate, []

    def evaluate(self: Any, overlay: Any, snapshot: Any, latest: tuple, incoming: list,
                 hands: tuple, base: dict, t_sec: float) -> dict:
        value = original(self, overlay, snapshot, latest, incoming, hands, base, t_sec)
        if begin <= t_sec <= end:
            row = capture(self, overlay, latest, incoming, hands, t_sec, value)
            row.update(details(self, overlay, latest, snapshot))
            row['multi_landing'] = value['multi_landing']
            rows.append(row)
        return value

    ExchangeLandingProjection._evaluate = evaluate
    dest = OUT/'replay'/source
    dest.mkdir(parents=True, exist_ok=True)
    status = replay(Path('logs/e16/records')/f'{source}.jsonl.gz', dest,
                    Path('models/exchange_event_v3'), True, **OPTIONS)
    save_rows(OUT/f'{source}_runtime.json', rows)
    status['equivalence'] = compare(event_path(source).parent, dest)
    status['evaluations'] = len(rows)
    save_json(OUT/f'{source}_equivalence.json', status)
    print(status, flush=True)


if __name__ == '__main__':
    main()
