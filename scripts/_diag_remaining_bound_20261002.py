"""証明関数の入力感度だけを別オブジェクトで測定する（本番挙動は変更しない）。"""
from __future__ import annotations

import json
from typing import Any

from scripts._diag_remaining_20261002 import BASE, OUT, read, save
from src.board import Board
from src.post_counter_death_bound import prove_post_counter

CASES = (('c5', 1728.1), ('mia8KCjr52g', 772.7666666666667))


def main() -> None:
    """操作中ツモ欠測・猶予手数の仮説を、死亡証明の感度として分離記録する。"""
    output: list[dict[str, Any]] = []
    for source, stamp in CASES:
        records = [json.loads(s) for s in (OUT / f'trace_{source}_all/trace.jsonl').read_text().splitlines()]
        trace = next(r for r in records if r['kind'] == 'death_inputs' and abs(r['t_sec']-stamp) < .001)
        audit = next(r for r in read(BASE / f'replay/all/{source}/post_counter_bound_audit.json')['rows']
                     if abs(r['t_sec']-stamp) < .001)
        board = Board.from_list(trace['context']['replies'][1])
        elapsed = next(iter(trace['elapsed'].values()))
        for name, queue, hands in (('original', tuple(audit['queue']), 1),
                                  ('unknown_current', (0, 0, *audit['queue']), 1),
                                  ('one_more_hand', tuple(audit['queue']), 2)):
            result = prove_post_counter(board, queue, audit['incoming'], hands, elapsed, tuple(audit['palette']))
            output.append(dict(source=source, t_sec=stamp, variant=name, queue=queue, hands=hands, result=result))
    save('bound_sensitivity.json', output)
    print(json.dumps([{k: v for k, v in r.items() if k != 'result'} | dict(
        result={k: r['result'][k] for k in ('dead', 'reason', 'nodes')}) for r in output]))


if __name__ == '__main__':
    main()
