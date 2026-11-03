"""誤確定後の実際の手番・連鎖継続を原票から追跡する。"""
from __future__ import annotations

import json
from scripts._diag_set2_falsedeath_extract import OUT, SET
from src.exchange_event_record import read_records

WINDOWS = (('s2', 13, 4417.4, 4435., 1), ('s5', 43, 6172.6, 6187., 1))


def main() -> None:
    """盤面・NEXT・状態・式更新の変化だけを保存する。"""
    result = []
    for part, game, start, end, idx in WINDOWS:
        previous, rows = None, []
        for record in read_records(SET / 'collect/records' / f'{part}.jsonl.gz'):
            if record['kind'] != 'update':
                continue
            value, stamp = record['args'][0], record['args'][3]
            if not start <= stamp <= end:
                continue
            side, attacker = (value.p1, value.p2)[idx], (value.p1, value.p2)[1-idx]
            event = attacker.chain_event
            item = dict(state=side.state.name, queue=[*(side.next_pair or ()), *(side.dnext_pair or ())],
                board=None if side.confirmed_board is None else side.confirmed_board._grid.tolist(), score=side.score,
                attacker_state=attacker.state.name,
                attacker_event=None if event is None else vars(event))
            if item != previous:
                rows.append(dict(t_sec=stamp, **item))
                previous = item
        result.append(dict(game=game, rows=rows))
    (OUT / 'ACTUAL_RESPONSES.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
