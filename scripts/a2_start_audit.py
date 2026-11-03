"""H1′が新たに受理した開始を、保存された実入力へ結び付ける。"""
from __future__ import annotations

import json
from scripts.a2_replay import OUT, INPUT
from scripts.falsedeath_replay_20261001 import records
from src.exchange_event_record import read_records
from src.exchange_history_integrity import MatchHistoryIntegrity


def audit(source: str) -> list[dict]:
    """最終フレームの開始証拠でなく、受理した当該時刻の観測を保存する。"""
    modes = {mode: json.loads((OUT/'replay'/mode/source/'history_integrity_audit.json').read_text())
             for mode in ('prod', 'a2')}
    old = {r['game']: r for r in modes['prod']['games']}
    added = {r['game']: r for r in modes['a2']['games']
             if r['history_complete'] and not old[r['game']]['history_complete']}
    result = []
    observer = MatchHistoryIntegrity(boundary_score_start=True, strict_integrity=True,
                                     menu_boundary_start=True, retain_audit=True)
    for row in read_records(INPUT/'collect/records'/f'{source}.jsonl.gz'):
        if row['kind'] != 'update':
            continue
        sides, _, _, stamp, game, _, displayed, _ = row['args']
        observer.observe((sides.p1, sides.p2), game, stamp)
        start = added.get(game)
        if start is None or stamp != start['match_start_observed_sec']:
            continue
        evidence = observer.evidence()
        assert evidence['history_complete'], (source, game, stamp)
        result.append(dict(record=source, game=game, t_sec=stamp,
            states=[s.state.name for s in (sides.p1, sides.p2)],
            side_scores=[s.score for s in (sides.p1, sides.p2)],
            displayed_scores=displayed, boundary=evidence['boundary_score_evidence']))
    assert len(result) == len(added), source
    return result


def main() -> None:
    """再受理の全事例を省略せず、入力画像なしの原票へ書く。"""
    rows = [r for source in records() for r in audit(source)]
    value = dict(rows=rows, accepted_games=len(rows),
                 both_raw_zero=sum(r['displayed_scores'] is not None and
                                   tuple(r['displayed_scores']) == (0, 0) for r in rows))
    (OUT/'START_AUDIT.json').write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in value.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
