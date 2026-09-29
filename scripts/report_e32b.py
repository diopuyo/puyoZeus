"""E32bの監査原票を同一時刻で突合し、根拠を集約する。"""
from __future__ import annotations

from collections import Counter
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
from scripts.audit_e32b import OUT, RECORDS, SOURCES, TOP_INTERVALS, load_rows, prior, save
from src.exchange_event_record import read_records

SCORE_RANGES = {'zenchi':(3089.,3112.,0), 'q_7gc4TgFig':(878.,884.,0),
                'fcXG83vInDY':(267.,278.,1)}


def events(source: str, version: str = 'e32') -> list[dict]:
    """確定原票を読み、値行と最終チェーン情報を区別する。"""
    return [json.loads(s) for s in (prior(version,source)/'events.jsonl').read_text().splitlines()]


def visibility(source: str) -> list[dict]:
    """採用・絞り込みの得点を、その時点までに見えた式へ照合する。"""
    wanted = {}
    for row in load_rows(source):
        for obs in row['observations']:
            wanted[(row['game'], row['side'], obs['t_sec'])] = (row, obs)
    seen, result = {}, []
    for item in read_records(RECORDS/f'{source}.jsonl.gz'):
        if item['kind'] != 'update':
            continue
        args = item['args']
        stamp, game = args[3:5]
        for idx, side in enumerate((args[0].p1,args[0].p2)):
            label, event = f'{idx+1}P', side.chain_event
            if event is not None and args[7][idx]:
                key = (game,label,event.trigger_sec,event.chain_count,event.total_score)
                seen.setdefault(key, stamp)
            if (game,label,stamp) not in wanted:
                continue
            row, obs = wanted[(game,label,stamp)]
            key = (game,label,event.trigger_sec,obs['count'],obs['score'])
            result.append(dict(game=game,side=label,chain_id=row['chain_id'],t=stamp,
                count=obs['count'],score=obs['score'],visible_now=bool(args[7][idx]),
                first_visible_sec=seen.get(key), event_trigger=event.trigger_sec,
                snapshot_end=row['snapshot'].get('end_sec'),
                snapshot_is_past=row['snapshot'].get('end_sec', stamp) < row['trigger_sec'] <= stamp))
    return result


def attribute(interval: dict, records: list[dict], visible: list[dict]) -> dict:
    """区間の出力を作った交換内の採用と、その時点の観測だけを取り出す。"""
    start, end, game = interval['start'], interval['end'], interval['internal_game']
    records = [r for r in records if r['game_idx'] == game]
    values = sorted([(v['t_sec'],r,v) for r in records for v in r['values'] if v['t_sec'] <= end], key=lambda x:x[0])
    earlier = [v for v in values if v[0] <= start]
    selected = ([earlier[-1]] if earlier else []) + [v for v in values if start < v[0] <= end]
    chain_ids = {c['chain_id'] for _,r,_ in selected for c in r['chains'] if c['trigger_sec'] <= end}
    active_ids = {p['chain_id'] for _,_,v in selected for p in v.get('prefire_prediction',[])}
    adoptions = []
    old = {r['chain_id']:r for r in load_rows('zenchi','e31') if r['game'] == game}
    for row in load_rows('zenchi'):
        if row['game'] != game or row['chain_id'] not in chain_ids or not row['accepted']:
            continue
        observations = [v for v in visible if v['game'] == game and v['chain_id'] == row['chain_id'] and v['t'] <= end]
        adoptions.append(dict(chain_id=row['chain_id'],side=row['side'],trigger=row['trigger_sec'],
            active_in_interval=row['chain_id'] in active_ids,
            snapshot_start=row['snapshot']['start_sec'],snapshot_end=row['snapshot']['end_sec'],
            initial_candidates=row['observations'][0]['remaining'],initial_score=row['predicted_score'],
            e31_reason=old.get(row['chain_id'],{}).get('reason'),
            observations_at_start=[v for v in observations if v['t'] <= start][-1:],
            observations_during=[v for v in observations if start < v['t'] <= end]))
    fields = ('source','t_sec','p1','base_p1','gfe_p1','incoming','death_incoming','hidden_row_weighted',
              'completion_sides','predicted_final_scores','prefire_prediction','dead_sides')
    return dict(interval,adoptions=adoptions,first_event={k:v for k,v in selected[0][2].items() if k in fields},
                event_count=len(selected))


def compare_prefix(cutoff: float) -> dict:
    """全列のdtypeと値ビット列、および各更新時のイベント全内容を照合する。"""
    full = OUT/'replays/zenchi'
    short = OUT/'replays'/f'zenchi_{cutoff:.3f}'
    left, right = np.load(full/'display.npz'), np.load(short/'display.npz')
    n = len(right['t_sec'])
    mask = left['t_sec'] <= cutoff
    assert int(mask.sum()) == n
    failures = []
    matched = np.ones(n,dtype=bool)
    assert left.files == right.files
    for name in left.files:
        a,b = left[name],right[name]
        if a.ndim and a.shape[0] == len(mask):
            a = a[mask]
            for index in range(n):
                if a[index].tobytes() != b[index].tobytes():
                    matched[index] = False
                    failures.append(dict(column=name,row=index,t=float(right['t_sec'][index])))
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            if not failures:
                failures.append(dict(column=name,metadata=True))
    with gzip.open(full/'trace.jsonl.gz','rt') as a, gzip.open(short/'trace.jsonl.gz','rt') as b:
        history = [json.loads(x) for x in a]
        prefix = [json.loads(y) for y in b]
    expected = [r for r in history if r['t'] <= cutoff]
    assert len(expected) == len(prefix) == json.loads((short/'DONE.json').read_text())['frames']
    pairs = list(zip(expected,prefix))
    bad = [dict(index=i,full=a,short=b) for i,(a,b) in enumerate(pairs) if a != b]
    prefix_events = full/f'checkpoint_{cutoff:.3f}.jsonl'
    exact = prefix_events.read_bytes() == (short/'events.jsonl').read_bytes()
    return dict(cutoff=cutoff,display_rows=n,display_matched=n if not failures else int(matched.sum()),
        display_columns=left.files,event_states=len(pairs),event_states_matched=len(pairs)-len(bad),
        event_records=len(prefix_events.read_text().splitlines()),event_file_exact=exact,
        first_display_mismatch=sorted(failures,key=lambda v:v.get('row',-1))[:1],first_event_mismatch=bad[:1],
        input_sha256=hashlib.sha256((OUT/'inputs'/f'zenchi_{cutoff:.3f}.jsonl.gz').read_bytes()).hexdigest())


def error_details(row: dict) -> dict:
    """初回・最終候補と実得点の確定根拠を別々に保存する。"""
    path = OUT/'replays'/row['source']
    key = (row['game'],row['chain_id'])
    states = [r for r in json.loads((path/'candidates.json').read_text()) if (r['game'],r['chain_id']) == key]
    chains = [c for r in events(row['source']) if r['game_idx'] == key[0] for c in r['chains'] if c['chain_id'] == key[1]]
    summary = []
    for state in states:
        dist: Counter = Counter()
        for value in state['distribution']:
            dist[value['prefix'][-1] if value['prefix'] else 0] += value['weight']
        summary.append(dict(state,score_distribution=dict(dist)))
    return dict(row,states=summary,final_chain=chains[0])


def reconstruct(row: dict) -> dict:
    """保存済みの隠し段分布から、各段で生き残った得点質量を再計算する。"""
    from src.board import Board
    from src.chain import ChainSimulator
    from src.hidden_row_belief import combinations
    from src.probabilistic_board import ProbabilisticCell
    from src.exchange_prefire_candidates import completion
    from src.production_config import GHOST_CHAIN_RULE_ENABLED
    simulator = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    cells = [ProbabilisticCell({int(k):v for k,v in c.items()}) for c in row['hidden_distributions']]
    choices, mass = combinations(cells)
    options = []
    for hidden,weight in choices:
        board = Board.from_list(row['snapshot']['board'])
        board._grid[0] = hidden
        options.append(dict(completion(board,simulator),weight=weight))
    states = []
    previous = 0
    for obs in row['observations']:
        count = obs['count']
        options = [v for v in options if previous <= count <= previous+1 and len(v['prefix']) >= count and v['prefix'][count-1] == obs['score']]
        total = sum(v['weight'] for v in options)
        dist: Counter = Counter()
        for value in options:
            dist[value['score']] += value['weight']/total
        states.append(dict(t=obs['t_sec'],count=count,n=len(options),observed=obs['score'],
            mean=sum(k*v for k,v in dist.items()),distribution=dict(dist)))
        assert len(options) == obs['remaining']
        previous = count
    np.testing.assert_allclose(states[0]['mean'],row['predicted_score'],rtol=0,atol=1e-8)
    return dict(source=row['source'],game=row['game'],chain_id=row['chain_id'],side=row['side'],
                error=row['error'],final_score=row['final_score'],states=states)


def score_inputs(source: str) -> list[dict]:
    """異常候補の式・OCR確定通知・表示スコアを入力順のまま保存する。"""
    begin,end,idx = SCORE_RANGES[source]
    rows, previous = [], None
    for item in read_records(RECORDS/f'{source}.jsonl.gz'):
        if item['kind'] != 'update':
            continue
        args = item['args']
        if not begin <= args[3] <= end:
            continue
        side = (args[0].p1,args[0].p2)[idx]
        event = side.chain_event
        value = dict(formula_total=args[5][idx],score=args[6][idx],visible=args[7][idx],
            state=side.state.name,finalization=vars(args[2]),
            event=None if event is None else {k:getattr(event,k) for k in ('trigger_sec','chain_count','total_score','mechanism')})
        if value != previous:
            rows.append(dict(t=args[3],**value))
            previous = value
    return rows


def report(inputs_only: bool = False) -> None:
    """検証の母数を保持し、欠損成果物は例外として止める。"""
    prelim = json.loads((OUT/'preliminary.json').read_text())
    seen = visibility('zenchi')
    save(OUT/'visibility.json',seen)
    intervals = [attribute(r,events('zenchi'),seen) for r in prelim['differences']['intervals'][:TOP_INTERVALS]]
    save(OUT/'gain_intervals.json',intervals)
    if inputs_only:
        save(OUT/'reconstructed_errors.json',[reconstruct(r) for r in prelim['top_errors']])
        save(OUT/'score_inputs.json',{s:score_inputs(s) for s in SCORE_RANGES})
        print(json.dumps(dict(observations=len(seen),unseen=[r for r in seen if r['first_visible_sec'] is None]),ensure_ascii=False))
        return
    checks = [compare_prefix(t) for t in json.loads((OUT/'cutoffs.json').read_text())]
    save(OUT/'prefix_comparison.json',checks)
    errors = [error_details(r) for r in prelim['top_errors']]
    save(OUT/'score_error_details.json',errors)
    result = dict(counts=prelim['differences']['counts'],prefix=checks,
        observations=len(seen),future_or_unseen=[r for r in seen if r['first_visible_sec'] is None or r['first_visible_sec'] > r['t']],
        snapshot_not_past=[r for r in seen if not r['snapshot_is_past']],
        intervals=len(intervals),full_equivalence={s:json.loads((OUT/'replays'/s/'DONE.json').read_text())['saved_e32_equivalence'] for s in ('zenchi',*SOURCES[:2])})
    save(OUT/'SUMMARY.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-only', action='store_true')
    parser.add_argument('--prefix-only', type=float)
    args = parser.parse_args()
    if args.prefix_only is not None:
        print(json.dumps(compare_prefix(args.prefix_only),ensure_ascii=False,indent=2))
    else:
        report(args.inputs_only)
