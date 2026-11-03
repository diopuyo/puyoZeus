"""全消失例を判定系列と一つ抜き再生へ結合する。合否や採否は扱わない。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts._diag_dv2d_loss_inventory import OUT, BASE, read, save, resource, events
from scripts._diag_dv2d_loss_inputs import ledger_failures

EPS = 1e-6
MODES = ('no_d1', 'no_e35', 'no_d3', 'no_d4', 'no_h1')
FLAGS = dict(no_d1='D1', no_e35='E35', no_d3='D3', no_d4='D4', no_h1='H1')
INTEGRITY_REASONS = {'color_score_mismatch', 'board_unsettled', 'invalid_start',
                    'missing_stable_board', 'missing_board_score', 'unknown_board', 'completion_prefix_unverified'}


def flat(path: Path) -> list[dict]:
    """当該時刻の値だけを読む。イベント終端の未来の連鎖属性は混ぜない。"""
    values = []
    for line in path.open(encoding='utf-8'):
        event = json.loads(line)
        values.extend(dict(game=event['game_idx'], **v) for v in event['values'])
    return sorted(values, key=lambda v:v['t_sec'])


def unknown_causes(post: dict) -> list[str]:
    """未知手持ちを省略できなかった条件を、実際の同時刻観測から分解する。"""
    current = post.get('current_puyo_observation') or {}
    if post.get('unknown_current_pair_prepended') is not True:
        return []
    conditions = dict(non_chain=current.get('state') != 'CHAIN',
        firing_unplaced=not current.get('firing_placed', False),
        next_changed=not current.get('next_unchanged', False),
        next_motion=bool(current.get('next_motion')), ended=current.get('ended', True),
        end_signal=current.get('end_signal_sec') is not None)
    return [name for name, failed in conditions.items() if failed]


def reasons(value: dict, side: str) -> dict:
    """拒否に直接使われた条件と入力を区別して短く保存する。"""
    idx = int(side == '2P')
    single = (value.get('single_death_proof') or [None, None])[idx] or {}
    post = next((v for v in value.get('post_counter_bound', []) if v['side'] == side), {})
    multi = (value.get('multi_landing') or [None, None])[idx] or {}
    history = value.get('history_integrity', {})
    return dict(single=single.get('reason'), post=post.get('reason'),
        post_proofs=sorted(set(v['reason'] for v in post.get('proofs', []))), multi=multi.get('reason'),
        unknown=post.get('unknown_current_pair_prepended'), current=post.get('current_puyo_observation'),
        unknown_causes=unknown_causes(post),
        h1=side in history.get('blocked_dead_sides', []), complete=history.get('history_complete'),
        safety=(value.get('state_safety') or [None, None])[idx],
        incoming=(value.get('death_incoming') or [None, None])[idx],
        hands=(value.get('hands') or [None, None])[idx],
        certain=(value.get('completion_certain') or [None, None])[idx],
        verified=(value.get('verified_attack') or [None, None])[idx])


def series(rows: list[dict], side: str) -> list[dict]:
    """同じ判定条件が連続する区間を圧縮する。入力変化の各原票は別に保持する。"""
    result = []
    for row in rows:
        if 'post_counter_bound' not in row:
            continue
        why = reasons(row, side)
        signature = {k:v for k,v in why.items() if k not in ('current', 'incoming', 'hands')}
        if result and result[-1]['signature'] == signature:
            result[-1]['last_sec'], result[-1]['evaluations'] = row['t_sec'], result[-1]['evaluations']+1
        else:
            result.append(dict(first_sec=row['t_sec'], last_sec=row['t_sec'], evaluations=1,
                               signature=signature, first_inputs=why))
    return result


def recovered(target: dict, rows: list[dict], history: bool = False) -> dict:
    """同一試合・同一死亡側で、元の確定時刻から観測死亡までに戻る秒数を測る。"""
    start, end, side = target['first_sec'], target['candidate_observed_death_sec'], target['side']
    candidates = []
    for row in rows:
        if row['game'] != target['local_game'] or row['t_sec'] >= end-EPS:
            continue
        sides = (row.get('history_integrity', {}).get('blocked_dead_sides', []) if history
                 else row.get('dead_sides', []) if row['source'] == 'unavoidable_death' else [])
        if side in sides:
            candidates.append(row['t_sec'])
    stamp = min(candidates) if candidates else None
    return dict(first_sec=stamp, restored_sec=0. if stamp is None else end-max(start, stamp))


def compact(value: Any) -> Any:
    """巨大な探索枝を件数へ縮め、盤面・入力・条件は残す。"""
    if isinstance(value, list):
        return [compact(v) for v in value]
    if isinstance(value, dict):
        return {k:len(v) if k in ('rounds', 'bounds') and isinstance(v, list) else compact(v)
                for k,v in value.items()}
    return value


def trace_inputs(target: dict) -> list[dict]:
    """入力スナップショットを行番号付きで抽出する。"""
    path = OUT/f"trace_{target['record']}_r2/trace.jsonl"
    result = []
    for number, line in enumerate(path.open(encoding='utf-8'), 1):
        row = json.loads(line)
        if (target['first_sec']-.05 <= row['t_sec'] < target['candidate_observed_death_sec']
                and 'context' in row):
            result.append(dict(line=number, **compact(row),
                               unverified_pending_details=ledger_failures(row, target['side'])))
    return result


def case(target: dict, current: list[dict], ablations: dict) -> dict:
    """実際の拒否理由と、解除による回復を別列にする。"""
    stamp, end = target['first_sec'], target['candidate_observed_death_sec']
    same = [v for v in current if v['game'] == target['local_game'] and 'post_counter_bound' in v]
    before = [v for v in same if v['t_sec'] <= stamp+EPS]
    first = before[-1] if before else None
    during = [v for v in same if stamp-EPS <= v['t_sec'] < end-EPS]
    recovery = {FLAGS[mode]:recovered(target, rows) for mode,rows in ablations.items()}
    for result in recovery.values():
        if result['first_sec'] is not None:
            at = [v for v in same if v['t_sec'] <= result['first_sec']+EPS]
            result['v2d_evaluation_sec'] = at[-1]['t_sec'] if at else None
            result['v2d_reasons'] = reasons(at[-1], target['side']) if at else None
    blocked = recovered(target, current, history=True)
    if 'H1' not in recovery:
        recovery['H1'] = dict(**blocked, method='抑止前の同時刻証明')
    signals = [reasons(v, target['side']) for v in during]
    flags = sorted({name for name,r in recovery.items() if r['first_sec'] is not None})
    observed = dict(D1=any(v['single'] in INTEGRITY_REASONS for v in signals),
        E35=any(v['unknown'] is True for v in signals), H1=any(v['h1'] for v in signals),
        D3=any(v['post'] == 'unverified_attack_or_budget' for v in signals))
    return {k:v for k,v in target.items() if k not in ('series', 'original', 'at_first')} | dict(
        first_evaluation_sec=None if first is None else first['t_sec'],
        first_reasons={} if first is None else reasons(first, target['side']),
        original=compact(target['original']), first=compact(first), observed=observed,
        recovery=recovery, h1_same_time=blocked, independently_restored_by=flags, reason_series=series(during, target['side']),
        inputs=trace_inputs(target))


def history_summary() -> dict:
    """全方式を元のH1とのORにし、判定時刻も保存する。"""
    observations = read(OUT/'OBSERVATIONS_v2.json')
    rows = []
    for source, value in observations.items():
        games = {g['game']:g for g in value['games']}
        for old in value['history']:
            g = games[old['game']]
            evidence = dict(boundary=g['boundary_sec'], zero3=g['zero_sec'],
                stable_zero=g['zero_stable_sec'], empty_zero=g['boundary_empty_sec'],
                death_and_zero3=g['zero_sec'] if g['preceding_death'] else None)
            rows.append(dict(record=source, game=old['game'], old_complete=old['history_complete'],
                             old_start=old['match_start_observed_sec'], alternatives=evidence))
    names = ('boundary', 'zero3', 'stable_zero', 'empty_zero', 'death_and_zero3')
    summary = {name:dict(reduced=sum(not r['old_complete'] and r['alternatives'][name] is not None for r in rows),
        remaining=sum(not r['old_complete'] and r['alternatives'][name] is None for r in rows)) for name in names}
    assert len(rows) == 176 and sum(not r['old_complete'] for r in rows) == 65
    return dict(total=176, baseline_incomplete=65, summary=summary, rows=rows)


def aggregate(rows: list[dict]) -> dict:
    """相互排他的分類と、重複する一つ抜き効果を分離する。"""
    groups = defaultdict(list)
    for row in rows:
        key = '+'.join(row['independently_restored_by']) or '単独解除で未回復'
        groups[key].append(row)
    partition = {key:dict(count=len(rr), total_delay_sec=sum(r['delay_sec'] for r in rr)) for key,rr in groups.items()}
    modes = list(FLAGS.values())
    effects = {mode:dict(count=sum(r['recovery'].get(mode, {}).get('first_sec') is not None for r in rows),
        evaluated_cases=sum(mode in r['recovery'] for r in rows),
        restored_sec=sum(r['recovery'].get(mode, {}).get('restored_sec', 0.) for r in rows)) for mode in modes}
    return dict(partition=partition, effects=effects, total_delay_sec=sum(r['delay_sec'] for r in rows),
                union_restored_cases=sum(bool(r['independently_restored_by']) for r in rows),
                union_restored_sec=sum(max(v['restored_sec'] for v in r['recovery'].values()) for r in rows))


def joint_results(rows: list[dict]) -> dict:
    """実施した重複解除だけを結合し、未実施をゼロ件に置き換えない。"""
    modes = ('no_d1+no_h1', 'no_e35+no_h1', 'no_d3+no_h1', 'no_d4+no_h1',
             'no_d1+no_d3', 'no_d1+no_d3+no_h1')
    effects = {}
    for mode in modes:
        selected = []
        for source in dict.fromkeys(r['record'] for r in rows):
            path = OUT/f'{mode}_{source}_r2'
            if not (path/'status.json').exists():
                assert not path.exists(), f'重複解除が未完了: {path}'
                continue
            values = flat(path/'events.jsonl')
            for row in rows:
                if row['record'] == source:
                    result = recovered(row, values)
                    row.setdefault('joint_recovery', {})[mode] = result
                    selected.append(row)
        if selected:
            extra = [r for r in selected if not r['independently_restored_by']
                     and r['joint_recovery'][mode]['first_sec'] is not None]
            effects[mode] = dict(evaluated_cases=len(selected), additional_count=len(extra),
                additional_sec=sum(r['joint_recovery'][mode]['restored_sec'] for r in extra),
                cases=[dict(group=r['group'], game=r['game'], **r['joint_recovery'][mode]) for r in extra])
    return effects


def risk_summary() -> dict:
    """短縮再生で実際に見えた勝者側への確定だけを数え、全体の安全性に外挿しない。"""
    from scripts._diag_set2_falsedeath_audit import scan
    from scripts.falsedeath_score_20261001 import SETS
    result = {}
    for mode in MODES:
        result[FLAGS[mode]] = {}
        for name, (sources, labels) in SETS.items():
            value = scan([OUT/f'{mode}_{source}_r2' for source in sources], read(labels)['games'])
            result[FLAGS[mode]][name] = dict(false_events=value['false'], false_frames=value['false_frames'],
                rows=[r for r in value['rows'] if r['false']], wrong_frames=value['wrong_frames'])
    return result


def verify_trace(source: str) -> dict:
    """観測用フックが表示結果を一切変えていないことを同一入力接頭部で確かめる。"""
    directory = OUT/f'trace_{source}_r2'
    with np.load(directory/'display.npz') as new, np.load(BASE/f'replay/v2d/{source}/display.npz') as old:
        count = len(new['t_sec'])
        for key in new.files:
            expected = old[key][:count] if old[key].ndim else old[key]
            np.testing.assert_array_equal(new[key], expected, err_msg=f'{source}/{key}')
    return dict(record=source, display_frames=count, all_display_fields_identical=True)


def main() -> None:
    """完成した再生だけを集計する。欠測モードをゼロ件扱いしない。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    print('RESOURCE', resource(), flush=True)
    targets = read(OUT/'EVENTS.json')
    results, checks = [], []
    for source in dict.fromkeys(r['record'] for r in targets):
        current = events(source)
        ablations = {}
        for mode in MODES:
            path = OUT/f'{mode}_{source}_r2'
            if (path/'status.json').exists():
                ablations[mode] = flat(path/'events.jsonl')
            elif not args.partial and source not in ('q_7gc4TgFig', 'fcXG83vInDY'):
                raise AssertionError(f'未完了: {path}')
        if not (OUT/f'trace_{source}_r2/status.json').exists():
            print('TRACE未完了', source, flush=True)
            continue
        checks.append(verify_trace(source))
        selected = [case(r, current, ablations) for r in targets if r['record'] == source]
        results.extend(selected)
        for r in selected:
            print(r['group'], r['game'], r['first_reasons'], r['recovery'], flush=True)
    summary = aggregate([r for r in results if r['group'] in ('set1', 'set2')])
    history = history_summary()
    print('AGGREGATE', json.dumps(summary, ensure_ascii=False), flush=True)
    print('HISTORY', json.dumps(history['summary'], ensure_ascii=False), flush=True)
    if not args.partial:
        save('ANALYSIS.json', dict(cases=results, aggregate=summary, history=history,
                                  trace_checks=checks, risks=risk_summary(), joints=joint_results(results)))
    else:
        save('PARTIAL_ANALYSIS.json', dict(cases=results, aggregate=summary, history=history,
                                         trace_checks=checks))


if __name__ == '__main__':
    main()
