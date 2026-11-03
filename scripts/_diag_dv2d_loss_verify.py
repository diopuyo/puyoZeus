"""診断の計数、再現性、資源条件と変更範囲を検算する。"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from scripts._diag_dv2d_loss_inventory import OUT, ROOT, BASE, read, save, resource
from scripts._diag_dv2d_loss_analyze import MODES, recovered
from scripts.falsedeath_replay_20261001 import input_record

MAX_FUNCTION_LINES = 50
EXPECTED_BASE = '42d82b1'
WINDOWS_GIT = '/mnt/c/Program Files/Git/mingw64/bin/git.exe'
WINDOWS_ROOT = 'D:/puyo_analyzer/wt_evalset'


def digest(path: Path) -> str:
    """大きいファイルも一定メモリでハッシュ化する。"""
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def syntax() -> dict:
    """構文・型ヒント・50行以内を、pycacheを書かず点検する。"""
    result = {}
    for path in sorted((ROOT/'scripts').glob('_diag_dv2d_loss_*.py')):
        source = path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(path))
        compile(tree, str(path), 'exec')
        functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node in functions:
            assert node.end_lineno-node.lineno+1 <= MAX_FUNCTION_LINES, (path.name,node.name,node.lineno,node.end_lineno)
            assert node.returns is not None, (path,node.name,'return annotation')
            assert all(arg.annotation is not None for arg in node.args.args), (path,node.name,'argument annotation')
        result[path.name] = dict(functions=len(functions), sha256=digest(path))
    return result


def arithmetic(data: dict) -> dict:
    """初回時刻・未来除外・複数旗の重複排除を独立した小例でも確かめる。"""
    target = dict(first_sec=10., candidate_observed_death_sec=15., side='2P', local_game=7)
    rows = [dict(game=7,t_sec=9.,source='unavoidable_death',dead_sides=['2P']),
            dict(game=8,t_sec=11.,source='unavoidable_death',dead_sides=['2P'])]
    assert recovered(target, rows) == dict(first_sec=9., restored_sec=5.)
    assert recovered(target, [dict(game=7,t_sec=15.,source='unavoidable_death',dead_sides=['2P'])])['first_sec'] is None
    cases = [r for r in data['cases'] if r['group'] in ('set1','set2')]
    assert Counter(r['group'] for r in cases) == dict(set1=30,set2=25)
    assert len({(r['group'],r['game'],r['side']) for r in cases}) == 55
    assert abs(sum(r['delay_sec'] for r in cases)-371.4) < 1e-6
    assert sum(r['count'] for r in data['aggregate']['partition'].values()) == 55
    assert all(r['evaluated_cases'] == 55 for r in data['aggregate']['effects'].values())
    return dict(case_count=55, delay_sec=371.4, arithmetic_examples=True)


def replay_receipts(data: dict) -> list[dict]:
    """未完了の原票を成果に含めず、計算プロセスの実測資源を確認する。"""
    sources = list(dict.fromkeys(r['record'] for r in data['cases'] if r['group'] in ('set1','set2')))
    directories = [OUT/f'{mode}_{source}_r2' for mode in MODES for source in sources]
    directories += [OUT/f"trace_{r['record']}_r2" for r in data['trace_checks']]
    directories += sorted(OUT.glob('no_*+no_*_r2'))
    rows = []
    for directory in directories:
        status, measured = read(directory/'status.json'), read(directory/'RESOURCE.json')
        assert status['state'] == 'completed', directory
        assert measured['nice'] == 10 and measured['available_gib'] >= 6, directory
        rows.append(dict(directory=directory.name, resource=measured, status=status,
                         command=read(directory/'COMMAND.json')))
    return rows


def input_checks(data: dict) -> dict:
    """保存値からのD3条件分解が、現行クラスの計算と一致することを確認する。"""
    from src.exchange_event_tracker import ExchangeChainRecord
    from src.exchange_death_integrity_v2 import independently_verified
    from scripts._diag_dv2d_loss_inputs import independent_failure
    count = 0
    for case in data['cases']:
        for sample in case['inputs']:
            for chain in sample.get('chains', []):
                restored = SimpleNamespace(**chain)
                value = ExchangeChainRecord.provisional_score.fget(restored)
                restored.provisional_score = value
                detail = independent_failure(chain)
                assert value == detail['provisional_score']
                assert independently_verified(restored) == (not detail['reasons'])
                count += 1
    return dict(independent_score_samples=count, implementation_matches=True)


def manifest(data: dict) -> dict:
    """再利用した既存原票とコードのSHA256を残す。"""
    sources = list(dict.fromkeys(r['record'] for r in data['cases']))
    paths = [input_record(mode, source) for mode in ('prod','v2d') for source in sources]
    paths += [BASE/f'replay/v2d/{source}/display.npz' for source in sources]
    paths += [BASE/'v2d'/name for name in ('REPORT.md','RESULT_v2d.json','HISTORY.json')]
    paths += [ROOT/'src'/name for name in ('production_config.py','exchange_history_integrity.py',
        'exchange_single_death_proof.py','exchange_post_counter_bound.py','exchange_death_integrity_v2.py',
        'exchange_event_tracker.py','exchange_landing_safety.py','board_state_machine.py','placement_inferrer.py')]
    return {str(path):dict(bytes=path.stat().st_size, sha256=digest(path)) for path in paths}


def recovery_summary(data: dict) -> dict:
    """単独解除で残った10例と、実施済み複数解除による回復を重複排除する。"""
    cases = [r for r in data['cases'] if r['group'] in ('set1','set2')]
    remaining = [r for r in cases if not r['independently_restored_by']]
    restored = [r for r in remaining if any(v['first_sec'] is not None
                for v in r.get('joint_recovery', {}).values())]
    best = lambda row: max(v['restored_sec'] for v in
                          [*row['recovery'].values(), *row.get('joint_recovery', {}).values()])
    assert len(remaining) == 10 and len(restored) == 10
    return dict(single_remaining=10, joint_additional_cases=10,
        joint_additional_sec=sum(best(r) for r in restored),
        best_observed_configurations_cases=55,
        best_observed_configurations_sec=sum(best(r) for r in cases),
        history_incomplete_cases=sum(not r['first_reasons']['complete'] for r in cases))


def main() -> None:
    """最終成果が揃った時だけ確認票を保存する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--syntax-only', action='store_true')
    args = parser.parse_args()
    measured, source_checks = resource(), syntax()
    if args.syntax_only:
        save('SYNTAX.json', dict(resource=measured, files=source_checks))
        print('構文・型ヒント・関数長の確認完了', flush=True)
        return
    data = read(OUT/'ANALYSIS.json')
    checks, receipts = arithmetic(data), replay_receipts(data)
    inputs = input_checks(data)
    assert all(r['all_display_fields_identical'] for r in data['trace_checks'])
    assert len(data['trace_checks']) == 16
    git = [WINDOWS_GIT, '-C', WINDOWS_ROOT]
    head = subprocess.check_output([*git,'rev-parse','--short','HEAD'], cwd=ROOT, text=True).strip()
    assert head == EXPECTED_BASE, head
    changed = subprocess.check_output([*git,'diff','--name-only'], cwd=ROOT, text=True).splitlines()
    assert not changed, changed
    save('MANIFEST.json', manifest(data))
    save('RECOVERY_SUMMARY.json', recovery_summary(data))
    save('VALIDATION.json', dict(base_commit=head, resource=measured, files=source_checks,
        arithmetic=checks, replay_count=len(receipts), receipts=receipts,
        trace_checks=data['trace_checks'], input_checks=inputs,
        tracked_changes_before_commit=changed))
    print('確認完了', len(receipts), '再生', flush=True)


if __name__ == '__main__':
    main()
