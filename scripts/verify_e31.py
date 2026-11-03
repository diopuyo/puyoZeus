"""E31の既定OFF・入力不変・同じ行での比較を固定証跡へ残す。"""
from __future__ import annotations
import ast
import argparse
import inspect
import json
import gzip
from itertools import zip_longest
from pathlib import Path
import numpy as np
from src.exchange_event_overlay import ExchangeEventOverlay
from scripts.replay_exchange_event_20260926 import replay, compare
from scripts.run_e3_exchange_eval_20260926 import digest, save_json
from scripts.run_e31 import OUT, prior


def check_functions() -> dict:
    """新規モジュールの全関数を50行上限で検査する。"""
    paths = ('src/prefire_snapshot_reader.py', 'src/exchange_prefire_snapshot.py',
        'scripts/enrich_e31_snapshots.py','scripts/diagnose_e31_scene.py','scripts/run_e31.py',
        'scripts/verify_e31.py','scripts/report_e31.py','src/exchange_event_overlay.py')
    lengths = {}
    for name in paths:
        tree = ast.parse(Path(name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                lengths[f'{name}:{node.name}'] = node.end_lineno-node.lineno+1
                assert node.returns is not None, (name,node.name)
    assert max(lengths.values()) <= 50, {k:v for k,v in lengths.items() if v>50}
    return lengths


def rejected_only_rows(source: str, baseline: Path) -> dict:
    """初回採用前の全列と、過去採用の表示EMAを含まない未採用確率を照合する。"""
    audit = json.loads((prior.directory('on',source)/'snapshot_final_audit.json').read_text())['rows']
    first: dict[int,float] = {}
    for row in audit:
        if row['accepted']:
            first[row['game']] = min(first.get(row['game'],float('inf')),row['trigger_sec'])
    if not first:
        compare(baseline, prior.directory('on',source))
    with np.load(baseline/'display.npz') as base, np.load(prior.directory('on',source)/'display.npz') as on:
        mask = np.array([t < first.get(int(g),float('inf')) for t,g in zip(base['t_sec'],base['game_idx'])])
        initial = base['t_sec'] < min(first.values(),default=float('inf'))
        for name in base.files:
            if base[name].ndim and len(base[name]) == len(mask):
                selected = initial if name == 'display_adv' else mask
                np.testing.assert_array_equal(base[name][selected],on[name][selected],err_msg=f'{source}:{name}')
        differences = np.abs(base['display_adv'][mask]-on['display_adv'][mask])
        return dict(all_columns_before_first_accept=int(initial.sum()), current_columns_before_game_accept=int(mask.sum()),
            historical_display_ema_differences=int(np.count_nonzero(differences)),
            historical_display_ema_max=float(differences.max()) if differences.size else 0.)


def verify_added_inputs() -> dict:
    """追加観測を除く原記録の全キー・全値・盤面dtypeを厳密に照合する。"""
    manifests = {}
    for source in prior.ALL_SOURCES:
        old = Path('logs/e26/records')/f'{source}.jsonl.gz'
        new = OUT/'records'/old.name
        metadata = json.loads(new.with_suffix('.json').read_text())
        audit = json.loads((prior.directory('on',source)/'snapshot_final_audit.json').read_text())
        assert metadata['window_count'] == len(audit['rows']), source
        assert digest(Path(metadata['windows_path'])) == metadata['windows_sha256'], source
        count = 0
        with gzip.open(old,'rt') as left, gzip.open(new,'rt') as right:
            for a,b in zip_longest(left,right):
                assert a is not None and b is not None, source
                before, after = json.loads(a), json.loads(b)
                if after['kind'] == 'update':
                    sides = after['args']['tuple'][0]['namespace']
                    for label in ('p1','p2'):
                        sides[label]['namespace'].pop('prefire_snapshot',None)
                assert json.dumps(before,sort_keys=True) == json.dumps(after,sort_keys=True), (source,count)
                count += 1
        manifests[source] = dict(original_sha256=digest(old), enriched_sha256=digest(new), rows=count)
    return manifests


def withdrawal_checks() -> list[dict]:
    """実測で撤回した行についてE27の採用確率への復帰を確認する。"""
    rows = []
    for source in prior.ALL_SOURCES:
        path = prior.directory('on',source)/'snapshot_final_audit.json'
        if not path.exists():
            continue
        suffix = source if source in ('review','zenchi') else f'renders/{source}/on'
        with np.load(Path('logs/e27/on')/suffix/'display.npz') as base, np.load(path.parent/'display.npz') as on:
            for row in json.loads(path.read_text())['rows']:
                if not row['withdrawn']:
                    continue
                mask = on['t_sec'] == row['withdraw_sec']
                rows.append(dict(source=source,game=row['game'],chain_id=row['chain_id'],
                    t_sec=row['withdraw_sec'], reason=row['withdraw_reason'],
                    baseline_p1=base['display_p1'][mask].tolist(), actual_p1=on['display_p1'][mask].tolist()))
    return rows


def main() -> None:
    """本番設定、既存記録、モデルを改変せず、評価行とOFF全列を照合する。"""
    for call in (ExchangeEventOverlay,replay):
        assert inspect.signature(call).parameters['prefire_snapshot'].default is False
    expected = json.loads(Path('logs/e25/INPUTS.json').read_text())
    for name, sha in expected.items():
        assert digest(Path(name)) == sha, name
    assert 'src/production_config.py' in expected
    prior.OUT = OUT
    matching = {}
    for source in prior.ALL_SOURCES:
        suffix = source if source in ('review','zenchi') else f'renders/{source}/on'
        with np.load(Path('logs/e27/on')/suffix/'display.npz') as base, np.load(prior.directory('on',source)/'display.npz') as on:
            for name in ('t_sec','game_idx','state1','state2'):
                np.testing.assert_array_equal(base[name],on[name])
            matching[source] = dict(frames=len(base['t_sec']), rejected_only_identical=
                rejected_only_rows(source,Path('logs/e27/on')/suffix))
    controls = {p.stem:json.loads(p.read_text()) for p in OUT.glob('OFF_*.json')}
    assert len(controls) == len(prior.ALL_SOURCES)
    records = {str(p):digest(p) for p in Path('logs/e26/records').glob('*.jsonl.gz') if '.linear.' not in p.name}
    for name, sha in records.items():
        assert sha == json.loads(Path(name.replace('.jsonl.gz','.json')).read_text())['sha256']
    save_json(OUT/'VERIFICATION.json',dict(default_off=True,production_unchanged=True,
        models_inputs_unchanged=True,same_rows=matching,off_controls=controls,
        max_function_lines=max(check_functions().values()),records=records,added_inputs=verify_added_inputs(),
        withdrawals=withdrawal_checks()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-only', action='store_true')
    parser.add_argument('--withdrawals-only', action='store_true')
    args = parser.parse_args()
    prior.OUT = OUT
    if args.code_only:
        print(max(check_functions().values()))
    elif args.withdrawals_only:
        print(json.dumps(withdrawal_checks()))
    else:
        main()
