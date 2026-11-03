"""E32のOFF一致、因果時刻、入力・本番設定不変を照合する。"""
from __future__ import annotations
import ast
import argparse
import inspect
import json
from pathlib import Path
import numpy as np
from scripts.run_e32 import OUT, e31
from scripts.run_e3_exchange_eval_20260926 import digest, save_json
from scripts.replay_exchange_event_20260926 import compare, replay
from src.exchange_event_overlay import ExchangeEventOverlay

NEW_FILES = ('src/hidden_row_belief.py', 'src/exchange_hidden_row_belief.py',
    'src/exchange_hidden_row_probability.py', 'scripts/run_e32.py', 'scripts/verify_e32.py',
    'scripts/report_e32.py', 'tests/test_e32_hidden_row_belief.py', 'src/exchange_event_overlay.py')


def code_checks() -> dict:
    """追加モジュールの型注釈と関数50行制限を検査する。"""
    lengths = {}
    for name in NEW_FILES:
        for node in ast.walk(ast.parse(Path(name).read_text(encoding='utf-8'))):
            if isinstance(node, ast.FunctionDef):
                assert node.returns is not None, (name,node.name)
                assert all(a.annotation for a in node.args.args if a.arg not in ('self','cls')), (name,node.name)
                lengths[f'{name}:{node.name}'] = node.end_lineno-node.lineno+1
    assert max(lengths.values()) <= 50, {k:v for k,v in lengths.items() if v>50}
    for call in (ExchangeEventOverlay, replay):
        assert inspect.signature(call).parameters['hidden_row_belief'].default is False
    return dict(max_function_lines=max(lengths.values()), default_off=True)


def main() -> None:
    """比較対象の全行とOFF全列・全バイトを検査し、監査入力の由来を保存する。"""
    e31.prior.OUT = OUT
    unchanged = json.loads(Path('logs/e25/INPUTS.json').read_text())
    for name, sha in unchanged.items():
        assert digest(Path(name)) == sha, name
    controls, inputs, rows, withdrawals = {}, {}, {}, []
    for source in e31.prior.ALL_SOURCES:
        suffix = source if source in ('review','zenchi') else f'renders/{source}/on'
        baseline = Path('logs/e27/on')/suffix
        controls[source] = compare(baseline, e31.prior.directory('off',source))
        directory = e31.prior.directory('on',source)
        with np.load(baseline/'display.npz') as old, np.load(directory/'display.npz') as new:
            for key in ('t_sec','game_idx','state1','state2'):
                np.testing.assert_array_equal(old[key],new[key])
            rows[source] = len(old['t_sec'])
            audit = json.loads((directory/'snapshot_final_audit.json').read_text())
            for row in audit['calibration']:
                assert row['prior_sec'] < row['observed_sec'], row
            for row in audit['rows']:
                if row['withdrawn']:
                    mask = new['t_sec'] == row['withdraw_sec']
                    withdrawals.append(dict(source=source, game=row['game'], chain=row['chain_id'],
                        stamp=row['withdraw_sec'], e27=old['display_p1'][mask].tolist(), e32=new['display_p1'][mask].tolist()))
        path = Path('logs/e31/records')/f'{source}.jsonl.gz'
        inputs[str(path)] = digest(path)
    save_json(OUT/'VERIFICATION.json', dict(**code_checks(), production_unchanged=True,
        same_rows=rows, off_controls=controls, inputs=inputs, immutable_assets=unchanged, withdrawals=withdrawals))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-only', action='store_true')
    if parser.parse_args().code_only:
        print(code_checks())
    else:
        main()
