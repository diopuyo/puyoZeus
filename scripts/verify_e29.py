"""E29の既定OFF・本番設定と固定入力の不変・関数長を検査する。"""
from __future__ import annotations
import ast
import inspect
import json
from pathlib import Path
from scripts.run_e3_exchange_eval_20260926 import digest, save_json
from scripts.replay_exchange_event_20260926 import replay
from src.exchange_event_overlay import ExchangeEventOverlay
from src.exchange_midchain_completion import MidchainCompletion
from src.exchange_hidden_row_death import HiddenRowDeathCompletion

OUT = Path('logs/e29')
NEW_FILES = ('scripts/run_e29.py', 'scripts/audit_e29_predictions.py', 'scripts/verify_e29.py',
             'scripts/render_e29_review.py', 'scripts/report_e29.py', 'scripts/measure_e29_scene.py',
             'tests/test_e29_prediction_audit.py',
             'tests/test_e29_single_observation.py')


def main() -> None:
    """検収後にも同じ検査を行い、入力・本番設定への変更を検出する。"""
    expected = json.loads(Path('logs/e25/INPUTS.json').read_text())
    for name, sha in expected.items():
        assert digest(Path(name)) == sha, name
    for call in (replay, ExchangeEventOverlay):
        assert inspect.signature(call).parameters['midchain_single_observation'].default is False
    for call in (MidchainCompletion, HiddenRowDeathCompletion):
        assert inspect.signature(call).parameters['single_observation'].default is False
    lengths = {}
    for name in (*NEW_FILES, 'src/exchange_midchain_completion.py',
                 'src/exchange_hidden_row_death.py', 'src/exchange_event_overlay.py'):
        tree = ast.parse(Path(name).read_text())
        compile(tree, name, 'exec')
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                lengths[f'{name}:{node.name}'] = node.end_lineno-node.lineno+1
    assert max(lengths.values()) <= 50, {k:v for k,v in lengths.items() if v > 50}
    records = {str(p): digest(p) for p in Path('logs/e26/records').glob('*.jsonl.gz') if '.linear.' not in p.name}
    for name, sha in records.items():
        assert sha == json.loads(Path(name.replace('.jsonl.gz', '.json')).read_text())['sha256']
    save_json(OUT/'VERIFICATION.json', dict(defaults_off=True, production_unchanged=True,
        models_and_inputs_unchanged=True, max_function_lines=max(lengths.values()), records=records,
        preregistration_sha256=digest(OUT/'PREREGISTRATION.md'),
        off_controls={p.stem: json.loads(p.read_text()) for p in OUT.glob('OFF_*.json')}))
    print('既定OFF・本番設定/モデル/入力不変・関数50行以内: OK', flush=True)


if __name__ == '__main__':
    main()
