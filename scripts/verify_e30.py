"""E30の既定OFF・既存入力と設定の不変・変更関数の長さを検証する。"""
from __future__ import annotations
import ast
import inspect
import json
from pathlib import Path
from scripts.run_e3_exchange_eval_20260926 import digest, save_json
from scripts.replay_exchange_event_20260926 import replay
from src.exchange_event_overlay import ExchangeEventOverlay

OUT = Path('logs/e30')


def main() -> None:
    """固定証跡と対照再生を検収後にも確認する。"""
    expected = json.loads(Path('logs/e25/INPUTS.json').read_text())
    for name, sha in expected.items():
        assert digest(Path(name)) == sha, name
    for call in (replay, ExchangeEventOverlay):
        assert inspect.signature(call).parameters['prefire_candidates'].default is False
    files = ['src/exchange_prefire_candidates.py', 'src/exchange_event_overlay.py',
        'src/exchange_event_landing.py', 'src/exchange_death_inputs.py',
        'scripts/audit_e30_predictions.py', 'scripts/run_e30.py', 'scripts/verify_e30.py',
        'scripts/report_e30.py', 'scripts/render_e30_review.py', 'scripts/review_data_panel.py',
        'tests/test_e30_prefire_candidates.py', 'tests/test_e30_integration.py']
    lengths = {}
    for name in files:
        tree = ast.parse(Path(name).read_text())
        compile(tree, name, 'exec')
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                lengths[f'{name}:{node.name}'] = node.end_lineno-node.lineno+1
    assert max(lengths.values()) <= 50, {k:v for k,v in lengths.items() if v > 50}
    records = {str(p): digest(p) for p in Path('logs/e26/records').glob('*.jsonl.gz') if '.linear.' not in p.name}
    for name, sha in records.items():
        assert sha == json.loads(Path(name.replace('.jsonl.gz', '.json')).read_text())['sha256']
    assert 'src/production_config.py' in expected
    save_json(OUT/'VERIFICATION.json', dict(defaults_off=True, production_unchanged=True,
        models_and_inputs_unchanged=True, max_function_lines=max(lengths.values()), records=records,
        sources={f:digest(Path(f)) for f in files}, preregistration_sha256=digest(OUT/'PREREGISTRATION.md'),
        off_controls={p.stem: json.loads(p.read_text()) for p in OUT.glob('OFF_*.json')}))
    print('既定OFF・本番設定/モデル/入力不変・関数50行以内: OK', flush=True)


if __name__ == '__main__':
    main()
