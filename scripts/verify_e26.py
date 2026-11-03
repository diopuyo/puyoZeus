"""E26の既定OFF・本番設定不変・ソース記録と検収結果を保存する。"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from scripts.run_e3_exchange_eval_20260926 import digest, save_json
from scripts.replay_exchange_event_20260926 import replay
from src.exchange_event_overlay import ExchangeEventOverlay
from src.exchange_event_landing import ExchangeLandingProjection

OUT = Path('logs/e26')
NEW_FILES = ('src/exchange_midchain_completion.py', 'src/midchain_board_reader.py',
    'scripts/run_e26_ablation.py', 'scripts/run_e26_completion.py', 'scripts/enrich_e26_midchain.py',
    'scripts/inspect_e26_inputs.py', 'scripts/verify_e26_records.py', 'scripts/verify_e26.py',
    'scripts/render_e26_review.py', 'scripts/progress_e26.py', 'tests/test_e26_ablation.py',
    'tests/test_e26_midchain.py', 'scripts/run_e26_controls.py', 'scripts/report_e26.py',
    'scripts/measure_e26_scene.py', 'scripts/diagnose_e26_settle.py')
FLAGS = ('pending_ledger', 'color_score_safety', 'completion_recovery', 'midchain_completion')
CHANGED_FILES = ('src/exchange_event_overlay.py', 'src/exchange_event_landing.py',
    'src/exchange_completion_recovery.py', 'src/exchange_landing_safety.py',
    'src/exchange_event_record.py', 'src/recognition_pipeline.py',
    'scripts/replay_exchange_event_20260926.py', 'scripts/visualize_advantage_overlay.py',
    'scripts/review_data_panel.py')


def checks() -> dict:
    """変更禁止ファイルと既定値を、実装の読み取りで確認する。"""
    inputs = json.loads(Path('logs/e25/INPUTS.json').read_text())
    expected = inputs['src/production_config.py']
    actual = digest(Path('src/production_config.py'))
    assert actual == expected
    for name, expected_hash in inputs.items():
        assert digest(Path(name)) == expected_hash, name
    for call in (replay, ExchangeEventOverlay):
        assert all(inspect.signature(call).parameters[f].default is False for f in FLAGS)
    assert ExchangeLandingProjection().safety is None
    maximum = 0
    for name in NEW_FILES:
        tree = ast.parse(Path(name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno-node.lineno+1
                assert length <= 50, (name, node.name, length)
                maximum = max(maximum, length)
    hashes = {name: digest(Path(name)) for name in (*NEW_FILES, *CHANGED_FILES)}
    return dict(production_config_sha256=actual, production_unchanged=True,
        original_inputs_and_models_unchanged=True, defaults_off=True,
        maximum_new_function_lines=maximum, sha256=hashes)


def main() -> None:
    """検収前の検査でも実行でき、数値完了後は同じ証跡へ合否を添付する。"""
    value = checks()
    if (OUT/'on/METRICS.json').exists():
        metrics = json.loads((OUT/'on/METRICS.json').read_text())
        value['candidate'] = metrics['candidate']
        value['gates'] = metrics['gates']
    save_json(OUT/'VERIFICATION.json', value)
    print({k: v for k, v in value.items() if k != 'sha256'}, flush=True)


if __name__ == '__main__':
    main()
