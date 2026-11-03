"""E27の既定OFF・入力不変・関数長と、S3/landingへの混入を確認する。"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from scripts.run_e3_exchange_eval_20260926 import digest, save_json
from scripts.replay_exchange_event_20260926 import replay
from src.exchange_event_overlay import ExchangeEventOverlay
from src.exchange_event_landing import ExchangeLandingProjection

OUT = Path('logs/e27')
NEW_FILES = ('src/exchange_hidden_row_death.py', 'src/exchange_death_inputs.py',
    'scripts/run_e27.py', 'scripts/verify_e27.py', 'tests/test_e27_hidden_death.py',
    'tests/test_e27_death_inputs.py', 'src/match_color_evidence.py', 'tests/test_e27_match_colors.py',
    'scripts/render_e27_review.py', 'scripts/measure_e27_scene.py', 'scripts/diagnose_e27_hidden.py',
    'scripts/report_e27.py')
FLAGS = ('death_pending_ledger', 'hidden_row_death')
CHANGED_FILES = ('src/exchange_event_landing.py', 'src/exchange_event_multilanding.py',
    'src/exchange_event_overlay.py', 'scripts/replay_exchange_event_20260926.py',
    'scripts/visualize_advantage_overlay.py')


def checks() -> dict:
    """既存モデルと入力を固定し、E26補完記録のハッシュも保存する。"""
    expected = json.loads(Path('logs/e25/INPUTS.json').read_text())
    for name, sha in expected.items():
        assert digest(Path(name)) == sha, name
    for call in (replay, ExchangeEventOverlay, ExchangeLandingProjection):
        assert all(inspect.signature(call).parameters[f].default is False for f in FLAGS)
    lengths = {}
    for name in NEW_FILES:
        for node in ast.walk(ast.parse(Path(name).read_text())):
            if isinstance(node, ast.FunctionDef):
                lengths[f'{name}:{node.name}'] = node.end_lineno-node.lineno+1
    assert max(lengths.values()) <= 50, lengths
    large = {}
    for name in ('src/exchange_event_landing.py', 'src/exchange_event_multilanding.py', 'src/exchange_event_overlay.py'):
        for node in ast.walk(ast.parse(Path(name).read_text())):
            if isinstance(node, ast.FunctionDef) and node.end_lineno-node.lineno+1 > 50:
                large[f'{name}:{node.name}'] = node.end_lineno-node.lineno+1
    assert not large, large
    records = {str(p): digest(p) for p in Path('logs/e26/records').glob('*.jsonl.gz') if '.linear.' not in p.name}
    for name, sha in records.items():
        assert sha == json.loads(Path(name.replace('.jsonl.gz', '.json')).read_text())['sha256']
    return dict(production_unchanged=True, models_and_original_inputs_unchanged=True,
        defaults_off=True, maximum_new_function_lines=max(lengths.values()),
        input_records=records, existing_large_functions=large,
        source_hashes={name: digest(Path(name)) for name in (*NEW_FILES, *CHANGED_FILES)})


def probability_isolation(source: str) -> dict:
    """完走予測と同時刻確率を照合し、死亡保持に伴う会計メタデータ差も隠さず残す。"""
    suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
    old = [json.loads(line) for line in (Path('logs/e26/on')/suffix/'events.jsonl').read_text().splitlines()]
    new = [json.loads(line) for line in (OUT/'on'/suffix/'events.jsonl').read_text().splitlines()]
    assert len(old) == len(new), source
    checked, chains, metadata, s3_checked, timing = 0, 0, [], 0, []
    for before, after in zip(old, new):
        assert len(before['chains']) == len(after['chains'])
        for a, b in zip(before['chains'], after['chains']):
            changed = {k: [a[k], b[k]] for k in a if a[k] != b[k]}
            assert not any(k.startswith('predicted_') for k in changed), (source, changed)
            if changed:
                metadata.append(dict(exchange=before['exchange_id'], side=a['side'], chain=a['chain_id'], changes=changed))
        chains += len(before['chains'])
        old_s3 = {(v['t_sec'], v['source']): v['p1'] for v in before['values']
                  if v['source'] in ('S3', 'S3_provisional') and 'gfe_p1' not in v}
        new_s3 = {(v['t_sec'], v['source']): v['p1'] for v in after['values']
                  if v['source'] in ('S3', 'S3_provisional') and 'gfe_p1' not in v}
        common = old_s3.keys() & new_s3.keys()
        assert all(old_s3[k] == new_s3[k] for k in common), (source, before['exchange_id'], 'S3')
        s3_checked += len(common)
        for kind, values, other in (('added', new_s3, old_s3), ('removed', old_s3, new_s3)):
            timing.extend(dict(exchange=before['exchange_id'], change=kind, t_sec=k[0], source=k[1], p1=values[k])
                          for k in sorted(values.keys()-other.keys()))
        previous = {}
        for row in before['values']:
            if 'gfe_p1' in row:
                previous.setdefault(row['t_sec'], []).append(row)
        for value in after['values']:
            if 'gfe_p1' not in value:
                continue
            refs = previous.get(value['t_sec'])
            if refs is not None:
                assert any(all(value[k] == ref[k] for k in ('incoming', 'hands', 'base_p1', 'gfe_p1'))
                           for ref in refs), (source, value['t_sec'])
                checked += 1
    return dict(chains=chains, shared_landing_probability_rows=checked, s3_rows=s3_checked, predictions_identical=True,
                shared_probabilities_identical=True, bookkeeping_changes=metadata, s3_timing_changes=timing)


def main() -> None:
    """全入力の再生完了後は確率入力不変の証跡も追加する。"""
    value = checks()
    from scripts.run_e17_ablation_20260928 import ALL_SOURCES
    value['probability_isolation'] = {}
    for source in ALL_SOURCES:
        suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
        if (OUT/'on'/suffix/'DONE.json').exists():
            value['probability_isolation'][source] = probability_isolation(source)
    save_json(OUT/'VERIFICATION.json', value)
    print({k: v for k, v in value.items() if k not in ('source_hashes', 'input_records')}, flush=True)


if __name__ == '__main__':
    main()
