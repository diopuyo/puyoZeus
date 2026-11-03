"""段2bの資産・採点・用途原票を独立に照合する。"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

import numpy as np
from scripts.gfe_phase_stage2b_train_20261002 import OUT, R1
from scripts.gfe_phase_stage2b_score_20261002 import root, MODES, STAGE2
from scripts.gfe_phase_stage2_replay_20261002 import SOURCES, EXISTING, record_path, EVALSET
from scripts.gfe_phase_stage1_20261002 import save_json, HGB_PARAMS
from src.exchange_event_evaluator import file_sha256

BOOTSTRAPS, SEED = 10000, 0


def assets() -> dict:
    """原票・候補・本番モデル・設定・採点器を事前のハッシュへ固定する。"""
    from scripts.gfe_phase_stage2_preflight_20261002 import evaluation_separation
    paths = [record_path(s) for s in SOURCES]
    paths += [EVALSET/'labels.json', EVALSET/'set2/labels.json', Path('src/production_config.py')]
    paths += [p for d in ('v1', 'v3', 'v1_userphase_v3', 'v1_r1_hgb')
              for p in Path('models/exchange_event_'+d).glob('*') if p.is_file()]
    prior = json.loads((STAGE2/'PREFLIGHT.json').read_text())
    for path, expected in prior['original_model_sha256'].items():
        assert file_sha256(Path(path)) == expected
    assert file_sha256(Path('src/production_config.py')) == prior['production_config_sha256']
    manifest = json.loads((R1/'manifest.json').read_text())
    v3 = json.loads(Path('models/exchange_event_v1_userphase_v3/manifest.json').read_text())
    assert manifest['rows'] == 1035639 and manifest['nice'] == 19
    assert all(manifest['models']['G_fe']['parameters'][k] == v for k, v in HGB_PARAMS.items())
    assert manifest['models']['G_fe']['parameters'] == v3['models']['G_fe']['parameters']
    return dict(sha256={str(p): file_sha256(p) for p in paths},
                separation=evaluation_separation(), original_models_unchanged=True)


def verify_calls(mode: str, source: str) -> dict:
    """計測全呼出しの用途と実モデル、通知ごとの足し戻しを確認する。"""
    directory = root(mode, source)
    with np.load(directory/'gfe_usage.npz') as calls, np.load(directory/'latency.npz') as timing:
        assert set(calls['purpose']) == {'static', 'landing'}
        for purpose, model in (('static', mode), ('landing', 'baseline')):
            mask = calls['purpose'] == purpose
            assert set(calls['model'][mask]) == {model}
            ids = calls['notification'][mask]
            totals = np.bincount(ids, weights=calls['total_ms'][mask])
            np.testing.assert_allclose(totals[np.unique(ids)], calls[purpose+'_notification_ms'], rtol=1e-14)
        ids = calls['notification']
        totals = np.bincount(ids, weights=calls['total_ms'])
        np.testing.assert_allclose(totals[np.unique(ids)], timing['total_ms'], rtol=1e-14)
        np.testing.assert_array_equal(calls['total_ms'], timing['call_ms'])
        np.testing.assert_array_equal(calls['inference_ms'], timing['call_inference_ms'])
        usage = json.loads((directory/'GFE_USAGE.json').read_text())
        assert usage['nice'] == 19 and usage['gfe_static_only']
        for value in usage['models'].values():
            meta = json.loads((Path(value['directory'])/'manifest.json').read_text())
            assert value['sha256'] == meta['models']['G_fe']['sha256']
        return dict(mode=mode, source=source, calls=len(ids), notifications=len(np.unique(ids)))


def independent_metrics(result: dict) -> dict:
    """保存試合値から同seedの区間を独立計算し、全寄与を足し戻す。"""
    prior = json.loads((STAGE2/'METRICS.json').read_text())
    for name in ('set1', 'set2'):
        for key in ('rows', 'flips', 'false_deaths', 'display_frames', 'evaluated_frames', 'late_frames'):
            assert result[name][key]['baseline'] == prior[name][key]['baseline'], (name, key)
    for key in ('cases', 'false', 'scene_first_sec', 'game14'):
        assert result['regression']['baseline'][key] == prior['regression']['baseline'][key], key
    reports = {}
    for mode in MODES[1:]:
        arrays = [np.asarray([r['m_time'] for s in ('set1', 'set2') for r in result[s]['rows'][m]])
                  for m in ('baseline', mode)]
        delta = arrays[1]-arrays[0]
        index = np.random.default_rng(SEED).integers(0, len(delta), (BOOTSTRAPS, len(delta)))
        ci = np.percentile(delta[index].mean(axis=1), [2.5, 97.5])
        saved = result['decisions'][mode]['overall']
        np.testing.assert_allclose(ci, saved['ci95'], atol=1e-12, rtol=0)
        assert abs(delta.mean()-saved['difference']) < 1e-12
        contribution = sum(r['contribution'] for r in result['routes'][mode]['rows'])
        assert abs(contribution-delta.mean()) < 1e-12
        agree = [np.asarray([r['agree'] for s in ('set1', 'set2') for r in result[s]['rows'][m]])
                 for m in ('baseline', mode)]
        aggregates = [a[index, 0].sum(axis=1)/a[index, 1].sum(axis=1) for a in agree]
        agree_ci = np.percentile(aggregates[1]-aggregates[0], [2.5, 97.5])
        np.testing.assert_allclose(agree_ci, result['decisions'][mode]['overall_agree']['ci95'], atol=1e-12)
        reports[mode] = dict(difference=float(delta.mean()), ci95=ci.tolist(),
                            route_sum=contribution, agree_ci95=agree_ci.tolist())
    return reports


def landing_identity() -> list[dict]:
    """候補を有効にしても5記録の仮想着弾生確率が本番LRと同じことを検査する。"""
    rows = []
    for source in EXISTING:
        with np.load(OUT/'baseline/replay'/source/'gfe_usage.npz') as base:
            for mode in MODES[1:]:
                with np.load(root(mode, source)/'gfe_usage.npz') as candidate:
                    for key in ('notification', 't_sec', 'game_idx', 'purpose'):
                        np.testing.assert_array_equal(candidate[key], base[key])
                    mask = base['purpose'] == 'landing'
                    assert candidate['p1'][mask].tobytes() == base['p1'][mask].tobytes()
                    rows.append(dict(source=source, mode=mode, calls=int(mask.sum()), identical=True))
    return rows


def function_lengths() -> dict:
    """追加コードの関数サイズを検査する。既存部分の長さは変更しない。"""
    paths = list(Path('scripts').glob('*stage2b*.py')) + [Path('tests/test_exchange_event_static_gfe.py')]
    violations = {}
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno-node.lineno+1 > 50:
                violations[f'{path}:{node.name}'] = node.end_lineno-node.lineno+1
    assert not violations, violations
    return dict(files=len(paths), violations=violations)


def verify_assets() -> None:
    """再生入力と、M0を含む本番既存資産を終了時にも全ハッシュ照合する。"""
    expected = json.loads((OUT/'PREFLIGHT.json').read_text())
    for path, value in expected['sha256'].items():
        assert file_sha256(Path(path)) == value
    prior = json.loads((STAGE2/'PREFLIGHT.json').read_text())
    for path, value in prior['original_model_sha256'].items():
        assert file_sha256(Path(path)) == value
    assert file_sha256(Path('src/production_config.py')) == prior['production_config_sha256']


def main() -> None:
    """実行前ハッシュと完了後の監査を分け、以前の原票は更新しない。"""
    os.nice(max(0, 19-os.nice(0)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--code-check', action='store_true')
    args = parser.parse_args()
    if args.code_check:
        print(json.dumps(function_lengths()), flush=True)
        return
    if args.preflight:
        save_json(OUT/'PREFLIGHT.json', assets())
        print('事前検証完了', flush=True)
        return
    verify_assets()
    calls = [verify_calls(m, s) for m in MODES[1:] for s in SOURCES]
    result = json.loads((OUT/'METRICS.json').read_text())
    samples = [json.loads(line) for line in (OUT/'resources.jsonl').read_text().splitlines()]
    minimum = min(v['available_bytes'] for v in samples)/1024**3
    assert minimum >= 6
    assert all(p['nice'] == 19 for v in samples for p in v['processes'])
    manifest = json.loads((R1/'manifest.json').read_text())
    v3 = json.loads(Path('models/exchange_event_v1_userphase_v3/manifest.json').read_text())
    assert manifest['models']['G_fe']['parameters'] == v3['models']['G_fe']['parameters']
    verification = dict(passed=True, calls=calls, independent=independent_metrics(result),
        landing_identity=landing_identity(),
        original_assets_unchanged=True, functions=function_lengths(),
        resources=dict(samples=len(samples), minimum_available_gib=minimum, nice=19, blas_threads=1))
    save_json(OUT/'VERIFICATION.json', verification)
    from scripts.gfe_phase_stage2b_report_20261002 import write_summary
    write_summary(result)
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
