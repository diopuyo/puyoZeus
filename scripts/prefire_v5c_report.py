"""5Cの照合・速度・再生監査を、未完了を除外せず集計する。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import pstats

from scripts import prefire_v5b_report as report
from scripts.run_prefire_replay_20260930 import BASELINE_DIRS

OUT = Path('logs/prefire_prediction/v5c')
SAMPLES = 100


def verification() -> dict:
    """固定100局面の原票をすべて要求する。"""
    rows = [json.loads((OUT/'final_verification'/f'case_{i:03d}.json').read_text()) for i in range(SAMPLES)]
    return dict(samples=len(rows), transitions=sum(r['transitions'] for r in rows),
        mismatches=sum(r['mismatches'] for r in rows),
        missed=sum(any(v['missed'] for v in r['values']) for r in rows),
        table_choice_mismatches=sum(not v['table_sequential_equal'] for r in rows for v in r['values']),
        max_value_error=max(r['max_value_error'] for r in rows),
        feature_boards=sum(r['feature_boards'] for r in rows),
        feature_mismatches=sum(r['feature_mismatches'] for r in rows),
        board_dependent_response_cases=sum(bool(r['dependencies']) for r in rows),
        scope='5A/5Bと同じ両側1手の独立全直積。3手全直積の100局面検証ではない。')


def manifest() -> dict:
    """変更コード・入力・本番フラグの指紋と関数規約を記録する。"""
    paths = [*Path('src').glob('prefire*v5*.py'), *Path('scripts').glob('prefire_v5c*.py'),
             *Path('tests').glob('test_prefire_v5c*.py'), Path('src/exchange_hidden_row_probability.py'),
             Path('tests/test_e32_hidden_row_belief.py')]
    violations, hashes = [], {}
    for path in paths:
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw)):
            if not isinstance(node, ast.FunctionDef):
                continue
            missing = [a.arg for a in node.args.args if a.arg not in ('self', 'cls') and a.annotation is None]
            lines = node.end_lineno-node.lineno+1
            if lines > 50 or missing or node.returns is None:
                violations.append(dict(file=str(path), function=node.name, lines=lines, missing=missing))
    for path in (Path('src/production_config.py'), Path('logs/prefire_prediction/v5_experiment/samples_ledger.json')):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sha256=hashes, violations=violations)


def profile_summary() -> dict:
    """累積時間を重複加算せず、関数別の呼出数と時間を列挙する。"""
    stats = pstats.Stats(str(OUT/'profile/before.prof'))
    names = ('candidates', 'unique', 'choose', 'resolve', 'evaluate', '_exchange_board_features',
             '_side_feats_full_base', 'evaluate_exchange_event', 'prove_post_counter')
    rows = [dict(file=Path(file).name, function=name, calls=stat[1], own_sec=stat[2], cumulative_sec=stat[3])
            for (file, _, name), stat in stats.stats.items() if name in names]
    return dict(total_profiled_sec=stats.total_tt, functions=rows,
                note='累積時間は包含関係があるため合計しない。プロファイラ負荷を含む。')


def build_report() -> dict:
    """同じ5記録とセット1の6区間がすべて完了してから採点する。"""
    report.OUT = OUT
    sources = {source: report.score(source) for source in (*BASELINE_DIRS, *(f'c{i}' for i in range(1, 7)))}
    checked = verification()
    latency = json.loads((OUT/'latency.json').read_text())
    audits = [sources[source]['audit'] for source in BASELINE_DIRS]
    gates = dict(a=checked['mismatches'] == 0, b=checked['missed'] == 0,
                 c=sum(a['early_referenced'] for a in audits) == 0)
    agreements = [sources[source]['agreement'] for source in BASELINE_DIRS]
    totals = {k: sum(a[k] for a in audits) for k in
              ('rows', 'used', 'early_referenced', 'early_used', 'held', 'missing_any', 'firing_any')}
    summary = dict(gates=gates, abc_passed=all(gates.values()), speed_passed=latency['within_200ms'],
                   passed=all(gates.values()) and latency['within_200ms'], **totals,
                   agreement={k: sum(a[k] for a in agreements) for k in agreements[0]})
    return dict(summary=summary, verification=checked, latency=latency, sources=sources,
                zenchi_set1=report.combined_set1(sources), profile=profile_summary(),
                enumeration=json.loads((OUT/'enumeration.json').read_text()),
                resources=json.loads((OUT/'replay_resources.json').read_text()))


def main() -> None:
    """採点前でも規約検査だけは実行できる。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-only', action='store_true')
    args = parser.parse_args()
    result = manifest()
    (OUT/'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('violations', result['violations'], flush=True)
    if not args.manifest_only:
        result = build_report()
        (OUT/'replay_report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
