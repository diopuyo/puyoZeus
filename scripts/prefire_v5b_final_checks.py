"""5Bの最終読み・規約・原票を、旧試行を保持してまとめる。"""
from __future__ import annotations

import ast
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts import prefire_v5b_refine as refined

OUT = Path('logs/prefire_prediction/v5b/final')


def manifest() -> dict:
    """実装・入力・採用フラグのSHAと、関数の規約違反を出力する。"""
    files = sorted({p for pattern in ('src/prefire*v5*.py', 'scripts/prefire_v5b*.py',
                                     'tests/test_prefire*v5*.py') for p in Path('.').glob(pattern)})
    violations, hashes = [], {}
    for path in files:
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw)):
            if not isinstance(node, ast.FunctionDef):
                continue
            size = node.end_lineno-node.lineno+1
            missing = [a.arg for a in node.args.args if a.arg not in ('self', 'cls') and a.annotation is None]
            if size > 50 or missing or node.returns is None:
                violations.append(dict(file=str(path), function=node.name, lines=size, missing=missing))
    for path in (Path('src/production_config.py'), Path('logs/prefire_prediction/v5_experiment/samples_ledger.json'),
                 *Path('native/puyo_core/src').glob('*.rs')):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sha256=hashes, violations=violations)


def summary() -> dict:
    """100局面の遷移・枝落ちを、取りこぼしなく集計する。"""
    root = OUT.parent
    before = [json.loads(p.read_text()) for p in sorted((root/'before').glob('case_*.json'))]
    after = [json.loads(p.read_text()) for p in sorted((root/'verification').glob('case_*.json'))]
    def totals(rows: list[dict]) -> dict:
        return dict(samples=len(rows), transitions=sum(r['transitions'] for r in rows),
                    mismatches=sum(r['mismatches'] for r in rows),
                    missed=sum(any(v['missed'] for v in r['values']) for r in rows))
    failed = [r for r in before if any(v['missed'] for v in r['values'])]
    branches = [b for row in failed for b in row['branches']]
    missing, observations = [Counter(), Counter()], [Counter(), Counter()]
    denominator = 0
    for path in (root/'before').glob('missing_*.json'):
        record = json.loads(path.read_text())
        denominator += record['denominator']
        for side in (0, 1):
            missing[side].update(record['missing_rows_by_side'][side])
            observations[side].update(record['all_rows_by_side'][side])
    return dict(before=totals(before), after=totals(after), failed_branches=len(branches),
                dropped_attack=sum(b['exact_attack']['dropped'] for b in branches),
                dropped_response=sum(b['exact_response']['dropped'] for b in branches),
                missing_by_side=missing, all_observation_reasons_by_side=observations,
                diagnostic_denominator_per_side=denominator)


def main() -> None:
    """読みの測定を済ませてから、原票の集計と実装指紋を保存する。"""
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    refined.OUT = OUT
    if not args.metadata_only:
        refined.measure_coverage()
    for name, result in (('summary', summary()), ('manifest', manifest())):
        (OUT/f'{name}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(name, result, flush=True)


if __name__ == '__main__':
    main()
