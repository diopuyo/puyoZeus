"""診断専用成果物の構文・関数長・原票整合・資源記録を検証する。"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any
from scripts._diag_review1003b_extract import ROOT, OUT, CASES, resources, save, records

MAX_FUNCTION_LINES = 50
EXPECTED_ROWS = 1382
TOLERANCE = 1e-10


def check_scripts() -> list[dict[str, Any]]:
    """許可名の診断スクリプトだけを静的検証する。"""
    rows = []
    for path in sorted(Path('scripts').glob('_diag_review1003b_*.py')):
        source = path.read_text(encoding='utf-8')
        compile(source, str(path), 'exec')
        functions = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)]
        oversized = [(n.name, n.end_lineno-n.lineno+1) for n in functions
                     if n.end_lineno-n.lineno+1 > MAX_FUNCTION_LINES]
        assert not oversized, (path, oversized)
        assert all(n.returns is not None for n in functions), path
        rows.append(dict(path=str(path), functions=len(functions),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return rows


def check_tables() -> list[dict[str, Any]]:
    """全フレーム数と未来の証拠を参照していないことを確認する。"""
    checks = []
    for case, _, _, start, end in CASES:
        rows = records(OUT / f'case{case}_timeline.jsonl')
        frames = list(csv.DictReader((OUT / f'case{case}_full.csv').open(encoding='utf-8-sig')))
        assert len(rows) == len(frames)
        for row in rows:
            stamp = float(row['frame']['t_sec'])
            assert start <= stamp <= end
            for name in ('input', 'landing', 'score_prediction'):
                assert row[name].get('t_sec', stamp) <= stamp
        single = [p for r in rows for p in r['landing'].get('single_death_proof', []) if p]
        checks.append(dict(case=case, frames=len(rows), single_proof_entries=len(single)))
    assert sum(r['frames'] for r in checks) == EXPECTED_ROWS
    return checks


def main() -> None:
    """再現性証拠と全資源記録の最低値を残す。"""
    current = resources()
    data = json.loads((OUT / 'gfe_same_input.json').read_text())
    error = max(abs(r['old_same_input']-r['old_record_p1']) for r in data)
    mismatches = [{k: r[k] for k in ('t_sec','old_same_input','old_record_p1')}
                  for r in data if abs(r['old_same_input']-r['old_record_p1']) >= TOLERANCE]
    logs = [json.loads(line) for case in (4, 6, 7)
            for line in (OUT / f'replay{case}.log').read_text().splitlines() if line.startswith('{')]
    result = dict(base_commit='802efbc', scripts=check_scripts(), tables=check_tables(),
        gfe_calls=len(data), gfe_old_max_error=error, gfe_old_record_mismatches=mismatches,
        replay_resource_samples=len(logs),
        min_host_gib=min(r['host_kib'] for r in logs)/(1024**2),
        min_linux_gib=min(r['linux_kib'] for r in logs)/(1024**2),
        nice_values=sorted({r['nice'] for r in logs}), current_resources=current,
        replay_equivalence={str(n): json.loads((OUT / f'case{n}_support.json').read_text())['equivalence']
                            for n in (4, 6, 7)})
    assert min(result['min_host_gib'], result['min_linux_gib']) >= 6
    save('RECEIPT.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
