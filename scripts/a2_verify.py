"""A2の既定OFF互換・資源条件・関数長を証拠付きで検収する。"""
from __future__ import annotations

import ast
import argparse
import hashlib
import json
from pathlib import Path

from scripts.a2_replay import OUT, INPUT
from scripts.falsedeath_replay_20261001 import records

SOURCES = ('review', 'q_7gc4TgFig', 'fcXG83vInDY', 'mia8KCjr52g', 'zenchi')
FILES = ('display.npz', 'events.jsonl', 'events.diagnostics.json')


def digest(path: Path) -> str:
    """大きな記録を丸ごとメモリへ積まずSHA-256を計算する。"""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compatibility() -> dict:
    """既存compareと同じく表示・イベントはバイト、診断は辞書で照合する。"""
    result = {}
    for source in SOURCES:
        rows = {}
        for name in FILES:
            before = OUT/'replay/reference'/source/name
            after = OUT/'replay/prod'/source/name
            rows[name] = dict(before=digest(before), after=digest(after),
                              byte_identical=before.read_bytes() == after.read_bytes())
            diagnostic = name == 'events.diagnostics.json'
            rows[name]['comparison'] = 'json' if diagnostic else 'bytes'
            rows[name]['equal'] = (json.loads(before.read_text()) == json.loads(after.read_text())
                                   if diagnostic else rows[name]['byte_identical'])
        result[source] = rows
    return result


def resources() -> dict:
    """全再生の開始時資源と入力チェックサムを保存する。"""
    result = {}
    for source in records():
        result[source] = dict(input_sha256=digest(INPUT/'collect/records'/f'{source}.jsonl.gz'))
        for mode in ('prod', 'a2'):
            base = OUT/'replay'/mode/source
            row = json.loads((base/'RESOURCE.json').read_text())
            assert row['nice'] == 10 and row['available_kib'] >= 6*1024**2
            assert set(row['pools'].values()) == {'1'}
            result[source][mode] = dict(**row, outputs={name: digest(base/name) for name in FILES})
    return result


def static_checks() -> dict:
    """追加した関数は全て50行以下とし、構文と既定OFFを確認する。"""
    root = Path(__file__).resolve().parents[1]
    names = ['src/exchange_formula_origin.py', 'tests/test_a2_safety.py']
    names += [str(p.relative_to(root)) for p in (root/'scripts').glob('a2_*.py')]
    rows = {}
    for name in names:
        tree = ast.parse((root/name).read_text(encoding='utf-8'))
        functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        rows[name] = {n.name: n.end_lineno-n.lineno+1 for n in functions}
        assert all(v <= 50 for v in rows[name].values()), (name, rows[name])
    from src.production_config import exchange_event_flags
    assert '--a2-safety-review' not in exchange_event_flags()
    return rows


def main() -> None:
    """採点条件の変更や本番ONを伴わない検収票だけを新規作成する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--static-only', action='store_true')
    if parser.parse_args().static_only:
        print(json.dumps(static_checks()), flush=True)
        return
    result = dict(base_commit='c13ea95', compatibility=compatibility(), resources=resources(),
                  static=static_checks())
    from src.production_config import exchange_event_flags
    result['production_flags'] = exchange_event_flags()
    root = Path(__file__).resolve().parents[1]
    names = ['src/exchange_formula_origin.py', 'src/exchange_history_integrity.py',
             'src/exchange_event_overlay.py', 'src/exchange_event_cli.py',
             'scripts/replay_exchange_event_20260926.py', 'scripts/visualize_advantage_overlay.py',
             'scripts/a2_resource_limits.sh']
    result['final_code_sha256'] = {name: digest(root/name) for name in names}
    result['passed'] = all(row['equal'] for group in result['compatibility'].values()
                           for row in group.values())
    (OUT/'VERIFICATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'])), flush=True)


if __name__ == '__main__':
    main()
