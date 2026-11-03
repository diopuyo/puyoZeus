"""D-v2gの入力同一性・本番とE35‴の互換性・資源制限を受領票へ保存する。"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

from scripts.falsedeath_replay_20261001 import OUT, records, input_record
from scripts.falsedeath_v2d_receipt import digest, package, SOURCES
from scripts.falsedeath_v2d_static_checks import check_python, ROOT

DEST = OUT/'v2g'
SHORT_SOURCES = ('c5', 'mia8KCjr52g', 'review')


def compatibility() -> dict:
    """既定OFFと従来E35‴の再生出力を実ファイルで再確認する。"""
    result = {}
    paths = [(f'prod/{source}', DEST/'tests_tmp'/f'test_production_off_is_byte_id{idx}',
              OUT/'replay/prod'/source) for idx, source in enumerate(SOURCES)]
    paths += [(f'v2f/{source}', DEST/'compat_v2f'/source, OUT/'v2f/short'/source) for source in SHORT_SOURCES]
    for name, actual, expected in paths:
        files = {}
        for filename in ('display.npz', 'events.jsonl'):
            assert (actual/filename).read_bytes() == (expected/filename).read_bytes(), (name, filename)
            files[filename] = dict(sha256=digest(actual/filename), comparison='byte_identical')
        result[name] = files
    return result


def static_checks() -> dict:
    """新規関数の長さ・型注釈と起動シェルの構文を検査する。"""
    paths = [ROOT/'src/exchange_post_counter_steps.py', ROOT/'src/exchange_post_counter_bound.py',
             ROOT/'src/exchange_post_counter_prefix.py', ROOT/'tests/test_falsedeath_v2g.py']
    paths += list((ROOT/'scripts').glob('falsedeath_v2g*.py'))
    python = [check_python(p) for p in paths]
    shell = list((ROOT/'scripts').glob('run_falsedeath_v2g*.sh'))
    for path in shell:
        subprocess.run(['bash', '-n', str(path)], check=True)
    return dict(python=python, shell=[str(p.relative_to(ROOT)) for p in shell])


def main() -> None:
    """実施済み範囲の原票のみを圧縮し、未来の実行を完了に数えない。"""
    ladder = json.loads((DEST/'LADDER.json').read_text())
    old = json.loads((OUT/'v2d/RECEIPT.json').read_text())['input_records']
    inputs = {s: dict(path=str(input_record('v2g', s)), sha256=digest(input_record('v2g', s))) for s in records()}
    assert all(v['sha256'] == old[s]['sha256'] for s, v in inputs.items())
    checks, compat = static_checks(), compatibility()
    directories = {f'short/{s}': DEST/'short'/s for s in SHORT_SOURCES}
    if ladder['passed']:
        directories.update({f'full/{s}': OUT/'replay/v2g'/s for s in records()})
    directories.update({f'compat_v2f/{s}': DEST/'compat_v2f'/s for s in SHORT_SOURCES})
    resources = {n: json.loads((p/'RESOURCE.json').read_text()) for n, p in directories.items()}
    assert all(v['nice'] == 10 and v['available_kib'] >= 6*1024**2 for v in resources.values())
    assert all(all(v == '1' for v in r['pools'].values()) for r in resources.values())
    times = {n: json.loads((p/'status.json').read_text())['elapsed_seconds'] for n, p in directories.items()}
    evidence = {n: package(p, DEST/'evidence'/n) for n, p in directories.items()}
    started = datetime.fromisoformat((DEST/'worker.log').read_text().splitlines()[0].removeprefix('START '))
    finished = datetime.now(timezone.utc)
    value = dict(input_records=inputs, identical_to_v2d=True, full_records=18 if ladder['passed'] else 0,
        compatibility=compat, evidence=evidence, resources=resources, static_checks=checks,
        replay_seconds=times, replay_seconds_sum=sum(times.values()), launcher='run_limited',
        started=started.isoformat(), finished=finished.isoformat(), wall_seconds=(finished-started).total_seconds())
    with (DEST/'RECEIPT.json').open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=1)
    print(json.dumps(dict(full_records=value['full_records'], wall_seconds=value['wall_seconds'])))


if __name__ == '__main__':
    main()
