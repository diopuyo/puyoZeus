"""E35‴の入力同一性・既定OFF互換・実行資源と再生成果物を保存する。"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from scripts.falsedeath_replay_20261001 import OUT, records, input_record
from scripts.falsedeath_v2d_receipt import digest, package, SOURCES
from scripts.falsedeath_v2d_static_checks import check_python, ROOT

DEST = OUT/'v2f'


def off_receipts() -> list[dict]:
    """pytestが再生した5記録の出力を、既存原票と再度バイト比較する。"""
    result = []
    for index, source in enumerate(SOURCES):
        actual = DEST/'tests_tmp'/f'test_production_off_is_byte_id{index}'
        expected = OUT/'replay/prod'/source
        files = {}
        for name in ('display.npz', 'events.jsonl'):
            assert (actual/name).read_bytes() == (expected/name).read_bytes(), (source, name)
            files[name] = dict(sha256=digest(actual/name), comparison='byte_identical')
        result.append(dict(source=source, files=files))
    actual = DEST/'final_off_tmp/test_production_off_is_byte_id0'
    for name in ('display.npz', 'events.jsonl'):
        assert (actual/name).read_bytes() == (OUT/'replay/prod/review'/name).read_bytes()
    result.append(dict(source='review', final_code=True, comparison='byte_identical'))
    return result


def main() -> None:
    """梯子で止まった場合も、実施済み範囲だけの受領票を残す。"""
    ladder = json.loads((DEST/'LADDER.json').read_text())
    old = json.loads((OUT/'v2d/RECEIPT.json').read_text())['input_records']
    inputs = {s: dict(path=str(input_record('v2f', s)), sha256=digest(input_record('v2f', s)))
              for s in records()}
    assert all(v['sha256'] == old[s]['sha256'] for s, v in inputs.items())
    directories = {f'short/{s}': DEST/'short'/s for s in ('c5', 'mia8KCjr52g', 'review')}
    if ladder['passed']:
        directories.update({f'full/{s}': OUT/'replay/v2f'/s for s in records()})
    directories.update({f'attempt1/{s}': DEST/'attempt1/short'/s for s in ('c5', 'mia8KCjr52g', 'review')})
    resources = {n: json.loads((p/'RESOURCE.json').read_text()) for n, p in directories.items()}
    assert all(v['nice'] == 10 and v['available_kib'] >= 6*1024**2 for v in resources.values())
    assert all(all(v == '1' for v in r['pools'].values()) for r in resources.values())
    evidence = {n: package(p, DEST/'evidence'/n) for n, p in directories.items()}
    times = {n: json.loads((p/'status.json').read_text())['elapsed_seconds'] for n, p in directories.items()}
    files = ['src/exchange_post_counter_prefix.py', 'src/exchange_post_counter_bound.py',
             'tests/test_falsedeath_v2f.py']
    files += [str(p.relative_to(ROOT)) for p in (ROOT/'scripts').glob('falsedeath_v2f*.py')]
    checks = [check_python(ROOT/p) for p in files]
    started = datetime.fromisoformat((DEST/'attempt1/worker.log').read_text().splitlines()[0].removeprefix('START '))
    finished = datetime.now(timezone.utc)
    value = dict(input_records=inputs, identical_to_v2d=True, full_records=18 if ladder['passed'] else 0,
        off_byte_comparisons=off_receipts(), evidence=evidence, resources=resources, static_checks=checks,
        replay_seconds=times, replay_seconds_sum=sum(times.values()), launcher='run_limited',
        started=started.isoformat(), finished=finished.isoformat(), wall_seconds=(finished-started).total_seconds())
    with (DEST/'RECEIPT.json').open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=1)
    print(json.dumps(dict(full_records=value['full_records'], wall_seconds=value['wall_seconds'])))


if __name__ == '__main__':
    main()
