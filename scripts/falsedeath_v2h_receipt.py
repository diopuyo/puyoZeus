"""D-v2hの入力同一性・本番と既存2方式の互換性・資源制限を保存する。"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

from scripts.falsedeath_replay_20261001 import OUT, records, input_record
from scripts.falsedeath_v2d_receipt import digest, package, SOURCES
from scripts.falsedeath_v2d_static_checks import check_python, ROOT
from scripts.falsedeath_v2d_report import table

DEST = OUT/'v2h'
SHORT_SOURCES = ('c5', 'mia8KCjr52g', 'review')
TASK_STARTED = '2026-10-02T23:18:00.5790285+09:00'


def compatibility() -> dict:
    """既定OFFと従来E35‴・E35⁗の再生出力を実ファイルで再確認する。"""
    result = {}
    paths = [(f'prod/{source}', DEST/'tests_tmp'/f'test_production_off_is_byte_id{idx}',
              OUT/'replay/prod'/source) for idx, source in enumerate(SOURCES)]
    paths += [(f'{mode}/{source}', DEST/f'compat_{mode}'/source, OUT/mode/'short'/source)
              for mode in ('v2f', 'v2g') for source in SHORT_SOURCES]
    for name, actual, expected in paths:
        files = {}
        for filename in ('display.npz', 'events.jsonl'):
            assert (actual/filename).read_bytes() == (expected/filename).read_bytes(), (name, filename)
            files[filename] = dict(sha256=digest(actual/filename), comparison='byte_identical')
        result[name] = files
    return result


def static_checks() -> dict:
    """新規関数の長さ・型注釈と起動シェルの構文を検査する。"""
    paths = [ROOT/'src/exchange_post_counter_recent.py', ROOT/'src/exchange_post_counter_bound.py',
             ROOT/'src/exchange_post_counter_prefix.py', ROOT/'tests/test_falsedeath_v2h.py']
    paths += list((ROOT/'scripts').glob('falsedeath_v2h*.py'))
    python = [check_python(p) for p in paths]
    shell = list((ROOT/'scripts').glob('run_falsedeath_v2h*.sh'))
    for path in shell:
        subprocess.run(['bash', '-n', str(path)], check=True)
    return dict(python=python, shell=[str(p.relative_to(ROOT)) for p in shell])


def verification_report(value: dict) -> None:
    """全表へ検査結果と作業全体の所要時間を追記する。"""
    text = table('検査結果', ['項目', '結果'], [
        ['単体', (DEST/'unit_tests.log').read_text().splitlines()[-1]],
        ['回帰', (DEST/'regression_tests.log').read_text().splitlines()[-1]],
        ['入力18記録のv2d原票一致', value['identical_to_v2d']],
        ['新規コードの型注釈・関数50行以内・シェル構文', '通過']])
    text += table('既定OFF・既存照合の互換性', ['構成/記録', '表示', 'イベント'],
        [[name, row['display.npz']['comparison'], row['events.jsonl']['comparison']]
         for name, row in value['compatibility'].items()])
    text += table('所要時間（秒）', ['範囲', '値'], [
        ['作業開始から受領票まで', value['task_wall_seconds']],
        ['梯子開始から受領票まで', value['wall_seconds']],
        ['短区間・全体・互換再生の計算時間合計', value['replay_seconds_sum']]])
    with (DEST/'REPORT.md').open('a', encoding='utf-8') as stream:
        stream.write(text)


def main() -> None:
    """実施済み範囲の原票のみを圧縮し、未来の実行を完了に数えない。"""
    ladder = json.loads((DEST/'LADDER.json').read_text())
    old = json.loads((OUT/'v2d/RECEIPT.json').read_text())['input_records']
    inputs = {s: dict(path=str(input_record('v2h', s)), sha256=digest(input_record('v2h', s))) for s in records()}
    assert all(v['sha256'] == old[s]['sha256'] for s, v in inputs.items())
    checks, compat = static_checks(), compatibility()
    directories = {f'short/{s}': DEST/'short'/s for s in SHORT_SOURCES}
    if ladder['passed']:
        directories.update({f'full/{s}': OUT/'replay/v2h'/s for s in records()})
    directories.update({f'compat_{m}/{s}': DEST/f'compat_{m}'/s
                        for m in ('v2f', 'v2g') for s in SHORT_SOURCES})
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
        started=started.isoformat(), finished=finished.isoformat(), wall_seconds=(finished-started).total_seconds(),
        task_started=TASK_STARTED, task_wall_seconds=(finished-datetime.fromisoformat(TASK_STARTED)).total_seconds())
    with (DEST/'RECEIPT.json').open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=1)
    verification_report(value)
    print(json.dumps(dict(full_records=value['full_records'], wall_seconds=value['wall_seconds'])))


if __name__ == '__main__':
    main()
