"""D-v2h2bの入力同一性・互換性・資源・圧縮原票を受領票へ保存する。"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts.falsedeath_replay_20261001 import OUT, records, input_record
from scripts.falsedeath_v2d_receipt import digest, package, SOURCES
from scripts.falsedeath_v2d_static_checks import check_python, ROOT
from scripts.falsedeath_v2d_report import table

DEST = OUT/'v2h2b'
TASK_STARTED = '2026-10-02T17:42:41+00:00'
SHORT_SOURCES = ('c5', 'mia8KCjr52g', 'review')


def static_checks() -> dict:
    """既存巨大関数以外の変更ファイルの型・長さとシェル構文を検査する。"""
    paths = [ROOT/'src/exchange_post_counter_recent.py', ROOT/'src/exchange_post_counter_bound.py',
             ROOT/'src/exchange_event_cli.py', ROOT/'tests/test_falsedeath_v2h2b.py']
    paths += sorted((ROOT/'scripts').glob('falsedeath_v2h2b*.py'))
    result = [check_python(path) for path in paths]
    shell = sorted((ROOT/'scripts').glob('run_falsedeath_v2h2b*.sh'))
    for path in shell:
        subprocess.run(['bash', '-n', str(path)], check=True)
    return dict(python=result, shell=[str(p.relative_to(ROOT)) for p in shell])


def compatibility() -> dict:
    """実再生した旧D-v2h2と本番の表示・イベントバイトを照合する。"""
    paths = {f'v2h2/{s}': (DEST/'compat_v2h2'/s, OUT/'v2h2/short'/s) for s in SHORT_SOURCES}
    paths.update({f'prod/{s}': (DEST/'production_tmp'/f'test_production_off_is_byte_id{i}',
                              OUT/'replay/prod'/s) for i, s in enumerate(SOURCES)})
    result = {}
    for name, (actual, expected) in paths.items():
        result[name] = {}
        for filename in ('display.npz', 'events.jsonl'):
            assert (actual/filename).read_bytes() == (expected/filename).read_bytes(), (name, filename)
            result[name][filename] = dict(sha256=digest(actual/filename), comparison='byte_identical')
    return result


def verification_report(value: dict) -> None:
    """検査と所要時間を全表へ追記する。"""
    text = table('検査結果', ['項目', '結果'], [
        ['単体（旧D-v2h2含む）', (DEST/'unit.log').read_text().splitlines()[-1]],
        ['予測段数欠測を含む最終単体', (DEST/'unit_final.log').read_text().splitlines()[-1]],
        ['周辺回帰', (DEST/'regression_tests_final.log').read_text().splitlines()[-1]],
        ['本番5再生', (DEST/'production_tests.log').read_text().splitlines()[-1]],
        ['skip理由', 'E2_SHORT_ARTIFACTS未設定（旧E2動画検証）'],
        ['18入力のv2d原票一致', value['identical_to_v2d']],
        ['18記録のH1開始判定・同時刻証拠D-v2h2一致', all(r['games_equal'] and r['shared_evidence_equal'] for r in value['history_identical_to_v2h2'].values())],
        ['型・関数50行・シェル構文', '通過']])
    text += table('既定OFFの互換性', ['構成/記録', '表示', 'イベント'],
        [[name, row['display.npz']['comparison'], row['events.jsonl']['comparison']]
         for name, row in value['compatibility'].items()])
    text += table('所要時間（秒）', ['範囲', '値'], [
        ['作業開始から受領票', value['task_wall_seconds']],
        ['短区間・全体・互換再生の計算時間合計', value['replay_seconds_sum']]])
    text += ('H1のgamesは18記録すべて一致。同時刻の開始証拠も一致。rowsは証明判定時だけ記録されるため、'
             '鮮度補正による評価経路変化で時刻・件数が変わる。全ファイル一致を要求した初回受領票検査は過剰条件で停止し、'
             '開始判定と共通時刻の証拠を比較する検査へ修正した。\n\n')
    with (DEST/'REPORT.md').open('a') as stream:
        stream.write(text)


def collect_evidence() -> tuple[dict, dict, dict]:
    """完了した再生だけを圧縮し、各実計算の資源制限を確認する。"""
    directories = {f'short/{s}': DEST/'short'/s for s in SHORT_SOURCES}
    directories.update({f'full/{s}': OUT/'replay/v2h2b'/s for s in records()})
    directories.update({f'compat_v2h2/{s}': DEST/'compat_v2h2'/s for s in SHORT_SOURCES})
    evidence, resources, times = {}, {}, {}
    for name, directory in directories.items():
        status = json.loads((directory/'status.json').read_text())
        assert status['state'] == 'completed', name
        times[name] = status['elapsed_seconds']
        resource = json.loads((directory/'RESOURCE.json').read_text())
        assert resource['nice'] == 10 and resource['available_kib'] >= 6*1024**2
        assert all(v == '1' for v in resource['pools'].values())
        resources[name] = resource
        evidence[name] = package(directory, DEST/'evidence'/name)
    return evidence, resources, times



def history_comparison() -> dict:
    """全観測で更新する開始判定と、証明時だけ追加される監査行を区別する。"""
    result = {}
    keys = ('match_start_observed_sec', 'history_complete', 'boundary_score_evidence')
    for source in records():
        old, new = [json.loads((OUT/'replay'/mode/source/'history_integrity_audit.json').read_text())
                    for mode in ('v2h2', 'v2h2b')]
        before, after = [{(r['game'], r['t_sec']): {k: r[k] for k in keys} for r in v['rows']}
                         for v in (old, new)]
        common = before.keys() & after.keys()
        result[source] = dict(games_equal=old['games'] == new['games'],
            shared_evidence_equal=all(before[k] == after[k] for k in common),
            shared_times=len(common), old_audit_rows=len(old['rows']), new_audit_rows=len(new['rows']))
        assert result[source]['games_equal'] and result[source]['shared_evidence_equal'], source
    return result


def main() -> None:
    """原票ハッシュと全検査を新規受領票に固定する。"""
    old = json.loads((OUT/'v2d/RECEIPT.json').read_text())['input_records']
    inputs = {s: dict(path=str(input_record('v2h2b', s)), sha256=digest(input_record('v2h2b', s)))
              for s in records()}
    assert all(v['sha256'] == old[s]['sha256'] for s, v in inputs.items())
    history_identical = history_comparison()
    checks, compat = static_checks(), compatibility()
    evidence, resources, times = collect_evidence()
    finished = datetime.now(timezone.utc)
    value = dict(input_records=inputs, identical_to_v2d=True, full_records=18,
        compatibility=compat, evidence=evidence, resources=resources, static_checks=checks,
        history_identical_to_v2h2=history_identical,
        replay_seconds=times, replay_seconds_sum=sum(times.values()), launcher='run_limited',
        task_started=TASK_STARTED, finished=finished.isoformat(),
        task_wall_seconds=(finished-datetime.fromisoformat(TASK_STARTED)).total_seconds())
    with (DEST/'RECEIPT.json').open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=1)
    verification_report(value)
    print(json.dumps(dict(full_records=18, task_wall_seconds=value['task_wall_seconds'])))


if __name__ == '__main__':
    main()
