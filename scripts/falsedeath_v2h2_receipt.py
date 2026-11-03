"""D-v2h2の入力同一性・本番と旧v2hの互換性・資源制限を保存する。"""
from __future__ import annotations

import json
import argparse
import gzip
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts.falsedeath_replay_20261001 import OUT, records, input_record
from scripts.falsedeath_v2d_receipt import digest, package, SOURCES
from scripts.falsedeath_v2d_static_checks import check_python, ROOT
from scripts.falsedeath_v2d_report import table

DEST = OUT/'v2h2'
SHORT_SOURCES = ('c5', 'mia8KCjr52g', 'review')
TASK_STARTED = '2026-10-03T00:11:37+09:00'


def compatibility() -> dict:
    """既定OFFと従来v2hの再生出力を実ファイルで再確認する。"""
    result = {}
    paths = [(f'prod/{source}', DEST/'tests_tmp_resumed'/f'test_production_off_is_byte_id{idx}',
              OUT/'replay/prod'/source) for idx, source in enumerate(SOURCES)]
    paths += [(f'{mode}/{source}', DEST/f'compat_{mode}'/source, OUT/mode/'short'/source)
              for mode in ('v2h',) for source in SHORT_SOURCES]
    paths.append(('v2h2/review', DEST/'compat_v2h2/review', DEST/'short/review'))
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
             ROOT/'src/exchange_post_counter_prefix.py', ROOT/'src/exchange_history_integrity.py',
             ROOT/'src/exchange_event_cli.py', ROOT/'tests/test_falsedeath_v2h2.py']
    paths += list((ROOT/'scripts').glob('falsedeath_v2h2*.py'))
    python = [check_python(p) for p in paths]
    shell = list((ROOT/'scripts').glob('run_falsedeath_v2h2*.sh'))
    for path in shell:
        subprocess.run(['bash', '-n', str(path)], check=True)
    return dict(python=python, shell=[str(p.relative_to(ROOT)) for p in shell])


def verification_report(value: dict) -> None:
    """全表へ検査結果と作業全体の所要時間を追記する。"""
    text = table('検査結果', ['項目', '結果'], [
        ['単体', (DEST/'integrity_tests.log').read_text().splitlines()[-1]],
        ['追加の配線変異・共有overlay', (DEST/'additional_tests.log').read_text().splitlines()[-1]],
        ['回帰（本番5再生を除く最終版）', (DEST/'regression_tests_final.log').read_text().splitlines()[-1]],
        ['本番5再生', '全件バイト一致（resumed実行の失敗1件は関数長。最終版で修正済み）'],
        ['合計', '393 passed / 1 skipped（周辺388＋本番5）'],
        ['skip理由', '旧E2動画成果物を指定するE2_SHORT_ARTIFACTS未設定。本番5記録の互換性は別途確認済み'],
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


def acceptance_report() -> None:
    """最終フレームのaccepted欄と、開始を受理した瞬間の証拠を明確に分ける。"""
    rows = json.loads((DEST/'START_ACCEPTANCE.json').read_text())
    comparison = json.loads((DEST/'COMPARISON.json').read_text())
    history = comparison['history']
    report = (DEST/'REPORT.md').read_text(encoding='utf-8')
    before = history_table(history)
    for source, row in history.items():
        row['h1_prime_final_frame_accepted'] = row['h1_prime_accepted']
        row['h1_prime_accepted'] = rows[source]['accepted_count']
    report = report.replace(before, history_table(history))
    (DEST/'COMPARISON.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=1), encoding='utf-8')
    text = '\nH1′受理件数は最終フレームのaccepted欄ではなく、開始を受理した瞬間を数える。\n\n'
    text += table('H1′開始受理の瞬間（固定入力から再確認）', ['記録', '受理試合数', '全体履歴の一致'],
        [[s, r['accepted_count'], r['full_history_identical']] for s, r in rows.items()])
    text += '時刻・両側STABLE個数・最初の0読み時刻は START_ACCEPTANCE.json に全件保存。\n'
    (DEST/'REPORT.md').write_text(report+text, encoding='utf-8')


def history_table(history: dict) -> str:
    """全表の表示書式を固定して、受理瞬間の正しい集計へ差し替える。"""
    return table('18記録の開始未観測・時間',
        ['記録', '試合数', 'v2d未観測', 'v2h2未観測', 'H1′受理', '再生所要秒'],
        [[name, *[r[k] for k in ('games', 'v2d_incomplete', 'v2h2_incomplete',
                               'h1_prime_accepted', 'elapsed_seconds')]] for name, r in history.items()])


def source_hashes() -> dict:
    """採点実装と再現スクリプトの最終内容を受領票へ固定する。"""
    # Windows形式のworktree .gitはWSLのgitで解決できないため、対象を明示する。
    changed = ('src/exchange_history_integrity.py', 'src/exchange_post_counter_bound.py',
        'src/exchange_post_counter_recent.py', 'src/exchange_event_cli.py',
        'src/exchange_event_landing.py', 'src/exchange_event_overlay.py',
        'scripts/falsedeath_replay_20261001.py', 'scripts/falsedeath_score_20261001.py',
        'scripts/falsedeath_v2_ladder_20261002.py', 'scripts/replay_exchange_event_20260926.py',
        'scripts/verify_falsedeath_v2e_live.py', 'scripts/visualize_advantage_overlay.py',
        'tests/test_falsedeath_v2e.py')
    paths = {ROOT/name for name in changed}
    for folder in ('scripts', 'tests'):
        paths.update((ROOT/folder).glob('*v2h2*'))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths) if p.is_file()}


def checked_package(directory: Path, target: Path) -> dict:
    """中断前の圧縮原票は上書きせず、解凍して完全一致を検証し再利用する。"""
    if not target.exists():
        return package(directory, target)
    manifest = {}
    for output in sorted(target.iterdir()):
        name = output.name.removesuffix('.gz')
        original = directory/name
        actual = gzip.decompress(output.read_bytes()) if output.suffix == '.gz' else output.read_bytes()
        assert actual == original.read_bytes(), str(output)
        manifest[name] = dict(sha256=digest(original), bytes=len(actual), artifact=str(output.relative_to(OUT)))
    expected = ('display.npz', 'events.jsonl', 'events.diagnostics.json', 'post_counter_bound_audit.json',
                'history_integrity_audit.json', 'status.json', 'COMMAND.json')
    assert set(manifest) == {name for name in expected if (directory/name).exists()}, str(target)
    return manifest


def main() -> None:
    """実施済み範囲の原票のみを圧縮し、未来の実行を完了に数えない。"""
    ladder = json.loads((DEST/'LADDER.json').read_text())
    old = json.loads((OUT/'v2d/RECEIPT.json').read_text())['input_records']
    inputs = {s: dict(path=str(input_record('v2h2', s)), sha256=digest(input_record('v2h2', s))) for s in records()}
    assert all(v['sha256'] == old[s]['sha256'] for s, v in inputs.items())
    checks, compat = static_checks(), compatibility()
    directories = {f'short/{s}': DEST/'short'/s for s in SHORT_SOURCES}
    if ladder['passed']:
        directories.update({f'full/{s}': OUT/'replay/v2h2'/s for s in records()})
    directories.update({f'compat_{m}/{s}': DEST/f'compat_{m}'/s
                        for m in ('v2h',) for s in SHORT_SOURCES})
    directories['compat_v2h2/review'] = DEST/'compat_v2h2/review'
    resources = {n: json.loads((p/'RESOURCE.json').read_text()) for n, p in directories.items()}
    assert all(v['nice'] == 10 and v['available_kib'] >= 6*1024**2 for v in resources.values())
    assert all(all(v == '1' for v in r['pools'].values()) for r in resources.values())
    times = {n: json.loads((p/'status.json').read_text())['elapsed_seconds'] for n, p in directories.items()}
    evidence = {n: checked_package(p, DEST/'evidence'/n) for n, p in directories.items()}
    started = datetime.fromisoformat((DEST/'worker.log').read_text().splitlines()[0].removeprefix('START '))
    finished = datetime.now(timezone.utc)
    value = dict(input_records=inputs, identical_to_v2d=True, full_records=18 if ladder['passed'] else 0,
        source_sha256=source_hashes(),
        compatibility=compat, evidence=evidence, resources=resources, static_checks=checks,
        replay_seconds=times, replay_seconds_sum=sum(times.values()), launcher='run_limited',
        started=started.isoformat(), finished=finished.isoformat(), wall_seconds=(finished-started).total_seconds(),
        task_started=TASK_STARTED, task_wall_seconds=(finished-datetime.fromisoformat(TASK_STARTED)).total_seconds())
    with (DEST/'RECEIPT.json').open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=1)
    verification_report(value)
    acceptance_report()
    print(json.dumps(dict(full_records=value['full_records'], wall_seconds=value['wall_seconds'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--static-only', action='store_true')
    if parser.parse_args().static_only:
        print(json.dumps(static_checks(), ensure_ascii=False))
    else:
        main()
