"""原票照合・再生成果物・実行時間の受領票を作り、圧縮証拠を保存する。"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.falsedeath_replay_20261001 import OUT, records

DEST = OUT/'v2c'
SOURCES = ('review', 'q_7gc4TgFig', 'fcXG83vInDY', 'mia8KCjr52g', 'zenchi')


def digest(path: Path) -> str:
    """受領票用のファイルハッシュ。比較自体はバイト列でも行う。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def off_receipts() -> list[dict]:
    """pytestが実再生した5出力を再度ファイルバイトで原票に照合する。"""
    result = []
    for index, source in enumerate(SOURCES):
        actual = DEST/'tests_tmp'/f'test_production_off_is_byte_id{index}'
        expected = OUT/'replay/prod'/source
        rows = {}
        for name in ('display.npz', 'events.jsonl'):
            assert (expected/name).read_bytes() == (actual/name).read_bytes(), (source, name)
            rows[name] = dict(sha256=digest(actual/name), bytes=(actual/name).stat().st_size,
                              comparison='byte_identical')
        status = json.loads((actual/'status.json').read_text())
        result.append(dict(source=source, files=rows, status=status))
    return result


def package(directory: Path, target: Path) -> dict:
    """再現に必要な表示・イベント・E35監査を無損失圧縮して保存する。"""
    target.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in ('display.npz','events.jsonl','events.diagnostics.json',
                 'post_counter_bound_audit.json','status.json','COMMAND.json'):
        path = directory/name
        if not path.exists():
            continue
        data = path.read_bytes()
        output = target/(name if name.endswith('.npz') else name+'.gz')
        with output.open('xb') as stream:
            stream.write(data if name.endswith('.npz') else gzip.compress(data, mtime=0))
        manifest[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                             artifact=str(output.relative_to(OUT)))
    return manifest


def main() -> None:
    """合否を変えずに、完了範囲とC6の実行証拠をひとつにまとめる。"""
    ladder = json.loads((DEST/'LADDER.json').read_text())
    off = off_receipts()
    directories = {f'short/{s}':DEST/'short'/s for s in ('c5','mia8KCjr52g','review')}
    if ladder['passed']:
        directories.update({f'full/{s}':OUT/'replay/v2c'/s for s in records()})
    manifests, times = {}, {}
    for name, directory in directories.items():
        status = json.loads((directory/'status.json').read_text())
        assert status['state'] == 'completed'
        times[name] = status['elapsed_seconds']
        manifests[name] = package(directory, DEST/'evidence'/name)
    value: dict[str, Any] = dict(off_byte_comparisons=off, evidence=manifests,
        replay_seconds=times, replay_seconds_total=sum(times.values()),
        off_replay_seconds_total=sum(v['status']['elapsed_seconds'] for v in off),
        full_records=18 if ladder['passed'] else 0, nice=10, blas_threads=1,
        min_available_memory_gib=6, launcher='scripts/run_falsedeath_v2c.sh -> run_limited')
    with (DEST/'RECEIPT.json').open('x', encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=1)
    print(json.dumps(dict(full_records=value['full_records'], off_records=len(off))))


if __name__ == '__main__':
    main()
