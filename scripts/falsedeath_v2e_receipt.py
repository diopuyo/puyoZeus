"""D-v2eの入力同一性、既定OFFのバイト一致、画像を含まない再生成果物を保存する。"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.falsedeath_replay_20261001 import OUT, input_record, records
from scripts.falsedeath_v2d_receipt import SOURCES, digest, package

DEST = OUT/'v2e'


def off_receipts() -> list[dict]:
    """今回のpytest合格ログを記録し、一時領域整理後の出力保持と区別する。"""
    log = (DEST/'regression_tests.log').read_text()
    failed = [line for line in log.splitlines() if line.startswith('FAILED ')]
    assert failed == ['FAILED tests/test_exchange_event_tracker.py::test_new_functions_within_fifty_lines']
    assert '291 passed, 1 skipped' in log
    result = []
    for source in SOURCES:
        expected = OUT/'replay/prod'/source
        files = {name: dict(expected_sha256=digest(expected/name), comparison='byte_identical')
                 for name in ('display.npz', 'events.jsonl')}
        result.append(dict(source=source, files=files, test='test_production_off_is_byte_identical',
            evidence='regression_tests.log', actual_artifact_retained=False,
            note='pytest一時出力は他テストの整理で消失。合格ログを保持。唯一の失敗は行数規約で別途修正・再試験。'))
    return result


def main() -> None:
    """D-v2dと同じ入力ハッシュを必須とし、新規の圧縮証拠だけを作る。"""
    off = off_receipts()
    old = json.loads((OUT/'v2d/RECEIPT.json').read_text())['input_records']
    inputs = {s: dict(path=str(input_record('v2e', s)), sha256=digest(input_record('v2e', s)))
              for s in records()}
    assert all(row['sha256'] == old[s]['sha256'] for s, row in inputs.items())
    directories = {f'short/{s}': DEST/'short'/s for s in ('c5', 'mia8KCjr52g', 'review')}
    directories.update({f'full/{s}': OUT/'replay/v2e'/s for s in records()})
    evidence = {name: package(path, DEST/'evidence'/name) for name, path in directories.items()}
    result = dict(input_records=inputs, identical_to_v2d=True, full_records=len(records()),
                  off_byte_comparisons=off, evidence=evidence)
    with (DEST/'RECEIPT.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=1)
    print(json.dumps(dict(full_records=len(records()), off_records=len(off), inputs_identical=True)))


if __name__ == '__main__':
    main()
