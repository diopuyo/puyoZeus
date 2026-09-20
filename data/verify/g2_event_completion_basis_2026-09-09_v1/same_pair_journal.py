"""実同色NEXT進行区間の原FIFO票だけを既存抽出器で確認する。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'g2_historical_next_hand_boundary_2026-09-09_v1/probe.py'
FIRST, LAST, SIDE = 33790, 33808, '1P'


def main(output: Path) -> int:
    output.mkdir(exist_ok=False)
    started, before = time.perf_counter(), hashlib.sha256(OLD.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location('_same_pair_existing_scan', OLD)
    old: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    saved = old.FIRST, old.LAST, old.SIDE
    try:
        old.FIRST, old.LAST, old.SIDE = FIRST, LAST, SIDE
        extracted = old.scan('atomic_journal.jsonl')
    finally:
        old.FIRST, old.LAST, old.SIDE = saved
    if hashlib.sha256(OLD.read_bytes()).hexdigest() != before:
        raise RuntimeError('scanner_changed')
    rows = [item['row'] for item in extracted['rows']]
    enqueue = [row for row in rows if row['kind'] == 'enqueue']
    steps = [row for row in rows if row['kind'] == 'step']
    expected = list(range(FIRST, LAST + 2, 2))
    if [row['frame_idx'] for row in enqueue] != expected or [row['frame_idx'] for row in steps] != expected:
        raise RuntimeError('coverage')
    facts = [{'frame': row['frame_idx'], 'pair': row['pair'], 'added': row['added_occurrence_tokens'],
              'pending_before': row['before']['pending_tsumo'],
              'pending_after': row['after']['pending_tsumo']} for row in enqueue]
    result = {'pid': os.getpid(), 'seconds': time.perf_counter() - started, 'scanner_sha256': before,
              'extracted': extracted, 'enqueue_facts': facts,
              'physical_labels_from': 'PARENT_OBSERVATIONS.md / 原PNG目視',
              'current_permission': False, 'quality_gate_clear': False}
    with (output / 'RESULT.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print({'pid': result['pid'], 'seconds': result['seconds'], 'enqueue': facts})
    return 0


if __name__ == '__main__':
    raise SystemExit(main(Path(sys.argv[1])))
