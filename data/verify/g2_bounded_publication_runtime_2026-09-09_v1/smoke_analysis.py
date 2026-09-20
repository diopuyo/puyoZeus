"""既保存の人工4appendで新解析の型復元経路だけを事前確認する。"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time
import analyze_metadata as A


def main() -> int:
    output = A.L.ROOT / sys.argv[1]
    output.mkdir(exist_ok=False)
    root = A.L.ROOT.parent / 'g2_collector_metadata_bounded_2026-09-09_v1/independent/cpu_v1'
    receipt = json.loads((root / 'COMPLETE.json').read_text())
    before = {str(root / n): h for n, h in receipt['sha256'].items()}
    before |= {str(p): A.L.F.sha(p) for p in (Path(__file__), Path(A.__file__))}
    A.L.F.require(all(A.L.F.sha(Path(p)) == h for p, h in before.items()), 'smoke_saved_changed')
    started, c = time.perf_counter(), A.collector()
    path = root / 'tmp/test_actual_four_calls_gray_an0/True/collector_metadata.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    results = [A.replay(c, item) for row in rows for item in row['appends']]
    A.L.F.require(len(results) == 4 and all(not r['differences'] for r in results), 'smoke_replay_changed')
    after = {p: A.L.F.sha(Path(p)) for p in before}
    A.L.F.require(before == after, 'smoke_input_changed')
    result = dict(pid=os.getpid(), actual_exit=0, seconds=time.perf_counter() - started,
        before=before, after=after, rows=results, artificial_only=True, quality_gate_clear=False)
    A.L.F.P.write(output / 'RESULT.json', result)
    A.L.F.P.write(output / 'COMPLETE.json', {'sha256': {'RESULT.json': A.L.F.sha(output / 'RESULT.json')}})
    print({k: v for k, v in result.items() if k not in ('before', 'after')}, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
