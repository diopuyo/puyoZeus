"""10/12録画の同一コマ比較を集計し、未測定の表示品質と区別する。"""
from __future__ import annotations

import ctypes
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bench_live_delta import check_resources
from scripts.diagnose_live_speed3 import stats
from scripts.summarize_live_delta import compare, ROOT, WARMUP


def main() -> None:
    check_resources()
    loaded = {}
    result = dict(status='recognition_only', runs={})
    for name in ('obs_baseline', 'obs_delta'):
        path = ROOT / (name + '.json')
        rows = loaded[name] = json.loads(path.read_text(encoding='utf-8'))
        result['runs'][name] = dict(
            timing_ms=stats([r['ms'] for r in rows[WARMUP:]]),
            receipt=json.loads(path.with_suffix('.receipt.json').read_text(encoding='utf-8')))
    baseline, candidate = loaded['obs_baseline'], loaded['obs_delta']
    result['comparison'] = compare(baseline, candidate)
    result['changed_indices'] = [a['index'] for a, b in zip(baseline, candidate)
                                 if a['digest'] != b['digest']]
    result['note'] = '元実機の受理画素列の再現ではなく、録画を同一順序で認識した比較。'
    (ROOT / 'obs_comparison.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    main()
