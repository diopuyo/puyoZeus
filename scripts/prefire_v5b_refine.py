"""5Bのキャッシュ容量是正後の診断。旧測定結果とは出力先を分ける。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from scripts import prefire_v5b_verify as verify
from scripts import prefire_v5b_coverage as coverage
from scripts import prefire_stable_queue_accuracy_20261001 as accuracy
from src.prefire_v5b_value import CachedM0, CachedStatic
from src.prefire_v5b_queue import SideQueueV5B

OUT = Path('logs/prefire_prediction/v5b/refined')


def proxy(overlay: SimpleNamespace) -> SimpleNamespace:
    """本体と同じM0と盤面特徴キャッシュを測定器へ渡す。"""
    return SimpleNamespace(**{**vars(overlay), '_m0': CachedM0(overlay._m0),
                              '_build_static': CachedStatic(overlay._build_static)})


def measure_coverage() -> None:
    """同じ診断行の欠測と、未来の設置を正解とした読みの正しさを分離する。"""
    results = {}
    for source in coverage.BASELINE_DIRS:
        result = coverage.coverage(source)
        frames = accuracy.frames(source)
        old = accuracy.SideQueue
        result['observed_pair_accuracy_before'] = [accuracy.evaluate_side(items) for items in frames]
        accuracy.SideQueue = SideQueueV5B
        try:
            result['observed_pair_accuracy_after'] = [accuracy.evaluate_side(items) for items in frames]
        finally:
            accuracy.SideQueue = old
        results[source] = result
        (OUT/f'coverage_{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(source, result, flush=True)


def main() -> None:
    """速度診断と欠測診断を独立したプロセスで起動できる。"""
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('benchmark', 'coverage', 'verify'))
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=1)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == 'coverage':
        measure_coverage()
        return
    verify.proxy = proxy
    samples = json.loads((verify.reference.OUT/'samples_ledger.json').read_text())
    overlay = verify.reference.model_overlay()
    for index in range(args.start, args.stop):
        path = OUT/f'{args.mode}_{index:03d}.json'
        if path.exists():
            continue
        result = (verify.benchmark if args.mode == 'benchmark' else verify.check)(samples[index], overlay)
        path.write_text(json.dumps(result), encoding='utf-8')
        print(index, result, flush=True)


if __name__ == '__main__':
    main()
