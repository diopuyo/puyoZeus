"""B3時間門専用の薄い計装と原票集計。固定モジュールは変更しない。"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator
import json
import os
import platform
import subprocess
import time

import numpy as np

MS = 1000.0
GIB = 1024 ** 3
CHAIN7 = 7
MEMORY_GATE_GIB = 6
# 2026-10-06 user伝授: 1列分(6マス)以上の空きを残す上限打切りを観測する。
LOOKAHEAD_EMPTY_THRESHOLD = 6


def write_json(path: Path, data: dict) -> None:
    """監査可能なJSONを書き出す。"""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def memory() -> dict:
    """WSLの利用可能メモリと自プロセスRSS。"""
    mem = {p[0].rstrip(':'): int(p[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
           if len(p := line.split()) == 3}
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
    return dict(available_gib=mem['MemAvailable'] / GIB, total_gib=mem['MemTotal'] / GIB,
                rss_gib=int(status['VmRSS'].split()[0]) * 1024 / GIB,
                threads=int(status['Threads']))


def environment() -> dict:
    """モデル・計測とは独立な端末情報を保存する。"""
    from threadpoolctl import threadpool_info
    from src import puyo_core_bridge
    cpu = next(x.split(':', 1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines()
               if x.startswith('model name'))
    return dict(cpu=cpu, logical_cpus=os.cpu_count(), platform=platform.platform(),
                python=platform.python_version(), pid=os.getpid(), memory=memory(),
                native=puyo_core_bridge.NATIVE_AVAILABLE, threadpools=threadpool_info(),
                env={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                  'MKL_NUM_THREADS', 'B3_HGB_THREADS')},
                processes=subprocess.check_output(['ps', '-eo', 'pid,comm,pcpu,rss', '--sort=-pcpu'], text=True))


class Instrumentation:
    """呼出先の引数・戻り値を変えず、探索・静止指標・台帳の時間を加算する。"""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.searches: list[dict] = []
        self.originals: list[tuple[Any, str, Callable]] = []

    def wrap(self, owner: Any, name: str, label: str) -> None:
        original = getattr(owner, name)

        def timed(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = original(*args, **kwargs)
            self.seconds[label] += time.perf_counter() - start
            self.calls[label] += 1
            if label == 'search':
                self.searches.append(dict(done=result.done_hands, hmax=result.hmax, died=result.beam_died))
            return result

        self.originals.append((owner, name, original))
        setattr(owner, name, timed)

    def install(self) -> None:
        from src import b3_features, b3_timeaxis
        from src.b3_score_ledger import ScoreLedger
        self.wrap(b3_features, 'ideal_fire_search', 'search')
        self.wrap(b3_timeaxis, 'ideal_fire_search', 'search')
        self.wrap(b3_features, 'static_indicators', 'indicators')
        self.wrap(ScoreLedger, 'update', 'ledger')

    def reset(self) -> None:
        self.seconds.clear()
        self.calls.clear()
        self.searches.clear()

    def restore(self) -> None:
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)


def quantiles(values: list[float]) -> dict:
    """空母数は未測定。補間法を固定する。"""
    if not values:
        return dict(n=0, p50=None, p95=None, p99=None, max=None)
    v = np.asarray(values, dtype=float)
    return dict(n=len(v), **dict(zip(('p50', 'p95', 'p99'), np.quantile(v, [.5, .95, .99]).tolist())),
                max=float(v.max()))


def search_status(row: dict) -> dict:
    """要求H未到達を更新単位のORで数える。K列の最大要求は既定の7手。"""
    from src.b3_column_spec import K_LEVELS
    from src.indicators_v2 import NEAR_FUTURE_KNOWN_HAND_SLOTS
    k_h = max(K_LEVELS) + NEAR_FUTURE_KNOWN_HAND_SLOTS
    bad_k = any(row[f'search_done_h_{s}'] < k_h for s in ('p1', 'p2'))
    bad_ta = any(row[f'ta_trunc_{s}'] == 1 or row[f'ta_done_h_{s}'] < row[f'ta_req_h_{s}']
                 for s in ('p1', 'p2'))
    req = [row[f'ta_req_h_{s}'] for s in ('p1', 'p2') if np.isfinite(row[f'ta_req_h_{s}'])]
    return dict(unreached=bool(bad_k or bad_ta), k_unreached=bool(bad_k), ta_unreached=bool(bad_ta),
                spacious_truncation=spacious_truncation(row),
                requested_h=max(req) if req else None,
                observed_chain7=any(row[f'chain_n_steps_{s}'] >= CHAIN7 or row[f'ep_chain7plus_{s}'] == 1
                                    for s in ('p1', 'p2')))


def spacious_truncation(row: dict, threshold: int = LOOKAHEAD_EMPTY_THRESHOLD) -> bool:
    """要求H未達・上限打切り・空き閾値以上を、両側/K/時間軸の更新単位ORで数える。"""
    from src.b3_column_spec import K_LEVELS
    from src.indicators_v2 import NEAR_FUTURE_KNOWN_HAND_SLOTS
    nan = float('nan')
    for prefix in ('search', 'ta'):
        for side in ('p1', 'p2'):
            req = (max(K_LEVELS) + NEAR_FUTURE_KNOWN_HAND_SLOTS if prefix == 'search'
                   else row.get(f'ta_req_h_{side}', nan))
            done = row.get(f'{prefix}_done_h_{side}', nan)
            if (req > done and row.get(f'{prefix}_trunc_empty_{side}', nan) >= threshold
                    and row.get(f'{prefix}_full_{side}', 0) != 1):
                return True
    return False


def lookahead_rates(rows: list[dict]) -> dict:
    """全更新を母数に、新しい先読み項目と従来の要求H未達率を併記する。"""
    result: dict = {}
    for label in ('unreached', 'spacious_truncation'):
        # 旧原票に新項目が無い場合は未測定。0%とは報告しない。
        values = [bool(r[label]) for r in rows if label in r and r[label] is not None]
        result[label] = dict(n=len(values), failures=sum(values),
                             rate=sum(values) / len(values) if values else None)
    return result


def summarize(directory: Path) -> dict:
    """計測原票だけを集計し、勝敗の採点はしない。"""
    rows = [json.loads(x) for x in (directory / 'updates.jsonl').read_text().splitlines()]
    frames = [json.loads(x) for x in (directory / 'frames.jsonl').read_text().splitlines()]
    costs = ('feature_ms', 'row_ms', 'state_ms', 'ledger_ms', 'search_ms', 'indicators_ms', 'inference_ms',
             'latency_ms', 'wait_ms', 'amortized_feature_ms')
    result = dict(updates=len(rows), frames=len(frames), games=len({r['game'] for r in rows}),
                  timings={k: quantiles([r[k] for r in rows]) for k in costs})
    result['by_cache'] = {label: {k: quantiles([r[k] for r in rows if bool(r['cache_misses']) == cold])
                                for k in costs[:8]} for label, cold in (('miss', True), ('hit_only', False))}
    result['first_update'] = rows[0] if rows else None
    result['lookahead_rates'] = lookahead_rates(rows)
    result['lookahead_empty_threshold'] = LOOKAHEAD_EMPTY_THRESHOLD
    result['search_rates'] = {label: dict(n=len(part), failures=sum(r['unreached'] for r in part),
                              rate=sum(r['unreached'] for r in part) / len(part) if part else None)
                             for label, part in (('all', rows), ('observed_chain7',
                                                 [r for r in rows if r['observed_chain7']]))}
    result['placement_latency'] = quantiles([r['latency_ms'] for r in rows if r['placement']])
    result['frame_wait'] = quantiles([r['wait_ms'] for r in frames])
    result['frames_wait_over_1s'] = sum(r['wait_ms'] > MS for r in frames)
    result['cache_hits'] = sum(r['cache_hits'] for r in rows)
    result['cache_misses'] = sum(r['cache_misses'] for r in rows)
    result['by_requested_h'] = {str(h): quantiles([r['feature_ms'] for r in rows if r['requested_h'] == h])
                                for h in sorted({r['requested_h'] for r in rows if r['requested_h'] is not None})}
    resources = [json.loads(x) for x in (directory / 'resources.jsonl').read_text().splitlines()]
    result['memory'] = dict(min_available_gib=min(r['available_gib'] for r in resources),
                            max_rss_gib=max(r['rss_gib'] for r in resources), samples=len(resources))
    write_json(directory / 'summary.json', result)
    return result
