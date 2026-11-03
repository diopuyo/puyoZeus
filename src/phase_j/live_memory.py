"""ライブ各processの保持量を動画時刻5分ごとに記録する任意計装。"""
from __future__ import annotations

from collections import Counter
import gc
import json
import logging
import os
from pathlib import Path
import time
import tracemalloc
from typing import Any

INTERVAL_SEC = 300.0
TOP_ALLOCATIONS = 25
INTEGRITY_AUDIT_LIMIT = 2048
CONTAINER_TYPES = frozenset({'ChainSimulator', 'RecognitionAudit', 'StableTransitionMonitor',
    'PuyoErasureMonitor', '_ColorStats', 'ExchangeEventRecorder', 'CellPatchFingerprint',
    'VideoChainTracker', 'OnlineHsvCalibrator', 'ExchangeRecord',
    'PostCounterDeathBound', 'MatchHistoryIntegrity'})


class RecentIntegrityAudit(list[dict]):
    """単一試合が長く続いても、live監査の直近件数を固定する。"""

    def __init__(self, rows: Any = ()) -> None:
        super().__init__()
        self.append_count = 0
        self.discarded_count = 0
        self.extend(rows)

    def append(self, row: dict) -> None:
        self.append_count += 1
        super().append(row)
        if len(self) > INTEGRITY_AUDIT_LIMIT:
            discarded = len(self)-INTEGRITY_AUDIT_LIMIT
            self.discarded_count += discarded
            del self[:discarded]

    def extend(self, rows: Any) -> None:
        for row in rows:
            self.append(row)

    def __reduce_ex__(self, protocol: int) -> tuple:
        """終了監査のprocess配送でも、append前にカウンタを初期化する。"""
        return type(self), (list(self),), dict(self.__dict__)

    def rows_since(self, cursor: int) -> list[dict]:
        """clearや先頭切り捨て後も、追加通番から今回の証明を取り出す。"""
        first = self.append_count-len(self)
        return self[max(0, cursor-first):]


def bound_integrity_audits(overlay: Any) -> None:
    """保存再生の全件保持指定は尊重し、通常liveだけ監査に上限を設ける。"""
    from src.exchange_integrity_memory import integrity_memory_roots
    for owner in integrity_memory_roots(overlay).values():
        if owner is None or owner.retain_audit or isinstance(owner.audit, RecentIntegrityAudit):
            continue
        owner.audit = RecentIntegrityAudit(owner.audit)


def retained_counts(roots: dict[str, Any]) -> dict[str, int]:
    """所有オブジェクトのコンテナ長と配列容量を記録する。中身を保持しない。"""
    counts: dict[str, int] = {}
    for name, owner in roots.items():
        if owner is None:
            continue
        for key, value in vars(owner).items() if hasattr(owner, '__dict__') else ():
            label = name+'.'+key
            if isinstance(value, (list, tuple, dict, set)) or type(value).__name__ == 'deque':
                counts[label] = len(value)
                if isinstance(value, RecentIntegrityAudit):
                    counts[label+'.appended'] = value.append_count
                    counts[label+'.discarded'] = value.discarded_count
                if key in ('_history', '_scores'):
                    counts[label+'.items'] = sum(len(v) for v in value)
            if hasattr(value, 'nbytes') and type(value).__module__ == 'numpy':
                counts[label+'.bytes'] = int(value.nbytes)
            if type(value).__name__ in ('DiskRows', 'DiskMap', 'DiskFIFO'):
                counts[label+'.disk_rows'] = len(value)
                counts[label+'.resident_rows'] = min(1, len(value)) if type(value).__name__ == 'DiskRows' else 0
            if type(value).__name__ == 'CounterRegistry':
                counts[label] = len(value)
        records = getattr(owner, 'records', ())
        if records:
            counts[name+'.values'] = sum(len(r.values) for r in records)
    return counts


def project_containers(objects: list[Any]) -> dict[str, int]:
    counts: Counter = Counter()
    for obj in objects:
        cls = type(obj)
        if cls.__name__ not in CONTAINER_TYPES:
            continue
        for key, value in vars(obj).items():
            if isinstance(value, (list, dict, set)):
                counts[cls.__name__+'.'+key] += len(value)
    return dict(counts.most_common(TOP_ALLOCATIONS))


class MemoryProbe:
    def __init__(self, role: str) -> None:
        directory = os.environ.get('PUYO_MEMORY_PROFILE')
        self.path = Path(directory)/(role+'.jsonl') if directory else None
        self.last = float('-inf')
        self.interval = float(os.environ.get('PUYO_MEMORY_INTERVAL', INTERVAL_SEC))
        self.gc_seconds = [0.0, 0.0, 0.0]
        self.gc_calls = [0, 0, 0]
        self.gc_started = 0.0
        self.trace = os.environ.get('PUYO_MEMORY_TRACE', '1') == '1'
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.trace:
                tracemalloc.start()
            gc.callbacks.append(self.observe_gc)

    def observe_gc(self, phase: str, info: dict[str, Any]) -> None:
        if phase == 'start':
            self.gc_started = time.perf_counter()
        else:
            generation = info['generation']
            self.gc_seconds[generation] += time.perf_counter()-self.gc_started
            self.gc_calls[generation] += 1

    def sample(self, t_sec: float, roots: dict[str, Any], force: bool = False) -> None:
        """計測の失敗を評価・認識処理へ伝播させない。"""
        try:
            self._sample(t_sec, roots, force)
        except Exception:
            logging.getLogger(__name__).exception('メモリ計測に失敗しました')

    def _sample(self, t_sec: float, roots: dict[str, Any], force: bool = False) -> None:
        if self.path is None or (not force and t_sec-self.last < self.interval):
            return
        self.last = t_sec
        objects = gc.get_objects()
        types = Counter(str(type(o).__module__)+'.'+type(o).__name__ for o in objects)
        containers = project_containers(objects)
        caches = {f'{o.__module__}.{getattr(o, "__qualname__", type(o).__name__)}': o.cache_info()._asdict()
                  for o in objects if type(o).__name__ == '_lru_cache_wrapper'}
        chain_caches = [dict(size=len(o._cache), capacity=getattr(o._cache, 'capacity', o._CACHE_MAX_SIZE),
            hits=getattr(o._cache, 'hits', None), misses=getattr(o._cache, 'misses', None),
            evictions=getattr(o._cache, 'evictions', None)) for o in objects if type(o).__name__ == 'ChainSimulator']
        del objects
        allocations = tracemalloc.take_snapshot().statistics('lineno')[:TOP_ALLOCATIONS] if self.trace else []
        current, peak = tracemalloc.get_traced_memory() if self.trace else (0, 0)
        row = dict(t_sec=t_sec, at=time.perf_counter(), pid=os.getpid(),
            traced_bytes=current, peak_bytes=peak, tracemalloc_enabled=self.trace, counts=retained_counts(roots),
            objects=dict(types), containers=containers, caches=caches, gc_seconds=self.gc_seconds,
            gc_calls=self.gc_calls, chain_caches=chain_caches, allocations=[dict(
                location=str(s.traceback), bytes=s.size, count=s.count)
                for s in allocations])
        statm = Path('/proc/self/statm')
        if statm.exists():
            row['rss_bytes'] = int(statm.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')


def evaluation_roots(local: dict[str, Any], bridge: Any) -> dict[str, Any]:
    from src.exchange_integrity_memory import integrity_memory_roots
    overlay = local.get('event_overlay')
    sink = getattr(bridge.callback, '__self__', None)
    publisher = getattr(sink, 'publisher', None)
    return dict(bridge=bridge, overlay=overlay, tracker=getattr(overlay, 'tracker', None),
        sink=sink, publisher=publisher, server=getattr(publisher, 'server', None),
        **integrity_memory_roots(overlay),
        meter=bridge.meter, **{k: local.get(k) for k in ('tracker', 'hcache', 'fctracker')
                             if k != 'tracker'})
