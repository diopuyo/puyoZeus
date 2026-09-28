"""試合をまたがない評価状態と、ディスク上の全件監査記録を分離する。"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from itertools import chain
import json
from pathlib import Path
from typing import Any
from weakref import WeakSet

from .live_spool import DiskRows, DiskMap

BRIDGE_ROWS = ('batch_starts', 'batch_sizes', 'recognition_rows', 'profile',
               'state_updates', 'calculation_rows', 'game_starts')
TIMELINE_ROWS = frozenset({'dump_rows', 'display_dump_rows', 'episode_dump_rows'})


class CounterRegistry:
    """executor所有者だけを保持し、旧試合・旧信号のMCを解放できる登録簿。"""
    def __init__(self) -> None:
        self.owner: Any = None
        self.references: WeakSet = WeakSet()

    def append(self, counter: Any) -> None:
        if self.owner is None:
            self.owner = counter
        self.references.add(counter)

    def __iter__(self) -> Any:
        if self.owner is not None:
            yield self.owner
        yield from (counter for counter in self.references if counter is not self.owner)

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> Any:
        return list(self)[index]


def spool(stack: ExitStack, directory: Path, name: str) -> DiskRows:
    rows = DiskRows(directory/(name+'.pickle'))
    stack.callback(rows.close)
    return rows


def install_logs(stack: ExitStack, bridge: Any, sink: Any, directory: Path) -> None:
    """計測列の意味と全件保存を保ったまま、RAM内の追記リストを置換する。"""
    bridge.spool_stack, bridge.spool_directory = stack, directory
    for name in BRIDGE_ROWS:
        setattr(bridge, name, spool(stack, directory, name))
    sink.rows = spool(stack, directory, 'evaluations')
    bridge.meter.rows = spool(stack, directory, 'evaluation_stages')
    publisher = sink.publisher
    for name in ('publications', 'input_events'):
        with publisher.lock:
            rows = spool(stack, directory, name)
            rows.extend(getattr(publisher, name))
            setattr(publisher, name, rows)
    with publisher.server.sent_lock:
        sent = DiskMap(directory/'sent.sqlite')
        for key, value in publisher.server.sent.items():
            sent.setdefault(key, value)
        publisher.server.sent = sent
        stack.callback(sent.close)


def bounded_overlay(base: type, directory: Path, stack: ExitStack) -> type:
    class BoundedOverlay(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.tracker = archived_tracker(type(self.tracker), directory, stack)(self.tracker.models)
    return BoundedOverlay


def archived_tracker(base: type, directory: Path, stack: ExitStack) -> type:
    class ArchivedTracker(base):
        def __init__(self, models: Any) -> None:
            super().__init__(models)
            self.archive = spool(stack, directory, 'exchanges')
            self.diagnostic_archive = spool(stack, directory, 'diagnostics')

        def boundary(self, game_idx: int, t_sec: float) -> None:
            previous = self._game_idx
            super().boundary(game_idx, t_sec)
            if previous is not None and previous != game_idx:
                self.archive.extend(self.records)
                self.diagnostic_archive.extend(self.diagnostics)
                self.records.clear()
                self._contexts.clear()
                self.diagnostics.clear()
                self._diagnostic_keys.clear()

        def save(self, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('w', encoding='utf-8') as stream:
                for record in chain(self.archive, self.records):
                    stream.write(json.dumps(asdict(record), ensure_ascii=False, allow_nan=False)+'\n')
            events = list(chain(self.diagnostic_archive, self.diagnostics))
            counts = dict(Counter(row['reason'] for row in events))
            path.with_suffix('.diagnostics.json').write_text(json.dumps(
                dict(counts=counts, events=events), ensure_ascii=False, allow_nan=False,
                indent=2), encoding='utf-8')
    return ArchivedTracker


def timeline_spool(bridge: Any, name: str) -> Any:
    if not hasattr(bridge, 'spool_directory'):
        return []
    return spool(bridge.spool_stack, bridge.spool_directory, name)


def archive_recognition_alerts(pipe: Any, rows: DiskRows) -> None:
    """認識出力に渡し終えた過去frameの診断だけを退避し、毎frameの全履歴走査を除く。"""
    for side in ('1p', '2p'):
        for name, field in (('_erasure_monitor_', 'alerts'), ('_transition_monitor_', '_alerts')):
            monitor = getattr(pipe, name+side, None)
            alerts = getattr(monitor, field, [])
            for alert in alerts:
                rows.append(dict(side=side, kind=name, alert=alert))
            alerts.clear()
