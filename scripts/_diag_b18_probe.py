"""B18診断用の実行中計装。src/を変更せず、環境変数PUYO_B18_PROBEがある時だけ
全spawn子processで (1) GC停止時間 (2) VideoFileSourceの読出/寝/消費側の1フレーム分解を記録する。
本体挙動は変えない (時計とsleepの薄いラッパーのみ)。"""
from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

MS = 1000.0
_GC = dict(total=0.0, start=0.0)
_KEEP: list = []


def _install_gc(directory: Path, pid: int) -> None:
    stream = (directory/f'gc_{pid}.jsonl').open('a', buffering=1)
    bucket: dict = {}

    def flush(key: int) -> None:
        if bucket:
            stream.write(json.dumps(dict(second=key, **bucket))+'\n')
            bucket.clear()

    def callback(phase: str, info: dict) -> None:
        if phase == 'start':
            _GC['start'] = time.perf_counter()
            return
        now = time.perf_counter()
        elapsed = now-_GC['start']
        _GC['total'] += elapsed
        key = int(now)
        if bucket.get('_key', key) != key:
            flush(bucket.pop('_key'))
        bucket['_key'] = key
        gen = str(info['generation'])
        bucket['n'+gen] = bucket.get('n'+gen, 0)+1
        bucket['ms'+gen] = bucket.get('ms'+gen, 0.0)+elapsed*MS
        bucket['max_ms'] = max(bucket.get('max_ms', 0.0), elapsed*MS)
    gc.callbacks.append(callback)


class _Proxy:
    """grab/readの所要時間だけを積算するcv2.VideoCapture代理。"""

    def __init__(self, capture, acc: dict) -> None:
        self.capture, self.acc = capture, acc

    def grab(self):
        t = time.perf_counter()
        result = self.capture.grab()
        self.acc['grab_ms'] += (time.perf_counter()-t)*MS
        self.acc['grab_n'] += 1
        return result

    def read(self):
        t = time.perf_counter()
        result = self.capture.read()
        self.acc['read_ms'] += (time.perf_counter()-t)*MS
        return result

    def __getattr__(self, name):
        return getattr(self.capture, name)


def _install_source(directory: Path, pid: int) -> None:
    from src.phase_j import live_source as ls
    original_init, original_iter = ls.VideoFileSource.__init__, ls.VideoFileSource.__iter__

    def init(self, capture, fps, start, end, stride, realtime=False, clock=time.perf_counter,
             sleep=time.sleep, fault=None, on_hold=None):
        acc = dict(grab_ms=0.0, grab_n=0, read_ms=0.0, sleep_req=0.0, sleep_act=0.0)

        def timed_sleep(seconds):
            t = clock()
            sleep(seconds)
            acc['sleep_req'] += seconds*MS
            acc['sleep_act'] += (clock()-t)*MS
        original_init(self, _Proxy(capture, acc), fps, start, end, stride, realtime, clock,
                      timed_sleep, fault=fault, on_hold=on_hold)
        self._b18 = acc

    def iterate(self):
        acc, stream = self._b18, (directory/f'source_{pid}.jsonl').open('a', buffering=1)
        previous = None
        resume = time.perf_counter()
        for frame in original_iter(self):
            arrive = time.perf_counter()
            row = dict(frame=frame.index, media=frame.media_sec, cap=frame.captured_at,
                       acq=frame.acquired_at, arrive=arrive, source_ms=(arrive-resume)*MS,
                       dropped_before=frame.dropped_before, gc_source_ms=(_GC['total']-_GC.get('mark', _GC['total']))*MS,
                       **acc)
            if previous is not None:
                row.update(previous)
            stream.write(json.dumps(row)+'\n')
            for key in acc:
                acc[key] = 0 if key == 'grab_n' else 0.0
            handed, gc_handed = time.perf_counter(), _GC['total']
            yield frame
            resume = time.perf_counter()
            _GC['mark'] = _GC['total']
            previous = dict(prev_consumer_ms=(resume-handed)*MS,
                            prev_consumer_gc_ms=(_GC['total']-gc_handed)*MS)
    ls.VideoFileSource.__init__, ls.VideoFileSource.__iter__ = init, iterate


def _install_frozen_fix(directory: Path, pid: int) -> None:
    """検証専用: exchange_event_overlay._fire_notificationsのfrozen dataclass代入(FrozenInstanceError)を
    dataclasses.replaceで回避した等価版へ実行時のみ差し替える。ディスクのsrc/は不変。適用回数を記録する。"""
    import src.exchange_event_overlay as module
    from dataclasses import replace
    from itertools import zip_longest
    calls = directory/f'frozen_fix_calls_{pid}.txt'
    original = module.ExchangeEventOverlay._fire_notifications

    def fixed(self, result, snapshot, stamp, ready):
        with calls.open('a') as stream:
            stream.write('call\n')
        for pair in zip_longest(*ready):
            saved = replace(result, p1=replace(result.p1, chain_event=pair[0]),
                            p2=replace(result.p2, chain_event=pair[1]))
            selected = [(i, event.trigger_sec) for i, event in enumerate(pair) if event is not None]
            stamps = tuple(e.trigger_sec if e is not None else None for e in pair)
            self._fire(saved, snapshot, stamp, stamps, selected)
    fixed.__wrapped__ = original
    module.ExchangeEventOverlay._fire_notifications = fixed
    (directory/f'frozen_fix_installed_{pid}.txt').write_text('installed')


def _install_no_terminal(directory: Path, pid: int) -> None:
    """検証専用(what-if): ObservedDeathDetector.update(全画面灰色化+matchTemplate x2, 約9.7ms/frame)を空実装にして、
    認識1フレーム費用からこの分を除いた時の遅延・捨て率を測る。本番挙動の候補ではない(死亡確定検出が無効)。"""
    import src.exchange_event_terminal as module
    calls = directory/f'no_terminal_calls_{pid}.txt'
    counter = dict(n=0)

    def stub(self, frame):
        counter['n'] += 1
        if counter['n'] % 1000 == 1:
            calls.write_text(str(counter['n']))
        return ()
    module.ObservedDeathDetector.update = stub
    (directory/f'no_terminal_installed_{pid}.txt').write_text('installed')


def install() -> None:
    root = os.environ.get('PUYO_B18_PROBE')
    if not root:
        return
    directory, pid = Path(root), os.getpid()
    directory.mkdir(parents=True, exist_ok=True)
    (directory/f'proc_{pid}.json').write_text(json.dumps(dict(pid=pid, ppid=os.getppid(),
        name=mp.current_process().name, started=time.perf_counter())))
    _install_gc(directory, pid)
    _install_source(directory, pid)
    if os.environ.get('PUYO_B18_NO_TERMINAL') == '1' and mp.current_process().name == 'recognition-process':
        _install_no_terminal(directory, pid)
    if os.environ.get('PUYO_B18_STACKS') == '2' and mp.current_process().name == 'event-evaluation-worker':
        import faulthandler
        import signal
        dump = (directory/f'stack_{pid}.txt').open('w', buffering=1)
        _KEEP.append(dump)
        faulthandler.register(signal.SIGUSR2, file=dump, all_threads=True, chain=False)  # 外からSIGUSR2で採取
    if os.environ.get('PUYO_B18_STACKS') == '1' and mp.current_process().name == 'event-evaluation-worker':
        import faulthandler
        dump = (directory/f'stack_{pid}.txt').open('w', buffering=1)
        _KEEP.append(dump)  # faulthandlerはfdを保持しないため参照を残す
        faulthandler.dump_traceback_later(15, repeat=True, file=dump)
    if os.environ.get('PUYO_B18_FIX_FROZEN') == '1' and mp.current_process().name == 'event-evaluation-worker':
        _install_frozen_fix(directory, pid)
