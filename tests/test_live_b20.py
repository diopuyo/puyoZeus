"""B20: 死亡検出の低解像度予備判定 (fast_terminal) と非同期通知送信口 (async_notice_queue)。

どちらも既定OFF。ONでも決定・通知内容が変わらないことを単体で確かめる。
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from queue import Queue
from threading import Event, Thread
import time

import cv2
import numpy as np
import pytest


# 全体実行時の src import 順序汚染 (test_g3_* が src 未import前提) を避けるため、src は遅延import。
FRAME_SHAPE = (1080, 1920, 3)
SEED = 20
POSITIVE_ROUNDS = 6


def _mods():
    from src import exchange_event_terminal as terminal
    from src.match_end_detector import SEARCH_P1, SEARCH_P2
    from src.phase_j import live_notice_sender as sender
    return terminal, SEARCH_P1, SEARCH_P2, sender


def _frame(rng: np.random.Generator, side: int | None) -> np.ndarray:
    """ノイズ画面。side を指定するとその探索領域へばたんきゅーを貼る。"""
    terminal, SEARCH_P1, SEARCH_P2, _ = _mods()
    image = rng.integers(0, 255, FRAME_SHAPE, dtype=np.uint8)
    if side is not None:
        template = cv2.imread(str(terminal.TEMPLATE), cv2.IMREAD_GRAYSCALE)
        x, y, *_ = (SEARCH_P1, SEARCH_P2)[side]
        height, width = template.shape
        image[y+20:y+20+height, x+30:x+30+width] = template[:, :, None]
    return image


def _sequence() -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    pattern = [None, None, 0, 0, 0, None, 1, None, 1, 1, None, 0, 1]
    return [_frame(rng, side) for side in pattern]


def test_fast_terminal_is_decision_identical_on_positive_and_negative_frames() -> None:
    terminal, SEARCH_P1, SEARCH_P2, sender = _mods()
    ObservedDeathDetector = terminal.ObservedDeathDetector
    slow, fast = ObservedDeathDetector(fast=False), ObservedDeathDetector(fast=True)
    frames = _sequence()
    decisions = [(slow.update(f), fast.update(f)) for f in frames]
    assert all(a == b for a, b in decisions)
    assert any(a for a, _ in decisions)          # 陽性フレームが実際に含まれる (母数の健全性)
    assert fast.frames == len(frames)
    assert sum(fast.full_matches) < 2 * len(frames)  # 予備判定で全解像度照合を省いた


def test_fast_terminal_default_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal, SEARCH_P1, SEARCH_P2, sender = _mods()
    ObservedDeathDetector = terminal.ObservedDeathDetector
    monkeypatch.delenv(terminal.FAST_ENV, raising=False)
    assert ObservedDeathDetector().fast is False
    monkeypatch.setenv(terminal.FAST_ENV, '1')
    assert ObservedDeathDetector().fast is True
    assert ObservedDeathDetector(fast=False).fast is False


def test_off_path_scores_equal_original_formula() -> None:
    """OFF経路は従来式(全画面灰色化→ROI→matchTemplate)とスコアが完全一致する。"""
    terminal, SEARCH_P1, SEARCH_P2, sender = _mods()
    ObservedDeathDetector = terminal.ObservedDeathDetector
    detector = ObservedDeathDetector(fast=False)
    frame = _sequence()[3]
    detector.update(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for idx, (x, y, w, h) in enumerate((SEARCH_P1, SEARCH_P2)):
        expected = float(cv2.minMaxLoc(cv2.matchTemplate(
            gray[y:y+h, x:x+w], detector.template, cv2.TM_CCOEFF_NORMED))[1])
        assert detector.scores[idx] == expected


def test_notice_sender_preserves_order_and_never_blocks_producer() -> None:
    """消費側が止まっていても、局所容量までは put が即時に返り、順序と件数は不変。"""
    terminal, SEARCH_P1, SEARCH_P2, sender = _mods()
    ObservedDeathDetector = terminal.ObservedDeathDetector
    queue: Queue = Queue(maxsize=2)
    channel = sender.AsyncNoticeSender(queue, Event(), capacity=50)
    started = time.perf_counter()
    for index in range(40):
        channel.put(('notice', index))
    assert time.perf_counter()-started < 0.5    # 消費者ゼロでも詰まらない
    received: list = []
    consumer = Thread(target=lambda: [received.append(queue.get()) for _ in range(40)], daemon=True)
    time.sleep(2*sender.STALL_SEC)               # 評価側の詰まりを模擬
    consumer.start()
    channel.close()
    consumer.join(5)
    assert received == [('notice', i) for i in range(40)]
    stats = channel.stats()
    assert stats['sent'] == 40 and stats['dropped_on_cancel'] == 0
    assert stats['backlog_max'] >= 30 and stats['sender_stall_count'] >= 1


def test_notice_sender_backpressure_when_local_buffer_full() -> None:
    terminal, SEARCH_P1, SEARCH_P2, sender = _mods()
    ObservedDeathDetector = terminal.ObservedDeathDetector
    queue: Queue = Queue(maxsize=1)
    channel = sender.AsyncNoticeSender(queue, Event(), capacity=3)
    done = Event()

    def produce() -> None:
        for index in range(10):
            channel.put(index)
        done.set()

    Thread(target=produce, daemon=True).start()
    time.sleep(0.4)
    assert not done.is_set()                     # 局所も満杯なら従来どおり背圧
    got = [queue.get(timeout=2) for _ in range(10)]
    assert done.wait(2) and got == list(range(10))
    channel.close()
    assert channel.stats()['producer_blocked_sec'] > 0


def test_notice_sender_cancel_drops_rest_without_hanging() -> None:
    terminal, SEARCH_P1, SEARCH_P2, sender = _mods()
    ObservedDeathDetector = terminal.ObservedDeathDetector
    queue: Queue = Queue(maxsize=1)
    cancel = Event()
    channel = sender.AsyncNoticeSender(queue, cancel, capacity=20)
    for index in range(10):
        channel.put(index)
    cancel.set()
    started = time.perf_counter()
    channel.close()
    assert time.perf_counter()-started < 2
    assert channel.stats()['dropped_on_cancel'] >= 1


def test_notice_sender_works_with_multiprocessing_queue() -> None:
    terminal, SEARCH_P1, SEARCH_P2, sender = _mods()
    ObservedDeathDetector = terminal.ObservedDeathDetector
    context = mp.get_context('spawn')
    queue = context.Queue(maxsize=2)
    channel = sender.AsyncNoticeSender(queue, context.Event(), capacity=20)
    for index in range(15):
        channel.put(('notice', index))
    got = [queue.get(timeout=5) for _ in range(15)]
    channel.close()
    assert got == [('notice', i) for i in range(15)]


def test_wrap_queue_default_off_returns_same_object(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal, SEARCH_P1, SEARCH_P2, sender = _mods()
    ObservedDeathDetector = terminal.ObservedDeathDetector
    queue: Queue = Queue()
    monkeypatch.delenv(sender.ENV_FLAG, raising=False)
    assert sender.wrap_queue(queue, Event()) is queue
    assert sender.queue_stats(queue) is None
    sender.close_queue(queue)                    # 素のqueueには何もしない
    monkeypatch.setenv(sender.ENV_FLAG, '1')
    wrapped = sender.wrap_queue(queue, Event())
    assert isinstance(wrapped, sender.AsyncNoticeSender)
    sender.close_queue(wrapped)


@pytest.mark.parametrize('field', ['async_notice_queue', 'fast_terminal'])
def test_cli_flags_default_off_and_propagate_to_env(monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    terminal, SEARCH_P1, SEARCH_P2, sender = _mods()
    ObservedDeathDetector = terminal.ObservedDeathDetector
    from scripts.run_live_pipeline_20260928 import configure_cpu, add_fault_arguments
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    parser = argparse.ArgumentParser()
    add_fault_arguments(parser)
    off = parser.parse_args([])
    assert getattr(off, field) is False
    on = parser.parse_args(['--' + field.replace('_', '-')])
    options = argparse.Namespace(cpu_threads=0, evaluation_nice=0, recognition_audit=False,
                                 worker_mode='process', source='video', **vars(on))
    configure_cpu(options, parser)
    assert os.environ['PUYO_' + field.upper()] == '1'


def test_async_flag_rejected_in_thread_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal, SEARCH_P1, SEARCH_P2, sender = _mods()
    ObservedDeathDetector = terminal.ObservedDeathDetector
    from scripts.run_live_pipeline_20260928 import configure_cpu
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    options = argparse.Namespace(cpu_threads=0, evaluation_nice=0, recognition_audit=False,
                                 worker_mode='thread', source='video', async_notice_queue=True)
    with pytest.raises(SystemExit):
        configure_cpu(options, argparse.ArgumentParser())


def test_local_peak_is_lower_bound_and_fallback_keeps_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    """近傍照合は全域最大の下界。粗い位置が外れても全域照合へ落ちて決定は同一。"""
    terminal, SEARCH_P1, SEARCH_P2, sender = _mods()
    frame = _sequence()[2]                      # P1 探索領域にばたんきゅーを貼ったフレーム
    x, y, w, h = SEARCH_P1
    roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    template = cv2.imread(str(terminal.TEMPLATE), cv2.IMREAD_GRAYSCALE)
    full = terminal._peak(roi, template)
    coarse, where = terminal._peak_at(terminal._shrink(roi), terminal._shrink(template))
    local = terminal._local_peak(roi, template, where)
    assert coarse >= terminal.PREFILTER_FLOOR and 0.55 <= local <= full + 1e-4
    assert terminal._local_peak(roi, template, (0, 0)) < 0.55   # 位置が外れれば下界は低い
    monkeypatch.setattr(terminal, '_peak_at', lambda r, t: (0.9, (0, 0)))  # 常に外れた位置を返す
    slow, fast = terminal.ObservedDeathDetector(fast=False), terminal.ObservedDeathDetector(fast=True)
    frames = _sequence()
    assert [slow.update(f) for f in frames] == [fast.update(f) for f in frames]
    assert sum(fast.local_hits) == 0            # 近傍では確定せず、全域照合が決定を担った
