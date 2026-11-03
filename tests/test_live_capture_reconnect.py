"""カメラの再接続 (live_capture_reconnect、既定 OFF) を実機なしで検査する。

時計と睡眠は差し替え、機器は偽物の VideoCapture。OFF では従来どおり 1 回だけ開き、開き直さないことも確かめる。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.phase_j import live_capture_reconnect as rc
from src.phase_j.live_capture_buffer import BufferedCapture
from src.phase_j.live_device import DeviceConfig, DirectShowSource
from src.phase_j.live_faults import STALL_DURATION_SEC
from src.phase_j.live_source import NATIVE_SIZE

DURATION_SEC = 60.0
NEVER = 10 ** 9


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (rc.FLAG_ENV, rc.OUTPUT_ENV, 'PUYO_EVENT_PRIORITY'):
        monkeypatch.delenv(name, raising=False)


class FakeCamera:
    """opened=False は開けない機器。outage=(最初, 最後) は read の何回目から何回目までを失敗させるか (1 始まり、両端含む)。
    それ以外は毎回画素が変わる映像を返す (映像停止扱いを避ける)。"""

    def __init__(self, name: str = 'cam', opened: bool = True, outage: tuple[int, int] | None = None,
                 log: list[str] | None = None) -> None:
        self.name, self.opened, self.outage = name, opened, outage
        self.log = log if log is not None else []
        self.reads, self.released = 0, False
        self.image = np.zeros((NATIVE_SIZE[1], NATIVE_SIZE[0], 3), np.uint8)

    def isOpened(self) -> bool:
        return self.opened

    def set(self, key: int, value: float) -> bool:
        return True

    def read(self) -> tuple[bool, Any]:
        self.reads += 1
        if not self.opened or (self.outage and self.outage[0] <= self.reads <= self.outage[1]):
            return False, None
        self.image[0, 0, 0] = self.reads % 251
        return True, self.image.copy()

    def release(self) -> None:
        self.released = True
        self.log.append(f'release {self.name}')


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class Run:
    """偽の時計と機器で DirectShowSource を回した結果。calls は VideoCapture を作った回数 (= open の回数)。"""

    def __init__(self, cameras: list[FakeCamera], duration: float = DURATION_SEC, frames_wanted: int = 3) -> None:
        self.clock, self.holds, self.frames, self.log = Clock(), [], [], cameras[0].log
        self.calls = 0

        def factory(index: int, backend: int) -> FakeCamera:
            camera = cameras[min(self.calls, len(cameras) - 1)]
            self.calls += 1
            self.log.append(f'open {camera.name}')
            return camera

        source = DirectShowSource(DeviceConfig('OBS', 3), duration, lambda: self.holds.append(self.clock.now),
                                  factory, lambda image: True, self.clock, self.clock.sleep)
        iterator = iter(source)
        for frame in iterator:
            self.frames.append(frame)
            if len(self.frames) >= frames_wanted:
                break
        iterator.close()     # finally (機器の解放) を走らせる


def read_records(directory: Path) -> list[dict]:
    path = directory / rc.LOG_FILE
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


# ---------------------------------------------------------------- 定数・設定

def test_thresholds_are_derived_from_the_stall_reference() -> None:
    assert rc.STALL_REFERENCE_SEC == STALL_DURATION_SEC == 5.0          # 複製の一致を固定
    assert rc.RECONNECT_AFTER_FAIL_SEC == 2 * STALL_DURATION_SEC > STALL_DURATION_SEC
    assert 0 < rc.OPEN_RETRY_INTERVAL_SEC <= rc.RECONNECT_AFTER_FAIL_SEC


def test_default_is_off_and_no_reconnector(monkeypatch: pytest.MonkeyPatch) -> None:
    assert rc.LIVE_CAPTURE_RECONNECT_DEFAULT is False and rc.enabled() is False
    assert rc.make_reconnector() is None
    monkeypatch.setenv(rc.FLAG_ENV, '1')
    assert isinstance(rc.make_reconnector(), rc.CaptureReconnector)


def test_enabled_flag_leaves_a_record_with_the_thresholds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(rc.FLAG_ENV, '1')
    monkeypatch.setenv(rc.OUTPUT_ENV, str(tmp_path))
    rc.make_reconnector()
    (row,) = read_records(tmp_path)
    assert row['event'] == rc.EVENT_ENABLED
    assert (row['reconnect_after_fail_sec'], row['open_retry_interval_sec']) == (10.0, 5.0)
    monkeypatch.setenv(rc.FLAG_ENV, '0')
    rc.make_reconnector()
    assert len(read_records(tmp_path)) == 1                     # OFF は何も書かない


def test_configure_output_never_leaves_a_stale_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(rc.OUTPUT_ENV, 'stale')
    rc.configure_output(False, tmp_path)
    assert rc.OUTPUT_ENV not in os.environ
    rc.configure_output(True, None)
    assert rc.OUTPUT_ENV not in os.environ
    rc.configure_output(True, tmp_path)
    assert Path(os.environ[rc.OUTPUT_ENV]) == tmp_path.resolve()


def test_record_write_failure_does_not_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(rc.OUTPUT_ENV, str(tmp_path / 'absent_dir'))
    rc.write_record(dict(event='x'))        # 保存先が無くても入力ループを止めない


# ---------------------------------------------------------------- 時機の決定 (CaptureReconnector 単体)

def make_policy() -> tuple[rc.CaptureReconnector, list[dict], list[str]]:
    rows: list[dict] = []
    lines: list[str] = []
    return rc.CaptureReconnector(rows.append, lines.append, lambda: 1_700_000_000.0), rows, lines


def test_open_failure_retries_at_fixed_interval_and_reports_once() -> None:
    policy, rows, lines = make_policy()
    assert [policy.due(False, False, t) for t in (0.0, 4.9)] == [False, False]
    assert policy.due(False, False, rc.OPEN_RETRY_INTERVAL_SEC)
    policy.record_attempt(False, rc.OPEN_RETRY_INTERVAL_SEC)
    assert not policy.due(False, False, rc.OPEN_RETRY_INTERVAL_SEC + 4.9)    # 試した直後は待つ (嵐にしない)
    assert policy.due(False, False, 2 * rc.OPEN_RETRY_INTERVAL_SEC)
    assert [row['event'] for row in rows] == [rc.EVENT_OPEN_FAILED, rc.EVENT_ATTEMPT]
    assert '開けません' in lines[0] and '5 秒おき' in lines[0]


def test_read_failure_reconnects_only_after_threshold_and_not_in_a_storm() -> None:
    policy, rows, _ = make_policy()
    assert not policy.due(False, True, 100.0)
    assert not policy.due(False, True, 100.0 + rc.RECONNECT_AFTER_FAIL_SEC - 0.1)
    assert policy.due(False, True, 100.0 + rc.RECONNECT_AFTER_FAIL_SEC)
    policy.record_attempt(True, 110.0)
    assert not policy.due(False, True, 119.9)       # 作り直した後もまた閾値分待つ
    assert policy.due(False, True, 120.0)
    assert [row['event'] for row in rows] == [rc.EVENT_ATTEMPT]      # 開けている失敗の始まりは記録しない


def test_short_blip_recovers_without_reconnect_or_record() -> None:
    policy, rows, lines = make_policy()
    assert not policy.due(False, True, 1.0)
    assert not policy.due(True, True, 1.0 + STALL_DURATION_SEC)      # 5 秒の映像停止は自力復帰 (作り直さない)
    assert rows == [] and lines == [] and policy.attempts == 0 and policy.recovered == 0


def test_recovery_after_reconnect_records_count_time_and_failing_seconds() -> None:
    policy, rows, lines = make_policy()
    policy.due(False, True, 10.0)
    assert policy.due(False, True, 20.0)
    policy.record_attempt(True, 20.0)
    assert not policy.due(True, True, 21.5)
    assert (policy.attempts, policy.recovered) == (1, 1)
    last = rows[-1]
    assert last['event'] == rc.EVENT_RECOVERED and last['attempts'] == 1 and last['recovered'] == 1
    assert last['failing_sec'] == pytest.approx(11.5) and last['at_mono'] == 21.5
    assert last['at_wall'] == 1_700_000_000.0 and last['at_local'].startswith('2023-11-1')
    assert '戻りました' in lines[-1] and '1 回' in lines[-1]
    assert policy.failing_since is None and policy.last_attempt is None   # 次の失敗は新しい出来事として数える


# ---------------------------------------------------------------- DirectShowSource 経由

def test_off_opens_once_and_never_reopens_when_the_camera_cannot_open() -> None:
    cameras = [FakeCamera('a', opened=False), FakeCamera('b')]
    run = Run(cameras, duration=30.0)
    assert run.frames == [] and len(run.holds) > 100        # 従来どおり保留のまま (欠けとして固定)
    assert run.calls == 1 and not cameras[1].released


def test_off_is_identical_for_a_healthy_camera() -> None:
    run = Run([FakeCamera()], frames_wanted=5)
    assert len(run.frames) == 5 and run.holds == [] and run.calls == 1


def test_on_retries_open_until_the_camera_appears(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(rc.FLAG_ENV, '1')
    monkeypatch.setenv(rc.OUTPUT_ENV, str(tmp_path))
    log: list[str] = []
    cameras = [FakeCamera('a', opened=False, log=log), FakeCamera('b', opened=False, log=log),
               FakeCamera('c', log=log)]
    run = Run(cameras)
    assert len(run.frames) == 3 and run.calls == 3
    # 作り直す前の機器は、次を開く前に必ず解放した
    assert run.log[:5] == ['open a', 'release a', 'open b', 'release b', 'open c']
    assert 2 * rc.OPEN_RETRY_INTERVAL_SEC <= run.frames[0].captured_at < 2 * rc.OPEN_RETRY_INTERVAL_SEC + 1
    rows = read_records(tmp_path)
    assert [row['event'] for row in rows] == [rc.EVENT_ENABLED, rc.EVENT_OPEN_FAILED, rc.EVENT_ATTEMPT,
                                              rc.EVENT_ATTEMPT, rc.EVENT_RECOVERED]
    assert rows[-1]['attempts'] == 2 and rows[-1]['recovered'] == 1


def test_on_reconnects_after_read_stops_midstream(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(rc.FLAG_ENV, '1')
    monkeypatch.setenv(rc.OUTPUT_ENV, str(tmp_path))
    log: list[str] = []
    broken = FakeCamera('broken', outage=(4, NEVER), log=log)        # 3 コマ読めた後、二度と読めない
    run = Run([broken, FakeCamera('fresh', log=log)], frames_wanted=8)
    assert len(run.frames) == 8 and run.calls == 2
    assert run.log[:3] == ['open broken', 'release broken', 'open fresh']
    gap = run.frames[3].captured_at - run.frames[2].captured_at
    assert gap >= rc.RECONNECT_AFTER_FAIL_SEC                # 閾値より早くは作り直していない
    rows = read_records(tmp_path)
    assert [row['event'] for row in rows] == [rc.EVENT_ENABLED, rc.EVENT_ATTEMPT, rc.EVENT_RECOVERED]
    assert rows[1]['attempts'] == 1 and rows[2]['failing_sec'] >= rc.RECONNECT_AFTER_FAIL_SEC


def test_on_does_not_reopen_for_a_stall_shorter_than_the_stall_reference(monkeypatch: pytest.MonkeyPatch,
                                                                         tmp_path: Path) -> None:
    monkeypatch.setenv(rc.FLAG_ENV, '1')
    monkeypatch.setenv(rc.OUTPUT_ENV, str(tmp_path))
    stalled = FakeCamera('stalled', outage=(4, 83))          # 保留の 0.05 秒 x 80 回 = 4 秒の映像停止
    run = Run([stalled, FakeCamera('unused')], frames_wanted=6)
    assert run.calls == 1 and not any('unused' in line for line in run.log)
    # 有効になったことだけ記録され、作り直していない自力復帰は記録しない
    assert [row['event'] for row in read_records(tmp_path)] == [rc.EVENT_ENABLED]


def test_on_does_not_reopen_a_healthy_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(rc.FLAG_ENV, '1')
    run = Run([FakeCamera(), FakeCamera('unused')], frames_wanted=30)
    assert run.calls == 1 and run.holds == []


def test_on_with_buffered_capture_reopens_a_camera_that_could_not_open(monkeypatch: pytest.MonkeyPatch,
                                                                      tmp_path: Path) -> None:
    """event_priority 経路 (BufferedCapture の取得 thread) でも、開けなかった機器を作り直して映像を受け取れる。
    実 thread と実時計を使うため再試行間隔だけ短くする。"""
    monkeypatch.setenv(rc.FLAG_ENV, '1')
    monkeypatch.setenv(rc.OUTPUT_ENV, str(tmp_path))
    monkeypatch.setenv('PUYO_EVENT_PRIORITY', '1')
    monkeypatch.setattr(rc, 'OPEN_RETRY_INTERVAL_SEC', 0.05)
    log: list[str] = []
    cameras = [FakeCamera('a', opened=False, log=log), FakeCamera('b', log=log)]
    calls: list[int] = []

    def factory(index: int, backend: int) -> FakeCamera:
        calls.append(index)
        return cameras[min(len(calls), len(cameras)) - 1]

    iterator = iter(DirectShowSource(DeviceConfig('OBS', 3), 20.0, lambda: None, factory, lambda image: True))
    frames = []
    for frame in iterator:
        frames.append(frame)
        if len(frames) >= 2:
            break
    iterator.close()
    assert len(frames) == 2 and len(calls) == 2 and cameras[0].released
    assert [row['event'] for row in read_records(tmp_path)][:3] == [rc.EVENT_ENABLED, rc.EVENT_OPEN_FAILED,
                                                                    rc.EVENT_ATTEMPT]


def test_buffered_capture_that_failed_to_open_stays_closed_without_reconnect() -> None:
    """DirectShowSource の再接続が必要な理由の固定: BufferedCapture 自身は 1 回しか開かず、保留のまま終わる。"""
    capture = BufferedCapture(lambda: FakeCamera('a', opened=False), (), None)
    assert capture.isOpened() is False
    assert capture.thread is not None
    capture.thread.join(2.0)
    assert capture.finished and capture.read() == (False, None)
