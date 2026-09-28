"""B12の汚染境界・縮退・復帰を検証する。"""
from pathlib import Path

import pytest

from src.phase_j.live_load import cpu_interval, read_cpu


def snapshot(at: float, total: int, idle: int, owned: dict) -> dict:
    return dict(at=at, unix=at, total=total, idle=idle, owned=owned, loadavg=(1, 2, 3))


@pytest.mark.parametrize('external, contaminated', [(0, False), (100, False), (101, True), (400, True)])
def test_external_cpu_subtracts_owned_group(external: int, contaminated: bool) -> None:
    row = cpu_interval(snapshot(0, 0, 0, {'1:0': 0}),
                       snapshot(1, 1600, 1500-external, {'1:0': 100}), 100)
    assert row['own_cores'] == 1
    assert row['external_cores'] == external/100
    assert row['contaminated'] is contaminated
    assert row['valid']


@pytest.mark.parametrize('elapsed', [0, 3, 10])
def test_missing_samples_are_not_clean(elapsed: float) -> None:
    assert not cpu_interval(snapshot(0, 0, 0, {}), snapshot(elapsed, 100, 50, {}), 100)['valid']


def test_reused_pid_is_a_new_process() -> None:
    row = cpu_interval(snapshot(0, 0, 0, {'1:0': 900}),
                       snapshot(1, 400, 0, {'1:1': 50}), 100)
    assert row['own_ticks'] == 50
    assert row['external_cores'] == 3.5


def test_proc_parser_excludes_guests_and_other_group(tmp_path: Path) -> None:
    (tmp_path/'stat').write_text('cpu 100 0 20 200 10 3 2 1 90 0\n')
    for pid, group in [(1, 42), (2, 43)]:
        directory = tmp_path/str(pid)
        directory.mkdir()
        fields = ['0']*22
        fields[2], fields[11], fields[12], fields[19] = str(group), '10', '5', '99'
        (directory/'stat').write_text(f'{pid} (name with ) space) '+' '.join(fields))
    row = read_cpu(42, tmp_path)
    assert row['total'] == 336
    assert row['idle'] == 210
    assert row['owned'] == {'1:99': 15}


def test_pacer_degrades_immediately_recovers_slowly() -> None:
    from src.phase_j.live_degrade import EvaluationPacer
    pacer = EvaluationPacer(True)
    pacer.update(2, 0)
    assert pacer.level == 2
    pacer.update(0, 1)
    pacer.update(0, 5)
    assert pacer.level == 2
    pacer.update(0, 6)
    assert pacer.level == 1
    pacer.update(0, 7)
    pacer.update(0, 12)
    assert pacer.level == 0


def test_offline_pacer_ignores_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_degrade import EvaluationPacer
    monkeypatch.setenv('PUYO_ADAPTIVE_EVALUATION', '1')
    assert EvaluationPacer(False).period(1, 100) == .5


@pytest.mark.parametrize('enabled, until, expected', [(True, 100, 26), (False, 100, 30), (True, 0, 30)])
def test_event_priority_bounded_backlog(enabled: bool, until: float, expected: int) -> None:
    from src.phase_j.live_degrade import EventPriority
    priority = EventPriority()
    priority.enabled, priority.until = enabled, until
    assert priority.select(10, 30, 2, 60) == expected
    assert priority.select(30, 30, 2, 60) == 30


@pytest.mark.parametrize('start,end,label', [(0, .5, 'clean'), (.5, 1.5, 'contaminated'),
    (1, 1.9, 'contaminated'), (2, 2.5, 'unknown'), (-1, 0, 'unknown'), (4, 5, 'unknown')])
def test_contamination_labels_full_latency_interval(start: float, end: float, label: str) -> None:
    from scripts.analyze_live_b12 import Contamination
    rows = [dict(start=i, end=i+1, valid=i != 2, contaminated=i == 1) for i in range(3)]
    assert Contamination(rows).label(start, end) == label


def test_clean_denominator_excludes_unknown_and_contaminated() -> None:
    from scripts.analyze_live_b12 import aggregate
    slots = [dict(label=label, dropped=True) for label in ('clean', 'contaminated', 'unknown')]
    assert aggregate(slots, [], [], True)['expected'] == 1
    assert aggregate(slots, [], [], False)['expected'] == 3
    assert aggregate([], [], [], True)['drop_fraction'] is None


def test_affinity_does_not_modify_external_process(monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    from src.phase_j.live_cpu import isolate_cpu
    calls = []
    monkeypatch.delenv('PUYO_LIVE_RESERVED_CPU', raising=False)
    monkeypatch.setattr('src.phase_j.live_cpu.cpu_siblings', lambda cpu: {0, 1})
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {0, 1, 2, 3})
    monkeypatch.setattr('src.phase_j.live_cpu.set_process_affinity', lambda cpus: calls.append((0, cpus)))
    isolate_cpu('recognition')
    isolate_cpu('evaluation')
    assert calls == [(0, [0]), (0, [2, 3])]


def test_controlled_stress_is_still_external() -> None:
    before = snapshot(0, 0, 0, {})
    after = snapshot(1, 1600, 1200, {'1:0': 100})
    after['controlled'] = {'2:0': 300}
    row = cpu_interval(before, after, 100)
    assert row['external_cores'] == row['controlled_cores'] == 3
    assert row['contaminated']
    assert row['unplanned_cores'] == 0
    assert not row['unplanned_contaminated']


@pytest.mark.parametrize('external,lag,period', [(0, 0, .5), (1.1, 0, 1), (2.1, 0, 2),
    (0, .11, 1), (0, .26, 2), (1.1, .26, 2)])
def test_pacer_uses_both_cpu_and_queue_lag(monkeypatch: pytest.MonkeyPatch,
                                        external: float, lag: float, period: float) -> None:
    from types import SimpleNamespace
    from src.phase_j.live_degrade import EvaluationPacer
    monkeypatch.setenv('PUYO_ADAPTIVE_EVALUATION', '1')
    pacer = EvaluationPacer(True)
    pacer.sampler = SimpleNamespace(sample=lambda: dict(valid=True, external_cores=external))
    assert pacer.period(1, lag) == period
    assert pacer.period(1.1, 0) == period


@pytest.mark.parametrize('trigger', ['placement', 'chain', 'landing', 'state', 'none'])
def test_priority_uses_physical_events(monkeypatch: pytest.MonkeyPatch, trigger: str) -> None:
    from types import SimpleNamespace as NS
    from src.phase_j.live_degrade import EventPriority
    monkeypatch.setenv('PUYO_EVENT_PRIORITY', '1')
    priority = EventPriority()
    priority.counts = (1, 1)
    side = NS(state=NS(name='STABLE' if trigger != 'state' else 'CHAIN'),
              chain_event=trigger == 'chain', landing_chain_started=trigger == 'landing')
    notice = NS(t_sec=5, pipeline=NS(counts=(2, 1) if trigger == 'placement' else (1, 1)),
                result=lambda: NS(p1=side, p2=side))
    priority.observe(notice)
    assert (priority.until > 5) is (trigger != 'none')


def test_event_priority_never_replays_expired_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_source import VideoFileSource
    monkeypatch.setenv('PUYO_EVENT_PRIORITY', '1')
    source = VideoFileSource(None, 30, 0, 6, 1, True, clock=lambda: 10)
    source.event_priority.until = 100
    assert source.latest_index(1, 0) == 6


@pytest.mark.parametrize('role', ['recognition', 'evaluation', 'mc'])
def test_affinity_handles_inherited_partition(monkeypatch: pytest.MonkeyPatch, role: str) -> None:
    import os
    from src.phase_j.live_cpu import isolate_cpu
    calls = []
    monkeypatch.setenv('PUYO_LIVE_RESERVED_CPU', '0')
    monkeypatch.setattr('src.phase_j.live_cpu.cpu_siblings', lambda cpu: {cpu})
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {1, 2, 3})
    monkeypatch.setattr('src.phase_j.live_cpu.set_process_affinity', lambda cpus: calls.append(cpus))
    isolate_cpu(role)
    assert calls == ([] if role == 'recognition' else [[1, 2, 3]])


@pytest.mark.parametrize('unexpected,expected', [(False, 2), (True, 0)])
def test_placement_clean_denominators(tmp_path: Path, unexpected: bool, expected: int) -> None:
    import json
    import numpy as np
    from scripts.analyze_live_b8 import board_comparison
    from scripts.analyze_live_b12 import placement_intervals
    from tests.test_live_b8 import sample_data
    data = sample_data([0, 2, 4])
    left, right = tmp_path/'reference', tmp_path/'candidate'
    for path in (left, right):
        path.mkdir()
        np.savez(path/'recognition.npz', **data)
    row = dict(start=97, end=103, valid=True, contaminated=True, unplanned_contaminated=unexpected)
    (right/'cpu_load.jsonl').write_text(json.dumps(row)+'\n')
    result = placement_intervals(left, right, 0, 1, board_comparison(data, data))
    assert result['including']['reference_placements'] == 2
    assert result['excluding_contaminated']['reference_placements'] == 0
    assert result['excluding_unplanned_contaminated']['reference_placements'] == expected


@pytest.mark.parametrize('phase,expected', [(0, 360), (1, 359)])
def test_denominator_preserves_source_phase(phase: int, expected: int) -> None:
    from scripts.analyze_live_b12 import normalized_slots
    bounds = dict(fps=60, start=int(2579.566*60)//2*2+phase, end=int(2592.566*60), stride=2)
    slots = normalized_slots(bounds, 2580.566)
    assert len(slots) == expected
    assert all(round(t*60) % 2 == phase for t in slots)


@pytest.mark.parametrize('legacy', [False, True])
def test_report_matches_meter_frame_and_drop_counts(tmp_path: Path, legacy: bool) -> None:
    import json
    import numpy as np
    from scripts.analyze_live_b12 import report, normalized_slots
    start, end, origin = 2580.566, 2592.566, 100
    bounds = dict(fps=60, start=int((start-1)*60), end=int(end*60), stride=2)
    times = np.array(normalized_slots(bounds, start))
    np.savez(tmp_path/'recognition.npz', t_sec=times, frame=np.round(times*60).astype(int),
             captured_at=origin+times)
    metrics = dict(expected_frames=len(times), capture_dropped_in_measured_window=5)
    if not legacy:
        metrics['frame_bounds'] = bounds
    (tmp_path/'metrics.json').write_text(json.dumps(metrics))
    (tmp_path/'recognition.json').write_text(json.dumps(dict(dropped_times=times[:5].tolist())))
    rows = [dict(start=origin+start+i, end=origin+start+i+1, valid=True, contaminated=i < 6)
            for i in range(12)]
    (tmp_path/'cpu_load.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    window = report(tmp_path, start, end)['windows'][0]
    assert window['including']['expected'] == 359
    assert window['including']['dropped'] == 5
    assert window['excluding']['expected'] == 179
    assert window['excluding']['dropped'] == 0


@pytest.mark.parametrize('field', ['cpu_isolation', 'adaptive_evaluation', 'event_priority'])
def test_degradation_rejects_shared_thread_mode(monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    import argparse
    import os
    from scripts.run_live_pipeline_20260928 import configure_cpu
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    options = argparse.Namespace(cpu_threads=0, evaluation_nice=0, recognition_audit=False,
                                 worker_mode='thread', source='video', **{field: True})
    with pytest.raises(SystemExit):
        configure_cpu(options, argparse.ArgumentParser())


@pytest.mark.parametrize('value,expected', [('0-1', {0, 1}), ('0,8', {0, 8})])
def test_cpu_sibling_topology(tmp_path: Path, value: str, expected: set[int]) -> None:
    from src.phase_j.live_cpu import cpu_siblings
    directory = tmp_path/'cpu0/topology'
    directory.mkdir(parents=True)
    (directory/'thread_siblings_list').write_text(value)
    assert cpu_siblings(0, tmp_path) == expected
    assert cpu_siblings(4, tmp_path) == {4}


def test_affinity_covers_existing_threads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    from src.phase_j.live_cpu import set_process_affinity
    for tid in (12, 13):
        (tmp_path/str(tid)).mkdir()
    calls = []
    monkeypatch.setattr(os, 'sched_setaffinity', lambda pid, cpus: calls.append((pid, cpus)))
    set_process_affinity([2, 4], tmp_path)
    assert set(pid for pid, _ in calls) == {0, 12, 13}
    assert all(cpus == [2, 4] for _, cpus in calls)


@pytest.mark.parametrize('critical,expected,dropped', [(False, 4, 4), (True, 2, 2)])
def test_capture_buffer_bounded_selection(critical: bool, expected: int, dropped: int) -> None:
    from src.phase_j.live_capture_buffer import BufferedCapture, CAPACITY
    from src.phase_j.live_degrade import EventPriority
    priority = EventPriority()
    priority.enabled, priority.until = True, 10 if critical else -1
    capture = BufferedCapture(lambda: None, (), priority)
    for index in range(5):
        capture._offer(index/30, index)
        assert len(capture.frames) <= CAPACITY
    assert capture.read() == (True, expected)
    assert capture.last_captured_at == expected/30
    assert capture.dropped == capture.dropped_before == dropped
    capture.release()
    assert capture.dropped == 4 and not capture.frames


@pytest.mark.parametrize('failure', [None, 'read', 'release'])
def test_capture_owner_thread_and_failure_cleanup(failure: str | None) -> None:
    from threading import get_ident
    from types import SimpleNamespace as NS
    from src.phase_j.live_capture_buffer import BufferedCapture
    from src.phase_j.live_degrade import EventPriority
    calls = []
    capture = BufferedCapture(lambda: None, (), EventPriority())
    def read() -> tuple:
        calls.append(('read', get_ident()))
        capture.stop.set()
        if failure == 'read':
            raise ValueError('read失敗')
        return True, 'frame'
    def release() -> None:
        calls.append(('release', get_ident()))
        if failure == 'release':
            raise ValueError('release失敗')
    def factory() -> NS:
        calls.append(('create', get_ident()))
        return NS(set=lambda *args: None, isOpened=lambda: True, read=read, release=release)
    capture.factory = factory
    try:
        capture.isOpened()
    except RuntimeError:
        assert failure is not None
    assert capture.thread is not None
    capture.thread.join(2)
    if failure:
        with pytest.raises(RuntimeError):
            capture.release()
    else:
        assert capture.read() == (True, 'frame')
        capture.release()
    assert capture.finished and not capture.thread.is_alive()
    assert [name for name, _ in calls] == ['create', 'read', 'release']
    assert len({tid for _, tid in calls}) == 1 and calls[0][1] != get_ident()


def test_native_priority_keeps_verification_on_consumer(monkeypatch: pytest.MonkeyPatch) -> None:
    from threading import get_ident
    import numpy as np
    from src.phase_j.live_device import DeviceConfig, DirectShowSource
    from tests.phase_j.test_live_device import FakeDevice
    monkeypatch.setenv('PUYO_EVENT_PRIORITY', '1')
    camera = FakeDevice(np.zeros((720, 1280, 3), np.uint8))
    verification_threads = []
    def verifier(image: np.ndarray) -> bool:
        verification_threads.append(get_ident())
        return True
    source = DirectShowSource(DeviceConfig('B12-test', 0), 10, lambda: None,
                              lambda *args: camera, verifier)
    stream = iter(source)
    try:
        frame = next(stream)
        assert frame.source_size == (1280, 720)
        assert frame.image.shape == (1080, 1920, 3)
        assert frame.captured_at <= frame.acquired_at
        assert verification_threads == [get_ident()]
    finally:
        stream.close()
    assert camera.released
