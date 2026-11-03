"""監視失敗・メモリ不足・親終了の回帰検証。"""
from pathlib import Path
import os
import signal
import subprocess
import time
from unittest.mock import Mock

import pytest

from scripts import image_free_parallel as parallel


def good_monitor(monkeypatch: pytest.MonkeyPatch) -> parallel.Monitor:
    result = Mock(stdout='{"pids": [], "free_bytes": 17179869184}')
    monkeypatch.setattr(parallel.subprocess, 'run', Mock(return_value=result))
    monkeypatch.setattr(parallel, 'linux_available', lambda: 16 * parallel.GIB)
    monitor = parallel.Monitor()
    assert not monitor.poll()
    return monitor


@pytest.mark.parametrize('error', [OSError('失敗'), subprocess.CalledProcessError(1, 'powershell'),
                                  subprocess.TimeoutExpired('powershell', 15), ValueError('JSON')])
def test_transient_failure_and_recovery(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    monitor = good_monitor(monkeypatch)
    parallel.subprocess.run.side_effect = error
    assert not monitor.poll()
    assert not monitor.can_start()
    assert not monitor.poll()
    assert monitor.poll()
    parallel.subprocess.run.side_effect = None
    assert not monitor.poll()
    assert monitor.failures == 0


def test_unknown_startup_stays_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel.subprocess, 'run', Mock(side_effect=OSError('失敗')))
    assert parallel.Monitor().poll()


def test_live_stops_all_and_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = good_monitor(monkeypatch)
    parallel.subprocess.run.return_value.stdout = '{"pids": [123], "free_bytes": 17179869184}'
    assert monitor.poll()
    assert not monitor.can_start()
    parallel.subprocess.run.return_value.stdout = '{"pids": [], "free_bytes": 17179869184}'
    assert not monitor.poll()


@pytest.mark.parametrize('platform', ['windows', 'wsl'])
def test_memory_reserve(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    monitor = good_monitor(monkeypatch)
    if platform == 'wsl':
        monkeypatch.setattr(parallel, 'linux_available', lambda: 5 * parallel.GIB)
    else:
        parallel.subprocess.run.return_value.stdout = '{"pids": [], "free_bytes": 5368709120}'
    assert monitor.poll()
    assert not monitor.can_start()
    children = {'a': Mock(), 'b': Mock()}
    send, stop = Mock(), Mock()
    monkeypatch.setattr(parallel, 'signal_worker', send)
    monkeypatch.setattr(parallel, 'stop_worker', stop)
    parallel.resource_control(monitor, children, True)
    assert len(children) == 1
    assert send.call_count == 2
    assert all(call.args[1] == signal.SIGSTOP for call in send.call_args_list)
    stop.assert_called_once()


def test_parent_loss_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel.os, 'getppid', lambda: 10)
    parallel.CentralGuard(10).checkpoint()
    with pytest.raises(RuntimeError, match='監視器'):
        parallel.CentralGuard(11).checkpoint()


def test_disabled_videos(tmp_path: Path) -> None:
    assert parallel.disabled_videos(tmp_path) == set()
    parallel.save_json(tmp_path/'parallel_control.json', {'disabled_videos': ['video_d02']})
    assert parallel.disabled_videos(tmp_path) == {'video_d02'}


def test_launch_uses_shared_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = Mock(config=tmp_path/'config.json', source=tmp_path/'source', output=tmp_path)
    monkeypatch.setattr(parallel.os, 'sched_getaffinity', lambda _: {0, 1})
    start = Mock()
    monkeypatch.setattr(parallel.subprocess, 'Popen', start)
    parallel.launch(args, 'c68', 0)
    command = start.call_args.args[0]
    assert command[command.index('--config')+1] == str(args.config)
    assert command[command.index('--worker')+1] == 'c68'
    assert start.call_args.kwargs['env']['OMP_NUM_THREADS'] == '1'


def test_partial_results_never_complete(tmp_path: Path) -> None:
    (tmp_path/'c68').mkdir()
    parallel.save_json(tmp_path/'c68'/'progress.json', dict(complete=True,
        display_changed_scenes=0, display_text_changed_scenes=0))
    parallel.collect_results(tmp_path, ['c68', 'video_d02'], {'c68': 0})
    assert not parallel.read_json(tmp_path/'result.json')['complete']


@pytest.mark.skipif(os.name == 'nt', reason='WSLのプロセスグループ専用')
def test_real_process_pause_resume_stop() -> None:
    child = subprocess.Popen(['sleep', '60'], start_new_session=True)
    state = Path(f'/proc/{child.pid}/status')
    try:
        parallel.signal_worker(child, signal.SIGSTOP)
        deadline = time.monotonic() + parallel.POLL_SECONDS
        while '\nState:\tT' not in state.read_text() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert '\nState:\tT' in state.read_text()
        parallel.signal_worker(child, signal.SIGCONT)
        deadline = time.monotonic() + parallel.POLL_SECONDS
        while '\nState:\tT' in state.read_text() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert '\nState:\tT' not in state.read_text()
        parallel.stop_worker(child)
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_cpu_selection_visits_cores_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel.os, 'sched_getaffinity', lambda _: {0, 1, 2, 3})
    def read(path: Path) -> str:
        return '0' if path.name == 'physical_package_id' else str(int(path.parts[-3][3:]) // 2)
    monkeypatch.setattr(Path, 'read_text', read)
    assert parallel.worker_cpus() == [0, 2, 1, 3]
