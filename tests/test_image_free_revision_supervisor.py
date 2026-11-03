"""改版時に旧プロセスへ触れず、新だけを0から取り直す。"""
from pathlib import Path
import json
import pytest
from scripts.image_free_revision_supervisor import completed, replace_new, revision_routes


def test_replace_only_new_preserves_old_process_and_completed_code() -> None:
    old, new = object(), object()
    routes = {'old': {'job': {'route': 'old'}}, 'new': {'job': {'route': 'new'}}}
    children = {'old': old, 'new': new}
    codes = {'old_done': 0, 'new': 0}
    stopped = []
    replacement = {'job': {'route': 'new'}, 'output': 'revision2'}
    replace_new(routes, {'new': replacement}, children, codes, stopped.append)
    assert children == {'old': old} and stopped == [new]
    assert codes == {'old_done': 0} and routes['new'] is replacement


def test_old_replacement_is_forbidden() -> None:
    with pytest.raises(ValueError, match='旧経路'):
        replace_new({'old': {'job': {'route': 'old'}}}, {'old': {}}, {}, {}, lambda _: None)


def test_revision_must_start_from_zero(tmp_path: Path) -> None:
    source = tmp_path/'source'
    source.mkdir()
    initial = {'output': str(tmp_path/'old'), 'videos': [
        {'id': 'v_new', 'route': 'new', 'path': 'v.mp4', 'end_frame': 100}]}
    config = {'output': str(tmp_path/'new'), 'videos': [dict(initial['videos'][0], start_frame=5)]}
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    record = dict(config=str(path), source=str(source), source_manifest={}, model_sha256={}, r2_exit=0, commit='abc')
    with pytest.raises(ValueError, match='0から'):
        revision_routes(record, initial)
    config['videos'][0]['start_frame'] = 0
    path.write_text(json.dumps(config))
    assert set(revision_routes(record, initial)) == {'v_new'}


def test_completed_route_requires_matching_final_checkpoint(tmp_path: Path) -> None:
    route = dict(output=str(tmp_path), job=dict(id='v', end_frame=300))
    directory = tmp_path/'v'
    directory.mkdir()
    (directory/'progress.json').write_text(json.dumps(dict(complete=True, frames=300)))
    assert not completed(route)
    (directory/'checkpoint.json').write_text(json.dumps(dict(frame=299)))
    assert not completed(route)
    (directory/'checkpoint.json').write_text(json.dumps(dict(frame=300)))
    assert completed(route)


def test_pending_affinity_is_reserved_and_guard_moves_are_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import image_free_revision_supervisor as supervisor
    for pid in (101, 102):
        directory = tmp_path/str(pid)
        directory.mkdir()
        (directory/'cmdline').write_bytes(b'\0'.join([b'python', b'--worker', b'v', b'--config', b'c', b'--cpu', b'4']))
    monkeypatch.setattr(supervisor.os, 'sched_getaffinity', lambda pid: {0, 1, 2, 3, 4} if pid == 101 else {7})
    assert supervisor.reserved_cpus(tmp_path) == {4, 7}


def test_consecutive_launches_assign_distinct_cpus_in_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import image_free_revision_supervisor as supervisor
    from scripts import image_free_parallel as parallel, image_free_resources as resources
    from types import SimpleNamespace
    used, assigned = set(), []
    monkeypatch.setattr(supervisor, 'CPU_LAUNCH_LOCK', tmp_path/'lock')
    monkeypatch.setattr(supervisor, 'reserved_cpus', lambda: set(used))
    monkeypatch.setattr(parallel, 'worker_cpus', lambda: [0, 1])
    monkeypatch.setattr(resources, 'unoccupied_cpus', lambda cpus: cpus)
    monkeypatch.setattr(supervisor.subprocess, 'Popen', lambda *args, **kwargs: SimpleNamespace(pid=100+len(assigned)))
    def assign(pid: int, cpus: set[int]) -> None:
        used.update(cpus)
        assigned.append((pid, cpus))
    monkeypatch.setattr(supervisor.os, 'sched_setaffinity', assign)
    route = dict(job=dict(id='v'), source=str(tmp_path), output=str(tmp_path), config='config')
    assert supervisor.launch_route(route).pid == 100
    assert supervisor.launch_route(route).pid == 101
    assert supervisor.launch_route(route) is None
    assert assigned == [(100, {0}), (101, {1})]


@pytest.mark.parametrize('error', [FileNotFoundError('置換中'), json.JSONDecodeError('途中', '', 0)])
def test_progress_guard_skips_race_and_records_nonzero_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    from scripts import image_free_revision_supervisor as supervisor, image_free_resources as resources
    calls = []
    def original(self: object, children: dict, output: Path) -> None:
        calls.extend(children)
        if 'a' in children:
            raise error
    monkeypatch.setattr(resources.ResourceBudget, 'observe', original)
    monkeypatch.setattr(supervisor, 'GUARD_COUNTS', {})
    supervisor.install_progress_guard(tmp_path)
    resources.ResourceBudget().observe({'a': object(), 'b': object()}, tmp_path)
    receipt = json.loads((tmp_path/'supervisor_guard.json').read_text())
    assert calls == ['a', 'b']
    assert receipt['counts'] == {'observe_replacements': 1, 'progress_read_skipped': 1}


def test_tick_exception_preserves_child_then_next_tick_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import image_free_revision_supervisor as supervisor
    monkeypatch.setattr(supervisor, 'GUARD_COUNTS', {})
    children, calls = {'still_running': object()}, []
    def failed() -> None:
        raise OSError('監督側の一時例外')
    assert not supervisor.guarded_tick(tmp_path, failed)
    assert supervisor.guarded_tick(tmp_path, lambda: calls.append('next'))
    assert len(children) == 1 and calls == ['next']
    assert supervisor.GUARD_COUNTS['tick_exceptions'] == 1
    with pytest.raises(SystemExit):
        supervisor.guarded_tick(tmp_path, lambda: (_ for _ in ()).throw(SystemExit(0)))


def test_main_keeps_workers_after_tick_error_until_explicit_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import image_free_revision_supervisor as supervisor, image_free_parallel as parallel
    events = []
    config = tmp_path/'config.json'
    gate = tmp_path/'gate.json'
    config.write_text(json.dumps(dict(output=str(tmp_path), host='pc1', videos=[])))
    gate.write_text(json.dumps(dict(source_manifest={})))
    (tmp_path/'route_plan.json').write_text(json.dumps(dict(worker_limits={'pc1': 4})))
    monkeypatch.setattr(supervisor.sys, 'argv', ['guard', '--config', str(config), '--source', str(tmp_path),
        '--gate', str(gate), '--execute', supervisor.AUTHORIZATION])
    monkeypatch.setattr(supervisor, 'verify_initial', lambda *args: None)
    monkeypatch.setattr(supervisor, 'install_progress_guard', lambda *args: None)
    monkeypatch.setattr(supervisor, 'GUARD_COUNTS', {})
    monkeypatch.setattr(supervisor.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(supervisor.time, 'sleep', lambda *args: None)
    monkeypatch.setattr(parallel, 'Monitor', object)
    monkeypatch.setattr(parallel, 'stop_worker', lambda child: events.append('explicit_stop'))
    def tick(root: Path, initial: dict, state: dict, *args: object) -> None:
        if not events:
            state['children']['running'] = object()
            events.append('read_error')
            raise FileNotFoundError('progress置換中')
        assert list(state['children']) == ['running']
        assert events == ['read_error']
        events.append('next_tick')
        raise SystemExit(0)
    monkeypatch.setattr(supervisor, 'tick', tick)
    with pytest.raises(SystemExit):
        supervisor.main()
    assert events == ['read_error', 'next_tick', 'explicit_stop']
