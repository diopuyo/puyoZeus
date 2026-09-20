"""原run codeと必要3参照の対応、明示GOなしの拒否を限定検査する。"""
from __future__ import annotations
from contextlib import ExitStack
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import FunctionType
import pytest
import target_entry as T


def test_actual_configured_run_binding() -> None:
    with ExitStack() as stack:
        main = T.A.configured(stack)
        original = main.__globals__['run_live']
        run = T.selected_run(main)
        assert run.__code__ is original.__code__
        assert run.__defaults__ is original.__defaults__ and run.__kwdefaults__ is original.__kwdefaults__
        for key in ('collect', 'S', 'K'):
            assert run.__globals__[key] is main.__globals__[key]
        for key, value in original.__globals__.items():
            if key not in ('collect', 'S', 'K'):
                assert run.__globals__[key] is value
        collected = run.__globals__['collect']
        guard = collected.__globals__['constructor_guard']
        assert 'original_guard' in inspect.getclosurevars(guard).nonlocals
        assert run.__globals__['K'].FRAMES[-1] == 36298


def test_selected_run_preserves_original_exception() -> None:
    namespace = dict(collect=lambda: (_ for _ in ()).throw(LookupError('original_collect')), S=None, K=None)
    exec('def run_live():\n return collect()\ndef main():\n pass\n', namespace)
    run = T.selected_run(namespace['main'])
    with pytest.raises(LookupError, match='original_collect'):
        run()


def test_missing_go_rejects_without_creating_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(T, 'GO', tmp_path / 'missing.json')
    output = tmp_path / 'not-created'
    with pytest.raises(FileNotFoundError):
        T.approved(output)
    assert not output.exists()


def test_probability_runtime_sources_are_pinned() -> None:
    names = {p.name for p in T.runtime_sources()}
    assert {'live_adapter.py', 'loader.py', 'context.py', 'probability_owner.py',
            'probability_finish.py', 'start_capture.py', 'target_entry.py', 'run_whole_target.sh'} <= names
    # 所有入口の検査scriptは除外。依存root一括の保護集合は実行一覧ではない。
    assert all(not p.name.startswith(('test_', 'probe_')) for p in T.runtime_sources()
               if p.parent in (T.ROOT, T.A.PRIOR))


def test_optimized_python_cannot_bypass_go(tmp_path: Path) -> None:
    go = tmp_path / 'go.json'
    go.write_text(json.dumps(dict(decision='NO_GO', frozen_files={})))
    code = ('import sys; sys.path.insert(0,sys.argv[1]); import target_entry as T; '
            'from pathlib import Path; T.GO=Path(sys.argv[2]); T.approved(Path(sys.argv[3]))')
    result = subprocess.run([sys.executable, '-O', '-c', code, str(T.ROOT), str(go), str(tmp_path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and 'target_not_approved' in result.stderr


def test_finalize_saves_wait_even_if_raw_import_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(T, 'ROOT', tmp_path / 'runner')
    output = tmp_path / T.OUTPUT_NAME
    output.mkdir()
    def fail(name: str) -> None:
        raise ImportError('original_finalizer_import')
    monkeypatch.setattr(T.importlib, 'import_module', fail)
    with pytest.raises(ImportError, match='original_finalizer_import'):
        T.finalize(output, 0, 0)
    saved = json.loads((output / 'TARGET_PARENT_WAIT.json').read_bytes())
    assert saved['child_exit_code'] == saved['resource_guard_exit'] == 0
    assert not (output / 'COMPLETE').exists()


@pytest.mark.parametrize('child_exit', [0, 1])
def test_raw_finalizer_without_generation_or_go(monkeypatch: pytest.MonkeyPatch,
                                               tmp_path: Path, child_exit: int) -> None:
    # 終了票だけの人工fixture。実動画/盤面/G2品質の合格には使わない。
    monkeypatch.setattr(T, 'ROOT', tmp_path / 'runner')
    monkeypatch.setattr(T, 'GO', tmp_path / 'missing-go.json')
    monkeypatch.syspath_prepend(str(T.FINALIZER_ROOT))
    closure = T.importlib.import_module('closure')
    common = closure.K
    output = tmp_path / T.OUTPUT_NAME
    output.mkdir()
    def forbidden(*args: object) -> None:
        raise AssertionError('generation_must_not_be_configured')
    monkeypatch.setattr(T.A, 'configured', forbidden)
    if child_exit == 0:
        common.write(output / 'ENTRY_RESULT.json', dict(pid=os.getpid(), exit_code=0, artificial=True))
        engine = dict(computation_closed=True, bounds=common.bounds(),
            summary=dict(references_restored=True, guards_unchanged=True,
                         goal=dict(current_event_observed=False, outer_publication_observed=False)),
            sha256={'ENTRY_RESULT.json': common.sha(output / 'ENTRY_RESULT.json')},
            guard_sha256={str(Path(T.__file__)): common.sha(Path(T.__file__))}, **closure.PERMISSIONS)
        common.write(output / closure.ENGINE, engine)
        resource = dict(pid=os.getpid(), safety_stop=False, rss_kib=1000, available_kib=4000000,
                        rss_limit_kib=8388608, minimum_available_kib=2097152)
        Path(str(output) + '.resources.jsonl').write_text(json.dumps(resource) + '\n')
        supervisor = dict(source='actual_Popen_wait', child_pid=os.getpid(), child_exit_code=0,
            resource_guard_exit=0, supervisor_error=None, guard_exited_first=False, artificial=True)
        Path(str(output) + '.supervisor.json').write_text(json.dumps(supervisor))
    result = T.finalize(output, child_exit, 0)
    assert (output / 'TARGET_PARENT_WAIT.json').exists() and (output / 'CHILD_EXIT.json').exists()
    assert result['quality_gate_clear'] is False
    assert (output / 'COMPLETE').exists() == (child_exit == 0)

def test_new_runtime_and_review_sources() -> None:
    names = {p.name for p in T.runtime_sources()}
    assert {'driver_status.py', 'thermal_child.py', 'review_m1.py', 'qualification_audit.py',
            'qualification_saved.py', 'second_pending_replay.py', 'schedule_saved.py'} <= names
    assert T.THREADS == 2
    assert T.OUTPUT_NAME == 'video38_m1_second_candidate_v21'
