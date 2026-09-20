"""新合成のcold依存・全scope復元・観測例外の伝播を検収する。"""
from __future__ import annotations
import ast
import contextlib
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
import pytest
import live_cli as L
import resource_guard as Q


def test_cold_combined_all_restored() -> None:
    code = '''
import live_cli as L,sys
p,d,a,e=L.bootstrap()
d.validate_addon(a)
assert not any(n=='src' or n.startswith('src.') for n in sys.modules)
with L.configuration(p,d):
 r,f=d.load_runtime()
 before=(r.run,r.write,r.finish,r.finalize,r.REQUIRED,f.OLD.F.validate_comparison,f.OLD.F.diagnostic_reports)
 with L.configured(r,f,d,a,e):
  assert L.M.REQUIRED | L.F.P.REQUIRED | L.P.REQUIRED <= r.REQUIRED
  assert hasattr(r,L.F.PRIVATE) and hasattr(f.OLD.F,L.D.MARKER)
  assert all(a.guards().get(p)==h for p,h in L.D.guards().items())
 assert (r.run,r.write,r.finish,r.finalize,r.REQUIRED,f.OLD.F.validate_comparison,f.OLD.F.diagnostic_reports)==before
 assert not hasattr(r,L.F.PRIVATE) and not hasattr(f.OLD.F,L.D.MARKER)
assert not any(n=='src' or n.startswith('src.') for n in sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=L.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_addon_metadata_order_and_failure(monkeypatch: Any) -> None:
    calls = []
    def note(name: str) -> Any:
        return lambda *args, **kwargs: calls.append(name)
    old = SimpleNamespace(REQUIRED=frozenset(('old',)), install=note('old_install'),
        finish=note('old_finish'), verify=note('old_verify'), guards=lambda: {}, bind=note('old_bind'))
    monkeypatch.setattr(L.C, 'bootstrap', lambda: (object(), object(), old, object()))
    monkeypatch.setattr(L, 'current_release', lambda state: object())
    for name in ('install', 'finish', 'verify'):
        monkeypatch.setattr(L.M, name, note('metadata_' + name))
        monkeypatch.setattr(L.P, name, note('pending_' + name))
    addon = L.bootstrap()[2]
    with contextlib.ExitStack() as stack:
        addon.install(stack, object(), object(), {})
    addon.bind(object())
    addon.finish({})
    addon.verify(L.ROOT)
    assert calls == ['old_install', 'pending_install', 'metadata_install', 'old_bind', 'old_finish',
        'metadata_finish', 'pending_finish', 'old_verify', 'metadata_verify', 'pending_verify']
    error = RuntimeError('人工metadata失敗')
    def fail(*args: Any) -> None:
        raise error
    monkeypatch.setattr(L.M, 'finish', fail)
    with pytest.raises(RuntimeError) as caught:
        addon.finish({})
    assert caught.value is error


def test_function_lengths() -> None:
    for name in L.OWN:
        if name.endswith('.py'):
            for node in ast.walk(ast.parse((L.ROOT / name).read_bytes())):
                if isinstance(node, ast.FunctionDef):
                    assert node.end_lineno - node.lineno + 1 <= 50, (name, node.name)


def test_resource_guard_stops_only_bound_child(tmp_path: Any, monkeypatch: Any) -> None:
    import json
    import signal
    child = subprocess.Popen([sys.executable, '-u', '-c', 'import time; time.sleep(5)'])
    output = tmp_path / 'resources.jsonl'
    monkeypatch.setattr(Q, 'RSS_LIMIT_KIB', 0)
    monkeypatch.setattr(sys, 'argv', ['guard', str(child.pid), '-c', str(output)])
    try:
        assert Q.main() == 0
        assert child.wait(timeout=3) == -signal.SIGTERM
        row = json.loads(output.read_text().splitlines()[0])
        assert row['pid'] == child.pid and row['safety_stop'] is True
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=3)


def test_resource_guard_rejects_other_command(tmp_path: Any, monkeypatch: Any) -> None:
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'])
    monkeypatch.setattr(sys, 'argv', ['guard', str(child.pid), '/not/the/runner.py', str(tmp_path / 'wrong.jsonl')])
    try:
        with pytest.raises(ValueError, match='process_identity'):
            Q.main()
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=3)
