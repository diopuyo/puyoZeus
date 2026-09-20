"""専有sleep子で実pidfd停止と実wait、元RAM identityを検証する。"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
from typing import Any
import pytest
import supervise as S
import resource_guard as G

ROOT = Path(__file__).resolve().parent


@pytest.mark.parametrize('case', ['normal', 'thermal', 'early'])
def test_actual_owned_processes(tmp_path: Any, case: str) -> None:
    output = tmp_path / (case + '.jsonl')
    runner = str(ROOT / 'probe_guard_child.py')
    result = S.run_pair([sys.executable, runner, '.5' if case == 'normal' else '20'],
                        str(ROOT / 'probe_guard_worker.py'), runner, output)
    assert result['source'] == 'actual_Popen_wait'
    assert result['child_exit_code'] == (0 if case == 'normal' else 143)
    assert result['resource_guard_exit'] == (1 if case == 'early' else 0)
    assert not Path(f"/proc/{result['child_pid']}").exists()
    assert not Path(f"/proc/{result['guard_pid']}").exists()
    if case != 'early':
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert rows and all(r['safety_stop'] is (case == 'thermal') for r in rows)
        assert all(r['schema'] == G.SCHEMA for r in rows)


def test_changed_identity_never_signals(monkeypatch: Any, tmp_path: Any) -> None:
    tokens = iter([('owned', '1'), ('owned', '2')])
    monkeypatch.setattr(G.R, 'identity', lambda *args: next(tokens))
    monkeypatch.setattr(G.os, 'pidfd_open', lambda pid: os.open('/dev/null', os.O_RDONLY))
    monkeypatch.setattr(G.signal, 'pidfd_send_signal', lambda *args: pytest.fail('PID変更後の送信'))
    with pytest.raises(ValueError, match='pid_changed'):
        G.monitor(os.getpid(), 'owned', tmp_path / 'unused.jsonl')


def test_guard_ignoring_term_is_reaped(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setattr(S, 'GUARD_EXIT_TIMEOUT', .2)
    runner = str(ROOT / 'probe_guard_child.py')
    result = S.run_pair([sys.executable, runner, '.5'], str(ROOT / 'probe_guard_worker.py'),
                        runner, tmp_path / 'hang.jsonl')
    assert result['child_exit_code'] == 0 and result['resource_guard_exit'] == 137
    assert result['supervisor_error'] == 'guard_did_not_exit_after_child'
    assert not Path(f"/proc/{result['guard_pid']}").exists()
