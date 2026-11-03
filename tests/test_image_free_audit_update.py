"""更新対象でないプロセスと、未検証の更新票を停止前に拒否する。"""
import importlib
import json
import os
from pathlib import Path

import pytest


def updater(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('image_free_audit_update')


def test_stop_refuses_unrelated_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = updater(monkeypatch)
    calls = []
    monkeypatch.setattr(module.os, 'kill', lambda *args: calls.append(args))
    with pytest.raises(ValueError, match='実コマンド'):
        module.stop_owned(tmp_path, dict(supervisor_pid=os.getpid(), worker_pids={}))
    assert not calls


def test_failed_gate_stops_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = updater(monkeypatch)
    (tmp_path/'audit_gate.json').write_text(json.dumps(dict(passed=False)))
    calls = []
    monkeypatch.setattr(module, 'stop_owned', lambda *args: calls.append(args))
    with pytest.raises(ValueError, match='合格票'):
        module.install(tmp_path, tmp_path, True)
    assert not calls
