"""実行子のpipe容量を超えるログも、全bytesを順番どおり保存する。"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def smoke_module() -> object:
    path = Path(__file__).resolve().parents[1]/'packaging/smoke_live_obs_20261020.py'
    spec = importlib.util.spec_from_file_location('obs_smoke_logging_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(os.name != 'nt', reason='Windows配布の検収口')
def test_stdout_pipe_is_drained_before_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = smoke_module()
    (tmp_path/'app').mkdir()
    expected = ('記録行\n'*100000).encode('utf-8')
    command = [sys.executable, '-c', "import sys; sys.stdout.buffer.write(('記録行\\n'*100000).encode('utf-8'))"]

    def observe(child: object, args: object, errors: list) -> dict:
        assert child.wait(timeout=30) == 0
        assert not errors
        return dict(polls=1)

    monkeypatch.setattr(module, 'observe', observe)
    value, code = module.run_child(SimpleNamespace(out=tmp_path, bundle=tmp_path), command, dict(os.environ))
    assert value == dict(polls=1) and code == 0
    assert (tmp_path/'smoke.log').read_bytes() == expected


def test_shutdown_rejects_forced_or_incomplete_audits(monkeypatch: pytest.MonkeyPatch) -> None:
    """有効表示が一度出ても、worker強制終了や監査入力欠落を合格にしない。"""
    import json
    shutdown_checks = smoke_module().shutdown_checks
    records = {
        'b3_async_status.json': dict(submitted=10, finished=dict(inputs=10), forced_termination=[]),
        'v49f_async_status.json': dict(exit_code=0, forced_termination=False),
        'v49f_display_status.json': dict(submitted=10, received=10, failure=None, pump_alive=False),
        'ama_async_status.json': dict(exit_code=0, forced_termination=False, pump_alive=False),
    }
    monkeypatch.setattr(Path, 'exists', lambda path: path.name in records)
    monkeypatch.setattr(Path, 'read_text', lambda path, **kwargs: json.dumps(records[path.name]))
    args = SimpleNamespace(out=Path('unused'), fast_audit=True, ama_off=False)
    assert shutdown_checks(args)['passed']
    records['ama_async_status.json']['forced_termination'] = True
    assert not shutdown_checks(args)['passed']
    records['ama_async_status.json']['forced_termination'] = False
    records['b3_async_status.json']['finished'] = None
    assert not shutdown_checks(args)['passed']
    records['b3_async_status.json']['finished'] = dict(inputs=10)
    records['v49f_display_status.json']['failure'] = 'KeyError: 0'
    assert not shutdown_checks(args)['passed']
    records['v49f_display_status.json']['failure'] = None
    records['v49f_display_status.json']['received'] = 9
    assert not shutdown_checks(args)['passed']
    records['v49f_display_status.json']['received'] = 10
    del records['v49f_async_status.json']
    assert not shutdown_checks(args)['passed']


def test_overlay_retains_config_when_bytecode_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """-B追加後も開発用経路が設定を失わず、配布検収と区別される。"""
    import json
    module = smoke_module()
    saved = {}
    monkeypatch.setattr(Path, 'read_text', lambda path, **kwargs: '{"changed":[]}')
    monkeypatch.setattr(Path, 'write_text',
                        lambda path, value, **kwargs: saved.update({path.name: json.loads(value)}))
    args = SimpleNamespace(out=Path('records'), bundle=Path('bundle'),
                           source_overlay=Path('source'), source_plan=Path('plan.json'))
    command = ['python.exe', '-B', '-m', 'src.phase_j.launcher',
               '--config', 'settings.json', '--require-manifest', '--dev-allow-video']
    environment = {}
    actual = module.source_overlay(args, command, environment)
    assert actual[1] == '-B'
    assert actual[3:] == ['--config', 'settings.json', '--skip-manifest', '--dev-allow-video']
    assert saved['source_overlay.json']['distribution_smoke'] is False
    args.source_overlay = None
    assert module.source_overlay(args, command, environment) == command
