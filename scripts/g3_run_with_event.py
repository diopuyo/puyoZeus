"""既存supervisorの実終了からWindows側の有限イベント処理へ一回接続する。"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any
from scripts import g3_run_outcome as O
from scripts import g3_agent_review as A

RELAY_TIMEOUT = 330


def windows_path(path: Path) -> str:
    """このworkspaceのC/Dマウントだけを変換し、任意shell文字列を生成しない。"""
    raw = str(path.resolve())
    for prefix, target in (('/mnt/c/', 'C:/'), ('/mnt/d/', 'D:/')):
        if raw.startswith(prefix):
            return target + raw[len(prefix):]
    raise ValueError('g3_event_relay_mount')


def relay(root: Path, label: str, windows_python: Path) -> dict:
    """応答不明時の再起動を禁止し、interopの実waitを別票へ保存する。"""
    O.destination(root, label)
    if os.name == 'nt' or not windows_python.is_file():
        raise ValueError('g3_event_relay_runtime')
    command = [str(windows_python), '-m', 'scripts.g3_end_event', '--root', windows_path(root), '--label', label]
    A.save(root / (label + '_EVENT_RELAY_REQUEST.json'), dict(command=command, automatic_retry=False))
    result: dict[str, Any] = dict(status='UNKNOWN_NO_RESEND', exit_code=None, quality_gate_clear=False)
    env = os.environ.copy()
    env['PYTHONPATH'] = windows_path(A.ROOT)
    try:
        with (root / (label + '_EVENT_RELAY.log')).open('xb') as log:
            waited = subprocess.run(command, cwd=A.ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=RELAY_TIMEOUT)
        result.update(status='EVENT_HANDLED_NOT_G3_PASS' if waited.returncode == 0 else 'FAILED',
                      exit_code=waited.returncode)
    except Exception as error:
        result['error'] = repr(error)
    A.save(root / (label + '_EVENT_RELAY_RESULT.json'), result)
    return result


def run(root: Path, label: str, command: list[str], runner: Path, windows_python: Path,
        *, supervisor: Any = None, relay_call: Any = relay) -> dict:
    """元親結果の失敗を通知成功で上書きせず、G3境界で終了する。"""
    result = O.run(root, label, command, runner, api=supervisor)
    try:
        event = relay_call(root, label, windows_python)
    except Exception as error:
        event = dict(status='FAILED', error=repr(error))
    combined = dict(parent=result, event=event, quality_gate_clear=False,
                    exit_code=0 if result['parent_exit_code'] == 0
                    and event['status'] == 'EVENT_HANDLED_NOT_G3_PASS' else 1)
    A.save(root / (label + '_RUN_EVENT_RESULT.json'), combined)
    return combined
