"""実際の待受ポートを削除用に伝える。残った記録だけでは停止を許可しない。"""
from __future__ import annotations

import json
import os
from pathlib import Path

INSTANCE_FILE = 'logs/instance.json'


def record_port(root: Path, port: int) -> None:
    path = root / INSTANCE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(root=str(root.resolve()), port=port, pid=os.getpid())),
                         encoding='utf-8')
    temporary.replace(path)


def recorded_port(root: Path, fallback: int) -> int:
    try:
        state = json.loads((root / INSTANCE_FILE).read_text(encoding='utf-8'))
        if not isinstance(state, dict) or state.get('root') != str(root.resolve()):
            return fallback
        port = state.get('port')
        return port if type(port) is int and 1 <= port <= 65535 else fallback
    except (OSError, ValueError, TypeError):
        return fallback
