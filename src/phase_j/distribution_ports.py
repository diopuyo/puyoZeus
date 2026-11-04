"""希望ポートの所有者を確認し、利用者の操作で別の展開先から引き継ぐ。"""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Any
from urllib.request import Request, urlopen

PROBE_TIMEOUT = 1
SWITCH_TIMEOUT = 10
RETRY_SECONDS = 0.1
MAX_STATE_BYTES = 1024 * 1024


def inspect_conflict(root: Path, port: int) -> dict[str, Any]:
    result = dict(kind='other', port=port)
    try:
        with urlopen(f'http://127.0.0.1:{port}/api/state', timeout=PROBE_TIMEOUT) as response:
            state = json.loads(response.read(MAX_STATE_BYTES))
        if (not isinstance(state, dict) or not isinstance(state.get('root'), str)
                or not Path(state['root']).is_absolute() or not isinstance(state.get('settings'), dict)
                or 'running' not in state or 'diagnostics' not in state):
            return result
        return dict(kind='same' if Path(state['root']).resolve() == root.resolve() else 'different',
                    root=state['root'], port=port)
    except (OSError, ValueError, TypeError):
        from .distribution_port_owner import inspect_owner
        return inspect_owner(port) or result


def stop_conflict(conflict: dict[str, Any]) -> None:
    if conflict['kind'] == 'unresponsive':
        from .distribution_port_owner import stop_owner
        stop_owner(conflict)
        return
    request = Request(f"http://127.0.0.1:{conflict['port']}/api/stop", data=b'{}',
                      headers={'Content-Type': 'application/json', 'Host': f"127.0.0.1:{conflict['port']}"})
    with urlopen(request, timeout=PROBE_TIMEOUT) as response:
        response.read(MAX_STATE_BYTES)


def switch_port(server: Any, runtime: Any) -> dict[str, Any]:
    """相手を再確認し、希望ポートだけを確保する。別ポートへは逃げない。"""
    from .local_http import LocalHTTPServer
    with runtime.control_lock:
        conflict = runtime.port_conflict
        if not conflict or conflict['kind'] not in ('different', 'unresponsive'):
            raise ValueError('切り替える別のアプリがありません / No other app to switch from')
        if inspect_conflict(runtime.root, conflict['port']) != conflict:
            raise ValueError('起動中のアプリが変わりました。再起動してください / The app changed; restart this app')
        stop_conflict(conflict)
        deadline = time.monotonic() + SWITCH_TIMEOUT
        while True:
            try:
                replacement = LocalHTTPServer(('127.0.0.1', conflict['port']), server.RequestHandlerClass)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise ValueError('ポートが空きませんでした / The port is still in use')
                time.sleep(RETRY_SECONDS)
        from .distribution_instance import record_port
        try:
            record_port(runtime.root, replacement.server_port)
        except OSError:
            replacement.server_close()
            raise
        runtime.port = replacement.server_port
        runtime.port_conflict = None
        server.next_server = replacement
        threading.Thread(target=server.shutdown, daemon=True).start()
        return dict(url=f'http://127.0.0.1:{runtime.port}/settings')
