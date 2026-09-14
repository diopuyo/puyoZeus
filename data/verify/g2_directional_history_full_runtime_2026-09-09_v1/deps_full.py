"""実fixtureと凍結history/handoffをcold読込する。foreign aliasは拒否。"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT.parent
HISTORY = VERIFY / 'g2_historical_completion_runtime_2026-09-09_v1'
HANDOFF = VERIFY / 'g2_directional_history_handoff_2026-09-09_v1'
LATEST = VERIFY / 'g2_directional_next_latest_runtime_2026-09-09_v1'
OWN = ('deps_full.py', 'assembly_history.py', 'fixture_full.py', 'run_cpu.py', 'PLAN.md')
FILES = {
    'connection': (LATEST / 'connection.py', '4926d87df624d5d359d51201fca73caccb18d7801b3a2094e06f812b1170cc52'),
    '_history_latest_cpu': (LATEST / 'actual_cpu.py', '83953222f582d21a81955bfd3be35c844a41380d6862e0bd400b2eb7607a25dc'),
    'history_state': (HISTORY / 'history_state.py', '9d66ab677dea73d596fc33623d67fbcbd111ba92e86147de5dc5b61c63392114'),
    'history_dependencies': (HISTORY / 'history_dependencies.py', '9d905f60bc013981dc5019d82feb350d6fadfadda08d44c76f68fd85f80ba62b'),
    'history_provider': (HISTORY / 'history_provider.py', 'fce0ac1ccfa8869a9bb71dcc5b4bc43f636ce7f6060b5321eb9750c46266447c'),
    'history_controller': (HISTORY / 'history_controller.py', '15880b3310207ca3119c29c41272fdc20a63a729812541c3352c1c47f9a839ef'),
    'history_ast': (HISTORY / 'history_ast.py', 'b1785c15eb1ae4ac5f6537c2a2019f58b66a1c93bde83fc2b18923f5cc7de027'),
    'fixed': (HANDOFF / 'fixed.py', '0b3d3f3cfe627186812941721e95fad96f3d0f96bf56674e990bdcc9a6640e46'),
    'evidence': (HANDOFF / 'evidence.py', '393e80a2168d0125879571f5679a4ead6f0d7421870894260dd6127f079798c2'),
    'handoff': (HANDOFF / 'handoff.py', 'c47f8384b50d618b9c3472cbc8ed551241a1a8b6b667bf4619118bbb57fb5064'),
}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, stack: Any) -> Any:
    path, expected = FILES[name]
    require(name not in sys.modules and sha(path) == expected, 'full_dependency:' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    stack.callback(sys.modules.pop, name, None)
    spec.loader.exec_module(value)
    return value


def history(stack: Any) -> dict[str, Any]:
    return {name: load(name, stack) for name in FILES if name not in ('connection', '_history_latest_cpu')}


def guards() -> dict[str, str]:
    require(all(sha(path) == expected for path, expected in FILES.values()), 'full_dependency_changed')
    return {str(path): expected for path, expected in FILES.values()} | {str(ROOT / n): sha(ROOT / n) for n in OWN}
