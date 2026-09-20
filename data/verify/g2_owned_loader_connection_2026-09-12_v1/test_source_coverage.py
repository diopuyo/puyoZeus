"""実constructorのMROと遅延依存がGO保護集合から脱落しない。"""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import target_entry as T


def test_actual_origins_and_late_dependencies_are_guarded() -> None:
    constructor = json.loads((T.ROOT / 'constructor_v1/CONSTRUCTOR_CLOSED.json').read_bytes())
    paths = T.runtime_sources()
    extra = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with ExitStack() as stack:
        selected = T.A.configured(stack)
        common = selected.__globals__['K']
        T.A.protect(stack, common, extra)
        pins = common.guards()
    assert set(constructor['context_perform_origins']) <= set(pins)
    assert all(str(p) in pins for p in T.A.START.glob('*.py'))
    for name in T.A.DEPENDENCY_ROOTS:
        files = list((T.A.VERIFY / name).glob('*.py'))
        assert files and all(str(p) in pins for p in files)
