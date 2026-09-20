"""既存最新runtimeを保持し原N直後の接続を同期setupへ合成する。"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT.parent
PROVIDER = VERIFY / 'g2_directional_next_provider_2026-09-09_v2'
FIXED = {
    'dependencies': (PROVIDER / 'dependencies.py', 'fbebc0a96fd04cad8c51d851485f4e013d74ea462b7c4207a35a5e77fed93fd2'),
    'stream': (PROVIDER / 'stream.py', '0738ce45ecd275a3172a4bc7ac0eb3bf10aa434b9c6f9722999a3959d41ec409'),
    '_directional_provider': (PROVIDER / 'provider.py', 'dc6e52908e33f9c5af2d17e485f6adebb7a3783c02485e7c223edf1b37bcb3ba'),
    '_directional_occurrence': (VERIFY / 'g2_directional_next_enqueue_2026-09-09_v2/occurrence.py', 'b71a992a94d5231278242ac14b73dab725786e713f543956a33a9f40179b86dc'),
    '_directional_runtime': (VERIFY / 'g2_directional_next_runtime_2026-09-09_v2/runtime.py', 'd575ac8ff8a14492f7a6fbe960426fa33825fb1a977dbcab4e2713c95ce965d0'),
}


def load(alias: str) -> Any:
    path, digest = FIXED[alias]
    if alias in sys.modules or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('directional_dependency_collision_or_changed:' + alias)
    spec = importlib.util.spec_from_file_location(alias, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[alias] = value
    try:
        spec.loader.exec_module(value)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return value


@contextmanager
def configured(runtime: Any, addon: Any, source_id: str, run_id: str) -> Iterator[dict[str, Any]]:
    loaded, binding = {}, {}
    with ExitStack() as stack:
        for alias in FIXED:
            loaded[alias] = load(alias)
            stack.callback(sys.modules.pop, alias, None)
        o, p, r = (loaded[name] for name in ('_directional_occurrence', '_directional_provider', '_directional_runtime'))
        original = runtime.M.instrument
        def instrument(inner: Any, collector: Any, history: Any, receipt: Any, state: Any) -> None:
            with r.before_observation(o, p, source_id=source_id, run_id=run_id) as captured:
                original(inner, collector, history, receipt, state)
                r.bind_journal(state, captured[0], addon.journal)
                binding.update(captured[0])
        runtime.M.instrument = instrument
        stack.callback(setattr, runtime.M, 'instrument', original)
        yield binding
