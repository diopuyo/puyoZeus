"""rtを変更せず、B用差分を一時コピーへ適用して保持量計装を検査する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import hashlib
import json
from pathlib import Path
import runpy
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS

from src.exchange_history_integrity import MatchHistoryIntegrity
from src.exchange_post_counter_bound import PostCounterDeathBound

RT = Path(_public_dev_text('<repo>'))
OUT = Path('logs/integration_r1s_dv2d_20261002')
PATCH = Path('docs/patches/rt_live_memory_dv2d_20261002.patch')


def main() -> None:
    """実live_memoryの3保持量が公開されることと外部原本の不変を確認する。"""
    source = RT/'src/phase_j/live_memory.py'
    original = source.read_bytes()
    OUT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUT) as temporary:
        root = Path(temporary)
        copied = root/'src/phase_j/live_memory.py'
        copied.parent.mkdir(parents=True)
        copied.write_bytes(original)
        subprocess.run(['patch', '-p1', '-d', str(root), '-i', str(PATCH.resolve())], check=True)
        module = runpy.run_path(str(copied))
        guard, engine = MatchHistoryIntegrity(), PostCounterDeathBound()
        guard.games[1] = dict(game=1)
        guard.audit.append(dict(game=1))
        engine.audit.append(dict(game=1))
        overlay = NS(_history_integrity=guard, _landing_projection=NS(post_counter_bound=engine))
        roots = module['evaluation_roots'](dict(event_overlay=overlay), NS(callback=None, meter=None))
        counts = module['retained_counts'](roots)
        expected = ('history_integrity.games', 'history_integrity.audit', 'post_counter_bound.audit')
        assert all(counts[key] == 1 for key in expected)
    assert source.read_bytes() == original
    result = dict(rt_unchanged=True, source_sha256=hashlib.sha256(original).hexdigest(),
        patch_sha256=hashlib.sha256(PATCH.read_bytes()).hexdigest(), counts=counts)
    (OUT/'MEMORY_PATCH.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
