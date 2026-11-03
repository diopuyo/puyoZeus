"""同一入力でキャッシュOFF/ONを独立再投入し、FIX4と同じ署名を保存する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from typing import Iterator
from unittest.mock import patch

ROOT = Path(_public_dev_text('<work>/wt_live_speed'))
MODE = os.environ.get('SPEED3_MODE', 'off')
OUT = Path(_public_dev_text('<work>/speed3')) / ('candidate_'+MODE)
sys.path[:0] = [str(ROOT), _public_dev_text('<work>/fix4')]
import replay_reference as base
from src.phase_j import live_cache
from src.phase_j.live_future_cache import FutureSearchCache

ORIGINAL = live_cache.cached_future_search


@contextmanager
def tracked_cache() -> Iterator[FutureSearchCache | None]:
    with ORIGINAL() as cache:
        try:
            yield cache
        finally:
            value = cache.summary() if cache is not None else dict(enabled=False)
            (OUT/'cache.json').write_text(json.dumps(value, indent=2), encoding='utf-8')


def main() -> None:
    assert MODE in ('off', 'on')
    OUT.mkdir(exist_ok=True)
    os.environ['PUYO_LIVE_REFERENCE_TIMES_ONLY'] = '1'
    os.environ['PUYO_LIVE_NEAR_FUTURE_CACHE'] = str(int(MODE == 'on'))
    base.OUT, base.MODE, base.PROFILE = OUT, '1', False
    # 子評価workerと同じ限定スコープ。認識側の連鎖キャッシュには適用しない。
    base.bounded_chain_caches = lambda: live_cache.bounded_chain_caches(future_search=True)
    with patch.object(live_cache, 'cached_future_search', tracked_cache):
        base.main()


if __name__ == '__main__':
    main()
