"""ライブ評価だけで近未来探索の完全同一入力を再利用する外部ラッパー。"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import logging
import os
from pathlib import Path
import pickle
from typing import Any, Callable, Iterator
from unittest.mock import patch

from src.chain import ChainSimulator
from src.scoring import compute_effective_rate
from . import live_stage_timing as timing

CACHE_SIZE = 2048
VERIFY_HITS = 32
LOGGER = logging.getLogger(__name__)


def source_sha() -> str:
    """配布版でも使えるコードSHA。依存するPython群とnative本体を含む。"""
    from src import puyo_core_bridge as bridge
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    native_path = getattr(bridge._native, '__file__', None)
    if native_path is not None:
        digest.update(Path(native_path).read_bytes())
    return digest.hexdigest()


class FutureSearchCache:
    """容量を制限し、先頭の命中を実計算で照合。不一致なら再利用を停止する。"""

    def __init__(self, function: Callable, code_sha: str, capacity: int = CACHE_SIZE,
                 verify_hits: int = VERIFY_HITS) -> None:
        self.function, self.signature = function, inspect.signature(function)
        self.code_sha, self.capacity, self.verify_hits = code_sha, capacity, verify_hits
        self.entries: OrderedDict = OrderedDict()
        self.hits = self.misses = self.verified = self.mismatches = 0
        self.disabled = False

    def key(self, args: tuple, kwargs: dict) -> bytes | None:
        """未知のシミュレータは共用しない。時刻は得点換算率が同じ時だけまとめる。"""
        from src.exchange_event_count_features import NativeSimulator
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        board, simulator = values.pop('board'), values.pop('sim')
        if type(simulator) not in (ChainSimulator, NativeSimulator):
            return None
        settings = (type(simulator).__module__, type(simulator).__qualname__,
                    getattr(simulator, '_exclude_hidden_row_from_pop', None),
                    getattr(simulator, '_fast_groups', None))
        # 探索は時刻を参照せず、終端のscore_to_ojamaだけが同じ規則で換算する。
        values['elapsed_sec'] = compute_effective_rate(values['elapsed_sec'])
        return pickle.dumps((self.code_sha, board._grid.dtype.str, board._grid.shape,
                             board._grid.tobytes(), settings, values))

    @timing.timed('event_near_future_search')
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.disabled:
            return self.function(*args, **kwargs)
        key = self.key(args, kwargs)
        if key is None:
            return self.function(*args, **kwargs)
        if key not in self.entries:
            self.misses += 1
            value = self.function(*args, **kwargs)
            self.entries[key] = deepcopy(value)
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
            return value
        self.hits += 1
        self.entries.move_to_end(key)
        cached = self.entries[key]
        if self.verified < self.verify_hits:
            value = self.function(*args, **kwargs)
            self.verified += 1
            if pickle.dumps(value) != pickle.dumps(cached):
                self.mismatches += 1
                self.disabled = True
                self.entries.clear()
                LOGGER.error('近未来探索キャッシュ照合不一致。以後は全件を実計算します。')
            return value
        return deepcopy(cached)

    def summary(self) -> dict:
        return dict(code_sha=self.code_sha, hits=self.hits, misses=self.misses,
                    verified=self.verified, mismatches=self.mismatches, disabled=self.disabled,
                    entries=len(self.entries), capacity=self.capacity)


@contextmanager
def cached_future_search() -> Iterator[FutureSearchCache | None]:
    """通知順序と純関数APIを保ち、ライブ評価の生存期間だけ置換する。"""
    from src import indicators_v2 as iv
    from src.production_config import LIVE_NEAR_FUTURE_CACHE_DEFAULT
    enabled = os.environ.get('PUYO_LIVE_NEAR_FUTURE_CACHE',
                             str(int(LIVE_NEAR_FUTURE_CACHE_DEFAULT))) == '1'
    if not enabled:
        yield None
        return
    cache = FutureSearchCache(iv._near_future_search, source_sha())
    try:
        with patch.object(iv, '_near_future_search', cache):
            yield cache
    finally:
        LOGGER.info('近未来探索キャッシュ: %s', cache.summary())
