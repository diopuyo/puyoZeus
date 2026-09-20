"""欠測は既確定の着地証拠の反証と区別する。原判定・公開・会計は再用する。"""
from __future__ import annotations
from contextlib import contextmanager
import functools
import hashlib
import inspect
from pathlib import Path
from typing import Any, Iterator

ORIGINAL_SHA = '15880b3310207ca3119c29c41272fdc20a63a729812541c3352c1c47f9a839ef'


@contextmanager
def installed(cls: Any) -> Iterator[None]:
    original = cls.observe_clear
    source = Path(inspect.getsourcefile(original))
    if hashlib.sha256(source.read_bytes()).hexdigest() != ORIGINAL_SHA:
        raise ValueError('missing_observation_original_changed')

    @functools.wraps(original)
    def observed(self: Any, binding: Any, item: Any, sm: Any, raw: Any,
                 signals: Any, view: Any) -> bool:
        # Noneは原observedがraw不一致/UNKNOWNを除外した結果。票は加算しない。
        if raw is None:
            return False
        return original(self, binding, item, sm, raw, signals, view)

    cls.observe_clear = observed
    try:
        yield
    finally:
        cls.observe_clear = original

