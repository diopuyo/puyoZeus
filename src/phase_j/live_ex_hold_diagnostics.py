"""配信で見送った採用行の検査失敗をプロセス単位で数える。"""
from __future__ import annotations

from collections import Counter
import logging
from threading import Lock

_COUNTS: Counter[str] = Counter()
_LOCK = Lock()
_LOGGER = logging.getLogger(__name__)


def record_rejection(reason: str) -> None:
    """同じ理由の警告は初回と件数が倍になる時だけ出す。"""
    with _LOCK:
        _COUNTS[reason] += 1
        count = _COUNTS[reason]
    if count & (count - 1) == 0:
        _LOGGER.warning('EX 採用を見送り: %s (累計 %d 件)', reason, count)


def rejection_diagnostics() -> dict:
    """テレメトリ側にはコピーを渡し、別スレッドから安全に読めるようにする。"""
    with _LOCK:
        counts = dict(_COUNTS)
    return dict(total=sum(counts.values()), by_reason=counts)
