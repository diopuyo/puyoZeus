"""D-v2dの保持量をライブ計装へ公開する。評価結果は変更しない。"""
from __future__ import annotations

from typing import Any


def integrity_memory_roots(overlay: Any) -> dict[str, Any]:
    """live_memory.retained_countsへ渡す所有者を列挙する。"""
    projection = getattr(overlay, '_landing_projection', None)
    return dict(post_counter_bound=getattr(projection, 'post_counter_bound', None),
                history_integrity=getattr(overlay, '_history_integrity', None))
