"""spawn の子で使う軽い関数 (重い import をしない)。pipe の枠より多い出力を積んで終わる。"""
from __future__ import annotations

from typing import Any

ITEMS = 64
ITEM_BYTES = 4096


def flood(queue: Any) -> None:
    for _ in range(ITEMS):
        queue.put(b'x' * ITEM_BYTES)
