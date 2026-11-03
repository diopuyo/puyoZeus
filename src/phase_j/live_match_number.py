"""初期化用の試合キーを維持し、入力回復だけを公開番号から除く。"""
from __future__ import annotations

from bisect import bisect_right
from typing import Any

BEFORE_FIRST_BOUNDARY = -1


class PublicMatchNumber:
    """評価状態を持たず、境界の対応だけを記録する。古い行にも同じ番号を返す。"""

    def __init__(self) -> None:
        self.recoveries: list[int] = []
        self.last_boundary = BEFORE_FIRST_BOUNDARY

    def boundary(self, game: int, forced: bool, scored: bool, recovery: bool) -> None:
        if game <= self.last_boundary:
            raise ValueError('内部試合キーは境界ごとに増加する必要があります')
        self.last_boundary = game
        # 得点境界との同時発生は実試合の開始を優先。不明な機器変更も省かない。
        if forced and recovery and not scored:
            self.recoveries.append(game)

    def number(self, game: int) -> int:
        return game - bisect_right(self.recoveries, game)

    def decorate(self, row: dict[str, Any]) -> dict[str, Any]:
        """保存評価行は書き換えず、公開用コピーにだけ番号を付ける。"""
        return dict(row, public_game=self.number(row['game']))
