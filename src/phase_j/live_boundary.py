"""入力復帰と直後の得点リセットを、初期化を省かず同じ試合番号へまとめる。"""
from __future__ import annotations

from dataclasses import dataclass

RECOVERY_BOUNDARY_WINDOW_SEC = 5.0


@dataclass
class RecoveryBoundary:
    """一つの入力復帰に対し、最初の得点境界だけを統合する。"""

    forced_at: float | None = None

    def increment(self, forced: bool, scored: bool, now: float,
                  recovery: bool = True) -> int:
        previous = self.forced_at
        if scored:
            self.forced_at = None
            if (previous is not None and (not forced or recovery)
                    and 0 <= now-previous < RECOVERY_BOUNDARY_WINDOW_SEC):
                return 0
        elif forced:
            self.forced_at = now if recovery else None
        return 1
