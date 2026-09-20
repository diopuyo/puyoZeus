"""
予告お邪魔の時系列追跡モジュール

各フレームでの OjamaWarningResult を時刻付きで蓄積し、
連鎖発火 → 予告生成 → 落下 → 相殺 のイベントを記録する。

Usage:
    tracker = OjamaTimelineTracker()
    for t, frame in frames:
        warning = detector.detect(frame)
        tracker.update(t, warning)
    pending = tracker.get_pending_ojama("1P")
    history = tracker.get_history()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.ojama_warning import OjamaWarningResult

# ============================
# 定数
# ============================
SIDE_P1: str = "1P"
SIDE_P2: str = "2P"
VALID_SIDES: tuple[str, ...] = (SIDE_P1, SIDE_P2)


# ============================
# データクラス
# ============================


@dataclass(frozen=True)
class OjamaTimelineEvent:
    """時系列上のイベント 1 件。

    Attributes:
        t_sec: イベント時刻 (動画開始からの秒数)。
        side: "1P" または "2P"。
        total_count: そのフレーム時点の予告おじゃま合計。
        change_from_previous: 直前イベントからの差分 (+:増加, -:相殺/減少)。
    """
    t_sec: float
    side: str
    total_count: int
    change_from_previous: int


# ============================
# トラッカー本体
# ============================


class OjamaTimelineTracker:
    """予告お邪魔の時系列追跡を行うトラッカー。

    update() に時刻と OjamaWarningResult ペアを与えると、
    side ごとの最新値と履歴を保持する。get_pending_ojama() で
    最新の予告合計、get_history() で全イベントを取得できる。
    """

    def __init__(self) -> None:
        """履歴とサイドごとの最新値を初期化する。"""
        self._history: list[OjamaTimelineEvent] = []
        self._latest: dict[str, int] = {SIDE_P1: 0, SIDE_P2: 0}
        self._last_t: float = -1.0

    # ---- 公開メソッド ---------------------------------------------

    def update(
        self,
        t_sec: float,
        warning: tuple[OjamaWarningResult, OjamaWarningResult],
    ) -> None:
        """1 フレーム分の予告状態を取り込む。

        Args:
            t_sec: フレーム時刻 (秒)。
            warning: (1P 結果, 2P 結果) のタプル。
        """
        if t_sec < self._last_t:
            raise ValueError(
                f"時刻が逆行しています: prev={self._last_t} now={t_sec}",
            )
        for res in warning:
            self._record_side(t_sec, res)
        self._last_t = t_sec

    def get_pending_ojama(self, side: str) -> int:
        """指定サイドの最新の予告おじゃま合計を返す。"""
        if side not in VALID_SIDES:
            raise ValueError(f"side は 1P/2P のいずれか: {side}")
        return self._latest[side]

    def get_history(self) -> list[OjamaTimelineEvent]:
        """全イベント履歴をリストで返す (シャローコピー)。"""
        return list(self._history)

    def filter_history(
        self, side: str | None = None, only_changes: bool = True,
    ) -> list[OjamaTimelineEvent]:
        """側面・差分有無でフィルタした履歴を返す。

        Args:
            side: 指定なら "1P" または "2P" のみ。
            only_changes: True なら change_from_previous != 0 のみ。
        """
        events: Iterable[OjamaTimelineEvent] = self._history
        if side is not None:
            if side not in VALID_SIDES:
                raise ValueError(f"side は 1P/2P のいずれか: {side}")
            events = (e for e in events if e.side == side)
        if only_changes:
            events = (e for e in events if e.change_from_previous != 0)
        return list(events)

    def reset(self) -> None:
        """状態をリセットする (試合開始時等)。"""
        self._history.clear()
        self._latest = {SIDE_P1: 0, SIDE_P2: 0}
        self._last_t = -1.0

    # ---- 内部 -----------------------------------------------------

    def _record_side(
        self, t_sec: float, result: OjamaWarningResult,
    ) -> None:
        """1 サイド分のイベントを記録する (差分計算込み)。"""
        if result.side not in VALID_SIDES:
            raise ValueError(f"未知 side: {result.side}")
        prev = self._latest[result.side]
        change = result.total_count - prev
        self._history.append(OjamaTimelineEvent(
            t_sec=t_sec,
            side=result.side,
            total_count=result.total_count,
            change_from_previous=change,
        ))
        self._latest[result.side] = result.total_count


__all__ = [
    "OjamaTimelineEvent",
    "OjamaTimelineTracker",
    "SIDE_P1",
    "SIDE_P2",
    "VALID_SIDES",
]
