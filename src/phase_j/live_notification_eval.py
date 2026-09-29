"""通知単位の本番評価と表示平滑。配信側は確定済みの最新値だけを読む。"""
from __future__ import annotations

from typing import Any
from src.exchange_event_overlay import ExchangeEventOverlay
from scripts.visualize_advantage_overlay import _ExchangeDisplayEMA, _exchange_display


class NotificationExchangeOverlay(ExchangeEventOverlay):
    """オフライン再生と同じ評価・EMA順序を配信頻度から独立させる。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.smoothing = _ExchangeDisplayEMA()
        self.display: tuple[float, float] | None = None

    def update(self, *args: Any, **kwargs: Any) -> None:
        super().update(*args, **kwargs)
        stamp = args[3] if len(args) > 3 else kwargs['t_sec']
        value = _exchange_display(self, 0., .5, self.smoothing, stamp)
        self.display = value if self.tracker.probability is not None else None

    def calculate(self) -> None:
        """互換用の公開境界。評価もEMAもupdateで完了している。"""

    def restore_smoothing(self, values: tuple) -> None:
        """試合journalの再実行前に、前試合末尾のEMAだけを戻す。"""
        self.smoothing.adv, self.smoothing.probability, self.smoothing.last_sec = values


def latest_display(overlay: Any, adv: float, probability: float,
                   smoothing: Any = None, t_sec: float | None = None) -> tuple:
    """本番値が欠測のときだけ、呼出元の既存フォールバックを保つ。"""
    if hasattr(overlay, 'display'):
        return overlay.display if overlay.display is not None else (adv, probability)
    return _exchange_display(overlay, adv, probability, smoothing, t_sec)
