"""通知単位の本番評価と表示平滑。配信側は確定済みの最新値だけを読む。"""
from __future__ import annotations

from typing import Any
from src.exchange_event_overlay import ExchangeEventOverlay
from scripts.visualize_advantage_overlay import _ExchangeDisplayEMA, _exchange_display
from src.exchange_display_smoothing import SwitchAwareDisplayEMA


class NotificationExchangeOverlay(ExchangeEventOverlay):
    """オフライン再生と同じ評価・EMA順序を配信頻度から独立させる。"""

    def __init__(self, *args: Any, switch_smoothing: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # 既定OFF。ONなら評価器切替の値飛び対策 (試合境界で古いEMAから再開しない・非イベント切替を短くブレンド)。
        self.switch_smoothing = switch_smoothing
        self.smoothing = SwitchAwareDisplayEMA() if switch_smoothing else _ExchangeDisplayEMA()
        self.display: tuple[float, float] | None = None

    def update(self, *args: Any, **kwargs: Any) -> None:
        super().update(*args, **kwargs)
        stamp = args[3] if len(args) > 3 else kwargs['t_sec']
        # 通知単位の評価は旧評価器の値を持たない (None)。切替対応では値の無い間は表示状態を保持する。
        fallback = (None, None) if self.switch_smoothing else (0., .5)
        value = _exchange_display(self, *fallback, self.smoothing, stamp)
        self.display = value if self.tracker.probability is not None else None

    def calculate(self) -> None:
        """互換用の公開境界。評価もEMAもupdateで完了している。"""

    def smoothing_state(self) -> tuple:
        """親プロセスへ返す表示平滑の状態。旧形式は (adv, 確率, last_sec) のまま。"""
        if self.switch_smoothing:
            return self.smoothing.export_state()
        return self.smoothing.adv, self.smoothing.probability, self.smoothing.last_sec

    def restore_smoothing(self, values: tuple) -> None:
        """試合journalの再実行前に、前試合末尾のEMAだけを戻す。"""
        if self.switch_smoothing:
            self.smoothing.restore_state(values)
            return
        self.smoothing.adv, self.smoothing.probability, self.smoothing.last_sec = values


def latest_display(overlay: Any, adv: float, probability: float,
                   smoothing: Any = None, t_sec: float | None = None) -> tuple:
    """本番値が欠測のときだけ、呼出元の既存フォールバックを保つ。"""
    if hasattr(overlay, 'display'):
        return overlay.display if overlay.display is not None else (adv, probability)
    return _exchange_display(overlay, adv, probability, smoothing, t_sec)
