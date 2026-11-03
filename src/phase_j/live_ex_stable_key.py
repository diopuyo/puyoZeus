"""共通の連鎖予測の番号を鍵と事象の判定に使う配信側アダプタ。"""
from __future__ import annotations

from ._live_ex_frozen.src.dl1k_engine import ExEngine, Probe


class StableKeyExEngine(ExEngine):
    """値の計算を変えず、予測がある側だけ識別子を置き換える。"""

    def step(self, t: float, game: int, raw: list[dict]) -> None:
        """retR 計算の有効無効にかかわらず、鍵用の予測を両プロセスで進める。"""
        from ._live_ex_frozen.src.dl1k_retr import fire_tracker
        self.observe(t, game, raw)
        if not hasattr(self, 'retr_fire'):
            self.retr_fire = fire_tracker(self.native)
        self.retr_fire.observe(t, game, raw)

    def probe(self) -> Probe:
        original = super().probe()
        tracker = getattr(self, 'retr_fire', None)
        if tracker is None or not hasattr(tracker, 'current'):
            return original
        ids = list(original.key[2])
        for side in range(len(ids)):
            forecast = tracker.current(side, self.state.t)
            if forecast is not None and forecast.has_prediction:
                ids[side] = forecast.chain_id
        return Probe(original.fight, (*original.key[:2], tuple(ids)))
