"""段2の保持・EMA契約。死亡証明は本番Rから受け取り、B3では作らない。"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from src.b3_eval_adapter import CERTAIN_SOURCES, MAX_HOLD_SEC
from src.b3_stage2_trace import to_adv
from src.exchange_display_smoothing import SwitchAwareDisplayEMA
from src.b3_chain_display import ChainDisplay
from src.production_config import DISPLAY_FAST_EMA_DEFAULT

FAIL_HOLD_SEC = 0.5


class B3Display:
    """未平滑確率・表示確率・死亡根拠を別々に保持する。"""

    def __init__(self, fast_ema: bool = DISPLAY_FAST_EMA_DEFAULT) -> None:
        self.fast_ema = fast_ema
        self.game: int | None = None
        self.probability: float | None = None
        self.source = 'null'
        self.last_success = -float('inf')
        self.failed = False
        self.smoother = SwitchAwareDisplayEMA(fast_ema=self.fast_ema)
        self.raw: float | None = None
        self.shown = (0.0, 0.5)
        self.mode = 'null'
        self.contents: dict | None = None
        self.chain = ChainDisplay()

    def boundary(self, game: int) -> None:
        if self.game != game:
            self.__init__(fast_ema=self.fast_ema)
            self.game = game

    def prediction(self, probability: float, mode: int, t: float) -> None:
        self.probability, self.last_success, self.failed = probability, t, False
        self.mode = 'exchange' if mode else 'static'

    def update(self, t: float, death_source: str | None = None,
               death_probability: float | None = None, physical_count: int = 0,
               k: float | None = None, active: bool = True) -> tuple[float, float]:
        cap = FAIL_HOLD_SEC if self.failed else MAX_HOLD_SEC
        self.raw = self.probability if active and t - self.last_success <= cap else None
        self.source = self.mode if self.raw is not None else 'null'
        if death_source in CERTAIN_SOURCES and death_probability is not None:
            self.raw, self.source = death_probability, death_source
        if self.raw is None:
            self.smoother = SwitchAwareDisplayEMA(fast_ema=self.fast_ema)
            self.shown = (0.0, 0.5)
        else:
            tracker = SimpleNamespace(source=self.source, _game_idx=self.game,
                                      current=None, records=range(physical_count))
            self.shown = self.smoother.apply(tracker, (to_adv(self.raw, k), self.raw), t)
        return self.shown
