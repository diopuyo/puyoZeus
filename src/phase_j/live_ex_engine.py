"""正式抽出器の状態と値を所有するEX専用アダプタ（既定OFF）。"""
from __future__ import annotations

from typing import Any
from types import SimpleNamespace

from ._live_ex_frozen.src.dl1k_engine import ExEngine
from .live_ex_compute import compute
from ._live_ex_frozen.src.dl1k_extractor import extra_from
from ._live_ex_frozen.src.dl1k_columns import COLUMNS, MASK_NOT_APPLICABLE, blank
from ._live_ex_frozen.src.dl1k_regen_state import EventMonitor
from ._live_ex_frozen.src.prefire_tsumo_v411 import Knowledge

TAPE_KEYS = (('next_pair', 'tape_next_pair'), ('dnext_pair', 'tape_dnext_pair'))


class TapeNextKnowledge(Knowledge):
    """テープにだけ毎コマの NEXT (tape_* の鍵) を渡す。鍵が無いコマは元と同じ値を使う。"""

    def observe(self, stamp: float, game: int, raw: list[dict]) -> None:
        super().observe(stamp, game, [dict(row, **{key: row[tape] for key, tape in TAPE_KEYS if tape in row})
                                      for row in raw])


class LiveExEngine:
    """確定した観測だけを順番に進め、B3推論を抽出へ戻さない。"""

    def __init__(self, policy: dict, native: Any, seed_lookup: Any = None,
                 epoch: int = 0, ledger_hook: Any = None, tape_next: bool = False,
                 stable_key: bool = False, xc_same_time: bool = False) -> None:
        from .live_ex_stable_key import StableKeyExEngine
        engine_type = StableKeyExEngine if stable_key else ExEngine
        self.engine = engine_type(policy, native, seed_lookup, epoch, ledger_hook)
        if tape_next:   # 観測前に差し替える (状態は空)。OFF では凍結の Knowledge のまま
            self.engine.knowledge = TapeNextKnowledge(self.engine.knowledge.lookup)
        self.monitor = EventMonitor()
        self.xc_track = xc_same_time and not stable_key
        self.observations = 0

    def observe(self, packet: dict) -> Any:
        self.engine.step(packet['t'], packet['game'], packet['raw'])
        if getattr(self, 'xc_track', False):
            from ._live_ex_frozen.src.dl1k_engine import retr_enabled
            from ._live_ex_frozen.src.dl1k_retr import fire_tracker
            if not retr_enabled():
                if not hasattr(self.engine, 'retr_fire'):
                    self.engine.retr_fire = fire_tracker(self.engine.native)
                self.engine.retr_fire.observe(packet['t'], packet['game'], packet['raw'])
        self.monitor.observe(self.engine, SimpleNamespace(t=packet['t'], index=self.observations))
        self.observations += 1
        return self.engine.probe()

    def take_resets(self) -> list[dict]:
        return self.monitor.take()

    def calculate(self) -> dict:
        self.monitor.take()
        probe = self.engine.probe()
        if not probe.fight:
            return dict(key=probe.key, fight=False, extra=blank(MASK_NOT_APPLICABLE), result=None)
        result = compute(self.engine)
        extra = extra_from(result)
        extra.update(result.get('xc', {}))
        return dict(key=probe.key, fight=True, extra=extra, result=result)

    def close(self) -> None:
        self.engine.close()
