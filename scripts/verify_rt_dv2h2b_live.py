"""従来の通知時間・RSS計測にH1′受理とE35段照合の原票を追加する。"""
from __future__ import annotations

from functools import wraps
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path.cwd()))
from scripts import verify_rt_integration_live_20261002 as base


def instrument_history() -> None:
    """辞書破棄後も、境界時と開始受理時の証拠を失わず保存する。"""
    from src.exchange_history_integrity import MatchHistoryIntegrity
    original = MatchHistoryIntegrity.observe

    @wraps(original)
    def observe(self: Any, sides: tuple, game: int, stamp: float) -> None:
        before = self.game
        complete = self.games.get(game, {}).get('history_complete', False)
        original(self, sides, game, stamp)
        if before != game or complete != self.games[game]['history_complete']:
            base.emit('h1_boundary', dict(game=game, t_sec=stamp, evidence=self.evidence()))

    MatchHistoryIntegrity.observe = observe


def instrument_steps() -> None:
    """式と得点表示の受理後の段履歴を変化時に保存し、入力配線を検査する。"""
    from src.exchange_post_counter_bound import PostCounterDeathBound
    original = PostCounterDeathBound.refresh_current_observation

    @wraps(original)
    def refresh(self: Any, overlay: Any, result: Any = None, stamp: float | None = None) -> None:
        original(self, overlay, result, stamp)
        state = repr(self.step_observations)
        if state != getattr(self, '_verification_steps', None):
            self._verification_steps = state
            base.emit('e35_observation', dict(game=overlay._game, t_sec=stamp,
                result_received=result is not None,
                entries=[dict(game=key[0], chain_id=key[1], **entry)
                         for key, entry in self.step_observations.items()]))

    PostCounterDeathBound.refresh_current_observation = refresh


def instrument_checks() -> None:
    """死亡証明へ進まない通知も、実際に呼ばれた照合結果を監査へ残す。"""
    from src.exchange_post_counter_bound import PostCounterDeathBound
    original = PostCounterDeathBound.prefix_check

    @wraps(original)
    def check(self: Any, projection: Any, overlay: Any, chain: Any, hidden: Any,
              stamp: float | None = None) -> dict:
        value = original(self, projection, overlay, chain, hidden, stamp)
        key = (overlay._game, chain.chain_id)
        seen = getattr(self, '_verification_checks', {})
        state = repr(value)
        if seen.get(key) != state:
            seen[key] = state
            self._verification_checks = seen
            base.emit('e35_check', dict(game=overlay._game, chain_id=chain.chain_id,
                t_sec=stamp, completion_prefix_check=value))
        return value

    PostCounterDeathBound.prefix_check = check


instrument_history()
instrument_steps()
instrument_checks()

if __name__ == '__main__':
    from scripts.run_live_pipeline_20260928 import main
    main()
