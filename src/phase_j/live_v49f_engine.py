"""v4.9f試験用の逐次評価口。入力順と計算原票を実時計から独立させる。"""
from __future__ import annotations

from functools import partial
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

MS = 1000.
ENV_ENABLED = 'PUYO_LIVE_V49F'
ENABLED_DEFAULT: bool = False


def enable_candidate() -> None:
    """専用子processだけで、検収済みの決定的探索と高速APIを選択する。"""
    from src.prefire_work_budget import FLAG
    from src import prefire_post_b3, exchange_count_prediction
    from src import (exchange_count_v310, exchange_count_v34, exchange_count_v39,
                     exchange_count_correction, exchange_correction_v310, prefire_chain_board)
    os.environ[FLAG] = '1'
    if getattr(prefire_post_b3, '_live_v49f_fast', False):
        return
    prefire_post_b3.forecast_features = partial(prefire_post_b3.forecast_features, fast_copy=True)
    compact = partial(exchange_count_prediction.simulate, compact=True)
    # 配布のimport閉包にも依存先を明示し、動的importによる同梱漏れを防ぐ。
    for module in (exchange_count_prediction, exchange_count_v310, exchange_count_v34,
                   exchange_count_v39, exchange_count_correction, exchange_correction_v310,
                   prefire_chain_board):
        module.simulate = compact
    prefire_post_b3._live_v49f_fast = True


def signature(engine: Any, old: dict) -> tuple:
    """原票と同じ入力変化判定。時刻やCPU負荷で入力を間引かない。"""
    state = engine.state
    return (state.game_idx, old['input_sec'],
        tuple(s.board.tobytes() if s.board is not None else None for s in state.sides),
        tuple(tuple(s.queue) for s in state.sides), tuple(s.state for s in state.sides),
        tuple(state.sl_snap.get(f'{key}_p{i+1}') for key in ('pending', 'generated') for i in (0, 1)))


class LiveV49FEngine:
    """配信と保存再生で共用する状態所有者。B3の本番状態とは共有しない。"""

    def __init__(self, predictor: Any, table: Any, policy: dict, native: Any,
                 pre_model: dict) -> None:
        from src.prefire_post_v47 import PostV47
        from src.prefire_v49f import PostV49F
        enable_candidate()
        self.base = PostV47(predictor, table, policy, native, True, pre_model)
        self.base.defer_all_clear = True
        self.engine = PostV49F(predictor, table, policy, native, True, pre_model)
        self.latest: dict | None = None
        self.previous: tuple | None = None
        self.game: int | None = None
        self.board_game: int | None = None
        self.board_start: float | None = None

    def observe(self, stamp: float, game: int, raw: list[dict],
                forecasts: list[dict], firepower_ledgers: list[dict] | None = None,
                adopted_game: int | None = None) -> tuple[dict | None, dict | None]:
        if game != self.game:
            self.latest, self.previous, self.game = None, None, game
        base = self.base.observe(stamp, game, raw, forecasts)
        if base is not None and 'v_post' in base:
            self.latest = base
        began = perf_counter()
        self.engine._boundary(stamp, game)
        self.engine._current(stamp, game, raw)
        current_ms = (perf_counter()-began)*MS
        from src.ledger_x_options import board_enabled
        if board_enabled():
            board_game = game if adopted_game is None else adopted_game
            if board_game != self.board_game:
                self.board_game, self.board_start = board_game, stamp
        if board_enabled() and self._needs_board_only():
            from src.ledger_x_board import board_record
            self.latest = None
            return base, board_record(stamp, game, raw, self.base.chains,
                                      firepower_ledgers, self.base.native, self.board_start)
        if self.latest is None or self.engine.state.sl_snap is None:
            return base, None
        if any(s.board is None for s in self.engine.state.sides):
            return base, None
        key = signature(self.engine, self.latest)
        if key == self.previous:
            return base, None
        value = self.engine.calculate_record(self.latest, raw)
        self.previous = key
        value.update(source_record_input=self.latest['input_sec'],
            evidence_max=max(value['evidence_times']), current_ms=current_ms,
            pre_delta_full=self.latest.get('pre_delta_full', 0))
        return base, value

    def _needs_board_only(self) -> bool:
        """部分台帳を相殺計算の確定値へ渡さず、既存の完全台帳経路を保つ。"""
        import math
        snap = self.base.state.sl_snap
        return (snap is None or snap.get('history_incomplete', False)
                or any(not math.isfinite(snap[f'pending_p{i+1}']) for i in range(2)))

    def close(self) -> None:
        self.base.pool.shutdown()
        self.engine.pool.shutdown()


def load_engine(model: Path, table: Path, policy: Path, pre_model: Path) -> LiveV49FEngine:
    import puyo_core
    from src.b3_predict import RowPredictor
    from src.b3_train_model import load_model
    from src.b3_timing import TimingTable
    return LiveV49FEngine(RowPredictor(load_model(model)),
        TimingTable.from_json(table.read_text(encoding='utf-8')),
        json.loads(policy.read_text(encoding='utf-8')), puyo_core,
        json.loads(pre_model.read_text(encoding='utf-8'))['models']['exchange_mixture'])
