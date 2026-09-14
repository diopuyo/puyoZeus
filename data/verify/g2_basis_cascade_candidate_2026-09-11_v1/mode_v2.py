"""同callの基準連鎖/新手競合と、未消費FIFO先頭からのorigin借用を拒否。"""
from __future__ import annotations
from types import FunctionType
from typing import Any
import mode as V1

B, M = V1.B, V1.M


class Mode(V1.Mode):
    def origin_token(self, item: Any) -> str:
        B.require(self.native is not None and len(self.native.pending) == 1,
                  'origin_without_native_consumption')
        return self.native.pending[0].occurrence_token

    def capture_origin(self, item: Any) -> None:
        if self.native.pending:
            value = self.connection.registry.current(self.connection.binding)
            for event in item['events']:
                origin = event.get('active_origin')
                if origin is None or (origin['object_id'], origin['trigger_sec']) in self.origin_ids:
                    continue
                if origin['before_board'] is None:
                    continue  # 元の必須before検査へ渡す。
                grid = B.grid(B.Board.from_dict({'grid': origin['before_board']['grid']}))
                if all(w.grid[B.HIDDEN_ROWS:] == grid[B.HIDDEN_ROWS:] for w in value.worlds):
                    result = B.ChainSimulator(exclude_hidden_row_from_pop=True).simulate(
                        B.Board.from_dict({'grid': value.worlds[0].grid}))
                    B.require(result.chain_count == 0, 'basis_chain_native_hand_ambiguous')
        super().capture_origin(item)


def install(stack: Any, connection: Any, state: dict[str, Any]) -> Mode:
    return FunctionType(M.install.__code__, dict(vars(M), Mode=Mode))(stack, connection, state)
