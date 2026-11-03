"""明示したworkerの文脈内だけ、連鎖物理をRust bitboardへ統一する。"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import accumulate
from typing import Any, Iterator
from src.board import Board

BACKEND: ContextVar[Any] = ContextVar('exchange_native_backend', default=None)
ROWS, COLS = 13, 6


@contextmanager
def native_chains(native: Any) -> Iterator[None]:
    if not hasattr(native, 'simulate_chain_trace_py'):
        raise RuntimeError('Rustの連鎖トレース対応拡張が必要です。配列版へ退避しません')
    token = BACKEND.set(native)
    try:
        yield
    finally:
        BACKEND.reset(token)


def as_board(raw: list[int]) -> Board:
    return Board.from_list([raw[i:i+COLS] for i in range(0, ROWS*COLS, COLS)])


def chain_result(board: Board, hidden: bool, native: Any) -> Any:
    from src.chain import ChainResult, ChainStep, PuyoGroup
    trace = native.simulate_chain_trace_py(board._grid.ravel().tolist(), hidden)
    steps = []
    for index, (before, after, groups, ojama, score) in enumerate(trace, 1):
        erased = [PuyoGroup(c, frozenset(map(tuple, cells)), len(cells), frozenset(map(tuple, adjacent)))
                  for c, cells, adjacent in groups]
        steps.append(ChainStep(index, erased, ojama, sum(g.size for g in erased), as_board(before), as_board(after)))
    total = sum(s.erased_count for s in steps)
    return ChainResult(steps, len(steps), total, sum(s.erased_ojama for s in steps),
                       steps[-1].board_after.copy() if steps else board.copy(), total)


def completion(raw: list[int], native: Any) -> dict:
    trace = native.simulate_chain_trace_py(raw, True)
    prefix = list(accumulate(step[4] for step in trace))
    return dict(prefix=prefix, score=prefix[-1] if prefix else 0,
                chain_count=len(trace), final_board=trace[-1][1] if trace else list(raw))
