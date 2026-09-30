"""E35 の候補証明の早期打切り (既定OFF) が、判定を変えず、証明する候補数だけを減らすことを確かめる。"""
from __future__ import annotations

import numpy as np
import pytest

import src.exchange_post_counter_bound as bound
from src.board import Board

PALETTE = (1, 2, 3, 4)
QUEUE = (1, 2, 3, 4)
INCOMING, HANDS, ELAPSED = 30, 2, 60.0


def board_with(value: int) -> Board:
    board = Board()
    board._grid[12, 0] = value  # 盤面ごとに別のバイト列にする
    return board


def run(early_exit: bool, deaths: list[bool], monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict], list[int]]:
    seen: list[int] = []

    def fake_prove(board: Board, *args: object) -> dict:
        index = int(board._grid[12, 0]) - 1
        seen.append(index)
        return dict(dead=deaths[index], reason='fake', nodes=index, pruned=0, bounds=[])
    monkeypatch.setattr(bound, 'prove_post_counter', fake_prove)
    engine = bound.PostCounterDeathBound(early_exit=early_exit)
    boards = [board_with(i + 1) for i in range(len(deaths))]
    return engine.proofs_for(boards, QUEUE, INCOMING, HANDS, ELAPSED, PALETTE), seen


@pytest.mark.parametrize('deaths', ([True, True, True], [True, False, True, True], [False, True, True]))
def test_decision_is_same_and_early_exit_stops_at_first_survivor(deaths: list[bool],
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    full, seen_full = run(False, deaths, monkeypatch)
    early, seen_early = run(True, deaths, monkeypatch)
    assert all(p['dead'] for p in full) == all(p['dead'] for p in early) == all(deaths)
    assert seen_full == list(range(len(deaths)))            # 従来は全候補を証明する
    assert early == full[:len(early)]                       # 早期打切りは従来の先頭部分と一致
    first_survivor = next((i for i, d in enumerate(deaths) if not d), None)
    assert len(early) == (len(deaths) if first_survivor is None else first_survivor + 1)


def test_default_is_off() -> None:
    assert bound.PostCounterDeathBound().early_exit is False
