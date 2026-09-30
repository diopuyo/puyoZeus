"""保留通知の復帰時の複製が frozen dataclass でも落ちないこと (B18 欠陥A)。"""
from __future__ import annotations

from dataclasses import dataclass, FrozenInstanceError
from types import SimpleNamespace

import pytest

from src.exchange_event_overlay import _replaced


@dataclass(frozen=True)
class FrozenSide:
    chain_event: object
    score: int


def test_frozen_side_is_replaced_without_mutating_original() -> None:
    side = FrozenSide(chain_event='old', score=10)
    with pytest.raises(FrozenInstanceError):
        side.chain_event = 'new'  # 旧実装の代入はここで落ちていた
    saved = _replaced(side, chain_event='new')
    assert saved == FrozenSide('new', 10) and side.chain_event == 'old'


def test_mutable_side_keeps_shallow_copy_behavior() -> None:
    side = SimpleNamespace(chain_event='old', score=10)
    saved = _replaced(side, chain_event='new')
    assert saved.chain_event == 'new' and saved.score == 10 and side.chain_event == 'old'
    assert saved is not side
