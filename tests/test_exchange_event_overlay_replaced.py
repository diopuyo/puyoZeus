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


def test_replay_timeout_scales_with_journal_length() -> None:
    """再起動時の再実行は件数比例の応答期限を持つ (B18 欠陥B)。"""
    from src.phase_j import live_eval_supervisor as sup
    captured = {}
    owner = SimpleNamespace(static_error=None, discarded_replies=[], request_id=0, directory=None,
                            connection=SimpleNamespace(send=lambda request: None))
    owner.receive = lambda timeout, **_: (captured.__setitem__('timeout', timeout), dict(kind='ok'))[1]
    sup.SupervisedOverlay.request(owner, dict(op='advance', commands=[b'x']*1000, reset=True))
    assert captured['timeout'] == pytest.approx(sup.REPLY_TIMEOUT_SEC+1000*sup.REPLAY_TIMEOUT_PER_COMMAND_SEC)
    sup.SupervisedOverlay.request(owner, dict(op='advance', commands=[b'x']*1000, reset=False))
    assert captured['timeout'] == sup.REPLY_TIMEOUT_SEC
