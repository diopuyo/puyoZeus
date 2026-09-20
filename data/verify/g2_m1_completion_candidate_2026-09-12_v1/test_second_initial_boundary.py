"""人工action境界を原J/観測へ通し、初回基準だけが厳格拒否する反例を固定する。"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace as N
from typing import Any

import pytest
from test_second_basis import saved, live, context, policy
from test_second_observation import generated
import test_journal_witness as T
import second_observation as O
import second_basis as B
import journal_witness as W
import second_basis_boundary as NEW


@pytest.mark.parametrize('change', [False, True])
@pytest.mark.parametrize('repaired', [False, True])
def test_original_first_basis_action_boundary(context: Any, change: bool, repaired: bool) -> None:
    c = context
    before = c.journal.scope()
    after = deepcopy(before)
    before['generation']['action_revision'] = before['generation']['action_revision'] or 0
    after['generation']['action_revision'] = before['generation']['action_revision'] + int(change)
    calls: list[None] = []

    def scope(*args: Any) -> dict:
        calls.append(None)
        return deepcopy(before if len(calls) == 1 else after)

    c.journal.scope = scope
    c.journal.tracker = N(generation=lambda side: T.Generation(**after['generation']))
    with ExitStack() as stack:
        witness = W.install(stack, c.journal)
        evidence = O.install(stack, c.journal, c.state)
        generated(c.journal, c.pipe, c.result, c.row)
        assert evidence.error is None and evidence.latest['state'] == 'stable'
        # 片側のみを生成する限定反例。原pairの両側要件は人工1P対照で補う。
        witness.rows['1P'] = witness.rows['2P']
        qualify = NEW.wrapped(B.qualify, B) if repaired else B.qualify
        if change:
            error = B.BasisHold if repaired else ValueError
            reason = 'second_action_boundary_wait' if repaired else 'second_basis_generation'
            with pytest.raises(error, match=reason):
                qualify(evidence, witness, c.pipe)
            if repaired:
                c.pipe._active_chain_2p = object()
                with pytest.raises(ValueError, match='second_basis_current'):
                    qualify(evidence, witness, c.pipe)
        else:
            row, _ = qualify(evidence, witness, c.pipe)
            assert row is evidence.latest
