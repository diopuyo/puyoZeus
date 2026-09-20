"""元J捕捉/context票でオンライン分類と独立再構成を比較。1P台帳は人工対照。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from test_session_flags_connection import saved, live, session, connected
import capture_eligibility as E
import qualification_saved as Q


def test_duplicate_application_not_hidden_by_count() -> None:
    ledger = dict(arrivals=[dict(token='A'), dict(token='B')], applied=['A', 'A'],
                  acknowledgements=[dict(token='A'), dict(token='B')])
    with pytest.raises(ValueError, match='arrival_prefix'): Q.arrival_pending(ledger)


@pytest.mark.parametrize('connected', ['ready', 'effect'], indirect=True)
@pytest.mark.parametrize('case', ['normal', 'pending', 'missing', 'tamper'])
def test_saved_reasons_match_live(session: Any, connected: Any, case: str) -> None:
    s = session.value
    s.evaluation_flags = connected.flags
    steps = tuple(s.witness.pair(35370))
    first = dict(scope={k: steps[0][k] for k in Q.SCOPE_KEYS}, journal_token=steps[0]['token'],
                 pending_occurrences=[], arrival_ledger=dict(arrivals=[], applied=[], acknowledgements=[]))
    pending = ()
    if case == 'pending':
        s.mode.native.pending.append(object())
        pending = ('q',)
    elif case == 'missing':
        s.mode = None
        pending = None
    flags = deepcopy(connected.flags.latest)
    if case == 'tamper':
        flags['1P']['state'] = 'chain'
        with pytest.raises(ValueError, match='flag_returned'):
            Q.reasons(s.state['provisional_context_observer'].rows[-1], flags, steps, first, pending)
    else:
        assert Q.reasons(s.state['provisional_context_observer'].rows[-1], flags, steps, first, pending) == E.reasons(s, 35370)
