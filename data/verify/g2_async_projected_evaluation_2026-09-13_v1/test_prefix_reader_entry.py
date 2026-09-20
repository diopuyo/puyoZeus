"""新reader入口から元complete_step/Witnessを通す。予測陽性は未検収。"""
from contextlib import ExitStack
from types import SimpleNamespace as N
import pytest
import journal_pair_reader as J
import prefix_live_reader as R
import prefix_owner_lease as L
from test_journal_pair_reader import setup, complete, FRAME, W


def test_original_j_pair_before_prefix_initialization() -> None:
    journal, pipe, _ = setup()
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        session = N(witness=witness, journal=journal, pipe=pipe, state={}, restored=False, error=None)
        lease = L.Lease()
        complete(journal, pipe, '1P', 0)
        assert R.read(session, lease, J, FRAME)[2] == 'both_side_J_not_available'
        complete(journal, pipe, '2P', 1)
        before = journal.stream.getvalue()
        assert R.read(session, lease, J, FRAME)[2] == 'prefix_live_not_installed'
        lease.live, lease.adapter = N(), N(Live=N)
        assert R.read(session, lease, J, FRAME)[2] == 'prefix_mode_not_initialized'
        assert journal.stream.getvalue() == before
        journal.history.frame += 2
        with pytest.raises(ValueError, match='projected_J_live_clock'):
            R.read(session, lease, J, FRAME)


@pytest.mark.parametrize('fault', ['lease_closed', 'session_closed', 'session_error'])
def test_end_before_initialization_is_not_a_successful_hold(fault: str) -> None:
    journal, pipe, _ = setup()
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        session = N(witness=witness, journal=journal, pipe=pipe, state={}, restored=False, error=None)
        lease = L.Lease()
        if fault == 'lease_closed': lease.close()
        elif fault == 'session_closed': session.restored = True
        else: session.error = 'original_failure'
        with pytest.raises(ValueError, match='closed'):
            R.read(session, lease, J, FRAME)
