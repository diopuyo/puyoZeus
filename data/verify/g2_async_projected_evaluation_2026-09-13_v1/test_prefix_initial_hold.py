"""240指摘：実Live.stable資格保存→実Live.progressのLane生成直後を再現する。"""
from contextlib import ExitStack
from dataclasses import replace
from io import StringIO
import json
import pytest
from types import SimpleNamespace as N
from typing import Any
import prefix_live_adapter as A
import prefix_live_reader as R
import prefix_owner_lease as L
import test_stable_evidence as E

source = E.source


def test_original_lane_initialization_holds_until_first_observation(source: Any, tmp_path: Any) -> None:
    parts, row, _, _, _ = source
    mode, item, qualification = E.synthetic_owner(source, tmp_path)
    mode.basis_cascade_closed, mode.error, mode.stream = True, None, StringIO()
    mode.origins, mode.origin_ids = {}, {}
    mode.state[R.MODE_KEY] = mode
    item['events'] = []
    c, ledger = mode.connection, mode.arrival_ledger
    current = replace(parts.mode.BASE.V1.S.decode(row['transition']['state']), scope=ledger.scope)
    c.registry, c.recovery.error = N(current=lambda binding: current), None
    c.recovery.pipe._sm_1p = item['frame'].f_locals['sm']
    result = N(state=N(value='stable'),
        confirmed_board=parts.mode.B.Board.from_dict({'grid': qualification['returned']}))
    with ExitStack() as stack:
        live = A.Live(stack, parts.mode)
        def original(*args: Any) -> dict:
            _, reason = live.stable(parts.mode.Mode.stable, mode, item, result)
            assert reason is None
            return {'artificial_basis_transition_already_completed': True}
        def progress(*args: Any) -> dict:
            return live.progress(original, mode, item, result, {})
        live.observe(progress, mode, item, result, None)
        lane = live.lane(mode)
        assert lane is not None and not lane.observations and id(mode) in live.recorder.owners
        assert lane.families[0].phase == A.N.P.SETTLED
        lease = L.Lease()
        lease.live, lease.adapter = live, A
        session = N(state=mode.state, journal=c.recovery.journal, pipe=c.recovery.pipe,
            factory=c.recovery.factory, witness=None, restored=False, error=None)
        step = item['scope'] | dict(software_reset=item['epoch'], generation_after=item['scope']['generation'])
        reader = N(read_pair=lambda *args: N(holds=(), rows_json=json.dumps([step])))
        assert R.read(session, lease, reader, ledger.clock)[2] == 'prefix_lane_observation_not_available'
        assert (tmp_path / 'PREFIX_LANE_INITIAL.jsonl').is_file()
        lane.families = (replace(lane.families[0], prefix=1),)
        with pytest.raises(ValueError, match='unobserved_state_changed'):
            R.read(session, lease, reader, ledger.clock)
