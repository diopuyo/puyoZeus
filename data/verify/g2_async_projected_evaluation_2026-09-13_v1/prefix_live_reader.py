"""update完了後の原Jと借用Liveを突合する。元frameの再利用や台帳更新はしない。"""
import json
from typing import Any

MODE_KEY = 'probabilistic_tracking_mode'
SIDE = '1P'


def owner(live: Any, adapter: Any, mode: Any, session: Any, step: dict) -> Any:
    require = live.module.B.require
    c, ledger = mode.connection, mode.arrival_ledger
    journal, pipe = session.journal, session.pipe
    sm = getattr(pipe, '_sm_' + SIDE.lower())
    require(c.recovery.journal is journal and c.recovery.pipe is pipe
            and c.recovery.factory is session.factory, 'projection_live_recovery_owner')
    require(c.recovery.error is None and not live.active, 'projection_live_incomplete_or_failed')
    require(mode.state is session.state and live.recorder.module is live.module,
            'projection_live_state_owner')
    live.module.L.check(ledger)
    scope = ledger.scope
    require(c.binding.scope == scope and id(pipe) == scope[3] and id(sm) == scope[4],
            'projection_live_binding_owner')
    require((step['source_id'], step['run_id'], step['software_reset'], step['pipe_object_id']) == scope[:4]
            and step['generation_after']['reset_epoch'] == scope[5]
            and step['side'] == scope[-1] == SIDE, 'projection_live_scope')
    require(ledger.clock == step['frame_idx'] <= ledger.deadline, 'projection_live_finished_clock')
    actual = (mode, c, c.binding, c.recovery.factory, pipe, sm)
    prior = live.recorder.owners.get(id(mode))
    require(prior is not None and len(prior) == len(actual)
            and all(old is new for old, new in zip(prior, actual)), 'projection_live_qualified_owner')
    lane = live.lane(mode)
    require(type(lane) is adapter.N.Lane and lane.parts is live.parts, 'projection_live_lane_owner')
    require(c.registry.current(c.binding) is lane.current, 'projection_live_current_owner')
    if not lane.observations:
        require(lane.current is lane.initial and not lane.committed and len(lane.families) == 1
                and lane.families[0].prefix == 0 and lane.families[0].phase == adapter.N.P.SETTLED
                and lane.families[0].value is lane.initial
                and live.recorder.last[id(mode)][0] == lane.initial.frame <= step['frame_idx'],
                'projection_live_unobserved_state_changed')
        return lane
    last = lane.observations[-1]
    require(live.recorder.last.get(id(mode)) == (last['frame'], last['evidence_key'])
            and last['frame'] <= step['frame_idx'], 'projection_live_qualification_token')
    return lane


def read(session: Any, lease: Any, reader: Any, frame: int) -> tuple[Any, dict | None, str | None]:
    """資格不足は明示HOLD、所有破壊はstrict。返却参照は同じ呼出内だけで使用する。"""
    pair = reader.read_pair(session.witness, session.journal, session.pipe, frame)
    if session.restored or session.error is not None:
        raise ValueError('projection_live_session_closed_or_failed')
    if lease.closed:
        raise ValueError('prefix_lease_closed')
    if pair.holds:
        return None, None, ';'.join(pair.holds)
    if lease.live is None:
        return None, None, 'prefix_live_not_installed'
    live, adapter = lease.current()
    mode = session.state.get(MODE_KEY)
    if mode is None:
        return None, None, 'prefix_mode_not_initialized'
    lease.mode_open(mode)
    if mode.connection.binding is None or mode.arrival_ledger is None or live.lane(mode) is None:
        return None, None, 'prefix_lane_not_initialized'
    if id(mode) not in live.recorder.owners:
        return None, None, 'prefix_qualified_owner_not_available'
    step = next(row for row in json.loads(pair.rows_json) if row['side'] == SIDE)
    lane = owner(live, adapter, mode, session, step)
    if not lane.observations:
        return None, step, 'prefix_lane_observation_not_available'
    if len(lane.families) != 1 or lane.families[0].phase != adapter.N.P.PREPOP:
        return None, step, 'single_prepop_family_not_available'
    if lane.families[0].value.frame != lane.observations[-1]['frame']:
        raise ValueError('projection_live_family_observation')
    return lane, step, None
