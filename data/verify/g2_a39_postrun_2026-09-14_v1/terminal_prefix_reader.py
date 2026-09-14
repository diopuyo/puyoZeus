"""update完了後の原Jと借用Liveを突合する。元frameの再利用や台帳更新はしない。"""
from dataclasses import asdict
import json
from typing import Any

MODE_KEY = 'probabilistic_tracking_mode'
SIDE = '1P'
FPS = 60


def verify_clock(live: Any, mode: Any, session: Any, step: dict) -> None:
    """認証済み同世代終了だけを分離し、元時計・所有条件は通常経路に維持する。"""
    require, ledger = live.module.B.require, mode.arrival_ledger
    capture = getattr(mode, 'arrival_capture', None)
    receipt = getattr(capture, 'terminal_receipt', None)
    if receipt is None:
        require(ledger.clock == step['frame_idx'] <= ledger.deadline, 'projection_live_finished_clock')
        return
    require(capture.mode is mode and capture.recovery is mode.connection.recovery
        and capture.journal is session.journal, 'projection_terminal_capture_owner')
    require(capture.frozen_ledger is ledger and receipt['last_live_ledger'] == asdict(ledger)
        and receipt['kind'] == 'arrival_scope_terminal/v1', 'projection_terminal_frozen_ledger')
    require(ledger.clock < receipt['frame'] <= capture.terminal_last == step['frame_idx'] <= ledger.deadline
        and step['time_sec'] == step['frame_idx'] / FPS, 'projection_terminal_clock')
    require(capture.pending is None and capture.error is None and not session.journal.errors
        and session.journal.active is None and capture.terminal_stream is not None
        and not capture.terminal_stream.closed, 'projection_terminal_incomplete')
    require(step['status'] == 'returned' and step['exception'] is None
        and step['generation'] == step['generation_after']
        and step['returned']['state'] == 'MENU'
        and step['returned']['active_origin'] is None, 'projection_terminal_original_step')
    require(capture.frozen_current is mode.connection.registry.current(mode.connection.binding),
        'projection_terminal_current_changed')


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
    verify_clock(live, mode, session, step)
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
    if getattr(getattr(mode, 'arrival_capture', None), 'terminal_receipt', None) is not None:
        return None, step, 'match_ended_scope_frozen'
    if not lane.observations:
        return None, step, 'prefix_lane_observation_not_available'
    if len(lane.families) != 1 or lane.families[0].phase != adapter.N.P.PREPOP:
        return None, step, 'single_prepop_family_not_available'
    if lane.families[0].value.frame != lane.observations[-1]['frame']:
        raise ValueError('projection_live_family_observation')
    return lane, step, None
