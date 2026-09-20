"""正常Bの二M1/親保存/全schedule/終了を検査。外側waitや物理GTとは区別する。"""
from pathlib import Path
from types import FunctionType
from typing import Any
import original_saved as V
import original_schedule_saved as S
import normal_settings as K


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('normal_saved:' + reason)


def schedule(output: Path) -> dict:
    """元検査codeの専用globalsへ範囲を与え、元moduleのENDを変更しない。"""
    namespace = dict(vars(S), EARLIEST=K.EARLIEST, END=K.LAST)
    namespace['order'] = FunctionType(S.order.__code__, namespace)
    verify = FunctionType(S.verify.__code__, namespace)
    return verify(output)


def session(output: Path) -> dict:
    saved = V.read(output / 'BELIEF_M1_SESSION.json')
    require(all(saved[key] is True for key in ('restored', 'observer_closed', 'witness_closed',
                                              'evaluation_flags_closed')), 'session_cleanup')
    require(all(saved[key] is None for key in ('error', 'session_error', 'witness_error',
                                              'evaluation_flags_error'))
            and saved['observer_errors'] == [None, None], 'session_errors')
    require(saved['normal_initialization_start'] == K.INITIALIZATION_START
            and saved['quality_gate_clear'] is False, 'session_scope')
    require({mode['side'] for mode in saved['modes']} == set(K.SIDES), 'mode_sides')
    for mode in saved['modes']:
        initial = mode['initial']['state']
        require(mode['closed'] is True and mode['error'] is None and initial['scope'][-1] == mode['side']
                and K.INITIALIZATION_START <= initial['frame'] <= K.LAST
                and initial['deadline'] == K.LAST, 'mode_scope_or_failure')
    owner = V.read(output / 'NORMAL_OWNER_STATUS.json')
    require(owner['closed'] is True and owner['error'] is None and owner['original_body'] is None
            and owner['start'] == K.INITIALIZATION_START and owner['deadline'] == K.LAST
            and owner['reset_claimed'] is False and owner['quality_gate_clear'] is False, 'owner_cleanup')
    require(not (output / 'M1_SESSION_CLOSE_FAILURE.json').exists(), 'close_failure_present')
    return saved


def producer(capture: dict, output: Path, frame: int) -> None:
    require(capture['first_frame'] == K.FIRST and capture['last_frame'] == frame
            and capture['observed_count'] == (frame - K.FIRST) // K.STRIDE + 1
            and capture['identity']['run_id'] == str(output), 'producer_scope')


def verify(output: Path) -> dict:
    saved = session(output)
    order = schedule(output)
    frames = tuple(order['frames'])
    names = tuple(f'BELIEF_M1_{frame}.json' for frame in frames)
    packets = V.packets(output, 'JOINT_EVENTS.jsonl', names, frames)
    require(len(packets) == len(K.EARLIEST) and all(packet['result']['supported'] is True
            for packet in packets), 'both_M1_supported')
    for index, frame in enumerate(frames):
        ticket = V.read(output / f'JOINT_EVENTS.jsonl.request{index}.json')
        producer(ticket['producer'], output, frame)
    producer(V.read(output / 'JOINT_PRODUCER_CAPTURE.json'), output, K.LAST)
    return dict(schema='normal-m1-window-child-review/v1', frames=list(frames),
        input_bounds=[K.FIRST, K.LAST], initialization_start=K.INITIALIZATION_START,
        parent_packets_and_child_exit_verified=True, normal_m1_window_verified=True,
        raw_J_qualification_replay_verified=False, physical_ground_truth_verified=False,
        outer_wait_verified=False, quality_gate_clear=False, production_permission=False)
