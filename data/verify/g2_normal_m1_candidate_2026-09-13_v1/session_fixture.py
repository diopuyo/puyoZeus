"""元J complete/emitを実行するCPU入力。画像/SM/採録前PBは人工と明示する。"""
from dataclasses import asdict, dataclass
import io
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import test_normal_basis as F

ROOT = Path(__file__).resolve().parent


@dataclass
class Generation:
    side: str
    reset_epoch: int = 0
    action_revision: int = 0
    identity_scope: str = 'software_observation_only_not_physical_identity'


def create(parts: Any, graph: Any, output: Path) -> Any:
    f = F.fixture(parts)
    del f.factory._g2_probabilistic_scope_registry
    source = ROOT.parent / 'g2_atomic_journal_capture_2026-09-09_v1/observer.py'
    original = parts.loader('_normal_session_original_journal', source)
    journal = original.Recorder.__new__(original.Recorder)
    journal.stream, journal.count, journal.errors = io.StringIO(), 0, []
    journal.steps, journal.expected = 0, []
    journal.active, journal.pipe, journal.codes, journal.closed = None, f.pipe, {step.__code__}, False
    journal.tracker = N(generation=lambda side: Generation(side))
    journal.scope = lambda pipe, side, frame, clock: dict(source_id='normal', run_id='fixture',
        side=side, frame_idx=frame, time_sec=clock, pipe_object_id=id(pipe), generation=asdict(Generation(side)))
    journal.epoch = lambda *args: 0
    f.journal, f.original = journal, original
    f.state = dict(private_suffix_factory=f.factory, output=output,
        hidden_probability_observer=N(active=None, failures=[], rows=[]),
        postcommit_current_receiver=N(rec=N(side_value=lambda result: result.typed)),
        provisional_context_observer=N(active=None, errors=[], rows=[], installed=True, closed=False,
                                      source_id='normal', run_id='fixture'))
    f.factory.provider = N(journal=journal, no_origin=lambda *args: True,
        raw=lambda pipe, side, view: (parts.original.mode.B.grid(getattr(pipe, '_sm_' + side.lower()).context.confirmed_board), None))
    for side in ('1P', '2P'):
        setattr(f.pipe, '_landing_grace_' + side.lower(), None)
    return f


def step(f: Any, side: str, frame_idx: int) -> None:
    time_sec = frame_idx / 60
    sm = getattr(f.pipe, '_sm_' + side.lower())
    sm.context.frame_idx = frame_idx
    board = sm.context.confirmed_board
    grid = f.original.board(board)
    cells = [[[[color, 1.0]] for color in row] for row in grid['grid']]
    probability = dict(present=True, type_valid=True, errors=[], cells=cells)
    scope = f.journal.scope(f.pipe, side, frame_idx, time_sec)
    f.state['hidden_probability_observer'].rows = [scope | dict(raw=grid, confirmed=grid,
        probability=probability, hold_reasons=[], instrumentation_errors=[])]
    signals = N(is_match_active=True, effect_gate_window_active=False, cnn_board=board)
    typed = dict(state_value='stable', confirmed=grid, probability=probability)
    result = N(confirmed_board=board, inferred_board=board, state=N(name='STABLE', value='stable'), typed=typed)
    item = dict(frame=sys._getframe(), pipe=f.pipe, scope=scope, epoch=0,
                token='step:' + str(f.journal.count), events=[], return_line=None)
    f.journal.complete_step(item, result, None, sys.getprofile())
    f.journal.steps += 1
    f.journal.expected.append((frame_idx, side))
    assert item['frame'] is None


def update(f: Any, session: Any, frame: int) -> None:
    for side in ('1P', '2P'):
        step(f, side, frame)
    f.state['provisional_context_observer'].rows.append(context_row(f, frame))
    session.completed(frame)


def context_row(f: Any, frame: int) -> dict:
    """純粋契約に必要な人工context。実観測票/モデル入力全体の代用ではない。"""
    generation = dict(observed=True, reason='actual_software_generation_not_physical_game',
                      value={side: asdict(Generation(side)) for side in ('1P', '2P')})
    sides = {}
    for side in ('1P', '2P'):
        board = getattr(f.pipe, '_sm_' + side.lower()).context.confirmed_board
        sides[side] = dict(hold_reasons=[], before_hold=dict(confirmed=f.original.board(board),
            probability=dict(present=True, type_valid=True, errors=[])))
    update = dict(returned=True, exception=None, call_index=frame, pipe_index=0,
                  frame_idx=frame, returned_frame_idx=frame, time_sec=frame / 60, returned_time_sec=frame / 60)
    for key, value in (('is_match_active', True), ('match_end_locked', False), ('post_match_lockdown_active', False)):
        update[key], update[key + '_observed'] = value, True
    return dict(schema_version='provisional-current-context/v1', source_id='normal', run_id='fixture',
        frame_idx=frame, time_sec=frame / 60, available_frame=frame, capture_token='fixture:' + str(frame),
        sides=sides, failures=[], capture_status='CAPTURED', hold_reasons=[], update=update,
        upstream_failures={key: [] for key in ('hidden_probability_observer', 'current_scope_sink',
                                              'provisional_current_connection')},
        game=dict(observed=False, value=None, reason='no_game_identity_on_this_pipeline_boundary'),
        generation=dict(before=generation, after=generation))
