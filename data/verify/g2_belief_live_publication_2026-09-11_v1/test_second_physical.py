"""2P登録→人工正常手→原物理候補→Registry一回更新を検査する。"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
from test_second_basis import saved,live,context,policy,observed
from test_second_tracking import advance
import second_physical as P
import second_tracking as T
import belief as B


@pytest.fixture(scope='module')
def physical() -> Any:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0,str(root/'g2_belief_hidden_landing_2026-09-11_v1'))
    sys.path.insert(0,str(root/'g2_reset_settled_basis_gate_2026-09-11_v1'))
    sys.path.insert(0,str(root/'g2_basis_cascade_candidate_2026-09-11_v1'))
    path = root/'g2_basis_cascade_candidate_2026-09-11_v1/mode.py'
    spec = importlib.util.spec_from_file_location('_second_physical_candidate',path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def step(o: Any, board: Any, token: str, events: list[Any], stable: bool = True) -> None:
    journal,pipe = o.witness.journal,o.pipe
    side,sm = '2P',pipe._sm_2p
    frame_idx,time_sec = sm.context.frame_idx,sm.context.frame_idx/60
    sm.context.confirmed_board = board
    signals = N(is_match_active=True,effect_gate_window_active=False,cnn_board=board)
    result = N(state=N(value='stable' if stable else 'tsumo_fall',name='STABLE'),confirmed_board=board,inferred_board=board)
    state = o.evidence.state
    typed = deepcopy(state['postcommit_current_receiver'].rec.side_value(result))
    typed['confirmed']['grid'] = board.to_dict()['grid']
    pb = state['hidden_probability_observer'].rows[-1]
    pb['confirmed'] = typed['confirmed']
    state['postcommit_current_receiver'].rec.side_value = lambda result: deepcopy(typed)
    item = dict(frame=inspect.currentframe(),pipe=pipe,scope=journal.scope(),epoch=0,
                token=token,events=events,return_line=None)
    journal.complete_step(item,result,None,sys.getprofile())
    assert item['frame'] is None


def setup(o: Any, policy: Any, physical: Any) -> Any:
    provider = N(journal=o.witness.journal,no_origin=lambda *args:True,
        raw=lambda *args:(B.grid(o.pipe._sm_2p.context.confirmed_board),{}))
    mode = P.Mode(o.evidence,o.witness,o.pipe,o.registry,o.factory,35410,policy,physical,provider)
    o.witness.journal.codes.add(step.__code__)
    return mode


def consumption(mode: Any) -> list[Any]:
    s = mode.connection.binding.scope
    token = f'{s[0]}:{s[1]}:reset:{s[2]}:2P:enqueue:fixture'
    return [dict(stage='fifo_before',accounting={'pending_tsumo':[[5,5]]},fifo_occurrence_tokens=[token]),
        dict(stage='fifo_after',accounting={'pending_tsumo':[]},committed=[5,5],enqueue_occurrence_token=token)]


@pytest.mark.parametrize('delay',[False,True])
def test_normal_hand_once(observed: Any,policy: Any,physical: Any,delay: bool) -> None:
    o = observed
    mode = setup(o,policy,physical)
    before = o.registry.current(mode.connection.binding)
    board = o.pipe._sm_2p.context.confirmed_board.copy()
    board.set(12,0,5)
    board.set(11,0,5)
    advance(o)
    with ExitStack() as stack:
        T.install(stack,mode)
        step(o,board,'step:601',consumption(mode),not delay)
        if delay:
            assert o.registry.current(mode.connection.binding) is before and len(mode.native.pending)==1
            advance(o)
            step(o,board,'step:603',[])
        assert len(mode.applied)==1 and not mode.native.pending
        after = o.registry.current(mode.connection.binding)
        assert after.frame==o.pipe._sm_2p.context.frame_idx and len(after.tokens)==1
        assert all(w.grid[12][0]==w.grid[11][0]==5 for w in after.worlds)
        assert not mode.latest['quality_gate_clear']
        advance(o)
        step(o,board,'step:605',[])
        assert len(mode.applied)==1 and o.registry.current(mode.connection.binding) is after
