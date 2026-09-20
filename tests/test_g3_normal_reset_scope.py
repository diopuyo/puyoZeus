"""元normal graph/原J completeを用いる。SM/PB/原scopeは人工初期条件。"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any
from types import SimpleNamespace as Box
import pytest
from scripts import g3_normal_reset_scope as N

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data/verify/g2_normal_m1_candidate_2026-09-13_v1'))
import test_normal_graph as T
import session_fixture as F

parts, graph = T.parts, T.graph
FRAME = 33722


def replace_owned(stack: Any, owner: Any, name: str, value: Any) -> None:
    """試験内の所有参照を元へ復元する。製品の所有機構は変更しない。"""
    previous = getattr(owner, name)
    stack.callback(setattr, owner, name, previous)
    setattr(owner, name, value)


@pytest.mark.parametrize('side', ['1P', '2P'])
@pytest.mark.parametrize('mode', ['normal', 'reset', 'registered', 'clock', 'reverse', 'old'])
def test_normal_observation_reset(parts: Any, graph: Any, tmp_path: Path, side: str, mode: str) -> None:
    """両sideと未登録HOLD、元PB正常、時計不正、原例外を分ける。"""
    f = F.create(parts, graph, tmp_path)
    observation = graph['second_observation']
    original, generation = observation.capture, dict(epoch=0)
    def scope(pipe: Any, which: str, frame: int, clock: float) -> dict:
        value = asdict(F.Generation(which, generation['epoch']))
        return dict(source_id='normal', run_id='fixture', side=which, frame_idx=frame,
                    time_sec=clock, pipe_object_id=id(pipe), generation=value)
    f.journal.scope = scope
    f.journal.tracker.generation = lambda which: F.Generation(which, generation['epoch'])
    expected_error = mode in ('clock', 'reverse', 'old')
    with ExitStack() as stack:
        if mode != 'old':
            receipt = N.install(stack, observation, replace_owned)
            assert receipt['dynamic_side'] and 'SIDE' not in N.R.INSERT.replace('SIDE', 'side')
        evidence = observation.install(stack, f.journal, f.state, side=side)
        if mode == 'registered':
            evidence.registered_scope = scope(f.pipe, side, FRAME, FRAME / 60)
        completed = f.journal.complete_step
        def resetting(item: Any, result: Any, error: Any, profile: Any) -> Any:
            if mode != 'normal':
                getattr(f.pipe, '_sm_' + side.lower()).context.frame_idx = 0
                generation['epoch'] = -1 if mode == 'reverse' else (0 if mode == 'clock' else 1)
            return completed(item, result, error, profile)
        replace_owned(stack, f.journal, 'complete_step', resetting)
        if expected_error:
            with pytest.raises((ValueError, RuntimeError)):
                F.step(f, side, FRAME)
        else:
            F.step(f, side, FRAME)
            assert evidence.latest['side'] == side if mode != 'normal' else evidence.latest['scope'][-1] == side
            assert (evidence.latest.get('hold_reason') == 'generation_changed') == (mode != 'normal')
            assert not evidence.latest['basis_registered']
    assert observation.capture is original and evidence.closed
    (tmp_path / 'J_CPU.jsonl').write_text(f.journal.stream.getvalue(), encoding='utf-8')


def test_normal_session_old_flags_counterexample(parts: Any, graph: Any, tmp_path: Path) -> None:
    """観測captureだけ直しても実Sessionの採録flagがSM_clockで停止する反例。"""
    f = F.create(parts, graph, tmp_path)
    contract = parts.loader('_g3_normal_session_contract', ROOT / 'data/verify/g2_model_process_bridge_2026-09-11_v1/pure_contract.py')
    epochs = {'1P': 0, '2P': 0}
    original_scope = f.journal.scope
    def scope(pipe: Any, side: str, frame: int, clock: float) -> dict:
        row = original_scope(pipe, side, frame, clock)
        row['generation'] = asdict(F.Generation(side, epochs[side]))
        return row
    f.journal.scope = scope
    f.journal.tracker.generation = lambda side: F.Generation(side, epochs[side])
    with pytest.raises(ValueError, match='SM_clock') as caught:
        with ExitStack() as stack:
            N.install(stack, graph['second_observation'], replace_owned)
            session = graph['normal_session'].Session(stack, dict(state=f.state, factory=f.factory, pipe=f.pipe),
                parts.policy, Box(Mode=parts.original.mode.BASE.Mode), contract, None, (33726, 33766))
            completed = f.journal.complete_step
            def resetting(item: Any, result: Any, error: Any, profile: Any) -> Any:
                side = item['scope']['side']
                getattr(f.pipe, '_sm_' + side.lower()).context.frame_idx = 0
                epochs[side] += 1
                return completed(item, result, error, profile)
            replace_owned(stack, f.journal, 'complete_step', resetting)
            F.update(f, session, FRAME)
    (tmp_path / 'SESSION_COUNTEREXAMPLE.json').write_text(json.dumps(dict(error=str(caught.value),
        saved_models=len(session.saved), owner_closed=session.owner.closed, artificial_sm=True)), encoding='utf-8')
    assert not session.saved and session.owner.closed


def test_normal_session_reset_hold(parts: Any, graph: Any, tmp_path: Path) -> None:
    """実Sessionと採録資格で両side保留を確認し、予定停止でも原票を残す。"""
    f = F.create(parts, graph, tmp_path)
    contract = parts.loader('_g3_normal_hold_contract', ROOT / 'data/verify/g2_model_process_bridge_2026-09-11_v1/pure_contract.py')
    epochs = {'1P': 0, '2P': 0}
    original_scope = f.journal.scope
    original_flags = graph['normal_session'].F.capture
    original_observation = graph['second_observation'].capture
    def scope(pipe: Any, side: str, frame: int, clock: float) -> dict:
        row = original_scope(pipe, side, frame, clock)
        row['generation'] = asdict(F.Generation(side, epochs[side]))
        return row
    f.journal.scope = scope
    f.journal.tracker.generation = lambda side: F.Generation(side, epochs[side])
    with pytest.raises(RuntimeError, match='CPU_PLANNED_STOP'):
        with ExitStack() as stack:
            N.install(stack, graph['second_observation'], replace_owned)
            N.install_flags(stack, graph['normal_session'].F, replace_owned)
            session = graph['normal_session'].Session(stack, dict(state=f.state, factory=f.factory, pipe=f.pipe),
                parts.policy, Box(Mode=parts.original.mode.BASE.Mode), contract, None, (33726, 33766))
            completed = f.journal.complete_step
            def resetting(item: Any, result: Any, error: Any, profile: Any) -> Any:
                side = item['scope']['side']
                getattr(f.pipe, '_sm_' + side.lower()).context.frame_idx = 0
                epochs[side] += 1
                return completed(item, result, error, profile)
            replace_owned(stack, f.journal, 'complete_step', resetting)
            F.update(f, session, 33724)
            F.update(f, session, 33726)
            assert not session.saved and not session.modes and len(session.holds) == 4
            for flag in session.evaluation_flags.latest.values():
                assert set(flag) == {'scope', 'token', 'software_reset', 'hold_reason'}
                assert flag['hold_reason'] == 'generation_changed'
                assert flag['scope']['generation']['reset_epoch'] == 1
            raise RuntimeError('CPU_PLANNED_STOP')
    rows = [json.loads(line) for line in (tmp_path / 'M1_CAPTURE_SCHEDULE.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and all(row['action'] != 'REQUEST' and not row['saved'] for row in rows)
    assert session.owner.closed and session.evaluation_flags.closed and session.stream.closed
    assert graph['normal_session'].F.capture is original_flags
    assert graph['second_observation'].capture is original_observation
    saved = json.loads((tmp_path / 'BELIEF_M1_SESSION.json').read_text())
    assert saved['session_error'] is None and saved['evaluation_flags_error'] is None


@pytest.mark.parametrize('side', ['1P', '2P'])
@pytest.mark.parametrize('mode', ['normal', 'reset', 'clock', 'reverse'])
def test_flags_reset_boundaries(parts: Any, graph: Any, tmp_path: Path, side: str, mode: str) -> None:
    """採録flagの正常対照と不正時計/逆行拒否を元J保存まで検査する。"""
    f = F.create(parts, graph, tmp_path)
    module = graph['normal_session'].F
    original, generation = module.capture, dict(epoch=0)
    prior_scope = f.journal.scope
    def scope(pipe: Any, which: str, frame: int, clock: float) -> dict:
        row = prior_scope(pipe, which, frame, clock)
        row['generation'] = asdict(F.Generation(which, generation['epoch']))
        return row
    f.journal.scope = scope
    with ExitStack() as stack:
        N.install_flags(stack, module, replace_owned)
        evidence = module.install(stack, f.journal)
        completed = f.journal.complete_step
        def resetting(item: Any, result: Any, error: Any, profile: Any) -> Any:
            if mode != 'normal':
                getattr(f.pipe, '_sm_' + side.lower()).context.frame_idx = 0
                generation['epoch'] = {'reset': 1, 'clock': 0, 'reverse': -1}[mode]
            return completed(item, result, error, profile)
        replace_owned(stack, f.journal, 'complete_step', resetting)
        if mode in ('clock', 'reverse'):
            with pytest.raises(ValueError):
                F.step(f, side, FRAME)
        else:
            F.step(f, side, FRAME)
            flag = evidence.latest[side]
            assert flag['hold_reason'] == ('generation_changed' if mode == 'reset' else None)
            assert flag['scope']['generation']['reset_epoch'] == 0
    assert evidence.closed and module.capture is original
    (tmp_path / 'J_FLAGS_CPU.jsonl').write_text(f.journal.stream.getvalue(), encoding='utf-8')
