"""Session接続の有限待機/原保存呼出契約を人工基底・人工資格・人工SAVEで検査する。"""
from __future__ import annotations

from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any

import pytest
import capture_schedule as POLICY
from test_evaluation_flags import journal

SAVE_ERROR = RuntimeError('人工SAVE障害')


def require(ok: bool, reason: str) -> None:
    if not ok: raise ValueError(reason)


def save(capture: Any, members: str, path: Path, seed: int) -> dict:
    before = capture()
    if members == 'raise': raise SAVE_ERROR
    assert before['frame'] == seed
    return dict(frame=seed + (2 if members == 'wrong_frame' else 0), artificial_save=True)


def write(path: Path, packet: dict) -> None:
    with path.open('x', encoding='utf-8') as stream: json.dump(packet, stream)


C, SAVE, S = N(require=require), N(run=save), N(encode=lambda v: v)


class BaseSession:
    def __init__(self, stack: Any, context: dict, policy: Any, physical: Any,
                 contract: Any, members: Any, frames: tuple) -> None:
        self.state, self.journal, self.members, self.frames = context['state'], context['journal'], members, frames
        self.saved, self.modes, self.holds, self.error, self.restored = [], [], [], None, True
        self.evidence, self.witness = N(closed=True, error=None), N(closed=True, error=None)
        stack.push(self.close)
    def completed(self, frame: int) -> None:
        raise AssertionError('人工基底completedは呼ばない')
    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        return False
    def basis(self) -> None:
        pass
    def capture(self) -> dict:
        return dict(frame=self.state['provisional_context_observer'].rows[-1]['frame_idx'])


@pytest.fixture
def module(monkeypatch: Any) -> Any:
    monkeypatch.setitem(sys.modules, 'old_session', N(Session=BaseSession))
    path = Path(__file__).with_name('scheduled_session.py')
    spec = importlib.util.spec_from_file_location('_scheduled_component_test', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def context(path: Path) -> dict:
    return dict(journal=journal(), state=dict(output=path,
        provisional_context_observer=N(active=None, errors=[], rows=[])))


def update(value: Any, frame: int) -> None:
    value.state['provisional_context_observer'].rows.append(dict(frame_idx=frame))
    value.completed(frame)


@pytest.mark.parametrize('ready', [35370, 35450])
def test_delayed_two_saves_and_actual_end(module: Any, monkeypatch: Any, tmp_path: Path, ready: int) -> None:
    monkeypatch.setattr(module.E, 'reasons', lambda session, frame: ('2P:not_stable',) if frame < ready else ())
    with ExitStack() as stack:
        value = module.Session(stack, context(tmp_path), None, None, None, 'ok', POLICY.EARLIEST)
        for frame in range(35368, POLICY.END + 1, 2): update(value, frame)
    packet = json.loads((tmp_path / 'BELIEF_M1_SESSION.json').read_bytes())
    assert [r['frame'] for r in packet['saved']] == [ready, ready + POLICY.MIN_GAP]
    assert packet['schedule']['last'] == POLICY.END and packet['evaluation_flags_closed']
    assert packet['original_targets'] == list(POLICY.EARLIEST)


def test_unqualified_end_fails_after_preserving_status(module: Any, monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(module.E, 'reasons', lambda session, frame: ('2P:not_stable',))
    with pytest.raises(ValueError, match='incomplete_coverage'):
        with ExitStack() as stack:
            value = module.Session(stack, context(tmp_path), None, None, None, 'ok', POLICY.EARLIEST)
            for frame in range(35368, POLICY.END + 1, 2): update(value, frame)
    packet = json.loads((tmp_path / 'BELIEF_M1_SESSION.json').read_bytes())
    assert not packet['saved'] and packet['schedule']['last'] == POLICY.END
    assert 'incomplete_coverage' in json.loads((tmp_path / 'M1_SESSION_CLOSE_FAILURE.json').read_bytes())['error']


@pytest.mark.parametrize('failure', ['raise', 'wrong_frame'])
def test_save_failure_never_acknowledged(module: Any, monkeypatch: Any, tmp_path: Path, failure: str) -> None:
    monkeypatch.setattr(module.E, 'reasons', lambda session, frame: ())
    with pytest.raises((RuntimeError, ValueError)) as caught:
        with ExitStack() as stack:
            value = module.Session(stack, context(tmp_path), None, None, None, failure, POLICY.EARLIEST)
            update(value, 35368)
            update(value, 35370)
    if failure == 'raise': assert caught.value is SAVE_ERROR
    assert value.saved == [] and value.schedule.pending == 35370
    rows = [json.loads(line) for line in (tmp_path / 'M1_CAPTURE_SCHEDULE.jsonl').read_text().splitlines()]
    assert not rows[-1]['saved'] and rows[-1]['error']


def test_registry_failure_still_has_diagnostic(module: Any, tmp_path: Path) -> None:
    error = RuntimeError('原Registry終了障害')
    def current(binding: Any) -> None:
        raise error
    with pytest.raises(RuntimeError) as caught:
        with ExitStack() as stack:
            value = module.Session(stack, context(tmp_path), None, None, None, 'ok', POLICY.EARLIEST)
            value.modes = [N(initial_receipt={}, retired_receipt=None, applied=[],
                             connection=N(binding=None, registry=N(current=current)))]
    assert caught.value is error
    packet = json.loads((tmp_path / 'M1_SESSION_CLOSE_FAILURE.json').read_bytes())
    assert packet['error'] == repr(error) and not packet['quality_gate_clear']
