"""原最終factoryのroot入力選択と未到達拒否。prime/Modeは人工、親M1 bodyは未実行。"""
from contextlib import ExitStack
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import session_runtime_binding as B
import test_projected_binding_scope as ORIGINAL
from test_journal_pair_reader import setup


def construct(module: Any, arrival: Any, state: dict, output: Path) -> None:
    journal, pipe, _ = setup()
    first = arrival.Mode()
    first.connection = N(binding=N(scope=('artificial', '1P')))
    context = dict(pipe=pipe, factory=N(provider=N(journal=journal)),
                   state=dict(output=output, probabilistic_tracking_mode=first))
    targets = module.Session.__init__.__wrapped__.__globals__['S'].EARLIEST
    planned = RuntimeError('planned_factory_root_input_HOLD_stop')
    with pytest.raises(RuntimeError) as caught:
        with ExitStack() as stack:
            value = module.Session(stack, context, None, N(Mode=arrival.Mode), None, None, targets)
            B.verify(state, require_instance=True, require_root_probability=True)
            with pytest.raises(ValueError, match='sample_missing'): B.verify(state, require_root_probability_sample=True)
            capture = value.projected_origin_binding.capture
            capture.completed(journal.history.frame)
            after = inspect.getclosurevars(state['binding']['completed']).nonlocals['after_completed']
            after(capture, journal.history.frame)
            assert state['binding']['root_probability_counts'] == dict(calls=1, READY=0, HOLD=1, last_frame=100)
            with pytest.raises(ValueError, match='sample_missing'): B.verify(state, require_root_probability_sample=True)
            raise planned
    assert caught.value is planned and capture.closed and value.witness.closed
    saved = [json.loads(line) for line in (output / 'PROJECTED_ORIGIN_CAPTURE.jsonl').read_text().splitlines()]
    assert saved[-1]['kind'] == 'conditional_fixed_root_input/v1' and saved[-1]['status'] == 'HOLD'
    assert len(saved) == 3


def test_original_factory_selects_root_hook_and_releases_aliases(monkeypatch: Any, tmp_path: Path) -> None:
    history = N()
    calls = []
    def prime(capture: Any, frame: int) -> None:
        calls.append(frame)
        history.journal, history.pipe, history.last_frame = capture.journal, capture.pipe, frame
        history.closed, history.sealed, history.transferred, history.error = False, True, True, None
        history.probability = N(closed=True)
        history.probability_inputs = lambda: ()
    history.prime = prime  # 人工primeは状態外殻だけ。実履歴移管の合格には使わない。
    original = B.install
    def install(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **kwargs, fixed_reference_enabled=True, root_probability_enabled=True,
                        history_getter=lambda: history)
    monkeypatch.setattr(B, 'install', install)
    monkeypatch.setattr(ORIGINAL, 'construct_and_close', construct)
    ORIGINAL.test_projected_callback_binding_scope_close(monkeypatch, tmp_path)
    assert calls == [100]
