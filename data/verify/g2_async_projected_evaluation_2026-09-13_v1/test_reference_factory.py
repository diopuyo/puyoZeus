"""原最終Session factoryで参考hookを選択し、HOLD保存と未到達拒否を検査する。"""
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
from test_fixed_origin_reference import source
import fixed_origin_reference as F


def construct(module: Any, arrival: Any, state: dict, output: Path) -> None:
    journal, pipe, _ = setup()
    first = arrival.Mode()
    first.connection = N(binding=N(scope=('artificial', '1P')))
    context = dict(pipe=pipe, factory=N(provider=N(journal=journal)),
                   state=dict(output=output, probabilistic_tracking_mode=first))
    targets = module.Session.__init__.__wrapped__.__globals__['S'].EARLIEST
    planned = RuntimeError('planned_original_factory_reference_HOLD_stop')
    with pytest.raises(RuntimeError) as caught:
        with ExitStack() as stack:
            value = module.Session(stack, context, None, N(Mode=arrival.Mode), None, None, targets)
            B.verify(state, require_instance=True, require_projected_input=True, require_fixed_reference=True)
            with pytest.raises(ValueError, match='sample_missing'): B.verify(state, require_reference_sample=True)
            capture = value.projected_origin_binding.capture
            capture.completed(journal.history.frame)
            after = inspect.getclosurevars(state['binding']['completed']).nonlocals['after_completed']
            after(capture, journal.history.frame)  # 原parent M1 bodyはこの人工Modeでは実行しない。
            assert state['binding']['reference_counts'] == dict(calls=1, REFERENCE_ONLY=0, HOLD=2, last_frame=100)
            with pytest.raises(ValueError, match='sample_missing'): B.verify(state, require_reference_sample=True)
            with pytest.raises(ValueError, match='duplicate_save'): after(capture, 100)
            raise planned
    assert caught.value is planned and value.witness.closed and capture.closed
    saved = [json.loads(row) for row in (output / 'PROJECTED_ORIGIN_CAPTURE.jsonl').read_text().splitlines()]
    assert len(saved) == 2 and saved[-1]['kind'] == 'fixed_origin_reference_pair/v1'
    assert json.loads((output / 'PROJECTED_ORIGIN_CAPTURE_CLOSE.json').read_text())['original_error'] == repr(planned)


def test_original_factory_selects_reference_and_preserves_cleanup(monkeypatch: Any, tmp_path: Path) -> None:
    original = B.install
    def install(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **kwargs, fixed_reference_enabled=True)
    monkeypatch.setattr(B, 'install', install)
    monkeypatch.setattr(ORIGINAL, 'construct_and_close', construct)
    ORIGINAL.test_projected_callback_binding_scope_close(monkeypatch, tmp_path)


def test_hook_positive_counts_and_disabled_gate(source: Any) -> None:
    capture, _, _, _, _ = source
    def load(alias: str, path: Path, injection: Any = None) -> Any:
        assert alias == '_async_live_fixed_origin_reference' and path == B.ROOT / 'fixed_origin_reference.py'
        assert injection['journal_pair_reader'] is F.R
        return F
    hook, counts = B.reference_hook(load, F.R, True)
    hook(capture, 100)
    assert counts == dict(calls=1, REFERENCE_ONLY=1, HOLD=1, last_frame=100)
    cls = type('ArtificialClassIdentity', (), {'__init__': lambda self: None, 'completed': lambda self: None})
    binding = dict(module=N(Session=cls), cls=cls, initialize=cls.__init__, completed=cls.completed,
        closed=False, created_frames=[100], projected_input_enabled=False, fixed_reference_enabled=True,
        reference_counts=counts)
    state = dict(patch=object(), binding=binding)
    B.verify(state, require_reference_sample=True)
    binding['fixed_reference_enabled'] = False
    with pytest.raises(ValueError, match='not_enabled'): B.verify(state, require_fixed_reference=True)
