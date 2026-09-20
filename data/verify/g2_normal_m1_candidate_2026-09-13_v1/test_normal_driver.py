"""元driver呼出codeの正常READY/生成frame非採録/例外停止。Bridge/Sessionは人工。"""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import test_normal_basis as F
parts = F.parts
ROOT = Path(__file__).resolve().parent
FRAMES = (33726, 33766)


def loaded(parts: Any) -> tuple:
    root = ROOT.parent / 'g2_live_probability_context_2026-09-12_v1'
    original = parts.loader('_normal_test_original_driver', root / 'whole_session_driver.py')
    normal = parts.loader('_normal_test_driver', ROOT / 'normal_driver.py')
    return original, normal.driver_class(original)


def setup(driver_type: type) -> tuple:
    observed = []
    state = dict(private_suffix_factory=N(), hidden_probability_observer=N(),
                 provisional_context_observer=N(active=None, errors=[], rows=[dict(frame_idx=29052)]))
    def create(stack: Any, context: Any) -> Any:
        assert context['factory'] is state['private_suffix_factory']
        session = N(frames=FRAMES, error=None, restored=False, completed=observed.append, closed=False)
        stack.callback(setattr, session, 'closed', True)
        return session
    driver = driver_type(state, create, FRAMES)
    bridge = N(state=state, capture=N(last=29052), completed_frames=[29052], lifetime=ExitStack(),
               initial=dict(pipeline=N()), consumer=driver, frames=tuple(range(29052, 34292, 2)))
    return driver, bridge, observed


def test_normal_driver_original_code_and_missing_reset(parts: Any) -> None:
    original, normal = loaded(parts)
    assert normal.original_call.__code__ is original.Driver.__call__.__code__
    driver, bridge, observed = setup(normal)
    driver(bridge, 29052)
    assert driver.session is not None and not observed
    assert 'probabilistic_basis_connection' not in driver.state
    bridge.capture.last = 29054
    bridge.completed_frames.append(29054)
    driver.state['provisional_context_observer'].rows.append(dict(frame_idx=29054))
    driver(bridge, 29054)
    assert observed == [29054]
    bridge.lifetime.__exit__(LookupError, LookupError('original_body'), None)
    assert driver.closed and driver.stopped and driver.session.closed and driver.session.restored
    assert bridge.consumer is original.forbidden_collect


def test_normal_driver_missing_real_ready_rejected(parts: Any) -> None:
    original, normal = loaded(parts)
    driver, bridge, observed = setup(normal)
    del driver.state['hidden_probability_observer']
    driver(bridge, 29052)
    assert driver.session is None
    bridge.capture.last = 33726
    bridge.completed_frames.append(33726)
    with pytest.raises(ValueError, match='session_not_ready_at_selected_frame'):
        driver(bridge, 33726)
    bridge.lifetime.__exit__(ValueError, driver.error, None)
    assert driver.closed and driver.session is None


def test_normal_driver_rejects_wrong_end_before_create(parts: Any) -> None:
    original, normal = loaded(parts)
    driver, bridge, observed = setup(normal)
    bridge.frames = tuple(range(29052, 34294, 2))
    with pytest.raises(ValueError, match='normal_run_bounds'):
        driver(bridge, 29052)
    assert driver.session is None and not observed
