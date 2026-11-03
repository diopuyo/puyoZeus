"""B3の完了時取り込みは判定値を保持し、EMAを同一通知で二重更新しない。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.test_b3_async_display import wrapper, result
from src.phase_j.live_b3 import LiveB3
from src.b3_live_display import B3Display


def event_wrapper() -> LiveB3:
    live = wrapper()
    live.completion_driven = True
    live.async_failed = False
    live.physical, live.physical_count = set(), 0
    live.latest_frame = dict(frame=60, t_sec=2., game_idx=1, match_active=True)
    live.latest_overlay = SimpleNamespace(b3_reference=dict(source=None, probability=None, physical=[]))
    return live


def test_completion_applies_without_next_observation() -> None:
    live = event_wrapper()
    live.async_engine = SimpleNamespace(poll=lambda: [result()])
    assert live.receive_completed()
    assert live.display.raw == .8
    output = {}
    live.decorate(output)
    assert output['raw_probability'] == .8
    assert output['judgment_id'][:2] == (2, 30)
    assert not live.receive_completed()


def test_invalid_completion_is_not_a_new_judgment() -> None:
    live = event_wrapper()
    live.async_engine = SimpleNamespace(poll=lambda: [result(epoch=1)])
    assert not live.receive_completed()
    assert live.display.raw is None


@pytest.mark.parametrize('fast, expected', [(False, .35), (True, .5)])
def test_completion_replaces_same_tick_instead_of_adding_ema_step(fast: bool, expected: float) -> None:
    live = event_wrapper()
    live.display = B3Display(fast_ema=fast)
    live.display.boundary(1)
    live.display.prediction(.2, 0, 1.)
    live._display(dict(t_sec=1., match_active=True), live.latest_overlay)
    live._display(live.latest_frame, live.latest_overlay)
    live.async_engine = SimpleNamespace(poll=lambda: [result(frame=60, t_sec=2.)])
    assert live.receive_completed()
    assert live.display.shown[1] == pytest.approx(expected)
    live._display(live.latest_frame, live.latest_overlay)
    assert live.display.shown[1] == pytest.approx(expected)


def test_completion_cannot_override_reference_death() -> None:
    live = event_wrapper()
    live.latest_overlay.b3_reference.update(source='confirmed_death', probability=1.)
    live.async_engine = SimpleNamespace(poll=lambda: [result()])
    assert live.receive_completed()
    assert live.display.raw == 1.
    assert live.display.probability == .8


def test_event_update_does_not_poll_twice() -> None:
    live = event_wrapper()
    calls = []
    live.async_engine = SimpleNamespace(submit=lambda *args, **kwargs: calls.append(args))
    live._async_update(live.latest_frame, (), (), SimpleNamespace(captured_at=0.))
    assert len(calls) == 1


def test_unknown_judgment_in_new_game_is_a_new_status() -> None:
    live = event_wrapper()
    first: dict = {}
    second: dict = {}
    live.decorate(first)
    live.display.boundary(2)
    live.decorate(second)
    assert first['judgment_id'] != second['judgment_id']
