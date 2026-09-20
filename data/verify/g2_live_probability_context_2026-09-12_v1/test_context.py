"""合成順序・元例外保持の製造単体試験。実J/実動画の到達証拠ではない。"""
from __future__ import annotations
from contextlib import ExitStack
from types import SimpleNamespace as N
from typing import Any
import pytest
import context as C

I = N(DEADLINE=114)


class Recovery:
    pass


def fixture_parts(trace: list, failure: str | None = None) -> tuple[type, C.Dependencies]:
    class Parent:
        def __init__(self, stack: Any) -> None:
            self.stack, self.rows, self.error = stack, [], None
            self.state = dict(live_history_sink=object(), repeat_scope_guard=N(reset_lease=object()))
            self.recovery = None
            self.pipe = N(_enable_match_transition_debounce=True)

        def perform(self, advisory: Any, frame: int, clock: float) -> None:
            trace.append('original')
            self.recovery = Recovery()

    def part(name: str) -> Any:
        def install(stack: Any, *args: Any) -> Any:
            trace.append(name)
            if name == failure:
                raise RuntimeError('original_failure:' + name)
            stack.callback(trace.append, 'close:' + name)
            return N(name=name)
        return N(install=install)

    def wrapper(name: str) -> Any:
        def derived(parent: type, *args: Any) -> type:
            class Derived(parent):
                def perform(self, *args: Any) -> None:
                    trace.append('enter:' + name)
                    super().perform(*args)
                    trace.append('exit:' + name)
            return Derived
        return N(derived=derived)

    parts = N(**{n: part(n) for n in ('actual', 'binding', 'mode', 'lease')})
    deps = C.Dependencies(lambda: parts, part('quarantine'), lambda: part('votes'),
        *(wrapper(n) for n in ('side', 'occurrence', 'baseline', 'snapshot', 'anchor')))
    return Parent, deps


def test_exact_compose_order_and_original_deadline() -> None:
    trace: list[str] = []
    parent, deps = fixture_parts(trace)
    cls = C.compose(parent, deps, end_frame=200)
    with ExitStack() as stack:
        value = cls(stack)
        value.perform(None, 100, 100 / 60)
        assert value.state[C.KEY]['acquisition_deadline'] == I.DEADLINE
        assert trace == ['enter:anchor', 'enter:snapshot', 'enter:baseline', 'enter:occurrence',
                         'enter:side', 'original', 'quarantine', 'actual', 'binding', 'mode', 'lease', 'votes',
                         'exit:side', 'exit:occurrence', 'exit:baseline', 'exit:snapshot', 'exit:anchor']
    assert trace[-6:] == ['close:votes', 'close:lease', 'close:mode', 'close:binding', 'close:actual', 'close:quarantine']


@pytest.mark.parametrize('failure', ['quarantine', 'actual', 'binding', 'mode', 'lease', 'votes'])
def test_original_error_preserved(failure: str) -> None:
    trace: list[str] = []
    parent, deps = fixture_parts(trace, failure)
    with ExitStack() as stack:
        value = C.compose(parent, deps, end_frame=200)(stack)
        with pytest.raises(RuntimeError, match='original_failure:' + failure):
            value.perform(None, 100, 100 / 60)
        assert failure in value.error and value.state[C.KEY]['error'] == value.error
        assert 'exit:anchor' not in trace


def test_missing_sink_and_duplicate_rejected() -> None:
    trace: list[str] = []
    parent, deps = fixture_parts(trace)
    with ExitStack() as stack:
        value = C.compose(parent, deps, end_frame=200)(stack)
        del value.state['live_history_sink']
        with pytest.raises(ValueError, match='missing_original_sink'):
            value.perform(None, 100, 100 / 60)
        assert 'original' not in trace
        value.state['live_history_sink'] = object()
        value.perform(None, 100, 100 / 60)
        with pytest.raises(ValueError, match='duplicate_perform'):
            value.perform(None, 102, 102 / 60)
        assert trace.count('original') == 1


@pytest.mark.parametrize('end', [True, 0, -1])
def test_invalid_tracking_end(end: Any) -> None:
    parent, deps = fixture_parts([])
    with pytest.raises(ValueError, match='tracking_end'):
        C.compose(parent, deps, end_frame=end)


def test_fixture_flag_is_not_silently_corrected() -> None:
    trace: list[str] = []
    parent, deps = fixture_parts(trace)
    with ExitStack() as stack:
        value = C.compose(parent, deps, end_frame=200)(stack)
        value.pipe._enable_match_transition_debounce = False
        with pytest.raises(ValueError, match='constructor_not_live_aligned'):
            value.perform(None, 100, 100 / 60)
        assert value.pipe._enable_match_transition_debounce is False and 'original' not in trace
