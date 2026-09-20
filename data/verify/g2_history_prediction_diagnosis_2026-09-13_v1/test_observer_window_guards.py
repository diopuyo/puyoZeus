"""窓束縛の負対照と復元。所有実型は別の実configured検査で確認する。"""
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
import pytest
import observer_window_candidate as W

FIRST, LAST, STRIDE = W.FIRST, W.LAST, W.STRIDE


def replace(stack: Any, owner: Any, key: str, value: Any) -> None:
    old = getattr(owner, key)
    stack.callback(setattr, owner, key, old)
    setattr(owner, key, value)


def values() -> dict[str, Any]:
    return dict(scope=SimpleNamespace(WINDOWS=((FIRST, W.OLD_LAST),)),
        context=SimpleNamespace(OUTER_FIRST=FIRST, OUTER_LAST=W.OLD_LAST, STRIDE=STRIDE),
        metadata=SimpleNamespace(FIRST=FIRST, LAST=W.OLD_LAST, STRIDE=STRIDE))


@pytest.mark.parametrize('failure', [False, True])
def test_binding_restore(monkeypatch: Any, failure: bool) -> None:
    selected = values()
    monkeypatch.setattr(W, 'owners', lambda latest: selected)
    try:
        with ExitStack() as stack:
            W.bind(stack, object(), replace)
            assert selected['scope'].WINDOWS == ((FIRST, LAST),)
            assert selected['context'].OUTER_LAST == selected['metadata'].LAST == LAST
            if failure:
                raise LookupError('original')
    except LookupError as error:
        assert failure and str(error) == 'original'
    assert selected['scope'].WINDOWS == ((FIRST, W.OLD_LAST),)
    assert selected['context'].OUTER_LAST == selected['metadata'].LAST == W.OLD_LAST


@pytest.mark.parametrize('key,field', [('scope', 'WINDOWS'), ('context', 'OUTER_LAST'), ('metadata', 'LAST')])
def test_foreign_bounds_no_partial_mutation(monkeypatch: Any, key: str, field: str) -> None:
    selected = values()
    setattr(selected[key], field, None)
    before = {name: vars(value).copy() for name, value in selected.items()}
    monkeypatch.setattr(W, 'owners', lambda latest: selected)
    with ExitStack() as stack, pytest.raises(ValueError, match='original_bounds'):
        W.bind(stack, object(), replace)
    assert before == {name: vars(value) for name, value in selected.items()}


class MetadataBase:
    def wrapper(self, original: Any) -> Any:
        return original


class Metadata(MetadataBase):
    pass


@pytest.mark.parametrize('bad', [None, 'pb', 'current', 'context', 'consumer'])
def test_start_scope_guard(bad: str | None) -> None:
    pairs = [(frame, side) for frame in W.FRAMES for side in W.SIDES]
    pb, current = SimpleNamespace(expected=list(pairs)), SimpleNamespace(expected=list(pairs))
    context = SimpleNamespace(expected=list(W.FRAMES), pb_expected=list(pairs))
    consumer = SimpleNamespace(recorder=context)
    if bad == 'pb': pb.expected.pop()
    if bad == 'current': current.expected.pop()
    if bad == 'context': context.expected.pop()
    if bad == 'consumer': consumer.recorder = object()
    state = dict(hidden_probability_observer=pb, current_scope_sink=current,
        provisional_context_observer=context, private_publication_consumer=consumer,
        collector_metadata_sink=Metadata())
    if bad is not None:
        with pytest.raises(ValueError, match='observer_window_'):
            W.verify_state(state)
    else:
        assert W.verify_state(state)['updates'] == len(W.FRAMES)
