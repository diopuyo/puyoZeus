"""実注入関数の模擬constructor検収。実pipeline陽性とは区別する。"""
from contextlib import ExitStack
from types import SimpleNamespace as N
from typing import Any, Iterator
import pytest
import constructor_fixture as C
import reproduce_constructor_v1 as R


def drive(kwargs: dict, fail: bool = False) -> tuple:
    fixture = R.load('_constructor_test_original', R.FULL / 'fixture_full.py')
    palette = R.load('_constructor_test_palette', R.PALETTE / 'palette_fixture.py')
    received: list[dict] = []
    class Pipeline:
        def __init__(self, **values: Any) -> None:
            received.append(values)
            if fail:
                raise RuntimeError('original_constructor_failure')
    original, original_select = Pipeline.__init__, palette.configured
    def real(frozen: Any, monkeypatch: Any) -> Iterator[Any]:
        yield frozen.RecognitionPipeline(**kwargs)
    evidence: dict[str, Any] = {}
    receipt: dict[str, Any] = {}
    active = N(P=N(load=lambda alias, path: N(real=N(__wrapped__=real))))
    error = None
    with ExitStack() as stack:
        C.install(stack, palette, receipt)
        palette.configured(fixture).install(stack, active, evidence)
        generator = active.P.load('_normal_active_legacy_fixture', None).real.__wrapped__(N(RecognitionPipeline=Pipeline), None)
        try:
            next(generator)
        except BaseException as caught:
            error = caught
        finally:
            generator.close()
    assert Pipeline.__init__ is original and palette.configured is original_select
    assert receipt['selected'] and receipt['restored']
    return received, evidence, error


def test_original_missing_flag() -> None:
    assert R.reproduce()['missing_flag_reproduced']


def test_three_flags_and_evidence() -> None:
    received, evidence, error = drive({})
    assert error is None and received == [dict(enable_ojama_write_accounting_guard=True,
        enable_next_history_starvation_fix=True, enable_match_transition_debounce=True)]
    for row in (evidence, evidence['palette_constructor_fixture'], evidence['debounce_constructor_fixture']):
        assert row['original_constructor_calls'] == 1 and row['constructor_restored']


def test_original_exception_preserved() -> None:
    received, evidence, error = drive({}, True)
    assert len(received) == 1 and str(error) == 'original_constructor_failure'
    assert evidence['debounce_constructor_fixture']['constructor_restored']


@pytest.mark.parametrize('flag', ('enable_ojama_write_accounting_guard', *C.FLAGS))
def test_explicit_flags_rejected(flag: str) -> None:
    received, evidence, error = drive({flag: False})
    assert not received and error is not None


def test_duplicate_install_rejected() -> None:
    palette = N(configured=lambda value: value)
    with ExitStack() as stack:
        C.install(stack, palette, {})
        with pytest.raises(AssertionError, match='duplicate'):
            C.install(stack, palette, {})
