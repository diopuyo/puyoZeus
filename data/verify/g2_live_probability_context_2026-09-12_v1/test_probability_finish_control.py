"""終了wrapperの非対象委譲・保存・解除・元例外。内部品質検査はstubと明記。"""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import probability_finish as F


def configured(stack: ExitStack, writer: Any) -> tuple[Any, Any, list[Any]]:
    calls: list[Any] = []
    def previous(*args: Any) -> Any:
        calls.append(args)
        return 'original'
    target = N(evaluate=previous)
    main = N(__globals__=dict(Q=N(FINAL=target), K=N(write=writer)))
    F.install(stack, main, None, None)
    return target, previous, calls


def test_unselected_original_arguments_and_restore(tmp_path: Path) -> None:
    with ExitStack() as stack:
        target, previous, calls = configured(stack, lambda *args: pytest.fail('非対象保存'))
        args = (object(), [], object(), tmp_path, {})
        assert target.evaluate(*args) == 'original' and calls == [args]
    assert target.evaluate is previous


def test_selected_once_saved_and_restored(tmp_path: Path, monkeypatch: Any) -> None:
    # evaluate本体の品質をこのstubで証明しない。
    expected, writes = dict(control_fixture_only=True), []
    monkeypatch.setattr(F, 'evaluate', lambda *args: expected)
    state = dict(probabilistic_tracking_mode=object())
    with ExitStack() as stack:
        target, previous, calls = configured(stack, lambda path, value: writes.append((path, value)))
        assert target.evaluate(None, [], None, tmp_path, state) is expected and not calls
        assert writes == [(tmp_path / 'LIVE_PROBABILITY_FINAL.json', expected)]
        with pytest.raises(AssertionError, match='duplicate_probability_finalize'):
            target.evaluate(None, [], None, tmp_path, state)
    assert target.evaluate is previous and state[F.KEY]['stage'] == 'saved'


@pytest.mark.parametrize('save_fails', (False, True))
def test_original_failure_identity_preserved(tmp_path: Path, monkeypatch: Any, save_fails: bool) -> None:
    original = RuntimeError('original_evaluation_failure')
    def fail(*args: Any) -> None:
        raise original
    def write(*args: Any) -> None:
        if save_fails:
            raise OSError('failure_evidence_disk_error')
    monkeypatch.setattr(F, 'evaluate', fail)
    state = dict(probabilistic_tracking_mode=object())
    with ExitStack() as stack:
        target, previous, calls = configured(stack, write)
        with pytest.raises(RuntimeError) as caught:
            target.evaluate(None, [], None, tmp_path, state)
        assert caught.value is original and not calls
    assert target.evaluate is previous
    assert ('failure_save_error' in state[F.KEY]) is save_fails
