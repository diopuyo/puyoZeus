"""G3のconstructor所有stackへ元例外を渡し、終了時の二次失敗を別記する。"""
from __future__ import annotations
from contextlib import ExitStack
import sys
import json
from typing import Any


def finish(owned: Any, candidate: Any, factory: Any, pipe: Any, state: Any,
           evidence: dict, before: Any, kind: Any, body: Any, trace: Any) -> bool:
    """原例外を維持。正常経路のcleanup/参照検査失敗は外側へ送出する。"""
    failures: list[BaseException] = []
    evidence['primary_error'] = None if body is None else repr(body)
    try:
        owned.__exit__(kind, body, trace)
    except BaseException as error:
        failures.append(error)
    try:
        restored = before == candidate.references(factory, pipe, state)
        evidence['references_restored'] = restored
        if not restored:
            failures.append(ValueError('g3_constructor_references_not_restored'))
    except BaseException as error:
        evidence['references_restored'] = False
        failures.append(error)
    evidence['closed'] = True
    evidence['cleanup_errors'] = [repr(error) for error in failures]
    if body is None and failures:
        raise failures[0]
    return False


def attach(stack: Any, candidate: Any, factory: Any, pipe: Any,
           state: Any, evidence: dict) -> None:
    """元install順を保ち、開始失敗も同じ所有出口で閉鎖する。"""
    owned = ExitStack()
    before = candidate.references(factory, pipe, state)
    try:
        candidate.install(owned, factory, pipe, state, evidence['rows'])
        owned.enter_context(candidate.P.Q.R.Q.installed(pipe, evidence['qualification']))
    except BaseException:
        finish(owned, candidate, factory, pipe, state, evidence, before, *sys.exc_info())
        raise
    def close(kind: Any, body: Any, trace: Any) -> bool:
        return finish(owned, candidate, factory, pipe, state, evidence, before, kind, body, trace)
    stack.push(close)
    evidence.update(installed=True, pipe_identity=id(pipe))


def release_wrapper(original: Any) -> Any:
    """元releaseの保存失敗をstderrだけにせず、所有stateと終了例外へ残す。"""
    def release(value: Any, kind: Any, body: Any, trace: Any) -> bool:
        state, output = value.state, value.output
        status = dict(valid=False, save_error=None, original_error=None if body is None else repr(body))
        state['g3_probability_release'] = status
        caught = None
        try:
            original(value, kind, body, trace)
        except BaseException as error:
            caught = error
        receipt = value.hook_receipt
        status['receipt'] = receipt
        if receipt is not None:
            status['valid'] = all(receipt.get(key) is True for key in
                ('sampling_closed', 'hook_closed', 'complete_restored', 'capture_restored', 'references_released'))
            status['valid'] = status['valid'] and receipt.get('hook_error') is None
        try:
            if receipt is None or json.loads((output / 'EARLY_PROBABILITY_LIFETIME.json').read_text()) != receipt:
                raise ValueError('g3_probability_release_saved_mismatch')
        except BaseException as error:
            status['save_error'] = repr(error)
            if caught is None:
                caught = error
        try:
            with (output / 'G3_PROBABILITY_RELEASE.json').open('x') as stream:
                json.dump(status, stream, allow_nan=False)
        except BaseException as error:
            status['save_error'] = repr(error)
            if caught is None:
                caught = error
        if caught is not None:
            raise caught
        return False
    return release
