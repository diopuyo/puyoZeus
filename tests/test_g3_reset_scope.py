"""元保存fixture・原complete cleanupでreset限定HOLDと正常拒否を検査。"""
from __future__ import annotations
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import sys
import importlib.util
import json
from typing import Any
import pytest
from scripts import g3_reset_scope as R
from scripts import g3_owned_exit as X

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       'data/verify/g2_belief_live_publication_2026-09-11_v1'))
from test_second_observation import context, live, saved, generated, O, T, N


def replace(stack: Any, module: Any, name: str, value: Any) -> None:
    """試験所有の置換と復元。"""
    original = getattr(module, name)
    stack.callback(setattr, module, name, original)
    setattr(module, name, value)


@pytest.mark.parametrize('case', ['reset', 'registered', 'software', 'clock', 'reverse', 'source', 'nonzero'])
def test_reset_boundary(context: Any, case: str) -> None:
    """人工世代/時計を元Jに入力する。GT/実動画結果ではない。"""
    c = context
    before, calls = c.journal.scope(), []
    after = deepcopy(before)
    after['generation']['reset_epoch'] += -1 if case == 'reverse' else 1
    if case in ('software', 'clock'):
        after = deepcopy(before)
    if case == 'source':
        after['source_id'] = 'foreign-source'
    def scope(*args: Any) -> dict:
        calls.append(None)
        return deepcopy(before if len(calls) == 1 else after)
    c.journal.scope = scope
    c.journal.epoch = lambda *args: 1 if case == 'software' else 0
    c.journal.tracker = N(generation=lambda side: T.Generation(**after['generation']))
    c.pipe._sm_2p.context.frame_idx = 1 if case == 'nonzero' else 0
    original = O.capture
    with ExitStack() as stack:
        R.install(stack, O, replace)
        assert O.capture.__globals__ is vars(O)
        evidence = O.install(stack, c.journal, c.state)
        evidence.registered_scope = before if case == 'registered' else None
        if case in ('reset', 'registered', 'software'):
            generated(c.journal, c.pipe, c.result, c.row)
            assert evidence.latest['hold_reason'] == 'generation_changed'
            assert not evidence.latest['basis_registered'] and not evidence.latest['quality_gate_clear']
            assert evidence.latest['observed_sm_frame'] == 0
        else:
            with pytest.raises(ValueError):
                generated(c.journal, c.pipe, c.result, c.row)
            assert evidence.latest is None
        assert c.journal.count == 1
    assert O.capture is original and evidence.closed


def test_normal_probability(context: Any) -> None:
    """元の正常PB内容と原cleanupをそのまま通す。"""
    c = context
    with ExitStack() as stack:
        R.install(stack, O, replace)
        evidence = O.install(stack, c.journal, c.state)
        generated(c.journal, c.pipe, c.result, c.row)
        assert evidence.latest['probability']['cells'] == c.state['hidden_probability_observer'].rows[0]['probability']['cells']
        assert c.journal.count == 1


@pytest.mark.parametrize('save_failure', [False, True])
def test_actual_capture_owner_lifo(context: Any, tmp_path: Path, save_failure: bool) -> None:
    """実Capture.__init__/releaseを通し、別ownerの閉鎖順とglobals認証を確認。"""
    c = context
    path = Path(O.__file__).parent.parent / 'g2_second_prefix_runtime_2026-09-14_v40/early_probability_capture.py'
    spec = importlib.util.spec_from_file_location('_g3_actual_early_capture_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.Capture.release = X.release_wrapper(module.Capture.release)
    observer = c.state['hidden_probability_observer']
    observer.installed, observer.closed = True, False
    observer.expected = ((c.row['frame_idx'], '2P'),)
    receiver = c.state['postcommit_current_receiver']
    receiver.journal, receiver.closed, receiver.errors = c.journal, False, []
    c.state['provisional_context_observer'] = receiver.rec
    c.state['output'] = tmp_path
    original = O.capture
    if save_failure:
        (tmp_path / 'EARLY_PROBABILITY_LIFETIME.json').write_text('{}')
    failure = None
    with ExitStack() as outer:
        R.install(outer, O, replace)
        transformed = O.capture
        try:
            with ExitStack() as hooks:
                with ExitStack() as sampling:
                    capture = module.Capture(sampling, c.journal, c.state, O, replace, hook_stack=hooks)
                    generated(c.journal, c.pipe, c.result, c.row)
                    assert capture.evidence.latest['state'] == 'stable'
                assert capture.closed
        except FileExistsError as error:
            failure = error
        assert O.capture is transformed and capture.journal is None
    assert O.capture is original
    assert (failure is not None) == save_failure
    status = json.loads((tmp_path / 'G3_PROBABILITY_RELEASE.json').read_text())
    assert bool(status['save_error']) == save_failure
    receipt = status['receipt']
    assert all(receipt[key] for key in ('sampling_closed', 'hook_closed', 'complete_restored',
                                      'capture_restored', 'references_released'))
