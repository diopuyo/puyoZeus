"""新binding callbackから元retired_owner認証を実行。archive/Registryは明示人工対照。"""
from __future__ import annotations
from dataclasses import asdict
from types import SimpleNamespace as N
from typing import Any
import pytest
import stage_retired_identity as S
import test_stage_identity_repro as R
import test_probability_owner as O


@pytest.mark.parametrize('defect', [None, 'reset', 'receipt', 'integer', 'frame', 'side'])
def test_actual_retired_authority_callback(defect: str | None) -> None:
    sample = O.sample()
    factory, lease = sample.factory, sample.lease
    factory.controller.provider = factory.provider
    factory.provider.journal = lease.recovery.journal = N(controller=object())
    rows = [dict(scope=dict(side='1P', frame_idx=lease.empty_evidence.frame),
                 decision=dict(history_state=asdict(sample.value)))]
    function = S.adapted(R.actual(), sample.runtime, lease, N(E=O.R.E))
    if defect == 'reset':
        lease.recovery.reset_count = 2
    elif defect == 'receipt':
        lease.empty_evidence.receipt_sha = 'wrong'
    elif defect == 'integer':
        factory.controller.history['1P'] = sample.old
    elif defect == 'frame':
        rows[-1]['scope']['frame_idx'] += 2
    elif defect == 'side':
        rows[-1]['scope']['side'] = '2P'
    def same(left: Any, right: Any, reason: str) -> None:
        if left != right:
            raise AssertionError(reason)
    module = N(E=N(same=same))
    if defect is None:
        assert function(factory.controller, factory, rows, module) is True
        assert sample.checked == [lease.archive]
        assert factory.controller.history == {}
    else:
        with pytest.raises(AssertionError):
            function(factory.controller, factory, rows, module)
