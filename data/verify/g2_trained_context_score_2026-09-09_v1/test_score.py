"""較正順序と既存binder/samplerの境界。モデルは人工対照。"""
from __future__ import annotations
import copy
from dataclasses import replace
import importlib.util
import sys
from types import SimpleNamespace
from typing import Any
import numpy as np
import pytest
import torch
import score as P

SPEC = importlib.util.spec_from_file_location('_trained_score_binding_fixture', P.BINDING_ROOT / 'test_binding.py')
T = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = T
SPEC.loader.exec_module(T)
candidates, fixture = T.candidates, T.fixture


class FakeModel(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset, self.calls = offset, 0

    def infer(self, boards: Any, queues: Any, values: Any, masks: Any,
              supported: Any, integrity: Any) -> Any:
        self.calls += 1
        assert not supported.any() and integrity.all() and not values.any()
        assert torch.equal(masks.sum(-1), torch.ones_like(masks.sum(-1)))
        p = boards[:, 0, 0, 0].to(torch.float64) * 0.15 + 0.01 + self.offset
        return SimpleNamespace(raw_probability=p)


def members() -> tuple[Any, ...]:
    return tuple(SimpleNamespace(seed=seed, slope=slope, model=FakeModel(offset).eval())
        for seed, slope, offset in zip(P.M1_SEEDS, (0.4, 0.3, 0.35), (0.0, 0.01, 0.02)))


def test_sample_nonlinear_order_and_shared_inputs(fixture: Any) -> None:
    bound = P.B.bind_provisional_context(*fixture)
    value = members()
    scorer = P.BatchScorer(bound, value, 'cpu')
    samples = np.repeat(np.asarray(fixture[0]['sides']['1P']['pb']['confirmed']['grid'])[None, None], 4, axis=0)
    samples = np.repeat(samples, 2, axis=1)
    for index, side in enumerate(P.B.SIDES):
        samples[:, index] = np.asarray(fixture[0]['sides'][side]['pb']['confirmed']['grid'])
    samples[:, 0, 0, 0] = (0, 1, 2, 4)
    before = samples.copy()
    output = scorer(samples)
    expected = P.equal_seed_probability_mean({m.seed: P.calibrate(scorer.raw[m.seed], m.slope) for m in value})
    assert np.array_equal(output, expected) and np.array_equal(samples, before)
    wrong = np.mean([P.calibrate(np.array([scorer.raw[m.seed].mean()]), m.slope)[0] for m in value])
    assert abs(output.mean() - wrong) > 1e-3
    assert np.array_equal(scorer.raw_cached(samples), P.equal_seed_probability_mean(scorer.raw))
    assert [m.model.calls for m in value] == [1, 1, 1]
    with pytest.raises(P.B.ContextFault, match='repeated'):
        scorer(samples)
    samples[0, 0, 0, 0] = 3
    with pytest.raises(P.B.ContextFault, match='samples_differ'):
        scorer.raw_cached(samples)


def test_pointmass_real_sampler_and_no_second_infer(fixture: Any, monkeypatch: Any) -> None:
    value = members()
    verified = []
    monkeypatch.setattr(P, 'members_ready', lambda source, models, device: verified.append(source))
    bound = P.B.bind_provisional_context(*fixture)
    result = P.evaluate(bound, value, sample_count=16, seed=5)
    assert verified == [fixture[1]['source_id']]
    assert [m.model.calls for m in value] == [1, 1, 1]
    expected = np.mean([P.calibrate(np.array(result['seed_raw'][str(m.seed)]), m.slope)[0] for m in value])
    assert result['calibrated']['win_probability_p1'] == expected
    assert len(result['seed_raw'][str(P.M1_SEEDS[0])]) == 1
    assert not result['supported'] and result['integrity_valid']
    assert not result['quality_gate_clear'] and not result['calibration_quality_measured']


@pytest.mark.parametrize('key,value', (('supported', True), ('integrity_valid', False), ('context_digest', 'fake')))
def test_bound_controls_rejected_before_model(fixture: Any, monkeypatch: Any, key: str, value: Any) -> None:
    bound = replace(P.B.bind_provisional_context(*fixture), **{key: value})
    monkeypatch.setattr(P, 'members_ready', lambda *a: pytest.fail('model touched'))
    with pytest.raises(P.B.ContextFault):
        P.evaluate(bound, members())


def test_future_and_fault_stay_rejected(fixture: Any, monkeypatch: Any) -> None:
    bound = P.B.bind_provisional_context(*fixture)
    payload = __import__('json').loads(bound.source_json)
    payload['row']['available_frame'] += 2
    changed = replace(bound, source_json=P.B.encoded(payload))
    monkeypatch.setattr(P, 'members_ready', lambda *a: pytest.fail('model touched'))
    with pytest.raises(P.B.ContextFault, match='context_available'):
        P.evaluate(changed, members())


def test_member_verifier_exception_not_swallowed(fixture: Any, monkeypatch: Any) -> None:
    error = ValueError('model_changed')
    def fail(*args: Any) -> None:
        raise error
    monkeypatch.setattr(P, 'members_ready', fail)
    with pytest.raises(ValueError) as caught:
        P.evaluate(P.B.bind_provisional_context(*fixture), members())
    assert caught.value is error


def test_visible_change_is_not_hidden_sampling(fixture: Any) -> None:
    bound = P.B.bind_provisional_context(*fixture)
    scorer = P.BatchScorer(bound, members(), 'cpu')
    batch = np.array([[fixture[0]['sides'][s]['pb']['confirmed']['grid'] for s in P.B.SIDES]])
    batch[0, 0, 12, 0] = (batch[0, 0, 12, 0] + 1) % 5
    with pytest.raises(ValueError, match='visible_board_changed'):
        scorer(batch)
