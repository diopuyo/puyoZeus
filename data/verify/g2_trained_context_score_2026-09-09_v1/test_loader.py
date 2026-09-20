"""保存学習済みcheckpointの復元だけを検査し、モデル推論は行わない。"""
from __future__ import annotations
import copy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from typing import Any
import pytest
import loader as L


@pytest.fixture(scope='module')
def loaded() -> tuple[Any, ...]:
    api = L.backend()
    api.torch.set_num_threads(2)
    members = L.load_members(L.SOURCE_ID)
    output = Path(os.environ['LOADER_OUTPUT'])
    values = [{k: v for k, v in asdict(replace(m, model=None)).items()} for m in members]
    with (output / 'MEMBERS.json').open('x', encoding='utf-8') as stream:
        json.dump({'members': values, 'model_inference_executed': False,
                   'trained_checkpoint_restored': True, 'quality_gate_clear': False}, stream, indent=2)
    return members


def test_real_checkpoint_restore_and_repeat_verification(loaded: tuple[Any, ...]) -> None:
    L.verify_members(L.SOURCE_ID, loaded)
    assert tuple(m.seed for m in loaded) == L.SEEDS
    assert all(m.fold == 6 for m in loaded)
    assert not L.backend().torch.cuda.is_initialized()


@pytest.mark.parametrize('value', [False, 0, '', L.SOURCE_ID.replace('b372', 'a372')])
def test_wrong_source_before_import(value: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(L, 'backend', lambda: pytest.fail('拒否前import'))
    with pytest.raises(ValueError, match='source_id'):
        L.load_members(value)


@pytest.mark.parametrize('device', ['cuda:0', False])
def test_cpu_only(device: Any) -> None:
    with pytest.raises(ValueError, match='cpu_only'):
        L.load_members(L.SOURCE_ID, device)


@pytest.mark.parametrize('field,value', [('seed', True), ('seed', 20260905), ('fold', 1),
    ('fold', 6.0), ('slope', 0.4), ('checkpoint_sha256', '0' * 64),
    ('m0_state_sha256', '0' * 64), ('model_state_sha256', '0' * 64)])
def test_forged_member_rejected(loaded: tuple[Any, ...], field: str, value: Any) -> None:
    members = (replace(loaded[0], **{field: value}), *loaded[1:])
    with pytest.raises(ValueError):
        L.verify_members(L.SOURCE_ID, members)


def test_member_coverage_order_and_type(loaded: tuple[Any, ...]) -> None:
    for members in (loaded[:2], loaded[::-1], list(loaded), (None, *loaded[1:])):
        with pytest.raises(ValueError):
            L.verify_members(L.SOURCE_ID, members)


@pytest.mark.parametrize('field,value', [('seed', True), ('fold', 1), ('fold', 6.0),
    ('not_production', 1), ('fallback_to_m0', True), ('model_version', 'wrong'),
    ('input_schema_version', 'wrong'), ('m0_state_sha256', '0' * 64)])
def test_checkpoint_metadata_negative(loaded: tuple[Any, ...], field: str, value: Any) -> None:
    api = L.backend()
    payload = L._payload(api, L.SEEDS[0])
    payload[field] = value
    with pytest.raises(ValueError, match='checkpoint_'):
        L._validate_payload(api, payload, L.SEEDS[0])


def test_checkpoint_sha_rejected_before_deserialization(monkeypatch: Any) -> None:
    api = L.backend()
    monkeypatch.setattr(L, 'sha', lambda path: '0' * 64)
    monkeypatch.setattr(api.torch, 'load', lambda *a, **k: pytest.fail('変造後load'))
    with pytest.raises(ValueError, match='checkpoint_changed'):
        L._payload(api, L.SEEDS[0])


def test_strict_shape_rejects_and_does_not_modify_original(loaded: tuple[Any, ...]) -> None:
    model = copy.deepcopy(loaded[0].model)
    state = model.state_dict()
    state.pop(next(iter(state)))
    with pytest.raises(RuntimeError):
        model.load_state_dict(state, strict=True)
    L.verify_members(L.SOURCE_ID, loaded)


@pytest.mark.parametrize('kind', ['weight', 'grad', 'training', 'method', 'hook', 'variant', 'activation', 'private_method'])
def test_live_model_mutation_rejected(loaded: tuple[Any, ...], kind: str) -> None:
    api = L.backend()
    model = copy.deepcopy(loaded[0].model)
    if kind == 'weight':
        with api.torch.no_grad():
            next(model.parameters()).reshape(-1)[0] += 1.0
    elif kind == 'grad':
        next(model.parameters()).requires_grad_(True)
    elif kind == 'training':
        model.train()
    elif kind == 'method':
        model.infer = lambda *args: None
    elif kind == 'hook':
        model.register_forward_hook(lambda *args: None)
    elif kind == 'activation':
        name = next(n for n, m in model.named_modules() if n.startswith('m0.') and type(m) is api.torch.nn.ReLU)
        parent, key = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), key, api.torch.nn.Sigmoid().eval())
    elif kind == 'private_method':
        model._evaluate_supported = lambda *args: None
    else:
        model.variant = 'values'
    claimed = api.fixed.frozen.model_state_sha256(model)
    members = (replace(loaded[0], model=model, model_state_sha256=claimed), *loaded[1:])
    with pytest.raises(ValueError):
        L.verify_members(L.SOURCE_ID, members)


def test_weights_only_and_no_prediction(loaded: tuple[Any, ...], monkeypatch: Any) -> None:
    api = L.backend()
    original, calls = api.torch.load, []
    def traced(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get('weights_only'))
        return original(*args, **kwargs)
    monkeypatch.setattr(api.torch, 'load', traced)
    monkeypatch.setattr(api.fixed.v3.AdvantageM1ZeroCounterfactualV3, 'forward',
                        lambda *args: pytest.fail('推論禁止'))
    L.verify_members(L.SOURCE_ID, loaded)
    assert calls == [True, True, True]
