"""2P観測→原J票→初回Registry登録。frame/pipeは人工、原completeは実本体。"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
from test_live_binding import saved, live
from test_second_observation import context, generated
import second_observation as O
import journal_witness as W
import second_basis as S
import registry as R


@pytest.fixture(scope='module')
def policy() -> Any:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root/'g2_reset_settled_basis_gate_2026-09-11_v1'))
    path = root/'g2_hidden_basis_initialization_2026-09-11_v1/hidden_basis_gate.py'
    spec = importlib.util.spec_from_file_location('_second_initial_hidden_policy', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def observed(context: Any) -> Any:
    c = context
    factory = N()
    registry = R.Registry(factory)
    setattr(factory, '_g2_probabilistic_scope_registry', registry)
    with ExitStack() as stack:
        witness = W.install(stack, c.journal)
        evidence = O.install(stack, c.journal, c.state)
        scope = dict(c.journal.scope(), side='1P')
        item = dict(scope=scope, token='step:598', epoch=1, events=[], return_line=None, frame=None)
        c.journal.complete_step(item, None, None, sys.getprofile())
        generated(c.journal, c.pipe, c.result, c.row)
        yield N(evidence=evidence, witness=witness, pipe=c.pipe, registry=registry, factory=factory)


def test_first_actual_observation_registration(observed: Any, policy: Any) -> None:
    o = observed
    binding, receipt = S.initialize(o.evidence, o.witness, o.pipe, o.registry, o.factory, 35410, policy)
    value = o.registry.current(binding)
    assert value.scope[-1] == '2P' and value.frame == 35370
    assert S.S.decode(receipt['state']) == value and not receipt['reset_or_fall_claimed']
    with pytest.raises(ValueError, match='registry_duplicate_scope'):
        S.initialize(o.evidence, o.witness, o.pipe, o.registry, o.factory, 35410, policy)
    assert o.registry.current(binding) is value


def test_foreign_J_token_rejected(observed: Any, policy: Any) -> None:
    o = observed
    o.evidence.latest['journal_token'] = 'step:601'
    with pytest.raises(ValueError, match='second_basis_J'):
        S.initialize(o.evidence, o.witness, o.pipe, o.registry, o.factory, 35410, policy)
    assert not o.registry._bindings


def test_unknown_prior_not_collapsed(observed: Any, policy: Any) -> None:
    # これは分布変換の人工対照。改変した票を原観測として登録しない。
    row = deepcopy(observed.evidence.latest)
    grid = deepcopy(row['confirmed']['grid'])
    for r in range(1,13):
        grid[r][0] = row['raw']['grid'][r][0] = 1+r%2
        row['probability']['cells'][r][0] = [[grid[r][0],1.0]]
    row['raw']['grid'][0][0] = 10
    probability, changed = S.distribution(row, tuple(map(tuple,grid)), policy)
    assert changed == (0,) and len(probability.cell(0,0).probs) == 7
    value, mass, removed = S.P.establish_conditioned(tuple(row['scope']),35370,35410,
        S.B.Board.from_dict({'grid':grid}),probability)
    assert len(value.worlds) == 7 and mass == pytest.approx(1.0) and removed == 0


def test_uniform_prior_override_rejected(observed: Any, policy: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(policy, 'UNOBSERVED_PRIOR', ((0,1.0),))
    row = observed.evidence.latest
    with pytest.raises(ValueError, match='second_policy_constants'):
        S.distribution(row, tuple(map(tuple,row['confirmed']['grid'])), policy)
