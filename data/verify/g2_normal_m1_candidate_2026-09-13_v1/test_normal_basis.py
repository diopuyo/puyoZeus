"""左右非resetの同call登録/隔離を元Registryで検査。原J/PB入力は人工。"""
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_m1_second_runtime_2026-09-13_v21'))
import target_entry as T


@pytest.fixture(scope='module')
def parts() -> Any:
    with ExitStack() as stack:
        T.A.configured(stack)
        import probe_native_merge
        owner = sys.modules[T.A.A.A.A.V4.OWNED_ALIAS]
        original = owner.dependencies().modules()
        loader = owner.bootstrap().load
        imports = dict(belief=original.mode.B, conditioning=original.binding.C,
                       serialization=original.binding.S)
        context = loader('_normal_test_context', ROOT.parent /
                         'g2_belief_live_publication_2026-09-11_v1/journal_context.py', imports)
        basis = loader('_normal_test_basis', ROOT / 'second_basis.py',
                       imports | dict(journal_context=context))
        yield N(original=original, basis=basis, context=context, loader=loader,
                policy=sys.modules['_g2_hidden_initialization_gate'])


def fixture(parts: Any) -> Any:
    B = parts.original.mode.B
    pipe, factory = N(), N()
    journal = N(pipe=pipe)
    registry = parts.original.binding.OLD.R.Registry(factory)
    factory._g2_probabilistic_scope_registry = registry
    steps, rows = [], []
    for index, side in enumerate(('1P', '2P')):
        board = B.Board()
        board.set(12, index, index + 1)
        sm = N(context=N(frame_idx=10, state=N(value='stable'), confirmed_board=board))
        setattr(pipe, '_sm_' + side.lower(), sm)
        setattr(pipe, '_active_chain_' + side.lower(), None)
        grid = [list(row) for row in B.grid(board)]
        step = dict(source_id='normal', run_id='fixture', software_reset=0, pipe_object_id=id(pipe),
                    side=side, status='returned', exception=None, token='step:' + side,
                    code_sha256='fixture', generation=dict(reset_epoch=0),
                    generation_after=dict(reset_epoch=0), returned=dict(confirmed=dict(grid=grid)))
        row = dict(scope=['normal', 'fixture', 0, id(pipe), id(sm), 0, side], frame=10,
                   state='stable', match_active=True, effect_window=False, origin_present=False,
                   landing_grace_expired=True, pb_holds=[], pb_errors=[], journal_token=step['token'],
                   original_code_sha256='fixture', confirmed=dict(grid=grid), raw=dict(grid=grid),
                   probability=dict(present=True, type_valid=True, errors=[],
                       cells=[[[[color, 1.0]] for color in line] for line in grid]))
        steps.append(step)
        rows.append(row)
    witness = N(journal=journal, pair=lambda frame: steps)
    evidence = [N(closed=False, error=None, latest=row, journal=journal) for row in rows]
    return N(pipe=pipe, factory=factory, registry=registry, witness=witness, evidence=evidence, steps=steps)


def test_both_sides_same_registry_without_reset(parts: Any) -> None:
    f = fixture(parts)
    bindings = []
    for side, evidence in zip(('1P', '2P'), f.evidence):
        binding, receipt = parts.basis.initialize(evidence, f.witness, f.pipe, f.registry,
            f.factory, 100, parts.policy, side=side)
        assert binding.scope[-1] == side and receipt['reset_or_fall_claimed'] is False
        bindings.append(binding)
    assert bindings[0] is not bindings[1]
    assert all(f.registry.current(binding).frame == 10 for binding in bindings)


@pytest.mark.parametrize('side,index', [('1P', 0), ('2P', 1)])
def test_foreign_side_journal_rejected(parts: Any, side: str, index: int) -> None:
    f = fixture(parts)
    f.steps[index]['side'] = '2P' if side == '1P' else '1P'
    with pytest.raises(ValueError, match='second_completed_J'):
        parts.basis.initialize(f.evidence[index], f.witness, f.pipe, f.registry,
            f.factory, 100, parts.policy, side=side)
    assert not f.registry._bindings
