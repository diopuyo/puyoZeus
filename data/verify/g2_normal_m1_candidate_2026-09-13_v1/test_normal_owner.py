"""正常Registryの元実体を使い、重複/異物/初期化下限/原例外の非隠蔽を検査。"""
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import test_normal_basis as F
parts = F.parts
ROOT = Path(__file__).resolve().parent
START, END = 33724, 34290


@pytest.fixture(scope='module')
def module(parts: Any) -> Any:
    return parts.loader('_normal_owner_test', ROOT / 'normal_owner.py',
        dict(belief=parts.original.mode.B, conditioning=parts.original.binding.C,
             serialization=parts.original.binding.S, registry=parts.original.binding.OLD.R,
             journal_context=parts.context))


def fixture(tmp_path: Path) -> tuple:
    pipe = N()
    factory = N(provider=N(journal=N(pipe=pipe)))
    state = dict(private_suffix_factory=factory, output=tmp_path)
    return factory, pipe, state


def test_normal_owner_bounds_and_cleanup(module: Any, tmp_path: Path) -> None:
    factory, pipe, state = fixture(tmp_path)
    with ExitStack() as stack:
        owner = module.install(stack, factory, pipe, state, START, END)
        assert not owner.registry._bindings
        for frame in (START - 2, START - 1):
            assert owner.eligible_frame(frame) is False
            with pytest.raises(ValueError, match='normal_basis_before_start'):
                owner.create_mode(None, N(latest=dict(frame=frame)), None, None, None, '1P')
        assert owner.eligible_frame(START) and owner.eligible_frame(END)
        with pytest.raises(ValueError, match='normal_frame_bound'):
            owner.eligible_frame(END + 1)
        with pytest.raises(ValueError, match='normal_duplicate_owner'):
            module.install(stack, factory, pipe, state, START, END)
    assert owner.closed and not hasattr(factory, module.REGISTRY_KEY)
    assert module.KEY not in state and module.REGISTRY_KEY not in state
    assert json.loads((tmp_path / 'NORMAL_OWNER_STATUS.json').read_bytes())['error'] is None


def test_foreign_owner_preserved_original_failure(module: Any, tmp_path: Path) -> None:
    factory, pipe, state = fixture(tmp_path)
    foreign = object()
    with pytest.raises(LookupError, match='original_body'):
        with ExitStack() as stack:
            owner = module.install(stack, factory, pipe, state, START, END)
            setattr(factory, module.REGISTRY_KEY, foreign)
            raise LookupError('original_body')
    assert getattr(factory, module.REGISTRY_KEY) is foreign
    assert owner.closed and 'normal_owner_lifetime' in str(owner.error)
    saved = json.loads((tmp_path / 'NORMAL_OWNER_STATUS.json').read_bytes())
    assert 'original_body' in saved['original_body'] and saved['quality_gate_clear'] is False


def test_foreign_pipe_rejected(module: Any, tmp_path: Path) -> None:
    factory, pipe, state = fixture(tmp_path)
    with pytest.raises(ValueError, match='normal_factory_journal_owner'):
        module.Owner(factory, N(), state, START, END)
    assert not hasattr(factory, module.REGISTRY_KEY)
