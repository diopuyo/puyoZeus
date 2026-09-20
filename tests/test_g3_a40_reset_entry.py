"""元late_loaderと後発runtime_patchの閉包契約を人工bootstrapで検証する。"""
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as Box
from typing import Any
import pytest
from scripts import g3_a40_reset_entry as A


def replace(stack: Any, owner: Any, name: str, value: Any) -> None:
    """人工所有者の参照をLIFO復元する。"""
    prior = getattr(owner, name)
    stack.callback(setattr, owner, name, prior)
    setattr(owner, name, value)


def loader(stack: Any) -> Any:
    """元ファイルを私有aliasへ読み、注入とsys.modulesを復元する。"""
    def load(alias: str, path: Path, injection: Any = None) -> Any:
        spec = importlib.util.spec_from_file_location(alias, path)
        module = importlib.util.module_from_spec(spec)
        old = {key: sys.modules.get(key) for key in (injection or {})}
        prior = sys.modules.get(alias)
        def restore() -> None:
            if prior is None:
                sys.modules.pop(alias, None)
            else:
                sys.modules[alias] = prior
        stack.callback(restore)
        sys.modules[alias] = module
        try:
            sys.modules.update(injection or {})
            spec.loader.exec_module(module)
        finally:
            for key, value in old.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value
        return module
    return load


def test_late_loader_and_patch(tmp_path: Path, monkeypatch: Any) -> None:
    """初期所有・後発閉包検査・alias限定・重複拒否・元capture復元を通す。"""
    with ExitStack() as stack:
        load = loader(stack)
        bootstrap = Box(load=load, __name__='_g3_cpu_bootstrap')
        owner = Box(bootstrap=lambda: bootstrap)
        monkeypatch.setitem(sys.modules, bootstrap.__name__, bootstrap)
        monkeypatch.setitem(sys.modules, '_g3_cpu_owner', owner)
        dependencies = Box(session_creator=lambda current, frames: current)
        version = Box(OWNED_ALIAS='_g3_cpu_owner', replace_owned=replace, A=Box(D=dependencies))
        adapter = Box(A=Box(A=Box(A=Box(V4=version))))
        late = load('_g3_cpu_late', A.G.ROOT / 'data/verify/g2_m1_loader_lifetime_2026-09-13_v1/late_loader.py')
        late.install(stack, dependencies, owner, replace)
        A.install(stack, adapter, tmp_path)
        current = dependencies.session_creator(bootstrap.load, (35370, 35410))
        patch = load('_g3_cpu_patch', late.PATCH)
        patch.install(stack, bootstrap, replace)
        context = current('_g3_cpu_context', A.PUBLICATION / 'journal_context.py')
        early = current('_early_probability_observation', A.OBSERVATION, dict(journal_context=context))
        observation = current('_g2_pub_runtime_second_observation', A.OBSERVATION, dict(journal_context=context))
        flags = current('_g2_evaluation_flags', A.FLAGS)
        assert Path(early.capture.__code__.co_filename).resolve() == A.OBSERVATION
        assert Path(observation.capture.__code__.co_filename).resolve() == Path(A.N.R.__file__).resolve()
        assert Path(flags.capture.__code__.co_filename).resolve() == Path(A.N.__file__).resolve()
        with pytest.raises(ValueError, match='duplicate_load'):
            current('_g2_evaluation_flags', A.FLAGS)
    assert bootstrap.load is load
    assert Path(flags.capture.__code__.co_filename).resolve() == A.FLAGS
    assert Path(observation.capture.__code__.co_filename).resolve() == A.OBSERVATION
