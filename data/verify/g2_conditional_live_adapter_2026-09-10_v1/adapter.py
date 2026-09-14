"""既存runtimeの設置口だけを共有candidateへ束縛し、元guardを保持する。"""
from __future__ import annotations
from pathlib import Path
from types import FunctionType, SimpleNamespace
from typing import Any
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT.parent
ATTACH = VERIFY / 'g2_conditional_constructor_attach_2026-09-10_v1'
RUNTIME = VERIFY / 'g2_history_publication_probe_runtime_2026-09-10_v13'
FACTORY_KEY = 'conditional_runtime_factory'


def load(alias: str, path: Path, stack: Any) -> Any:
    assert alias not in sys.modules
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    stack.callback(sys.modules.pop, alias, None)
    spec.loader.exec_module(module)
    return module


def patch(stack: Any, obj: Any, name: str, value: Any) -> None:
    original = getattr(obj, name)
    stack.callback(setattr, obj, name, original)
    setattr(obj, name, value)


def save_extra(k: Any, state: Any, env: Any) -> None:
    factory = env['factory']
    assert state[FACTORY_KEY] is factory
    control, base = factory.controller, env['runtime'].M.base
    value = dict(installs=state.get('conditional_full_installs', 0),
        revision=state.get('conditional_revision_connection'),
        installation=state.get('combined_install'),
        rows=state.get('conditional_full_rows', []),
        history=getattr(control, 'hidden_history_rows', []),
        events=getattr(control, 'hidden_current_events', []),
        lifetimes=getattr(control, 'hidden_lifetime_rows', []),
        checkpoint_count=len(getattr(control, 'conditional_next_hands', {})),
        physical_certified=False, quality_gate_clear=False)
    k.write(state['output'] / 'CONDITIONAL_RUNTIME.json', base.json_value(value))


def connect(stack: Any, cli: Any, candidate: Any, hook: Any) -> None:
    original_install, original_verify = cli.REPEAT.install, cli.REPEAT.verify
    def supplied(inner: Any) -> tuple[Any, Any]:
        return candidate, hook
    def install(inner: Any, collector: Any, state: Any, env: Any) -> None:
        assert FACTORY_KEY not in state
        state[FACTORY_KEY] = env['factory']
        inner.callback(save_extra, cli.K, state, env)
        original_install(inner, collector, state, env)
    def verify(state: Any) -> dict[str, Any]:
        result = original_verify(state)
        state['combined_restored']()
        assert state['conditional_full_installs'] == 1
        assert state['combined_install']['configure_calls'] == 1
        assert state['conditional_revision_connection']['installs'] == 1
        return result | dict(conditional_shared_installed=True)
    for name, value in (('load', supplied), ('install', install), ('verify', verify)):
        patch(stack, cli.REPEAT, name, value)


def configured(stack: Any) -> Any:
    previous = list(sys.path)
    stack.callback(sys.path.__setitem__, slice(None), previous)
    sys.path.insert(0, str(ATTACH))
    reused = load('_live_adapter_attach_assets', ATTACH / 'run_cpu.py', stack)
    candidate, scope, hook, smoke = reused.modules(stack)
    sys.path.insert(0, str(RUNTIME))
    cli = load('_conditional_original_live_cli', RUNTIME / 'live_cli.py', stack)
    assert cli.K is smoke.K and Path(cli.K.__file__).resolve() == RUNTIME / 'common.py'
    extra = candidate.guards() | {str(p): cli.K.sha(p)
        for p in [*ROOT.glob('*.py'), ROOT / 'PLAN.md', *ATTACH.glob('*.py')]}
    original_guards = cli.K.guards
    def guards() -> dict[str, str]:
        assert all(cli.K.sha(Path(path)) == digest for path, digest in extra.items())
        return original_guards() | extra
    patch(stack, cli.K, 'guards', guards)
    connect(stack, cli, candidate, hook)
    local = SimpleNamespace(**(vars(cli.K) | dict(ROOT=ROOT)))
    main = FunctionType(cli.main.__code__, dict(vars(cli), K=local),
        cli.main.__name__, cli.main.__defaults__, cli.main.__closure__)
    assert main.__code__ is cli.main.__code__ and main.__closure__ is cli.main.__closure__
    return main
