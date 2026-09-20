"""固定history installerにcurrentの明示導出/factoryだけを合成する。"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'g2_history_current_recovery_2026-09-09_v1'
NAMES = ('lifecycle', 'merge', 'policy', 'current_ast', '_current_history_controller', 'controller_v3')
FILES = ('g2_history_current_lifecycle_2026-09-09_v1/lifecycle.py',
    *('g2_history_current_recovery_2026-09-09_v1/' + n + '.py' for n in ('merge', 'policy', 'current_ast', 'controller')),
    'g2_history_current_recovery_2026-09-09_v3/controller_v3.py')
AST_V2 = ROOT.parent / 'g2_history_current_ast_2026-09-09_v2/current_ast.py'
AST_SHA = '8a6c510bc42e724927bb17a134223418fb9643f556078ce9f3edc17dfe588fd8'
CONTROLLER_V3 = ROOT.parent / FILES[-1]
CONTROLLER_SHA = '61bb252025d7d004a187103d828683490c42b830cc0017df400ad16c438a3357'


MERGE_V3 = ROOT.parent / 'g2_history_current_recovery_2026-09-09_v3/merge.py'
MERGE_SHA = '92a618958e29f295e68127d75d48a6bee5d9fb5fbc1a4665faf79d189c32d38c'


def guards() -> dict[str, str]:
    value = json.loads((SOURCE / 'FREEZE.json').read_bytes())['guards']
    for path, expected in value.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('current_candidate_source_changed:' + path)
    if hashlib.sha256(AST_V2.read_bytes()).hexdigest() != AST_SHA:
        raise RuntimeError('current_AST_v2_changed')
    if hashlib.sha256(CONTROLLER_V3.read_bytes()).hexdigest() != CONTROLLER_SHA:
        raise RuntimeError('current_controller_v3_changed')
    if hashlib.sha256(MERGE_V3.read_bytes()).hexdigest() != MERGE_SHA:
        raise RuntimeError('current_merge_v3_changed')
    return value | {str(AST_V2): AST_SHA, str(CONTROLLER_V3): CONTROLLER_SHA, str(MERGE_V3): MERGE_SHA}


def load(stack: Any) -> dict[str, Any]:
    guards()
    result = {}
    for name, relative in zip(NAMES, FILES):
        if name in sys.modules:
            raise RuntimeError('current_module_collision:' + name)
        path = AST_V2 if name == 'current_ast' else ROOT.parent / relative
        if name == 'merge':
            path = MERGE_V3
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        stack.callback(sys.modules.pop, name, None)
        spec.loader.exec_module(module)
        result[name] = module
    return result


def factory_type(a: Any, loaded: Any) -> type:
    class Factory(a.Factory):
        def make_controller(self, provider: Any, legal: Any, *, enabled: bool = False) -> Any:
            a.D.require(self.controller is None and provider is self.provider, 'current_single_controller')
            cc, policy = loaded['_current_history_controller'], loaded['policy']
            cls = loaded['controller_v3'].make_type(self.types.Controller)
            self.types = SimpleNamespace(**(vars(self.types) | {'Controller': cls}))
            inventory = self.modules['history_dependencies'].P
            facade = SimpleNamespace(**(vars(inventory) | {'BoundPolicy': policy.policy_type(inventory)}))
            self.controller = cls(provider, legal, enabled=enabled, inventory=facade)
            attach = provider.attach
            def attached(stack: Any, controller: Any) -> None:
                attach(stack, controller)
                cc.attach_returns(stack, provider.journal, controller)
            provider.attach = attached
            return self.controller
    return Factory


def session(a: Any, c: Any, modules: Any, loaded: Any) -> Any:
    original = modules['history_ast']
    def add_history(tree: Any) -> Any:
        return loaded['current_ast'].add_current(original.add_history(tree))
    combined = SimpleNamespace(add_history=add_history, guards=lambda: original.guards() | guards())
    cloned = a.clone(a.session.__wrapped__, Factory=factory_type(a, loaded))
    return contextmanager(cloned)(c, dict(modules, history_ast=combined))
