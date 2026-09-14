"""最終Scheduled型の生成時にだけ2P到来接続を追加する。私有aliasは同scopeで解放。"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
PATCH = ROOT.parent / 'g2_m1_completion_candidate_2026-09-12_v1/runtime_patch.py'
ARRIVAL = ROOT.parent / 'g2_arrival_ack_candidate_2026-09-12_v1/arrival_mode.py'
NAMES = ('_g2_second_source_state', '_g2_second_source_base_tap', '_g2_second_source_tap',
         '_g2_second_source_binding', '_g2_second_source_session', '_g2_second_warning_observation',
         '_g2_second_warning_source', '_g2_second_terminal_math', '_g2_second_terminal_recovery')


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('second_source_selection:' + reason)


def services(arrival: Any, modules: tuple) -> Any:
    from src import ojama_warning
    _, _, _, _, _, observation, warning, math, recovery = modules
    detector = ojama_warning.OjamaWarningDetector()
    require(detector._cnn is not None and next(detector._cnn.parameters()).device.type == 'cpu',
            'warning_model_missing_or_non_cpu')
    commit, lane, notice = (sys.modules[name] for name in
        ('_g2_prefix_commit', '_g2_prefix_lane', '_g2_second_notice_v21'))
    require(Path(commit.__file__).name == 'prefix_commit.py'
            and Path(lane.__file__).name == 'lane_state.py'
            and Path(notice.__file__).resolve() == ROOT.parent /
                'g2_second_origin_postrun_2026-09-13_v1/settled_notice.py', 'terminal_dependencies')
    return SimpleNamespace(parts=SimpleNamespace(mode=arrival), commit=commit, lane=lane,
        notice=notice, warning=warning, recovery=recovery,
        sampler=observation.Sampler(ojama_warning, detector))


class Selection:
    def __init__(self, stack: Any, scope_getter: Any) -> None:
        self.stack, self.scope_getter = stack, scope_getter
        self.scope, self.closed = None, False
        self.modules: dict[str, Any] = {}
        self.bound: list[Any] = []

    def claim(self) -> None:
        scope = self.stack if self.scope_getter is None else self.scope_getter()
        require(not self.closed and (self.scope is None or self.scope is scope), 'scope_changed')
        if self.scope is None:
            require(not any(name in sys.modules for name in NAMES), 'foreign_alias')
            self.scope = scope
            scope.push(self.release)

    def release(self, kind: Any, body: Any, trace: Any) -> bool:
        changed = []
        for name, module in self.modules.items():
            if sys.modules.get(name) is module:
                sys.modules.pop(name)
            else:
                changed.append(name)
        self.closed = True
        if changed and body is None:
            raise ValueError('second_source_selection:alias_replaced:' + ','.join(changed))
        return False

    def dependencies(self, load: Any) -> tuple:
        if not self.modules:
            specs = (('second_arrival_state.py', None), ('second_enqueue_tap.py', None),
                     ('witness_enqueue_tap.py', 'base'),
                     ('second_arrival_binding.py', 'state'), ('session_connection.py', None),
                     ('warning_observation.py', None), ('warning_source.py', None),
                     ('observed_terminal_drop.py', None), ('terminal_recovery.py', 'recovery'))
            for alias, (name, injection) in zip(NAMES, specs):
                values = {'second_arrival_state': self.modules[NAMES[0]]} if injection == 'state' else None
                if injection == 'base':
                    values = {'second_enqueue_tap': self.modules[NAMES[1]]}
                if injection == 'recovery':
                    values = {'warning_source': self.modules[NAMES[6]],
                              'observed_terminal_drop': self.modules[NAMES[7]]}
                self.modules[alias] = load(alias, ROOT / name, values)
        require(all(sys.modules.get(name) is module for name, module in self.modules.items()),
                'dependency_replaced')
        return tuple(self.modules[name] for name in NAMES)

    def bind(self, module: Any, replace: Any) -> None:
        if self.bound:
            require(self.bound[0] is module and module.scheduled is self.bound[2], 'factory_replaced')
            return
        before = module.scheduled
        def scheduled(load: Any, parent: type) -> type:
            self.claim()
            base = before(load, parent)
            candidates = [value for value in tuple(sys.modules.values())
                if getattr(value, '__file__', None) and Path(value.__file__).resolve() == ARRIVAL]
            require(len(candidates) == 1, 'arrival_owner')
            arrival = candidates[0]
            modules = self.dependencies(load)
            _, _, tap, binding, connection = modules[:5]
            return connection.connected(base, arrival.L, arrival.CORE.V1.N, tap, binding,
                                        services_factory=lambda: services(arrival, modules))
        self.bound[:] = (module, before, scheduled)
        replace(self.stack, module, 'scheduled', scheduled)


def install(stack: Any, bootstrap: Any, replace: Any, scope_getter: Any = None) -> Selection:
    selected = Selection(stack, scope_getter)
    original = bootstrap.load
    def load(alias: str, path: Any, injection: Any = None) -> Any:
        value = original(alias, path, injection)
        if Path(path).resolve() == PATCH:
            selected.bind(value, replace)
        return value
    for value in tuple(sys.modules.values()):
        if getattr(value, '__file__', None) and Path(value.__file__).resolve() == PATCH:
            selected.bind(value, replace)
    replace(stack, bootstrap, 'load', load)
    return selected


def install_owner(stack: Any, owner: Any, replace: Any, scope_getter: Any = None) -> None:
    """元bootstrapが凍結Boardを生成できる時点までload接続を遅延する。"""
    original, installed = owner.bootstrap, []
    def bootstrap() -> Any:
        value = original()
        if not installed:
            selected = install(stack, value, replace, scope_getter)
            installed.append((value, selected))
        else:
            require(installed[0][0] is value, 'bootstrap_owner_changed')
        return value
    replace(stack, owner, 'bootstrap', bootstrap)
