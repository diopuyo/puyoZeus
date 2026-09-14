"""cold loaderの原Mode生成後にだけprefix接続を設置。旧LIFEは変更しない。"""
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
BACK = ROOT.parent / 'g2_arrival_backlog_2026-09-13_v1'
UNBOUND = ROOT.parent / 'g2_m1_unbound_start_2026-09-13_v1'
SECOND = ROOT.parent / 'g2_m1_completion_candidate_2026-09-12_v1/second_mode_binding.py'
MODE = ROOT.parent / 'g2_arrival_ack_candidate_2026-09-12_v1/arrival_mode.py'
FINAL = MODE.parent / 'arrival_final_saved.py'
ALIAS = '_g2_prefix_unbound_second_binding'
METHODS = ('observe', 'stable', 'capture_origin', 'progress', 'new_origin', 'close')
PRIVATE_NAMES = tuple('_g2_prefix_' + name for name in
    ('split', 'math', 'engine', 'saved', 'origin', 'commit', 'lane', 'stable', 'decision', 'initial_origin', 'live', 'replay', 'final'))


def private_loader(stack: Any, original: Any) -> Any:
    if any(name in sys.modules for name in PRIVATE_NAMES):
        raise ValueError('prefix_private_foreign_alias')
    owned: dict[str, Any] = {}
    def release(kind: Any, body: Any, trace: Any) -> bool:
        changed = []
        for alias, module in reversed(tuple(owned.items())):
            current = sys.modules.get(alias)
            if current is module:
                sys.modules.pop(alias)
            elif current is not None:
                changed.append(alias)
        owned.clear()
        if changed and body is None:
            raise ValueError('prefix_private_alias_changed:' + ','.join(changed))
        return False
    stack.push(release)
    def load(alias: str, path: Any, injection: Any = None) -> Any:
        if alias not in PRIVATE_NAMES:
            raise ValueError('prefix_private_alias_out_of_scope')
        if alias in owned and sys.modules.get(alias) is not owned[alias]:
            raise ValueError('prefix_private_alias_changed:' + alias)
        try:
            return original(alias, path, injection)
        finally:
            module = sys.modules.get(alias)
            actual = getattr(module, '__file__', None)
            if alias not in owned and isinstance(actual, (str, Path)) and Path(actual).resolve() == Path(path).resolve():
                owned[alias] = module
    return load


def dependencies(load: Any) -> Any:
    """専用aliasと明示注入で旧module名を汚染せず同一Family型を共有する。"""
    split = load('_g2_prefix_split', BACK / 'split_landing.py')
    math = load('_g2_prefix_math', BACK / 'prefix_phase_math_v2.py', {'split_landing': split})
    engine = load('_g2_prefix_engine', BACK / 'engine_identity.py')
    saved = load('_g2_prefix_saved', BACK / 'prefix_phase_saved_v4.py',
        dict(prefix_phase_math_v2=math, split_landing=split, engine_identity=engine))
    origin = load('_g2_prefix_origin', BACK / 'identified_origin_candidate.py', {'prefix_phase_math_v2': math})
    commit = load('_g2_prefix_commit', ROOT / 'prefix_commit.py', {'prefix_phase_math_v2': math})
    lane = load('_g2_prefix_lane', ROOT / 'lane_state.py',
        dict(prefix_phase_math_v2=math, prefix_phase_saved_v4=saved, prefix_commit=commit))
    stable = load('_g2_prefix_stable', ROOT / 'stable_evidence.py')
    decision = load('_g2_prefix_decision', ROOT / 'stable_decision.py')
    initial = load('_g2_prefix_initial_origin', ROOT / 'prefix_origin_initial.py')
    return load('_g2_prefix_live', ROOT / 'prefix_live_adapter.py',
        dict(identified_origin_candidate=origin, lane_state=lane, stable_evidence=stable,
            stable_decision=decision, prefix_origin_initial=initial))


def methods(value: Any) -> tuple:
    return (value,) + tuple(getattr(value.Mode, name) for name in METHODS)


def finish_selection(stack: Any, load: Any, replace: Any, value: Any,
                     adapter: Any, mode: Any, saved: list) -> None:
    if saved:
        if saved[0] != (value, value.verify):
            raise ValueError('prefix_final_owner_changed')
        return
    replay = load('_g2_prefix_replay', ROOT / 'prefix_replay.py', dict(lane_state=adapter.N,
        prefix_live_adapter=adapter, prefix_commit=adapter.N.C, stable_evidence=adapter.E,
        stable_decision=adapter.D, prefix_origin_initial=adapter.I))
    final = load('_g2_prefix_final', ROOT / 'prefix_final_saved.py', {'prefix_replay': replay})
    def verify(state: dict, serializer: Any, native: Any) -> dict:
        return final.verify(value, mode, state, serializer, native)
    replace(stack, value, 'verify', verify)
    saved.append((value, value.verify))


def second_selection(stack: Any, original: Any, replace: Any, value: Any, owned: list) -> None:
    if owned and sys.modules.get(ALIAS) is not owned[0]:
        raise ValueError('prefix_unbound_alias_changed')
    try:
        patched = original(ALIAS, UNBOUND / 'second_mode_binding.py')
    finally:
        partial = sys.modules.get(ALIAS)
        partial_path = getattr(partial, '__file__', None)
        if not owned and isinstance(partial_path, (str, Path)) and Path(partial_path).resolve() == UNBOUND / 'second_mode_binding.py':
            owned.append(partial)
    if value.session_class is not patched.session_class:
        replace(stack, value, 'session_class', patched.session_class)


def install_load(stack: Any, bootstrap: Any, replace: Any) -> None:
    original = bootstrap.load
    if ALIAS in sys.modules:
        raise ValueError('prefix_unbound_foreign_alias')
    private = private_loader(stack, original)
    owned: list[Any] = []
    observed: list[tuple] = []
    adapters, saved = [], []
    def release(kind: Any, body: Any, trace: Any) -> bool:
        if owned and sys.modules.get(ALIAS) is owned[0]:
            sys.modules.pop(ALIAS)
        elif owned and ALIAS in sys.modules and body is None:
            raise ValueError('prefix_unbound_alias_changed')
        return False
    stack.push(release)
    def load(alias: str, path: Any, injection: Any = None) -> Any:
        value = original(alias, path, injection)
        resolved = Path(path).resolve()
        if resolved == MODE:
            if not observed:
                adapter = dependencies(private)
                adapter.install(stack, value, replace)
                observed.append(methods(value))
                adapters.append(adapter)
            elif observed[0] != methods(value):
                raise ValueError('prefix_mode_owner_changed')
        if resolved == FINAL:
            if not adapters:
                raise ValueError('prefix_saved_before_mode')
            finish_selection(stack, private, replace, value, adapters[0], observed[0][0], saved)
        if resolved == SECOND:
            second_selection(stack, original, replace, value, owned)
        return value
    replace(stack, bootstrap, 'load', load)


def install(stack: Any, owner: Any, replace: Any) -> None:
    original = owner.bootstrap
    installed: list[Any] = []
    def bootstrap() -> Any:
        value = original()
        if not installed:
            install_load(stack, value, replace)
            installed.append(value)
        elif installed[0] is not value:
            raise ValueError('prefix_bootstrap_owner_changed')
        return value
    replace(stack, owner, 'bootstrap', bootstrap)
