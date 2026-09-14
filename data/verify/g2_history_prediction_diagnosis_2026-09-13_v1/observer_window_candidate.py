"""実構成の窓定義だけを一括束縛し、原採録・拒否・復元を保持する。"""
from contextlib import ExitStack, contextmanager
from pathlib import Path
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT.parent
FIRST, LAST, STRIDE, OLD_LAST = 29052, 36900, 2, 36298
FRAMES = tuple(range(FIRST, LAST + STRIDE, STRIDE))
SIDES = ('1P', '2P')
PATHS = {
    'scope': VERIFY / 'g2_full_observation_scope_2026-09-09_v1/scope.py',
    'context': VERIFY / 'g2_provisional_context_capture_2026-09-09_v1/observer.py',
    'metadata': VERIFY / 'g2_collector_metadata_capture_2026-09-09_v1/observer.py',
}
COMPOSITION = VERIFY / 'g2_fifo_palette_composition_2026-09-09_v1/adapter.py'
FULL_CLI = VERIFY / 'g2_full_observation_scope_2026-09-09_v1/live_cli.py'


def owners(latest: Any) -> dict[str, Any]:
    function = latest.configuration.__wrapped__
    if Path(function.__code__.co_filename).resolve() != FULL_CLI:
        raise ValueError('observer_window_configuration_source')
    namespace = function.__globals__
    values = dict(scope=namespace['O'], context=namespace['L'].O,
                  metadata=latest.B.B.B.B.M.O)
    for key, module in values.items():
        if Path(module.__file__).resolve() != PATHS[key] or sys.modules.get(module.__name__) is not module:
            raise ValueError('observer_window_module_owner:' + key)
    return values


def bind(stack: Any, latest: Any, replace: Any) -> dict[str, Any]:
    values = owners(latest)
    scope, context, metadata = (values[k] for k in ('scope', 'context', 'metadata'))
    if (scope.WINDOWS, context.OUTER_FIRST, context.OUTER_LAST,
        metadata.FIRST, metadata.LAST, context.STRIDE, metadata.STRIDE) != (
        ((FIRST, OLD_LAST),), FIRST, OLD_LAST, FIRST, OLD_LAST, STRIDE, STRIDE):
        raise ValueError('observer_window_original_bounds')
    # 元scope.configuredが観測器へ展開する前の設定元だけを変更する。
    replace(stack, scope, 'WINDOWS', ((FIRST, LAST),))
    replace(stack, context, 'OUTER_LAST', LAST)
    replace(stack, metadata, 'LAST', LAST)
    return values


def install(stack: Any, main: Any, replace: Any) -> None:
    assembly = main.__globals__['S'].A
    original = assembly.session
    @contextmanager
    def session(inner: Any, a: Any, c: Any, modules: Any) -> Iterator[Any]:
        if Path(c.A.__file__).resolve() != COMPOSITION:
            raise ValueError('observer_window_composition_owner')
        previous = c.A.loaded_latest
        @contextmanager
        def loaded_latest() -> Iterator[Any]:
            with previous() as latest:
                # 値の復元は外側まで遅延。元moduleの削除/stream閉鎖は原順のまま。
                bind(stack, latest, replace)
                yield latest
        with ExitStack() as local:
            replace(local, c.A, 'loaded_latest', loaded_latest)
            with original(inner, a, c, modules) as value:
                yield value
    replace(stack, assembly, 'session', session)


def verify_state(state: dict[str, Any]) -> dict[str, Any]:
    pairs = [(frame, side) for frame in FRAMES for side in SIDES]
    pb, current, context = (state[k] for k in ('hidden_probability_observer',
        'current_scope_sink', 'provisional_context_observer'))
    if list(pb.expected) != pairs or list(current.expected) != pairs:
        raise ValueError('observer_window_inner_scope')
    if list(context.expected) != list(FRAMES) or list(context.pb_expected) != pairs:
        raise ValueError('observer_window_outer_scope')
    if state['private_publication_consumer'].recorder is not context:
        raise ValueError('observer_window_consumer_identity')
    metadata = state['collector_metadata_sink']
    original = type(metadata).__mro__[1].wrapper.__globals__
    if (original['FIRST'], original['LAST'], original['STRIDE']) != (FIRST, LAST, STRIDE):
        raise ValueError('observer_window_metadata_scope')
    return dict(first=FIRST, last=LAST, end_exclusive=LAST + STRIDE,
        updates=len(FRAMES), both_side_scopes=len(pairs), actual_state_checked=True,
        quality_gate_clear=False)
