"""原v8へA3の同call HSV意味正規化だけを私有接続する。"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'g2_tracking_observation_runtime_2026-09-12_v8'
POLICY = ROOT.parent / 'g2_ui_hsv_background_candidate_2026-09-12_v1'
SNAPSHOT = ROOT.parents[2] / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30'
TEMPLATES = SNAPSHOT / 'models/ui_templates'
SPEC = importlib.util.spec_from_file_location('_g2_ui_hsv_base', BASE / 'owned_adapter.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
V5, V4 = A.A.A, A.A.A.A
PRIOR, START, SIDE, protect = A.PRIOR, A.START, A.SIDE, A.protect


def sources() -> tuple[Path, ...]:
    return tuple(sorted(set(A.sources()) | {ROOT / 'owned_adapter.py', BASE / 'owned_adapter.py',
                        POLICY / 'ui_background_policy.py', POLICY / 'live_connection.py',
                        SNAPSHOT / 'src/ui_mask.py'} | set(TEMPLATES.glob('x_mark*.png'))))


def attach(stack: Any, recovery: Any, output: Path) -> Any:
    owner = sys.modules[V4.OWNED_ALIAS]
    load = owner.bootstrap().load
    policy = load('_g2_ui_hsv_policy', POLICY / 'ui_background_policy.py')
    connection = load('_g2_ui_hsv_connection', POLICY / 'live_connection.py', {'ui_background_policy': policy})
    ui = sys.modules['src.ui_mask']
    V5.C.require(Path(ui.__file__).resolve() == SNAPSHOT / 'src/ui_mask.py', 'original_UI_module')
    matcher = ui.UiMaskMatcher.load_default(TEMPLATES)
    V5.C.require(len(matcher._templates) == 6, 'original_UI_templates')
    return connection.install(stack, recovery, matcher, output, side='1P')


def configured(stack: Any) -> Any:
    previous = V5.C.install
    def install(scope: Any, observer: Any, output: Path) -> Any:
        value = previous(scope, observer, output)
        state = observer.recovery.state
        V5.C.require('ui_hsv_background' not in state, 'duplicate_UI_connection')
        state['ui_hsv_background'] = attach(scope, observer.recovery, output)
        return value
    V4.replace_owned(stack, V5.C, 'install', install)
    return A.configured(stack)
