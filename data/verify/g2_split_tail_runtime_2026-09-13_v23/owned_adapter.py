"""凍結v22を保持し、実envのRecorder排他上限のみ追加接続する。"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
from typing import Any
import history_bound as H

ROOT = Path(__file__).resolve().parent
PREVIOUS_ROOT = ROOT.parent / 'g2_split_tail_runtime_2026-09-13_v22'
SPEC = importlib.util.spec_from_file_location('_history_bound_previous_v22', PREVIOUS_ROOT / 'owned_adapter.py')
PREVIOUS = importlib.util.module_from_spec(SPEC)
_paths = list(sys.path)
try:
    sys.path.insert(0, str(PREVIOUS_ROOT))
    SPEC.loader.exec_module(PREVIOUS)
finally:
    sys.path[:] = _paths
A, W = PREVIOUS.A, PREVIOUS.W
PRIOR, START, SIDE, protect = PREVIOUS.PRIOR, PREVIOUS.START, PREVIOUS.SIDE, PREVIOUS.protect
CANDIDATE, SAFETY, BASE = PREVIOUS.CANDIDATE, PREVIOUS.SAFETY, PREVIOUS.BASE
UNBOUND_MODULE, NOTICE = PREVIOUS.UNBOUND_MODULE, PREVIOUS.NOTICE
PREVIOUS_ADAPTER = PREVIOUS.PREVIOUS_ADAPTER


def sources() -> tuple[Path, ...]:
    return tuple(sorted(set(PREVIOUS.sources()) | {ROOT / 'owned_adapter.py', ROOT / 'history_bound.py'}))


def configured(stack: Any) -> Any:
    selected = PREVIOUS.configured(stack)
    assert (W.FIRST, W.LAST, W.STRIDE) == (H.FIRST, H.LAST, H.STRIDE)
    H.install(stack, selected, A.A.A.V4.replace_owned)
    return selected
