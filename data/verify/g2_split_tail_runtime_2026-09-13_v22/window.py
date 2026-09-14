"""A限定観測の範囲。初回基準取得期限・安定条件・二M1要求は変更しない。"""
from __future__ import annotations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
FIRST, LAST, STRIDE = 29052, 36900, 2
OLD_LAST = 36298
COMMON = ROOT.parent / 'g2_history_publication_probe_runtime_2026-09-10_v13/common.py'
CANDIDATE = ROOT.parent / 'g2_m1_completion_candidate_2026-09-12_v1'


def common(stack: Any, module: Any, replace: Any) -> None:
    if Path(module.__file__).resolve() != COMMON or module.FIRST != FIRST or module.STRIDE != STRIDE:
        raise ValueError('split_window_common_owner')
    if module.END != OLD_LAST + STRIDE or module.FRAMES[-1] != OLD_LAST:
        raise ValueError('split_window_original_bounds')
    replace(stack, module, 'END', LAST + STRIDE)
    replace(stack, module, 'FRAMES', tuple(range(FIRST, LAST + STRIDE, STRIDE)))


def load_wrapper(stack: Any, bootstrap: Any, replace: Any) -> None:
    original, changed = bootstrap.load, {}
    def load(alias: str, path: Path, injection: Any = None) -> Any:
        value = original(alias, path, injection)
        target = Path(path).resolve()
        if target in (CANDIDATE / 'capture_schedule.py', CANDIDATE / 'schedule_saved.py'):
            if value not in changed:
                if value.END != OLD_LAST:
                    raise ValueError('split_window_original_schedule')
                replace(stack, value, 'END', LAST)
                changed[value] = LAST
            elif value.END != LAST:
                raise ValueError('split_window_schedule_changed')
        return value
    replace(stack, bootstrap, 'load', load)
