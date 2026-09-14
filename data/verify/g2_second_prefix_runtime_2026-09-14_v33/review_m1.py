"""元の保存検査を再利用し、採録終端だけを子と同じ限定観測区間へ束縛する。"""
from __future__ import annotations

from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
import owned_adapter as A
W = A.W

SOURCE = W.ROOT.parent / 'g2_m1_second_runtime_2026-09-13_v21/review_m1.py'


def inspect(output: Path) -> dict:
    """旧source/資格/採録数を変えず、成功票と全metadataを元検査へ渡す。"""
    with ExitStack() as stack:
        previous = list(sys.path)
        stack.callback(sys.path.__setitem__, slice(None), previous)
        spec = importlib.util.spec_from_file_location('_split_original_m1_review', SOURCE)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        replace = A.A.A.A.V4.replace_owned
        original.WHOLE.require(original.Q.S.END == original.WHOLE.LAST == W.OLD_LAST,
                               'split_original_bounds')
        replace(stack, original.Q.S, 'END', W.LAST)
        replace(stack, original.WHOLE, 'LAST', W.LAST)
        return original.inspect(output)
