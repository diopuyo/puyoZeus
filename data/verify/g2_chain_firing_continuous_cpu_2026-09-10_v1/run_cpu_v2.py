"""保存 flag を constructor へ渡す新版。v1 source/失敗票は変更しない。"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
from types import FunctionType, SimpleNamespace
from typing import Any
import constructor_input as C
import run_cpu as R


def execute(output: Path) -> Any:
    evidence: dict[str, Any] = {}
    try:
        with ExitStack() as stack, C.installed(stack, R.K, evidence):
            return R.execute(output)
    finally:
        R.K.write(output/'CONSTRUCTOR_INPUT.json', evidence)


def guards() -> dict[str, str]:
    plan = R.VERIFY/'video38_history_publication_probe_live_2026-09-10_v6/PLAN.json'
    return R.K.guards() | {str(plan): R.K.sha(plan), str(R.ROOT/'PLAN_V2.md'): R.K.sha(R.ROOT/'PLAN_V2.md')}


def main() -> int:
    common = SimpleNamespace(**(vars(R.K) | {'guards': guards}))
    return FunctionType(R.main.__code__, dict(vars(R), K=common, execute=execute))()


if __name__ == '__main__':
    raise SystemExit(main())
