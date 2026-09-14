"""既存39更新と式constructor入力を再用する前段診断。原停止は保持する。"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
import sys
from types import FunctionType
from typing import Any
import front_probe as F

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent/'g2_chain_firing_continuous_cpu_2026-09-10_v1'
sys.path.insert(0, str(PRIOR))
import run_cpu_v2 as V2
R = V2.R


def execute(output: Path) -> Any:
    rows: list[Any] = []
    try:
        with ExitStack() as stack:
            F.installed(stack, R.D.O, rows)
            return V2.execute(output)
    finally:
        R.K.write(output/'FRONT.json', rows)


def accepted(output: Path) -> Any:
    value = R.accepted(output)
    rows = R.K.read(output/'FRONT.json')
    assert len(rows) == 1 and rows[0]['infer']['inferred'] == rows[0]['raw']
    row = rows[0]
    assert row['score_value'] is None and row['formula_valid'] is True
    assert row['infer']['actual']['chains'] == 0 and row['infer']['missing_control']['chains'] == 1
    value.update(front_actual_caller=True, inference_matches_raw=True, score_missing_is_zero=True,
        origin_repair_closed=False, prediction_only_control=True)
    return value


def main() -> int:
    return FunctionType(R.main.__code__, dict(vars(R), ROOT=ROOT, execute=execute, accepted=accepted))()


if __name__ == '__main__':
    raise SystemExit(main())
