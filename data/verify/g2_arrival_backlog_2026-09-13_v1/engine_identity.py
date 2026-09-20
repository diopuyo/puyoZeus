"""CPU数学の主要な実依存/定数を固定する。実走全target guardを代替しない。"""
from __future__ import annotations

import hashlib
import inspect
import marshal
from pathlib import Path
import sys
from typing import Any


def callable_identity(function: Any) -> dict:
    actual = getattr(function, '__func__', function)
    path = Path(inspect.getsourcefile(actual)).resolve()
    return dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(marshal.dumps(actual.__code__)).hexdigest())


def fingerprint(engine: Any) -> dict:
    simulator, board = engine.B.ChainSimulator, engine.B.Board
    functions = dict(simulate=simulator.simulate, groups=simulator.find_erasable_groups,
        gravity=simulator.apply_gravity, prior=engine.H.uncalibrated_uniform,
        enumerate=engine.H.enumerate_hypotheses, apply=engine.H.apply_hypothesis,
        validate=engine.B.validate, board_from_dict=board.from_dict, board_set=board.set, board_copy=board.copy)
    return dict(functions={name: callable_identity(function) for name, function in functions.items()},
        constants=dict(max_worlds=engine.B.MAX_WORLDS, rows=engine.B.BOARD_ROWS, cols=engine.B.BOARD_COLS,
            hidden_rows=engine.B.HIDDEN_ROWS, mass_tolerance=engine.B.SUM_TOLERANCE),
        exclude_hidden_row_from_pop=True, python_version=list(sys.version_info[:3]),
        identity_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        full_runtime_target_guard_required=True)
