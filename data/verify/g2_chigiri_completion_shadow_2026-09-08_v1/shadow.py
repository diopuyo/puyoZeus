"""正増分 TSUMO 退出の私有 SM preview。CPU 比較専用。"""
from __future__ import annotations

import copy
import hashlib
import inspect
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

SM_SHA = "5cde1678718bbde42987034fde4f0029055e3ce5f305c5745cb8731f620f9770"
COLORS = frozenset(range(1, 6))
VALID = COLORS | {0, 9, 10}
ROWS, COLS, PAIR_SIZE = 13, 6, 2
TRANSITION_FIRST_LINE = 1237
_ACTIVE = False


def digest(value: Any) -> str:
    """私有値全体の非干渉比較。公開 board SHA とは別。"""
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def grid(board: Any) -> list[list[int]] | None:
    if board is None:
        return None
    return [[int(board.get(r, c)) for c in range(COLS)] for r in range(ROWS)]


def exact_addition(before: Any, after: Any, gravity: Callable[..., None]) -> dict[str, Any]:
    """UNKNOWN は全体を未対応にし、支持や欠測補完には使わない。"""
    if before is None or after is None:
        return {"ok": False, "reason": "missing_board", "cells": []}
    changes = []
    for r in range(ROWS):
        for c in range(COLS):
            old, new = int(before.get(r, c)), int(after.get(r, c))
            if old not in VALID or new not in VALID:
                return {"ok": False, "reason": "invalid_cell", "cells": []}
            if 10 in (old, new):
                return {"ok": False, "reason": "unknown_unresolved", "cells": []}
            if old != new:
                if old != 0 or new not in COLORS:
                    return {"ok": False, "reason": "baseline_changed", "cells": []}
                changes.append((r, c, new))
    if len(changes) != PAIR_SIZE:
        return {"ok": False, "reason": "not_exact_two", "cells": changes}
    for name, value in (("baseline", before), ("observed", after)):
        private = value.copy()
        gravity(private)
        if private != value:
            return {"ok": False, "reason": name + "_unsupported", "cells": changes}
    return {"ok": True, "reason": "exact_two_supported", "cells": changes}


def scope_snapshot(sm: Any) -> dict[str, Any]:
    ctx = sm.context
    return {"state": ctx.state.value, "frame_idx": ctx.frame_idx, "time_sec": ctx.time_sec,
            "confirmed": grid(ctx.confirmed_board), "pending": grid(ctx.pending_board),
            "history": [grid(value) for value in ctx.non_stable_cnn_history],
            "next_queue": list(ctx.next_queue), "private_digest": digest(sm.__dict__)}


class Controller:
    def __init__(self, cls: type, original: Callable[..., Any], observer: Any = None) -> None:
        self.cls, self.original, self.observer = cls, original, observer
        self.module = sys.modules[cls.__module__]
        self.within = cls._update_within_current_state
        self.records: list[dict[str, Any]] = []
        self.sticky_error: str | None = None

    def emit(self, row: dict[str, Any]) -> None:
        self.records.append(copy.deepcopy(row))
        if self.observer is not None:
            self.observer(copy.deepcopy(row))

    def raw_check(self, sm: Any, signals: Any) -> dict[str, Any]:
        from src.next_slide_detector import validate_tsumo_placement
        result = exact_addition(sm.context.confirmed_board, signals.cnn_board,
                                self.module._apply_gravity_filter)
        queue = sm.context.next_queue
        pair = queue[-2] if len(queue) >= PAIR_SIZE else None
        known = isinstance(pair, (tuple, list)) and len(pair) == PAIR_SIZE
        known = known and all(type(value) is int and value in COLORS for value in pair)
        result["pair_used"] = tuple(pair) if known else None
        result["physical_pair_certified"] = False
        if result["ok"] and known:
            valid = validate_tsumo_placement(sm.context.confirmed_board, signals.cnn_board,
                                             pair, tolerance=0)
            if not valid.consistent:
                result.update(ok=False, reason="next_color_mismatch")
        return result

    def preview(self, sm: Any, state: Any, signals: Any) -> tuple[Any, dict[str, Any]]:
        before, signal_before = digest(sm.__dict__), digest(signals)
        cloned, private_signals = copy.deepcopy(sm), copy.deepcopy(signals)
        self.original(cloned, state, private_signals)
        if digest(sm.__dict__) != before or digest(signals) != signal_before:
            raise RuntimeError("preview_mutated_original")
        merged = exact_addition(sm.context.confirmed_board, cloned.context.confirmed_board,
                                self.module._apply_gravity_filter)
        return cloned, {"original_unchanged": True, "merge_check": merged,
                        "preview": scope_snapshot(cloned), "preview_calls": 1}

    def guarded_transition(self, sm: Any, state: Any, signals: Any) -> None:
        before, raw = scope_snapshot(sm), self.raw_check(sm, signals)
        row = {"before": before, "raw": grid(signals.cnn_board), "raw_check": raw,
               "preview_calls": 0, "live_transition_calls": 0, "within_calls": 0,
               "publication_allowed": False, "accounting_allowed": False}
        cloned = None
        if raw["ok"]:
            cloned, preview = self.preview(sm, state, signals)
            row.update(preview)
            match = preview["merge_check"]["ok"] and cloned.context.confirmed_board == signals.cnn_board
            row["reason"] = "accepted" if match else "merge_incomplete"
        else:
            row["reason"] = raw["reason"]
        if row["reason"] == "accepted":
            self.original(sm, state, signals)
            row["live_transition_calls"] = 1
            if scope_snapshot(sm) != scope_snapshot(cloned):
                raise RuntimeError("live_transition_preview_diverged")
        else:
            self.within(sm, signals)
            row["within_calls"] = 1
        row["after"] = scope_snapshot(sm)
        self.emit(row)

    def transition(self, sm: Any, state: Any, signals: Any) -> None:
        try:
            ctx = sm.context
            target = ctx.state == self.module.BoardState.TSUMO_FALL and state == self.module.BoardState.STABLE
            if not target or ctx.confirmed_board is None:
                return self.original(sm, state, signals)
            delta = signals.cnn_board.count_puyos() - ctx.confirmed_board.count_puyos()
            if delta <= 0:
                return self.original(sm, state, signals)
            self.guarded_transition(sm, state, signals)
        except Exception as exc:
            self.sticky_error = type(exc).__name__ + ":" + str(exc)
            raise


@contextmanager
def install(cls: type, observer: Any = None) -> Iterator[Controller]:
    """class 原本だけを保存し、instance closure を deep copy しない。"""
    global _ACTIVE
    path = Path(inspect.getfile(cls)).resolve()
    original = cls.__dict__["_apply_transition"]
    if _ACTIVE or hashlib.sha256(path.read_bytes()).hexdigest() != SM_SHA:
        raise RuntimeError("unfrozen_or_nested_install")
    if tuple(inspect.signature(original).parameters) != ("self", "new_state", "signals"):
        raise RuntimeError("transition_signature_mismatch")
    if original.__code__.co_filename != str(path) or original.__code__.co_firstlineno != TRANSITION_FIRST_LINE:
        raise RuntimeError("transition_code_mismatch")
    controller = Controller(cls, original, observer)
    def wrapped(sm: Any, new_state: Any, signals: Any) -> None:
        controller.transition(sm, new_state, signals)
    _ACTIVE = True
    cls._apply_transition = wrapped
    try:
        yield controller
    finally:
        cls._apply_transition = original
        _ACTIVE = False
