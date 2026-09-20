"""NEXT一時中断をformula実前進だけで回復するv2契約。"""

from __future__ import annotations

import ast
import contextlib
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from typing import Any

import pytest

from scripts import chain_end_epoch_shadow_v2 as shadow


def grid(value: int = 0) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(value for _ in range(6)) for _ in range(13))


class Board:
    def __init__(self, value: Any) -> None:
        self.value = value

    def copy(self) -> "Board":
        return Board(tuple(tuple(row) for row in self.value))

    def to_dict(self) -> dict[str, Any]:
        return {"grid": [list(row) for row in self.value]}


def point(frame: int) -> Any:
    return NS(frame_idx=frame, time_sec=frame / 60)


def formula(step: int, total: int, product: int | None, frame: int, seq: int) -> Any:
    return NS(session_id=2, step_index=step, total_power=total,
              step_product=product, observed_at=point(frame), ledger_sequence=seq,
              source="formula_reset/update" if step == 0 else "formula_observation")


def score(frame: int, raw: int, previous: int, delta: int) -> Any:
    return NS(observed_at=point(frame), raw_value=raw, prev_score=previous,
              cur_score=raw, delta=delta, is_valid=True,
              source="ScoreDelta/_apply_read")


def make_snapshot() -> Any:
    generation = NS(side="2P", reset_epoch=1, action_revision=4)
    origin = NS(prediction_revision=1, episode_revision=1, is_origin_reference=True,
                scope=NS(value="total_from_instance_origin"), final_grid=grid(),
                final_sha256="f" * 64, chain_count=3, calculated_total_score=140)
    return NS(status=NS(value="provisional"), generation=generation,
              origin_prediction_revision=1, predictions=(origin,),
              formula_session_binding=NS(session_id=2, ledger_sequence=5,
                                         source="formula_reset/update"),
              formula_evidence=(formula(0, 0, None, 80, 5),
                                formula(1, 40, 40, 86, 80),
                                formula(2, 100, 60, 96, 105)),
              raw_score_evidence=(score(86, 140, 100, 40),
                                  score(96, 200, 140, 60)))


class Recorder:
    def __init__(self) -> None:
        self.snap = make_snapshot()
        self.handle = NS(instance_id=7)
        self.active_handles = {"2P": self.handle}
        self.ledger = NS(snapshot=lambda handle: self.snap)
        self.generation_recorder = object()
        identity = NS(side="2P", reset_epoch=1, action_revision=4)
        candidate = NS(identity=identity, final_grid=grid())
        self.active_candidates = {"2P": {
            "instance_id": 7, "status": "pending",
            "transaction": NS(state=NS(value="pending")), "candidate": candidate}}
        self.rows: list[dict[str, Any]] = []
        self.frame, self.time_sec = 100, 100 / 60
        self.clock_active = True

    def _clock_active(self) -> bool:
        return self.clock_active

    def _current_generation(self, side: str) -> Any:
        return self.snap.generation

    def emit(self, row: dict[str, Any]) -> None:
        self.rows.append(row)


class Pipe:
    def __init__(self) -> None:
        self._chain_start_next_2p = (5, 5)
        self._last_seen_next_2p = (5, 5)
        self._active_chain_2p = NS(chain_count=3)
        self._last_chain_event_for_settle_2p = None

    def _stash_and_clear_active_chain(self, side: str) -> None:
        self._last_chain_event_for_settle_2p = self._active_chain_2p
        self._active_chain_2p = None


def call(
    observer: Any, rec: Recorder, pipe: Pipe, frame: int, *, slide: bool = False,
    current: Any = (5, 5), start: Any = (5, 5), active: bool = True,
    event: Any = NS(chain_count=3), board: Any | None = None,
) -> Any:
    rec.frame, rec.time_sec = frame, frame / 60
    pipe._last_seen_next_2p, pipe._chain_start_next_2p = current, start
    pipe._active_chain_2p = NS(chain_count=3) if active else None
    sm = NS(context=NS(state=NS(value="chain")))
    return observer.before_exit(pipe, "2P", frame, frame / 60, True,
                                Board(grid()) if board is None else board, event,
                                current, start, slide, sm)


def add_step3(rec: Recorder, frame: int = 110, seq: int = 149) -> None:
    rec.snap.formula_evidence += (formula(3, 140, 40, frame, seq),)
    rec.snap.raw_score_evidence += (score(frame, 240, 200, 40),
                                    score(frame + 2, 240, 240, 0))


@pytest.mark.parametrize(("slide", "current", "start", "reason"), [
    (True, (5, 5), (5, 5), "next_slide_progressed"),
    (False, None, (5, 5), "next_evidence_unknown"),
    (False, (1, 1), (5, 5), "next_value_progressed"),
])
def test_next_anomaly_is_pending_not_sticky(
    slide: bool, current: Any, start: Any, reason: str,
) -> None:
    rec, pipe = Recorder(), Pipe()
    observer = shadow.ChainEndEpochObserver(rec)
    assert call(observer, rec, pipe, 100, slide=slide, current=current, start=start) is None
    assert rec.rows[-1]["reason"] == reason
    assert ("2P", 7) in observer._disruptions
    assert ("2P", 7) not in observer._invalidated


def test_34920_recapture_same_step_does_not_recover() -> None:
    rec, pipe = Recorder(), Pipe()
    observer = shadow.ChainEndEpochObserver(rec)
    call(observer, rec, pipe, 100, slide=True)
    rec.snap.formula_evidence += (formula(2, 100, 60, 102, 123),)
    assert call(observer, rec, pipe, 102) is None
    assert rec.rows[-1]["reason"] == "next_disruption_pending"


def test_34968_strict_formula_step_recovers_then_restarts_votes() -> None:
    rec, pipe = Recorder(), Pipe()
    observer = shadow.ChainEndEpochObserver(rec)
    call(observer, rec, pipe, 100, slide=True)
    add_step3(rec, 110, 149)
    assert call(observer, rec, pipe, 110) is None
    assert rec.rows[-1]["reason"] == "origin_final_needs_second_observation"
    assert any(row["reason"] == "next_disruption_recovered_by_formula_step"
               for row in rec.rows)
    evidence = call(observer, rec, pipe, 112)
    assert evidence is not None and evidence.consecutive_full_grid == 2


@pytest.mark.parametrize("mutation", [
    "session", "source", "old_clock", "same_step", "seq", "bad_frame", "bad_time",
])
def test_false_formula_progress_does_not_recover(mutation: str) -> None:
    rec, pipe = Recorder(), Pipe()
    observer = shadow.ChainEndEpochObserver(rec)
    call(observer, rec, pipe, 100, slide=True)
    row = formula(3, 140, 40, 110, 149)
    if mutation == "session": row.session_id = 3
    if mutation == "source": row.source = "formula_reset/update"
    if mutation == "old_clock": row.observed_at = point(98)
    if mutation == "same_step": row.step_index = 2
    if mutation == "seq": row.ledger_sequence = 100
    if mutation == "bad_frame": row.observed_at.frame_idx = False
    if mutation == "bad_time": row.observed_at.time_sec = False
    rec.snap.formula_evidence += (row,)
    assert call(observer, rec, pipe, 110) is None
    assert rec.rows[-1]["reason"] == "next_disruption_pending"


def test_35786_true_exit_cannot_reacquire_when_next_returns() -> None:
    rec, pipe = Recorder(), Pipe()
    observer = shadow.ChainEndEpochObserver(rec)
    call(observer, rec, pipe, 100, slide=True)
    assert call(observer, rec, pipe, 102, current=(1, 1)) is None
    assert call(observer, rec, pipe, 104, current=(5, 5)) is None
    assert rec.rows[-1]["reason"] == "next_disruption_pending"


def test_generation_change_is_permanently_invalid() -> None:
    rec, pipe = Recorder(), Pipe()
    observer = shadow.ChainEndEpochObserver(rec)
    original = rec.snap.generation
    rec._current_generation = lambda side: NS(side=side, reset_epoch=2, action_revision=5)
    assert call(observer, rec, pipe, 100) is None
    rec._current_generation = lambda side: original
    add_step3(rec)
    assert call(observer, rec, pipe, 110) is None
    assert rec.rows[-1]["reason"] == "end_epoch_invalidated"


def test_stash_without_ready_creates_disruption() -> None:
    rec, pipe = Recorder(), Pipe()
    observer = shadow.ChainEndEpochObserver(rec)
    observer.note_active_stash("2P")
    disruption = observer._disruptions[("2P", 7)]
    assert disruption.reason == "active_chain_stashed"
    assert disruption.max_step_index == 2
    assert rec.rows[-1]["reason"] == "active_chain_stashed"
    assert rec.rows[-1]["barrier_after"] == disruption.__dict__
    assert rec.rows[-1]["commit_permission_issued"] is False


def test_persistent_disruption_refreshes_recovery_barrier() -> None:
    rec, pipe = Recorder(), Pipe()
    observer = shadow.ChainEndEpochObserver(rec)
    call(observer, rec, pipe, 100, slide=True)
    add_step3(rec, 108, 149)
    call(observer, rec, pipe, 102, current=None)
    call(observer, rec, pipe, 110, current=(1, 1))
    barrier = observer._disruptions[("2P", 7)]
    assert barrier.frame_idx == 110 and barrier.max_step_index == 3
    assert call(observer, rec, pipe, 112) is None
    assert rec.rows[-1]["reason"] == "next_disruption_pending"


def test_true_stash_refreshes_barrier_after_old_formula_progress() -> None:
    rec, pipe = Recorder(), Pipe()
    observer = shadow.ChainEndEpochObserver(rec)
    call(observer, rec, pipe, 100, slide=True)
    add_step3(rec, 108, 149)
    rec.frame, rec.time_sec = 120, 2.0
    observer.note_active_stash("2P")
    barrier = observer._disruptions[("2P", 7)]
    assert barrier.frame_idx == 120 and barrier.max_step_index == 3
    assert call(observer, rec, pipe, 122) is None
    assert rec.rows[-1]["reason"] == "next_disruption_pending"


def test_recovery_still_requires_current_next_match() -> None:
    rec, pipe = Recorder(), Pipe()
    observer = shadow.ChainEndEpochObserver(rec)
    call(observer, rec, pipe, 100, slide=True)
    add_step3(rec)
    assert call(observer, rec, pipe, 110, current=(1, 1)) is None
    assert rec.rows[-1]["reason"] == "next_value_progressed"


def test_install_records_unready_stash_and_restores() -> None:
    rec = Recorder()
    module = ModuleType("tests._chain_end_v2_pipeline")
    module._is_game_event_chain_exit = lambda **kwargs: False
    Pipe.__module__ = module.__name__
    module.Pipeline = Pipe
    sys.modules[module.__name__] = module
    original_stash = Pipe._stash_and_clear_active_chain
    try:
        with contextlib.ExitStack() as stack:
            observer = shadow.install(stack, NS(RecognitionPipeline=Pipe), rec)
            pipe = Pipe()
            pipe._stash_and_clear_active_chain("2P")
            assert ("2P", 7) in observer._disruptions
    finally:
        sys.modules.pop(module.__name__, None)
        Pipe.__module__ = __name__
    assert Pipe._stash_and_clear_active_chain is original_stash


def test_install_does_not_record_noop_stash_without_active() -> None:
    rec = Recorder()
    module = ModuleType("tests._chain_end_v2_noop_stash")
    module._is_game_event_chain_exit = lambda **kwargs: False
    Pipe.__module__ = module.__name__
    module.Pipeline = Pipe
    sys.modules[module.__name__] = module
    try:
        with contextlib.ExitStack() as stack:
            observer = shadow.install(stack, NS(RecognitionPipeline=Pipe), rec)
            pipe = Pipe()
            pipe._active_chain_2p = None
            pipe._stash_and_clear_active_chain("2P")
            assert observer._disruptions == {}
            assert rec.rows == []
    finally:
        sys.modules.pop(module.__name__, None)
        Pipe.__module__ = __name__


def test_invalid_formula_snapshot_stash_fails_closed() -> None:
    rec, pipe = Recorder(), Pipe()
    rec.snap.formula_evidence = ()
    observer = shadow.ChainEndEpochObserver(rec)
    observer.note_active_stash("2P")
    assert ("2P", 7) in observer._invalidated


def test_function_size_and_guard_contract() -> None:
    tree = ast.parse(Path(shadow.__file__).read_text(encoding="utf-8"))
    functions = [node for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert max(node.end_lineno - node.lineno + 1 for node in functions) <= 50
    assert shadow.GUARD_PATHS == ("scripts/chain_end_epoch_shadow_v1.py",)
