"""連鎖終了入口shadowのCPU契約。実盤面公開・commitは扱わない。"""

from __future__ import annotations

import contextlib
import inspect
import sys
from types import ModuleType, SimpleNamespace as NS
from typing import Any

import pytest

from scripts import chain_end_epoch_shadow_v1 as shadow


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


def formula(step: int, total: int, product: int | None, frame: int,
            sequence: int | None = None) -> Any:
    return NS(session_id=2, step_index=step, total_power=total,
              step_product=product, observed_at=point(frame),
              ledger_sequence=step if sequence is None else sequence,
              source="formula_reset/update" if step == 0 else "formula_observation")


def score(frame: int, raw: int, prev: int, delta: int) -> Any:
    return NS(observed_at=point(frame), raw_value=raw, prev_score=prev,
              cur_score=raw, delta=delta, is_valid=True, source="ScoreDelta/_apply_read")


def snapshot(*, final: Any | None = None, generation: Any | None = None) -> Any:
    final = grid() if final is None else final
    generation = generation or NS(side="2P", reset_epoch=1, action_revision=4)
    origin = NS(prediction_revision=1, episode_revision=1, is_origin_reference=True,
                scope=NS(value="total_from_instance_origin"), final_grid=final,
                final_sha256="f" * 64, chain_count=2, calculated_total_score=100)
    return NS(status=NS(value="provisional"), generation=generation,
              origin_prediction_revision=1, predictions=(origin,),
              formula_session_binding=NS(session_id=2, ledger_sequence=5,
                                         source="formula_reset/update"),
              formula_evidence=(formula(0, 0, None, 5, 5),
                                formula(1, 40, 40, 10), formula(2, 100, 60, 20)),
              raw_score_evidence=(score(30, 140, 100, 40),
                                  score(40, 200, 140, 60),
                                  score(50, 200, 200, 0)))


class Recorder:
    def __init__(self, snap: Any | None = None) -> None:
        self.snap = snap or snapshot()
        self.handle = NS(instance_id=7)
        self.active_handles = {"2P": self.handle}
        self.ledger = NS(snapshot=lambda handle: self.snap)
        self.generation_recorder = object()
        tx = NS(state=NS(value="pending"))
        identity = NS(side="2P", reset_epoch=1, action_revision=4)
        candidate = NS(identity=identity, final_grid=grid())
        self.active_candidates = {"2P": {"instance_id": 7, "status": "pending",
                                         "transaction": tx, "candidate": candidate}}
        self.rows: list[dict[str, Any]] = []
        self.clock_active = True
        self.frame, self.time_sec = 50, 50 / 60

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
        self._active_chain_2p = NS(name="chain")
        self._last_chain_event_for_settle_2p = None
        self.stash_calls: list[str] = []

    def _stash_and_clear_active_chain(self, side: str) -> None:
        self.stash_calls.append(side)
        self._last_chain_event_for_settle_2p = self._active_chain_2p
        self._active_chain_2p = None


def step_signature() -> inspect.Signature:
    def step(pipe: Any, side: str, frame_idx: int, time_sec: float,
             is_active: bool, cnn_board: Any, chain_event: Any, *,
             sm: Any = None, next_pair: Any = None, slide_motion: bool = False) -> Any:
        return None
    return inspect.signature(step)


def bound(frame: int = 50, board: Any | None = None, **changes: Any) -> Any:
    values = dict(side="2P", frame_idx=frame, time_sec=frame / 60,
                  is_active=True, cnn_board=Board(grid()) if board is None else board,
                  chain_event=NS(chain_count=13), sm=NS(context=NS(state=NS(value="chain"))),
                  next_pair=(5, 5), slide_motion=False)
    values.update(changes)
    return step_signature().bind(NS(), **values)


def ready_observer() -> tuple[Any, Recorder, Pipe]:
    rec, pipe = Recorder(), Pipe()
    return shadow.ChainEndEpochObserver(rec), rec, pipe


def observe(observer: Any, rec: Recorder, pipe: Pipe, value: Any) -> Any:
    rec.frame = value.arguments["frame_idx"]
    rec.time_sec = value.arguments["time_sec"]
    args = value.arguments
    suffix = "1p" if args["side"] == "1P" else "2p"
    return observer.before_exit(
        pipe, args["side"], args["frame_idx"], args["time_sec"], args["is_active"],
        args["cnn_board"], args["chain_event"], getattr(pipe, f"_last_seen_next_{suffix}"),
        getattr(pipe, f"_chain_start_next_{suffix}"),
        args["slide_motion"], args["sm"],
    )


def test_two_current_observations_close_once() -> None:
    observer, rec, pipe = ready_observer()
    assert observe(observer, rec, pipe, bound(frame=50)) is None
    rec.snap.raw_score_evidence += (score(52, 200, 200, 0),)
    evidence = observe(observer, rec, pipe, bound(frame=52))
    assert evidence == shadow.EndEvidence(7, "2P", 1, 2, 100, 200, "f" * 64, 2)
    observer.mark_closed("2P")
    assert observe(observer, rec, pipe, bound(frame=54)) is None
    assert rec.rows[-1]["reason"] == "already_closed_once"


@pytest.mark.parametrize(("mutation", "reason"), [
    ("clock", "step_and_recorder_clock_mismatch"),
    ("status", "ledger_not_provisional"),
    ("generation", "software_generation_mismatch"),
    ("owner", "pending_owner_missing_or_mismatch"),
    ("transaction", "pending_transaction_not_pending"),
    ("candidate_generation", "pending_candidate_generation_mismatch"),
    ("candidate_final", "pending_candidate_not_origin_final"),
    ("origin", "origin_prediction_missing"),
    ("origin_scope", "origin_prediction_missing"),
    ("initial_session", "initial_formula_session_reset_missing"),
    ("formula_missing", "formula_steps_incomplete"),
    ("formula_total", "formula_total_not_origin_total"),
    ("formula_prefix", "formula_prefix_not_causal"),
    ("score_delta", "score_deltas_not_formula_products"),
    ("score_current", "current_raw_score_not_origin_total"),
    ("unknown", "current_full_grid_unknown"),
    ("different_grid", "current_grid_not_origin_final"),
])
def test_fail_closed_reasons(mutation: str, reason: str) -> None:
    observer, rec, pipe = ready_observer()
    arg = bound()
    if mutation == "clock": rec.clock_active = False
    if mutation == "status": rec.snap.status = NS(value="invalidated")
    if mutation == "generation": rec._current_generation = lambda side: NS(side=side)
    if mutation == "owner": rec.active_candidates.clear()
    if mutation == "transaction": rec.active_candidates["2P"]["transaction"].state.value = "prepared"
    if mutation == "candidate_generation":
        rec.active_candidates["2P"]["candidate"].identity.action_revision = 5
    if mutation == "candidate_final": rec.active_candidates["2P"]["candidate"].final_grid = grid(1)
    if mutation == "origin": rec.snap.origin_prediction_revision = None
    if mutation == "origin_scope": rec.snap.predictions[0].scope.value = "remaining_from_revision"
    if mutation == "initial_session": rec.snap.formula_evidence = rec.snap.formula_evidence[1:]
    if mutation == "formula_missing": rec.snap.formula_evidence = rec.snap.formula_evidence[:1]
    if mutation == "formula_total": rec.snap.formula_evidence[2].total_power = 99
    if mutation == "formula_prefix": rec.snap.formula_evidence[1].total_power = 39
    if mutation == "score_delta": rec.snap.raw_score_evidence[0].delta = 41
    if mutation == "score_current": rec.snap.raw_score_evidence[-1].raw_value = 199
    if mutation == "unknown":
        bad = [list(row) for row in grid()]; bad[0][0] = 10
        arg.arguments["cnn_board"] = Board(tuple(tuple(row) for row in bad))
    if mutation == "different_grid": arg.arguments["cnn_board"] = Board(grid(1))
    assert observe(observer, rec, pipe, arg) is None
    assert rec.rows[-1]["reason"] == reason
    assert all(row.get("commit_permission_issued") is False for row in rec.rows)


def test_conflicting_formula_revision_is_rejected() -> None:
    observer, rec, pipe = ready_observer()
    rec.snap.formula_evidence += (formula(1, 41, 41, 11),)
    assert observe(observer, rec, pipe, bound()) is None
    assert rec.rows[-1]["reason"] == "formula_step_revision_conflict"


def test_active_chain_must_still_exist() -> None:
    observer, rec, pipe = ready_observer()
    pipe._active_chain_2p = None
    assert observe(observer, rec, pipe, bound()) is None
    assert rec.rows[-1]["reason"] == "active_chain_missing"


def test_left_side_uses_symmetric_runtime_fields() -> None:
    observer, rec, pipe = ready_observer()
    rec.snap.generation = NS(side="1P", reset_epoch=1, action_revision=4)
    rec.handle = NS(instance_id=8)
    rec.active_handles = {"1P": rec.handle}
    identity = NS(side="1P", reset_epoch=1, action_revision=4)
    rec.active_candidates = {"1P": {"instance_id": 8, "status": "pending",
                                    "transaction": NS(state=NS(value="pending")),
                                    "candidate": NS(identity=identity, final_grid=grid())}}
    pipe._chain_start_next_1p = pipe._last_seen_next_1p = (5, 5)
    pipe._active_chain_1p = NS(chain_count=2)
    first = bound(side="1P", chain_event=pipe._active_chain_1p)
    assert observe(observer, rec, pipe, first) is None
    rec.snap.raw_score_evidence += (score(52, 200, 200, 0),)
    second = bound(frame=52, side="1P", chain_event=pipe._active_chain_1p)
    assert observe(observer, rec, pipe, second).instance_id == 8


@pytest.mark.parametrize("field", ["slide", "start", "current", "unknown"])
def test_next_progress_invalidates_episode_permanently(field: str) -> None:
    observer, rec, pipe = ready_observer()
    arg = bound()
    if field == "slide": arg.arguments["slide_motion"] = True
    if field == "start": pipe._chain_start_next_2p = (4, 5)
    if field == "current": pipe._last_seen_next_2p = (4, 5)
    if field == "unknown": pipe._last_seen_next_2p = None
    observe(observer, rec, pipe, arg)
    pipe._chain_start_next_2p = pipe._last_seen_next_2p = (5, 5)
    arg.arguments.update(next_pair=(5, 5), slide_motion=False)
    rec.snap.raw_score_evidence += (score(52, 200, 200, 0),)
    arg.arguments.update(frame_idx=52, time_sec=52 / 60)
    assert observe(observer, rec, pipe, arg) is None
    assert rec.rows[-1]["reason"] == "end_epoch_invalidated"


def test_nonmatching_grid_resets_only_consecutive_counter() -> None:
    observer, rec, pipe = ready_observer()
    assert observe(observer, rec, pipe, bound()) is None
    assert observe(observer, rec, pipe, bound(frame=51, board=Board(grid(1)))) is None
    rec.snap.raw_score_evidence += (score(52, 200, 200, 0), score(54, 200, 200, 0))
    assert observe(observer, rec, pipe, bound(frame=52)) is None
    assert observe(observer, rec, pipe, bound(frame=54)) is not None


def test_duplicate_frame_is_not_second_observation() -> None:
    observer, rec, pipe = ready_observer()
    assert observe(observer, rec, pipe, bound()) is None
    assert observe(observer, rec, pipe, bound()) is None


def test_skipped_frame_is_not_consecutive_observation() -> None:
    observer, rec, pipe = ready_observer()
    assert observe(observer, rec, pipe, bound()) is None
    rec.snap.raw_score_evidence += (score(54, 200, 200, 0), score(56, 200, 200, 0))
    assert observe(observer, rec, pipe, bound(frame=54)) is None
    assert observe(observer, rec, pipe, bound(frame=56)) is not None


def test_generation_mismatch_is_sticky_for_same_instance() -> None:
    observer, rec, pipe = ready_observer()
    current = rec.snap.generation
    rec._current_generation = lambda side: NS(side=side)
    assert observe(observer, rec, pipe, bound()) is None
    rec._current_generation = lambda side: current
    assert observe(observer, rec, pipe, bound()) is None
    assert rec.rows[-1]["reason"] == "end_epoch_invalidated"


def test_install_stashes_through_unwrapped_generic_instrument_and_restores() -> None:
    rec, calls = Recorder(), []
    module = ModuleType("tests._chain_end_fake_pipeline")
    def actual(*, current_next: Any, start_next: Any) -> bool:
        return current_next != start_next
    def decide(*args: Any, **kwargs: Any) -> bool:
        return actual(*args, **kwargs)
    module._is_game_event_chain_exit = decide
    class Pipeline(Pipe):
        def __init__(self) -> None:
            super().__init__()
            self._sm_2p = NS(context=NS(state=NS(value="chain")))
        def _step_side(self, side: str, frame_idx: int, time_sec: float,
                       is_active: bool, cnn_board: Any, chain_event: Any, *,
                       next_pair: Any = None, slide_motion: bool = False,
                       own_chain_active: bool = False) -> Any:
            calls.append((self._active_chain_2p, chain_event, own_chain_active))
            return "ok"
        def update(self, frame_idx: int, time_sec: float) -> Any:
            is_active, cnn_2p, chain_ev_2p, slide_2p = True, Board(grid()), self._active_chain_2p, False
            if module._is_game_event_chain_exit(
                    current_next=self._last_seen_next_2p,
                    start_next=self._chain_start_next_2p):
                self._stash_and_clear_active_chain("2P")
                chain_ev_2p = None
            return self._step_side("2P", frame_idx, time_sec, is_active, cnn_2p,
                                   chain_ev_2p, next_pair=self._last_seen_next_2p,
                                   slide_motion=slide_2p,
                                   own_chain_active=chain_ev_2p is not None)
    Pipeline.__module__ = module.__name__
    module.Pipeline = Pipeline
    sys.modules[module.__name__] = module
    original = Pipeline._step_side
    original_exit = module._is_game_event_chain_exit
    collector = NS(RecognitionPipeline=Pipeline)
    pipe = Pipeline()
    try:
        with contextlib.ExitStack() as stack:
            shadow.install(stack, collector, rec)
            rec.frame, rec.time_sec = 50, 50 / 60
            pipe.update(50, 50 / 60)
            rec.snap.raw_score_evidence += (score(52, 200, 200, 0),)
            rec.frame, rec.time_sec = 52, 52 / 60
            assert pipe.update(52, 52 / 60) == "ok"
            assert calls[-1] == (None, None, False)
            assert pipe.stash_calls == ["2P"]
            mutation = [row for row in rec.rows if row["kind"] == "boundary_repair_mutation"]
            assert len(mutation) == 1 and mutation[0]["stable_or_release_permission_issued"] is False
    finally:
        sys.modules.pop(module.__name__, None)
    assert Pipeline._step_side is original
    assert module._is_game_event_chain_exit is original_exit


def test_failed_stash_does_not_consume_one_shot() -> None:
    rec = Recorder()
    module = ModuleType("tests._chain_end_stash_failure")
    module._is_game_event_chain_exit = lambda **kwargs: False
    class Pipeline(Pipe):
        def _stash_and_clear_active_chain(self, side: str) -> None:
            raise RuntimeError("stash failed")
    Pipeline.__module__ = module.__name__
    sys.modules[module.__name__] = module
    collector, pipe = NS(RecognitionPipeline=Pipeline), Pipeline()
    try:
        with contextlib.ExitStack() as stack:
            observer = shadow.install(stack, collector, rec)
            evidence = shadow.EndEvidence(7, "2P", 1, 2, 100, 200, "f" * 64, 2)
            observer._ready["2P"] = evidence
            with pytest.raises(RuntimeError, match="stash failed"):
                pipe._stash_and_clear_active_chain("2P")
            assert ("2P", 7) not in observer._closed and "2P" not in observer._ready
    finally:
        sys.modules.pop(module.__name__, None)


def test_silent_stash_failure_does_not_consume_one_shot() -> None:
    rec = Recorder()
    module = ModuleType("tests._chain_end_silent_stash_failure")
    module._is_game_event_chain_exit = lambda **kwargs: False
    class Pipeline(Pipe):
        def _stash_and_clear_active_chain(self, side: str) -> None:
            return None
    Pipeline.__module__ = module.__name__
    sys.modules[module.__name__] = module
    try:
        with contextlib.ExitStack() as stack:
            observer = shadow.install(stack, NS(RecognitionPipeline=Pipeline), rec)
            evidence = shadow.EndEvidence(7, "2P", 1, 2, 100, 200, "f" * 64, 2)
            observer._ready["2P"] = evidence
            with pytest.raises(RuntimeError, match="退避が完了"):
                Pipeline()._stash_and_clear_active_chain("2P")
            assert ("2P", 7) not in observer._closed and "2P" not in observer._ready
    finally:
        sys.modules.pop(module.__name__, None)


def test_install_requires_existing_pending_ledger_runtime() -> None:
    with pytest.raises(RuntimeError, match="PendingCommit"):
        shadow.install(contextlib.ExitStack(), NS(RecognitionPipeline=object), NS())


def test_function_size_and_guard_paths_contract() -> None:
    import ast
    from pathlib import Path
    tree = ast.parse(Path(shadow.__file__).read_text(encoding="utf-8"))
    functions = [node for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert max(node.end_lineno - node.lineno + 1 for node in functions) <= 50
    assert shadow.GUARD_PATHS == ()
