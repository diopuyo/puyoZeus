"""連鎖完走の同時証拠だけで active-chain を退避する既定OFF shadow。

既存 prediction ledger / software generation / pending C6 transaction を読み、
実 ``_step_side`` 入力が origin final と一致した時だけ連鎖イベントを終了する。
これは state/STABLE、盤面公開、Counter、commit/release の許可ではない。
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import sys
from dataclasses import dataclass
from typing import Any


GUARD_PATHS: tuple[str, ...] = ()
SIDES = ("1P", "2P")
UNKNOWN_COLOR = 10
CONSECUTIVE_FULL_GRID_OBSERVATIONS = 2
OBSERVATION_FRAME_STRIDE = 2
VALID_PUYO_COLORS = frozenset(range(1, 6))
STICKY_REASONS = frozenset(("ledger_not_provisional", "software_generation_mismatch",
                            "pending_candidate_generation_mismatch"))


@dataclass(frozen=True)
class EndEvidence:
    """同じ実frameで成立した、終了入口専用の不変な根拠。"""

    instance_id: int
    side: str
    origin_prediction_revision: int
    formula_session_id: int
    anchor_score: int
    expected_score: int
    final_sha256: str
    consecutive_full_grid: int


class ChainEndEpochObserver:
    """既存台帳を変更せず、連鎖終了入口だけを一回限り判定する。"""

    def __init__(self, recorder: Any) -> None:
        self.rec = recorder
        self._matches: dict[tuple[str, int], tuple[Any, int, int]] = {}
        self._closed: set[tuple[str, int]] = set()
        self._invalidated: set[tuple[str, int]] = set()
        self._ready: dict[str, EndEvidence] = {}

    def before_exit(
        self, pipe: Any, side: Any, frame_idx: Any, time_sec: Any,
        is_active: Any, board: Any, chain_event: Any,
        current_next: Any, start_next: Any, slide_motion: Any, sm: Any,
    ) -> EndEvidence | None:
        """元game-event終了helperの実入力だけで終了入口の成立を返す。"""
        if side not in SIDES:
            return None
        handle = self.rec.active_handles.get(side)
        if not self._clock_matches(frame_idx, time_sec):
            return self._observe(side, "step_and_recorder_clock_mismatch", None)
        state = getattr(getattr(sm, "context", None), "state", None)
        if not bool(is_active) or _state_value(state) != "chain":
            return self._observe(side, "context_not_active_chain", None)
        if handle is None or chain_event is None:
            return self._observe(side, "inactive_or_clock_unknown", None)
        key = (side, int(handle.instance_id))
        snapshot = self.rec.ledger.snapshot(handle)
        suffix = "1p" if side == "1P" else "2p"
        if getattr(pipe, f"_active_chain_{suffix}", None) is None:
            return self._observe(side, "active_chain_missing", key)
        reason = self._identity_reason(side, key, snapshot)
        if reason is not None:
            if reason in STICKY_REASONS:
                self._invalidated.add(key)
            return self._observe(side, reason, key)
        next_reason = self._next_reason(side, current_next, start_next, slide_motion)
        if next_reason is not None:
            self._invalidated.add(key)
            return self._observe(side, next_reason, key)
        evidence, reason = self._evidence(snapshot, board, frame_idx, time_sec, key)
        if evidence is None:
            return self._observe(side, reason, key)
        self._ready[side] = evidence
        return evidence

    def mark_closed(self, side: str) -> EndEvidence | None:
        """stash成功後だけ同instanceの一回性を確定する。"""
        evidence = self._ready.pop(side, None)
        if evidence is None:
            return None
        self._closed.add((evidence.side, evidence.instance_id))
        return evidence

    def _clock_matches(self, frame_idx: Any, time_sec: Any) -> bool:
        """step、recorder、generation hookが同じ実update内にあることを要求する。"""
        if type(frame_idx) is not int or isinstance(time_sec, bool):
            return False
        if not isinstance(time_sec, (int, float)) or not self.rec._clock_active():
            return False
        return self.rec.frame == frame_idx and self.rec.time_sec == float(time_sec)

    def _identity_reason(self, side: str, key: tuple[str, int], snapshot: Any) -> str | None:
        """ledger世代・pending単一所有者・origin固定を検査する。"""
        if key in self._closed:
            return "already_closed_once"
        if key in self._invalidated:
            return "end_epoch_invalidated"
        if getattr(snapshot.status, "value", snapshot.status) != "provisional":
            return "ledger_not_provisional"
        if snapshot.generation != self.rec._current_generation(side):
            return "software_generation_mismatch"
        pending = self.rec.active_candidates.get(side)
        if pending is None or pending.get("instance_id") != key[1]:
            return "pending_owner_missing_or_mismatch"
        candidate = pending.get("candidate")
        identity = getattr(candidate, "identity", None)
        origin = _origin_prediction(snapshot)
        if (identity is None or identity.side != side
                or identity.reset_epoch != snapshot.generation.reset_epoch
                or identity.action_revision != snapshot.generation.action_revision):
            return "pending_candidate_generation_mismatch"
        if origin is None:
            return "origin_prediction_missing"
        if tuple(candidate.final_grid) != tuple(origin.final_grid):
            return "pending_candidate_not_origin_final"
        state = getattr(pending.get("transaction"), "state", None)
        if getattr(state, "value", state) != "pending" or pending.get("status") != "pending":
            return "pending_transaction_not_pending"
        return None

    def _next_reason(
        self, side: str, current: Any, start: Any, slide_motion: Any,
    ) -> str | None:
        """NEXT進行を終了許可にせず、旧episodeの永久失効にだけ使う。"""
        if bool(slide_motion):
            return "next_slide_progressed"
        if any(not _valid_next(value) for value in (start, current)):
            return "next_evidence_unknown"
        if tuple(start) != tuple(current):
            return "next_value_progressed"
        return None

    def _evidence(
        self, snapshot: Any, board: Any, frame_idx: Any, time_sec: Any,
        key: tuple[str, int],
    ) -> tuple[EndEvidence | None, str]:
        """origin/formula/score/finalの同時成立と2回の同盤面を検査する。"""
        grid = _board_grid(board)
        if grid is None or any(UNKNOWN_COLOR in row for row in grid):
            self._matches.pop(key, None)
            return None, "current_full_grid_unknown"
        if origin := _origin_prediction(snapshot):
            if grid != origin.final_grid:
                self._matches.pop(key, None)
                return None, "current_grid_not_origin_final"
        else:
            self._matches.pop(key, None)
            return None, "origin_or_formula_binding_missing"
        formula, reason = _formula_contract(snapshot, origin)
        if formula is None:
            self._matches.pop(key, None)
            return None, reason
        anchor, reason = _score_contract(snapshot, formula, origin, frame_idx, time_sec)
        if anchor is None:
            self._matches.pop(key, None)
            return None, reason
        count = self._advance_match(key, grid, frame_idx)
        if count < CONSECUTIVE_FULL_GRID_OBSERVATIONS:
            return None, "origin_final_needs_second_observation"
        return EndEvidence(key[1], key[0], origin.prediction_revision, formula.session_id,
                           anchor, anchor + origin.calculated_total_score,
                           origin.final_sha256, count), "ready"

    def _advance_match(self, key: tuple[str, int], grid: Any, frame_idx: Any) -> int:
        """重複frameを票にせず、連続した実updateだけを数える。"""
        previous = self._matches.get(key)
        adjacent = previous is not None and frame_idx == previous[2] + OBSERVATION_FRAME_STRIDE
        count = previous[1] + 1 if adjacent and previous[0] == grid else 1
        self._matches[key] = (grid, count, int(frame_idx))
        return count

    def _observe(self, side: str, reason: str, key: Any) -> None:
        """不成立も許可へ読み替えず実時計の診断行へ残す。"""
        if reason != "origin_final_needs_second_observation":
            if key is None:
                for known in [value for value in self._matches if value[0] == side]:
                    self._matches.pop(known, None)
            else:
                self._matches.pop(key, None)
        self.rec.emit({"kind": "boundary_repair_observation", "repair": "chain_end",
                       "side": side, "instance_id": None if key is None else key[1],
                       "reason": reason, "state_or_release_modified": False,
                       "commit_permission_issued": False})
        return None


def _origin_prediction(snapshot: Any) -> Any | None:
    """後から最大値を選ばず、固定済みorigin revisionだけを返す。"""
    revision = snapshot.origin_prediction_revision
    if type(revision) is not int:
        return None
    values = [value for value in snapshot.predictions
              if value.prediction_revision == revision and value.is_origin_reference]
    if len(values) != 1:
        return None
    origin = values[0]
    scope = getattr(origin.scope, "value", origin.scope)
    if origin.episode_revision != 1 or scope != "total_from_instance_origin":
        return None
    return origin


def _formula_contract(snapshot: Any, origin: Any) -> tuple[Any | None, str]:
    """最初にbindしたsessionがorigin全段・総scoreへ到達したことを検査する。"""
    binding = snapshot.formula_session_binding
    if origin is None or binding is None:
        return None, "origin_or_formula_binding_missing"
    if binding.source != "formula_reset/update":
        return None, "formula_binding_source_untrusted"
    initial = [row for row in snapshot.formula_evidence
               if row.session_id == binding.session_id and row.step_index == 0
               and row.ledger_sequence == binding.ledger_sequence]
    if len(initial) != 1:
        return None, "initial_formula_session_reset_missing"
    rows = [row for row in snapshot.formula_evidence
            if row.session_id == binding.session_id and row.step_index > 0]
    first_by_step: dict[int, Any] = {}
    for row in rows:
        prior = first_by_step.get(row.step_index)
        if prior is not None and (prior.total_power, prior.step_product) != (
                row.total_power, row.step_product):
            return None, "formula_step_revision_conflict"
        first_by_step.setdefault(row.step_index, row)
    expected = list(range(1, origin.chain_count + 1))
    if sorted(first_by_step) != expected:
        return None, "formula_steps_incomplete"
    ordered = [first_by_step[index] for index in expected]
    if any(row.step_product is None for row in ordered):
        return None, "formula_step_product_unknown"
    if any(row.source != "formula_observation" for row in ordered):
        return None, "formula_step_source_untrusted"
    if ordered[-1].total_power != origin.calculated_total_score:
        return None, "formula_total_not_origin_total"
    if any(row.total_power != sum(item.step_product for item in ordered[:index])
           for index, row in enumerate(ordered, start=1)):
        return None, "formula_prefix_not_causal"
    return binding, "ready"


def _score_contract(
    snapshot: Any, binding: Any, origin: Any, frame_idx: Any, time_sec: Any,
) -> tuple[int | None, str]:
    """初段後最初の正増分をanchorへ固定し、現frameの総score一致を要求する。"""
    steps = [row for row in snapshot.formula_evidence
             if row.session_id == binding.session_id and row.step_index == 1]
    if not steps:
        return None, "first_formula_step_missing"
    positives = [row for row in snapshot.raw_score_evidence
                 if row.is_valid and row.delta is not None and row.delta > 0
                 and steps[0].observed_at.frame_idx <= row.observed_at.frame_idx <= frame_idx]
    by_step: dict[int, Any] = {}
    for row in snapshot.formula_evidence:
        if row.session_id == binding.session_id and row.step_index > 0:
            by_step.setdefault(row.step_index, row)
    formulas = [by_step[index] for index in sorted(by_step)]
    if [row.delta for row in positives] != [row.step_product for row in formulas]:
        return None, "score_deltas_not_formula_products"
    first = positives[0]
    if first.source != "ScoreDelta/_apply_read":
        return None, "score_anchor_source_untrusted"
    if first.prev_score is None or first.cur_score != first.prev_score + first.delta:
        return None, "score_anchor_inconsistent"
    current = [row for row in snapshot.raw_score_evidence
               if row.observed_at.frame_idx == frame_idx
               and row.observed_at.time_sec == float(time_sec) and row.is_valid]
    target = first.prev_score + origin.calculated_total_score
    if (len(current) != 1 or current[0].source != "ScoreDelta/_apply_read"
            or current[0].raw_value != target or current[0].cur_score != target):
        return None, "current_raw_score_not_origin_total"
    return first.prev_score, "ready"


def _valid_next(value: Any) -> bool:
    """NEXTは実色1..5の厳密2要素だけを現在証拠に使う。"""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return False
    return all(type(color) is int and color in VALID_PUYO_COLORS for color in value)


def _state_value(value: Any) -> str:
    """実state enumを推測せず比較用の小文字へ揃える。"""
    return str(getattr(value, "value", value)).lower()


def _board_grid(board: Any) -> tuple[tuple[int, ...], ...] | None:
    """実Boardをimmutable gridへ変換し、異常形は不成立にする。"""
    if board is None:
        return None
    try:
        grid = tuple(tuple(int(cell) for cell in row)
                     for row in board.copy().to_dict()["grid"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if len(grid) != 13 or any(len(row) != 6 for row in grid):
        return None
    return grid


def install(stack: contextlib.ExitStack, collector: Any, rec: Any) -> ChainEndEpochObserver:
    """pending shadow済み実runtimeへ終了入口を一軸だけ一時接続する。"""
    required = ("ledger", "generation_recorder", "active_handles", "active_candidates",
                "_clock_active", "_current_generation", "emit")
    if any(not hasattr(rec, name) for name in required) or rec.ledger is None:
        raise RuntimeError("PredictionLedger/PendingCommit recorderが先に必要です")
    cls = collector.RecognitionPipeline
    module = sys.modules.get(cls.__module__)
    if module is None or not callable(getattr(module, "_is_game_event_chain_exit", None)):
        raise RuntimeError("実pipelineのgame-event終了helperが必要です")
    original_exit, original_stash = module._is_game_event_chain_exit, cls._stash_and_clear_active_chain
    observer = ChainEndEpochObserver(rec)

    @functools.wraps(original_exit)
    def exit_signal(*args: Any, **kwargs: Any) -> bool:
        if bool(original_exit(*args, **kwargs)):
            return True
        frame = sys._getframe(1)
        inputs = _exit_inputs(args, kwargs)
        runtime = None if inputs is None else _runtime_call(frame, *inputs)
        if runtime is None:
            rec.emit({"kind": "boundary_repair_observation", "repair": "chain_end",
                      "side": None, "reason": "runtime_side_or_input_ambiguous",
                      "commit_permission_issued": False})
            return False
        return observer.before_exit(**runtime) is not None

    @functools.wraps(original_stash)
    def stash(pipe: Any, side: str) -> Any:
        active = getattr(pipe, f"_active_chain_{'1p' if side == '1P' else '2p'}", None)
        try:
            result = original_stash(pipe, side)
        except Exception:
            observer._ready.pop(side, None)
            raise
        suffix = "1p" if side == "1P" else "2p"
        if active is not None and getattr(pipe, f"_active_chain_{suffix}", None) is not None:
            observer._ready.pop(side, None)
            raise RuntimeError("active chainの退避が完了しませんでした")
        evidence = observer.mark_closed(side)
        if evidence is not None:
            rec.emit({"kind": "boundary_repair_mutation", "repair": "chain_end",
                      "side": side, "before": {"active": _active_value(active)},
                      "after": {"active": None}, "evidence": evidence.__dict__,
                      "stable_or_release_permission_issued": False})
        return result

    _patch(stack, module, "_is_game_event_chain_exit", exit_signal)
    _patch(stack, cls, "_stash_and_clear_active_chain", stash)
    return observer


def _exit_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any] | None:
    """既存の汎用計装wrapper越しでも実helperのkeyword入力だけを読む。"""
    if args or set(kwargs) != {"current_next", "start_next"}:
        return None
    return kwargs["current_next"], kwargs["start_next"]


def _runtime_call(frame: Any, current_next: Any, start_next: Any) -> dict[str, Any] | None:
    """元update callerのlive局所値をside推測なしで一組へ束縛する。"""
    pipe = frame.f_locals.get("self")
    if pipe is None:
        return None
    matches = []
    for side, suffix in (("1P", "1p"), ("2P", "2p")):
        if (current_next is getattr(pipe, f"_last_seen_next_{suffix}", None)
                and start_next is getattr(pipe, f"_chain_start_next_{suffix}", None)):
            matches.append((side, suffix))
    if len(matches) != 1:
        return None
    side, suffix = matches[0]
    return {"pipe": pipe, "side": side, "frame_idx": frame.f_locals.get("frame_idx"),
            "time_sec": frame.f_locals.get("time_sec"),
            "is_active": frame.f_locals.get("is_active"),
            "board": frame.f_locals.get(f"cnn_{suffix}"),
            "chain_event": frame.f_locals.get(f"chain_ev_{suffix}"),
            "current_next": current_next, "start_next": start_next,
            "slide_motion": frame.f_locals.get(f"slide_{suffix}"),
            "sm": getattr(pipe, f"_sm_{suffix}", None)}


def _patch(stack: contextlib.ExitStack, owner: Any, name: str, value: Any) -> None:
    """例外終了でもdescriptorを厳密に復元する。"""
    original = inspect.getattr_static(owner, name)
    setattr(owner, name, value)
    stack.callback(setattr, owner, name, original)


def _active_value(event: Any) -> dict[str, Any] | None:
    """owner addressを保存せず、終了直前eventの監査値だけを返す。"""
    if event is None:
        return None
    return {name: getattr(event, name, None) for name in (
        "chain_count", "total_score", "mechanism", "trigger_sec", "end_sec")}


__all__ = ["ChainEndEpochObserver", "EndEvidence", "GUARD_PATHS", "install"]
