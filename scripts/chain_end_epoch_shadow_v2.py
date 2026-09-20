"""NEXTの一時揺れと真の連鎖退出を分離するchain-end既定OFF shadow。"""

from __future__ import annotations

import contextlib
import functools
import inspect
import sys
from dataclasses import dataclass
from typing import Any

from scripts import chain_end_epoch_shadow_v1 as base


GUARD_PATHS: tuple[str, ...] = ("scripts/chain_end_epoch_shadow_v1.py",)
SIDES = base.SIDES
EndEvidence = base.EndEvidence


@dataclass(frozen=True)
class NextDisruption:
    """NEXT異常を観測した時点のformula進捗を固定する。"""

    session_id: int
    max_step_index: int
    max_ledger_sequence: int
    frame_idx: int
    time_sec: float
    reason: str


class ChainEndEpochObserver(base.ChainEndEpochObserver):
    """NEXT復帰だけでは再取得せず、同一sessionの実step前進を要求する。"""

    def __init__(self, recorder: Any) -> None:
        super().__init__(recorder)
        self._disruptions: dict[tuple[str, int], NextDisruption] = {}

    def before_exit(
        self, pipe: Any, side: Any, frame_idx: Any, time_sec: Any,
        is_active: Any, board: Any, chain_event: Any,
        current_next: Any, start_next: Any, slide_motion: Any, sm: Any,
    ) -> EndEvidence | None:
        """一時中断を保存し、formula実前進後だけ新しい2票を開始する。"""
        initial = self._initial_context(pipe, side, frame_idx, time_sec, is_active,
                                        chain_event, sm)
        if isinstance(initial, str):
            return self._observe(side if side in SIDES else str(side), initial, None)
        key, snapshot = initial
        reason = self._identity_reason(side, key, snapshot)
        if reason is not None:
            if reason in base.STICKY_REASONS:
                self._invalidated.add(key)
            return self._observe(side, reason, key)
        next_reason = self._next_reason(side, current_next, start_next, slide_motion)
        if next_reason is not None:
            self._remember_disruption(key, snapshot, frame_idx, time_sec, next_reason)
            return self._observe(side, next_reason, key)
        disruption = self._disruptions.get(key)
        if disruption is not None:
            if not self._formula_advanced(snapshot, disruption, frame_idx, time_sec):
                return self._observe(side, "next_disruption_pending", key)
            self._disruptions.pop(key, None)
            self._matches.pop(key, None)
            self._emit_recovery(side, key, disruption)
        evidence, reason = self._evidence(snapshot, board, frame_idx, time_sec, key)
        if evidence is None:
            return self._observe(side, reason, key)
        self._ready[side] = evidence
        return evidence

    def note_active_stash(self, side: str) -> None:
        """helper外のactive消失も一時中断として保存する。"""
        handle = self.rec.active_handles.get(side)
        if handle is None or not self.rec._clock_active():
            return
        key = (side, int(handle.instance_id))
        try:
            snapshot = self.rec.ledger.snapshot(handle)
        except Exception:
            self._invalidated.add(key)
            return
        before = self._disruptions.get(key)
        self._remember_disruption(key, snapshot, self.rec.frame, self.rec.time_sec,
                                  "active_chain_stashed")
        after = self._disruptions.get(key)
        self.rec.emit({"kind": "boundary_repair_observation", "repair": "chain_end",
                       "side": side, "instance_id": key[1],
                       "reason": "active_chain_stashed",
                       "barrier_before": None if before is None else before.__dict__,
                       "barrier_after": None if after is None else after.__dict__,
                       "commit_permission_issued": False})

    def _initial_context(
        self, pipe: Any, side: Any, frame_idx: Any, time_sec: Any,
        is_active: Any, chain_event: Any, sm: Any,
    ) -> tuple[tuple[str, int], Any] | str:
        """v1と同じ実clock・CHAIN・owner入口を検査する。"""
        if side not in SIDES:
            return "invalid_side"
        handle = self.rec.active_handles.get(side)
        if not self._clock_matches(frame_idx, time_sec):
            return "step_and_recorder_clock_mismatch"
        state = getattr(getattr(sm, "context", None), "state", None)
        if not bool(is_active) or base._state_value(state) != "chain":
            return "context_not_active_chain"
        if handle is None or chain_event is None:
            return "inactive_or_clock_unknown"
        suffix = "1p" if side == "1P" else "2p"
        if getattr(pipe, f"_active_chain_{suffix}", None) is None:
            return "active_chain_missing"
        key = (side, int(handle.instance_id))
        return key, self.rec.ledger.snapshot(handle)

    def _remember_disruption(
        self, key: tuple[str, int], snapshot: Any, frame_idx: Any,
        time_sec: Any, reason: str,
    ) -> None:
        """異常が続く間の最新formula進捗を回復barrierへ固定する。"""
        self._matches.pop(key, None)
        progress = _formula_progress(snapshot, frame_idx, time_sec)
        if progress is None:
            self._invalidated.add(key)
            return
        session_id, max_step, max_sequence = progress
        previous = self._disruptions.get(key)
        if previous is not None and previous.session_id != session_id:
            self._invalidated.add(key)
            return
        self._disruptions[key] = NextDisruption(
            session_id, max_step, max_sequence, int(frame_idx), float(time_sec), reason)

    def _formula_advanced(
        self, snapshot: Any, disruption: NextDisruption,
        frame_idx: Any, time_sec: Any,
    ) -> bool:
        """同一sessionの事後formula step厳密増加だけを受理する。"""
        if type(frame_idx) is not int or not isinstance(time_sec, (int, float)):
            return False
        binding = getattr(snapshot, "formula_session_binding", None)
        if getattr(binding, "session_id", None) != disruption.session_id:
            return False
        for row in getattr(snapshot, "formula_evidence", ()):
            point = getattr(row, "observed_at", None)
            if (_is_formula_step(row, disruption.session_id)
                    and _valid_point(point)
                    and row.step_index > disruption.max_step_index
                    and row.ledger_sequence > disruption.max_ledger_sequence
                    and point.frame_idx > disruption.frame_idx
                    and point.frame_idx <= frame_idx
                    and float(point.time_sec) > disruption.time_sec
                    and float(point.time_sec) <= float(time_sec)):
                return True
        return False

    def _emit_recovery(
        self, side: str, key: tuple[str, int], disruption: NextDisruption,
    ) -> None:
        """回復は終了許可でないことを診断行へ固定する。"""
        self.rec.emit({"kind": "boundary_repair_observation", "repair": "chain_end",
                       "side": side, "instance_id": key[1],
                       "reason": "next_disruption_recovered_by_formula_step",
                       "disruption": disruption.__dict__,
                       "commit_permission_issued": False})


def _formula_progress(
    snapshot: Any, frame_idx: Any, time_sec: Any,
) -> tuple[int, int, int] | None:
    """中断時までに実観測済みの同一session最大step/sequenceを返す。"""
    if type(frame_idx) is not int or isinstance(time_sec, bool):
        return None
    if not isinstance(time_sec, (int, float)):
        return None
    binding = getattr(snapshot, "formula_session_binding", None)
    session_id = getattr(binding, "session_id", None)
    if type(session_id) is not int:
        return None
    rows = []
    for row in getattr(snapshot, "formula_evidence", ()):
        point = getattr(row, "observed_at", None)
        if (_is_formula_step(row, session_id) and _valid_point(point)
                and point.frame_idx <= frame_idx and float(point.time_sec) <= float(time_sec)):
            rows.append(row)
    if not rows:
        return None
    return session_id, max(row.step_index for row in rows), max(row.ledger_sequence for row in rows)


def _is_formula_step(row: Any, session_id: int) -> bool:
    """再捕捉episodeでなく出所付きformula stepだけを数える。"""
    return (getattr(row, "source", None) == "formula_observation"
            and getattr(row, "session_id", None) == session_id
            and type(getattr(row, "step_index", None)) is int
            and type(getattr(row, "ledger_sequence", None)) is int
            and getattr(row, "observed_at", None) is not None)


def _valid_point(point: Any) -> bool:
    """改竄clockを比較例外や暗黙整数へ通さない。"""
    frame = getattr(point, "frame_idx", None)
    value = getattr(point, "time_sec", None)
    return (type(frame) is int and not isinstance(value, bool)
            and isinstance(value, (int, float)))


def install(
    stack: contextlib.ExitStack, collector: Any, rec: Any,
) -> ChainEndEpochObserver:
    """pending runtimeへv2 observerを一時接続する。"""
    required = ("ledger", "generation_recorder", "active_handles", "active_candidates",
                "_clock_active", "_current_generation", "emit")
    if any(not hasattr(rec, name) for name in required) or rec.ledger is None:
        raise RuntimeError("PredictionLedger/PendingCommit recorderが先に必要です")
    cls = collector.RecognitionPipeline
    module = sys.modules.get(cls.__module__)
    if module is None or not callable(getattr(module, "_is_game_event_chain_exit", None)):
        raise RuntimeError("実pipelineのgame-event終了helperが必要です")
    original_exit = module._is_game_event_chain_exit
    original_stash = cls._stash_and_clear_active_chain
    observer = ChainEndEpochObserver(rec)
    _install_exit(stack, module, observer, original_exit)
    _install_stash(stack, cls, observer, original_stash, rec)
    return observer


def _install_exit(
    stack: contextlib.ExitStack, module: Any, observer: ChainEndEpochObserver,
    original_exit: Any,
) -> None:
    """既存helperを一回だけ呼び、false時だけshadow証拠を検査する。"""
    @functools.wraps(original_exit)
    def exit_signal(*args: Any, **kwargs: Any) -> bool:
        if bool(original_exit(*args, **kwargs)):
            return True
        frame = sys._getframe(1)
        inputs = base._exit_inputs(args, kwargs)
        runtime = None if inputs is None else base._runtime_call(frame, *inputs)
        if runtime is None:
            observer.rec.emit({"kind": "boundary_repair_observation",
                               "repair": "chain_end", "side": None,
                               "reason": "runtime_side_or_input_ambiguous",
                               "commit_permission_issued": False})
            return False
        return observer.before_exit(**runtime) is not None
    base._patch(stack, module, "_is_game_event_chain_exit", exit_signal)


def _install_stash(
    stack: contextlib.ExitStack, cls: Any, observer: ChainEndEpochObserver,
    original_stash: Any, rec: Any,
) -> None:
    """stash成功時だけcloseまたは一時中断を記録する。"""
    @functools.wraps(original_stash)
    def stash(pipe: Any, side: str) -> Any:
        suffix = "1p" if side == "1P" else "2p"
        active = getattr(pipe, f"_active_chain_{suffix}", None)
        try:
            result = original_stash(pipe, side)
        except Exception:
            observer._ready.pop(side, None)
            raise
        if active is not None and getattr(pipe, f"_active_chain_{suffix}", None) is not None:
            observer._ready.pop(side, None)
            raise RuntimeError("active chainの退避が完了しませんでした")
        evidence = observer.mark_closed(side)
        if evidence is None and active is not None:
            observer.note_active_stash(side)
        elif evidence is not None:
            rec.emit({"kind": "boundary_repair_mutation", "repair": "chain_end",
                      "side": side, "before": {"active": base._active_value(active)},
                      "after": {"active": None}, "evidence": evidence.__dict__,
                      "stable_or_release_permission_issued": False})
        return result
    base._patch(stack, cls, "_stash_and_clear_active_chain", stash)


__all__ = ["ChainEndEpochObserver", "EndEvidence", "GUARD_PATHS", "NextDisruption", "install"]
