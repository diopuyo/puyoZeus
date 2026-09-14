"""旧確定公開を保留しつつ、同呼出の別current候補streamへ接続する。"""
from __future__ import annotations

from dataclasses import asdict
import functools
import json
from typing import Any

import adapter as A
C: Any = None
POLICY_FILES = (
    A.ROOT.parent / "g2_hidden_probability_capture_2026-09-08_v1/replay.py",
    A.ROOT.parent / "g2_hidden_probability_basis_2026-09-08_v1/support_policy.py",
)

SIDECAR = "provisional_current.jsonl"


def policy() -> Any:
    """実frozen collectorが確立する前にはsrcをimportしない。"""
    global C
    if C is None:
        import current_policy
        C = current_policy
    return C


class Connection:
    """私有の診断出力だけを持つ。collectorの確定盤面/会計は変更しない。"""
    def __init__(self, history: Any, state: dict[str, Any]) -> None:
        self.history, self.state = history, state
        self.next_rows: dict[tuple[int, str], dict[str, Any]] = {}
        self.rows: list[dict[str, Any]] = []
        self.failures: list[str] = []
        self.closed = False
        self.stream = (state["output"] / SIDECAR).open("x", encoding="utf-8")

    def next_row(self, value: dict[str, Any]) -> None:
        if value.get("kind") != "next_enqueue_live_decision":
            return
        row = {"frame_idx": self.history.frame, "time_sec": self.history.time_sec, **value}
        key = row["frame_idx"], row["side"]
        if key in self.next_rows:
            raise RuntimeError("duplicate_current_next_observation")
        self.next_rows[key] = json.loads(json.dumps(row, allow_nan=False))

    def current(self, side: str, before: Any, after: Any) -> None:
        if self.failures:
            raise RuntimeError("provisional_connection_failed")
        frame = self.history.frame
        pb = [r for r in self.state["hidden_probability_observer"].rows
              if r["frame_idx"] == frame and r["side"] == side]
        if not pb:
            return
        sm = [r for r in self.state["sink"].rows if r["frame_idx"] == frame
              and r["side"] == side and r["kind"] == "current_entry_sm_return"]
        row = {"frame": frame, "side": side, "current_stage": "before_confirmed_publication_hold",
               "confirmed_output_held": after.confirmed_board is None,
               "current_candidate": None, "accounting_permission": False, "quality_gate_clear": False}
        try:
            C.require(len(pb) == len(sm) == 1, "current_capture_count")
            final = self.final_row(side, before)
            next_row = self.next_rows.get((frame, side))
            C.require(next_row is not None, "current_next_observation_missing")
            row["current_candidate"] = asdict(C.current_candidate(pb[0], sm[0], final, next_row))
        except ValueError as error:
            row["hold_reason"] = str(error)
        self.rows.append(row)
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()

    def final_row(self, side: str, result: Any) -> dict[str, Any]:
        """実updateの保留前SideResultを取得。最終確定出力の成功とは呼ばない。"""
        board_value = self.state["hidden_probability_observer"].board_value
        return {"kind": "frame_side", "side": side, "frame_idx": self.history.frame,
            "time_sec": self.history.time_sec, "state": getattr(result.state, "value", result.state),
            "cnn": board_value(result.cnn_board), "confirmed": board_value(result.confirmed_board),
            "board_provenance": result.board_provenance, "board_none_reason": result.board_none_reason}

    def close(self) -> None:
        self.stream.close()
        self.closed = True


def install(stack: Any, collector: Any, history: Any, state: dict[str, Any]) -> Connection:
    """既存計装完了後に呼ぶ。元update/stepを再compileせず、旧確定はholdする。"""
    policy()
    rec = state["pending_rec"]
    hold = stack.enter_context(A.installed(type(rec)))
    connection = Connection(history, state)
    stack.callback(connection.close)
    original_emit, original_side = history.emit, rec.isolate_side_result
    own_side = vars(rec).get("isolate_side_result")
    had_own_side = "isolate_side_result" in vars(rec)
    @functools.wraps(original_emit)
    def emit(value: dict[str, Any]) -> Any:
        result = original_emit(value)
        connection.next_row(value)
        return result
    @functools.wraps(original_side)
    def isolate(side: str, value: Any) -> Any:
        result = original_side(side, value)
        try:
            connection.current(side, value, result)
        except BaseException as error:
            connection.failures.append(repr(error))
            raise
        return result
    history.emit, rec.isolate_side_result = emit, isolate
    stack.callback(setattr, history, "emit", original_emit)
    if had_own_side:
        stack.callback(setattr, rec, "isolate_side_result", own_side)
    else:
        stack.callback(delattr, rec, "isolate_side_result")
    state.update(generation_publication_hold=hold, provisional_current_connection=connection)
    return connection
