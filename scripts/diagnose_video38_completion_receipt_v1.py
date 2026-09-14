"""atomic landing対照を変えず、連鎖完了証拠の時系列receiptだけを採取する。"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any

from scripts import diagnose_video38_atomic_landing_shadow_v1 as atomic


prechain, details, base = atomic.prechain, atomic.prechain.details, atomic.base
FORMAT = "video38-completion-receipt/v1"
START_SEC, END_SEC = 560.0, 605.0
PREFIX_END_SEC, SOURCE_FPS, FRAME_STRIDE = 583.0, 60, 2
START_FRAME, PREFIX_END_FRAME, END_FRAME = 33600, 34980, 36300
LAUNCHER = base.ROOT / "scripts/launch_video38_completion_receipt_v1.sh"
TEST = base.ROOT / "tests/test_diagnose_video38_completion_receipt_v1.py"
DOC = base.ROOT / "docs/agent_coordination/C6_COMMIT_VERIFICATION_WIRING_2026-09-07.md"
ATOMIC_REFERENCE = base.VERIFY / "video38_atomic_landing_shadow_2026-09-07_v1"
ORIGINAL_ATOMIC_INSTRUMENT = atomic.atomic_instrument
ORIGINAL_SHADOW_FINISH = prechain.shadow_finish
ORIGINAL_COMPLETION_PREPARE = atomic.ORIGINAL_PREPARE


def _prediction(result: Any) -> dict[str, Any] | None:
    """既存simulate戻り値だけから物理予測と公式素点を複製する。"""
    if result is None:
        return None
    from src.scoring import calculate_chain_score
    value = details.result_value(result)
    value["calculated_total_score"] = calculate_chain_score(result).total_score
    return value


def _caller_side(frame: FrameType, owner: Any | None = None) -> str | None:
    """呼出stackのside、またはpipelineが所有するaccumulatorのsideを得る。"""
    current: FrameType | None = frame
    for _ in range(8):
        if current is None:
            break
        side = current.f_locals.get("side")
        if side in ("1P", "2P"):
            return side
        pipe = current.f_locals.get("self")
        for side, suffix in (("1P", "1p"), ("2P", "2p")):
            if (owner is not None and pipe is not None
                    and getattr(pipe, f"_formula_accum_{suffix}", None) is owner):
                return side
        current = current.f_back
    return None


class CompletionRecorder(atomic.AtomicRecorder):
    """raw score・formula revision・物理予測を同じframe時計で保持する。"""

    def __init__(self, stream: Any, target: dict[str, Any]) -> None:
        super().__init__(stream, target)
        self.score_rows: list[dict[str, Any]] = []
        self.formula_rows: list[dict[str, Any]] = []
        self.prediction_rows: list[dict[str, Any]] = []
        self.frame_rows: list[dict[str, Any]] = []
        self.lineages: list[dict[str, Any]] = []
        self.current_lineage: dict[str, int] = {}
        self.formula_sessions: dict[int, int] = {}
        self.last_raw_score: dict[str, dict[str, Any]] = {}
        self.latest_positive_delta: dict[str, dict[str, Any]] = {}
        self.verify_frames = 5

    def _add(self, target: list[dict[str, Any]], row: dict[str, Any]) -> None:
        value = base.json_value({"frame_idx": self.frame, "time_sec": self.time_sec, **row})
        target.append(value)
        self.emit(value)

    def record_start(self, side: str, event: Any, result: Any) -> None:
        lineage_id = len(self.lineages) + 1
        self.current_lineage[side] = lineage_id
        row = {"lineage_id": lineage_id, "side": side, "start_frame": self.frame,
               "start_time_sec": self.time_sec, "event": details.event_value(event),
               "entry_score_anchor": None, "anchor_status": "unknown_until_raw_score_jump",
               "raw_score_before_start": self.last_raw_score.get(side),
               "generation_binding": "unverified_no_reset_action_receipt"}
        self.lineages.append(base.json_value(row))
        self.emit({"kind": "completion_lineage_start", **row})
        jump = self.latest_positive_delta.get(side)
        if jump is not None and jump["frame_idx"] == self.frame:
            self._freeze_anchor(lineage_id, jump)
        if result is not None:
            self.record_prediction(side, result, "_start_chain_estimate/result")

    def record_score(self, delta: Any, raw_value: int | None) -> None:
        side = str(delta.side)
        lineage_id = self.current_lineage.get(side)
        row = {"kind": "raw_score_ocr", "side": side, "lineage_id": lineage_id,
               "raw_value": raw_value, "prev_score": delta.prev_score,
               "cur_score": delta.cur_score, "delta": delta.delta,
               "is_valid": bool(delta.is_valid), "source": "ScoreDelta/_apply_read"}
        prior = self.last_raw_score.get(side)
        self._add(self.score_rows, row)
        if raw_value is not None:
            self.last_raw_score[side] = {"value": int(raw_value), "frame_idx": self.frame,
                                         "time_sec": self.time_sec, "source": "raw_score_ocr"}
        if delta.is_valid and delta.delta > 0:
            jump = {"prev_score": int(delta.prev_score), "frame_idx": self.frame,
                    "time_sec": self.time_sec, "previous_raw_observation": prior}
            self.latest_positive_delta[side] = jump
            if lineage_id is not None:
                self._freeze_anchor(lineage_id, jump)

    def _freeze_anchor(self, lineage_id: int, jump: dict[str, Any]) -> None:
        lineage = self.lineages[lineage_id - 1]
        if lineage["entry_score_anchor"] is not None:
            return
        previous = jump.get("previous_raw_observation")
        consistent = previous is not None and previous.get("value") == jump["prev_score"]
        lineage.update(entry_score_anchor=jump["prev_score"],
                       anchor_status="candidate_unverified_generation",
                       anchor_source_frame=jump["frame_idx"],
                       anchor_source_time_sec=jump["time_sec"],
                       anchor_previous_raw_observation=previous,
                       anchor_prev_matches_saved_raw=consistent,
                       anchor_source="ScoreDelta.prev_score; first positive jump")
        self.emit({"kind": "entry_score_anchor_candidate", "lineage_id": lineage_id,
                   "side": lineage["side"], "anchor": jump["prev_score"],
                   "source_frame": jump["frame_idx"], "source_time_sec": jump["time_sec"],
                   "previous_raw_observation": previous,
                   "generation_binding": "unverified_no_reset_action_receipt"})

    def record_formula(self, side: str | None, owner: Any, step: Any,
                       before_count: int) -> None:
        key = id(owner)
        session = self.formula_sessions.setdefault(key, 0)
        row = {"kind": "formula_observation", "side": side,
               "lineage_id": self.current_lineage.get(side or ""), "session": session,
               "step_count": owner.step_count, "total_power": owner.total_power,
               "last_valid_t": owner.last_valid_t,
               "new_step": None if step is None else {
                   "t_sec": step.t_sec, "left": step.left, "right": step.right,
                   "product": step.product}}
        if step is not None or owner.step_count != before_count:
            self._add(self.formula_rows, row)

    def record_formula_reset(self, side: str | None, owner: Any,
                             caller: str) -> None:
        key = id(owner)
        session = self.formula_sessions.get(key, 0) + 1
        self.formula_sessions[key] = session
        self._add(self.formula_rows, {"kind": "formula_session_reset", "side": side,
            "lineage_id": self.current_lineage.get(side or ""), "session": session,
            "caller": caller, "step_count": owner.step_count,
            "total_power": owner.total_power, "last_valid_t": owner.last_valid_t})

    def record_prediction(self, side: str | None, result: Any, caller: str) -> None:
        row = {"kind": "physical_prediction", "prediction_id": len(self.prediction_rows) + 1,
               "side": side,
               "lineage_id": self.current_lineage.get(side or ""), "caller": caller,
               "prediction": _prediction(result)}
        self._add(self.prediction_rows, row)

    def record_side(self, pipeline: Any, side: str, result: Any) -> None:
        super().record_side(pipeline, side, result)
        self.verify_frames = int(getattr(pipeline, "CHAIN_VERIFY_FRAMES", 5))
        suffix = "1p" if side == "1P" else "2p"
        memory = getattr(pipeline, f"_stable_color_memory_{suffix}")
        captured = self.capture.get(id(memory))
        row = {"kind": "completion_frame_observation", "side": side,
            "lineage_id": self.current_lineage.get(side),
            "state": getattr(result.state, "value", str(result.state)),
            "raw_captured_this_frame": captured is not None,
            "raw_before_accounting": None if captured is None else captured["raw"],
            "published_cnn": base.board_value(result.cnn_board)}
        self._add(self.frame_rows, row)


def instrument_score(stack: contextlib.ExitStack, cls: Any,
                     rec: CompletionRecorder) -> None:
    """OCRの生戻り値とScoreDeltaを再読せず保存する。"""
    original = cls._apply_read

    @functools.wraps(original)
    def apply(tracker: Any, cur: int | None) -> Any:
        result = original(tracker, cur)
        rec.record_score(result, cur)
        return result

    base.patch(stack, cls, "_apply_read", apply)


def instrument_formula(stack: contextlib.ExitStack, cls: Any,
                       rec: CompletionRecorder) -> None:
    """既存accumulator更新の戻りと確定済み段だけを保存する。"""
    original, original_reset = cls.update, cls._reset_session

    @functools.wraps(original)
    def update(owner: Any, *args: Any, **kwargs: Any) -> Any:
        before = owner.step_count
        result = original(owner, *args, **kwargs)
        rec.record_formula(_caller_side(sys._getframe(1), owner), owner, result, before)
        return result

    @functools.wraps(original_reset)
    def reset(owner: Any) -> Any:
        caller = sys._getframe(1)
        side = _caller_side(caller, owner)
        result = original_reset(owner)
        rec.record_formula_reset(side, owner, caller.f_code.co_name)
        return result

    base.patch(stack, cls, "update", update)
    base.patch(stack, cls, "_reset_session", reset)


def instrument_start(stack: contextlib.ExitStack, cls: Any,
                     rec: CompletionRecorder) -> None:
    """新規event開始とその時点のprecomputed物理予測を結ぶ。"""
    original = cls._start_chain_estimate

    @functools.wraps(original)
    def start(pipe: Any, side: str, event: Any,
              precomputed_result: Any = None) -> Any:
        result = original(pipe, side, event, precomputed_result)
        suffix = "1p" if side == "1P" else "2p"
        stored = getattr(pipe, f"_chain_estimate_result_{suffix}", None)
        rec.record_start(side, event, stored)
        return result

    base.patch(stack, cls, "_start_chain_estimate", start)


def instrument_simulate(stack: contextlib.ExitStack, cls: Any,
                        rec: CompletionRecorder) -> None:
    """C-6直呼びの既存simulate戻り値だけを追加計算なしで保存する。"""
    original = cls.simulate

    @functools.wraps(original)
    def simulate(owner: Any, board: Any) -> Any:
        caller = sys._getframe(1)
        result = original(owner, board)
        if caller.f_code.co_name == "_step_side":
            rec.record_prediction(_caller_side(caller), result, "_step_side/C-6")
        return result

    base.patch(stack, cls, "simulate", simulate)


def completion_prepare(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """45秒固定窓と全import・対照rootを開始終了SHA guardへ追加する。"""
    if args.start_sec != START_SEC or args.end_sec != END_SEC:
        raise ValueError("completion receiptは560～605秒の固定区間です")
    receipt, config = ORIGINAL_COMPLETION_PREPARE(args)
    _, ref_hashes = prechain._reference_receipt(ATOMIC_REFERENCE)
    paths = (Path(__file__), LAUNCHER, TEST, DOC, Path(details.__file__),
             Path(prechain.__file__), Path(atomic.__file__), details.DETAIL_TEST,
             prechain.TEST, atomic.TEST)
    receipt["input_and_code_sha256"].update(ref_hashes)
    receipt["input_and_code_sha256"].update({str(path): base.sha256(path) for path in paths})
    import torch
    receipt.update(format_version=FORMAT, completion_receipt_scope={
        "interval_sec": [START_SEC, END_SEC], "baseline": str(ATOMIC_REFERENCE),
        "source_frames": [START_FRAME, END_FRAME], "frame_stride": FRAME_STRIDE,
        "expected_pipeline_frames": (END_FRAME - START_FRAME) // FRAME_STRIDE,
        "behavior": "atomic landing shadow unchanged; observation wrappers only",
        "raw_score_source": "ScoreTracker._apply_read -> ScoreDelta",
        "row0_policy": "candidate row0 never authenticates itself",
        "prefix_comparison": "560-583 only; 583-605 has no same-policy reference",
        "commit_or_publication_modified": False}, diagnostic_resources={
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads()})
    return receipt, config


def completion_instrument(stack: contextlib.ExitStack, collector: ModuleType,
                          rec: CompletionRecorder) -> None:
    """既存atomic shadow設置後に戻り値透過の観測wrapperだけを加える。"""
    ORIGINAL_ATOMIC_INSTRUMENT(stack, collector, rec)
    from src.chain import ChainSimulator
    from src.score_ocr import FormulaStepAccumulator, ScoreTracker
    cls = collector.RecognitionPipeline
    instrument_score(stack, ScoreTracker, rec)
    instrument_formula(stack, FormulaStepAccumulator, rec)
    instrument_start(stack, cls, rec)
    instrument_simulate(stack, ChainSimulator, rec)


def _visible_match(cnn: dict[str, Any] | None,
                   final: dict[str, Any] | None) -> bool:
    """row0を除く72cellの厳密一致だけを判定する。"""
    if not cnn or not final:
        return False
    raw_visible, final_visible = cnn["grid"][1:], final["grid"][1:]
    known = all(value != 10 for grid in (raw_visible, final_visible)
                for row in grid for value in row)
    return known and raw_visible == final_visible


def _consensus_evidence(rows: list[dict[str, Any]], final: dict[str, Any],
                        required: int) -> dict[str, Any]:
    """連続STABLE厳密一致と、その生row1に由来する重力EMPTY列を返す。"""
    run: list[dict[str, Any]] = []
    best: list[dict[str, Any]] = []
    for row in rows:
        raw = row.get("raw_before_accounting")
        if str(row["state"]).lower() == "stable" and _visible_match(raw, final):
            run.append(row)
            if len(run) > len(best):
                best = list(run)
        else:
            run = []
    chosen = best[:required] if len(best) >= required else []
    row1 = None if not chosen else chosen[-1]["raw_before_accounting"]["grid"][1]
    gravity = [] if row1 is None else [index for index, value in enumerate(row1) if value == 0]
    candidate_row0 = final["grid"][0]
    return {"required_consecutive_frames": required, "max_exact_run": len(best),
            "consensus_frame_indices": [row["frame_idx"] for row in chosen],
            "observed_row1": row1, "gravity_empty_columns": gravity,
            "row0_source": "same-frame accounting-capture raw STABLE row1; GRAVITY_EMPTY rule",
            "candidate_row0": candidate_row0,
            "row0_candidate_supported": bool(chosen) and len(gravity) == 6
                and all(value == 0 for value in candidate_row0),
            "commit_permission_issued": False}


def _score_evidence(lineage: dict[str, Any], rows: list[dict[str, Any]],
                    predicted: int) -> dict[str, Any]:
    """凍結anchorからのraw score増分を比較し、欠測をunknownのまま残す。"""
    anchor = lineage.get("entry_score_anchor")
    timeline = []
    for row in rows:
        cur = row.get("raw_value")
        gain = None if anchor is None or cur is None else int(cur) - int(anchor)
        status = "unknown" if gain is None else (
            "numeric_exact_unverified_generation" if gain == predicted
            else "below" if gain < predicted else "above")
        timeline.append({"frame_idx": row["frame_idx"], "time_sec": row["time_sec"],
                         "raw_score": cur, "gain_from_anchor": gain, "status": status})
    exact = next((row for row in timeline
                  if row["status"] == "numeric_exact_unverified_generation"), None)
    return {"anchor": anchor, "anchor_status": lineage["anchor_status"],
            "anchor_source_frame": lineage.get("anchor_source_frame"),
            "predicted_total_score": predicted, "first_numeric_exact_candidate": exact,
            "completion_verified": False,
            "timeline": timeline}


def build_receipt(rec: CompletionRecorder) -> dict[str, Any]:
    """各eventをanchor・物理予測・可視/row0証拠へ結ぶ診断台帳を作る。"""
    output = []
    for prediction_row in rec.prediction_rows:
        lid = prediction_row["lineage_id"]
        lineage = next((row for row in rec.lineages if row["lineage_id"] == lid), None)
        if lineage is None or prediction_row["prediction"] is None:
            output.append({**prediction_row, "status": "unknown_unbound_prediction"})
            continue
        prediction = prediction_row["prediction"]
        scores = [row for row in rec.score_rows if row["lineage_id"] == lid
                  and row["frame_idx"] >= prediction_row["frame_idx"]]
        frames = [row for row in rec.frame_rows if row["lineage_id"] == lid
                  and row["frame_idx"] >= prediction_row["frame_idx"]]
        score = _score_evidence(lineage, scores, prediction["calculated_total_score"])
        visible = _consensus_evidence(frames, prediction["final_board"], rec.verify_frames)
        output.append({**prediction_row, "lineage": lineage, "score_evidence": score,
                       "visible_and_row0_evidence": visible,
                       "combined_closure_status": "not_evaluated_missing_epoch_action_binding",
                       "formula_revisions": [row for row in rec.formula_rows
                           if row["lineage_id"] == lid
                           and row["frame_idx"] >= prediction_row["frame_idx"]],
                       "commit_permission_issued": False})
    return {"format_version": FORMAT, "status": "diagnostic_only_not_adopted",
            "prediction_revisions": output,
            "lineages_without_prediction": [row for row in rec.lineages
                if not any(p["lineage_id"] == row["lineage_id"]
                           for p in rec.prediction_rows)],
            "missing_generation_binding": ["reset_epoch", "action_revision"],
            "missing_observation_is_never_zero_or_pass": True,
            "row0_candidate_is_not_evidence": True, "commit_permission_issued": False}


def _canonical_rows(path: Path, kind: str) -> list[str]:
    """指定kindだけを順序・全field込みのcanonical JSON列として読む。"""
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if value.get("kind") == kind:
                rows.append(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")))
    return rows


def _prefix_comparison(candidate: Path) -> dict[str, Any]:
    """同条件atomic 23秒prefixの公開行・collector行を完全一致照合する。"""
    result: dict[str, Any] = {}
    for kind, expected in (("frame_side", 1380), ("collector_snapshot", 53)):
        reference = _canonical_rows(ATOMIC_REFERENCE / "frames.jsonl", kind)
        current = _canonical_rows(candidate, kind)
        if len(reference) != expected or len(current) < expected:
            raise RuntimeError(f"{kind} prefix行数不一致: {len(reference)}/{len(current)}")
        prefix = current[:expected]
        if prefix != reference:
            mismatch = next(index for index, pair in enumerate(zip(prefix, reference))
                            if pair[0] != pair[1])
            raise RuntimeError(f"atomic prefix不一致: {kind}[{mismatch}]")
        digest = hashlib.sha256(("\n".join(prefix) + "\n").encode()).hexdigest()
        result[kind] = {"reference_count": expected, "prefix_count": len(prefix),
                        "prefix_bit_exact": True, "prefix_sha256": digest,
                        "tail_count": len(current) - expected}
    result["scope"] = "560<=time<583 only; extended tail is observed without reference"
    return result


def completion_finish(output: Path, receipt: dict[str, Any],
                      rec: CompletionRecorder, elapsed: float) -> dict[str, Any]:
    """診断receiptを排他保存し、既存COMPLETEのSHA台帳へ含める。"""
    if rec.frames != (END_FRAME - START_FRAME) // FRAME_STRIDE:
        raise RuntimeError(f"45秒frame数不一致: {rec.frames}")
    result = build_receipt(rec)
    result["atomic_prefix_comparison"] = _prefix_comparison(output / "frames.jsonl")
    result["extended_tail_comparison_performed"] = False
    base.assert_unchanged(receipt["input_and_code_sha256"])
    base.write_json(output / "COMPLETION_RECEIPT.json", result)
    original_write = base.write_json

    def write(path: Path, value: Any) -> None:
        if path.name == "COMPLETE":
            value = dict(value)
            value["sha256"] = {**value["sha256"], "COMPLETION_RECEIPT.json":
                               base.sha256(output / "COMPLETION_RECEIPT.json")}
        original_write(path, value)

    with contextlib.ExitStack() as stack:
        base.patch(stack, base, "write_json", write)
        summary = ORIGINAL_SHADOW_FINISH(output, receipt, rec, elapsed)
    return {**summary, "completion_receipt": result}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """fresh process内だけ45秒上限と既存shadow構成点を差し替える。"""
    with contextlib.ExitStack() as stack:
        base.patch(stack, base, "MAX_DURATION_SEC", END_SEC - START_SEC)
        base.patch(stack, base, "END_SEC", END_SEC)
        base.patch(stack, atomic, "ORIGINAL_PREPARE", completion_prepare)
        base.patch(stack, atomic, "AtomicRecorder", CompletionRecorder)
        base.patch(stack, atomic, "atomic_instrument", completion_instrument)
        base.patch(stack, prechain, "shadow_finish", completion_finish)
        return atomic.run(args)


def _configure_torch_threads() -> None:
    """fresh診断processのtorch CPU thread数を他数値libraryと同じ2へ固定する。"""
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)


def main() -> int:
    """固定45秒・新規root・native差明示許可だけを受け付ける。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, default=START_SEC)
    parser.add_argument("--end-sec", type=float, default=END_SEC)
    parser.add_argument("--allow-native-runtime-mismatch", action="store_true")
    _configure_torch_threads()
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
