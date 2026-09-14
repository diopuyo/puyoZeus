"""completion receipt v1を変えず、予測ledgerとsoftware世代だけを横に記録する。"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import hashlib
import importlib.util
import json
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

from scripts import chain_prediction_generation_hooks_v1 as generation_hooks
from scripts import diagnose_video38_completion_receipt_v1 as completion


base = completion.base
FORMAT = "video38-prediction-ledger-shadow/v1"
START_SEC, END_SEC = completion.START_SEC, completion.END_SEC
REFERENCE = base.VERIFY / "video38_completion_receipt_2026-09-07_v1"
LAUNCHER = base.ROOT / "scripts/launch_video38_prediction_ledger_shadow_v1.sh"
TEST = base.ROOT / "tests/test_diagnose_video38_prediction_ledger_shadow_v1.py"
LEDGER_SOURCE = base.ROOT / "src/chain_prediction_ledger_v1.py"
LEDGER_TEST = base.ROOT / "tests/test_chain_prediction_ledger_v1.py"
COMMIT_SOURCE = base.ROOT / "src/chain_commit_candidate_v1.py"
COMMIT_TEST = base.ROOT / "tests/test_chain_commit_candidate_v1.py"
GENERATION_TEST = base.ROOT / "tests/test_chain_prediction_generation_hooks_v1.py"
ORIGINAL_PREPARE = completion.completion_prepare
ORIGINAL_INSTRUMENT = completion.completion_instrument
ORIGINAL_FINISH = completion.completion_finish
_PREPARED_GUARDS: dict[str, str] | None = None
_MISSING = object()


def _json_value(value: Any) -> Any:
    """dataclass/EnumをJSON化し、handleのprocess-local tokenを除く。"""
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name))
                for field in dataclasses.fields(value) if field.name != "_owner_token"}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return base.json_value(value)


def _restore_module(name: str, previous: Any) -> None:
    """動的load前のsys.modules状態へ厳密に戻す。"""
    if previous is _MISSING:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous


def _load_guarded(
    stack: contextlib.ExitStack, name: str, path: Path, expected: str,
) -> ModuleType:
    """current helper一つを開始時SHAに照合して明示module名でloadする。"""
    if base.sha256(path) != expected:
        raise ValueError(f"動的load直前のSHA不一致: {path}")
    if name in sys.modules:
        loaded_path = getattr(sys.modules[name], "__file__", None)
        raise RuntimeError(f"current helperが予期せず既loadです: {name}={loaded_path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module specを作成できません: {path}")
    previous = _MISSING
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    stack.callback(_restore_module, name, previous)
    try:
        spec.loader.exec_module(module)
    except Exception:
        _restore_module(name, previous)
        raise
    loaded_path = Path(str(getattr(module, "__file__", ""))).resolve()
    if loaded_path != path.resolve() or base.sha256(loaded_path) != expected:
        _restore_module(name, previous)
        raise RuntimeError(f"load後module identityがguardと不一致です: {name}")
    return module


def _load_current_ledger(
    stack: contextlib.ExitStack, guards: dict[str, str],
) -> ModuleType:
    """frozen srcを維持し、新規commit型とledgerだけをcurrentからloadする。"""
    _load_guarded(
        stack, "src.chain_commit_candidate_v1", COMMIT_SOURCE,
        guards[str(COMMIT_SOURCE)],
    )
    return _load_guarded(
        stack, "src.chain_prediction_ledger_v1", LEDGER_SOURCE,
        guards[str(LEDGER_SOURCE)],
    )


class PredictionLedgerRecorder(completion.CompletionRecorder):
    """既存completion行を保持し、ledger保存結果を追加kindへ分離する。"""

    def __init__(self, stream: Any, target: dict[str, Any]) -> None:
        super().__init__(stream, target)
        self.ledger_module: ModuleType | None = None
        self.ledger: Any = None
        self.generation_recorder: Any = None
        self.active_handles: dict[str, Any] = {}
        self.all_handles: list[Any] = []
        self.generation_rows: list[dict[str, Any]] = []
        self.ledger_rows: list[dict[str, Any]] = []
        self.rejections: list[dict[str, Any]] = []
        self.last_generation_row: dict[str, dict[str, Any]] = {}
        self.start_episode: dict[str, tuple[Any, Any]] = {}

    def bind_runtime(self, ledger_module: ModuleType, generation_recorder: Any) -> None:
        """collector load後のcurrent ledgerとgeneration recorderを一度だけ接続する。"""
        if self.ledger is not None or self.generation_recorder is not None:
            raise ValueError("ledger runtimeは一度だけ接続できます")
        self.ledger_module = ledger_module
        self.ledger = ledger_module.ChainPredictionLedger()
        self.generation_recorder = generation_recorder

    def record_generation(self, row: dict[str, Any]) -> None:
        """software世代行を原時計のまま保存し、既handleを既知時刻で失効する。"""
        value = base.json_value(row)
        self.generation_rows.append(value)
        self.last_generation_row[str(row["side"])] = value
        self.emit(value)
        if row.get("frame_idx") is not None:
            self._sync_handle(str(row["side"]), value)

    def record_start(self, side: str, event: Any, result: Any) -> None:
        """初回landingだけprovisionalを作り、再startはepisodeへ追加する。"""
        episode = self._record_ledger_episode(side, event)
        if episode is not None:
            self.start_episode[side] = (episode, event.before_board.copy())
        try:
            super().record_start(side, event, result)
        finally:
            self.start_episode.pop(side, None)
        if result is None:
            self._reject("prediction", side, "start_result_unknown", None)

    def record_prediction(self, side: str | None, result: Any, caller: str) -> None:
        """既存prediction行を先に保存し、同じresultを再計算なしでledgerへ渡す。"""
        super().record_prediction(side, result, caller)
        if side not in ("1P", "2P"):
            self._reject("prediction", side, "side_unknown", caller)
            return
        handle, current = self._active(side, "prediction")
        if current is None:
            return
        if handle is None:
            self._reject("prediction", side, "no_landing_provisional", caller)
            return
        start_value = self.start_episode.get(side)
        if start_value is None:
            self._reject("prediction", side, "prediction_without_exact_start_episode", caller)
            return
        if result is None or not getattr(result, "steps", None):
            self._reject("prediction", side, "chain_result_empty_or_unknown", caller)
            return
        episode, input_board = start_value
        self._add_prediction(handle, current, episode, input_board, result, caller)

    def record_c6_prediction(
        self, side: str | None, event: Any, input_board: Any,
        result: Any, caller: str,
    ) -> None:
        """旧C6行とactual event/input/resultを同じsimulate呼出しから保存する。"""
        super().record_prediction(side, result, caller)
        if side not in ("1P", "2P"):
            self._reject("c6_prediction", side, "side_unknown", caller)
            return
        handle, current = self._active(side, "c6_prediction")
        if current is None:
            return
        if handle is None or event is None:
            reason = "no_landing_provisional" if handle is None else "effective_event_unknown"
            self._reject("c6_prediction", side, reason, caller)
            return
        try:
            episode = self.ledger.add_episode(
                handle, generation=current, frame_idx=self.frame, time_sec=self.time_sec,
                event=event, capture_source="_step_side/C-6/effective_chain_event",
            )
            self._emit_ledger("episode", episode)
        except self.ledger_module.ChainPredictionLedgerError as exc:
            self._handle_ledger_error("c6_episode", side, exc, caller)
            return
        self._add_prediction(handle, current, episode, input_board, result, caller)

    def record_score(self, delta: Any, raw_value: int | None) -> None:
        """既存raw行の後に、承認を作らないscore証拠だけをledgerへ追加する。"""
        super().record_score(delta, raw_value)
        side = str(delta.side)
        handle, current = self._active(side, "raw_score")
        if handle is None:
            return
        try:
            evidence = self.ledger.record_raw_score(
                handle, generation=current, frame_idx=self.frame, time_sec=self.time_sec,
                source="ScoreDelta/_apply_read", raw_value=raw_value,
                prev_score=delta.prev_score, cur_score=delta.cur_score,
                delta=delta.delta, is_valid=bool(delta.is_valid),
            )
        except self.ledger_module.ChainPredictionLedgerError as exc:
            self._handle_ledger_error("raw_score", side, exc)
        else:
            self._emit_ledger("raw_score", evidence)

    def record_formula(self, side: str | None, owner: Any, step: Any,
                       before_count: int) -> None:
        """既存formula行と同じ有意更新だけをbound sessionへ保存する。"""
        super().record_formula(side, owner, step, before_count)
        if side not in ("1P", "2P") or (step is None and owner.step_count == before_count):
            return
        session = self.formula_sessions.get(id(owner), 0)
        self._record_formula_value(side, session, owner, step, "formula_observation")

    def record_formula_reset(self, side: str | None, owner: Any, caller: str) -> None:
        """初回step0 resetをsession接続根拠として保存する。"""
        super().record_formula_reset(side, owner, caller)
        if side not in ("1P", "2P"):
            return
        session = self.formula_sessions[id(owner)]
        self._record_formula_value(side, session, owner, None, f"formula_reset/{caller}")

    def _record_ledger_episode(self, side: str, event: Any) -> Any | None:
        """世代同期後にorigin openまたはepisode追記を行う。"""
        handle, current = self._active(side, "episode")
        if current is None:
            return None
        mechanism = getattr(event, "mechanism", None)
        try:
            if handle is None:
                if mechanism != "landing":
                    self._reject("episode", side, "first_event_not_landing", mechanism)
                    return None
                handle = self.ledger.open_landing_provisional(
                    generation=current, frame_idx=self.frame, time_sec=self.time_sec,
                    origin_before_board=event.before_board, landing_event=event,
                    capture_source="_start_chain_estimate/landing",
                )
                self.active_handles[side] = handle
                self.all_handles.append(handle)
                self._emit_ledger("provisional_open", self.ledger.snapshot(handle))
                return self.ledger.snapshot(handle).episodes[0]
            episode = self.ledger.add_episode(
                handle, generation=current, frame_idx=self.frame, time_sec=self.time_sec,
                event=event, capture_source=f"_start_chain_estimate/{mechanism or 'unknown'}",
            )
            self._emit_ledger("episode", episode)
            return episode
        except self.ledger_module.ChainPredictionLedgerError as exc:
            self._handle_ledger_error("episode", side, exc)
            return None

    def _record_formula_value(
        self, side: str, session: int, owner: Any, step: Any, source: str,
    ) -> None:
        """formula値をanchorへ変換せず、出所とsessionを保ったまま追記する。"""
        handle, current = self._active(side, "formula")
        if handle is None:
            return
        try:
            evidence = self.ledger.record_formula(
                handle, generation=current, frame_idx=self.frame, time_sec=self.time_sec,
                source=source, session_id=session, step_index=owner.step_count,
                total_power=owner.total_power,
                step_product=None if step is None else step.product,
            )
        except self.ledger_module.ChainPredictionLedgerError as exc:
            self._handle_ledger_error("formula", side, exc)
        else:
            self._emit_ledger("formula", evidence)

    def _active(self, side: str, stage: str) -> tuple[Any | None, Any | None]:
        """次の既知時計で世代差を反映し、現在generationを返す。"""
        if not self._clock_active():
            self._reject(stage, side, "outside_update_clock_unknown", None)
            return None, None
        current = self._current_generation(side)
        self._sync_handle(side, self.last_generation_row.get(side))
        return self.active_handles.get(side), current

    def _clock_active(self) -> bool:
        """前回frameを流用せずgeneration hookの現在update時計と照合する。"""
        recorder = self.generation_recorder
        if recorder is None or not bool(getattr(recorder, "_in_frame", False)):
            return False
        frame, time_sec = getattr(recorder, "_frame", None), getattr(recorder, "_time", None)
        return frame == self.frame and time_sec == self.time_sec

    def _current_generation(self, side: str) -> Any:
        """software generationをUNKNOWN補完せずledger型へ写す。"""
        if self.ledger is None or self.generation_recorder is None:
            raise RuntimeError("ledger/generation runtimeが未接続です")
        value = self.generation_recorder.generation(side)
        return self.ledger_module.ChainGeneration(
            value.side, value.reset_epoch, value.action_revision,
        )

    def _sync_handle(self, side: str, transition: dict[str, Any] | None) -> None:
        """世代差があれば、遷移時計と失効可能時計を分けて旧handleを失効する。"""
        handle = self.active_handles.get(side)
        if handle is None or not self._clock_active():
            return
        current = self._current_generation(side)
        snapshot = self.ledger.snapshot(handle)
        if snapshot.generation == current:
            return
        old = snapshot.generation
        receipt = self.ledger.invalidate(
            handle, generation=old, frame_idx=self.frame, time_sec=self.time_sec,
            reason="software_generation_changed_not_physical_identity",
        )
        self.active_handles.pop(side, None)
        self._emit_ledger("generation_invalidation", {
            "receipt": receipt, "old_generation": old, "new_generation": current,
            "transition_observation": transition,
            "invalidation_available_at": {"frame_idx": self.frame, "time_sec": self.time_sec},
        })

    def _add_prediction(
        self, handle: Any, current: Any, episode: Any,
        input_board: Any, result: Any, caller: str,
    ) -> None:
        """実resultを一度だけledgerへ渡し、拒否も診断行として残す。"""
        try:
            prediction = self.ledger.add_prediction(
                handle, generation=current, frame_idx=self.frame, time_sec=self.time_sec,
                episode_revision=episode.episode_revision,
                input_board=input_board, result=result,
            )
        except self.ledger_module.ChainPredictionLedgerError as exc:
            self._handle_ledger_error("prediction", str(current.side), exc, caller)
        else:
            self._emit_ledger("prediction", {"caller": caller, "value": prediction})

    def _handle_ledger_error(
        self, stage: str, side: str, exc: Exception, detail: Any = None,
    ) -> None:
        """ledgerの期待されたfail-closedを旧診断を止めず理由化する。"""
        handle = self.active_handles.get(side)
        if handle is not None:
            snapshot = self.ledger.snapshot(handle)
            if snapshot.status.value == "invalidated":
                self.active_handles.pop(side, None)
        self._reject(stage, side, f"{type(exc).__name__}:{exc}", detail)

    def _reject(self, stage: str, side: Any, reason: str, detail: Any) -> None:
        """unknown/chain0/未bindingを成功や0へ変換せず保存する。"""
        unknown_clock = reason == "outside_update_clock_unknown"
        row = base.json_value({"stage": stage, "side": side, "reason": reason,
            "detail": detail, "frame_idx": None if unknown_clock else self.frame,
            "time_sec": None if unknown_clock else self.time_sec,
            "clock_source": ("outside_update_time_unknown" if unknown_clock
                             else "pipeline_update_input"),
            "commit_permission_issued": False})
        self.rejections.append(row)
        self._emit_ledger("rejected", row, clock_unknown=unknown_clock)

    def _emit_ledger(self, name: str, value: Any, *, clock_unknown: bool = False) -> None:
        """ledger追加行を旧kindと衝突しないnamespaceで保存する。"""
        row = {"kind": f"prediction_ledger_{name}", "value": _json_value(value),
               "commit_permission_issued": False}
        if clock_unknown:
            row.update(frame_idx=None, time_sec=None,
                       clock_source="outside_update_time_unknown")
        self.ledger_rows.append(base.json_value(row))
        self.emit(row)

    def ledger_receipt(self) -> dict[str, Any]:
        """全handleを失効後も含め、owner tokenなしの不変receiptへ変換する。"""
        snapshots = [_json_value(self.ledger.snapshot(handle)) for handle in self.all_handles]
        prediction_count = sum(len(item["predictions"]) for item in snapshots)
        return {"format_version": FORMAT, "status": "diagnostic_only_not_adopted",
                "identity_scope": generation_hooks.IDENTITY_SCOPE,
                "software_generation_is_not_physical_identity": True,
                "snapshot_count": len(snapshots), "prediction_count": prediction_count,
                "snapshots": snapshots, "generation_rows": self.generation_rows,
                "rejections": self.rejections, "anchor_binding_performed": False,
                "completion_or_publication_evaluated": False,
                "commit_permission_issued": False}


def ledger_prepare(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """completion v1全guardと全45秒referenceへ新adapter資産を追加する。"""
    global _PREPARED_GUARDS
    if args.start_sec != START_SEC or args.end_sec != END_SEC:
        raise ValueError("prediction ledger shadowは560～605秒の固定区間です")
    receipt, config = ORIGINAL_PREPARE(args)
    reference_receipt, reference_hashes = completion.prechain._reference_receipt(REFERENCE)
    paths = (Path(__file__), LAUNCHER, TEST, LEDGER_SOURCE, LEDGER_TEST,
             COMMIT_SOURCE, COMMIT_TEST, Path(generation_hooks.__file__), GENERATION_TEST)
    own_hashes = {str(path): base.sha256(path) for path in paths}
    receipt["input_and_code_sha256"].update(reference_hashes)
    receipt["input_and_code_sha256"].update(own_hashes)
    receipt.update(format_version=FORMAT, prediction_ledger_shadow={
        "interval_sec": [START_SEC, END_SEC], "reference": reference_receipt,
        "legacy_comparison": "all completion-v1 JSONL rows after excluding new kinds",
        "origin_policy": "first landing provisional only; no late origin promotion",
        "generation_scope": generation_hooks.IDENTITY_SCOPE,
        "resimulation_performed": False, "anchor_binding_performed": False,
        "commit_or_publication_modified": False, "production_modified": False})
    _PREPARED_GUARDS = dict(receipt["input_and_code_sha256"])
    return receipt, config


def instrument_simulate_exact(
    stack: contextlib.ExitStack, cls: Any, rec: PredictionLedgerRecorder,
) -> None:
    """C-6のactual event/input/resultを旧行と同じ一回の呼出しで記録する。"""
    original = cls.simulate

    @functools.wraps(original)
    def simulate(owner: Any, board: Any) -> Any:
        caller = sys._getframe(1)
        result = original(owner, board)
        if caller.f_code.co_name == "_step_side":
            event = caller.f_locals.get("_effective_chain_event")
            rec.record_c6_prediction(
                completion._caller_side(caller), event, board, result, "_step_side/C-6",
            )
        return result

    base.patch(stack, cls, "simulate", simulate)


def ledger_instrument(
    stack: contextlib.ExitStack, collector: ModuleType, rec: PredictionLedgerRecorder,
) -> None:
    """旧simulate入口を一個のexact wrapperへ差替え、世代hookを追加する。"""
    base.patch(stack, completion, "instrument_simulate", instrument_simulate_exact)
    ORIGINAL_INSTRUMENT(stack, collector, rec)
    if _PREPARED_GUARDS is None:
        raise RuntimeError("prepare済みSHA guardがありません")
    ledger_module = _load_current_ledger(stack, _PREPARED_GUARDS)
    generation_recorder = generation_hooks.PipelineGenerationRecorder(rec.record_generation)
    rec.bind_runtime(ledger_module, generation_recorder)
    pipeline_class = collector.RecognitionPipeline
    pipeline_module = sys.modules[pipeline_class.__module__]
    machine_class = getattr(pipeline_module, "BoardStateMachine")
    generation_hooks.install_generation_hooks(
        stack, pipeline_class, machine_class, generation_recorder,
    )


def _is_new_kind(kind: Any) -> bool:
    """adapter追加行だけをlegacy比較から除く。"""
    return kind == "software_generation" or str(kind).startswith("prediction_ledger_")


def _canonical_legacy(path: Path, *, candidate: bool) -> list[str]:
    """旧JSONL行を順序・全field込みのcanonical列にする。"""
    rows: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if candidate and _is_new_kind(value.get("kind")):
                continue
            rows.append(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")))
    return rows


def _full_legacy_comparison(candidate: Path) -> dict[str, Any]:
    """completion v1の45秒全行が追加計装後も完全一致することを検査する。"""
    expected = _canonical_legacy(REFERENCE / "frames.jsonl", candidate=False)
    actual = _canonical_legacy(candidate, candidate=True)
    if actual != expected:
        mismatch = next((index for index, pair in enumerate(zip(actual, expected))
                         if pair[0] != pair[1]), min(len(actual), len(expected)))
        raise RuntimeError(
            f"completion v1全行不一致: index={mismatch}, {len(actual)}/{len(expected)}",
        )
    payload = ("\n".join(actual) + "\n").encode()
    return {"row_count": len(actual), "all_legacy_rows_bit_exact": True,
            "canonical_sha256": hashlib.sha256(payload).hexdigest(),
            "scope": "full 560<=time<605; all legacy kinds and fields"}


def ledger_finish(
    output: Path, receipt: dict[str, Any], rec: PredictionLedgerRecorder, elapsed: float,
) -> dict[str, Any]:
    """ledger receiptを先に排他保存し、既存COMPLETEのSHA台帳へ追加する。"""
    result = rec.ledger_receipt()
    result["legacy_completion_v1_comparison"] = _full_legacy_comparison(
        output / "frames.jsonl",
    )
    base.assert_unchanged(receipt["input_and_code_sha256"])
    base.write_json(output / "LEDGER_RECEIPT.json", result)
    original_write = base.write_json

    def write(path: Path, value: Any) -> None:
        if path.name == "COMPLETE":
            value = dict(value)
            value["sha256"] = {**value["sha256"], "LEDGER_RECEIPT.json":
                               base.sha256(output / "LEDGER_RECEIPT.json")}
        original_write(path, value)

    with contextlib.ExitStack() as stack:
        base.patch(stack, base, "write_json", write)
        summary = ORIGINAL_FINISH(output, receipt, rec, elapsed)
    return {**summary, "prediction_ledger_receipt": result}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """completion v1拡張点だけをfresh process内で一時差替えする。"""
    global _PREPARED_GUARDS
    try:
        with contextlib.ExitStack() as stack:
            base.patch(stack, completion, "completion_prepare", ledger_prepare)
            base.patch(stack, completion, "CompletionRecorder", PredictionLedgerRecorder)
            base.patch(stack, completion, "completion_instrument", ledger_instrument)
            base.patch(stack, completion, "completion_finish", ledger_finish)
            return completion.run(args)
    finally:
        _PREPARED_GUARDS = None


def main() -> int:
    """固定45秒・新規root・native差明示許可だけを受け付ける。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, default=START_SEC)
    parser.add_argument("--end-sec", type=float, default=END_SEC)
    parser.add_argument("--allow-native-runtime-mismatch", action="store_true")
    completion._configure_torch_threads()
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
