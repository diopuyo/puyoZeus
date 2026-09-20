"""最新背景＋既存C6のsoftware所有と各実盤面段階を計測する。"""
from __future__ import annotations
import ast
import contextlib
import dataclasses
import functools
import inspect
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

HOME = Path(__file__).resolve().parent
ROOT = HOME.parents[2]
CONNECTION = HOME.parent / "g2_c6_latest_background_connection_2026-09-08_v1/connection.py"
CONNECTION_SHA = "8d555caa0b1cc9a4a61a2b736b529177fdd43ae4c2b3c3550f6f160eefe70364"
WRITER = HOME.parent / "g2_landing_grace_writer_observer_2026-09-08_v1/observer.py"
WRITER_SHA = "a43ebd79769a35e891934ad8563ba422027cc64eb0232e6ca7814a02b7f6f0c9"
WINDOWS = ((32640, 32800), (34700, 36000))
SIDECAR, LEDGER = "split_runtime_evidence.jsonl", "split_pending_ledger.jsonl"
REPORT, STATUS = "SPLIT_EVIDENCE_RECEIPT.json", "SPLIT_EVIDENCE_STATUS.json"
EXTERNAL_SOURCE_SHA = {
    "src.chain_commit_candidate_v1": "443b79a896c8e46a268fab8e9f54b0a5238940bf4495706041423ee7de9596c5",
    "src.chain_prediction_ledger_v1": "2dcf09dcdad08c62fe0384a34e4d1d6ed85adf5a1ee4f3884c76dde49f5319d7",
}
A: Any = None
base: Any = None
pending: Any = None
hooks: Any = None
writer: Any = None


def initialize(latest: Any) -> None:
    """CPUでは既ロード実adapterを受け、liveではrunnerが正本をロードする。"""
    global A, base, pending, hooks, writer
    A, base = latest, latest.base
    from scripts import diagnose_video38_c6_pending_commit_shadow_v1 as module
    pending = module
    base.assert_unchanged({str(CONNECTION): CONNECTION_SHA, str(WRITER): WRITER_SHA})
    names = {"insert_before_grace", "publication_after_next", "ledger_hooks"}
    tree = ast.parse(CONNECTION.read_text(encoding="utf-8"), filename=str(CONNECTION))
    selected_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(selected_nodes) != len(names):
        raise RuntimeError("connection_functions_missing")
    namespace = dict(A=A, base=base, pending=pending, Path=Path, Any=Any, sys=sys, functools=functools)
    exec(compile(ast.Module(body=selected_nodes, type_ignores=[]), str(CONNECTION), "exec"), namespace)
    hooks = SimpleNamespace(**{name: namespace[name] for name in names})
    writer = A.load("_split_reused_writer", WRITER)


def selected(frame: int, side: str = "2P") -> bool:
    return type(frame) is int and side == "2P" and frame % 2 == 0 and any(a <= frame <= b for a, b in WINDOWS)


def expected_scopes() -> list[int]:
    return [frame for start, end in WINDOWS for frame in range(start, end + 1, 2)]


def owner_value(rec: Any, side: str) -> dict[str, Any]:
    handle = rec.active_handles.get(side)
    snapshot = None if handle is None else pending.prediction._json_value(rec.ledger.snapshot(handle))
    candidate = rec.active_candidates.get(side)
    return {"active_snapshot": snapshot, "active_candidate": None if candidate is None
            else pending.prediction._json_value(candidate["candidate"]), "game_identity": None,
            "accounting_basis_verified": False, "physical_identity_certified": False}


class Sink:
    """既存Sidecarの例外sticky／排他保存を利用する。"""
    def __init__(self, path: Path, history: Any, rec: Any, tracker: Any) -> None:
        self.base_sink = A.entry.Sidecar(path, history)
        self.rec, self.tracker = rec, tracker

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self.base_sink.rows

    def failed(self, exc: BaseException, location: str) -> None:
        self.base_sink.failed(exc, location)

    def emit(self, row: dict[str, Any]) -> None:
        try:
            frame, clock, side = row["frame_idx"], row["time_sec"], row["side"]
            if not selected(frame, side) or type(clock) not in (int, float) or not math.isfinite(clock):
                raise ValueError("evidence_scope")
            if clock != frame / 60 or (self.tracker._frame, self.tracker._time) != (frame, clock):
                raise ValueError("generation_clock_mismatch")
            if self.tracker._pipeline is None:
                raise ValueError("unbound_pipeline")
            value = {**row, "generation": dataclasses.asdict(self.tracker.generation(side)),
                "owner": owner_value(self.rec, side), "game_identity": None,
                "accounting_basis_verified": False, "physical_identity_certified": False,
                "current_publication_allowed": False, "accounting_commit_allowed": False}
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
            self.base_sink.stream.write(encoded + "\n")
            self.rows.append(json.loads(encoded))
        except BaseException as exc:
            self.failed(exc, "evidence_emit")
            raise

    def close(self) -> None:
        self.base_sink.close()

    def summary(self, expected: list[int] | None = None) -> dict[str, Any]:
        self.base_sink.require_ok()
        wanted = expected_scopes() if expected is None else expected
        for suffix in ("step_enter", "step_return", "public_return"):
            frames = [row["frame_idx"] for row in self.rows if row["kind"].endswith("_" + suffix)]
            if frames != wanted:
                raise ValueError("scope_coverage_" + suffix)
        for frame in wanted:
            rows = [row for row in self.rows if row["frame_idx"] == frame]
            for kind in ("sm", "infer", "resolve", "detector"):
                calls = [r for r in rows if r["kind"] == "chigiri_exit_" + kind + "_call"]
                returns = [r for r in rows if r["kind"] == "chigiri_exit_" + kind + "_return"]
                if len(calls) != len(returns) or (kind == "sm" and len(calls) != 1):
                    raise ValueError("actual_call_pair_missing_" + kind)
            if any(r.get("unwinding_exception") or r["kind"].endswith("_exception") for r in rows):
                raise ValueError("observed_exception")
        return {"scope_count": len(wanted), "rows": len(self.rows), "physical_identity_certified": False,
                "quality_gate_clear": False, "accounting_basis_verified": False}


def runtime_codes(history: Any, original: Any) -> tuple[Any, Any]:
    """原本unwrapではなく実C6/grace合成後のhistory codeを選ぶ。"""
    raw = inspect.unwrap(original)
    infer = inspect.unwrap(raw.__globals__["infer_placement"])
    if Path(history.step_code.co_filename).resolve() != (base.SNAPSHOT / "src/recognition_pipeline.py").resolve():
        raise ValueError("actual_composite_code_path")
    if history.step_code is raw.__code__:
        raise ValueError("missing_composite_code")
    return history.step_code, infer.__code__


def install_trace(stack: Any, collector: Any, history: Any, sink: Sink, pending_code: Any) -> Any:
    code = runtime_codes(history, collector.RecognitionPipeline._step_side)
    facade = SimpleNamespace(WriterTrace=writer.WriterTrace, validate_code=lambda entry, original: code)
    base.patch(stack, A.serializer, "selected", lambda frame: selected(frame))
    trace = A.serializer.make_trace(facade, A.entry, collector, history, sink)
    # grace窓外はC6変換だけの実codeを呼ぶ。同じfilenameでも別codeを登録する。
    if Path(pending_code.co_filename).resolve() != Path(code[0].co_filename).resolve():
        raise ValueError("pending_code_path")
    trace.registry[pending_code] = "step"
    original = collector.RecognitionPipeline._step_side
    traced = trace.wrapper(original)
    @functools.wraps(original)
    def step(pipe: Any, side: str, frame: int, clock: float, *args: Any, **kwargs: Any) -> Any:
        if not selected(frame, side):
            return original(pipe, side, frame, clock, *args, **kwargs)
        if sink.tracker._pipeline is not pipe or sink.tracker._machines.get(side) is not kwargs.get("sm"):
            error = ValueError("actual_pipe_side_sm_mismatch")
            sink.failed(error, "step_identity")
            raise error
        return traced(pipe, side, frame, clock, *args, **kwargs)
    base.patch(stack, collector.RecognitionPipeline, "_step_side", step)
    return trace


def install_public(stack: Any, collector: Any, history: Any, sink: Sink) -> None:
    original = collector.RecognitionPipeline.update
    @functools.wraps(original)
    def update(pipe: Any, frame: int, clock: float, *args: Any, **kwargs: Any) -> Any:
        try:
            result = original(pipe, frame, clock, *args, **kwargs)
            if selected(frame):
                side = result.p2
                value = {name: A.serializer.json_value(SimpleNamespace(entry=A.entry), getattr(side, name, None))
                         for name in ("state", "confirmed_board", "prob_board", "inferred_board", "board_provenance")}
                sink.emit({"kind": "split_public_return", "frame_idx": frame, "time_sec": clock,
                           "side": "2P", "result": value, "snapshot": A.snapshot(pipe, "2P")})
            if sink.base_sink.error is not None:
                raise RuntimeError("sticky_evidence_error")
            return result
        except BaseException as exc:
            sink.failed(exc, "actual_update")
            raise
    base.patch(stack, collector.RecognitionPipeline, "update", update)


def close(state: dict[str, Any]) -> None:
    sink, rec = state["evidence_sink"], state["pending_rec"]
    try:
        sink.close()
        state["ledger_stream"].close()
    finally:
        base.write_json(state["output"] / STATUS, {"closed": sink.base_sink.closed,
            "sticky_error": sink.base_sink.error, "installed": True})


def external_guards() -> dict[str, str]:
    """追加を承認した型2名のみ固定pathと固定SHAへ束縛する。"""
    return {str((ROOT / (name.replace(".", "/") + ".py")).resolve()): digest
            for name, digest in EXTERNAL_SOURCE_SHA.items()}


def runtime_modules(guards: dict[str, str], identities: dict[str, Any], modules: dict[str, Any]) -> dict[str, str]:
    """全srcを検査し、追加2型以外のsnapshot条件は旧validatorと同じ。"""
    fixed = external_guards()
    if set(identities) != set(EXTERNAL_SOURCE_SHA) or any(guards.get(p) != h for p, h in fixed.items()):
        raise ValueError("explicit_source_prepare_guard_missing")
    result = {}
    for name, module in tuple(modules.items()):
        if name != "src" and not name.startswith("src."):
            continue
        path = Path(module.__file__).resolve()
        if name in identities:
            expected = (ROOT / (name.replace(".", "/") + ".py")).resolve()
            if module is not identities[name] or module.__name__ != name or path != expected:
                raise ValueError("explicit_source_module_identity_mismatch")
            if base.sha256(path) != fixed[str(expected)]:
                raise ValueError("explicit_source_byte_sha_mismatch")
        elif not path.is_relative_to(base.SNAPSHOT.resolve()):
            raise RuntimeError(f"非凍結srcが混入しています: {name}: {path}")
        result[name] = str(path)
    if not set(identities) <= result.keys():
        raise ValueError("explicit_source_module_missing")
    return result


def install_runtime_source_guard(stack: Any, receipt: Any, rec: Any) -> dict[str, Any]:
    """旧関数は編集せず、この診断の動的2型だけを明示した全src検査へ配線する。"""
    history = A.entry.previous.history
    original = history.frozen_modules
    source = str(Path(original.__code__.co_filename).resolve())
    if base.sha256(Path(source)) != receipt["input_and_code_sha256"].get(source):
        raise ValueError("original_source_validator_guard_mismatch")
    identities = {"src.chain_commit_candidate_v1": rec.commit_module,
                  "src.chain_prediction_ledger_v1": rec.ledger_module}
    report = {"fixed_additional_sources": external_guards(), "inspection_count": 0,
              "original_validator_path": source, "unknown_src_allowed": False}
    def checked() -> dict[str, str]:
        result = runtime_modules(receipt["input_and_code_sha256"], identities, sys.modules)
        report["inspection_count"] += 1
        report["last_module_count"] = len(result)
        return result
    base.patch(stack, history, "frozen_modules", checked)
    checked()
    return report


def instrument(stack: Any, collector: Any, history: Any, receipt: Any, state: dict[str, Any]) -> None:
    stream = (state["output"] / LEDGER).open("x", encoding="utf-8")
    rec = pending.PendingCommitRecorder(stream, {})
    state.update(pending_rec=rec, ledger_stream=stream)
    hooks.publication_after_next(stack, rec)
    hooks.insert_before_grace(stack, collector, rec, state)
    A_INSTRUMENT(stack, collector, history, receipt, state)
    tracker = hooks.ledger_hooks(stack, collector, rec)
    state["runtime_source_guard"] = install_runtime_source_guard(stack, receipt, rec)
    pending.install_pending_verifier(stack, collector.RecognitionPipeline, rec)
    sink = Sink(state["output"] / SIDECAR, history, rec, tracker)
    state.update(tracker=tracker, evidence_sink=sink, history=history)
    stack.callback(close, state)
    state["evidence_trace"] = install_trace(stack, collector, history, sink, state["pending_code"])
    install_public(stack, collector, history, sink)


A_INSTRUMENT: Any = None


def bind(latest: Any) -> None:
    global A_INSTRUMENT
    initialize(latest)
    A_INSTRUMENT = latest.instrument
