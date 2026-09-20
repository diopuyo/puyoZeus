"""旧NEXT/entry engineへ限定grace更新を接続し、全差分を保持する。"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import functools
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

from scripts import diagnose_video38_confirmed_collapse_v1 as base

HOME = Path(__file__).resolve().parent
ROOT = HOME.parents[2]
ENTRY_ROOT = HOME.parent / "g2_current_entry_observer_2026-09-08_v1/adapter"
SHADOW_ROOT = HOME.parent / "g2_resolved_grace_guard_shadow_2026-09-08_v1"
ENTRY_SHAS = ("99422d24bc74f8a39fb41991735de755c577bacc81ec110128da85a262c1bdd6",
              "72a62881b0c4c2a7f6aeb62c1e6030150c065a9a2fd49d39dcb13c3d530225ff",
              "8cc1008885bfca73787676940d84d5729d15456990f8aaca3e7f49ca4a0de0dd")
SHADOW_SHAS = {"shadow.py": "f20028f16c980825ee435ff57df649d1527b740c04861de0e621d62011af7561",
    "test_shadow.py": "66c4c820c80672c87d25dd31b27b442b35ca7793507bf342db5249164c34ca19",
    "ASSET_PREFLIGHT.md": "1d1378b9c9624ac23a9ab620f11f1a90570e5e65991b5a5a12ebab4e89a78a9f",
    "run_cpu.py": "7035092afc6c8a4cd59c7a1427d5f902f101676d64298675695a1cdf1f219760"}
REFERENCE = HOME.parent / "video38_current_entry_live_observation_2026-09-08_v1"
REFERENCE_SHA = "58ef09ac8b7f89060b3715ddf6d72697b00afe2a1a6504c21241a44e370ad48c"
FORMAT = "video38-resolved-grace-guard-shadow/v1"
SIDECAR, REPORT = "resolved_grace_guard.jsonl", "RESOLVED_GRACE_RECEIPT.json"
DIFF, STATUS = "RESOLVED_GRACE_COMPARISON.json", "RESOLVED_GRACE_STATUS.json"
WINDOWS = {"1P": (34340, 34370), "2P": (35838, 35854)}
PREFIX = "resolved_grace_"


def load(name: str, path: Path) -> Any:
    """固定絶対path専用aliasだけを読み、想定外既loadを拒否する。"""
    if name in sys.modules:
        raise RuntimeError("unexpected_loaded_alias")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
        return value
    except BaseException:
        sys.modules.pop(name, None)
        raise


base.assert_unchanged({str(ENTRY_ROOT / "adapter.py"): ENTRY_SHAS[0]})
entry = load("_resolved_grace_old_entry", ENTRY_ROOT / "adapter.py")
OLD_PREPARE, OLD_INSTRUMENT = entry.prepare, entry.instrument
OLD_ADDITIONAL, OLD_FORMAT = entry.additional_guards, entry.FORMAT
OLD_RUN, OLD_FINALIZE = entry.run, entry.finalize
EXTRAS = {SIDECAR, REPORT, DIFF, STATUS}


def targeted(frame: int, side: str) -> bool:
    """評価後に窓を増やさず、指定sideだけに介入する。"""
    return side in WINDOWS and WINDOWS[side][0] <= frame <= WINDOWS[side][1]


def expected() -> list[tuple[int, str]]:
    return [(frame, side) for frame, side in entry.expected_scopes() if targeted(frame, side)]


def additional_guards(args: Any) -> dict[str, str]:
    """旧3SHA/部品4SHA/新3SHA/同NEXT対照を別pathで結ぶ。"""
    old = argparse.Namespace(**vars(args))
    for name, sha in zip(("script", "test", "launcher"), ENTRY_SHAS, strict=True):
        setattr(old, name + "_sha256", sha)
    hashes = OLD_ADDITIONAL(old)
    hashes.update({str(SHADOW_ROOT / name): sha for name, sha in SHADOW_SHAS.items()})
    for name, file in (("script", "adapter.py"), ("test", "test_adapter.py"), ("launcher", "launcher.sh")):
        sha = getattr(args, name + "_sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError("new_sha_required")
        hashes[str(HOME / file)] = sha
    hashes[str(HOME / "ASSET_PREFLIGHT.md")] = base.sha256(HOME / "ASSET_PREFLIGHT.md")
    hashes[str(REFERENCE / "COMPLETE")] = REFERENCE_SHA
    base.assert_unchanged(hashes)
    complete = base.read_json(REFERENCE / "COMPLETE")
    if (complete.get("format_version") != OLD_FORMAT or type(complete.get("child_exit_code")) is not int
            or complete["child_exit_code"] != 0 or len(complete["sha256"]) != 12):
        raise ValueError("reference_completion_contract")
    hashes.update({str(REFERENCE / name): sha for name, sha in complete["sha256"].items()})
    hashes[str(base.SNAPSHOT / "src/chain.py")] = "911c400dfa5c0881839a7ba20a688f16da6c195acc71adfc693efed63b23d57c"
    base.assert_unchanged(hashes)
    return hashes


def prepare(args: Any, new_args: Any) -> Any:
    """元prepareを一回だけ使い、変更範囲と品質未認証を明示する。"""
    receipt, config = OLD_PREPARE(args, new_args)
    receipt["resolved_grace_guard"] = {"windows_by_side": WINDOWS, "scope_count": len(expected()),
        "reference_root": str(REFERENCE), "shadow_sha256": SHADOW_SHAS["shadow.py"],
        "predicate": "actual_same_step_sm_return_and_positive_chain_full_grid_equal",
        "preserved": ["grace_deadlines", "original_two_board_copies", "in_grace_local"],
        "saved_sm_injection": False, "completion_time_certified": False,
        "quality_gate_clear": False, "publication_permission": False, "accounting_permission": False}
    return receipt, config


def snapshot(pipe: Any, side: str) -> dict[str, Any]:
    """同scope内の実値だけdetachし、欠測を推論で補完しない。"""
    ctx = getattr(pipe, "_sm_" + side.lower()).context
    grace = getattr(pipe, "_landing_grace_" + side.lower())
    return {"state": getattr(ctx.state, "value", ctx.state),
        "confirmed": base.board_value(ctx.confirmed_board), "pending": base.board_value(ctx.pending_board),
        "grace": None if grace is None else {"frame_deadline": grace[0],
            "board": base.board_value(grace[1]), "time_deadline": grace[2]},
        "accounting": entry.previous.history.accounting_snapshot(pipe, side)}


class Sink(entry.Sidecar):
    """既存sticky/epoch/flushを再利用した別stream。"""

    def emit(self, row: dict[str, Any]) -> None:
        try:
            frame, clock, side = row["frame_idx"], row["time_sec"], row["side"]
            if (type(frame) is not int or frame % entry.STRIDE or not targeted(frame, side)
                    or isinstance(clock, bool) or not isinstance(clock, (int, float))
                    or not math.isfinite(clock) or clock != frame / entry.FPS):
                raise ValueError("guard_observation_clock")
            value = {**row, "software_epoch": self.epoch(row), "physical_hand_certified": False,
                     "publication_permission": False, "accounting_permission": False}
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
            self.stream.write(encoded + "\n")
            self.rows.append(json.loads(encoded))
        except BaseException as error:
            self.failed(error, "resolved_grace_sink")
            raise

    def summary(self) -> dict[str, Any]:
        self.require_ok()
        entered, returned, active, sm = [], [], None, 0
        for row in self.rows:
            key, kind = (row["frame_idx"], row["side"]), row["kind"]
            if kind == PREFIX + "scope_enter":
                if active is not None:
                    raise ValueError("nested_guard_scope")
                active, sm = key, 0
                entered.append(key)
            elif active != key:
                raise ValueError("guard_row_outside_scope")
            if kind == PREFIX + "sm_return":
                sm += 1
            if kind == PREFIX + "scope_return":
                if sm != 1:
                    raise ValueError("missing_or_duplicate_sm_return")
                returned.append(key)
                active = None
        if entered != expected() or returned != entered or active is not None:
            raise ValueError("guard_scope_coverage")
        if any(row["software_epoch"] is None for row in self.rows):
            raise ValueError("software_epoch_missing")
        return {"scopes": len(entered), "rows": len(self.rows), "runtime": self.runtime,
                "mutation_count": sum(row["kind"] == PREFIX + "guard_mutation" for row in self.rows),
                "quality_gate_clear": False, "completion_time_certified": False}


class RecProxy:
    """guardの新mutationだけ別sinkへ送信し旧rec.emitを変更しない。"""

    def __init__(self, rec: Any, sink: Sink) -> None:
        self.rec, self.sink = rec, sink

    def __getattr__(self, name: str) -> Any:
        if name == "step_code":
            raise AttributeError(name)
        return getattr(self.rec, name)

    def emit(self, row: dict[str, Any]) -> None:
        self.sink.emit(row)


def scope_step(original: Any, modified: Any, guard: Any, sink: Sink) -> Any:
    """指定side窓だけに介入し、他stepは元関数へ透過する。"""
    @functools.wraps(original)
    def step(pipe: Any, side: str, frame_idx: int, time_sec: float, *args: Any, **kwargs: Any) -> Any:
        if not targeted(frame_idx, side):
            return original(pipe, side, frame_idx, time_sec, *args, **kwargs)
        def emit(kind: str, **fields: Any) -> None:
            sink.emit({"kind": PREFIX + kind, "side": side, "frame_idx": frame_idx, "time_sec": time_sec, **fields})
        try:
            emit("scope_enter", snapshot=snapshot(pipe, side))
            result = modified(pipe, side, frame_idx, time_sec, *args, **kwargs)
            guard.assert_clean()
            emit("scope_return", snapshot=snapshot(pipe, side), confirmed_return=base.board_value(result.confirmed_board))
            return result
        except BaseException as error:
            sink.failed(error, "guard_scope")
            raise
    return step


def observe_callbacks(stack: Any, transformed: Any, sink: Sink) -> None:
    """実AST hookを一回通し、同時刻SMとwriter前後だけを保存する。"""
    namespace = transformed.__globals__
    capture, advance = namespace["__resolved_capture"], namespace["__resolved_advance"]
    def emit(kind: str, pipe: Any, side: str, frame: int, time: float, **fields: Any) -> None:
        sink.emit({"kind": PREFIX + kind, "side": side, "frame_idx": frame, "time_sec": time, **fields})
    def captured(pipe: Any, side: str, frame: int, time: float, *args: Any) -> Any:
        result = capture(pipe, side, frame, time, *args)
        emit("sm_return", pipe, side, frame, time, snapshot=snapshot(pipe, side))
        return result
    def advanced(pipe: Any, side: str, frame: int, time: float, *args: Any) -> Any:
        before = snapshot(pipe, side)
        result = advance(pipe, side, frame, time, *args)
        emit("advance", pipe, side, frame, time, before=before, after=snapshot(pipe, side))
        return result
    for name, value, old in (("__resolved_capture", captured, capture), ("__resolved_advance", advanced, advance)):
        namespace[name] = value
        stack.callback(namespace.__setitem__, name, old)


def instrument(stack: Any, collector: Any, rec: Any, receipt: Any, state: dict[str, Any]) -> None:
    """最初のstep観測器の直前へ一回だけ差込し、旧全計装順序を保つ。"""
    base.assert_unchanged(receipt["input_and_code_sha256"])
    sink = Sink(state["output"] / SIDECAR, rec)
    state["guard_sink"] = sink
    stack.callback(close_sink, state)
    module = load("_resolved_grace_runtime_shadow", SHADOW_ROOT / "shadow.py")
    stack.callback(sys.modules.pop, "_resolved_grace_runtime_shadow", None)
    observed = entry.previous.runner.observed
    original_end, count = observed.install_end, []
    def before_end(inner: Any, actual: Any, actual_rec: Any) -> None:
        if count or actual is not collector or actual_rec is not rec:
            raise RuntimeError("guard_install_identity")
        cls, original = collector.RecognitionPipeline, collector.RecognitionPipeline._step_side
        guard = module.install(inner, collector, RecProxy(rec, sink))
        state["guard"] = guard
        modified = cls._step_side
        observe_callbacks(inner, modified.__wrapped__, sink)
        base.patch(inner, cls, "_step_side", scope_step(original, modified, guard, sink))
        base.patch(inner, rec, "step_code", modified.__wrapped__.__code__)
        count.append(True)
        sink.runtime = {**guard.receipt, "saved_sm_injection": False, "install_before_end_observer": True}
        original_end(inner, actual, actual_rec)
    with contextlib.ExitStack() as setup:
        base.patch(setup, observed, "install_end", before_end)
        OLD_INSTRUMENT(stack, collector, rec, receipt, state)
    if len(count) != 1:
        raise RuntimeError("guard_not_installed")


def save_status(state: dict[str, Any]) -> None:
    """stack終了後のstickyをengineの入力として保存する。"""
    sink, guard = state.get("guard_sink"), state.get("guard")
    if sink is not None:
        base.write_json(state["output"] / STATUS, {"sticky_error": sink.error,
            "guard_error": None if guard is None else guard.error, "closed": sink.closed,
            "guard_installed": guard is not None, "rows": len(sink.rows)})


def close_sink(state: dict[str, Any]) -> None:
    """finishより前にflush/状態保存し、失敗しても元観測を保持する。"""
    try:
        state["guard_sink"].close()
    finally:
        save_status(state)


def indexed(path: Path) -> dict[str, Any]:
    """旧/新全行を捨てずkind/frame/side/出現番号と原文で保存する。"""
    values, counts = {}, Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identity = (row["kind"], row.get("frame_idx"), row.get("side"))
            key = json.dumps((*identity, counts[identity]), ensure_ascii=False)
            counts[identity] += 1
            values[key] = {"row": row, "text": line.rstrip("\r\n")}
    return values


def compare(reference: Path, candidate: Path, mutations: list[dict[str, Any]]) -> dict[str, Any]:
    """意図差も全保存。prefix以外を無条件合格とは呼ばない。"""
    hashes = {str(path): base.sha256(path) for path in (reference, candidate)}
    old, new = indexed(reference), indexed(candidate)
    first = min((row["frame_idx"] for row in mutations), default=math.inf)
    prefix = lambda values: [(key, value["text"]) for key, value in values.items()
                            if (value["row"].get("frame_idx") or -1) < first]
    differences = {key: {"before": old[key], "after": value} for key, value in new.items()
                   if key in old and old[key]["text"] != value["text"]}
    output = {"input_sha256": hashes, "reference_rows": len(old), "candidate_rows": len(new),
        "pre_mutation_prefix_bit_exact": prefix(old) == prefix(new),
        "all_text_and_order_equal": list(old.items()) == list(new.items()),
        "changed_rows": differences, "missing_rows": {key: value for key, value in old.items() if key not in new},
        "added_rows": {key: value for key, value in new.items() if key not in old},
        "reference_key_order": list(old), "candidate_key_order": list(new),
        "first_mutation_frame": None if first == math.inf else first, "quality_gate_clear": False}
    base.assert_unchanged(hashes)
    return output


def invariants(reference: Path, candidate: Path, mutations: list[dict[str, Any]]) -> dict[str, Any]:
    """raw/NEXT/会計/side別の差を分離し、実影響を後付けで許容しない。"""
    old, new = indexed(reference), indexed(candidate)
    def projection(values: Any, category: str) -> Any:
        result = []
        for key, item in values.items():
            row, kind = item["row"], item["row"]["kind"]
            if category == "raw" and kind == "frame_side":
                capture = row.get("accounting_capture")
                result.append((key, row.get("raw_captured_this_frame"), None if capture is None else
                               {name: capture.get(name) for name in ("captured_frame", "raw")}))
            elif category == "next" and kind == "live_accounting_invocation":
                result.append((key, row["raw"]))
            elif category == "accounting" and (kind == "accounting_update" or kind == "live_accounting_accounting"):
                result.append((key, item["text"]))
            elif category == "2P" and row.get("side") in ("2P", "p2"):
                result.append((key, item["text"]))
        return result
    report = {}
    for category in ("raw", "next", "accounting", "2P"):
        left, right = projection(old, category), projection(new, category)
        report[category] = {"equal": left == right, "reference_rows": len(left), "candidate_rows": len(right)}
    for side in WINDOWS:
        first = min((row["frame_idx"] for row in mutations if row["side"] == side), default=math.inf)
        subset = lambda values: [(key, item["text"]) for key, item in values.items()
            if item["row"].get("side") == side and (item["row"].get("frame_idx") or -1) < first]
        report[side + "_prefix_equal"] = subset(old) == subset(new)
    return report


def target_rows(path: Path) -> list[dict[str, Any]]:
    """指定改善/正常対照の全盤面を保存し、色数だけで合格にしない。"""
    targets = {(34344, "1P"), (34360, "1P"), (35838, "2P"), (35846, "2P")}
    return [item["row"] for item in indexed(path).values()
            if item["row"]["kind"] == "frame_side" and
            (item["row"].get("frame_idx"), item["row"].get("side")) in targets]


def finish(output: Path, receipt: Any, rec: Any, elapsed: float, state: dict[str, Any]) -> Any:
    """旧NEXTの完走/全5artifactを再利用し、新entry比較を別条件で結ぶ。"""
    sink, guard = state["guard_sink"], state["guard"]
    guard.assert_clean()
    report = sink.summary()
    report["entry_observation"] = state["sink"].summary()
    mutations = [row for row in sink.rows if row["kind"] == PREFIX + "guard_mutation"]
    comparisons = {"frames": compare(REFERENCE / "frames.jsonl", output / "frames.jsonl", mutations),
                  "entry": compare(REFERENCE / entry.SIDECAR, output / entry.SIDECAR, mutations)}
    comparisons["invariants"] = invariants(REFERENCE / "frames.jsonl", output / "frames.jsonl", mutations)
    comparisons["target_before"] = target_rows(REFERENCE / "frames.jsonl")
    comparisons["target_after"] = target_rows(output / "frames.jsonl")
    base.write_json(output / DIFF, comparisons)
    base.write_json(output / entry.COMPARISON, comparisons["frames"])
    base.write_json(output / entry.REPORT, report["entry_observation"])
    base.write_json(output / REPORT, report)
    if not all(comparisons[key]["pre_mutation_prefix_bit_exact"] for key in ("frames", "entry")):
        raise ValueError("unexplained_pre_mutation_difference")
    if not comparisons["invariants"]["raw"]["equal"]:
        raise ValueError("raw_input_changed")
    result = entry.ORIGINAL_FINISH(output, receipt, rec, elapsed)
    engine = base.read_json(output / "ENGINE_COMPLETE")
    if engine.get("format_version") != entry.previous.FORMAT or set(engine["sha256"]) != entry.previous.ENGINE_NAMES:
        raise ValueError("legacy_engine_contract")
    hashes = {**engine["sha256"], **{name: base.sha256(output / name)
              for name in entry.ENGINE_NAMES - entry.previous.ENGINE_NAMES}}
    base.assert_unchanged(receipt["input_and_code_sha256"])
    base.assert_unchanged({str(output / name): sha for name, sha in hashes.items()})
    guard.assert_clean()
    base.write_json(output / entry.ENGINE, {"format_version": FORMAT, "sha256": hashes, "awaiting_actual_child_exit": True})
    return {**result, "guard_mutations": len(mutations), "quality_gate_clear": False}


@contextlib.contextmanager
def configured() -> Any:
    """元runnerのrun/終了を再利用し、変更したmodule属性を復元する。"""
    with contextlib.ExitStack() as stack:
        values = {"FORMAT": FORMAT, "additional_guards": additional_guards,
            "prepare": prepare, "instrument": instrument, "finish": finish,
            "ENGINE_NAMES": entry.ENGINE_NAMES | EXTRAS}
        for name, value in values.items():
            base.patch(stack, entry, name, value)
        yield


def run(args: Any) -> Any:
    """run私有stateをfinish後まで保持し、guardエラーを最終statusへ結ぶ。"""
    original, state = instrument, {}
    def capture(stack: Any, collector: Any, rec: Any, receipt: Any, inner: dict[str, Any]) -> None:
        state["value"] = inner
        original(stack, collector, rec, receipt, inner)
    try:
        with configured(), contextlib.ExitStack() as stack:
            base.patch(stack, entry, "instrument", capture)
            return OLD_RUN(args)
    finally:
        inner = state.get("value", {})
        if inner.get("guard_sink") is not None and not (inner["output"] / STATUS).exists():
            save_status(inner)


def finalize(output: Path, child_exit: int) -> Any:
    """数値exit/全SHA/guardを旧finalizerで照合し、観測のみという旧statusを訂正する。"""
    if type(child_exit) is not int or not 0 <= child_exit <= 255:
        raise ValueError("actual_child_exit_required")
    if child_exit == 0:
        status = base.read_json(output / STATUS)
        if (status.get("sticky_error") is not None or status.get("guard_error") is not None
                or status.get("closed") is not True or status.get("guard_installed") is not True):
            raise ValueError("guard_failed_or_missing")
    writer = base.write_json
    def write(path: Path, value: Any) -> None:
        if path == output / "COMPLETE":
            value["status"] = "shadow_complete_requires_independent_quality_review"
            value["completion_time_certified"] = False
        writer(path, value)
    with configured(), contextlib.ExitStack() as stack:
        base.patch(stack, base, "write_json", write)
        return OLD_FINALIZE(output, child_exit)


def main() -> int:
    """既存CLI設定は保ち、run私有statusと新finalizeだけへ委譲する。"""
    with configured(), contextlib.ExitStack() as stack:
        base.patch(stack, entry, "run", run)
        base.patch(stack, entry, "finalize", finalize)
        return entry.main()


if __name__ == "__main__":
    raise SystemExit(main())
