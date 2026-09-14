"""固定開始epochを背景に、NEXT会計だけを同一履歴で比較する開発診断。"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts import diagnose_video38_initial_pair_observation_v1 as initial
from scripts import next_enqueue_live_shadow_v1 as live

runner, base, history = initial.runner, initial.base, initial.history
FORMAT = "video38-next-enqueue-live-shadow/v1"
PREFIX = live.PREFIX
KINDS = frozenset(PREFIX + name for name in ("decision", "accounting", "invocation"))
TEST = base.ROOT / "tests/test_diagnose_video38_next_enqueue_live_shadow_v1.py"
LAUNCHER = base.ROOT / "scripts/launch_video38_next_enqueue_live_shadow_v1.sh"
PLAN = base.ROOT / "docs/agent_coordination/NEXT_LIVE_RUN_PLAN_2026-09-08.md"
PLAN_SHA = "3ebe8f4110fbc6528720dad50d80c9e423a79735e339143df4735a21f2cfa263"
LIVE_SHA = "e5ebff6827119c616319b4598fce1428c98643b621ff02e985736b43783d9237"
LIVE_TEST_SHA = "3aba1e71d7e915d5875d06489936917a5a0719ac881ed9d98d37fc0b85e1eb66"
INITIAL_SHA = "af19446390a9786d9684ffd7f3c5d13d62fb3ddb411fd0381ba3ed2a5cd068cb"
SUPPLEMENT = base.VERIFY / "video38_initial_pair_observation_2026-09-07_v2"
SUPPLEMENT_SHA = "c87ceadd5e07fd78284db101bb33c879165615e799cd991e8ef80b6bcef012e9"
SUPPLEMENT_QA_SHA = "6630a988da3dc632976405b58c72b5194f58078f8870a4d120a0208a12b68cff"
FIRST_DIFFERENCE_FRAME = 32698
OBS_FIRST, OBS_LAST = initial.FIRST_FRAME, initial.LAST_FRAME
EXPECTED_UPDATES = len(range(live.FIRST_FRAME, live.LAST_FRAME + 1, history.STRIDE))
COMPARE_NAME, EXTRA_NAME = "NEXT_LIVE_COMPARISON.json", "NEXT_LIVE_OBSERVATION.json"
ENGINE_NAMES = frozenset(("PLAN.json", "frames.jsonl", "SUMMARY.json", COMPARE_NAME, EXTRA_NAME))
NORMAL_WINDOWS = {"1P_cold": ["1P", 34080, 34380], "1P_landing": ["1P", 34700, 34792],
                  "2P_end_next": ["2P", 35704, 36298]}


def fixed_guards() -> dict[str, str]:
    """実行する固定依存だけを列挙し、別レーンのgraceに依存しない。"""
    result = {str((base.ROOT / name).resolve()): digest for name, digest in initial.FIXED_SHA.items()}
    result.update({str((base.SNAPSHOT / "src" / name).resolve()): digest
                   for name, digest in initial.FROZEN_SHA.items()})
    result.update(live.REQUIRED_INPUT_SHA256)
    result.update(live.cpu.REQUIRED_INPUT_SHA256)
    result.update({str(Path(live.__file__).resolve()): LIVE_SHA,
                   str(base.ROOT / "tests/test_next_enqueue_live_shadow_v1.py"): LIVE_TEST_SHA,
                   str(Path(initial.__file__).resolve()): INITIAL_SHA, str(PLAN): PLAN_SHA})
    return result


def supplement_guards() -> dict[str, str]:
    """旧追加観測は入力証拠としてのみ読み、独立推論は起動しない。"""
    result = {str(SUPPLEMENT / "COMPLETE"): SUPPLEMENT_SHA,
              str(SUPPLEMENT / "INDEPENDENT_RUN_QA.md"): SUPPLEMENT_QA_SHA}
    base.assert_unchanged(result)
    value = base.read_json(SUPPLEMENT / "COMPLETE")
    expected = {"PLAN.json", "frames.jsonl", "SUMMARY.json", runner.DIFFERENCE_NAME, initial.EXTRA_NAME}
    if set(value["sha256"]) != expected:
        raise ValueError("補助対照の5artifact契約が違います")
    result.update({str(SUPPLEMENT / name): digest for name, digest in value["sha256"].items()})
    base.assert_unchanged(result)
    return result


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """候補・対照・実行3fileを事前SHAへ結合して同じ履歴を使う。"""
    if (args.mode != "start_epoch" or args.module_sha256 != initial.FIXED_SHA["scripts/match_start_epoch_shadow_v1.py"]
            or args.module_test_sha256 != initial.FIXED_SHA["tests/test_match_start_epoch_shadow_v1.py"]):
        raise ValueError("背景start_epoch以外は混ぜません")
    guards = fixed_guards()
    for path, digest in zip((Path(__file__), TEST, LAUNCHER),
                            (args.script_sha256, args.test_sha256, args.launcher_sha256), strict=True):
        guards[str(path.resolve())] = digest
    base.assert_unchanged(guards)
    guards.update(initial.reference_guards())
    guards.update(supplement_guards())
    receipt, config = runner.prepare(args)
    receipt["input_and_code_sha256"].update(guards)
    runner.validate_dependencies(Path(live.__file__), receipt["input_and_code_sha256"])
    base.assert_unchanged(receipt["input_and_code_sha256"])
    receipt["next_enqueue_live"] = {"format_version": FORMAT, "module_sha256": LIVE_SHA,
        "reference_root": str(initial.REFERENCE), "supplement_root": str(SUPPLEMENT),
        "history_frame_range": [history.FIRST_FRAME, history.END_FRAME - history.STRIDE],
        "repair_frame_range": [live.FIRST_FRAME, live.LAST_FRAME], "expected_updates": EXPECTED_UPDATES,
        "raw_observation_frame_range": [OBS_FIRST, live.LAST_FRAME],
        "fixed_legacy_prefix_before_frame": FIRST_DIFFERENCE_FRAME, "normal_windows": NORMAL_WINDOWS,
        "extra_detector_calls": 0, "inactive_inference": False, "grace_repair": False,
        "same_color_placement_verified": False, "initial_yellow_pair_repaired": False,
        "accounting_basis_verified": False, "quality_gate_clear": False,
        "raw_reference_limit": "535–548秒だけ詳細return対照あり。以後は比較不能を補完しない"}
    return receipt, config


def selected(frame: int) -> bool:
    return live.FIRST_FRAME <= frame <= live.LAST_FRAME


def strict(value: Any) -> Any:
    """追加receiptの非有限値や未知objectをNone/文字列へ救済しない。"""
    json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value


def slide_value(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return strict({name: getattr(value, name) for name in ("slide_motion", "diff_score", "threshold_used")})


def observed_invocation(controller: Any, calls: dict[str, Any]) -> dict[str, Any]:
    """捕捉済み実returnだけを読む。追加検出/再度captureは行わない。"""
    inv = controller.active
    if inv is None or calls.get("invocation") is not inv:
        raise RuntimeError("実invocationの捕捉が欠けています")
    if calls.get("observer_error"):
        raise RuntimeError(calls["observer_error"])
    main, slides = calls["main"], calls["slides"]
    if main["count"] > 1 or any(value["count"] > 1 for value in slides.values()):
        raise RuntimeError("原本detector追加呼出を検出しました")
    if main["returned"] is not inv.main:
        raise RuntimeError("主NEXTの実returnと会計捕捉が不一致です")
    for side, value in slides.items():
        if value["returned"] is not inv.slides.get(side):
            raise RuntimeError("slideの実returnと会計捕捉が不一致です")
    return strict({"main": {"count": main["count"], "exception": main["exception"],
            "returned": None if inv.main is None else initial.pairs(inv.main)},
        "slides": {side: {"count": value["count"], "exception": value["exception"],
                    "returned": slide_value(value["returned"]), "after": value.get("after")}
                   for side, value in slides.items()}, "same_return_identity": True,
        "capture_error": inv.error, "extra_detector_calls": 0})


def empty_call() -> dict[str, Any]:
    return {"count": 0, "exception": None, "returned": None}


def install_actual_calls(stack: contextlib.ExitStack, controller: Any) -> dict[str, Any]:
    """元呼出を一回、例外も同型のまま通す。保存は認識catchの外側。"""
    from src.next_detector import NextDetector
    from src.next_slide_detector import NextSlideDetector
    state: dict[str, Any] = {}
    original_begin = controller.begin
    def begin(pipe: Any, frame: int, time_sec: float) -> None:
        original_begin(pipe, frame, time_sec)
        state.clear()
        state.update(invocation=controller.active, main=empty_call(),
                     slides={side: empty_call() for side in history.SIDES})
    def wrapped(original: Any, is_slide: bool) -> Any:
        def call(detector: Any, *args: Any, **kwargs: Any) -> Any:
            inv = controller.active
            if inv is None:
                return original(detector, *args, **kwargs)
            pipe, side = inv.runtime.pipe, None
            if is_slide:
                sides = [s for s in history.SIDES if detector is getattr(pipe, "_slide_detector_" + s.lower())]
                side = sides[0] if len(sides) == 1 else None
            valid = side is not None if is_slide else detector is pipe._next_detector
            if not valid:
                state["observer_error"] = "未知detectorの呼出を実NEXTへ混ぜません"
                return original(detector, *args, **kwargs)
            record = state["slides"][side] if is_slide else state["main"]
            record["count"] += 1
            try:
                result = original(detector, *args, **kwargs)
            except BaseException as exc:
                record["exception"] = type(exc).__name__
                raise
            record["returned"] = result
            return result
        return call
    base.patch(stack, controller, "begin", begin)
    base.patch(stack, NextDetector, "detect_both", wrapped(NextDetector.detect_both, False))
    base.patch(stack, NextSlideDetector, "update", wrapped(NextSlideDetector.update, True))
    return state


def install_observation(stack: contextlib.ExitStack, controller: Any, rec: Any) -> None:
    """enqueueの外側へ前後採録。commit内callback、Counter書戻しはない。"""
    calls = install_actual_calls(stack, controller)
    original_enqueue, original_end = controller.enqueue, controller.end
    def enqueue(pipe: Any, side: str, frame: int, time_sec: float, active: bool, pair: Any) -> None:
        if not selected(frame):
            return original_enqueue(pipe, side, frame, time_sec, active, pair)
        for name, value in calls["slides"].items():
            detector = getattr(pipe, "_slide_detector_" + name.lower())
            value["after"] = None if detector is None else initial.slide_snapshot(detector)
        raw = observed_invocation(controller, calls)
        before = history.accounting_snapshot(pipe, side)
        old = strict(asdict(controller.active.runtime.histories[side]))
        original_enqueue(pipe, side, frame, time_sec, active, pair)
        rec.emit({"kind": PREFIX + "accounting", "frame_idx": frame, "time_sec": time_sec,
            "side": side, "active": active, "raw": raw, "before": before,
            "after": history.accounting_snapshot(pipe, side), "history_before": old,
            "history_after": strict(asdict(controller.active.runtime.histories[side])),
            "counter_written_here": False, "physical_placement_verified": False})
    def end(success: bool) -> None:
        inv = controller.active
        try:
            if success and inv is not None and OBS_FIRST <= inv.frame <= live.LAST_FRAME:
                value = observed_invocation(controller, calls)
                rec.emit({"kind": PREFIX + "invocation", "frame_idx": inv.frame,
                          "time_sec": inv.time_sec, "raw": value, "production_permission": False})
        finally:
            original_end(success)
    base.patch(stack, controller, "enqueue", enqueue)
    base.patch(stack, controller, "end", end)


def detector_config(pipe: Any, *, require_cuda: bool = True) -> dict[str, Any]:
    """実factoryのdirect型と実parameter deviceを読む。推論/clone/toはしない。"""
    from src.next_detector import NextDetector
    from src.patch_classifier import CnnPatchClassifier
    if type(pipe._next_detector) is not NextDetector:
        raise RuntimeError("実NextDetectorが固定型と違います")
    classifier = pipe._next_detector._classifier
    if type(classifier) is not CnnPatchClassifier:
        raise RuntimeError("固定factoryのdirect CNN配置ではありません")
    device = str(next(classifier._model.parameters()).device)
    if require_cuda and device != "cuda:0":
        raise RuntimeError("実NEXT CNNがcuda:0ではありません")
    return {"detector_type": type(pipe._next_detector).__qualname__,
            "classifier_type": type(classifier).__qualname__, "classifier_layout": "direct_cnn",
            "parameter_device": device, "extra_inference": False,
            "slide_configuration": {side: initial.slide_snapshot(getattr(pipe, "_slide_detector_" + side.lower()))
                                    for side in history.SIDES}}


def instrument(stack: contextlib.ExitStack, collector: Any, rec: Any, receipt: dict[str, Any]) -> None:
    """NEXTはraw updateへ先設置し、その後に元history/start/endのclosureを作る。"""
    base.assert_unchanged(fixed_guards())
    controller = live.install(stack, collector, rec)
    install_observation(stack, controller, rec)
    runner.instrument(stack, collector, rec, receipt)
    cls, original = collector.RecognitionPipeline, inspect.getattr_static(collector.RecognitionPipeline, "load_default")
    def load(inner_cls: type, *args: Any, **kwargs: Any) -> Any:
        pipe = original.__func__(inner_cls, *args, **kwargs)
        rec.pipeline_receipt["next_enqueue_live"] = {"configuration": detector_config(pipe),
                                                    "transform": controller.transform_receipt}
        return pipe
    base.patch(stack, cls, "load_default", classmethod(load))
    rec.next_enqueue_controller = controller


def read_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """背景start_epoch行を残し、新NEXT prefixだけ別母数にする。"""
    result, extra, occurrences = {}, [], Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"].startswith(PREFIX):
                if row["kind"] not in KINDS:
                    raise ValueError("未知NEXT行です")
                extra.append(row)
                continue
            if row["kind"].startswith(initial.PREFIX):
                raise ValueError("外した初手追加観測が混入しました")
            identity = row["kind"], row["frame_idx"], row.get("side")
            key = json.dumps((*identity, occurrences[identity]), ensure_ascii=False)
            occurrences[identity] += 1
            result[key] = {"row": row, "text": line.rstrip("\r\n")}
    return result, extra


def compare_rows(reference: Path, candidate: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """先に固定した32698前は完全一致。以後は全差分を保存する。"""
    guards = {str(path): base.sha256(path) for path in (reference, candidate)}
    old, background_extra = read_rows(reference)
    new, extra = read_rows(candidate)
    if background_extra or len(old) != initial.EXPECTED_LEGACY_ROWS:
        raise ValueError("主対照15722行契約が違います")
    def prefix(values: dict[str, Any]) -> list[Any]:
        return [(key, value["text"]) for key, value in values.items()
                if (value["row"]["frame_idx"] if value["row"]["frame_idx"] is not None else -1) < FIRST_DIFFERENCE_FRAME]
    if prefix(old) != prefix(new):
        raise ValueError("事前固定prefixに差分があります")
    changes = {key: runner.changed_fields(old[key]["row"], value["row"]) for key, value in new.items()
               if key in old and old[key]["text"] != value["text"]}
    result = {"reference_rows": len(old), "candidate_rows": len(new),
        "fixed_prefix_before_frame": FIRST_DIFFERENCE_FRAME, "prefix_bit_exact": True,
        "all_legacy_bit_exact": list(old.items()) == list(new.items()),
        "changed_fields": changes, "missing_rows": {key: value for key, value in old.items() if key not in new},
        "added_rows": {key: value for key, value in new.items() if key not in old},
        "reference_key_order": list(old), "candidate_key_order": list(new),
        "coverage": runner.coverage(new, "start_epoch"), "input_sha256": guards,
        "normal_windows": NORMAL_WINDOWS, "quality_gate_clear": False}
    base.assert_unchanged(guards)
    return result, extra


def observation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """全1903 updateと左右受理を要求し、欠測returnを静止で埋めない。"""
    expected = list(range(live.FIRST_FRAME, live.LAST_FRAME + 1, history.STRIDE))
    observed = list(range(OBS_FIRST, live.LAST_FRAME + 1, history.STRIDE))
    by_kind = {kind: [row for row in rows if row["kind"] == kind] for kind in KINDS}
    for row in rows:
        frame = row.get("frame_idx")
        if type(frame) is not int or frame not in observed or not runner.valid_time(row.get("time_sec"), frame):
            raise ValueError("追加観測clock/窓が不正です")
    invocation = by_kind[PREFIX + "invocation"]
    if [row["frame_idx"] for row in invocation] != observed:
        raise ValueError("invocation coverageに欠落/重複があります")
    for kind in (PREFIX + "accounting", PREFIX + "decision"):
        if [(row["frame_idx"], row["side"]) for row in by_kind[kind]] != [
                (frame, side) for frame in expected for side in history.SIDES]:
            raise ValueError("会計side coverageに欠落/重複があります")
    totals = Counter()
    for row in invocation:
        raw = row["raw"]
        if raw["same_return_identity"] is not True or raw["extra_detector_calls"] != 0:
            raise ValueError("実return/追加呼出契約が違います")
        for name, value in {"main": raw["main"], **raw["slides"]}.items():
            if type(value["count"]) is not int or value["count"] not in (0, 1):
                raise ValueError("実call countが不正です")
            totals[name + "_calls"] += value["count"]
            totals[name + "_exceptions"] += value["exception"] is not None
    return {"counts": {kind: len(value) for kind, value in by_kind.items()}, "actual_calls": dict(totals),
            "initial_yellow_pair_repaired": False, "same_color_placement_verified": False,
            "accounting_basis_verified": False, "quality_gate_clear": False}


def raw_comparison(rows: list[dict[str, Any]], reference: Path) -> dict[str, Any]:
    """旧535–548秒の既存returnだけ比較し、後半の旧未観測を補完しない。"""
    old = {}
    with reference.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] in (initial.PREFIX + "main_next", initial.PREFIX + "main_slide"):
                key = (row["frame_idx"], row.get("side"))
                if key in old:
                    raise ValueError("補助実callが重複しています")
                value = row["returned"]
                if row.get("side") is not None and value is not None:
                    value = {name: value[name] for name in ("slide_motion", "diff_score", "threshold_used")}
                old[key] = {"returned": value, "exception": row["exception"]}
    new = {}
    for row in rows:
        if row["kind"] != PREFIX + "invocation" or not OBS_FIRST <= row["frame_idx"] <= OBS_LAST:
            continue
        for side, value in {None: row["raw"]["main"], **row["raw"]["slides"]}.items():
            if value["count"]:
                new[(row["frame_idx"], side)] = {"returned": value["returned"], "exception": value["exception"]}
    encode = lambda values: {json.dumps(key): value for key, value in values.items()}
    return {"reference_calls": len(old), "candidate_calls": len(new), "all_common_returns_equal": old == new,
        "changed": encode({key: {"before": old[key], "after": value} for key, value in new.items()
                           if key in old and old[key] != value}),
        "missing": encode({key: value for key, value in old.items() if key not in new}),
        "added": encode({key: value for key, value in new.items() if key not in old}),
        "compared_frame_range": [OBS_FIRST, OBS_LAST], "later_reference_raw_unobserved": True}


def finish(output: Path, receipt: dict[str, Any], rec: Any, elapsed: float) -> dict[str, Any]:
    """旧finishを再利用し、実child exitまで最終COMPLETEを保留する。"""
    comparison, rows = compare_rows(initial.REFERENCE / "frames.jsonl", output / "frames.jsonl")
    extra = observation_summary(rows)
    extra["raw_comparison"] = raw_comparison(rows, SUPPLEMENT / "frames.jsonl")
    writer, deferred = base.write_json, []
    def defer(path: Path, value: Any) -> None:
        if path == output / "COMPLETE":
            deferred.append(value)
        else:
            writer(path, value)
    with contextlib.ExitStack() as stack:
        base.patch(stack, base, "write_json", defer)
        summary = runner.ORIGINAL_HISTORY_FINISH(output, receipt, rec, elapsed)
    if len(deferred) != 1:
        raise RuntimeError("元finishのCOMPLETEが一意ではありません")
    writer(output / COMPARE_NAME, comparison)
    writer(output / EXTRA_NAME, extra)
    hashes = {**deferred[0]["sha256"], COMPARE_NAME: base.sha256(output / COMPARE_NAME),
              EXTRA_NAME: base.sha256(output / EXTRA_NAME)}
    if set(hashes) != ENGINE_NAMES:
        raise RuntimeError("ENGINEの5artifact契約が違います")
    base.assert_unchanged(comparison["input_sha256"])
    base.assert_unchanged(receipt["input_and_code_sha256"])
    base.assert_unchanged({str(output / name): digest for name, digest in hashes.items()})
    writer(output / "ENGINE_COMPLETE", {"format_version": FORMAT, "sha256": hashes,
                                       "status": "awaiting_real_child_exit", "quality_gate_clear": False})
    return {"frame_count": summary["frame_count"], "elapsed_sec": elapsed,
            "status": "awaiting_real_child_exit", "quality_gate_clear": False}


def finalize(output: Path, child_exit: int) -> dict[str, Any]:
    """shellが実際に待ったchild exitを保存し、0＋全SHAだけ最終完了へ結ぶ。"""
    if type(child_exit) is not int or not 0 <= child_exit <= 255:
        raise ValueError("実shell child exit codeが必要です")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "COMPLETE").exists() or (output / "CHILD_EXIT.json").exists():
        raise FileExistsError("終了receiptを上書きしません")
    base.write_json(output / "CHILD_EXIT.json", {"child_exit_code": child_exit,
        "source": "launcher_waited_actual_process", "pid_disappearance_is_not_exit_zero": True})
    if child_exit != 0:
        return {"status": "child_failed", "child_exit_code": child_exit}
    engine = base.read_json(output / "ENGINE_COMPLETE")
    if engine.get("format_version") != FORMAT or set(engine["sha256"]) != ENGINE_NAMES:
        raise ValueError("別engineまたは欠落artifactです")
    receipt = base.read_json(output / "PLAN.json")
    hashes = dict(engine["sha256"])
    hashes.update({name: base.sha256(output / name) for name in ("ENGINE_COMPLETE", "CHILD_EXIT.json")})
    base.assert_unchanged({str(output / name): digest for name, digest in hashes.items()})
    base.assert_unchanged(receipt["input_and_code_sha256"])
    complete = {"format_version": FORMAT, "status": "experiment_complete_not_adopted",
                "sha256": hashes, "child_exit_code": 0, "quality_gate_clear": False}
    base.write_json(output / "COMPLETE", complete)
    return complete


def run(args: argparse.Namespace) -> dict[str, Any]:
    """原本historyだけを一時配線し、例外時もmodule属性を復元する。"""
    state: dict[str, Any] = {}
    def prepared(value: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
        result = prepare(value)
        state["receipt"] = result[0]
        return result
    def installed(stack: contextlib.ExitStack, collector: Any, rec: Any) -> None:
        instrument(stack, collector, rec, state["receipt"])
    with contextlib.ExitStack() as stack:
        for name, function in (("prepare", prepared), ("instrument_pipeline", installed), ("finish", finish)):
            base.patch(stack, history, name, function)
        return history.run(args)


def main() -> int:
    """親の独立QA後のみ起動。実child終了確認はlauncherが別processで呼ぶ。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--finalize-child-exit", type=int)
    for name in ("script", "test", "launcher"):
        parser.add_argument(f"--{name}-sha256")
    parser.add_argument("--allow-native-runtime-mismatch", action="store_true")
    args = parser.parse_args()
    if args.finalize_child_exit is not None:
        print(json.dumps(finalize(args.output_root.resolve(), args.finalize_child_exit), ensure_ascii=False))
        return 0
    if any(not getattr(args, name + "_sha256") for name in ("script", "test", "launcher")):
        parser.error("実行には検収済み3SHAが必要です")
    import cv2
    cv2.setNumThreads(history.THREADS)
    runner.pending.completion._configure_torch_threads()
    args.mode, (args.start_sec, args.end_sec) = "start_epoch", runner.INTERVALS["start_epoch"]
    args.module_sha256 = initial.FIXED_SHA["scripts/match_start_epoch_shadow_v1.py"]
    args.module_test_sha256 = initial.FIXED_SHA["tests/test_match_start_epoch_shadow_v1.py"]
    print(json.dumps(run(args), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
