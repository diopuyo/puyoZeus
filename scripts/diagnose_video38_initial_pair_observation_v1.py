"""固定開始epoch runを変更せず、欠けていた初手NEXT観測だけを追加する。"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import inspect
import json
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from scripts import diagnose_video38_boundary_repair_shadow_v1 as runner


base, history = runner.base, runner.history
FORMAT = "video38-initial-pair-observation/v1"
PREFIX = "initial_pair_"
KINDS = frozenset(PREFIX + name for name in ("result", "main_next", "main_slide", "inactive_next"))
FIRST_FRAME, LAST_FRAME, EXPECTED_UPDATES = 32100, 32880, 391
EXPECTED_LEGACY_ROWS = 15722
EXTRA_NAME = "INITIAL_PAIR_OBSERVATION_SUMMARY.json"
REFERENCE = base.VERIFY / "video38_start_epoch_shadow_2026-09-07_v1"
REFERENCE_COMPLETE_SHA = "0dee7d346a0231ed3f95347d549b1a1e5436dc512ce405e5f597db6587b0c326"
TEST = base.ROOT / "tests/test_diagnose_video38_initial_pair_observation_v1.py"
LAUNCHER = base.ROOT / "scripts/launch_video38_initial_pair_observation_v1.sh"
FIXED_SHA = {
    "scripts/diagnose_video38_boundary_repair_shadow_v1.py": "e65d368be8a68074184450811a9a112f8517b945d424530f47ebab41217ff453",
    "tests/test_diagnose_video38_boundary_repair_shadow_v1.py": "dd978ceff4ec1eb5b0f67b8966eb0d45a1f0db0d708968f8ce6bd30da6715d35",
    "scripts/launch_video38_boundary_repair_shadow_v1.sh": "7c4851027202873d58b6d087cdcfcdba082a4a14a7995ebc431f9bb6f0665806",
    "scripts/match_start_epoch_shadow_v1.py": "d9fbb9f2de11cd48535c64bce51a0d282100aeefcc067632316a51f87cd6fe70",
    "tests/test_match_start_epoch_shadow_v1.py": "817bb95430edfca28bed93496e364aafd4fc57770357c4bcc67efa3c496d8914",
}
FROZEN_SHA = {
    "recognition_pipeline.py": "6e945d7584025ae9803e14c9ac079b1a468f483d0beac681a1f0e580cdd10e02",
    "next_detector.py": "6f57af41704c8438ce172ea5dacc3c34d804014d0c88665c0860dc73d5ffb434",
    "next_slide_detector.py": "ce7fcb447429578b6b591a455fe900f99ae55b845ce7f7b3d2e98cde2270ad05",
    "patch_classifier.py": "ccadc1af68a9d99c8ceeb660d583b2ec11ea4dbaafbbedacf3656fa422dce3b2",
}
ORIGINAL_PREPARE, ORIGINAL_INSTRUMENT, ORIGINAL_FINISH = runner.prepare, runner.instrument, runner.finish


def reference_guards() -> dict[str, str]:
    """開始修復の完成済み4artifactを対照として固定する。"""
    guards = {str(REFERENCE / "COMPLETE"): REFERENCE_COMPLETE_SHA}
    base.assert_unchanged(guards)
    value = base.read_json(REFERENCE / "COMPLETE")
    if set(value["sha256"]) != {"PLAN.json", "frames.jsonl", "SUMMARY.json", runner.DIFFERENCE_NAME}:
        raise ValueError("開始epoch対照の4artifact契約が違います")
    guards.update({str(REFERENCE / name): digest for name, digest in value["sha256"].items()})
    base.assert_unchanged(guards)
    return guards


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """固定依存と新規3fileをguardし、元run条件を一切変更しない。"""
    if args.mode != "start_epoch" or args.module_sha256 != FIXED_SHA["scripts/match_start_epoch_shadow_v1.py"]:
        raise ValueError("固定start_epochだけが許可されています")
    if args.module_test_sha256 != FIXED_SHA["tests/test_match_start_epoch_shadow_v1.py"]:
        raise ValueError("開始修復testの固定SHAが違います")
    guards = {str((base.ROOT / name).resolve()): digest for name, digest in FIXED_SHA.items()}
    guards.update({str((base.SNAPSHOT / "src" / name).resolve()): digest for name, digest in FROZEN_SHA.items()})
    for path, digest in zip((Path(__file__), TEST, LAUNCHER),
                            (args.script_sha256, args.test_sha256, args.launcher_sha256), strict=True):
        guards[str(path.resolve())] = digest
    base.assert_unchanged(guards)
    guards.update(reference_guards())
    receipt, config = ORIGINAL_PREPARE(args)
    receipt["input_and_code_sha256"].update(guards)
    base.assert_unchanged(receipt["input_and_code_sha256"])
    receipt["initial_pair_observation"] = {
        "format_version": FORMAT, "frame_range_inclusive": [FIRST_FRAME, LAST_FRAME],
        "expected_updates": EXPECTED_UPDATES, "reference_root": str(REFERENCE),
        "expected_legacy_rows": EXPECTED_LEGACY_ROWS, "main_policy_unchanged": True,
        "extra_calls": "inactiveだけ独立NEXT.detect_bothを1回。両side計8ROI、CNN実呼出数/秒数を保存",
        "independent_state": "NEXT・分類器wrapper・gate/UI/centroidは別object、eval重みのみ読取共有",
        "comparison": "新prefixだけ除外。既存修復行も含め全文字列/順序/occurrence一致必須",
        "main_slide_extra_calls": False, "pending_initialization": False,
        "accounting_basis_verified": False, "quality_gate_clear": False,
        "clock_caveat": "target_decoded_frame外側旧clockは既知留保。新行は実update frameのみ"}
    return receipt, config


def pairs(result: Any) -> dict[str, Any]:
    """生値をそのまま保存し、色妥当性やGTを追加認証しない。"""
    return {side: {"next": list(value.next_pair), "dnext": list(value.dnext_pair)}
            for side, value in zip(history.SIDES, (result.p1, result.p2), strict=True)}


def slide_snapshot(detector: Any) -> dict[str, Any]:
    """pulse抑制の状態を読み、updateは追加で呼ばない。"""
    return {"cooldown": detector._cooldown, "diff_history": list(detector._diff_history),
            "last_diff_score": detector._last_diff_score, "diff_threshold": detector._diff_threshold,
            "adaptive_k": detector._adaptive_k, "cooldown_frames": detector._cooldown_frames}


def model_stamp(model: Any) -> tuple[Any, ...]:
    """eval状態と共有tensorの世代を検査し、eval/to_device自体は呼ばない。"""
    modules = tuple((name, item.training) for name, item in model.named_modules())
    if any(training for _, training in modules):
        raise RuntimeError("共有CNNがeval状態ではありません")
    tensors = tuple((kind, name, id(value), value._version, value.data_ptr())
                    for kind, values in (("parameter", model.named_parameters()), ("buffer", model.named_buffers()))
                    for name, value in values)
    return modules, tensors


def rng_stamp() -> str:
    """追加推論による乱数変化を拒否し、CUDA未初期化なら起動しない。"""
    values: list[Any] = [random.getstate(), history.np.random.get_state()]
    torch = sys.modules.get("torch")
    if torch is not None:
        values.append(torch.get_rng_state().cpu().numpy().tobytes())
        if torch.cuda.is_initialized():
            values.extend(value.cpu().numpy().tobytes() for value in torch.cuda.get_rng_state_all())
    return hashlib.sha256(pickle.dumps(values, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def independent_detector(main: Any, detector_type: Any) -> tuple[Any, Any, dict[str, int]]:
    """分類器の可変wrapperを分離し、既存重みの再ロードや主履歴更新をしない。"""
    if type(main) is not detector_type or set(vars(main)) != {"_classifier", "_centroid"}:
        raise RuntimeError("独立複製対象が固定NextDetectorと違います")
    classifier = copy.copy(main._classifier)
    classifier._color = copy.copy(main._classifier._color)
    for name in ("_gate", "_strict_gate", "_ui_matcher"):
        setattr(classifier, name, copy.deepcopy(getattr(main._classifier, name)))
    model = classifier._color._model
    model_stamp(model)
    counts = {"cnn_patch_calls": 0}
    classify = classifier._color.classify
    def counted(patch: Any) -> Any:
        counts["cnn_patch_calls"] += 1
        return classify(patch)
    classifier._color.classify = counted
    detector = detector_type(classifier, centroid_classifier=copy.deepcopy(main._centroid))
    if detector is main or detector._classifier is main._classifier:
        raise RuntimeError("独立NEXTが主観測器を共有しています")
    return detector, model, counts


def inactive_read(pipeline: Any, frame: Any, detector_type: Any,
                  original_detect: Callable[..., Any], cache: dict[str, Any]) -> dict[str, Any]:
    """inactiveの独立観測は記録専用。会計・主検出器・乱数を変えない。"""
    main = pipeline._next_detector
    if main is None:
        raise RuntimeError("独立観測に必要な既存NEXT検出器がありません")
    if cache.get("main") is not main:
        detector, model, counts = independent_detector(main, detector_type)
        cache.update(main=main, detector=detector, model=model, counts=counts)
    detector, model, counts = cache["detector"], cache["model"], cache["counts"]
    main_state = tuple((name, id(value)) for name, value in vars(main).items())
    accounts = {side: history.accounting_snapshot(pipeline, side) for side in history.SIDES}
    before_model, before_rng, before_calls = model_stamp(model), rng_stamp(), counts["cnn_patch_calls"]
    started = time.perf_counter()
    result = original_detect(detector, frame.copy())
    elapsed = time.perf_counter() - started
    if main_state != tuple((name, id(value)) for name, value in vars(main).items()):
        raise RuntimeError("独立NEXTが主検出器の状態を変更しました")
    if accounts != {side: history.accounting_snapshot(pipeline, side) for side in history.SIDES}:
        raise RuntimeError("独立NEXTが会計状態を変更しました")
    if before_model != model_stamp(model) or before_rng != rng_stamp():
        raise RuntimeError("独立NEXTが共有CNNまたは乱数状態を変更しました")
    return {"raw_pairs": pairs(result), "detect_both_calls": 1,
            "cnn_patch_calls": counts["cnn_patch_calls"] - before_calls, "elapsed_sec": elapsed,
            "classifier_device": str(detector._classifier._color._device),
            "main_detector_unchanged": True, "accounting_unchanged": True,
            "shared_model_unchanged": True, "rng_unchanged": True, "is_ground_truth": False}


def observe_call(rec: Any, kind: str, operation: Callable[[], Any],
                 fields: dict[str, Any], serialize: Callable[[Any], Any]) -> Any:
    """元呼出を一回だけ行い、例外の型を変えず実返却を記録する。"""
    try:
        result = operation()
    except BaseException as exc:
        try:
            rec.emit({"kind": kind, **fields, "exception": type(exc).__name__, "returned": None})
        except BaseException:
            pass
        raise
    rec.emit({"kind": kind, **fields, "exception": None, "returned": serialize(result)})
    return result


def instrument_next(stack: contextlib.ExitStack, rec: Any, detector_type: Any,
                    state: dict[str, Any]) -> Callable[..., Any]:
    """activeの主detect_bothを再推論せず、実際の返却だけを記録する。"""
    original = detector_type.detect_both
    def detect(self: Any, frame: Any) -> Any:
        pipeline = state.get("pipeline")
        if not runner.observed.in_window(rec) or pipeline is None or self is not pipeline._next_detector:
            return original(self, frame)
        state["main_next_calls"] += 1
        return observe_call(rec, PREFIX + "main_next", lambda: original(self, frame),
                            {"caller": runner.observed.caller_site(), "extra_call": False}, pairs)
    base.patch(stack, detector_type, "detect_both", detect)
    return original


def instrument_slide(stack: contextlib.ExitStack, rec: Any, slide_type: Any,
                     state: dict[str, Any]) -> None:
    """主slideの前後と実diffを保存する。Falseを静止認証にはしない。"""
    original = slide_type.update
    def update(self: Any, prev_frame: Any, current_frame: Any) -> Any:
        pipeline = state.get("pipeline")
        sides = [] if pipeline is None else [side for side in history.SIDES
                if self is getattr(pipeline, "_slide_detector_" + side.lower())]
        if not runner.observed.in_window(rec) or not sides:
            return original(self, prev_frame, current_frame)
        side, before, caller = sides[0], slide_snapshot(self), runner.observed.caller_site()
        state["slide_calls"][side] += 1
        fields = {"side": side, "before": before, "caller": caller, "extra_call": False}
        def operation() -> Any:
            try:
                return original(self, prev_frame, current_frame)
            finally:
                fields["after"] = slide_snapshot(self)
        def serialized(result: Any) -> dict[str, Any]:
            return {"slide_motion": bool(result.slide_motion), "diff_score": result.diff_score,
                    "threshold_used": result.threshold_used, "after": slide_snapshot(self)}
        return observe_call(rec, PREFIX + "main_slide", operation, fields, serialized)
    base.patch(stack, slide_type, "update", update)


def install_update(stack: contextlib.ExitStack, cls: Any, rec: Any, detector_type: Any,
                   original_detect: Callable[..., Any], state: dict[str, Any]) -> None:
    """元update後にだけ独立観測を追加し、返却resultを変更しない。"""
    original, cache = cls.update, {}
    def update(self: Any, frame_idx: int, time_sec: float, frame: Any) -> Any:
        if state.get("pipeline") is not None:
            raise RuntimeError("再入updateは観測契約外です")
        state.update(pipeline=self, main_next_calls=0, slide_calls=Counter())
        try:
            result = original(self, frame_idx, time_sec, frame)
            if not FIRST_FRAME <= frame_idx <= LAST_FRAME:
                return result
            if rec.frame != frame_idx or not runner.valid_time(time_sec, frame_idx):
                raise RuntimeError("初手観測の実clockが不一致です")
            active = bool(result.is_match_active)
            if not active:
                value = inactive_read(self, frame, detector_type, original_detect, cache)
                rec.emit({"kind": PREFIX + "inactive_next", "is_match_active": False, **value})
            rec.emit({"kind": PREFIX + "result", "is_match_active": active,
                      "main_next_calls": state["main_next_calls"],
                      "main_slide_calls": {side: state["slide_calls"][side] for side in history.SIDES},
                      "extra_detect_both_calls": int(not active), "pending_initialized": False})
            return result
        finally:
            state["pipeline"] = None
    base.patch(stack, cls, "update", update)


def instrument(stack: contextlib.ExitStack, collector: Any, rec: Any,
               receipt: dict[str, Any]) -> dict[str, Any]:
    """既存修復を設置後、固定srcの実APIへ追加観測を接続する。"""
    ORIGINAL_INSTRUMENT(stack, collector, rec, receipt)
    from src.next_detector import NextDetector
    from src.next_slide_detector import NextSlideDetector
    modules = history.frozen_modules()
    guards = receipt["input_and_code_sha256"]
    for kind in (collector.RecognitionPipeline, NextDetector, NextSlideDetector):
        path = str(Path(inspect.getfile(kind)).resolve())
        if path not in guards or path not in modules.values():
            raise RuntimeError("初手観測の実classが開始guard/frozen moduleにありません")
    state: dict[str, Any] = {}
    original = instrument_next(stack, rec, NextDetector, state)
    instrument_slide(stack, rec, NextSlideDetector, state)
    install_update(stack, collector.RecognitionPipeline, rec, NextDetector, original, state)
    return state


def read_rows(path: Path, allow_extra: bool) -> tuple[list[Any], list[dict[str, Any]]]:
    """修復行を含む全旧行を文字列で保持し、新prefixだけを分離する。"""
    legacy, extra, counts = [], [], Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"].startswith(PREFIX):
                if not allow_extra or row["kind"] not in KINDS:
                    raise ValueError("未知または対照に混入した初手観測行です")
                extra.append(row)
                continue
            key = (row["kind"], row["frame_idx"], row.get("side"))
            occurrence = counts[key]
            counts[key] += 1
            legacy.append(((*key, occurrence), line.rstrip("\r\n")))
    return legacy, extra


def compare_legacy(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """追加観測を除いた全15722行の値・順序・重複数を拒否型で検査する。"""
    reference = REFERENCE / "frames.jsonl"
    hashes = {str(item): base.sha256(item) for item in (reference, path)}
    old, _ = read_rows(reference, False)
    new, extra = read_rows(path, True)
    if len(old) != EXPECTED_LEGACY_ROWS or old != new:
        raise ValueError("start_epoch対照との全文字列/順序/occurrence一致が成立しません")
    base.assert_unchanged(hashes)
    return {"reference_rows": len(old), "candidate_legacy_rows": len(new),
            "all_legacy_bit_exact": True, "includes_original_repair_rows": True,
            "input_sha256": hashes}, extra


def validate_observations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """全391実updateとinactive追加呼出を検査し、認識合格とは分ける。"""
    expected = list(range(FIRST_FRAME, LAST_FRAME + 1, history.STRIDE))
    for row in rows:
        frame = row["frame_idx"]
        if type(frame) is not int or frame not in expected or not runner.valid_time(row.get("time_sec"), frame):
            raise ValueError("初手観測frame/clockが不正です")
    results = [row for row in rows if row["kind"] == PREFIX + "result"]
    if len(results) != EXPECTED_UPDATES or [row["frame_idx"] for row in results] != expected:
        raise ValueError("初手resultのcoverageが不正です")
    inactive = [row for row in rows if row["kind"] == PREFIX + "inactive_next"]
    inactive_frames = [row["frame_idx"] for row in results if not row["is_match_active"]]
    if [row["frame_idx"] for row in inactive] != inactive_frames:
        raise ValueError("inactive独立観測のcoverageが不正です")
    for row in results:
        at_frame = [value for value in rows if value["frame_idx"] == row["frame_idx"]]
        validate_frame(row, at_frame)
    for row in inactive:
        validate_inactive(row)
    return {"update_count": len(results), "inactive_extra_detect_both_calls": len(inactive),
            "inactive_cnn_patch_calls": sum(row["cnn_patch_calls"] for row in inactive),
            "inactive_inference_elapsed_sec": sum(row["elapsed_sec"] for row in inactive),
            "kind_counts": dict(Counter(row["kind"] for row in rows)),
            "accounting_basis_verified": False, "quality_gate_clear": False}


def validate_frame(row: dict[str, Any], values: list[dict[str, Any]]) -> None:
    """追加呼出数と実主呼出数を照合し、inactive主推論の混入を拒否する。"""
    if type(row["is_match_active"]) is not bool or row["pending_initialized"] is not False:
        raise ValueError("active/pending観測契約が不正です")
    count = sum(value["kind"] == PREFIX + "main_next" for value in values)
    if row["main_next_calls"] != count or count not in (0, 1):
        raise ValueError("主NEXT呼出数が不正です")
    if (not row["is_match_active"] and count) or row["extra_detect_both_calls"] != int(not row["is_match_active"]):
        raise ValueError("inactiveの主/独立観測分離が不正です")
    for side in history.SIDES:
        count = sum(value["kind"] == PREFIX + "main_slide" and value.get("side") == side for value in values)
        if row["main_slide_calls"][side] != count or count not in (0, 1):
            raise ValueError("主slide呼出数が不正です")
    if any(value.get("extra_call") is not False for value in values
           if value["kind"] in (PREFIX + "main_next", PREFIX + "main_slide")):
        raise ValueError("主観測の追加呼出が混入しています")


def validate_inactive(row: dict[str, Any]) -> None:
    """副作用なし・実追加量を確認し、GTやpending投入権は発行しない。"""
    required = ("main_detector_unchanged", "accounting_unchanged", "shared_model_unchanged", "rng_unchanged")
    if any(row.get(key) is not True for key in required) or row.get("is_ground_truth") is not False:
        raise ValueError("独立推論の非干渉receiptが不正です")
    if row["detect_both_calls"] != 1 or type(row["cnn_patch_calls"]) is not int or not 0 <= row["cnn_patch_calls"] <= 8:
        raise ValueError("独立推論の追加量が不正です")
    if type(row["elapsed_sec"]) not in (int, float) or not runner.math.isfinite(row["elapsed_sec"]) or row["elapsed_sec"] < 0:
        raise ValueError("独立推論の実測時間が不正です")


def common_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """既存共通比較には新prefixだけを除き、同じ旧対照比較を継承する。"""
    legacy, _ = read_rows(path, True)
    rows, repair = {}, []
    for key, text in legacy:
        row = json.loads(text)
        if row["kind"].startswith(runner.PREFIX):
            repair.append(row)
        else:
            rows[json.dumps(key, ensure_ascii=False)] = {"row": row, "text": text}
    return rows, repair


def finish(output: Path, receipt: dict[str, Any], rec: Any, elapsed: float) -> dict[str, Any]:
    """元4artifactに観測検収を結合し、全guard後だけCOMPLETEを一回保存する。"""
    comparison, extra = compare_legacy(output / "frames.jsonl")
    observations = validate_observations(extra)
    writer, deferred = base.write_json, []
    def defer(path: Path, value: Any) -> None:
        if path == output / "COMPLETE":
            deferred.append(value)
        else:
            writer(path, value)
    with contextlib.ExitStack() as stack:
        base.patch(stack, base, "write_json", defer)
        base.patch(stack, runner, "read_rows", common_rows)
        summary = ORIGINAL_FINISH(output, receipt, rec, elapsed)
    if len(deferred) != 1:
        raise RuntimeError("元COMPLETEが一意ではありません")
    base.write_json(output / EXTRA_NAME, {"format_version": FORMAT, **comparison, **observations,
                                        "production_adoption": False})
    hashes = {**deferred[0]["sha256"], EXTRA_NAME: base.sha256(output / EXTRA_NAME)}
    base.assert_unchanged(comparison["input_sha256"])
    base.assert_unchanged({str(output / name): digest for name, digest in hashes.items()})
    base.assert_unchanged(receipt["input_and_code_sha256"])
    base.write_json(output / "COMPLETE", {"format_version": FORMAT, "status": "observation_complete_not_adopted",
        "sha256": hashes, "all_legacy_bit_exact": True, "accounting_basis_verified": False})
    return {"frame_count": summary["frame_count"], "elapsed_sec": elapsed, **observations,
            "all_legacy_bit_exact": True, "production_adoption": False}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """固定runnerの三接続点だけを一時変更し、終了/例外時に必ず戻す。"""
    with contextlib.ExitStack() as stack:
        for name, value in (("prepare", prepare), ("instrument", instrument), ("finish", finish)):
            base.patch(stack, runner, name, value)
        return runner.run(args)


def main() -> int:
    """新3fileの検収済みSHAを必須とする。GPU起動は親の承認後だけ。"""
    runner.pending.completion._configure_torch_threads()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    for name in ("script", "test", "launcher"):
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--allow-native-runtime-mismatch", action="store_true")
    args = parser.parse_args()
    args.mode, (args.start_sec, args.end_sec) = "start_epoch", runner.INTERVALS["start_epoch"]
    args.module_sha256 = FIXED_SHA["scripts/match_start_epoch_shadow_v1.py"]
    args.module_test_sha256 = FIXED_SHA["tests/test_match_start_epoch_shadow_v1.py"]
    print(json.dumps(run(args), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
