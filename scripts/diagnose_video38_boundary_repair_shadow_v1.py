"""開始・終了・重力の独立修正を既存runへ一軸ずつ接続して比較する。"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

from scripts import diagnose_video38_start_gate_v1 as observed
from scripts import diagnose_video38_c6_pending_commit_shadow_v1 as pending


base, history = observed.base, observed.history
FORMAT = "video38-boundary-repair-shadow/v1"
PREFIX = "boundary_repair_"
DIFFERENCE_NAME = "BOUNDARY_REPAIR_COMPARISON.json"
TEST = base.ROOT / "tests/test_diagnose_video38_boundary_repair_shadow_v1.py"
LAUNCHER = base.ROOT / "scripts/launch_video38_boundary_repair_shadow_v1.sh"
MODULES = {"start_epoch": "match_start_epoch_shadow_v1",
           "chain_end": "chain_end_epoch_shadow_v1", "gravity_grid": "gravity_full_grid_shadow_v1"}
REFERENCE_ROOTS = {"start_epoch": "video38_start_gate_2026-09-07_v1",
                   "gravity_grid": "video38_start_gate_2026-09-07_v1",
                   "chain_end": "video38_c6_pending_commit_shadow_2026-09-07_v1"}
REFERENCE_HASHES = {"start_epoch": "d75b48fb91157cb4be162902faa407cb19f8aa389f094b706b67498d0a02c9af",
                    "gravity_grid": "d75b48fb91157cb4be162902faa407cb19f8aa389f094b706b67498d0a02c9af",
                    "chain_end": "056af44feb510a8d06e187c7c39accaec3b7e4501b2056016b0de7dc3c846631"}
INTERVALS = {"start_epoch": (484.2, 605.0), "gravity_grid": (484.2, 605.0),
             "chain_end": (560.0, 605.0)}
START_EVIDENCE = {
    "data/verify/video38_accounting_start_review_2026-09-07_v1/PHYSICAL_QA.md":
        "7f6fa93386267f786165488e17e9a3b7f95b09bc4ad0816bb6c8a1186d18d822",
    "data/verify/video38_start_gate_2026-09-07_v1/START_GATE_CAUSAL_QA.md":
        "fb825bbf31341f6cfe4db7cbf6ed129af6f2e8cf285911f6d45a221a26846203"}
ORIGINAL_HISTORY_PREPARE = observed.prepare
ORIGINAL_HISTORY_INSTRUMENT = observed.instrument
ORIGINAL_HISTORY_FINISH = history.finish
ORIGINAL_PENDING_PREPARE = pending.pending_prepare
ORIGINAL_PENDING_INSTRUMENT = pending.pending_instrument
ORIGINAL_PENDING_FINISH = pending.pending_finish


def module_paths(mode: str) -> tuple[Path, Path]:
    """指定一軸のmodule/testだけをguard対象にする。"""
    name = MODULES[mode]
    return base.ROOT / "scripts" / f"{name}.py", base.ROOT / "tests" / f"test_{name}.py"


def reference_guards(mode: str) -> dict[str, str]:
    """対照の完了receiptと全artifactを実SHAで固定する。"""
    root = base.VERIFY / REFERENCE_ROOTS[mode]
    guards = {str(root / "COMPLETE"): REFERENCE_HASHES[mode]}
    base.assert_unchanged(guards)
    value = base.read_json(root / "COMPLETE")
    guards.update({str(root / name): digest for name, digest in value["sha256"].items()})
    base.assert_unchanged(guards)
    return guards


def validate_dependencies(module: Path, guards: dict[str, str]) -> None:
    """moduleを実行せず、literal依存pathが既存承認guard内か確認する。"""
    tree = ast.parse(module.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "GUARD_PATHS" for target in node.targets):
            continue
        values = ast.literal_eval(node.value)
        if not isinstance(values, (tuple, list)) or not all(isinstance(v, str) for v in values):
            raise ValueError("GUARD_PATHSはpath文字列のliteral tuple/listだけです")
        for value in values:
            path = (base.ROOT / value).resolve()
            if str(path) not in guards:
                raise ValueError(f"未承認の追加依存です: {path}")


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """元の実行条件を保ち、選択候補と固定対照だけを追加する。"""
    expected = INTERVALS[args.mode]
    if (args.start_sec, args.end_sec) != expected:
        raise ValueError("modeごとの事前登録区間と違います")
    original = ORIGINAL_PENDING_PREPARE if args.mode == "chain_end" else ORIGINAL_HISTORY_PREPARE
    receipt, config = original(args)
    module, test = module_paths(args.mode)
    hashes = receipt["input_and_code_sha256"]
    hashes.update(reference_guards(args.mode))
    if args.mode == "start_epoch":
        hashes.update({str((base.ROOT / path).resolve()): digest for path, digest in START_EVIDENCE.items()})
    hashes.update({str(module.resolve()): args.module_sha256, str(test.resolve()): args.module_test_sha256})
    hashes.update({str(path.resolve()): base.sha256(path) for path in (Path(__file__), TEST, LAUNCHER)})
    validate_dependencies(module, hashes)
    base.assert_unchanged(hashes)
    receipt["boundary_repair"] = {"format_version": FORMAT, "mode": args.mode,
        "module": str(module.resolve()), "module_sha256": args.module_sha256,
        "reference_root": str(base.VERIFY / REFERENCE_ROOTS[args.mode]),
        "reference_complete_sha256": REFERENCE_HASHES[args.mode], "one_repair_axis": True,
        "production_adoption": False, "quality_gate_clear": False,
        "comparison_contract": "全行差分と初回介入より前の完全一致。欠落/追加も保存し改善を自動認定しない"}
    receipt["observation_only"] = False
    return receipt, config


def load_module(stack: contextlib.ExitStack, receipt: dict[str, Any]) -> ModuleType:
    """snapshot探索先を使わず、固定候補を一時的に絶対pathから読む。"""
    info = receipt["boundary_repair"]
    path, name = Path(info["module"]), f"_video38_boundary_repair_{info['mode']}"
    base.assert_unchanged({str(path): info["module_sha256"]})
    if name in sys.modules:
        raise RuntimeError("候補moduleが予期せず既loadです")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("候補moduleをloadできません")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    stack.callback(sys.modules.pop, name, None)
    spec.loader.exec_module(module)
    base.assert_unchanged({str(path): info["module_sha256"]})
    guards = receipt["input_and_code_sha256"]
    for dependency, expected in getattr(module, "REQUIRED_INPUT_SHA256", {}).items():
        if guards.get(str(Path(dependency).resolve())) != expected:
            raise RuntimeError(f"候補の実依存が開始guardにありません: {dependency}")
    return module


def install_generation(stack: contextlib.ExitStack, collector: Any, rec: Any) -> None:
    """重力単独runにも既存software世代trackerを一つだけ提供する。"""
    hooks = pending.prediction.generation_hooks
    def record(value: dict[str, Any]) -> None:
        rec.emit({"kind": PREFIX + "observation", "repair": "gravity_grid",
                  "source_kind": "software_generation", "payload": value})
    tracker = hooks.PipelineGenerationRecorder(record)
    rec.generation_recorder = tracker
    cls = collector.RecognitionPipeline
    machine = sys.modules[cls.__module__].BoardStateMachine
    hooks.install_generation_hooks(stack, cls, machine, tracker)


def instrument(stack: contextlib.ExitStack, collector: Any, rec: Any,
               receipt: dict[str, Any]) -> None:
    """既存計装の後へ修正を一軸だけ設置する。"""
    mode = receipt["boundary_repair"]["mode"]
    original = ORIGINAL_PENDING_INSTRUMENT if mode == "chain_end" else ORIGINAL_HISTORY_INSTRUMENT
    original(stack, collector, rec)
    if mode == "gravity_grid":
        install_generation(stack, collector, rec)
    module = load_module(stack, receipt)
    module.install(stack, collector, rec)


def read_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """全kind/frame/side/同frame順を保存し、新候補行だけを分離する。"""
    rows, repair, occurrences = {}, [], Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"].startswith(PREFIX):
                repair.append(row)
                continue
            identity = (row["kind"], row["frame_idx"], row.get("side"))
            index = occurrences[identity]
            occurrences[identity] += 1
            key = json.dumps((*identity, index), ensure_ascii=False)
            rows[key] = {"row": row, "text": line.rstrip("\r\n")}
    return rows, repair


def validate_repairs(rows: list[dict[str, Any]], mode: str) -> list[int]:
    """実clockのない介入と別候補の混入を拒否する。"""
    mutations = []
    first, last = (round(value * history.FPS) for value in INTERVALS[mode])
    for row in rows:
        if row.get("repair") != mode or row["kind"] not in (PREFIX + "mutation", PREFIX + "observation"):
            raise ValueError("未知の修正行または別modeが混入しました")
        if row["kind"] != PREFIX + "mutation":
            continue
        frame = row["frame_idx"]
        if type(frame) is not int or not first <= frame < last or frame % history.STRIDE:
            raise ValueError("介入frameが実区間外です")
        if not valid_time(row.get("time_sec"), frame):
            raise ValueError("介入clockが不一致です")
        if "before" not in row or "after" not in row or "evidence" not in row:
            raise ValueError("介入前後または根拠が欠けています")
        mutations.append(frame)
    return mutations


def valid_time(value: Any, frame: int) -> bool:
    """NaN、bool、欠測を正常clockへ含めない。"""
    return (type(value) in (int, float) and math.isfinite(value)
            and abs(value - frame / history.FPS) <= 1e-8)


def coverage(rows: dict[str, Any], mode: str) -> dict[str, Any]:
    """各engineの実frame/side母数を独立に検査する。"""
    first, last = (round(value * history.FPS) for value in INTERVALS[mode])
    expected = [(frame, side) for frame in range(first, last, history.STRIDE) for side in history.SIDES]
    kinds = ("frame_side", "raw_score_ocr", "completion_frame_observation") if mode == "chain_end" else (
        "frame_side", "accounting_update")
    counts = {}
    for kind in kinds:
        selected = [value["row"] for value in rows.values() if value["row"]["kind"] == kind]
        if [(row["frame_idx"], row.get("side")) for row in selected] != expected:
            raise ValueError(f"必須coverageに欠落/重複/順序差: {kind}")
        if any(not valid_time(row.get("time_sec"), row["frame_idx"]) for row in selected):
            raise ValueError(f"必須coverageのclock不一致: {kind}")
        counts[kind] = len(selected)
    return counts


def changed_fields(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """欠列とNoneも区別し、変更fieldの全前後値を保存する。"""
    result = {}
    for field in before.keys() | after.keys():
        if (field in before) == (field in after) and before.get(field) == after.get(field):
            continue
        result[field] = {"before_present": field in before, "after_present": field in after,
                         "before": before.get(field), "after": after.get(field)}
    return result


def compare_rows(reference: Path, candidate: Path, mode: str) -> dict[str, Any]:
    """介入前の非干渉を要求し、介入後は差分を欠落込みで全保存する。"""
    inputs = {str(path): base.sha256(path) for path in (reference, candidate)}
    old, prior_repairs = read_rows(reference)
    new, repairs = read_rows(candidate)
    if prior_repairs:
        raise ValueError("対照に新修正行が既にあります")
    mutations = validate_repairs(repairs, mode)
    first = min(mutations) if mutations else INTERVALS[mode][1] * history.FPS
    prefix = lambda values: [(key, value["text"]) for key, value in values.items()
                             if (value["row"]["frame_idx"] if value["row"]["frame_idx"] is not None else -1) < first]
    if prefix(old) != prefix(new):
        raise ValueError("初回介入より前に未説明の差分があります")
    common = old.keys() & new.keys()
    changed = [key for key in new if key in common and old[key]["text"] != new[key]["text"]]
    fields, differences = Counter(), {}
    for key in changed:
        before, after = old[key]["row"], new[key]["row"]
        differences[key] = changed_fields(before, after)
        for field in differences[key]:
            fields[f"{after['kind']}.{field}"] += 1
    result = {"reference_rows": len(old), "candidate_rows": len(new), "repair_rows": len(repairs),
        "mutation_count": len(mutations), "first_mutation_frame": min(mutations) if mutations else None,
        "pre_mutation_prefix_bit_exact": True, "all_legacy_rows_bit_exact": list(old.items()) == list(new.items()),
        "changed_keys": changed, "changed_field_counts": dict(fields), "changed_fields": differences,
        "missing_keys": [key for key in old if key not in new], "added_keys": [key for key in new if key not in old],
        "missing_rows": {key: value for key, value in old.items() if key not in new},
        "added_rows": {key: value for key, value in new.items() if key not in old},
        "coverage": coverage(new, mode), "input_sha256": inputs, "quality_gate_clear": False,
        "status": "INTERVENTION_REQUIRES_CAUSAL_QA" if mutations else "NO_INTERVENTION_NOT_QUALITY_CLEAR"}
    base.assert_unchanged(inputs)
    return result


def finish(output: Path, receipt: dict[str, Any], rec: Any, elapsed: float) -> dict[str, Any]:
    """元成果物を保持し、差分/最終guard照合後だけCOMPLETEを保存する。"""
    mode = receipt["boundary_repair"]["mode"]
    reference = Path(receipt["boundary_repair"]["reference_root"]) / "frames.jsonl"
    comparison = compare_rows(reference, output / "frames.jsonl", mode)
    original = ORIGINAL_PENDING_FINISH if mode == "chain_end" else ORIGINAL_HISTORY_FINISH
    writer, deferred = base.write_json, []
    def defer(path: Path, value: Any) -> None:
        if path == output / "COMPLETE":
            deferred.append(value)
        else:
            writer(path, value)
    with contextlib.ExitStack() as stack:
        base.patch(stack, base, "write_json", defer)
        summary = original(output, receipt, rec, elapsed)
    if len(deferred) != 1:
        raise RuntimeError("元COMPLETEが一意ではありません")
    base.write_json(output / DIFFERENCE_NAME, comparison)
    base.assert_unchanged(comparison["input_sha256"])
    base.assert_unchanged(receipt["input_and_code_sha256"])
    hashes = {**deferred[0]["sha256"], DIFFERENCE_NAME: base.sha256(output / DIFFERENCE_NAME)}
    base.assert_unchanged({str(output / name): digest for name, digest in hashes.items()})
    base.write_json(output / "COMPLETE", {"format_version": FORMAT, "mode": mode,
        "status": "experiment_complete_not_adopted", "sha256": hashes})
    return {"frame_count": summary["frame_count"], "elapsed_sec": elapsed,
            "mode": mode, "comparison": comparison, "production_adoption": False}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """別process・別rootで一候補だけを元engineへ一時接続する。"""
    state: dict[str, Any] = {}
    def prepared(value: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
        result = prepare(value)
        state["receipt"] = result[0]
        return result
    def installed(stack: contextlib.ExitStack, collector: Any, rec: Any) -> None:
        instrument(stack, collector, rec, state["receipt"])
    with contextlib.ExitStack() as stack:
        if args.mode == "chain_end":
            owner, names, execute = pending, ("pending_prepare", "pending_instrument", "pending_finish"), pending.run
        else:
            owner, names, execute = history, ("prepare", "instrument_pipeline", "finish"), history.run
        for name, function in zip(names, (prepared, installed, finish), strict=True):
            base.patch(stack, owner, name, function)
        return execute(args)


def main() -> int:
    """固定SHAを必須とし、検収済みmodeだけを明示起動する。"""
    pending.completion._configure_torch_threads()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=tuple(MODULES))
    parser.add_argument("--module-sha256", required=True)
    parser.add_argument("--module-test-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-native-runtime-mismatch", action="store_true")
    args = parser.parse_args()
    args.start_sec, args.end_sec = INTERVALS[args.mode]
    result = run(args)
    print(json.dumps({key: value for key, value in result.items() if key != "comparison"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
