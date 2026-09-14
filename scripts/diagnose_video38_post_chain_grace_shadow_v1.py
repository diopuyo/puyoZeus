"""終了v2へpost-chain grace隔離だけを追加する固定45秒runner。"""

from __future__ import annotations

import argparse
import ast
import contextlib
import functools
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any, Callable

from scripts import diagnose_video38_chain_end_shadow_v2 as chain


base, common = chain.base, chain.common
MODE, FORMAT = chain.MODE, "video38-post-chain-grace-shadow/v1"
SOURCE = Path(__file__).resolve()
TEST = base.ROOT / "tests/test_diagnose_video38_post_chain_grace_shadow_v1.py"
LAUNCHER = base.ROOT / "scripts/launch_video38_post_chain_grace_shadow_v1.sh"
GRACE = base.ROOT / "scripts/post_chain_grace_shadow_v1.py"
GRACE_TEST = base.ROOT / "tests/test_post_chain_grace_shadow_v1.py"
GRACE_SHA = "6989c717f3e61244a6f1a554011dac1a84cb776b0ce866f60c3a7ea2a88b91ed"
GRACE_TEST_SHA = "a51d54e8a05b2a0c2027cb6cce026ca405d53eefe85ee714c474322782b8ffad"
REFERENCE = base.VERIFY / "video38_chain_end_shadow_2026-09-07_v2"
REFERENCE_COMPLETE_SHA = "15a3153d7c9a0529fbfeaa9ce929bf66259c1c2c66d4d396619af88ed765f1c7"
PENDING_REFERENCE = base.VERIFY / "video38_c6_pending_commit_shadow_2026-09-07_v1"
POST_NAME, ENGINE_NAME, EXIT_NAME = (
    "POST_CHAIN_GRACE_RECEIPT.json", "ENGINE_COMPLETE.json", "CHILD_EXIT.json")
GRACE_KINDS = frozenset(("post_chain_grace_observation", "post_chain_grace_lifecycle"))
ORIGINAL_PREPARE, ORIGINAL_INSTRUMENT = chain.prepare, common.instrument
ORIGINAL_FINISH, ORIGINAL_RUN = chain.finish, chain.run


def reference_guards() -> dict[str, str]:
    """終了v2と旧pendingのCOMPLETE全artifactを別々に固定する。"""
    result: dict[str, str] = {}
    for root, expected in ((REFERENCE, REFERENCE_COMPLETE_SHA), (PENDING_REFERENCE,
                            common.REFERENCE_HASHES[MODE])):
        complete = root / "COMPLETE"
        result[str(complete)] = expected
        base.assert_unchanged({str(complete): expected})
        value = base.read_json(complete)
        result.update({str(root / name): digest for name, digest in value["sha256"].items()})
    base.assert_unchanged(result)
    return result


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """終了v2の全guardへgrace一軸と直接対照を追加する。"""
    receipt, config = ORIGINAL_PREPARE(args)
    hashes = receipt["input_and_code_sha256"]
    hashes.update(reference_guards())
    hashes.update({str(GRACE): GRACE_SHA, str(GRACE_TEST): GRACE_TEST_SHA})
    hashes.update({str(path): base.sha256(path) for path in (SOURCE, TEST, LAUNCHER)})
    base.assert_unchanged(hashes)
    receipt["boundary_repair"]["reference_root"] = str(REFERENCE)
    receipt["boundary_repair"]["reference_complete_sha256"] = REFERENCE_COMPLETE_SHA
    receipt["boundary_repair"]["comparison_contract"] = (
        "終了v2既存repairを保持し、legacy/chain-end/grace三群を全差分比較")
    receipt["post_chain_grace_shadow"] = {
        "format_version": FORMAT, "reference": str(REFERENCE),
        "one_axis": "invalidate/block post-chain landing grace only",
        "landing_pending_p7_counter_modified": False,
        "chain_end_v2_preserved": True, "production_adoption": False,
        "quality_gate_clear": False, "child_exit_finalization_required": True}
    return receipt, config


def _remove_module(name: str) -> None:
    sys.modules.pop(name, None)


def load_grace(stack: contextlib.ExitStack, receipt: dict[str, Any]) -> ModuleType:
    """固定graceだけを予期しない既loadなしで絶対pathから読む。"""
    name = "_video38_post_chain_grace_v1"
    if name in sys.modules:
        raise RuntimeError("grace候補が予期せず既loadです")
    guards = receipt["input_and_code_sha256"]
    if guards.get(str(GRACE)) != GRACE_SHA or guards.get(str(GRACE_TEST)) != GRACE_TEST_SHA:
        raise RuntimeError("grace source/test guardが不一致です")
    spec = importlib.util.spec_from_file_location(name, GRACE)
    if spec is None or spec.loader is None:
        raise RuntimeError("grace候補をloadできません")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    stack.callback(_remove_module, name)
    spec.loader.exec_module(module)
    base.assert_unchanged({str(GRACE): GRACE_SHA, str(GRACE_TEST): GRACE_TEST_SHA})
    expected_dependency = "scripts/diagnose_video38_c6_pending_commit_shadow_v1.py"
    if module.GUARD_PATHS != (expected_dependency,):
        raise RuntimeError("graceのpending依存宣言が変化しました")
    dependency = str((base.ROOT / expected_dependency).resolve())
    if dependency not in guards:
        raise RuntimeError("graceのpending依存が開始guardにありません")
    return module


def _board_sha(board: Any) -> str | None:
    value = base.board_value(board)
    return None if value is None else value["sha256"]


def _grace_lines() -> dict[int, tuple[str, int]]:
    """固定pipeline ASTからgrace代入・復元・期限clear行を導出する。"""
    tree = ast.parse(chain.PIPELINE.read_text(encoding="utf-8-sig"))
    result: dict[int, tuple[str, int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        text = ast.unparse(node)
        if "self._landing_grace_" in text and "grace_until" in text:
            result[node.lineno] = ("create_or_replace", node.end_lineno or node.lineno)
        elif "ctx.confirmed_board = grace_state[1].copy()" in text:
            result[node.lineno] = ("consume", node.end_lineno or node.lineno)
        elif "self._landing_grace_" in text and text.endswith("= None"):
            result[node.lineno] = ("expire_or_invalidate", node.end_lineno or node.lineno)
    kinds = Counter(value[0] for value in result.values())
    if kinds != Counter({"create_or_replace": 2, "consume": 1,
                         "expire_or_invalidate": 2}):
        raise RuntimeError(f"grace固定行契約が変化しました: {result}")
    return result


def _actual_branch(branch: str, after: dict[str, Any]) -> str:
    """生成guard拒否をgrace生成成功へ数えない。"""
    if branch == "create_or_replace" and after.get("grace_present") is False:
        return "create_attempt_rejected"
    return branch


class LifecycleTrace:
    """既存traceへ委譲し、固定実行行の直後だけgrace状態を記録する。"""

    def __init__(self, rec: Any) -> None:
        self.rec, self.lines = rec, _grace_lines()
        self.pending: dict[int, tuple[int, int, str, dict[str, Any]]] = {}
        self.rows: list[dict[str, Any]] = []

    def observe(self, frame: FrameType, event: str) -> None:
        if frame.f_code is not self.rec.step_code:
            return
        identity = id(frame)
        current = frame.f_lineno if event == "line" else None
        pending = self.pending.get(identity)
        if event == "exception":
            self.pending.pop(identity, None)
            return
        if pending is not None and (current is None or not pending[0] <= current <= pending[1]):
            self._record(frame, pending)
            self.pending.pop(identity, None)
        if event == "return":
            return
        if current in self.lines and identity not in self.pending:
            branch, end = self.lines[current]
            self.pending[identity] = (current, end, branch, self._snapshot(frame))

    def _snapshot(self, frame: FrameType) -> dict[str, Any]:
        """実行行の前後を盤面SHAと期限に限定して取得する。"""
        local, pipe = frame.f_locals, frame.f_locals.get("self")
        side = local.get("side")
        if pipe is None or side not in ("1P", "2P"):
            return {}
        grace = getattr(pipe, f"_landing_grace_{side.lower()}", None)
        ctx = local.get("ctx")
        return {"grace_present": grace is not None,
                "grace_board_sha": None if grace is None else _board_sha(grace[1]),
                "grace_deadline_frame": None if grace is None else grace[0],
                "grace_deadline_time": None if grace is None else grace[2],
                "confirmed_sha": _board_sha(getattr(ctx, "confirmed_board", None))}

    def _record(self, frame: FrameType, pending: tuple[int, int, str, dict[str, Any]]) -> None:
        line, end, branch, before = pending
        side, after = frame.f_locals.get("side"), self._snapshot(frame)
        if not before or not after or side not in ("1P", "2P"):
            return
        branch = _actual_branch(branch, after)
        row = {"kind": "post_chain_grace_lifecycle", "side": side,
               "branch": branch, "executed_source_line": line,
               "executed_source_end_line": end, "before": before, "after": after,
               "commit_permission_issued": False, "quality_gate_clear": False}
        marker = (self.rec.frame, side, line, branch, json.dumps(after, sort_keys=True))
        if any(value.get("_marker") == marker for value in self.rows):
            return
        self.rows.append({**row, "_marker": marker})
        self.rec.emit(row)


def install_trace(stack: contextlib.ExitStack, rec: Any) -> LifecycleTrace:
    """既存traceのreturn契約を保ち、step frameだけ追加観測する。"""
    original = rec.trace
    observer = LifecycleTrace(rec)

    @functools.wraps(original)
    def trace(frame: FrameType, event: str, arg: Any) -> Callable | None:
        next_trace = original(frame, event, arg)
        observer.observe(frame, event)
        return trace if next_trace is not None else None

    existed, old = "trace" in vars(rec), vars(rec).get("trace")
    rec.trace = trace
    stack.callback(setattr, rec, "trace", old) if existed else stack.callback(vars(rec).pop, "trace", None)
    return observer


def _patch_optional(stack: contextlib.ExitStack, obj: Any, name: str, value: Any) -> None:
    """instance追加属性も終了時に不在へ戻す。"""
    existed, old = name in vars(obj), vars(obj).get(name)
    setattr(obj, name, value)
    stack.callback(setattr, obj, name, old) if existed else stack.callback(vars(obj).pop, name, None)


def instrument(stack: contextlib.ExitStack, collector: Any, rec: Any,
               receipt: dict[str, Any]) -> None:
    """終了v2の全計装後にgraceと実branch traceを一度だけ設置する。"""
    ORIGINAL_INSTRUMENT(stack, collector, rec, receipt)
    module = load_grace(stack, receipt)
    controller = module.install(stack, collector, rec)
    trace = install_trace(stack, rec)
    _patch_optional(stack, rec, "post_chain_grace_controller", controller)
    _patch_optional(stack, rec, "post_chain_grace_trace", trace)


def _partition(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """legacy・chain-end・graceを同frame出現順付きで分離する。"""
    groups: tuple[dict[str, Any], dict[str, Any]] = ({}, {})
    grace, seen = [], Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") in GRACE_KINDS:
                grace.append(row)
                continue
            group = groups[1] if str(row.get("kind")).startswith("boundary_repair_") else groups[0]
            key0 = (row.get("kind"), row.get("frame_idx"), row.get("side"))
            key = json.dumps((*key0, seen[key0]), ensure_ascii=False)
            seen[key0] += 1
            group[key] = {"row": row, "text": line.rstrip("\r\n")}
    return groups[0], groups[1], grace


def _group_diff(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """一群の全文差・欠落・追加・順序を除外なしで返す。"""
    common_keys = old.keys() & new.keys()
    changed = [key for key in new if key in common_keys and old[key]["text"] != new[key]["text"]]
    return {"changed_keys": changed,
            "changed_fields": {key: common.changed_fields(old[key]["row"], new[key]["row"])
                               for key in changed},
            "missing_keys": [key for key in old if key not in new],
            "added_keys": [key for key in new if key not in old],
            "common_key_order_equal": [key for key in old if key in new]
                                      == [key for key in new if key in old]}


def _prefix_exact(old: dict[str, Any], new: dict[str, Any], first: int) -> bool:
    """最初の実chain-end介入前は行・原文・順序の一字差も拒否する。"""
    select = lambda values: [(key, value["text"]) for key, value in values.items()
                             if (value["row"].get("frame_idx") or -1) < first]
    return select(old) == select(new)


def _validate_grace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """grace行の時計・side・無許可を全件検査する。"""
    reasons = Counter()
    for row in rows:
        frame, side = row.get("frame_idx"), row.get("side")
        if type(frame) is not int or side not in ("1P", "2P") or not common.valid_time(row.get("time_sec"), frame):
            raise ValueError("grace観測の実clock/sideが不正です")
        if row.get("commit_permission_issued") is not False:
            raise ValueError("grace観測がcommit許可を発行しました")
        reasons[str(row.get("reason") or row.get("branch"))] += 1
    return {"count": len(rows), "reason_or_branch_counts": dict(reasons)}


def compare_rows(reference: Path, candidate: Path, mode: str) -> dict[str, Any]:
    """既存終了介入を保持したままgrace一軸の3群差を保存する。"""
    if mode != MODE or reference.resolve() != (REFERENCE / "frames.jsonl").resolve():
        raise ValueError("固定終了v2対照以外は比較しません")
    inputs = {str(path): base.sha256(path) for path in (reference, candidate)}
    old, old_repairs, old_grace = _partition(reference)
    new, new_repairs, grace = _partition(candidate)
    if old_grace:
        raise ValueError("対照にgrace候補行が混入しています")
    old_mutations = common.validate_repairs([value["row"] for value in old_repairs.values()], mode)
    new_mutations = common.validate_repairs([value["row"] for value in new_repairs.values()], mode)
    if len(old_mutations) != 1 or len(new_mutations) != 1:
        raise ValueError("終了介入mutationが一意ではありません")
    first = old_mutations[0]
    if not _prefix_exact(old, new, first) or not _prefix_exact(old_repairs, new_repairs, first):
        raise ValueError("grace実変更前に終了v2対照との差があります")
    result = {"reference": str(REFERENCE), "legacy": _group_diff(old, new),
              "chain_end": _group_diff(old_repairs, new_repairs),
              "grace": _validate_grace(grace), "first_chain_end_mutation_frame": first,
              "coverage": common.coverage(new, mode), "quality_gate_clear": False,
              "production_adoption": False, "input_sha256": inputs}
    base.assert_unchanged(inputs)
    return result


def _post_receipt(output: Path, comparison: dict[str, Any]) -> dict[str, Any]:
    """全grace行と正常窓を既存raw行から集計する。"""
    legacy, _repairs, grace = _partition(output / "frames.jsonl")
    raw_rows = {key: value["row"] for key, value in legacy.items()}
    windows = {"normal_1p_cold": ("1P", 34080, 34380),
               "normal_1p_landing": ("1P", 34700, 34792),
               "target_2p_completion": ("2P", 35704, 36298)}
    return {"format_version": FORMAT, "grace_rows": grace,
            "grace_summary": comparison["grace"],
            "windows": {name: common.pending._window_stats(raw_rows, *spec)
                        for name, spec in windows.items()},
            "direct_reference_collector_count": 11,
            "older_pending_recovery_target_collector_count": 12,
            "quality_gate_clear": False, "production_adoption": False}


def finish(output: Path, receipt: dict[str, Any], rec: Any, elapsed: float) -> dict[str, Any]:
    """元COMPLETEを保留し、post receiptとENGINE_COMPLETEまで保存する。"""
    comparison = compare_rows(REFERENCE / "frames.jsonl", output / "frames.jsonl", MODE)
    base.write_json(output / POST_NAME, _post_receipt(output, comparison))
    deferred: list[dict[str, Any]] = []
    writer = base.write_json

    def defer(path: Path, value: Any) -> None:
        deferred.append(value) if path == output / "COMPLETE" else writer(path, value)

    with contextlib.ExitStack() as stack:
        base.patch(stack, base, "write_json", defer)
        summary = ORIGINAL_FINISH(output, receipt, rec, elapsed)
    if len(deferred) != 1:
        raise RuntimeError("終了v2 COMPLETEが一意ではありません")
    hashes = {**deferred[0]["sha256"], POST_NAME: base.sha256(output / POST_NAME)}
    base.assert_unchanged({str(output / name): digest for name, digest in hashes.items()})
    base.assert_unchanged(receipt["input_and_code_sha256"])
    base.write_json(output / ENGINE_NAME, {"format_version": FORMAT, "sha256": hashes,
                                           "finalization_pending": True})
    return {**summary, "post_chain_grace": _post_receipt(output, comparison),
            "finalization_pending": True}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """終了v2 runnerの4拡張点だけを一時差替えする。"""
    with contextlib.ExitStack() as stack:
        for owner, name, value in ((chain, "prepare", prepare), (chain, "finish", finish),
                                   (chain, "compare_rows", compare_rows),
                                   (common, "instrument", instrument)):
            base.patch(stack, owner, name, value)
        return ORIGINAL_RUN(args)


def finalize(output: Path, child_exit_code: int) -> dict[str, Any]:
    """実child終了後だけ数値codeと全SHAを結合して最終COMPLETEを作る。"""
    output.mkdir(parents=True, exist_ok=True)
    exit_value = {"format_version": FORMAT, "child_exit_code": child_exit_code}
    base.write_json(output / EXIT_NAME, exit_value)
    if child_exit_code != 0:
        return {**exit_value, "complete_written": False}
    engine = base.read_json(output / ENGINE_NAME)
    plan = base.read_json(output / "PLAN.json")
    guards = plan.get("input_and_code_sha256")
    if not isinstance(guards, dict) or not guards:
        raise RuntimeError("PLANの開始guardが欠落しています")
    base.assert_unchanged(guards)
    hashes = dict(engine["sha256"])
    hashes.update({ENGINE_NAME: base.sha256(output / ENGINE_NAME),
                   EXIT_NAME: base.sha256(output / EXIT_NAME)})
    base.assert_unchanged({str(output / name): digest for name, digest in hashes.items()})
    base.write_json(output / "COMPLETE", {"format_version": FORMAT,
                                            "status": "experiment_complete_not_adopted",
                                            "sha256": hashes})
    return {**exit_value, "complete_written": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--finalize-child-exit", type=int)
    parser.add_argument("--allow-native-runtime-mismatch", action="store_true")
    args = parser.parse_args()
    if args.finalize_child_exit is not None:
        print(json.dumps(finalize(args.output_root, args.finalize_child_exit), ensure_ascii=False))
        return 0
    common.pending.completion._configure_torch_threads()
    args.mode, args.module_sha256 = MODE, chain.MODULE_SHA
    args.module_test_sha256 = chain.MODULE_TEST_SHA
    args.start_sec, args.end_sec = common.INTERVALS[MODE]
    print(json.dumps(run(args), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
