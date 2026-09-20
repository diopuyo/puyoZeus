"""video38短区間で、連鎖final先取り停止の1軸だけをshadow測定する。"""

from __future__ import annotations

import argparse
import contextlib
import functools
import inspect
import json
import sys
from collections import Counter
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any, Callable

from scripts import diagnose_video38_confirmed_collapse_details_v1 as details


base = details.base
FORMAT = "video38-prechain-publication-shadow/v1"
LAUNCHER = base.ROOT / "scripts/launch_video38_prechain_shadow_v1.sh"
TEST = base.ROOT / "tests/test_diagnose_video38_prechain_shadow_v1.py"
FIX_PLAN = base.ROOT / "docs/agent_coordination/CONFIRMED_ATOMIC_PUBLICATION_FIX_PLAN_2026-09-07.md"
BASE_REFERENCE = details.REFERENCE
DETAIL_REFERENCE = base.VERIFY / "video38_confirmed_collapse_2026-09-07_v3_details"
ORIGINAL_DETAIL_PREPARE = details.detail_prepare
ORIGINAL_DETAIL_INSTRUMENT = details.instrument_details
ORIGINAL_BASE_FINISH = base.finish


def _reference_receipt(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """完成済み診断rootを全構成fileのSHAまで検証する。"""
    complete_path = root / "COMPLETE"
    complete = base.read_json(complete_path)
    hashes = {str(complete_path): base.sha256(complete_path)}
    recorded = complete.get("sha256")
    required = {"PLAN.json", "frames.jsonl", "SUMMARY.json"}
    if not isinstance(recorded, dict) or not required.issubset(recorded):
        raise ValueError(f"参照診断rootの必須SHA receiptが不足: {root}")
    for name, expected in recorded.items():
        path = root / name
        if not path.is_file() or base.sha256(path) != expected:
            raise ValueError(f"参照診断rootのSHA不一致: {path}")
        hashes[str(path)] = expected
    return {"root": str(root.resolve()), "complete": complete,
            "complete_sha256": hashes[str(complete_path)]}, hashes


def _pipeline_caller(start: FrameType) -> dict[str, Any]:
    """詳細wrapperの内側からでも実際のpipeline呼出行を探す。"""
    frame: FrameType | None = start
    fallback = details.caller_value(start)
    for _ in range(8):
        if frame is None:
            break
        if frame.f_code.co_name == "_step_side":
            return {**details.caller_value(frame), "side": frame.f_locals.get("side")}
        frame = frame.f_back
    return fallback


class ShadowRecorder(details.DetailRecorder):
    """既存details行に候補policyと対象frameの返値だけを追記する。"""

    def __init__(self, stream: Any, target: dict[str, Any]) -> None:
        super().__init__(stream, target)
        self.shadow_policies: list[dict[str, Any]] = []
        self.target_results: list[dict[str, Any]] = []

    def record_shadow_policy(self, value: dict[str, Any]) -> None:
        """元finalと候補prechainをgrid込みで保存する。"""
        row = {"kind": "shadow_policy", **value}
        self.shadow_policies.append(base.json_value(row))
        self.emit(row)

    def record_side(self, pipeline: Any, side: str, result: Any) -> None:
        """base記録を維持し、対象frameの候補公開結果を別に固定する。"""
        super().record_side(pipeline, side, result)
        if self.frame == base.TARGET_FRAME and side == base.TARGET_SIDE:
            self.target_results.append({"frame_idx": self.frame, "side": side,
                                        "state": getattr(result.state, "value", str(result.state)),
                                        "confirmed": base.board_value(result.confirmed_board),
                                        "estimated": base.board_value(getattr(result, "estimated_board", None)),
                                        "chain": details.event_value(getattr(result, "chain_event", None))})


def instrument_shadow_resolve(stack: contextlib.ExitStack, module: ModuleType,
                              rec: ShadowRecorder) -> None:
    """chain_count>0の返却boardだけprechain copyへ置換する。"""
    original = module.resolve_after_placement
    signature = inspect.signature(original)

    @functools.wraps(original)
    def shadow_resolve(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        new_confirmed = bound.arguments["new_confirmed"]
        before = base.board_value(new_confirmed)
        caller = _pipeline_caller(sys._getframe(1))
        try:
            original_result = original(*args, **kwargs)
        except Exception as exc:
            rec.record_shadow_policy({"status": "original_exception", "policy_applied": False,
                                      "input_prechain": before, "exception_type": type(exc).__name__,
                                      "input_unchanged": before == base.board_value(new_confirmed), **caller})
            raise
        after = base.board_value(new_confirmed)
        if before != after:
            raise RuntimeError("resolve_after_placementが入力盤面を変更しました")
        original_board, chain_count = original_result
        returned = new_confirmed.copy() if int(chain_count) > 0 else original_board
        rec.record_shadow_policy({"status": "ok", "policy_applied": int(chain_count) > 0,
                                  "chain_count": int(chain_count), "input_prechain": before,
                                  "original_final": base.board_value(original_board),
                                  "returned_prechain": base.board_value(returned),
                                  "input_unchanged": True, **caller})
        return (returned, chain_count) if int(chain_count) > 0 else original_result

    base.patch(stack, module, "resolve_after_placement", shadow_resolve)


def shadow_prepare(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """固定区間・参照root・新規コードを開始終了SHAguardへ足す。"""
    if args.start_sec != base.START_SEC or args.end_sec != base.END_SEC:
        raise ValueError("shadowは560～583秒の固定区間です")
    receipt, config = ORIGINAL_DETAIL_PREPARE(args)
    refs, extra_hashes = {}, {}
    for name, root in (("base", BASE_REFERENCE), ("details", DETAIL_REFERENCE)):
        refs[name], hashes = _reference_receipt(root)
        extra_hashes.update(hashes)
    for path in (Path(__file__), LAUNCHER, TEST, FIX_PLAN):
        extra_hashes[str(path)] = base.sha256(path)
    receipt["input_and_code_sha256"].update(extra_hashes)
    receipt["format_version"] = FORMAT
    receipt["shadow_policy"] = {
        "adoption_status": "diagnostic_only_not_adopted",
        "one_axis": "if chain_count > 0, return new_confirmed.copy() instead of final_board",
        "unchanged": ["chain_count", "event", "accounting", "C-6", "rejection_control"],
        "pending_rollback_issue": "known_unfixed_separate_axis",
        "stale_next_exit_issue": "known_unfixed_separate_axis",
        "stale_next_observation": (
            "v3 details: frame34704でerasable gate前にgame-event NEXT変化がactiveを消去; "
            "start_next/entry_tが旧値のまま"),
        "source_and_production_flags_modified": False,
        "original_native_extension_identity_verified": False,
        "scope": "snapshot Python + current native probe; not original-native reproduction"}
    receipt["shadow_references"] = refs
    return receipt, config


def shadow_instrument(stack: contextlib.ExitStack, collector: ModuleType,
                      rec: ShadowRecorder) -> None:
    """候補をdetails resolve計装より先に差し込み、既存traceを維持する。"""
    cls = collector.RecognitionPipeline
    module = sys.modules[cls.__module__]
    instrument_shadow_resolve(stack, module, rec)
    ORIGINAL_DETAIL_INSTRUMENT(stack, collector, rec)


def _shadow_summary(receipt: dict[str, Any], rec: ShadowRecorder) -> dict[str, Any]:
    """候補だけの結果をbase SUMMARYと分離する。"""
    callers = Counter(f"{row.get('caller_function')}:{row.get('caller_line')}"
                      for row in rec.shadow_policies)
    statuses = Counter(str(row.get("status")) for row in rec.shadow_policies)
    return {"format_version": FORMAT, "status": "shadow_completed_not_adopted",
            "adoption": False, "policy": receipt["shadow_policy"],
            "policy_call_count": len(rec.shadow_policies),
            "policy_applied_count": sum(bool(row.get("policy_applied"))
                                        for row in rec.shadow_policies),
            "status_counts": dict(statuses), "caller_counts": dict(callers),
            "target_results": rec.target_results, "policy_records": rec.shadow_policies,
            "references": receipt["shadow_references"],
            "native_runtime_identity": receipt["runtime"]["native_identity"],
            "pending_rollback_problem_fixed": False,
            "stale_next_exit_problem_fixed": False,
            "source_modified": False, "production_flag_modified": False}


def shadow_finish(output: Path, receipt: dict[str, Any], rec: ShadowRecorder,
                  elapsed: float) -> dict[str, Any]:
    """別summaryを排他保存し、そのSHAも既存COMPLETEへ含める。"""
    shadow = _shadow_summary(receipt, rec)
    base.assert_unchanged(receipt["input_and_code_sha256"])
    base.write_json(output / "SHADOW_SUMMARY.json", shadow)
    original_write = base.write_json

    def write_with_shadow(path: Path, value: Any) -> None:
        if path.name == "COMPLETE":
            value = dict(value)
            value["sha256"] = {**value["sha256"],
                               "SHADOW_SUMMARY.json": base.sha256(output / "SHADOW_SUMMARY.json")}
        original_write(path, value)

    with contextlib.ExitStack() as stack:
        base.patch(stack, base, "write_json", write_with_shadow)
        result = ORIGINAL_BASE_FINISH(output, receipt, rec, elapsed)
    return {**result, "shadow_summary": shadow}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """既存base/detailsをfresh process内で再利用し全patchを復元する。"""
    with contextlib.ExitStack() as stack:
        base.patch(stack, details, "detail_prepare", shadow_prepare)
        base.patch(stack, details, "DetailRecorder", ShadowRecorder)
        base.patch(stack, details, "instrument_details", shadow_instrument)
        base.patch(stack, base, "finish", shadow_finish)
        return details.run(args)


def main() -> int:
    """原本native未証明を明示した新規rootだけを受け付ける。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, default=base.START_SEC)
    parser.add_argument("--end-sec", type=float, default=base.END_SEC)
    parser.add_argument("--allow-native-runtime-mismatch", action="store_true")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
