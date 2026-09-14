"""旧品質判定を変えず、C6診断の保存だけを実historyへ接続する。"""
from __future__ import annotations

import contextlib
import functools
import math
from pathlib import Path
from typing import Any

from scripts import diagnose_video38_next_enqueue_live_shadow_v1 as next_runner

TASK_DIR = Path(__file__).resolve().parent
PROJECT = TASK_DIR.parents[2]
RUNTIME = TASK_DIR.parent / "g2_split_runtime_evidence_adapter_2026-09-08_v1/runner.py"
FORMAT = "split-diagnostic-history-finisher/v1"
STREAMS = ("frames.jsonl", "current_entry.jsonl", "resolved_grace_guard.jsonl", "chigiri_completion.jsonl")
COUNTS = {"raw": 7248, "next": 2100}
HISTORY_FINISH = next_runner.runner.ORIGINAL_HISTORY_FINISH
FIXED = {
    str(RUNTIME): "78bdc5fdc58e2b2300ddff2215bc0276be4d810ce36bfa87aeff7505924fd93e",
    str(PROJECT / "scripts/diagnose_video38_next_enqueue_live_shadow_v1.py"):
        "9747198bf5b89e963a8fc8b95aeaee0997ae7e3069e8e5c5e77209489bb3fe2e",
    str(PROJECT / "scripts/diagnose_video38_accounting_history_v1.py"):
        "23836d917a97c4667c2a23b31ff863bc4e848c26d450506aa51134345b53768e",
    str(PROJECT / "scripts/diagnose_video38_confirmed_collapse_v1.py"):
        "b20525d5b7423a34125a45b11f76aea68d0942276109463007dfd991e46fa144",
}
base = next_runner.base


def guards() -> dict[str, str]:
    """旧prepare全guardに追加する。モデル等の既存guardを置換しない。"""
    base.assert_unchanged(FIXED)
    return {**FIXED, **{str(TASK_DIR / name): base.sha256(TASK_DIR / name)
                       for name in ("finisher.py", "ASSET_PREFLIGHT.md")}}


def validate_start(output: Path, receipt: dict[str, Any], elapsed: float) -> None:
    """既完了・失敗runの再利用と未束縛codeを拒否する。"""
    blocked = ("COMPLETE", "CHILD_EXIT.json", "SPLIT_ENGINE.json", "ENGINE_COMPLETE",
               "SUMMARY.json", next_runner.COMPARE_NAME, next_runner.EXTRA_NAME)
    if any((output / name).exists() for name in blocked):
        raise FileExistsError("exclusive_new_diagnostic_save_required")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid_elapsed")
    if base.read_json(output / "PLAN.json") != receipt:
        raise ValueError("receipt_plan_mismatch")
    actual = receipt["input_and_code_sha256"]
    if any(actual.get(path) != sha for path, sha in guards().items()):
        raise ValueError("finisher_prepare_guard_missing")
    base.assert_unchanged(actual)


def validate_comparison(output: Path, runtime: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """直前の実4stream比較の入力を束縛し、0対0や未知scopeを拒否する。"""
    path = output / "SPLIT_COMPARISON.json"
    report, bound = base.read_json(path), {str(path): base.sha256(path)}
    if set(report["streams"]) != set(STREAMS) or set(report["invariants"]) != set(COUNTS):
        raise ValueError("unknown_comparison_scope")
    if type(report["first_c6"]) is not int or report["quality_gate_clear"] is not False:
        raise ValueError("invalid_diagnostic_contract")
    for name in STREAMS:
        value = report["streams"][name]
        expected = {str(runtime.REFERENCE / name), str(output / name)}
        if set(value["input_sha256"]) != expected or value["pre_mutation_prefix_bit_exact"] is not True:
            raise ValueError("unbound_or_changed_prefix")
        bound.update(value["input_sha256"])
    for name, count in COUNTS.items():
        value = report["invariants"][name]
        if any(value[key] is not True for key in ("equal", "count_ok", "structure_ok")):
            raise ValueError("raw_next_changed_or_missing")
        if any(type(value[key]) is not int or value[key] != count for key in ("old", "new")):
            raise ValueError("nonempty_call_coverage_required")
    base.assert_unchanged(bound)
    return report, bound


def diagnostic_reports(output: Path, runtime: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """全NEXT実callを既存validatorへ渡し、対照欠測を補完しない。"""
    report, bound = validate_comparison(output, runtime)
    _, rows = next_runner.read_rows(output / "frames.jsonl")
    extra = next_runner.observation_summary(rows)
    supplement = next_runner.SUPPLEMENT / "frames.jsonl"
    bound[str(supplement)] = base.sha256(supplement)
    extra["raw_comparison"] = next_runner.raw_comparison(rows, supplement)
    comparison = {"format_version": FORMAT, "input_sha256": bound,
        "full_difference_artifact": "SPLIT_COMPARISON.json",
        "fixed_prefix_before_frame": report["first_c6"], "prefix_bit_exact": True,
        "invariants": report["invariants"], "legacy_next_quality_gate_executed": False,
        "post_intervention_differences_require_review": True, "quality_gate_clear": False}
    return comparison, extra


def history_save(output: Path, receipt: dict[str, Any], rec: Any, elapsed: float) -> tuple[Any, Any]:
    """実history保存を一回だけ呼び、最終COMPLETEだけ保留する。"""
    writer, deferred = base.write_json, []
    def defer(path: Path, value: Any) -> None:
        if path == output / "COMPLETE":
            deferred.append(value)
        else:
            writer(path, value)
    with contextlib.ExitStack() as stack:
        base.patch(stack, base, "write_json", defer)
        result = HISTORY_FINISH(output, receipt, rec, elapsed)
    if len(deferred) != 1:
        raise RuntimeError("original_history_complete_not_unique")
    return result, deferred[0]


def finish(output: Path, receipt: dict[str, Any], rec: Any, elapsed: float, *, runtime: Any) -> Any:
    """NEXT旧prefixを迂回するのは新診断だけ。公開・会計の許可は発行しない。"""
    validate_start(output, receipt, elapsed)
    comparison, extra = diagnostic_reports(output, runtime)
    summary, deferred = history_save(output, receipt, rec, elapsed)
    base.write_json(output / next_runner.COMPARE_NAME, comparison)
    base.write_json(output / next_runner.EXTRA_NAME, extra)
    hashes = {**deferred["sha256"], **{name: base.sha256(output / name)
              for name in (next_runner.COMPARE_NAME, next_runner.EXTRA_NAME)}}
    if set(hashes) != next_runner.ENGINE_NAMES:
        raise RuntimeError("history_five_artifacts_required")
    base.assert_unchanged(comparison["input_sha256"])
    base.assert_unchanged(receipt["input_and_code_sha256"])
    base.assert_unchanged({str(output / name): value for name, value in hashes.items()})
    base.write_json(output / "ENGINE_COMPLETE", {"format_version": FORMAT, "sha256": hashes,
        "status": "awaiting_real_child_exit", "quality_gate_clear": False})
    return {"frame_count": summary["frame_count"], "elapsed_sec": elapsed,
            "status": "awaiting_real_child_exit", "quality_gate_clear": False}


def install(stack: contextlib.ExitStack, runtime: Any) -> None:
    """固定外側runnerのcaptured保存だけを差替え、成功/例外とも元identityへ戻す。"""
    guards()
    if Path(runtime.__file__).resolve() != RUNTIME or runtime.M.A.entry.previous is not next_runner:
        raise ValueError("unexpected_runtime_identity")
    entry = runtime.M.A.entry
    if entry.ORIGINAL_FINISH is not next_runner.finish:
        raise ValueError("unexpected_captured_finish_or_reentry")
    base.patch(stack, entry, "ORIGINAL_FINISH", functools.partial(finish, runtime=runtime))
