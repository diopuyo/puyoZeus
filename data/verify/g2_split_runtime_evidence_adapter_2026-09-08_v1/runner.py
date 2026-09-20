"""CPU製造／独立診断完了契約。GPUは親検収後の明示起動のみ。"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

HOME = Path(__file__).resolve().parent
LATEST = HOME.parent / "g2_chigiri_completion_adapter_2026-09-08_v1"
REFERENCE = HOME.parent / "video38_chigiri_completion_live_2026-09-08_v1"
OLD_SHA = ("03b1c9be43df7b2b0a1a6510c8f7b54d1fae97c89dcda155d7c40d9bde8e5697",
           "dfd101afdb7b024693a5a212cae4e522fd69c8f283b704ef145b2573af33f54a",
           "230cac96b06788e32db6ce46a62d077aee741314bddedd732bcb993acf059df4")
FORMAT, ENGINE = "split-runtime-evidence-diagnostic/v1", "SPLIT_ENGINE.json"
REQUIRED = frozenset(("PLAN.json", "frames.jsonl", "SUMMARY.json", "NEXT_LIVE_COMPARISON.json",
    "NEXT_LIVE_OBSERVATION.json", "ENGINE_COMPLETE", "current_entry.jsonl", "CURRENT_ENTRY_STATUS.json",
    "CURRENT_ENTRY_RECEIPT.json", "resolved_grace_guard.jsonl", "RESOLVED_GRACE_STATUS.json",
    "RESOLVED_GRACE_RECEIPT.json", "chigiri_completion.jsonl", "CHIGIRI_COMPLETION_STATUS.json",
    "CHIGIRI_COMPLETION_RECEIPT.json", "split_runtime_evidence.jsonl", "split_pending_ledger.jsonl",
    "SPLIT_EVIDENCE_RECEIPT.json", "SPLIT_EVIDENCE_STATUS.json", "SPLIT_COMPARISON.json"))
M: Any = None
OLD_ADDITIONAL: Any = None


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load(name: str, path: Path) -> Any:
    if name in sys.modules:
        raise RuntimeError("unexpected_module_alias")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def own_guards() -> dict[str, str]:
    return {str(HOME / name): sha(HOME / name) for name in
            ("adapter.py", "test_adapter.py", "runner.py", "launcher.sh", "ASSET_PREFLIGHT.md")}


def extra(args: Any) -> dict[str, str]:
    old = argparse.Namespace(**vars(args))
    for name, digest in zip(("script", "test", "launcher"), OLD_SHA, strict=True):
        setattr(old, name + "_sha256", digest)
    result = OLD_ADDITIONAL(old)
    result.update(own_guards())
    for name, file in (("script", "adapter.py"), ("test", "test_adapter.py"), ("launcher", "launcher.sh")):
        if getattr(args, name + "_sha256") != result[str(HOME / file)]:
            raise ValueError("caller_new_sha_mismatch")
    result[str(M.CONNECTION)], result[str(M.WRITER)] = M.CONNECTION_SHA, M.WRITER_SHA
    for module in (M.pending, M.pending.prediction, M.pending.completion, M.pending.prediction.generation_hooks):
        path = Path(module.__file__).resolve()
        result[str(path)] = sha(path)
    result.update(M.external_guards())
    result[str(REFERENCE / "COMPLETE")] = "373455e3131db33c8c7c5afb9f1497f39f47c1a3974575fd01326b410b8064b3"
    complete = read(REFERENCE / "COMPLETE")
    result.update({str(REFERENCE / name): digest for name, digest in complete["sha256"].items()})
    M.base.assert_unchanged(result)
    return result


def comparisons(output: Path, state: dict[str, Any]) -> dict[str, Any]:
    rec = state["pending_rec"]
    mutations = [row for row in rec.c6_rows if row.get("kind") == "c6_pending_block"]
    first = min((row.get("frame_idx", 10**12) for row in mutations), default=10**12)
    # c6_rowsの内部値にclockが無い場合、原emit streamの実clockを使う。
    emitted = [json.loads(line) for line in (output / M.LEDGER).read_text().splitlines()]
    first = min((r["frame_idx"] for r in emitted if r.get("kind") == "c6_pending_block"), default=first)
    streams = ("frames.jsonl", M.A.entry.SIDECAR, M.A.prior.SIDECAR, M.A.SIDECAR)
    report = {name: M.A.prior.compare(REFERENCE / name, output / name, [{"frame_idx": first}]) for name in streams}
    old, new = M.A.prior.indexed(REFERENCE / "frames.jsonl"), M.A.prior.indexed(output / "frames.jsonl")
    invariants = {}
    for kind, count in (("raw", M.A.RAW_COUNT), ("next", M.A.NEXT_COUNT)):
        a, b = M.A.projections(old, kind), M.A.projections(new, kind)
        valid = all(M.A.input_present(v["row"], kind) for v in new.values())
        invariants[kind] = {"equal": a == b, "old": len(a), "new": len(b), "count_ok": len(a) == len(b) == count, "structure_ok": valid}
    return {"streams": report, "invariants": invariants, "first_c6": first,
            "post_intervention_differences_require_review": True, "quality_gate_clear": False}


def finish(output: Path, receipt: Any, history: Any, elapsed: float, state: dict[str, Any]) -> Any:
    summary = state["evidence_sink"].summary()
    report = comparisons(output, state)
    write(output / M.REPORT, {**summary, "ledger": state["pending_rec"].ledger_receipt(),
          "combined": state["combined_receipt"], "runtime_source_guard": state["runtime_source_guard"],
          "saved_sm_injection": False})
    write(output / "SPLIT_COMPARISON.json", report)
    if not all(r["equal"] and r["count_ok"] and r["structure_ok"] for r in report["invariants"].values()):
        raise ValueError("raw_next_changed")
    if not all(r["pre_mutation_prefix_bit_exact"] for r in report["streams"].values()):
        raise ValueError("pre_intervention_prefix_changed")
    # 旧A.finishの2P品質gateを変更せず、原NEXT/history保存だけを一回実行する。
    result = M.A.entry.ORIGINAL_FINISH(output, receipt, history, elapsed)
    for name, sink in ((M.A.entry.REPORT, state["sink"]), (M.A.REPORT, state["completion_sink"]),
                       (M.A.prior.REPORT, state["guard_sink"])):
        write(output / name, sink.summary())
    artifacts = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    M.base.assert_unchanged(receipt["input_and_code_sha256"])
    M.base.assert_unchanged({str(output / n): h for n, h in artifacts.items()})
    write(output / ENGINE, {"format_version": FORMAT, "sha256": artifacts,
          "awaiting_actual_child_exit": True, "quality_gate_clear": False})
    return result


def finalize(output: Path, child_exit: int) -> dict[str, Any]:
    if type(child_exit) is not int or not 0 <= child_exit <= 255:
        raise ValueError("actual_numeric_child_exit_required")
    if (output / "COMPLETE").exists() or (output / "CHILD_EXIT.json").exists():
        raise FileExistsError("exclusive_finalization")
    write(output / "CHILD_EXIT.json", {"child_exit_code": child_exit, "source": "actual_wait"})
    if child_exit:
        return {"status": "child_failed", "child_exit_code": child_exit}
    engine, plan = read(output / ENGINE), read(output / "PLAN.json")
    status = read(output / "SPLIT_EVIDENCE_STATUS.json")
    if engine.get("format_version") != FORMAT or not REQUIRED <= engine.get("sha256", {}).keys():
        raise ValueError("invalid_engine")
    if status.get("closed") is not True or status.get("sticky_error") is not None or status.get("installed") is not True:
        raise ValueError("missing_or_failed_observer")
    for name, digest in engine["sha256"].items():
        if Path(name).name != name or sha(output / name) != digest:
            raise ValueError("artifact_hash_mismatch")
    for path, digest in plan["input_and_code_sha256"].items():
        if sha(Path(path)) != digest:
            raise ValueError("guard_changed")
    hashes = {**engine["sha256"], ENGINE: sha(output / ENGINE), "CHILD_EXIT.json": sha(output / "CHILD_EXIT.json")}
    value = {"format_version": FORMAT, "sha256": hashes, "child_exit_code": 0,
             "quality_gate_clear": False, "physical_identity_certified": False}
    write(output / "COMPLETE", value)
    return value


@contextlib.contextmanager
def configured() -> Any:
    with contextlib.ExitStack() as stack:
        M.base.patch(stack, M.A, "additional_guards", extra)
        M.base.patch(stack, M.A, "instrument", M.instrument)
        M.base.patch(stack, M.A, "finish", finish)
        yield


def run(args: Any) -> Any:
    with configured():
        return M.A.run(args)


def live_main() -> int:
    global M, OLD_ADDITIONAL
    M = load("_split_evidence_live", HOME / "adapter.py")
    latest = load("_split_latest_live", LATEST / "adapter.py")
    M.bind(latest)
    OLD_ADDITIONAL = latest.additional_guards
    with configured(), latest.configured(), latest.prior.configured(), contextlib.ExitStack() as stack:
        M.base.patch(stack, latest.entry, "run", run)
        M.base.patch(stack, latest.entry, "finalize", finalize)
        return latest.entry.main()


def cpu_main() -> int:
    import pytest
    sys.path.insert(0, str(HOME.parents[2]))
    output = HOME / sys.argv[1]
    output.mkdir(exist_ok=False)
    os.environ["MANUFACTURING_OUTPUT"] = str(output)
    before, start = own_guards(), time.perf_counter()
    write(output / "PLAN.json", {"pid": os.getpid(), "sha256": before})
    with (output / "PYTEST.log").open("x", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            code = int(pytest.main([str(HOME / "test_adapter.py"), "-q", "-s", "-p", "no:cacheprovider",
                                   "--basetemp", str(output / "tmp"), *sys.argv[2:]]))
    after = own_guards()
    result = {"pid": os.getpid(), "pytest_exit_code": code, "elapsed_sec": time.perf_counter() - start,
              "before": before, "after": after, "equal": before == after, "gpu_executed": False}
    write(output / "RESULT.json", result)
    if code == 0 and before == after:
        write(output / "COMPLETE.json", {"status": "manufacturing_cpu_only",
            "sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps(result), flush=True)
    return code if before == after else 2


if __name__ == "__main__":
    raise SystemExit(cpu_main() if len(sys.argv) > 1 and sys.argv[1].startswith("cpu_") else live_main())
