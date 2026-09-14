"""固定NEXT診断へ入口観測sidecarだけを接続する。GPU起動は独立QA後。"""
from __future__ import annotations

import argparse
import contextlib
import functools
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable

from scripts import diagnose_video38_next_enqueue_live_shadow_v1 as previous

base = previous.base
ROOT = Path(__file__).resolve().parents[4]
HOME = Path(__file__).resolve().parent
OBSERVER = HOME.parent / "observer.py"
OBSERVER_TEST = HOME.parent / "test_observer.py"
PLAN = ROOT / "docs/agent_coordination/G2_CURRENT_ENTRY_OBSERVATION_PLAN_2026-09-08.md"
REFERENCE = base.VERIFY / "video38_next_enqueue_live_shadow_2026-09-08_v1"
REFERENCE_SHA = "cbee113e62bbe460bab241d096fdd59801e3ffe8eca40c53461401e1fe5e9452"
OLD_SHA = ("9747198bf5b89e963a8fc8b95aeaee0997ae7e3069e8e5c5e77209489bb3fe2e",
           "3028247fdb0c5b3749cb4a9c082f53b5d0eeb6d3bf2ab1fad90038b5aeaf90bd",
           "f7b30ba90d73ff33af4632bce56d3315c201cae87d0112aa1317d37b2e59589d")
HOOK_SHA = ("ea55257a4d178a9f2976da3c3a05ad09080f6f9a5d263033b23858e2f45432e7",
            "7266f10377c22eca6607d6fd48adb7e5a6ca433625614228c33aa67e76202f61")
PLAN_SHA = "7dc8db435775e88aa98c234d4220b841fd723f75c1d799ef2b7c14246c61f49e"
BOUNDARY_NAME = "_video38_start_gate_end_observer"
HOOK_NAME = "_video38_current_entry_observer"
WINDOWS, SIDES, FPS, STRIDE = ((34340, 34370), (35838, 35854)), ("1P", "2P"), 60, 2
FORMAT = "video38-current-entry-observation/v1"
SIDECAR, REPORT, COMPARISON = "current_entry.jsonl", "CURRENT_ENTRY_RECEIPT.json", "CURRENT_ENTRY_COMPARISON.json"
ENGINE, STATUS = "CURRENT_ENTRY_ENGINE.json", "CURRENT_ENTRY_STATUS.json"
ENGINE_NAMES = previous.ENGINE_NAMES | {"ENGINE_COMPLETE", SIDECAR, REPORT, COMPARISON}
SUFFIXES = ("step_enter", "detector_return", "sm_return", "step_return")
ORIGINAL_PREPARE, ORIGINAL_INSTRUMENT, ORIGINAL_FINISH = previous.prepare, previous.instrument, previous.finish


def selected(frame: int) -> bool:
    """事前固定窓だけを選ぶ。"""
    return any(first <= frame <= last for first, last in WINDOWS)


def expected_scopes() -> list[tuple[int, str]]:
    """全stepの順序を固定し、実測後には変えない。"""
    return [(frame, side) for first, last in WINDOWS for frame in range(first, last + 1, STRIDE) for side in SIDES]


def additional_guards(args: argparse.Namespace) -> dict[str, str]:
    """旧3SHAと新3SHAを別pathで結合する。"""
    paths = (Path(previous.__file__), previous.TEST, previous.LAUNCHER)
    hashes = {str(path.resolve()): digest for path, digest in zip(paths, OLD_SHA, strict=True)}
    hashes.update({str(path): digest for path, digest in zip((OBSERVER, OBSERVER_TEST), HOOK_SHA, strict=True)})
    hashes[str(PLAN)] = PLAN_SHA
    for name, path in (("script", Path(__file__)), ("test", HOME / "test_adapter.py"), ("launcher", HOME / "launcher.sh")):
        digest = getattr(args, name + "_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("新adapterの検収済み3SHAが必要です")
        hashes[str(path.resolve())] = digest
    base.assert_unchanged(hashes)
    return hashes


def reference_guards() -> dict[str, str]:
    """成功した同NEXT runの実7artifactだけを対照にする。"""
    hashes = {str(REFERENCE / "COMPLETE"): REFERENCE_SHA}
    base.assert_unchanged(hashes)
    complete = base.read_json(REFERENCE / "COMPLETE")
    if complete.get("format_version") != previous.FORMAT or type(complete.get("child_exit_code")) is not int:
        raise ValueError("元NEXT完了形式が違います")
    if complete["child_exit_code"] != 0 or set(complete["sha256"]) != previous.ENGINE_NAMES | {"ENGINE_COMPLETE", "CHILD_EXIT.json"}:
        raise ValueError("元NEXT実完了契約が違います")
    hashes.update({str(REFERENCE / name): digest for name, digest in complete["sha256"].items()})
    base.assert_unchanged(hashes)
    return hashes


def legacy_arguments(args: argparse.Namespace) -> argparse.Namespace:
    """元runnerへは元runner自身のSHAを渡す。"""
    value = argparse.Namespace(**vars(args))
    for name, digest in zip(("script", "test", "launcher"), OLD_SHA, strict=True):
        setattr(value, name + "_sha256", digest)
    return value


def prepare(args: argparse.Namespace, new_args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """元prepareを一回使い、追加scopeと固定依存を保存する。"""
    hashes = {**additional_guards(new_args), **reference_guards()}
    receipt, config = ORIGINAL_PREPARE(args)
    receipt["input_and_code_sha256"].update(hashes)
    base.assert_unchanged(receipt["input_and_code_sha256"])
    receipt["current_entry_observation"] = {"format_version": FORMAT, "windows": WINDOWS,
        "expected_step_scopes": len(expected_scopes()), "reference_root": str(REFERENCE),
        "same_run_reference_complete_sha256": REFERENCE_SHA, "separate_sidecar": SIDECAR,
        "legacy_runner_sha256": OLD_SHA[0], "observer_sha256": HOOK_SHA[0],
        "current_publication_allowed": False, "accounting_commit_allowed": False}
    return receipt, config


class Sidecar:
    """run私有sink。失敗は外側catchがあっても消さない。"""
    def __init__(self, path: Path, rec: Any) -> None:
        self.path, self.rec = path, rec
        self.stream = path.open("x", encoding="utf-8")
        self.error: dict[str, str] | None = None
        self.rows: list[dict[str, Any]] = []
        self.closed = False
        self.runtime: dict[str, Any] = {}

    def failed(self, exc: BaseException, location: str) -> None:
        """最初の失敗を保持し、後続成功でリセットしない。"""
        if self.error is None:
            self.error = {"type": type(exc).__name__, "message": str(exc), "location": location}

    def epoch(self, row: dict[str, Any]) -> int | None:
        """同invocationのsoftware epochだけを読み、欠測はNone。"""
        controller = self.rec.next_enqueue_controller
        inv = controller.active
        if inv is None:
            return None
        if (inv.frame, inv.time_sec) != (row["frame_idx"], row["time_sec"]):
            raise RuntimeError("観測invocation clock不一致")
        value = inv.runtime.histories[row["side"]].epoch
        if type(value) is not int or value < 0:
            raise ValueError("観測epoch型不正")
        return value

    def emit(self, row: dict[str, Any]) -> None:
        """厳密JSONとして同時刻値をdetachし、別streamにだけ書く。"""
        try:
            frame, clock = row["frame_idx"], row["time_sec"]
            if type(frame) is not int or not selected(frame) or frame % STRIDE:
                raise ValueError("追加観測frame不正")
            if isinstance(clock, bool) or not isinstance(clock, (int, float)) or not math.isfinite(clock) or clock != frame / FPS:
                raise ValueError("追加観測time不正")
            if row.get("side") not in SIDES or row.get("kind") not in {"current_entry_" + suffix for suffix in SUFFIXES}:
                raise ValueError("追加観測side/kind不正")
            value = {**row, "software_epoch": self.epoch(row), "physical_hand_certified": False}
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
            self.stream.write(encoded + "\n")
            self.rows.append(json.loads(encoded))
        except BaseException as exc:
            self.failed(exc, "sink")
            raise

    def close(self) -> None:
        """stack終了時に全行をflushし、失敗もsticky化する。"""
        try:
            self.stream.flush()
            os.fsync(self.stream.fileno())
        except BaseException as exc:
            self.failed(exc, "close")
            raise
        finally:
            self.stream.close()
            self.closed = True

    def summary(self) -> dict[str, Any]:
        """全stepを要求し、SM未呼出を正常呼出へ補完しない。"""
        self.require_ok()
        entered, returned, counts = [], [], {}
        for row in self.rows:
            key, kind = (row["frame_idx"], row["side"]), row["kind"].removeprefix("current_entry_")
            current = counts.setdefault(key, {suffix: 0 for suffix in SUFFIXES})
            if kind == "step_enter":
                entered.append(key)
            elif not current["step_enter"] or current["step_return"]:
                raise ValueError("追加観測のstep内順序違反")
            current[kind] += 1
            if kind == "step_return":
                returned.append(key)
            if any(current[name] > 1 for name in ("step_enter", "sm_return", "step_return")):
                raise ValueError("追加観測が重複しています")
        if entered != expected_scopes() or returned != entered:
            raise ValueError("追加観測step coverage不足/順序違反")
        return {"row_count": len(self.rows), "step_scopes": len(entered),
                "missing_sm_return": [list(key) for key, value in counts.items() if not value["sm_return"]],
                "counts": [{"frame": key[0], "side": key[1], **value} for key, value in counts.items()],
                "runtime": self.runtime, "software_epoch_missing_rows": sum(r["software_epoch"] is None for r in self.rows),
                "current_publication_allowed": False, "accounting_commit_allowed": False}

    def require_ok(self) -> None:
        """観測失敗または未flushを完了させない。"""
        if self.error is not None or not self.closed:
            raise RuntimeError(f"追加観測が未完了です: {self.error}")


def sticky(original: Callable[..., Any], sink: Sidecar, name: str) -> Callable[..., Any]:
    """観測wrapper例外を外側pipelineがcatchしても完了を拒否する。"""
    @functools.wraps(original)
    def call(*args: Any, **kwargs: Any) -> Any:
        try:
            if name == "_step_side" and selected(sink.rec.frame):
                inv = sink.rec.next_enqueue_controller.active
                if inv is not None and (not args or args[0] is not inv.runtime.pipe):
                    raise RuntimeError("別pipelineのstepを実invocationへ結合しません")
            return original(*args, **kwargs)
        except BaseException as exc:
            if selected(sink.rec.frame):
                sink.failed(exc, name)
            raise
    return call


def load_hook(stack: contextlib.ExitStack) -> Any:
    """想定外既loadを拒否し、固有aliasを完全復元する。"""
    if HOOK_NAME in sys.modules:
        raise RuntimeError("新observer aliasが既に存在します")
    spec = importlib.util.spec_from_file_location(HOOK_NAME, OBSERVER)
    if spec is None or spec.loader is None:
        raise ImportError("固定observerをロードできません")
    module = importlib.util.module_from_spec(spec)
    sys.modules[HOOK_NAME] = module
    stack.callback(sys.modules.pop, HOOK_NAME, None)
    spec.loader.exec_module(module)
    return module


def instrument(stack: contextlib.ExitStack, collector: Any, rec: Any, receipt: dict[str, Any], state: dict[str, Any]) -> None:
    """既存全計装の後に、新窓のsidecarだけを接続する。"""
    base.assert_unchanged(receipt["input_and_code_sha256"])
    ORIGINAL_INSTRUMENT(stack, collector, rec, receipt)
    boundary = sys.modules.get(BOUNDARY_NAME)
    if boundary is None or Path(boundary.__file__).resolve() != previous.runner.observed.END_MODULE.resolve():
        raise RuntimeError("旧BoundaryObserver実moduleがありません")
    runtime = previous.history.frozen_modules()
    for path in runtime.values():
        if str(Path(path).resolve()) not in receipt["input_and_code_sha256"]:
            raise RuntimeError("実runtimeの開始guardがありません")
    sink = Sidecar(state["output"] / SIDECAR, rec)
    state["sink"] = sink
    stack.callback(sink.close)
    hook = load_hook(stack)
    sm = sys.modules["src.board_state_machine"].BoardStateMachine
    hook.install(stack, collector, rec, boundary, sm, sink.emit)
    from src import state_detectors
    targets = [(collector.RecognitionPipeline, "_step_side"), (sm, "update")]
    targets += [(getattr(state_detectors, name), "detect") for name in boundary.DETECTORS]
    for owner, name in targets:
        base.patch(stack, owner, name, sticky(getattr(owner, name), sink, name))
    sink.runtime = {"src": {name: {"path": path, "sha256": base.sha256(Path(path))} for name, path in runtime.items()},
                    "observer": {"path": str(OBSERVER), "sha256": base.sha256(OBSERVER)},
                    "boundary": {"path": boundary.__file__, "sha256": base.sha256(Path(boundary.__file__))}}


def compare_all(reference: Path, candidate: Path) -> dict[str, Any]:
    """新sidecarを除外する必要なく、元JSONL全行を全文順序比較する。"""
    guards = {str(path): base.sha256(path) for path in (reference, candidate)}
    differences, old_count, new_count = [], 0, 0
    with reference.open(encoding="utf-8") as old, candidate.open(encoding="utf-8") as new:
        for index, (before, after) in enumerate(itertools.zip_longest(old, new), 1):
            old_count += before is not None
            new_count += after is not None
            if before != after:
                differences.append({"line": index, "before": before, "after": after})
    base.assert_unchanged(guards)
    return {"input_sha256": guards, "reference_rows": old_count, "candidate_rows": new_count,
            "all_text_and_order_equal": not differences, "differences": differences}


def finish(output: Path, receipt: dict[str, Any], rec: Any, elapsed: float, state: dict[str, Any]) -> dict[str, Any]:
    """旧engine保存を再利用し、最終COMPLETEは実child終了まで作らない。"""
    sink = state["sink"]
    report = sink.summary()
    comparison = compare_all(REFERENCE / "frames.jsonl", output / "frames.jsonl")
    base.write_json(output / COMPARISON, comparison)
    if not comparison["all_text_and_order_equal"]:
        raise ValueError("元NEXT全行に差分があります")
    result = ORIGINAL_FINISH(output, receipt, rec, elapsed)
    base.write_json(output / REPORT, report)
    engine = base.read_json(output / "ENGINE_COMPLETE")
    if engine.get("format_version") != previous.FORMAT or set(engine["sha256"]) != previous.ENGINE_NAMES:
        raise ValueError("旧NEXT engineが不正です")
    hashes = {**engine["sha256"], **{name: base.sha256(output / name) for name in ENGINE_NAMES - previous.ENGINE_NAMES}}
    base.assert_unchanged(receipt["input_and_code_sha256"])
    base.assert_unchanged({str(output / name): digest for name, digest in hashes.items()})
    sink.require_ok()
    base.write_json(output / ENGINE, {"format_version": FORMAT, "sha256": hashes, "awaiting_actual_child_exit": True})
    return {**result, "current_entry_rows": len(sink.rows), "quality_gate_clear": False}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """元NEXT run一回だけを一時配線し、失敗状態も別receiptへ残す。"""
    state: dict[str, Any] = {"output": args.output_root.resolve(), "sink": None}
    def prepared(value: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
        return prepare(value, args)
    def installed(stack: contextlib.ExitStack, collector: Any, rec: Any, receipt: dict[str, Any]) -> None:
        instrument(stack, collector, rec, receipt, state)
    def finished(output: Path, receipt: dict[str, Any], rec: Any, elapsed: float) -> dict[str, Any]:
        return finish(output, receipt, rec, elapsed, state)
    try:
        with contextlib.ExitStack() as stack:
            for name, value in (("prepare", prepared), ("instrument", installed), ("finish", finished)):
                base.patch(stack, previous, name, value)
            return previous.run(legacy_arguments(args))
    finally:
        sink = state["sink"]
        if sink is not None:
            base.write_json(state["output"] / STATUS, {"sticky_error": sink.error, "closed": sink.closed,
                            "row_count": len(sink.rows), "format_version": FORMAT})


def finalize(output: Path, child_exit: int) -> dict[str, Any]:
    """旧finalizeは呼ばず、実exitと全sidecar/guardを合わせて最後に完了する。"""
    if type(child_exit) is not int or not 0 <= child_exit <= 255:
        raise ValueError("実child終了コードが必要です")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "COMPLETE").exists() or (output / "CHILD_EXIT.json").exists():
        raise FileExistsError("終了receiptを上書きしません")
    base.write_json(output / "CHILD_EXIT.json", {"child_exit_code": child_exit, "source": "launcher_waited_actual_process"})
    if child_exit:
        return {"status": "child_failed", "child_exit_code": child_exit}
    engine, status = base.read_json(output / ENGINE), base.read_json(output / STATUS)
    if engine.get("format_version") != FORMAT or set(engine["sha256"]) != ENGINE_NAMES:
        raise ValueError("追加engineのartifact契約不正")
    if status.get("sticky_error") is not None or status.get("closed") is not True:
        raise ValueError("観測エラー/未closeを完了へ昇格しません")
    receipt = base.read_json(output / "PLAN.json")
    hashes = {**engine["sha256"], **{name: base.sha256(output / name) for name in (ENGINE, STATUS, "CHILD_EXIT.json")}}
    base.assert_unchanged({str(output / name): digest for name, digest in hashes.items()})
    base.assert_unchanged(receipt["input_and_code_sha256"])
    complete = {"format_version": FORMAT, "child_exit_code": child_exit, "sha256": hashes,
                "status": "observation_complete_not_repaired", "quality_gate_clear": False,
                "current_publication_allowed": False, "accounting_commit_allowed": False}
    base.write_json(output / "COMPLETE", complete)
    return complete


def main() -> int:
    """CPU/配線独立QA後だけ使用する新診断入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--finalize-child-exit", type=int)
    for name in ("script", "test", "launcher"):
        parser.add_argument("--" + name + "-sha256")
    parser.add_argument("--allow-native-runtime-mismatch", action="store_true")
    args = parser.parse_args()
    if args.finalize_child_exit is not None:
        print(json.dumps(finalize(args.output_root.resolve(), args.finalize_child_exit), ensure_ascii=False))
        return 0
    import cv2
    cv2.setNumThreads(previous.history.THREADS)
    previous.runner.pending.completion._configure_torch_threads()
    args.mode, (args.start_sec, args.end_sec) = "start_epoch", previous.runner.INTERVALS["start_epoch"]
    args.module_sha256 = previous.initial.FIXED_SHA["scripts/match_start_epoch_shadow_v1.py"]
    args.module_test_sha256 = previous.initial.FIXED_SHA["tests/test_match_start_epoch_shadow_v1.py"]
    print(json.dumps(run(args), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
