"""NEXT動画runnerのCPU接続検収。実動画/GPUの品質合格ではない。"""
from __future__ import annotations

import contextlib
import ast
import importlib.util
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import diagnose_video38_next_enqueue_live_shadow_v1 as subject


@pytest.fixture(scope="module")
def frozen() -> Any:
    saved, paths, directory = dict(sys.modules), list(sys.path), Path.cwd()
    os.chdir(subject.base.SNAPSHOT)
    collector = subject.base.load_collector()
    try:
        yield collector
    finally:
        os.chdir(directory)
        sys.path[:] = paths
        for name in list(sys.modules):
            if name == "src" or name.startswith("src."):
                sys.modules.pop(name)
        sys.modules.update({name: value for name, value in saved.items()
                            if name == "src" or name.startswith("src.")})


def test_real_property_value_is_supported_after_p1_fix(frozen: Any) -> None:
    """旧c2de2e33では一度ValueErrorを実測。新実型契約は同値propertyを受理。"""
    from src.next_detector import NextDetectionBothResult, NextDetectionResult
    from src.next_slide_detector import SlideMotionResult
    value = NextDetectionBothResult(NextDetectionResult(5, 5, 2, 3), NextDetectionResult(5, 5, 2, 3))
    pair = value.p1.next_pair
    assert pair == value.p1.next_pair and pair is not value.p1.next_pair
    inv = SimpleNamespace(main=value, slides={"1P": SlideMotionResult(False, 1.0, 8.0)})
    controller = subject.live.NextEnqueueController(frozen.RecognitionPipeline, None, {})
    assert controller._quiet(inv, "1P", pair)[0] is True


def helper() -> Any:
    """凍結済みcomponentのCPU reader/実DTO fixtureだけ再利用する。"""
    path = subject.base.ROOT / "tests/test_next_enqueue_live_shadow_v1.py"
    assert subject.base.sha256(path) == subject.LIVE_TEST_SHA
    spec = importlib.util.spec_from_file_location("_next_live_test_reuse", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def receipt_fixture() -> dict[str, Any]:
    guards = subject.base.read_json(subject.initial.REFERENCE / "PLAN.json")["input_and_code_sha256"]
    guards.update(subject.fixed_guards())
    guards.update({str(subject.base.ROOT / path): digest for path, digest in subject.runner.START_EVIDENCE.items()})
    return {"input_and_code_sha256": guards, "boundary_repair": {"mode": "start_epoch",
        "module": str(subject.base.ROOT / "scripts/match_start_epoch_shadow_v1.py"),
        "module_sha256": subject.initial.FIXED_SHA["scripts/match_start_epoch_shadow_v1.py"]}}


@pytest.fixture
def real(frozen: Any, monkeypatch: Any) -> Any:
    """実frozen update/全wrapper、人工画像返却。モデル推論は実行しない。"""
    import numpy as np
    from src.board import Board
    from src.match_state import MatchState
    from src.next_detector import NextDetector
    from src.next_slide_detector import NextSlideDetector, SlideMotionResult
    helpers, stream = helper(), io.StringIO()
    rec = subject.history.HistoryRecorder(stream, subject.base.board_value(Board()))
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    main = object.__new__(NextDetector)
    source = {"pair": (5, 5), "diff": 1.0, "calls": Counter(), "fail": None}
    def detect(detector: Any, frame: Any) -> Any:
        source["calls"]["next"] += 1
        if source["fail"] == "next":
            raise RuntimeError("actual_main_failure")
        return helpers.actual_next(source["pair"])
    def slide(detector: Any, previous: Any, current: Any) -> Any:
        source["calls"]["slide"] += 1
        if source["fail"] == "slide":
            raise RuntimeError("actual_slide_failure")
        return SlideMotionResult(False, source["diff"], 8.0)
    monkeypatch.setattr(NextDetector, "detect_both", detect)
    monkeypatch.setattr(NextSlideDetector, "update", slide)
    pipe = frozen.RecognitionPipeline(image_reader=helpers.CpuReader(Board()),
        match_state_detector=helpers.CpuMatch(MatchState.IN_MATCH), score_ocr=None,
        chain_tracker_1p=None, chain_tracker_2p=None, next_detector=main, stable_frame_count=2)
    pipe._prev_frame = image.copy()
    originals = {name: inspect_static(type(pipe), name) for name in ("update", "reset", "load_default", "_step_side")}
    with contextlib.ExitStack() as stack:
        subject.instrument(stack, frozen, rec, receipt_fixture())
        yield pipe, rec, image, source, stream
    assert all(inspect_static(type(pipe), name) is value for name, value in originals.items())
    assert "__next_live" not in originals["update"].__globals__


def inspect_static(owner: Any, name: str) -> Any:
    import inspect
    return inspect.getattr_static(owner, name)


def drive(real: Any, frame: int, pair: Any = (5, 5), diff: float = 1.0) -> Any:
    pipe, rec, image, source, _ = real
    rec.frames = (frame - subject.history.FIRST_FRAME) // subject.history.STRIDE
    rec.decoded_frame = frame
    source.update(pair=pair, diff=diff)
    return pipe.update(frame, frame / subject.history.FPS, image)


def recorded(real: Any) -> list[dict[str, Any]]:
    return [json.loads(line) for line in real[-1].getvalue().splitlines()]


def test_actual_all_instrument_closure_calls_once_and_after_helper(real: Any, monkeypatch: Any) -> None:
    from src import recognition_pipeline as module
    pipe, rec, _, source, _ = real
    controller = rec.next_enqueue_controller
    original = module._is_score_reset_boundary
    calls = []
    def helper_call(*args: Any, **kwargs: Any) -> Any:
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "_is_score_reset_boundary", helper_call)
    pipe._enable_match_start_full_clear = True
    drive(real, 32696)
    drive(real, 32698, (2, 3), 63.0)
    drive(real, 32700, (3, 5))
    assert source["calls"] == Counter(next=3, slide=6) and calls
    assert controller.active is None and controller.instances[id(pipe)].completed == 32700
    rows = recorded(real)
    extra = [row for row in rows if row["kind"] == subject.PREFIX + "invocation"]
    assert len(extra) == 3 and all(row["raw"]["same_return_identity"] for row in extra)
    assert pipe._last_seen_next_1p == (3, 5)
    assert all(row["caller"]["source"] == str(subject.live.BASE.PIPELINE)
               for row in rows if row["kind"] == "start_gate_latch")


@pytest.mark.parametrize("failure,expected", [("next", (1, 0)), ("slide", (1, 2))])
def test_actual_detector_exception_preserved_not_quiet(real: Any, failure: str, expected: Any) -> None:
    real[3]["fail"] = failure
    drive(real, 32696)
    raw = next(row["raw"] for row in recorded(real) if row["kind"] == subject.PREFIX + "invocation")
    assert (raw["main"]["count"], sum(value["count"] for value in raw["slides"].values())) == expected
    assert raw["main" if failure == "next" else "slides"]["exception" if failure == "next" else "1P"] is not None
    assert all(not row["quiet"] for row in recorded(real) if row["kind"] == subject.PREFIX + "decision")


def test_serialization_failure_is_outside_recognition_catch(real: Any, monkeypatch: Any) -> None:
    def fail(value: Any) -> Any:
        raise RuntimeError("serializer_outside_catch")
    monkeypatch.setattr(subject.initial, "pairs", fail)
    with pytest.raises(RuntimeError, match="serializer_outside_catch"):
        drive(real, 32696)
    controller = real[1].next_enqueue_controller
    assert controller.active is None and controller.instances[id(real[0])].completed is None
    assert not real[0]._pending_tsumo_1p


def test_direct_next_config_actual_class_cpu_device(real: Any) -> None:
    from src.patch_classifier import CnnPatchClassifier
    classifier = CnnPatchClassifier()
    real[0]._next_detector._classifier = classifier
    value = subject.detector_config(real[0], require_cuda=False)
    assert value["classifier_layout"] == "direct_cnn" and value["parameter_device"] == "cpu"
    with pytest.raises(RuntimeError, match="cuda"):
        subject.detector_config(real[0])


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def legacy(frame: int, *, kind: str = "frame_side", **fields: Any) -> dict[str, Any]:
    return {"kind": kind, "frame_idx": frame, "time_sec": frame / 60, "side": "1P", **fields}


def comparison_fixture(tmp_path: Path, monkeypatch: Any, old: Any, new: Any) -> Any:
    left, right = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    write_rows(left, old)
    write_rows(right, new)
    monkeypatch.setattr(subject.initial, "EXPECTED_LEGACY_ROWS", len(old))
    monkeypatch.setattr(subject.runner, "coverage", lambda *args: {"fixture_only": True})
    return subject.compare_rows(left, right)


def test_partition_keeps_background_and_occurrences(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [legacy(32494, kind="boundary_repair_mutation"), legacy(32698), legacy(32698),
            legacy(32698, kind=subject.PREFIX + "decision")]
    write_rows(path, rows)
    normal, extra = subject.read_rows(path)
    assert len(normal) == 3 and len(extra) == 1
    assert [json.loads(key)[-1] for key in normal] == [0, 0, 1]


@pytest.mark.parametrize("kind", ["initial_pair_result", subject.PREFIX + "unknown"])
def test_partition_rejects_unregistered_prefix(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "rows.jsonl"
    write_rows(path, [legacy(32698, kind=kind)])
    with pytest.raises(ValueError):
        subject.read_rows(path)


def test_all_missing_added_changed_include_full_values(tmp_path: Path, monkeypatch: Any) -> None:
    old = [legacy(32696), legacy(32698), legacy(32700, tag="missing")]
    new = [legacy(32696), legacy(32698, optional=None), legacy(32702, tag="added")]
    result, _ = comparison_fixture(tmp_path, monkeypatch, old, new)
    field = next(iter(result["changed_fields"].values()))["optional"]
    assert field == dict(before_present=False, after_present=True, before=None, after=None)
    assert next(iter(result["missing_rows"].values()))["row"]["tag"] == "missing"
    assert next(iter(result["added_rows"].values()))["row"]["tag"] == "added"
    assert result["prefix_bit_exact"] and not result["quality_gate_clear"]


@pytest.mark.parametrize("failure", ["value", "order", "missing"])
def test_fixed_prefix_failure_never_moves_cutoff(tmp_path: Path, monkeypatch: Any, failure: str) -> None:
    old = [legacy(32694), legacy(32696)]
    new = [legacy(32694, changed=True), legacy(32696)] if failure == "value" else list(reversed(old))
    if failure == "missing":
        new = old[:1]
    with pytest.raises(ValueError, match="prefix"):
        comparison_fixture(tmp_path, monkeypatch, old, new)


def minimal_extra(frame: int) -> list[dict[str, Any]]:
    raw = {"main": dict(count=0, exception=None, returned=None),
           "slides": {side: dict(count=0, exception=None, returned=None) for side in subject.history.SIDES},
           "same_return_identity": True, "extra_detector_calls": 0}
    result = [legacy(frame, kind=subject.PREFIX + "invocation", raw=raw)]
    for kind in ("decision", "accounting"):
        result.extend({**legacy(frame, kind=subject.PREFIX + kind), "side": side} for side in subject.history.SIDES)
    return result


@pytest.mark.parametrize("failure", ["missing", "duplicate", "clock", "bool", "calls"])
def test_observation_rejects_bad_coverage(monkeypatch: Any, failure: str) -> None:
    frame = subject.live.FIRST_FRAME
    monkeypatch.setattr(subject.live, "LAST_FRAME", frame)
    monkeypatch.setattr(subject, "OBS_FIRST", frame)
    rows = minimal_extra(frame)
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows.append(rows[-1])
    elif failure in ("clock", "bool"):
        rows[0]["time_sec"] = True if failure == "bool" else 0.0
    else:
        rows[0]["raw"]["main"]["count"] = 2
    with pytest.raises(ValueError):
        subject.observation_summary(rows)


def test_observation_missing_calls_not_ground_truth(monkeypatch: Any) -> None:
    frame = subject.live.FIRST_FRAME
    monkeypatch.setattr(subject.live, "LAST_FRAME", frame)
    monkeypatch.setattr(subject, "OBS_FIRST", frame)
    value = subject.observation_summary(minimal_extra(frame))
    assert value["actual_calls"]["main_calls"] == 0
    assert not value["initial_yellow_pair_repaired"] and not value["accounting_basis_verified"]


def test_raw_reference_missing_saved_without_rescue(tmp_path: Path) -> None:
    path = tmp_path / "reference.jsonl"
    write_rows(path, [{**legacy(32698, kind="initial_pair_main_next"), "side": None,
                       "returned": {"value": 5}, "exception": None}])
    value = subject.raw_comparison(minimal_extra(32698), path)
    assert value["missing"] and not value["all_common_returns_equal"]
    assert value["later_reference_raw_unobserved"]


def engine_fixture(root: Path) -> dict[str, Any]:
    root.mkdir()
    for name in subject.ENGINE_NAMES:
        subject.base.write_json(root / name, {"input_and_code_sha256": {}} if name == "PLAN.json" else {})
    value = {"format_version": subject.FORMAT,
             "sha256": {name: subject.base.sha256(root / name) for name in subject.ENGINE_NAMES}}
    subject.base.write_json(root / "ENGINE_COMPLETE", value)
    return value


def test_finalize_actual_exit_zero_seven_artifacts_exclusive(tmp_path: Path) -> None:
    root = tmp_path / "output"
    engine_fixture(root)
    value = subject.finalize(root, 0)
    assert len(value["sha256"]) == 7 and not value["quality_gate_clear"]
    with pytest.raises(FileExistsError):
        subject.finalize(root, 0)


@pytest.mark.parametrize("code", [1, 2, 137, 143])
def test_real_failed_exit_never_complete(tmp_path: Path, code: int) -> None:
    root = tmp_path / "failed"
    assert subject.finalize(root, code)["child_exit_code"] == code
    assert (root / "CHILD_EXIT.json").exists() and not (root / "COMPLETE").exists()


@pytest.mark.parametrize("code", [None, True, -1, 256])
def test_unknown_exit_refused(tmp_path: Path, code: Any) -> None:
    with pytest.raises(ValueError):
        subject.finalize(tmp_path / "bad", code)


def test_exit_zero_missing_engine_not_complete(tmp_path: Path) -> None:
    root = tmp_path / "no_engine"
    with pytest.raises(FileNotFoundError):
        subject.finalize(root, 0)
    assert not (root / "COMPLETE").exists()


def test_finalize_detects_artifact_changed(tmp_path: Path) -> None:
    root = tmp_path / "changed"
    engine_fixture(root)
    (root / "frames.jsonl").unlink()
    subject.base.write_json(root / "frames.jsonl", {"changed": True})
    with pytest.raises(ValueError):
        subject.finalize(root, 0)
    assert not (root / "COMPLETE").exists()


def test_run_exception_restores_original_functions(monkeypatch: Any, tmp_path: Path) -> None:
    names = ("prepare", "instrument_pipeline", "finish")
    originals = {name: getattr(subject.history, name) for name in names}
    def fail(args: Any) -> Any:
        assert all(getattr(subject.history, name) is not value for name, value in originals.items())
        raise RuntimeError("fake_collect_failure")
    monkeypatch.setattr(subject.history, "run", fail)
    with pytest.raises(RuntimeError, match="fake_collect"):
        subject.run(SimpleNamespace(output_root=tmp_path / "absent"))
    assert all(getattr(subject.history, name) is value for name, value in originals.items())


def test_no_grace_import_and_function_lengths() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "import diagnose_video38_post_chain_grace" not in source
    for path in (Path(subject.__file__), Path(__file__)):
        assert all(node.end_lineno - node.lineno + 1 <= 50 for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    assert subject.FIRST_DIFFERENCE_FRAME == 32698 and subject.EXPECTED_UPDATES == 1903


def arguments(output: Path) -> Any:
    return SimpleNamespace(output_root=output, mode="start_epoch", start_sec=484.2, end_sec=605.0,
        module_sha256=subject.initial.FIXED_SHA["scripts/match_start_epoch_shadow_v1.py"],
        module_test_sha256=subject.initial.FIXED_SHA["tests/test_match_start_epoch_shadow_v1.py"],
        script_sha256=subject.base.sha256(Path(subject.__file__)), test_sha256=subject.base.sha256(Path(__file__)),
        launcher_sha256=subject.base.sha256(subject.LAUNCHER), allow_native_runtime_mismatch=True)


def test_prepare_adds_all_three_files_and_dependencies(tmp_path: Path, monkeypatch: Any) -> None:
    calls = []
    def prior(args: Any) -> Any:
        calls.append(args)
        return receipt_fixture(), {"fixture_only": True}
    monkeypatch.setattr(subject.runner, "prepare", prior)
    monkeypatch.setattr(subject.initial, "reference_guards", lambda: {})
    monkeypatch.setattr(subject, "supplement_guards", lambda: {})
    args = arguments(tmp_path / "output")
    receipt, config = subject.prepare(args)
    assert calls == [args] and config["fixture_only"]
    guards = receipt["input_and_code_sha256"]
    assert all(str(path.resolve()) in guards for path in (Path(subject.__file__), Path(__file__), subject.LAUNCHER))
    assert guards[str(Path(subject.live.__file__).resolve())] == subject.LIVE_SHA
    assert receipt["next_enqueue_live"]["history_frame_range"] == [29052, 36298]


@pytest.mark.parametrize("field", ["module_sha256", "module_test_sha256", "script_sha256"])
def test_prepare_refuses_wrong_fixed_sha(tmp_path: Path, field: str) -> None:
    args = arguments(tmp_path / "output")
    setattr(args, field, "0" * 64)
    with pytest.raises(ValueError):
        subject.prepare(args)


def test_original_history_finish_five_artifacts_and_child_boundary(tmp_path: Path, monkeypatch: Any) -> None:
    root = tmp_path / "finish"
    root.mkdir()
    receipt = {"input_and_code_sha256": {str(Path(subject.__file__)): subject.base.sha256(Path(subject.__file__))}}
    subject.base.write_json(root / "PLAN.json", receipt)
    write_rows(root / "frames.jsonl", [legacy(32698)])
    guard = {str(root / "frames.jsonl"): subject.base.sha256(root / "frames.jsonl")}
    monkeypatch.setattr(subject, "compare_rows", lambda *args: ({"input_sha256": guard}, []))
    monkeypatch.setattr(subject, "observation_summary", lambda rows: {"cpu_fixture_only": True})
    monkeypatch.setattr(subject, "raw_comparison", lambda *args: {"cpu_fixture_only": True})
    rec = SimpleNamespace(frames=subject.history.EXPECTED_FRAMES, rows=1, frame=36298, snapshots=0,
                          model_loads=["cpu_fixture_not_actual_model_load"], pipeline_receipt={"cpu_fixture_only": True})
    result = subject.finish(root, receipt, rec, 1.0)
    assert result["status"] == "awaiting_real_child_exit" and not (root / "COMPLETE").exists()
    engine = subject.base.read_json(root / "ENGINE_COMPLETE")
    assert set(engine["sha256"]) == subject.ENGINE_NAMES
    complete = subject.finalize(root, 0)
    assert len(complete["sha256"]) == 7
