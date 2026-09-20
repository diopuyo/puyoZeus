"""初手追加観測の主履歴非干渉・厳密比較・凍結保存をCPUで検査する。"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import json
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import diagnose_video38_initial_pair_observation_v1 as subject


class FakeModel:
    """重み非ロードの共有tensor世代fixture。"""
    def __init__(self) -> None:
        self.training = False
        self.version = 0
    def named_modules(self) -> list[Any]:
        return [("", self)]
    def named_parameters(self) -> list[Any]:
        return [("weight", self)]
    def named_buffers(self) -> list[Any]:
        return []
    @property
    def _version(self) -> int:
        return self.version
    def data_ptr(self) -> int:
        return 123


class FakeColor:
    """classifier自身の可変cacheが主側へ戻らないことも調べる。"""
    def __init__(self) -> None:
        self._model, self._device, self.calls = FakeModel(), "cpu", 0
    def classify(self, patch: Any) -> int:
        self.calls += 1
        return 5


class FakeDetector:
    """固定NextDetectorと同じ二fieldだけを持つ人工検出器。"""
    calls: list[Any] = []
    def __init__(self, classifier: Any, centroid_classifier: Any = None) -> None:
        self._classifier, self._centroid = classifier, centroid_classifier
    def detect_both(self, frame: Any) -> Any:
        type(self).calls.append(self)
        self._classifier._color.classify(frame)
        value = SimpleNamespace(next_pair=(5, 5), dnext_pair=(3, 5))
        return SimpleNamespace(p1=value, p2=value)


class FakeSlide:
    """cooldown中もdiffが大きいという既存反例を保存する。"""
    def __init__(self) -> None:
        self._cooldown, self._diff_history, self._last_diff_score = 6, deque([20.0]), 20.0
        self._diff_threshold, self._adaptive_k, self._cooldown_frames = 8.0, 2.0, 6
        self.calls = 0
    def update(self, previous: Any, current: Any) -> Any:
        self.calls += 1
        self._cooldown -= 1
        return SimpleNamespace(slide_motion=False, diff_score=20.0, threshold_used=8.0)


class FakeRecorder:
    """実frame位置を明示する記録専用fixture。"""
    def __init__(self, frame: int = subject.FIRST_FRAME) -> None:
        self.frame, self.rows = frame, []
    def emit(self, row: dict[str, Any]) -> None:
        self.rows.append({"frame_idx": self.frame, "time_sec": self.frame / 60, **copy.deepcopy(row)})


def pipeline() -> Any:
    """pending空・Counter空を変えてはならない対象を作る。"""
    color = SimpleNamespace(_color=FakeColor(), _gate={}, _strict_gate={}, _ui_matcher={})
    result = SimpleNamespace(_next_detector=FakeDetector(color, {"centroids": [1, 2]}))
    for side in subject.history.SIDES:
        for name in subject.history.FIELDS:
            value = Counter() if name == "tsumo_count" else deque() if name == "pending_tsumo" else None
            setattr(result, f"_{name}_{side.lower()}", value)
        setattr(result, "_slide_detector_" + side.lower(), FakeSlide())
    return result


@pytest.fixture(autouse=True)
def reset_calls() -> None:
    """各testの主/独立呼出数を混ぜない。"""
    FakeDetector.calls = []


def test_independent_detector_separates_all_mutable_wrappers() -> None:
    main = pipeline()._next_detector
    clone, model, counts = subject.independent_detector(main, FakeDetector)
    assert clone is not main and clone._classifier is not main._classifier
    assert clone._classifier._color is not main._classifier._color
    assert model is main._classifier._color._model
    assert clone._centroid is not main._centroid
    for name in ("_gate", "_strict_gate", "_ui_matcher"):
        assert getattr(clone._classifier, name) is not getattr(main._classifier, name)
    assert counts == {"cnn_patch_calls": 0}


def test_inactive_read_never_calls_main_or_populates_pending() -> None:
    value, frame, cache = pipeline(), subject.history.np.zeros((2, 2)), {}
    accounts = {side: subject.history.accounting_snapshot(value, side) for side in subject.history.SIDES}
    result = subject.inactive_read(value, frame, FakeDetector, FakeDetector.detect_both, cache)
    assert len(FakeDetector.calls) == 1 and FakeDetector.calls[0] is not value._next_detector
    assert value._next_detector._classifier._color.calls == 0
    assert result["cnn_patch_calls"] == result["detect_both_calls"] == 1
    assert accounts == {side: subject.history.accounting_snapshot(value, side) for side in subject.history.SIDES}
    assert result["is_ground_truth"] is False


def test_independent_cached_across_frames_but_not_replaced_main() -> None:
    value, cache, frame = pipeline(), {}, subject.history.np.zeros((1, 1))
    for _ in range(2):
        subject.inactive_read(value, frame, FakeDetector, FakeDetector.detect_both, cache)
    assert FakeDetector.calls[0] is FakeDetector.calls[1]
    value._next_detector = pipeline()._next_detector
    subject.inactive_read(value, frame, FakeDetector, FakeDetector.detect_both, cache)
    assert FakeDetector.calls[1] is not FakeDetector.calls[2]


def test_shared_training_model_rejected_without_calling_eval() -> None:
    value = pipeline()
    value._next_detector._classifier._color._model.training = True
    with pytest.raises(RuntimeError, match="eval"):
        subject.independent_detector(value._next_detector, FakeDetector)
    assert value._next_detector._classifier._color._model.training is True


@pytest.mark.parametrize("mutation", ["model", "random", "accounting", "main"])
def test_inactive_side_effect_rejected(mutation: str) -> None:
    value = pipeline()
    def bad(detector: Any, frame: Any) -> Any:
        if mutation == "model":
            detector._classifier._color._model.version += 1
        elif mutation == "random":
            subject.random.random()
        elif mutation == "accounting":
            value._tsumo_count_1p[4] += 2
        else:
            value._next_detector._centroid = {}
        return FakeDetector.detect_both(detector, frame)
    with pytest.raises(RuntimeError, match="変更"):
        subject.inactive_read(value, subject.history.np.zeros((1, 1)), FakeDetector, bad, {})


def test_independent_frame_copy_prevents_source_write() -> None:
    value, frame = pipeline(), subject.history.np.zeros((1, 1))
    def bad(detector: Any, image: Any) -> Any:
        image[:] = 100
        return FakeDetector.detect_both(detector, image)
    subject.inactive_read(value, frame, FakeDetector, bad, {})
    assert frame[0, 0] == 0


@pytest.mark.parametrize("problem", ["missing", "extra_field", "subclass"])
def test_unapproved_detector_contract_rejected(problem: str) -> None:
    value = pipeline()
    if problem == "missing":
        value._next_detector = None
    elif problem == "extra_field":
        value._next_detector._history = []
    else:
        class Sub(FakeDetector):
            pass
        value._next_detector = Sub(value._next_detector._classifier)
    with pytest.raises(RuntimeError):
        subject.inactive_read(value, subject.history.np.zeros((1, 1)), FakeDetector, FakeDetector.detect_both, {})


def test_actual_next_once_return_identity_and_restore() -> None:
    value, rec = pipeline(), FakeRecorder()
    original = FakeDetector.detect_both
    state = {"pipeline": value, "main_next_calls": 0}
    with contextlib.ExitStack() as stack:
        subject.instrument_next(stack, rec, FakeDetector, state)
        result = value._next_detector.detect_both(None)
    assert FakeDetector.detect_both is original
    assert len(FakeDetector.calls) == state["main_next_calls"] == 1
    assert rec.rows[0]["returned"] == subject.pairs(result)
    assert rec.rows[0]["extra_call"] is False


def test_actual_slide_once_with_false_high_diff_and_restore() -> None:
    value, rec = pipeline(), FakeRecorder()
    state = {"pipeline": value, "slide_calls": Counter()}
    original = FakeSlide.update
    with contextlib.ExitStack() as stack:
        subject.instrument_slide(stack, rec, FakeSlide, state)
        returned = value._slide_detector_1p.update(None, None)
    assert FakeSlide.update is original and value._slide_detector_1p.calls == 1
    row = rec.rows[0]
    assert returned.slide_motion is False and row["before"]["cooldown"] == 6
    assert row["returned"]["after"]["cooldown"] == 5
    assert row["returned"]["diff_score"] > row["returned"]["threshold_used"]


def test_original_exception_preserved_even_if_recorder_fails() -> None:
    def error() -> Any:
        raise LookupError("original")
    rec = SimpleNamespace(emit=lambda row: (_ for _ in ()).throw(RuntimeError("logger")))
    with pytest.raises(LookupError, match="original"):
        subject.observe_call(rec, "fixture", error, {}, lambda value: value)


@pytest.mark.parametrize("active,frame", [(False, 32100), (True, 32100), (False, 32098), (False, 32882)])
def test_update_extra_only_inactive_in_window(active: bool, frame: int) -> None:
    rec, value = FakeRecorder(frame), pipeline()
    class Pipeline:
        def update(self, frame_idx: int, time_sec: float, image: Any) -> Any:
            return response
    response = SimpleNamespace(is_match_active=active)
    owner = Pipeline()
    owner.__dict__.update(vars(value))
    original, state = Pipeline.update, {}
    with contextlib.ExitStack() as stack:
        subject.install_update(stack, Pipeline, rec, FakeDetector, FakeDetector.detect_both, state)
        result = owner.update(frame, frame / 60, subject.history.np.zeros((1, 1)))
    assert result is response and Pipeline.update is original and state["pipeline"] is None
    expected = int(not active and subject.FIRST_FRAME <= frame <= subject.LAST_FRAME)
    assert len(FakeDetector.calls) == expected


def test_update_exception_restores_descriptor_and_context() -> None:
    class Pipeline:
        def update(self, frame_idx: int, time_sec: float, frame: Any) -> Any:
            raise LookupError("original")
    original, state = Pipeline.update, {}
    with pytest.raises(LookupError), contextlib.ExitStack() as stack:
        subject.install_update(stack, Pipeline, FakeRecorder(), FakeDetector, FakeDetector.detect_both, state)
        Pipeline().update(32100, 535, None)
    assert Pipeline.update is original and state["pipeline"] is None


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """一時fixtureだけを保存し、本番資産には触らない。"""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def baseline() -> list[dict[str, Any]]:
    """既存修復行と同frame occurrenceを含む人工対照。"""
    return [{"kind": kind, "frame_idx": 32100, "time_sec": 535, "side": "1P", "value": 1}
            for kind in ("frame_side", "boundary_repair_mutation", "collector_snapshot", "collector_snapshot")]


@pytest.fixture
def short_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """比較契約だけを4行へ縮小し、実15722行の代用とはしない。"""
    root = tmp_path / "reference"
    root.mkdir()
    write_rows(root / "frames.jsonl", baseline())
    monkeypatch.setattr(subject, "REFERENCE", root)
    monkeypatch.setattr(subject, "EXPECTED_LEGACY_ROWS", 4)
    return tmp_path / "candidate.jsonl"


def test_compare_includes_repair_and_duplicate_occurrence(short_reference: Path) -> None:
    rows = baseline()
    rows.insert(1, {"kind": subject.PREFIX + "result", "frame_idx": 32100})
    write_rows(short_reference, rows)
    comparison, extra = subject.compare_legacy(short_reference)
    assert comparison["all_legacy_bit_exact"] and len(extra) == 1
    assert comparison["includes_original_repair_rows"]


@pytest.mark.parametrize("edit", ["repair_value", "missing", "added", "order", "unknown_prefix", "unknown_kind"])
def test_any_legacy_difference_rejected(short_reference: Path, edit: str) -> None:
    rows = baseline()
    if edit == "repair_value":
        rows[1]["value"] = 2
    elif edit == "missing":
        rows.pop()
    elif edit == "added":
        rows.append(copy.deepcopy(rows[-1]))
    elif edit == "order":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows.append({"kind": subject.PREFIX + "unknown" if edit == "unknown_prefix" else "unknown", "frame_idx": 32100})
    write_rows(short_reference, rows)
    with pytest.raises(ValueError):
        subject.compare_legacy(short_reference)


def observation_fixture() -> list[dict[str, Any]]:
    """1frameのinactive追加観測receiptを作る。"""
    common = {"frame_idx": 32100, "time_sec": 535}
    result = {**common, "kind": subject.PREFIX + "result", "is_match_active": False,
              "main_next_calls": 0, "main_slide_calls": {"1P": 0, "2P": 0},
              "extra_detect_both_calls": 1, "pending_initialized": False}
    inactive = {**common, "kind": subject.PREFIX + "inactive_next", "detect_both_calls": 1,
                "cnn_patch_calls": 8, "elapsed_sec": 0.05, "main_detector_unchanged": True,
                "accounting_unchanged": True, "shared_model_unchanged": True, "rng_unchanged": True,
                "is_ground_truth": False}
    return [inactive, result]


@pytest.fixture
def one_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """検査器のfixture母数を明示短縮する。"""
    monkeypatch.setattr(subject, "LAST_FRAME", 32100)
    monkeypatch.setattr(subject, "EXPECTED_UPDATES", 1)


def test_observation_is_not_accounting_or_gt_permission(one_frame: None) -> None:
    result = subject.validate_observations(observation_fixture())
    assert result["inactive_extra_detect_both_calls"] == 1
    assert result["inactive_cnn_patch_calls"] == 8
    assert result["accounting_basis_verified"] is False and result["quality_gate_clear"] is False


@pytest.mark.parametrize("problem", ["missing_result", "duplicate", "clock", "frame", "missing_inactive",
                                    "counter_permission", "rng", "too_many_calls", "nan_elapsed", "main_calls"])
def test_observation_invalid_receipt_rejected(one_frame: None, problem: str) -> None:
    rows = observation_fixture()
    if problem == "missing_result":
        rows.pop()
    elif problem == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
    elif problem == "clock":
        rows[0]["time_sec"] = 534.0
    elif problem == "frame":
        rows[0]["frame_idx"] = True
    elif problem == "missing_inactive":
        rows.pop(0)
    elif problem == "counter_permission":
        rows[1]["pending_initialized"] = True
    elif problem == "rng":
        rows[0]["rng_unchanged"] = False
    elif problem == "too_many_calls":
        rows[0]["cnn_patch_calls"] = 9
    elif problem == "nan_elapsed":
        rows[0]["elapsed_sec"] = float("nan")
    else:
        rows[1]["main_next_calls"] = 1
    with pytest.raises(ValueError):
        subject.validate_observations(rows)


def test_common_comparison_filters_only_new_prefix(short_reference: Path) -> None:
    write_rows(short_reference, baseline() + [{"kind": subject.PREFIX + "result", "frame_idx": 32100}])
    rows, repairs = subject.common_rows(short_reference)
    assert len(rows) == 3 and len(repairs) == 1
    assert [json.loads(key)[-1] for key in rows][-2:] == [0, 1]


def test_run_restores_common_three_functions_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    before = (subject.runner.prepare, subject.runner.instrument, subject.runner.finish)
    def fail(args: Any) -> Any:
        assert subject.runner.prepare is subject.prepare
        raise RuntimeError("run")
    monkeypatch.setattr(subject.runner, "run", fail)
    with pytest.raises(RuntimeError, match="run"):
        subject.run(argparse.Namespace())
    assert before == (subject.runner.prepare, subject.runner.instrument, subject.runner.finish)


def test_functions_within_fifty_lines() -> None:
    tree = ast.parse(Path(subject.__file__).read_text(encoding="utf-8"))
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert all(node.end_lineno - node.lineno + 1 <= 50 for node in functions)


def test_fixed_reference_and_frozen_sources_unchanged() -> None:
    paths = {str((subject.base.ROOT / path).resolve()): value for path, value in subject.FIXED_SHA.items()}
    paths.update({str((subject.base.SNAPSHOT / "src" / path).resolve()): value
                  for path, value in subject.FROZEN_SHA.items()})
    subject.base.assert_unchanged(paths)


def arguments(tmp_path: Path) -> argparse.Namespace:
    """実新3fileのSHAをCLI相当で明示する。"""
    return argparse.Namespace(mode="start_epoch", start_sec=484.2, end_sec=605.0, output_root=tmp_path,
        module_sha256=subject.FIXED_SHA["scripts/match_start_epoch_shadow_v1.py"],
        module_test_sha256=subject.FIXED_SHA["tests/test_match_start_epoch_shadow_v1.py"],
        script_sha256=subject.base.sha256(Path(subject.__file__)), test_sha256=subject.base.sha256(subject.TEST),
        launcher_sha256=subject.base.sha256(subject.LAUNCHER))


def test_prepare_includes_new_code_and_fixed_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "ORIGINAL_PREPARE", lambda args: ({"input_and_code_sha256": {}}, {"config": 1}))
    monkeypatch.setattr(subject, "reference_guards", lambda: {})
    receipt, config = subject.prepare(arguments(tmp_path))
    hashes = receipt["input_and_code_sha256"]
    for path in (Path(subject.__file__), subject.TEST, subject.LAUNCHER):
        assert hashes[str(path.resolve())] == subject.base.sha256(path)
    assert receipt["initial_pair_observation"]["pending_initialization"] is False
    assert config == {"config": 1}


@pytest.mark.parametrize("field", ["script_sha256", "test_sha256", "launcher_sha256", "module_sha256", "module_test_sha256"])
def test_prepare_fixed_hash_mismatch_rejected(tmp_path: Path, field: str) -> None:
    args = arguments(tmp_path)
    setattr(args, field, "0" * 64)
    with pytest.raises((ValueError, RuntimeError)):
        subject.prepare(args)


def test_reference_complete_mismatch_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "COMPLETE").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(subject, "REFERENCE", tmp_path)
    with pytest.raises((ValueError, RuntimeError)):
        subject.reference_guards()


@pytest.mark.parametrize("mutate", [False, True])
def test_finish_five_hashes_and_complete_last(tmp_path: Path, short_reference: Path, one_frame: None,
                                             monkeypatch: pytest.MonkeyPatch, mutate: bool) -> None:
    output = tmp_path / "output"
    output.mkdir()
    write_rows(output / "frames.jsonl", baseline() + observation_fixture())
    guard = tmp_path / "guard"
    guard.write_text("before", encoding="utf-8")
    receipt = {"input_and_code_sha256": {str(guard): subject.base.sha256(guard)}}
    original_reader, original_writer = subject.runner.read_rows, subject.base.write_json
    def fake_finish(root: Path, manifest: Any, rec: Any, elapsed: float) -> Any:
        rows, repairs = subject.runner.read_rows(root / "frames.jsonl")
        assert len(rows) == 3 and len(repairs) == 1
        for name in ("PLAN.json", "SUMMARY.json", subject.runner.DIFFERENCE_NAME):
            subject.base.write_json(root / name, {"cpu_fixture": True})
        names = ("PLAN.json", "frames.jsonl", "SUMMARY.json", subject.runner.DIFFERENCE_NAME)
        subject.base.write_json(root / "COMPLETE", {"sha256": {name: subject.base.sha256(root / name) for name in names}})
        assert not (root / "COMPLETE").exists()
        if mutate:
            guard.write_text("changed", encoding="utf-8")
        return {"frame_count": 1}
    monkeypatch.setattr(subject, "ORIGINAL_FINISH", fake_finish)
    if mutate:
        with pytest.raises((RuntimeError, ValueError)):
            subject.finish(output, receipt, None, 0.1)
        assert not (output / "COMPLETE").exists()
    else:
        result = subject.finish(output, receipt, None, 0.1)
        complete = subject.base.read_json(output / "COMPLETE")
        assert len(complete["sha256"]) == 5 and result["all_legacy_bit_exact"]
        subject.base.assert_unchanged({str(output / name): digest for name, digest in complete["sha256"].items()})
    assert subject.runner.read_rows is original_reader and subject.base.write_json is original_writer


FROZEN_API_PROBE = r'''
import contextlib, inspect, io, json, os, sys
from collections import Counter, deque
from pathlib import Path
from scripts import diagnose_video38_initial_pair_observation_v1 as s
before_paths = {str((s.base.ROOT / name).resolve()): digest for name,digest in s.FIXED_SHA.items()}
before_paths.update({str((s.base.SNAPSHOT / "src" / name).resolve()): digest for name,digest in s.FROZEN_SHA.items()})
s.base.assert_unchanged(before_paths)
reference = s.reference_guards()
receipt = s.base.read_json(s.REFERENCE / "PLAN.json")
s.base.assert_unchanged({key:value for key,value in receipt["input_and_code_sha256"].items() if key in before_paths})
os.chdir(s.base.SNAPSHOT)
collector = s.base.load_collector()
import numpy as np
import torch
from src.next_detector import NextDetector
from src.next_slide_detector import NextSlideDetector
from src.patch_classifier import CnnPatchClassifier, GatedCnnClassifier
from src.ui_mask import UiMaskMatcher
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
color = CnnPatchClassifier()
gated = GatedCnnClassifier(color, ui_matcher=UiMaskMatcher([]))
detector = NextDetector(gated)
pipeline = object.__new__(collector.RecognitionPipeline)
pipeline._next_detector = detector
for side in s.history.SIDES:
    for name in s.history.FIELDS:
        value = Counter() if name=="tsumo_count" else deque() if name=="pending_tsumo" else None
        setattr(pipeline,"_"+name+"_"+side.lower(),value)
    setattr(pipeline,"_slide_detector_"+side.lower(),NextSlideDetector(side=side))
image = np.zeros((1080,1920,3),dtype=np.uint8)
before_rng = s.rng_stamp()
cache={}
inactive = s.inactive_read(pipeline,image,NextDetector,NextDetector.detect_both,cache)
assert s.rng_stamp()==before_rng and not torch.cuda.is_initialized()
model_state=s.model_stamp(color._model)
cnn_before=s.rng_stamp()
cpu_class=cache["detector"]._classifier._color.classify(np.zeros((16,16,3),dtype=np.uint8))
assert s.rng_stamp()==cnn_before and s.model_stamp(color._model)==model_state
stream=io.StringIO()
rec=s.history.HistoryRecorder(stream,receipt["target_board"])
rec.frame,rec.time_sec=32100,535.0
patches=[]
original_patch=s.base.patch
def traced(stack,obj,name,replacement):
    patches.append((obj,name,inspect.getattr_static(obj,name)))
    original_patch(stack,obj,name,replacement)
with contextlib.ExitStack() as stack:
    original_patch(stack,s.base,"patch",traced)
    state=s.instrument(stack,collector,rec,receipt)
    state.update(pipeline=pipeline,main_next_calls=0,slide_calls=Counter())
    raw=pipeline._next_detector.detect_both(image)
    slide=pipeline._slide_detector_1p.update(image,np.full_like(image,255))
for obj,name,value in reversed(patches):
    first=next(old for owner,field,old in patches if owner is obj and field==name)
    assert inspect.getattr_static(obj,name) is first
rows=[json.loads(line) for line in stream.getvalue().splitlines()]
assert [row["kind"] for row in rows]==["initial_pair_main_next","initial_pair_main_slide"]
assert state["main_next_calls"]==1 and state["slide_calls"]["1P"]==1
s.base.assert_unchanged(before_paths)
s.base.assert_unchanged(reference)
print(json.dumps({"frozen_module_count":len(s.history.frozen_modules()),"patch_operations":len(patches),
 "unique_restored":len({(id(obj),name) for obj,name,_ in patches}),"inactive":inactive,
 "actual_cpu_random_weight_classifier_result":cpu_class,"actual_next_and_slide_once":True,
 "gpu_initialized":torch.cuda.is_initialized(),"source_reference_sha_unchanged":True,
 "all_15722_gpu_comparison":"not_run","all_parameters_device":str(next(color._model.parameters()).device)},ensure_ascii=False))
'''


def test_fresh_frozen_real_api_cpu_probe() -> None:
    """実modelはCPU乱数初期化fixture。学習重み/動画/GPUを読み込まない。"""
    import os
    import subprocess
    import sys
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    result = subprocess.run([sys.executable, "-B", "-c", FROZEN_API_PROBE], cwd=subject.base.ROOT,
                            env=environment, text=True, capture_output=True, timeout=90, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["actual_next_and_slide_once"] and report["source_reference_sha_unchanged"]
    assert report["unique_restored"] == 15 and report["gpu_initialized"] is False
    assert report["all_15722_gpu_comparison"] == "not_run"
