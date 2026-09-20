"""開始gate追加計装の非干渉・保護・完了順をCPUだけで検査する。"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import diagnose_video38_start_gate_v1 as diag

REAL_INSTALL_END = diag.install_end


@pytest.fixture(autouse=True)
def isolate_parent_observer(monkeypatch: Any) -> None:
    """親moduleの実API検査は別のfresh frozen probeで行う。"""
    monkeypatch.setattr(diag, "install_end", lambda stack, collector, rec: None)


@pytest.mark.parametrize("existed", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_absolute_end_loader_restores_module_on_success_and_exception(tmp_path: Path, monkeypatch: Any,
                                                                     existed: bool, failure: bool) -> None:
    module_path = tmp_path / "observer.py"
    module_path.write_text("raise ImportError('fixture')\n" if failure else
                           "def install(stack, collector, rec):\n    rec.calls += 1\n")
    monkeypatch.setattr(diag, "END_MODULE", module_path)
    name, previous = "_video38_start_gate_end_observer", SimpleNamespace(old=True)
    if existed: monkeypatch.setitem(sys.modules, name, previous)
    else: monkeypatch.delitem(sys.modules, name, raising=False)
    rec = SimpleNamespace(calls=0)
    if failure:
        with pytest.raises(ImportError, match="fixture"):
            with contextlib.ExitStack() as stack:
                REAL_INSTALL_END(stack, SimpleNamespace(), rec)
    else:
        with contextlib.ExitStack() as stack:
            REAL_INSTALL_END(stack, SimpleNamespace(), rec)
            assert sys.modules[name] is not previous and rec.calls == 1
    assert sys.modules.get(name) is previous if existed else name not in sys.modules


def recorder(frame: int = diag.FIRST_OBSERVED_FRAME) -> Any:
    """指定frame直前まで進んだ旧recorderのCPU fixture。"""
    rec = diag.history.HistoryRecorder(io.StringIO(), diag.base.board_value(np.zeros((13, 6))))
    rec.frames = (frame - diag.history.FIRST_FRAME) // diag.history.STRIDE
    rec.frame, rec.time_sec, rec.decoded_frame = frame - 2, (frame - 2) / 60, frame
    return rec


class Pipeline:
    """各判定器を一度だけ呼ぶ小さなCPU対照。認識モデルは持たない。"""

    def __init__(self) -> None:
        for name in diag.LATCH_FIELDS:
            setattr(self, "_" + name, False if "active" in name or "locked" in name else -1.0)
        self._match_end_detector = SimpleNamespace(last_detected_t=540.0)
        self.calls: Counter[str] = Counter()
        self.failure: str | None = None
        self.error = RuntimeError("fixture error")
        grid = np.zeros((13, 6), dtype=np.int8)
        side = SimpleNamespace(state=SimpleNamespace(value="menu"), confirmed_board=grid,
                               cnn_board=grid, score=0)
        self.result = SimpleNamespace(p1=side, p2=side, is_match_active=False, match_end_locked=True)
        for suffix in ("1p", "2p"):
            fields = dict(tsumo_count=Counter(), pending_tsumo=deque(), last_seen_next=None,
                          landing_pending=None, last_consumed_color=None, first_move_sec=None,
                          constraint_valid=True)
            for name, value in fields.items():
                setattr(self, f"_{name}_{suffix}", value)
            setattr(self, "_stable_color_memory_" + suffix, {})
            setattr(self, "_sm_" + suffix, SimpleNamespace(context=side))

    def _board_shows_real_gameplay(self, frame: Any) -> bool:
        self.calls["board"] += 1
        if self.failure == "board":
            raise self.error
        return False

    def _update_post_match_lockdown_latch(self, frame: Any, time_sec: float,
                                        match_end_locked: bool, score_zero_both: bool,
                                        cur_score_1p: int | None, cur_score_2p: int | None,
                                        score_actively_moving: bool) -> None:
        self.calls["latch"] += 1
        self._post_match_lockdown_active = True
        self._board_shows_real_gameplay(frame)
        if self.failure == "latch":
            raise self.error

    def reset(self, match_start_sec: float | None = None) -> str:
        self.calls["reset"] += 1
        self._post_match_lockdown_active = False
        if self.failure == "reset":
            raise self.error
        return "reset-return"

    def update(self, frame_idx: int, time_sec: float, frame: Any) -> Any:
        self.calls["update"] += 1
        self._update_post_match_lockdown_latch(frame, time_sec, True, False, 13, 13, True)
        self._board_shows_real_gameplay(frame)
        self.reset(match_start_sec=time_sec)
        if self.failure == "update":
            raise self.error
        return self.result

    @classmethod
    def load_default(cls, **kwargs: Any) -> Any:
        return cls()


def rows(rec: Any) -> list[dict[str, Any]]:
    return [json.loads(line) for line in rec.stream.getvalue().splitlines()]


def instrumented(frame: int, failure: str | None = None) -> tuple[Any, Any]:
    rec, pipe = recorder(frame), Pipeline()
    pipe.failure = failure
    with contextlib.ExitStack() as stack:
        diag.instrument(stack, SimpleNamespace(RecognitionPipeline=Pipeline), rec)
        pipe.update(frame, frame / 60, None)
    return pipe, rec


@pytest.mark.parametrize("frame,expected", [(32098, False), (32100, True), (32880, True), (32882, False)])
def test_only_registered_window_is_observed(frame: int, expected: bool) -> None:
    pipe, rec = instrumented(frame)
    extra = [row for row in rows(rec) if row["kind"] in diag.GATE_KINDS]
    assert bool(extra) is expected
    assert pipe.calls == {"update": 1, "latch": 1, "board": 2, "reset": 1}


def test_live_wrapping_preserves_identity_fields_calls_and_restores(monkeypatch: Any) -> None:
    names = ("update", "reset", "_board_shows_real_gameplay", "_update_post_match_lockdown_latch")
    originals = {name: getattr(Pipeline, name) for name in names}
    def reject(*args: Any) -> None:
        raise AssertionError("line traceは禁止")
    monkeypatch.setattr(sys, "settrace", reject)
    pipe, rec = instrumented(diag.FIRST_OBSERVED_FRAME)
    assert all(getattr(Pipeline, name) is value for name, value in originals.items())
    gate = [row for row in rows(rec) if row["kind"] == "start_gate_latch"][0]
    assert gate["arguments"] == dict(time_sec=535.0, match_end_locked=True, score_zero_both=False,
        cur_score_1p=13, cur_score_2p=13, score_actively_moving=True)
    assert gate["before"]["post_match_lockdown_active"] is False
    assert gate["after"]["post_match_lockdown_active"] is True
    assert pipe._post_match_lockdown_active is False
    assert len([row for row in rows(rec) if row["kind"] == "start_gate_board_return"]) == 2


@pytest.mark.parametrize("failure", ["board", "latch", "reset", "update"])
def test_actual_exception_and_restoration_are_preserved(failure: str) -> None:
    originals = {name: getattr(Pipeline, name) for name in ("update", "reset", "_board_shows_real_gameplay",
                                                          "_update_post_match_lockdown_latch")}
    rec, pipe = recorder(), Pipeline()
    pipe.failure = failure
    with pytest.raises(RuntimeError) as raised:
        with contextlib.ExitStack() as stack:
            diag.instrument(stack, SimpleNamespace(RecognitionPipeline=Pipeline), rec)
            pipe.update(diag.FIRST_OBSERVED_FRAME, 535.0, None)
    assert raised.value is pipe.error
    assert all(getattr(Pipeline, name) is value for name, value in originals.items())
    assert not any(row["kind"] == "start_gate_result" for row in rows(rec))


def test_result_returns_original_object_and_zero_reset_is_not_added() -> None:
    rec, pipe = recorder(), Pipeline()
    with contextlib.ExitStack() as stack:
        diag.instrument_board(stack, Pipeline, rec)
        rec.frame = diag.FIRST_OBSERVED_FRAME
        assert pipe._board_shows_real_gameplay(None) is False
    assert pipe.calls == {"board": 1}
    assert rows(rec)[0]["returned"] is False


def test_reset_none_returns_exact_object_and_before_copy_survives() -> None:
    rec, pipe = recorder(), Pipeline()
    rec.frame = diag.FIRST_OBSERVED_FRAME
    pipe._post_match_lockdown_active = True
    with contextlib.ExitStack() as stack:
        diag.instrument_reset(stack, Pipeline, rec)
        assert pipe.reset() == "reset-return"
    row = rows(rec)[0]
    assert row["arguments"] == {"match_start_sec": None}
    assert row["before"]["post_match_lockdown_active"] is True
    assert row["after"]["post_match_lockdown_active"] is False


def reference(tmp_path: Path, monkeypatch: Any, legacy: str = "{}\n") -> Path:
    old = tmp_path / "old"
    old.mkdir()
    for name, content in (("PLAN.json", "{}\n"), ("SUMMARY.json", "{}\n"), ("frames.jsonl", legacy)):
        (old / name).write_text(content, encoding="utf-8")
    complete = {"sha256": {name: diag.base.sha256(old / name)
                           for name in ("PLAN.json", "SUMMARY.json", "frames.jsonl")}}
    diag.base.write_json(old / "COMPLETE", complete)
    monkeypatch.setattr(diag, "REFERENCE", old)
    monkeypatch.setattr(diag, "REFERENCE_COMPLETE_SHA", diag.base.sha256(old / "COMPLETE"))
    return old


def args(output: Path) -> argparse.Namespace:
    return argparse.Namespace(start_sec=484.2, end_sec=605.0, output_root=output,
                              allow_native_runtime_mismatch=True)


def test_prepare_guards_new_files_and_all_reference_artifacts(tmp_path: Path, monkeypatch: Any) -> None:
    old = reference(tmp_path, monkeypatch)
    monkeypatch.setattr(diag, "ORIGINAL_PREPARE", lambda args: ({"input_and_code_sha256": {}}, {}))
    receipt, _ = diag.prepare(args(tmp_path / "output"))
    guards = receipt["input_and_code_sha256"]
    assert all(str(path.resolve()) in guards for path in (diag.TEST, diag.LAUNCHER, Path(diag.__file__)))
    assert all(str(path.resolve()) in guards for path in old.iterdir())
    assert all(path in guards for path in diag.END_HASHES)
    assert receipt["extra_detector_calls"] is False


@pytest.mark.parametrize("name", ["COMPLETE", "PLAN.json", "SUMMARY.json", "frames.jsonl"])
def test_prepare_rejects_changed_reference(tmp_path: Path, monkeypatch: Any, name: str) -> None:
    old = reference(tmp_path, monkeypatch)
    (old / name).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError):
        diag.prepare(args(tmp_path / "output"))


def test_prepare_rejects_history_code_change(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(diag, "HISTORY_SHA", "0" * 64)
    with pytest.raises(ValueError, match="history script SHA"):
        diag.prepare(args(tmp_path))


def legacy_line(frame: int = 32100, side: str = "1P", value: int = 0) -> str:
    return json.dumps({"kind": "frame_side", "frame_idx": frame, "side": side, "value": value}) + "\n"


def test_legacy_comparison_preserves_occurrences_and_outer_decode_clock(tmp_path: Path, monkeypatch: Any) -> None:
    decoded = json.dumps(dict(kind="target_decoded_frame", frame_idx=34700, actual_frame=34702)) + "\n"
    legacy = legacy_line() + legacy_line() + decoded
    reference(tmp_path, monkeypatch, legacy)
    path = tmp_path / "new.jsonl"
    path.write_text(legacy_line() + json.dumps(dict(kind="start_gate_reset")) + "\n" + legacy_line() + decoded)
    report, gate = diag.compare_legacy(path)
    assert report["legacy_row_count"] == 3 and report["global_order_equal"] is True
    assert report["legacy_counts"]["target_decoded_frame"] == 1 and len(gate) == 1


@pytest.mark.parametrize("changed", [legacy_line(value=1), "", legacy_line()+legacy_line(),
                                      legacy_line(side="2P"), legacy_line(frame=32102),
                                      legacy_line().replace(": ", ":")])
def test_legacy_mismatch_fails_closed(tmp_path: Path, monkeypatch: Any, changed: str) -> None:
    reference(tmp_path, monkeypatch, legacy_line())
    path = tmp_path / "new.jsonl"
    path.write_text(changed)
    with pytest.raises(ValueError, match="bit exact"):
        diag.compare_legacy(path)


def gate_rows() -> list[dict[str, Any]]:
    values = []
    for frame in range(32100, 32882, 2):
        for kind in ("start_gate_result", "start_gate_latch", "start_gate_board_return"):
            values.append(dict(kind=kind, frame_idx=frame, time_sec=frame/60))
    return values


def end_rows() -> list[dict[str, Any]]:
    return [dict(kind=diag.END_PREFIX+suffix, frame_idx=frame, time_sec=frame/60, side=side)
            for frame in range(35770, 35814, 2) for side in ("1P", "2P")
            for suffix in ("step_enter", "step_return")]


def test_end_coverage_is_separate_from_start_and_does_not_invent_calls() -> None:
    summary = diag.end_summary(end_rows())
    assert summary["step_frame_side_count"] == 44
    assert summary["normal_release_verified"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "window", "clock", "side", "kind"])
def test_end_coverage_rejects_wrong_contract(mutation: str) -> None:
    values = end_rows()
    if mutation == "missing": values.pop(0)
    elif mutation == "duplicate": values.append(values[0])
    elif mutation == "window": values[0]["frame_idx"] = 35768
    elif mutation == "clock": values[0]["time_sec"] = 0
    elif mutation == "side": values[0]["side"] = None
    else: values[0]["kind"] = "end_boundary_unknown"
    with pytest.raises(ValueError):
        diag.end_summary(values)


def test_gate_coverage_allows_zero_actual_reset_calls() -> None:
    summary = diag.gate_summary(gate_rows())
    assert summary["result_update_count"] == 391
    assert summary["counts"].get("start_gate_reset", 0) == 0
    assert summary["individual_gate_cause_verified"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "window", "clock", "exception", "latch", "pixel"])
def test_gate_coverage_rejects_incomplete_observation(mutation: str) -> None:
    values = gate_rows()
    if mutation == "missing": values.pop(0)
    elif mutation == "duplicate": values.append(values[0])
    elif mutation == "window": values[0]["frame_idx"] = 32098
    elif mutation == "clock": values[0]["time_sec"] = 0
    elif mutation == "exception": values[0]["exception"] = "RuntimeError"
    else:
        remove = "start_gate_latch" if mutation == "latch" else "start_gate_board_return"
        values = [row for row in values if row["kind"] != remove]
    with pytest.raises(ValueError):
        diag.gate_summary(values)


def finish_fixture(tmp_path: Path, monkeypatch: Any) -> tuple[Path, Any]:
    reference(tmp_path, monkeypatch, legacy_line())
    output = tmp_path / "output"
    output.mkdir()
    (output / "PLAN.json").write_text("{}\n")
    (output / "frames.jsonl").write_text(legacy_line()+"".join(json.dumps(row)+"\n" for row in gate_rows()+end_rows()))
    rec = recorder()
    rec.frames, rec.frame = 3624, 36298
    rec.model_loads, rec.pipeline_receipt = ["CPU fixture"], {"cpu_fixture": True}
    return output, rec


def test_finish_defers_complete_and_binds_four_artifacts(tmp_path: Path, monkeypatch: Any) -> None:
    output, rec = finish_fixture(tmp_path, monkeypatch)
    original = diag.base.write_json
    observed = []
    def write(path: Path, value: Any) -> None:
        observed.append(path.name)
        if path.name == diag.EXTRA_NAME:
            assert not (output / "COMPLETE").exists()
        original(path, value)
    monkeypatch.setattr(diag.base, "write_json", write)
    result = diag.finish(output, {"input_and_code_sha256": {}}, rec, 1.0)
    complete = diag.base.read_json(output / "COMPLETE")
    assert observed == ["SUMMARY.json", diag.EXTRA_NAME, "COMPLETE"]
    assert len(complete["sha256"]) == 4
    assert all(diag.base.sha256(output/name) == digest for name, digest in complete["sha256"].items())
    assert result["start_gate"]["production_adoption"] is False
    assert diag.base.write_json is write


@pytest.mark.parametrize("when", ["SUMMARY.json", diag.EXTRA_NAME])
def test_finish_rechecks_guard_and_restores_writer_on_failure(tmp_path: Path, monkeypatch: Any, when: str) -> None:
    output, rec = finish_fixture(tmp_path, monkeypatch)
    asset = tmp_path / "guard"
    asset.write_text("fixed")
    receipt = {"input_and_code_sha256": {str(asset): diag.base.sha256(asset)}}
    original = diag.base.write_json
    def write(path: Path, value: Any) -> None:
        original(path, value)
        if path.name == when: asset.write_text("changed")
    monkeypatch.setattr(diag.base, "write_json", write)
    with pytest.raises(ValueError):
        diag.finish(output, receipt, rec, 1.0)
    assert not (output / "COMPLETE").exists() and diag.base.write_json is write


def test_finish_rechecks_compared_frames_after_extra_write(tmp_path: Path, monkeypatch: Any) -> None:
    output, rec = finish_fixture(tmp_path, monkeypatch)
    original = diag.base.write_json
    def write(path: Path, value: Any) -> None:
        original(path, value)
        if path.name == diag.EXTRA_NAME:
            (output / "frames.jsonl").write_text("changed")
    monkeypatch.setattr(diag.base, "write_json", write)
    with pytest.raises(ValueError):
        diag.finish(output, {"input_and_code_sha256": {}}, rec, 1.0)
    assert not (output / "COMPLETE").exists()


def test_run_patch_chain_is_restored_after_exception(tmp_path: Path, monkeypatch: Any) -> None:
    originals = (diag.history.prepare, diag.history.instrument_pipeline, diag.history.finish)
    def run(args: Any) -> Any:
        assert diag.history.prepare is diag.prepare
        assert diag.history.instrument_pipeline is diag.instrument
        assert diag.history.finish is diag.finish
        raise RuntimeError("stop")
    monkeypatch.setattr(diag.history, "run", run)
    with pytest.raises(RuntimeError, match="stop"):
        diag.run(args(tmp_path))
    assert (diag.history.prepare, diag.history.instrument_pipeline, diag.history.finish) == originals


def cpu_collection_environment(tmp_path: Path, monkeypatch: Any) -> None:
    """実run/collect/finishをCPU fixtureへ接続する。動画・CNNは呼ばない。"""
    box: dict[str, Any] = {}
    def collect(video: Path, output: Path, **kwargs: Any) -> None:
        pipe, rec = Pipeline(), box["rec"]
        for frame in range(29052, 36300, 2):
            rec.decoded_frame = frame
            pipe.update(frame, frame/60, None)
        raise diag.base.CollectionFinished()
    def video(stack: Any, collector: Any, rec: Any) -> None:
        box["rec"] = rec
    def model(stack: Any, rec: Any, hashes: Any) -> None:
        rec.model_loads, rec.pipeline_receipt = ["CPU fixture"], {"cpu_fixture": True}
    def prepare(args: Any) -> Any:
        return {"input_and_code_sha256": {}, "target_board": diag.base.board_value(np.zeros((13, 6))),
                "video_path": str(tmp_path / "unused.mp4")}, {}
    collector = SimpleNamespace(RecognitionPipeline=Pipeline, collect_lean=collect)
    monkeypatch.setattr(diag.base, "SNAPSHOT", tmp_path)
    monkeypatch.setattr(diag.base, "load_collector", lambda: collector)
    monkeypatch.setattr(diag.base, "collection_arguments", lambda c, config:
                        {"sample_interval_sec": 0, "normalize_fps_30": True})
    monkeypatch.setattr(diag.base, "instrument_storage", lambda *args: None)
    monkeypatch.setattr(diag.base, "instrument_model_load", model)
    monkeypatch.setattr(diag.base, "instrument_video", video)
    monkeypatch.setattr(diag.history, "frozen_modules", lambda: {})
    monkeypatch.setattr(diag.history, "prepare", prepare)
    monkeypatch.setattr(diag, "ORIGINAL_PREPARE", prepare)


def cpu_end_observer(stack: Any, collector: Any, rec: Any) -> None:
    """親側の追加行を模す。親の実installは別fresh processで検査する。"""
    original = collector.RecognitionPipeline.update
    def update(pipe: Any, frame_idx: int, time_sec: float, frame: Any) -> Any:
        result = original(pipe, frame_idx, time_sec, frame)
        if 35770 <= frame_idx <= 35812:
            for side in ("1P", "2P"):
                for suffix in ("step_enter", "step_return"):
                    rec.emit({"kind": diag.END_PREFIX+suffix, "side": side})
        return result
    diag.base.patch(stack, collector.RecognitionPipeline, "update", update)


def test_real_run_prepare_instrument_collect_finish_path_and_exclusive_root(tmp_path: Path, monkeypatch: Any) -> None:
    cpu_collection_environment(tmp_path, monkeypatch)
    initial_cwd, initial_path = Path.cwd(), list(sys.path)
    old = tmp_path / "reference"
    diag.history.run(args(old))
    monkeypatch.setattr(diag, "REFERENCE", old)
    monkeypatch.setattr(diag, "REFERENCE_COMPLETE_SHA", diag.base.sha256(old / "COMPLETE"))
    monkeypatch.setattr(diag, "install_end", cpu_end_observer)
    output = tmp_path / "new"
    result = diag.run(args(output))
    assert result["frame_count"] == 3624
    assert result["start_gate"]["observation"]["result_update_count"] == 391
    assert result["start_gate"]["end_boundary_observation"]["step_frame_side_count"] == 44
    assert result["start_gate"]["comparison"]["legacy_row_count"] == 14496
    assert Path.cwd() == initial_cwd and sys.path == initial_path
    before = diag.base.sha256(output / "COMPLETE")
    with pytest.raises(FileExistsError):
        diag.run(args(output))
    assert diag.base.sha256(output / "COMPLETE") == before
