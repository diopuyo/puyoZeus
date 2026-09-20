"""前試合境界から同一の凍結pipelineを継続し、会計基準の欠落を観測する。"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from scripts import diagnose_video38_confirmed_collapse_v1 as base


FORMAT = "video38-accounting-history-diagnostic/v1"
START_SEC, END_SEC = 484.2, 605.0
FPS, STRIDE = 60, 2
THREADS = 2
FIRST_FRAME, END_FRAME = 29052, 36300
EXPECTED_FRAMES = len(range(FIRST_FRAME, END_FRAME, STRIDE))
REFERENCE = base.VERIFY / "video38_c6_pending_commit_shadow_2026-09-07_v1"
LAUNCHER = base.ROOT / "scripts/launch_video38_accounting_history_v1.sh"
TEST = base.ROOT / "tests/test_diagnose_video38_accounting_history_v1.py"
SIDES = ("1P", "2P")
FIELDS = ("tsumo_count", "pending_tsumo", "last_seen_next", "landing_pending",
          "last_consumed_color", "first_move_sec", "constraint_valid")


def accounting_snapshot(pipeline: Any, side: str) -> dict[str, Any]:
    """内部会計値をコピーする。constraint flagは基準の正当性へ読み替えない。"""
    suffix = side.lower()
    value = {name: getattr(pipeline, f"_{name}_{suffix}") for name in FIELDS}
    value["pending_tsumo"] = list(value["pending_tsumo"])
    value["tsumo_count"] = dict(value["tsumo_count"])
    return base.json_value(value)


def frozen_modules() -> dict[str, str]:
    """認識srcが凍結snapshotに属することを実体pathで検査する。"""
    result = {}
    for name, module in sys.modules.items():
        if name != "src" and not name.startswith("src."):
            continue
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(base.SNAPSHOT.resolve()):
            raise RuntimeError(f"非凍結srcが混入しています: {name}: {path}")
        result[name] = str(path)
    return result


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """実範囲を偽らず既存SHA検査を再利用し、新規3fileも凍結する。"""
    if (args.start_sec, args.end_sec) != (START_SEC, END_SEC):
        raise ValueError("事前登録した484.2から605秒だけが対象です")
    with contextlib.ExitStack() as stack:
        base.patch(stack, base, "MAX_DURATION_SEC", END_SEC - START_SEC)
        receipt, config = base.prepare(args)
    reference = base.read_json(REFERENCE / "PLAN.json")
    hashes = receipt["input_and_code_sha256"]
    hashes.update(reference["input_and_code_sha256"])
    paths = (Path(__file__), TEST, LAUNCHER, REFERENCE / "PLAN.json")
    hashes.update({str(path.resolve()): base.sha256(path) for path in paths})
    base.assert_unchanged(hashes)
    receipt.update(format_version=FORMAT, requested_interval_sec=[START_SEC, END_SEC],
                   expected_frame_count=EXPECTED_FRAMES, line_trace_enabled=False,
                   initialization="cold_at_484.2_then_same_pipeline_through_605",
                   recognition_variant="unchanged_frozen_baseline_not_pending_shadow",
                   accounting_basis_verified=False, production_adoption=False,
                   comparison_limit="前史追加とpending解除の効果は比較しない")
    return receipt, config


class HistoryRecorder(base.Recorder):
    """時計欠落を拒否し、各update前後の会計と公開値を記録する。"""

    def begin_frame(self, frame: int, time_sec: float) -> None:
        expected = FIRST_FRAME + self.frames * STRIDE
        if frame != expected or frame >= END_FRAME or abs(time_sec - frame / FPS) > 1e-8:
            raise RuntimeError(f"履歴clock欠落/重複: {frame}, expected={expected}")
        super().begin_frame(frame, time_sec)


def instrument_pipeline(stack: contextlib.ExitStack, collector: ModuleType,
                        rec: HistoryRecorder) -> None:
    """line traceを使わず、戻り値・内部状態を変更しない観測だけを設置する。"""
    from src import ojama_write_accounting as accounting

    cls = collector.RecognitionPipeline
    original_update = cls.update
    original_filter = accounting.apply_ojama_write_accounting_filter
    original_load = inspect.getattr_static(cls, "load_default")

    def update(self: Any, frame_idx: int, time_sec: float, frame: np.ndarray) -> Any:
        if rec.decoded_frame != frame_idx:
            raise RuntimeError("復号frameとpipeline frameが一致しません")
        rec.begin_frame(frame_idx, time_sec)
        before = {side: accounting_snapshot(self, side) for side in SIDES}
        result = original_update(self, frame_idx, time_sec, frame)
        for side, value in zip(SIDES, (result.p1, result.p2), strict=True):
            rec.record_side(self, side, value)
            rec.emit({"kind": "accounting_update", "side": side,
                      "before": before[side], "after": accounting_snapshot(self, side),
                      "is_match_active": result.is_match_active,
                      "accounting_basis_verified": False})
        return result

    def filtered(board: Any, memory: Any, credit: Any, *args: Any, **kwargs: Any) -> Any:
        raw = board._grid.copy()
        result = original_filter(board, memory, credit, *args, **kwargs)
        rec.capture_raw(memory, raw, result)
        return result

    def load(inner_cls: Any, *args: Any, **kwargs: Any) -> Any:
        pipeline = original_load.__func__(inner_cls, *args, **kwargs)
        cnn = pipeline._reader._classifier._cnn
        device = str(next(cnn._model.parameters()).device)
        if device != "cuda:0":
            raise RuntimeError(f"実CNNが指定CUDAではありません: {device}")
        rec.pipeline_receipt = {"load_default_kwargs": base.json_value(kwargs),
                                "board_cnn_device": device, "src_modules": frozen_modules()}
        return pipeline

    base.patch(stack, cls, "update", update)
    base.patch(stack, cls, "load_default", classmethod(load))
    base.patch(stack, accounting, "apply_ojama_write_accounting_filter", filtered)


def finish(output: Path, receipt: dict[str, Any], rec: HistoryRecorder,
           elapsed: float) -> dict[str, Any]:
    """全frameとCUDA/guardを確認し、会計正当性とは別の診断完了を保存する。"""
    if rec.frames != EXPECTED_FRAMES or not rec.model_loads or not rec.pipeline_receipt:
        raise RuntimeError("履歴frame/CUDA/model receiptが不足しています")
    base.assert_unchanged(receipt["input_and_code_sha256"])
    summary = {"format_version": FORMAT, "status": "diagnostic_complete_not_adopted",
               "frame_count": rec.frames, "row_count": rec.rows, "elapsed_sec": elapsed,
               "first_frame": FIRST_FRAME, "last_frame": rec.frame,
               "collector_snapshot_count": rec.snapshots, "model_loads": rec.model_loads,
               "pipeline": rec.pipeline_receipt, "accounting_basis_verified": False,
               "line_trace_enabled": False, "production_adoption": False}
    base.write_json(output / "SUMMARY.json", summary)
    names = ("PLAN.json", "frames.jsonl", "SUMMARY.json")
    complete = {
        "format_version": FORMAT, "status": summary["status"],
        "sha256": {name: base.sha256(output / name) for name in names}}
    base.assert_unchanged(receipt["input_and_code_sha256"])
    base.write_json(output / "COMPLETE", complete)
    return summary


def collect(output: Path, receipt: dict[str, Any], collector: ModuleType,
            kwargs: dict[str, Any]) -> HistoryRecorder:
    """既存の保存遮断・checkpointガード・復号時計検査を再利用する。"""
    with (output / "frames.jsonl").open("x", encoding="utf-8") as stream:
        rec = HistoryRecorder(stream, receipt["target_board"])
        with contextlib.ExitStack() as stack:
            instrument_pipeline(stack, collector, rec)
            base.instrument_storage(stack, collector, rec)
            base.instrument_model_load(stack, rec, receipt["input_and_code_sha256"])
            base.instrument_video(stack, collector, rec)
            try:
                collector.collect_lean(Path(receipt["video_path"]), output / "not_written.npz", **kwargs)
            except base.CollectionFinished:
                pass
            else:
                raise RuntimeError("collectorが保存境界へ到達せず終了しました")
        stream.flush()
        os.fsync(stream.fileno())
    return rec


def run(args: argparse.Namespace) -> dict[str, Any]:
    """開始後にpipelineを作り直さず、履歴を含む観測を新規出力へ保存する。"""
    receipt, config = prepare(args)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    initial_cwd, initial_path = Path.cwd(), list(sys.path)
    try:
        os.chdir(base.SNAPSHOT)
        collector = base.load_collector()
        kwargs = base.collection_arguments(collector, config)
        receipt["original_collector_kwargs"] = base.json_value(kwargs)
        kwargs.update(start_sec=START_SEC, max_sec=END_SEC - START_SEC, precise_seek=False)
        if kwargs.get("sample_interval_sec") != 0 or kwargs.get("normalize_fps_30") is not True:
            raise RuntimeError("原本の30fps連続観測条件が一致しません")
        receipt.update(actual_collector_kwargs=base.json_value(kwargs),
                       diagnostic_argv=sys.argv, seek_policy="absolute_frame_clock_checked",
                       initial_src_modules=frozen_modules())
        base.write_json(output / "PLAN.json", receipt)
        started = time.perf_counter()
        rec = collect(output, receipt, collector, kwargs)
        return finish(output, receipt, rec, time.perf_counter() - started)
    finally:
        os.chdir(initial_cwd)
        sys.path[:] = initial_path


def main() -> int:
    """CPUのスレッド過多を抑えた独立CUDA診断として起動する。"""
    import cv2
    import torch

    cv2.setNumThreads(THREADS)
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(THREADS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, default=START_SEC)
    parser.add_argument("--end-sec", type=float, default=END_SEC)
    parser.add_argument("--allow-native-runtime-mismatch", action="store_true")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
