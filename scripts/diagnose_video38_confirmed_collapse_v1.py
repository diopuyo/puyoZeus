"""固定収集実装を短区間だけ計装し、2P confirmed縮退の代入経路を記録する。"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import inspect
import json
import linecache
import os
import sys
import time
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any, Callable, TextIO

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / ".runtime_snapshots/event_first30_observed_context_v5_2026-08-30"
VERIFY = ROOT / "data/verify"
RESULT = VERIFY / "event_source_v1_first30_observed_context_v5_2026-08-30/results/target=38.json"
SOURCE_NPZ = VERIFY / ("event_source_v1_48pilot_full_2026-08-29/targets/target=38/"
                       "attempt=event-pilot-48-v1-38/retry=0000/work/collection.npz")
FORMAT = "video38-confirmed-collapse-diagnostic/v1"
START_SEC, END_SEC = 560.0, 583.0
TARGET_FRAME, TARGET_SIDE, TARGET_ROW = 34702, "2P", 829
MAX_DURATION_SEC, SMALL_COLOR_LIMIT = 30.0, 12
LAUNCHER = ROOT / "scripts/launch_video38_confirmed_collapse_v1.sh"


class CollectionFinished(Exception):
    """元収集の保存開始直前に正常終了し、NPZと事後sidecarを作らない。"""


def sha256(path: Path) -> str:
    """大きい動画もメモリを圧迫せずハッシュ化する。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """固定receiptを読み込む。"""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """既存成果物を置換せず排他保存する。"""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def json_value(value: Any) -> Any:
    """実行引数等を決定的なJSON型へ変換する。"""
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def board_value(board: Any) -> dict[str, Any] | None:
    """盤面を複製し、色・お邪魔・unknownと内容SHAを記録する。"""
    if board is None:
        return None
    grid = np.asarray(getattr(board, "_grid", board), dtype=np.int8)
    if grid.shape != (13, 6):
        raise ValueError("診断盤面のshapeが13x6ではありません")
    return {"grid": grid.tolist(), "color": int(np.isin(grid, (1, 2, 3, 4, 5)).sum()),
            "garbage": int((grid == 9).sum()), "unknown": int((grid == 10).sum()),
            "sha256": hashlib.sha256(grid.tobytes()).hexdigest()}


def assert_unchanged(hashes: dict[str, str]) -> None:
    """開始後に入力・コードが変わった実行を完成扱いしない。"""
    for name, expected in hashes.items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"診断中に保護assetが変更されました: {name}")


def runtime_guard(manifest: dict[str, Any], hashes: dict[str, str], allow_mismatch: bool = False) -> dict[str, Any]:
    """旧receiptがpackageのみをhashした限界を明示し、実extensionを前後固定する。"""
    import platform
    import torch
    spec = importlib.util.find_spec("puyo_core")
    if spec is None or spec.origin is None:
        raise RuntimeError("元収集に必要なpuyo_coreがありません")
    paths = list(Path(spec.origin).parent.glob("*.so"))
    expected = manifest["runtime"]["native_puyo_core_sha256"]
    if len(paths) != 1:
        raise ValueError("ネイティブ物理ライブラリの実体が一意ではありません")
    package_sha, extension_sha = sha256(Path(spec.origin)), sha256(paths[0])
    if not allow_mismatch:
        raise ValueError("当時extension SHA未記録: 明示allow-native-runtime-mismatchが必要です")
    hashes[str(paths[0])] = extension_sha
    hashes[spec.origin] = package_sha
    return {"python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "thread_environment": {name: os.environ.get(name) for name in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "original": manifest["runtime"], "native_identity": {
                "original_receipt_field_sha256": expected, "original_receipt_field_scope": "package___init__.py",
                "actual_package_sha256": package_sha, "package_matches_original": package_sha == expected,
                "actual_extension_path": str(paths[0]), "actual_extension_sha256": extension_sha,
                "original_extension_sha256": None, "original_extension_identity_verified": False,
                "allow_native_runtime_mismatch": allow_mismatch,
                "reason": "original receipt hashes package __init__.py, not native extension",
                "scope": "snapshot Python + current native probe; not original-environment reproduction"}}


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """原本の設定とsnapshot全assetを確認し、現行との差も明示する。"""
    if args.output_root.exists():
        raise FileExistsError("診断出力は新規directory必須です")
    if not 0 < args.end_sec - args.start_sec <= MAX_DURATION_SEC:
        raise ValueError("診断区間は正の30秒以内が必要です")
    result = read_json(RESULT)
    run = Path(result["run_dir"])
    manifest, config = read_json(run / "manifest.json"), read_json(run / "recognition-config.json")
    hashes, differences = {}, []
    for record in manifest["code_artifacts"]:
        name, expected = record["relative_path"], record["sha256"]
        path = SNAPSHOT / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"固定snapshotが元収集assetと不一致: {name}")
        hashes[str(path)] = expected
        current = ROOT / name
        current_sha = sha256(current) if current.is_file() else None
        if current_sha != expected:
            differences.append({"path": name, "original_sha256": expected, "current_sha256": current_sha})
    video = ROOT / manifest["source"]["source_video_path"]
    for path, expected in ((video, manifest["source"]["source_video_sha256"]),
                           (SOURCE_NPZ, result["input_npz_sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"元動画/NPZがreceiptと不一致: {path}")
        hashes[str(path)] = expected
    paths = (RESULT, run / "manifest.json", run / "recognition-config.json", Path(__file__), LAUNCHER)
    hashes.update({str(path): sha256(path) for path in paths})
    with np.load(SOURCE_NPZ, allow_pickle=False) as source:
        if int(source["frame_idx"][TARGET_ROW]) != TARGET_FRAME or str(source["side"][TARGET_ROW]) != TARGET_SIDE:
            raise ValueError("元NPZ target行のframe/side不一致")
        target = board_value(source["grids"][TARGET_ROW])
    receipt = {"format_version": FORMAT, "input_and_code_sha256": hashes,
               "runtime": runtime_guard(manifest, hashes, args.allow_native_runtime_mismatch),
               "current_vs_original_differences": differences, "original_source": manifest["source"],
               "original_collection_tokens": config["collection_tokens"], "target_board": target,
               "target_frame": TARGET_FRAME, "target_npz_row": TARGET_ROW, "video_path": str(video),
               "initialization": "cold_at_start_then_interval_warmup; no_pre_start_recognition_history",
               "short_interval_failure_does_not_disprove_original_collapse": True,
               "training_performed": False, "source_modified": False, "video_created": False}
    return receipt, config


def load_collector() -> ModuleType:
    """固定snapshotをimport元に使い、現行srcとの混成を拒否する。"""
    if any(name == "src" or name.startswith("src.") for name in sys.modules):
        raise RuntimeError("src import前の独立プロセスで起動してください")
    sys.path.insert(0, str(SNAPSHOT))
    path = SNAPSHOT / "scripts/collect_boards_lean.py"
    spec = importlib.util.spec_from_file_location("_video38_frozen_collector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("固定collectorのロードに失敗しました")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def collection_arguments(collector: ModuleType, config: dict[str, Any]) -> dict[str, Any]:
    """元CLI parserで引数を解決し、認識を起動せずkwargsを捕捉する。"""
    captured: dict[str, Any] = {}
    original, argv = collector.collect_lean, sys.argv
    def capture(video: Path, output: Path, **kwargs: Any) -> int:
        captured.update(kwargs)
        return 0
    try:
        collector.collect_lean = capture
        sys.argv = ["frozen-collector", "--video", "data/frames/video_38.mp4",
                    "--out-npz", "diagnostic-unused.npz", *config["collection_tokens"]]
        collector.main()
    finally:
        collector.collect_lean, sys.argv = original, argv
    return captured


def patch(stack: contextlib.ExitStack, obj: Any, name: str, replacement: Any) -> None:
    """例外時も元descriptorを復元する一時差替え。"""
    original = inspect.getattr_static(obj, name)
    setattr(obj, name, replacement)
    stack.callback(setattr, obj, name, original)


class Recorder:
    """各frameの観測と、_step_side内でconfirmedが変わった実行行を保持する。"""

    def __init__(self, stream: TextIO, target: dict[str, Any]) -> None:
        self.stream, self.target = stream, target
        self.frame, self.time_sec = -1, 0.0
        self.capture: dict[int, dict[str, Any]] = {}
        self.previous: dict[int, tuple[Any, int]] = {}
        self.rows, self.frames, self.snapshots = 0, 0, 0
        self.exact_target_snapshots: list[dict[str, Any]] = []
        self.small_stable_frames: list[dict[str, Any]] = []
        self.model_loads: list[dict[str, Any]] = []
        self.pipeline_receipt: dict[str, Any] = {}
        self.step_code: Any = None
        self.decoded_frame: int | None = None

    def emit(self, value: dict[str, Any]) -> None:
        """副作用は新規診断JSONLへの追記だけに限定する。"""
        row = {"frame_idx": self.frame, "time_sec": self.time_sec, **value}
        self.stream.write(json.dumps(json_value(row), ensure_ascii=False, allow_nan=False) + "\n")
        self.rows += 1

    def begin_frame(self, frame: int, time_sec: float) -> None:
        """捕捉しなかったframeに前回rawを持ち越さない。"""
        self.frame, self.time_sec = frame, time_sec
        self.capture.clear()
        self.previous.clear()
        self.frames += 1

    def capture_raw(self, memory: Any, raw: Any, filtered: Any) -> None:
        """会計filterの前後を当該frame stamp付きで記録する。"""
        self.capture[id(memory)] = {"captured_frame": self.frame,
                                    "raw": board_value(raw), "filtered": board_value(filtered)}

    def record_side(self, pipeline: Any, side: str, result: Any) -> None:
        """raw欠測と、publishされたCNN/confirmedを別々に保存する。"""
        memory = getattr(pipeline, f"_stable_color_memory_{'1p' if side == '1P' else '2p'}")
        entry = self.capture.get(id(memory))
        state = getattr(result.state, "value", str(result.state))
        current = board_value(result.confirmed_board)
        fields = ("board_provenance", "board_none_reason", "answer_check_result", "score")
        chain = getattr(result, "chain_event", None)
        row = {"kind": "frame_side", "side": side, "state": state,
               "raw_captured_this_frame": entry is not None, "accounting_capture": entry,
               "cnn": board_value(result.cnn_board), "confirmed": current,
               "estimated": board_value(getattr(result, "estimated_board", None)),
               "chain": {key: getattr(chain, key, None) for key in
                         ("chain_count", "mechanism", "trigger_sec", "projected_end_sec")},
               **{key: getattr(result, key, None) for key in fields}}
        self.emit(row)
        if side == TARGET_SIDE and str(state).lower() == "stable" and current is not None:
            if current["color"] <= SMALL_COLOR_LIMIT:
                self.small_stable_frames.append({"frame_idx": self.frame, "color": current["color"],
                                                 "matches_target": current["sha256"] == self.target["sha256"]})

    def trace(self, frame: FrameType, event: str, arg: Any) -> Callable | None:
        """対象関数だけをline traceし、nested変更は呼出元行へ帰属させる。"""
        if frame.f_code is not self.step_code:
            return None
        if event == "call":
            self.previous.pop(id(frame), None)
            return self.trace
        if event not in ("line", "return"):
            return self.trace
        local = frame.f_locals
        ctx = local.get("ctx")
        if ctx is not None:
            board = getattr(ctx, "confirmed_board", None)
            grid = None if board is None else board._grid.tobytes()
            signature = (str(getattr(ctx, "state", None)), grid)
            previous = self.previous.get(id(frame))
            if previous is not None and previous[0] != signature:
                line = previous[1]
                self.emit({"kind": "context_change", "side": local.get("side"),
                           "after_caller_line": line, "next_line": frame.f_lineno,
                           "source": linecache.getline(frame.f_code.co_filename, line).strip(),
                           "function": frame.f_code.co_name, "state": signature[0],
                           "confirmed": board_value(board),
                           "chain_count": local.get("chain_count"),
                           "chain_event": str(local.get("chain_event")),
                           "final_board": board_value(local.get("final_board")),
                           "score_delta_self": local.get("score_d_for_self")})
            self.previous[id(frame)] = (signature, frame.f_lineno)
        if event == "return":
            self.previous.pop(id(frame), None)
        return self.trace

    def record_append(self, grid: np.ndarray, side: str, frame_idx: int, kwargs: dict[str, Any]) -> None:
        """実際に元collectorが採用したsnapshotを別種の行として記録する。"""
        board = board_value(grid)
        row = {"kind": "collector_snapshot", "side": side, "frame_idx": frame_idx,
               "board": board, "board_provenance": kwargs.get("board_provenance"),
               "chain_trigger_sec": kwargs.get("chain_trigger_sec"),
               "chain_mechanism": kwargs.get("mechanism"),
               "stable_persistence_confidence": kwargs.get("stable_persistence_confidence")}
        self.emit(row)
        self.snapshots += 1
        if side == TARGET_SIDE and board is not None and board["sha256"] == self.target["sha256"]:
            self.exact_target_snapshots.append({"frame_idx": frame_idx, "time_sec": self.time_sec})


def instrument_pipeline(stack: contextlib.ExitStack, collector: ModuleType, rec: Recorder) -> None:
    """毎frame捕捉と元load_default実引数・CUDA利用を計装する。"""
    from src import ojama_write_accounting as accounting
    cls = collector.RecognitionPipeline
    original_update, original_filter = cls.update, accounting.apply_ojama_write_accounting_filter
    original_load = inspect.getattr_static(cls, "load_default")
    rec.step_code = cls._step_side.__code__
    def update(self: Any, frame_idx: int, time_sec: float, frame: np.ndarray) -> Any:
        if rec.decoded_frame != frame_idx:
            raise RuntimeError(f"復号frameとcollector時計が不一致: {rec.decoded_frame}/{frame_idx}")
        rec.begin_frame(frame_idx, time_sec)
        previous = sys.gettrace()
        try:
            sys.settrace(rec.trace)
            result = original_update(self, frame_idx, time_sec, frame)
        finally:
            sys.settrace(previous)
        rec.record_side(self, "1P", result.p1)
        rec.record_side(self, "2P", result.p2)
        return result
    def filtered(board: Any, memory: Any, credit: Any, *args: Any, **kwargs: Any) -> Any:
        raw = board._grid.copy()
        result = original_filter(board, memory, credit, *args, **kwargs)
        rec.capture_raw(memory, raw, result)
        return result
    def load(inner_cls: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_load.__func__(inner_cls, *args, **kwargs)
        cnn = getattr(result._reader._classifier, "_cnn", None)
        device = str(next(cnn._model.parameters()).device) if cnn is not None else "missing"
        rec.pipeline_receipt = {"load_default_kwargs": json_value(kwargs), "board_cnn_device": device,
                                "per_video_hsv_profile_autoload_enabled": False}
        if not device.startswith("cuda"):
            raise RuntimeError(f"指定CUDAの盤面CNNが使われていません: {device}")
        return result
    patch(stack, cls, "update", update)
    patch(stack, cls, "load_default", classmethod(load))
    patch(stack, accounting, "apply_ojama_write_accounting_filter", filtered)


def instrument_video(stack: contextlib.ExitStack, collector: ModuleType, rec: Recorder) -> None:
    """短区間seekの位置と各復号frameを元collectorの絶対frame時計へ照合する。"""
    cv2, original = collector.cv2, collector.cv2.VideoCapture
    class Capture:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.capture = original(*args, **kwargs)
        def __getattr__(self, name: str) -> Any:
            return getattr(self.capture, name)
        def set(self, key: int, value: float) -> bool:
            result = self.capture.set(key, value)
            actual = self.capture.get(key)
            if key == cv2.CAP_PROP_POS_FRAMES:
                if not result or abs(actual - value) > 0.1:
                    raise RuntimeError(f"動画frame seekに失敗: {value}/{actual}")
                rec.emit({"kind": "seek", "requested_frame": value, "actual_frame": actual})
            return result
        def read(self) -> tuple[bool, Any]:
            ok, frame = self.capture.read()
            if ok:
                rec.decoded_frame = round(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if rec.decoded_frame == TARGET_FRAME:
                    rec.emit({"kind": "target_decoded_frame", "actual_frame": rec.decoded_frame,
                              "pixels_sha256": hashlib.sha256(frame.tobytes()).hexdigest()})
            return ok, frame
    patch(stack, cv2, "VideoCapture", Capture)


def instrument_storage(stack: contextlib.ExitStack, collector: ModuleType, rec: Recorder) -> None:
    """元採用判定を維持し、NPZ・勝敗・事後sidecar保存のみ止める。"""
    cls, original = collector._LeanNpzAccumulator, collector._LeanNpzAccumulator.append
    def append(self: Any, grid: np.ndarray, video: str, side: str, seconds: float,
               game: int, frame: int, **kwargs: Any) -> Any:
        rec.record_append(grid, side, frame, kwargs)
        return original(self, grid, video, side, seconds, game, frame, **kwargs)
    def save(self: Any, path: Path) -> None:
        raise CollectionFinished()
    def skip_labels(*args: Any, **kwargs: Any) -> None:
        return None
    def skip_panels(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}
    patch(stack, cls, "append", append)
    patch(stack, cls, "save", save)
    patch(stack, cls, "assign_won_labels", skip_labels)
    patch(stack, collector, "_detect_panel_winners_crosscheck", skip_panels)


def instrument_model_load(stack: contextlib.ExitStack, rec: Recorder, hashes: dict[str, str]) -> None:
    """実際にロードしたCNN checkpointを固定assetへ照合する。"""
    import torch
    original = torch.load
    def load(path: Any, *args: Any, **kwargs: Any) -> Any:
        resolved = Path(path).resolve()
        digest = sha256(resolved)
        if hashes.get(str(resolved)) != digest:
            raise ValueError(f"guard範囲外のtorch checkpoint: {resolved}")
        rec.model_loads.append({"path": str(resolved), "sha256": digest,
                                "map_location": str(kwargs.get("map_location"))})
        return original(path, *args, **kwargs)
    patch(stack, torch, "load", load)


def finish(output: Path, receipt: dict[str, Any], rec: Recorder, elapsed: float) -> dict[str, Any]:
    """未再現と実行失敗を区別し、入力不変確認後だけCOMPLETEを作る。"""
    if not rec.frames or not rec.model_loads or not rec.pipeline_receipt:
        raise RuntimeError("診断frameまたは実CUDA/model receiptがありません")
    summary = {"format_version": FORMAT, "elapsed_sec": elapsed, "frame_count": rec.frames,
               "row_count": rec.rows, "collector_snapshot_count": rec.snapshots,
               "status": "target_grid_reproduced" if rec.exact_target_snapshots else "short_interval_not_reproduced",
               "exact_target_snapshots": rec.exact_target_snapshots,
               "target_grid_status_does_not_imply_original_frame_reproduction": True,
               "original_target_frame_reproduced": any(
                   row["frame_idx"] == TARGET_FRAME for row in rec.exact_target_snapshots),
               "small_stable_frames": rec.small_stable_frames,
               "model_loads": rec.model_loads, "pipeline": rec.pipeline_receipt,
               "native_runtime_identity": receipt["runtime"]["native_identity"],
               "root_cause_automatically_asserted": False, "scope_expanded": False}
    assert_unchanged(receipt["input_and_code_sha256"])
    write_json(output / "SUMMARY.json", summary)
    complete = {"format_version": FORMAT, "status": summary["status"],
                "sha256": {name: sha256(output / name) for name in
                           ("PLAN.json", "frames.jsonl", "SUMMARY.json")}}
    assert_unchanged(receipt["input_and_code_sha256"])
    write_json(output / "COMPLETE", complete)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    """固定実装・固定assetsで限定区間を実行し、全一時差替えを復元する。"""
    receipt, config = prepare(args)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    initial_cwd, initial_path = Path.cwd(), list(sys.path)
    try:
        os.chdir(SNAPSHOT)
        collector = load_collector()
        kwargs = collection_arguments(collector, config)
        original_kwargs = json_value(kwargs)
        kwargs.update(start_sec=args.start_sec, max_sec=args.end_sec - args.start_sec, precise_seek=False)
        receipt.update(original_collector_kwargs=original_kwargs, actual_collector_kwargs=json_value(kwargs),
                       diagnostic_argv=sys.argv, seek_policy="frame-index OpenCV seek; collector absolute frame clock",
                       posthoc_outcome_and_file_writes_intercepted=True)
        write_json(output / "PLAN.json", receipt)
        started = time.perf_counter()
        with (output / "frames.jsonl").open("x", encoding="utf-8") as stream:
            rec = Recorder(stream, receipt["target_board"])
            with contextlib.ExitStack() as stack:
                instrument_pipeline(stack, collector, rec)
                instrument_storage(stack, collector, rec)
                instrument_model_load(stack, rec, receipt["input_and_code_sha256"])
                instrument_video(stack, collector, rec)
                try:
                    collector.collect_lean(Path(receipt["video_path"]), output / "not_written.npz", **kwargs)
                except CollectionFinished:
                    pass
                else:
                    raise RuntimeError("元collectorが正常保存到達せず終了しました")
            stream.flush()
            os.fsync(stream.fileno())
        return finish(output, receipt, rec, time.perf_counter() - started)
    finally:
        os.chdir(initial_cwd)
        sys.path[:] = initial_path


def main() -> int:
    """実行前失敗でもCOMPLETEを作らず、既存出力を再利用しない。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, default=START_SEC)
    parser.add_argument("--end-sec", type=float, default=END_SEC)
    parser.add_argument("--allow-native-runtime-mismatch", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
