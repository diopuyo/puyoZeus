"""read_board の Rust 経路 (live_native_read_board) の同一性検証台と、同じ入力での時間測定。

比べる相手 (基準) は、配信 C2 の構成そのもの: 一括経路 (live_board_vectorized) + ネイティブ HSV で、
Rust 経路だけが OFF。Rust 経路は一括経路の 1st pass の置き換えなので、基準と ON の差は
`first_pass` (盤面 1 枚の 1st pass) と、それが後段へ渡す値だけに現れる。

モード:
  firstpass : 基準の `first_pass` の呼出に毎回 Rust 経路を並走させ、盤・セル一覧・bg 距離 (ビット単位)・
              NCC (numpy との差の最大、閾値からの最小の余裕、判定の一致) を全セルで比べる。
              本番と同じ入力 (背景指紋・ROI 補正・skip_tier1・彩度較正) がそのまま入る。
  lockstep  : 基準と ON の 2 本のパイプラインを同じコマで回し、PipelineResult・盤面・CNN 確率・HSV 画素・
              内部の記録・状態を生バイトで比べる (scripts/verify_live_board_vectorized.py の比較器を使う)。
  bench     : 同じコマ列で 2 本を交互に回して認識の時間を測り、さらに同じ入力の `first_pass` を何度も
              呼んで 1 回あたりの時間 (P50/P95) を出す。
  vote      : LiveImages が使う 2 つの分類器 (中央値・投票) を、実コマの全セルで Python 版と Rust 版に
              1 セルずつ照合する (色コードが全件一致、時間は 1 セルあたり)。

測定器の健全性 (fail-silent 防止):
  - selftest: 比較器が 1 セル・1 ulp・NCC 判定の違いを必ず検出すること
  - 陽性対照 (--perturb): 閾値をずらした Rust 経路が、firstpass で必ず不一致を出すこと
  - control: 基準同士 (lockstep --control) の違いが 0 であること

使い方 (配布 Python・配布 app の models を指す --assets。実行中の cwd は app になる):
  python verify_live_native_read_board.py firstpass --assets APP --config recognition_config.json \
      --video VIDEO --start 3900 --count 6000 --stride 1 --crop 1557x876 --color-correction 601to709 --output out.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import cv2
import numpy as np

# 作業木の src を配布 app の src より先に読む (配布 Python は _pth で app を探索路に持つ)。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- 定数 ----
FRAME_SHAPE = (1080, 1920, 3)
REPORT_MISMATCH_LIMIT = 50
PERCENTILES = (50, 95)
PERTURB_THRESHOLD_SHIFT = 0.02       # 陽性対照: NCC 閾値をこれだけ下げる
PERTURB_BG_SHIFT = 5.0               # 陽性対照: bg 閾値をこれだけ下げる
MICRO_INPUTS = 40                    # 1 回あたりの時間を測る入力の数 (メモリ約 0.5GB)
MICRO_REPEATS = 20
STATE_EVERY_FRAMES = 20
ENV_NATIVE_READ_BOARD = "PUYO_LIVE_NATIVE_READ_BOARD"
ENV_VECTORIZED = "PUYO_LIVE_BOARD_VECTORIZED"
ENV_NATIVE_HSV = "PUYO_LIVE_NATIVE_HSV_MAIN"


# ----------------------------------------------------------------------------
# 入力コマ (ffmpeg のパイプ。配布 Python の cv2 は mp4 を開けないため)
# ----------------------------------------------------------------------------

def iter_frames(options: argparse.Namespace) -> Iterator[tuple[int, float, np.ndarray]]:
    """録画の連続コマ (stride 間隔で採用) を、配信と同じ前処理 (切り出し→1080p→601to709) で返す。"""
    ffmpeg = options.ffmpeg or shutil.which("ffmpeg")
    assert ffmpeg, "ffmpeg が見つかりません (--ffmpeg で指定)"
    command = [ffmpeg, "-v", "error", "-ss", f"{options.start / 30.0:.6f}", "-i", str(options.video),
               "-frames:v", str(options.count), "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1 << 24)
    size = int(np.prod(FRAME_SHAPE))
    try:
        for i in range(options.count):
            raw = process.stdout.read(size)
            if len(raw) < size:
                break
            if i % options.stride:
                continue
            number = options.start + i
            yield number, number / 30.0, preprocess(np.frombuffer(raw, dtype=np.uint8).reshape(FRAME_SHAPE), options)
    finally:
        process.kill()


def preprocess(image: np.ndarray, options: argparse.Namespace) -> np.ndarray:
    """録画のゲーム画面の切り出し (配信入力の再現) と色補正。"""
    if options.crop:
        width, height = options.crop
        image = cv2.resize(image[:height, :width], (FRAME_SHAPE[1], FRAME_SHAPE[0]), interpolation=cv2.INTER_AREA)
    if options.color_correction == "601to709":
        from src.phase_j.dshow_color import correct_601_as_709
        image = correct_601_as_709(image)
    return np.ascontiguousarray(image)


# ----------------------------------------------------------------------------
# パイプラインの構築
# ----------------------------------------------------------------------------

def build_pipeline(config_path: Path, native_read_board: bool) -> Any:
    """配信の実効認識設定で、Rust 経路の旗だけ違うパイプラインを作る (一括経路・ネイティブ HSV は ON)。

    cwd は呼出側が配布 app にしておく (ONNX の相対パス models/onnx の解決が cwd 依存のため)。
    """
    from src.phase_j import live_speed
    from src.recognition_pipeline import RecognitionPipeline
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("enable_native_hsv_classifier", "enable_live_board_vectorized", "enable_live_native_read_board"):
        config.pop(key, None)
    names = {ENV_VECTORIZED: True, ENV_NATIVE_HSV: True, ENV_NATIVE_READ_BOARD: native_read_board}
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name, enabled in names.items():
            os.environ[name] = str(int(enabled))
        pipe = RecognitionPipeline.load_default(**live_speed.recognition_config(config, True))
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    assert pipe._native_hsv_active and pipe._live_board_vectorized_active
    assert getattr(pipe, "_live_native_read_board_active", False) is native_read_board
    assert getattr(pipe._reader._classifier, "_cnn", None) is not None, "CNN が読み込まれていません"
    return pipe


# ----------------------------------------------------------------------------
# firstpass: 基準の first_pass に Rust 経路を並走させて全セルを比べる
# ----------------------------------------------------------------------------

def float_bits(value: float) -> bytes:
    return struct.pack("<d", float(value))


def reference_ncc(bg_fp: Any, table: Any, index: int, hsv_patch: np.ndarray) -> float | None:
    """従来経路と同じ NCC (CellPatchFingerprint.ncc_to)。tier1 を実行しないセルは None。"""
    from src.background_fingerprint import BG_PATCH_VALID_V_MIN, CellPatchFingerprint
    from src.board import HIDDEN_ROWS
    row, col = table.positions[index]
    cell = bg_fp.cell_at_patch(row - HIDDEN_ROWS, col)
    if cell.v_median < BG_PATCH_VALID_V_MIN:
        return None
    return float(CellPatchFingerprint(patch_hsv=hsv_patch.astype(np.float32)).ncc_to(cell))


class FirstPassComparer:
    """`live_board_vectorized.first_pass` を包み、基準の結果に Rust 経路の結果を毎回照合する。"""

    def __init__(self, original: Any, perturb: bool) -> None:
        self.original = original
        self.perturb = perturb
        self.totals: Counter = Counter()
        self.mismatches: list[dict[str, Any]] = []
        self.ncc_max_abs_diff = 0.0
        self.ncc_min_margin = float("inf")
        self.distance_checked = 0
        self.calls = 0

    def __call__(self, reader: Any, frame: np.ndarray, region: Any, bg_fp: Any,
                 hsv_full: np.ndarray | None, skip_tier1: bool, board: Any) -> Any:
        cells = self.original(reader, frame, region, bg_fp, hsv_full, skip_tier1, board)
        self.calls += 1
        self.compare(reader, frame, region, bg_fp, hsv_full, skip_tier1, board, cells)
        return cells

    def note(self, kind: str, **detail: Any) -> None:
        self.totals["mismatch_" + kind] += 1
        if len(self.mismatches) < REPORT_MISMATCH_LIMIT:
            self.mismatches.append(dict(kind=kind, call=self.calls, **detail))

    def compare(self, reader: Any, frame: np.ndarray, region: Any, bg_fp: Any,
                hsv_full: np.ndarray | None, skip_tier1: bool, board: Any, cells: Any) -> None:
        from src import live_native_read_board as native
        from src.board import Board
        from src.live_board_vectorized import cell_rect_table
        reason = native.unsupported_reason(reader, frame, bg_fp, hsv_full, region)
        if reason is not None or cells is None:
            self.totals["skipped_" + (reason or "reference_fell_back")] += 1
            return
        table = cell_rect_table(region, frame.shape[1], frame.shape[0])
        arrays = native.table_arrays(reader, table)
        pack = native.bg_pack(reader, bg_fp, arrays)
        status, distance, ncc = native.call_native(reader, frame, hsv_full, arrays, pack, skip_tier1)
        mine = Board()
        mine_cells = native.materialize(frame, hsv_full, arrays, status, distance, mine)
        self.totals["calls_compared"] += 1
        self.totals["skip_tier1_calls"] += int(skip_tier1)
        self.compare_outputs(board, cells, mine, mine_cells, status)
        self.compare_ncc(reader, bg_fp, table, hsv_full, arrays, ncc, status, skip_tier1)

    def compare_outputs(self, board: Any, cells: list, mine: Any, mine_cells: list, status: np.ndarray) -> None:
        self.totals["board_cells_compared"] += board._grid.size
        if board._grid.tobytes() != mine._grid.tobytes():
            self.note("board", diff=int((board._grid != mine._grid).sum()))
        for code, name in ((0, "classify"), (1, "tier1_empty"), (2, "tier2_empty")):
            self.totals["status_" + name] += int((status == code).sum())
        if len(cells) != len(mine_cells):
            self.note("cell_count", reference=len(cells), native=len(mine_cells))
            return
        for ref, got in zip(cells, mine_cells):
            self.totals["cells_compared"] += 1
            same = (ref[0], ref[1]) == (got[0], got[1]) and same_view(ref[2], got[2]) and same_view(ref[4], got[4])
            self.distance_checked += 1
            if not same or float_bits(ref[3]) != float_bits(got[3]):
                self.note("cell", position=[ref[0], ref[1]], reference_distance=ref[3], native_distance=got[3])

    def compare_ncc(self, reader: Any, bg_fp: Any, table: Any, hsv_full: np.ndarray, arrays: Any,
                    ncc: np.ndarray, status: np.ndarray, skip_tier1: bool) -> None:
        from src import background_fingerprint as bf
        threshold = bf.PATCH_NCC_EMPTY_THRESHOLD if reader._patch_ncc_threshold is None else reader._patch_ncc_threshold
        for index, value in enumerate(ncc.tolist()):
            x1, y1, x2, y2 = arrays.rects[index].tolist()
            expected = None if skip_tier1 else reference_ncc(bg_fp, table, index, hsv_full[y1:y2, x1:x2])
            if (expected is None) != (value != value):
                self.note("ncc_presence", cell=index, native=value, reference=expected)
                continue
            if expected is None:
                continue
            self.totals["ncc_cells"] += 1
            delta = abs(value - expected)
            self.ncc_max_abs_diff = max(self.ncc_max_abs_diff, delta)
            self.ncc_min_margin = min(self.ncc_min_margin, abs(expected - threshold))
            self.totals["ncc_bit_identical"] += int(float_bits(value) == float_bits(expected))
            if (value >= threshold) != (expected >= threshold):
                self.note("ncc_decision", cell=index, native=value, reference=expected)
            self.totals["tier1_candidates"] += int(expected >= threshold)

    def summary(self) -> dict[str, Any]:
        return dict(calls_seen=self.calls, totals=dict(self.totals), mismatch_count=sum(
            v for k, v in self.totals.items() if k.startswith("mismatch_")),
            ncc_max_abs_diff=self.ncc_max_abs_diff,
            ncc_min_margin_to_threshold=None if self.ncc_min_margin == float("inf") else self.ncc_min_margin,
            mismatches=self.mismatches)


def same_view(a: np.ndarray, b: np.ndarray) -> bool:
    """同じ画像の同じ位置の view か (アドレス・形・歩幅が同じ)。"""
    return (a.__array_interface__["data"][0] == b.__array_interface__["data"][0]
            and a.shape == b.shape and a.strides == b.strides)


def run_firstpass(options: argparse.Namespace) -> dict[str, Any]:
    from src import live_board_vectorized as vectorized
    from src import live_native_read_board as native
    vectorized.reset_stats()
    pipe = build_pipeline(options.config, native_read_board=False)
    if options.perturb:
        original_params = native.first_pass_params

        def perturbed(reader: Any, skip_tier1: bool) -> dict[str, Any]:
            values = original_params(reader, skip_tier1)
            values["ncc_threshold"] -= PERTURB_THRESHOLD_SHIFT
            values["bg_threshold"] -= PERTURB_BG_SHIFT
            return values
        native.first_pass_params = perturbed
    comparer = FirstPassComparer(vectorized.first_pass, options.perturb)
    vectorized.first_pass = comparer
    frames = 0
    for index, time_sec, frame in iter_frames(options):
        pipe.update(index, time_sec, frame)
        frames += 1
    vectorized.first_pass = comparer.original
    return dict(mode="firstpass", perturb=options.perturb, frames=frames, stride=options.stride,
                start=options.start, comparer=comparer.summary(), path_usage=dict(vectorized.STATS),
                selftest=selftest(), code_dir=code_dir())


# ----------------------------------------------------------------------------
# 測定器の自己検査
# ----------------------------------------------------------------------------

def selftest() -> dict[str, bool]:
    """比較の部品が 1 か所の違いを必ず検出し、同一入力では違いを出さないことを確かめる。"""
    a = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    view, other = a[1:3, 1:3], a[1:3, 0:2]
    return dict(
        identical_view_is_same=same_view(view, a[1:3, 1:3]),
        shifted_view_is_detected=not same_view(view, other),
        one_ulp_is_detected=float_bits(1.0) != float_bits(np.nextafter(1.0, 2.0)),
        equal_floats_have_equal_bits=float_bits(0.1 + 0.2) == float_bits(0.30000000000000004),
    )


def code_dir() -> str:
    import src
    return str(Path(src.__file__).resolve().parent)


# ----------------------------------------------------------------------------
# lockstep: 基準と ON を同じコマで回して、結果・盤面・確率・内部状態を生バイトで比べる
# ----------------------------------------------------------------------------

def run_lockstep(options: argparse.Namespace) -> dict[str, Any]:
    from scripts import verify_live_board_vectorized as base
    from src import live_board_vectorized as vectorized
    vectorized.reset_stats()
    pipes = [build_pipeline(options.config, native_read_board=flag) for flag in (False, not options.control)]
    totals: dict[str, int] = {}
    mismatches: list[dict[str, Any]] = []
    frames = 0
    for index, time_sec, frame in iter_frames(options):
        with_state = frames % STATE_EVERY_FRAMES == 0
        left, right = (base.observe(p, index, time_sec, frame, with_state, tuple(options.exclude)) for p in pipes)
        counts, diffs = base.compare_observations(left, right)
        for name, count in counts.items():
            totals[name] = totals.get(name, 0) + count
        mismatches.extend(diffs)
        frames += 1
    return dict(mode="lockstep", control=bool(options.control), frames=frames, compared_values=totals,
                compared_total=sum(totals.values()), mismatch_count=len(mismatches),
                mismatch_groups=mismatch_groups(mismatches), excluded_state_names=list(options.exclude),
                mismatches=mismatches[:REPORT_MISMATCH_LIMIT], selftest=base.selftest(),
                path_usage=dict(vectorized.STATS), code_dir=code_dir())


def mismatch_groups(mismatches: list[dict[str, Any]]) -> dict[str, int]:
    """違いを「節: パスの先頭 3 要素 (添字は [])」でまとめた件数 (上位 20)。どこが違うかを 50 件の見本に頼らず示す。"""
    groups: Counter = Counter()
    for item in mismatches:
        path = ".".join(re.sub(r"\[\d+\]", "[]", item["path"]).split(".")[:3])
        groups[f"{item['section']}:{path}"] += 1
    return dict(groups.most_common(20))


# ----------------------------------------------------------------------------
# bench: 認識の時間と、同じ入力の first_pass の 1 回あたりの時間
# ----------------------------------------------------------------------------

def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return dict(n=0)
    return dict(n=len(values), mean_ms=statistics.fmean(values),
                p50_ms=float(np.percentile(values, PERCENTILES[0])),
                p95_ms=float(np.percentile(values, PERCENTILES[1])), max_ms=max(values))


def attach_read_timers(pipe: Any) -> dict[str, list[float]]:
    """read_both_boards の呼出時間 (ms) を貯める (インスタンスだけ差し替える)。"""
    store: dict[str, list[float]] = {"read_both_boards": []}
    reader = pipe._reader
    original = reader.read_both_boards

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            store["read_both_boards"].append((perf_counter() - started) * 1000)
    reader.read_both_boards = wrapper
    return store


def collect_micro_inputs(pipe: Any, options: argparse.Namespace) -> list[tuple]:
    """本番の呼出の引数を一定間隔で保存する (frame・hsv は次のコマで書き換わりうるので複製する)。"""
    from src import live_board_vectorized as vectorized
    saved: list[tuple] = []
    original = vectorized.first_pass
    counter = {"n": 0}

    def tap(reader: Any, frame: np.ndarray, region: Any, bg_fp: Any, hsv_full: Any, skip: bool, board: Any) -> Any:
        counter["n"] += 1
        if (bg_fp is not None and hsv_full is not None and len(saved) < MICRO_INPUTS
                and counter["n"] % options.micro_every == 0):
            saved.append((reader, frame.copy(), region, bg_fp, hsv_full.copy(), skip))
        return original(reader, frame, region, bg_fp, hsv_full, skip, board)
    vectorized.first_pass = tap
    try:
        for index, time_sec, frame in iter_frames(options):
            pipe.update(index, time_sec, frame)
            if len(saved) >= MICRO_INPUTS:
                break
    finally:
        vectorized.first_pass = original
    return saved


def run_micro(pipe: Any, inputs: list[tuple]) -> dict[str, Any]:
    """同じ入力で、基準 (一括 Python) と Rust 経路の first_pass を交互に何度も測る (盤面 1 枚あたり)。"""
    from src import live_board_vectorized as vectorized
    from src.board import Board
    reader = pipe._reader
    samples: dict[str, list[float]] = {"reference": [], "native": []}
    for _, frame, region, bg_fp, hsv_full, skip in inputs:
        for repeat in range(MICRO_REPEATS):
            for mode in (("reference", "native") if repeat % 2 == 0 else ("native", "reference")):
                reader._live_native_read_board = mode == "native"
                started = perf_counter()
                cells = vectorized.first_pass(reader, frame, region, bg_fp, hsv_full, skip, Board())
                samples[mode].append((perf_counter() - started) * 1000)
                assert cells is not None
    reader._live_native_read_board = False
    return {name: summarize(values) for name, values in samples.items()}


BENCH_ARMS = {"off": False, "on": True}


def measure_pipelines(options: argparse.Namespace, pipes: dict[str, Any],
                      stores: dict[str, dict[str, list[float]]]) -> tuple[dict[str, list[float]], list[float]]:
    """同じコマ列を腕ごとに交互の順で回し、update の時間 (ms) と、両腕があれば on - off の差を返す。"""
    update_ms: dict[str, list[float]] = {name: [] for name in pipes}
    paired: list[float] = []
    for count, (index, time_sec, frame) in enumerate(iter_frames(options)):
        if count == options.warmup:
            for name in pipes:
                update_ms[name].clear()
                stores[name]["read_both_boards"].clear()
        order = tuple(pipes) if count % 2 == 0 else tuple(reversed(tuple(pipes)))
        elapsed: dict[str, float] = {}
        for name in order:
            started = perf_counter()
            pipes[name].update(index, time_sec, frame)
            elapsed[name] = (perf_counter() - started) * 1000
            update_ms[name].append(elapsed[name])
        if count >= options.warmup and len(elapsed) == len(BENCH_ARMS):
            paired.append(elapsed["on"] - elapsed["off"])
    return update_ms, paired


def run_bench(options: argparse.Namespace) -> dict[str, Any]:
    from src import live_native_read_board as native
    pipes = {name: build_pipeline(options.config, flag) for name, flag in BENCH_ARMS.items() if name in options.arms}
    stores = {name: attach_read_timers(pipe) for name, pipe in pipes.items()}
    update_ms, paired = measure_pipelines(options, pipes, stores)
    arms = {name: dict(update=summarize(update_ms[name]), read_both_boards=summarize(stores[name]["read_both_boards"]))
            for name in pipes}
    micro_options = argparse.Namespace(**{**vars(options), "start": options.micro_start, "count": options.micro_count})
    inputs = collect_micro_inputs(pipes["off"], micro_options) if "off" in pipes and native.native_available() else []
    return dict(
        mode="bench", frames_measured=len(paired), warmup=options.warmup, start=options.start,
        stride=options.stride, python=sys.version.split()[0], puyo_core=_puyo_core_path(),
        paired_update_diff_mean_ms=statistics.fmean(paired) if paired else None,
        micro_inputs=len(inputs), micro_repeats=MICRO_REPEATS,
        micro_first_pass_per_board=run_micro(pipes["off"], inputs) if inputs else None,
        unit="update/read_both_boards = 1 コマあたり ms (壁時計)。micro = 盤面 1 枚の first_pass 1 回あたり ms",
        **arms)


def _puyo_core_path() -> str | None:
    try:
        import puyo_core
        return str(Path(puyo_core.__file__).resolve())
    except (ImportError, OSError):
        return None


# ----------------------------------------------------------------------------
# vote: LiveImages の分類器 (中央値・投票) の Python 版と Rust 版を全セルで照合
# ----------------------------------------------------------------------------

def make_classifier_pairs() -> dict[str, tuple[Any, Any]]:
    """(Python 版, Rust 版) の対。Rust 版が有効にならなければ検証にならないので失敗させる。"""
    from src.image_reader import ColorClassifier
    pairs = {"median": (ColorClassifier(), ColorClassifier()),
             "vote": (ColorClassifier(vote_mode=True), ColorClassifier(vote_mode=True))}
    assert pairs["median"][1].enable_native_hsv(True), "ネイティブ HSV が有効にならない"
    assert pairs["vote"][1].enable_native_vote(True), "ネイティブ投票が有効にならない"
    return pairs


def run_vote(options: argparse.Namespace) -> dict[str, Any]:
    from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
    from src.live_board_vectorized import cell_rect_table, slice_patches
    pairs = make_classifier_pairs()
    compared: Counter = Counter()
    classes: dict[str, Counter] = {name: Counter() for name in pairs}
    seconds: dict[str, list[float]] = {f"{name}_{side}": [0.0, 0] for name in pairs for side in ("python", "native")}
    mismatches: list[dict[str, Any]] = []
    frames = 0
    for index, _time, frame in iter_frames(options):
        frames += 1
        for region in (DEFAULT_P1_REGION, DEFAULT_P2_REGION):
            patches = slice_patches(frame, cell_rect_table(region, frame.shape[1], frame.shape[0]))
            for name, (python_side, native_side) in pairs.items():
                for cell, patch in enumerate(patches):
                    started = perf_counter()
                    expected = int(python_side.classify(patch))
                    middle = perf_counter()
                    actual = int(native_side.classify(patch))
                    ended = perf_counter()
                    seconds[f"{name}_python"][0] += middle - started
                    seconds[f"{name}_native"][0] += ended - middle
                    seconds[f"{name}_python"][1] += 1
                    seconds[f"{name}_native"][1] += 1
                    compared[name] += 1
                    classes[name][expected] += 1
                    if expected != actual and len(mismatches) < REPORT_MISMATCH_LIMIT:
                        mismatches.append(dict(frame=index, classifier=name, cell=cell, python=expected, native=actual))
                    elif expected != actual:
                        compared["mismatch_beyond_report_limit"] += 1
    per_cell_us = {key: (total / count * 1e6 if count else None) for key, (total, count) in seconds.items()}
    return dict(mode="vote", frames=frames, stride=options.stride, compared_cells=dict(compared),
                mismatch_count=len(mismatches) + compared["mismatch_beyond_report_limit"],
                class_distribution={name: dict(sorted(c.items())) for name, c in classes.items()},
                per_cell_us=per_cell_us, mismatches=mismatches, selftest=selftest(), code_dir=code_dir())


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------

def parse_crop(text: str) -> tuple[int, int] | None:
    if not text:
        return None
    width, height = text.lower().split("x")
    return int(width), int(height)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("mode", choices=("firstpass", "lockstep", "bench", "vote", "selftest"))
    parser.add_argument("--assets", type=Path, help="配布 app (models/ を含む)。実行中の cwd にする")
    parser.add_argument("--config", type=Path, help="recognition_config.json")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--native-dir", type=Path, help="puyo_core パッケージを含むフォルダ (探索路の先頭へ)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--crop", type=parse_crop, default=None, help="例 1557x876 (録画の左上を 1080p へ拡大)")
    parser.add_argument("--color-correction", default="", choices=("", "601to709"))
    parser.add_argument("--perturb", action="store_true", help="陽性対照: 閾値をずらした Rust 経路が不一致を出すこと")
    parser.add_argument("--control", action="store_true", help="lockstep: 基準同士を比べる (測定器の雑音)")
    parser.add_argument("--exclude", nargs="*", default=[], help="状態の比較から除く属性名の断片")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--arms", nargs="+", default=["off", "on"], choices=("off", "on"),
                        help="bench: 測る腕 (基点の pyd には Rust 経路が無いので off だけ)")
    parser.add_argument("--micro-start", type=int, default=0)
    parser.add_argument("--micro-count", type=int, default=3000)
    parser.add_argument("--micro-every", type=int, default=7)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def configure_runtime() -> None:
    """配信の認識 process と同じ実行環境にする (src/phase_j/launcher.py: ONNX の CNN、live_cpu.py: 各ライブラリ 1 スレッド)。

    setdefault なので、環境変数を先に指定すれば (例: PUYO_CNN_BACKEND=torch) そちらが優先される。
    """
    os.environ.setdefault("PUYO_CNN_BACKEND", "onnx")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    if options.mode == "selftest":
        print(json.dumps(selftest()))
        return 0
    configure_runtime()
    cv2.setNumThreads(1)
    if options.native_dir is not None:
        sys.path.insert(0, str(options.native_dir.resolve()))
    for name in ("config", "video", "output"):
        value = getattr(options, name)
        if value is not None:
            setattr(options, name, value.resolve())
    os.chdir(options.assets)
    runners = dict(firstpass=run_firstpass, lockstep=run_lockstep, bench=run_bench, vote=run_vote)
    report = runners[options.mode](options)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if options.output:
        options.output.write_text(text, encoding="utf-8")
    print(text[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
