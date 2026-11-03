"""盤面読取の一括経路 (live_board_vectorized) の ON/OFF をビット単位で比較する検証台。

測るもの (どれも「違いが 0」でなければ不合格。許容誤差なし):
- パイプライン結果: RecognitionPipeline.update の PipelineResult の全フィールド
- 盤面: ImageReader.read_both_boards / read_board_hsv_only の盤面 (各セルの色)
- 確信度: CNN の各クラス確率 (生バイト) と classify_batch の色
- 内部の記録: ライブ窓の観測 (_live_raw_boards / _live_hsv_boards / _live_hsv_pixels)
- パイプライン内部の状態: 属性を再帰的にたどった値のハッシュ (較正の履歴・履歴バッファなど)

測定器の健全性 (fail-silent 防止):
1. selftest: 比較器が盤面 1 セル・確率 1 ビット・HSV 画素 1 つ・内部状態の最終桁の違いを必ず検出すること
2. control: OFF 同士を比べて違いが 0 であること (時刻など実行ごとに変わる値の除外が正しいこと)

使い方 (配布 Python と、配布 app のモデルを指す --assets。実行中の cwd は app になる):
  lockstep : 同一プロセスで OFF/ON 2 本を同じコマで回して比較 (--control で OFF 同士)
  dump     : 1 本を回して観測を保存 (OFF と ON を別プロセスで取り、compare で比べる)
  compare  : dump 2 つを比較
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import pickle
import struct
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

try:  # 基点 (一括経路の無いコミット) のコードに対しても、OFF の観測を取れるようにする
    from src import live_board_vectorized
except ImportError:  # pragma: no cover - 基点コミットでだけ起きる
    live_board_vectorized = None  # type: ignore[assignment]

# ---- 比較から除外する名前 ----
# 一括経路の有効フラグそのもの (ON/OFF で違うのが正しい)。全節で除外する。
FLAG_NAME_PARTS: tuple[str, ...] = ("vectorized",)
# 内部状態の節だけで除く、実行ごとに変わる値 (出力ではなく計時)。
# OFF 同士の control で違いが出た属性だけをここへ名前で足す (部分一致で広く除かない)。
STATE_NOISE_NAMES: frozenset[str] = frozenset({"_live_raw_board_sec"})
# 値としてたどらない型の所属モジュール (推論セッション・スレッド部品など)。
SKIP_MODULE_ROOTS: frozenset[str] = frozenset({
    "torch", "onnxruntime", "threading", "_thread", "io", "queue", "concurrent",
    "multiprocessing", "weakref", "ctypes", "logging",
})
MAX_DEPTH = 40
STATE_EVERY_FRAMES = 20
REPORT_MISMATCH_LIMIT = 50
SKIP_PATTERNS: tuple[tuple[bool, bool], ...] = (
    (False, False), (True, False), (False, True), (True, True),
)


# ----------------------------------------------------------------------------
# ハッシュ化 (値のバイト単位の同一性)
# ----------------------------------------------------------------------------

def _skip_type(obj: Any) -> bool:
    """値として比べない型 (関数・クラス・モジュール・推論セッション・スレッド部品など)。"""
    if isinstance(obj, type) or inspect.isroutine(obj) or inspect.ismodule(obj):
        return True
    return (type(obj).__module__ or "").split(".")[0] in SKIP_MODULE_ROOTS


def _hash_bytes(*parts: bytes) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


class DigestMap(dict):
    """{パス: ハッシュ}。elements は比べた「値」の個数 (配列は要素数、スカラは 1)。"""

    elements: int = 0


def _leaf(obj: Any) -> tuple[str, int] | None:
    """末端の値の (ハッシュ, 値の個数)。末端でなければ None。"""
    if obj is None or isinstance(obj, (bool, int, str, bytes)):
        return _hash_bytes(type(obj).__name__.encode(), repr(obj).encode("utf-8", "replace")), 1
    if isinstance(obj, float):
        return _hash_bytes(b"float", struct.pack("<d", obj)), 1
    if isinstance(obj, np.generic):
        return _hash_bytes(str(obj.dtype).encode(), obj.tobytes()), 1
    if isinstance(obj, np.ndarray) and obj.dtype != object:
        return _hash_bytes(str(obj.dtype).encode(), repr(obj.shape).encode(),
                           np.ascontiguousarray(obj).tobytes()), int(obj.size)
    if isinstance(obj, Enum):
        return _hash_bytes(type(obj).__name__.encode(), obj.name.encode()), 1
    return None


def _children(obj: Any) -> list[tuple[str, Any]]:
    """再帰でたどる子 (名前, 値)。順序は値によらず決まるようにする。"""
    if isinstance(obj, dict):
        return [(f"[{k!r}]", v) for k, v in sorted(obj.items(), key=lambda kv: repr(kv[0]))]
    if isinstance(obj, (list, tuple)) or type(obj).__name__ == "deque":
        return [(f"[{i}]", v) for i, v in enumerate(obj)]
    if isinstance(obj, np.ndarray):
        return [(f"[{i}]", v) for i, v in enumerate(obj.ravel().tolist())]
    if isinstance(obj, (set, frozenset)):
        return [(f"{{{i}}}", v) for i, v in enumerate(sorted(obj, key=repr))]
    attributes = dict(getattr(obj, "__dict__", {}))
    for cls in type(obj).__mro__:
        for slot in getattr(cls, "__slots__", ()):
            if hasattr(obj, slot):
                attributes[slot] = getattr(obj, slot)
    return [(f".{k}", v) for k, v in sorted(attributes.items())]


def _excluded(attribute: str, noise_names: frozenset[str], extra_parts: tuple[str, ...]) -> bool:
    name = attribute.lstrip(".")
    lowered = name.lower()
    return (name in noise_names
            or any(part in lowered for part in FLAG_NAME_PARTS + extra_parts))


def digest_tree(
    obj: Any, root: str = "", noise_names: frozenset[str] = frozenset(),
    extra_parts: tuple[str, ...] = (),
) -> DigestMap:
    """オブジェクトを再帰でたどり、{パス: 値のハッシュ} を返す。属性は除外名を除く。"""
    out = DigestMap()
    seen: set[int] = set()

    def walk(value: Any, path: str, depth: int) -> None:
        leaf = _leaf(value)
        if leaf is not None:
            out[path] = leaf[0]
            out.elements += leaf[1]
            return
        if depth > MAX_DEPTH or _skip_type(value):
            return
        if id(value) in seen:
            out[path + "<cycle>"] = "cycle"
            return
        seen.add(id(value))
        for name, child in _children(value):
            if name.startswith(".") and _excluded(name, noise_names, extra_parts):
                continue
            walk(child, path + name, depth + 1)
        seen.discard(id(value))

    walk(obj, root, 0)
    return out


def diff_maps(left: dict[str, str], right: dict[str, str]) -> list[tuple[str, str, str]]:
    """2 つのハッシュ表の違い (パス, 左, 右)。片方にしか無いパスも違いとして数える。"""
    paths = sorted(set(left) | set(right))
    return [(p, left.get(p, "<none>"), right.get(p, "<none>"))
            for p in paths if left.get(p) != right.get(p)]


# ----------------------------------------------------------------------------
# 1 コマの観測
# ----------------------------------------------------------------------------

@dataclass
class Observation:
    """1 本のパイプラインについての 1 コマ分の観測 (ハッシュ表)。"""

    index: int
    sections: dict[str, DigestMap] = field(default_factory=dict)


def cell_patches(reader: Any, frame: np.ndarray) -> list[np.ndarray]:
    """検証用に独立して書いた切り出し (本番の矩形表を使わない)。1P→2P、可視行の行優先。"""
    from src.board import BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS
    patches = []
    for region in (reader._p1_region, reader._p2_region):
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            for col in range(BOARD_COLS):
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                patches.append(frame[y1:y2, x1:x2])
    return patches


def reader_section(reader: Any, frame: np.ndarray, skips: tuple[bool, bool]) -> dict[str, Any]:
    """盤面・HSV 単独盤面・CNN 確率・分類色・HSV 画素を、実際の呼出で取る。"""
    boards = reader.read_both_boards(frame, skip_tier1_1p=skips[0], skip_tier1_2p=skips[1])
    hsv_only = [reader.read_board_hsv_only(frame, region)
                for region in (reader._p1_region, reader._p2_region)]
    patches = cell_patches(reader, frame)
    classifier = reader._classifier
    cnn = getattr(classifier, "_cnn", None)
    section: dict[str, Any] = dict(
        boards=[b._grid for b in boards], hsv_only=[b._grid for b in hsv_only],
        colors=np.asarray(classifier.classify_batch(patches), dtype=np.int64),
        telop=reader._cached_telop_bbox,
        hsv_pixels=reader._board_hsv_frame(frame, (reader._p1_region, reader._p2_region)),
    )
    if cnn is not None:
        section["probabilities"] = cnn.predict_proba_batch(patches)
    return section


def observe(
    pipe: Any, index: int, time_sec: float, frame: np.ndarray,
    with_state: bool, extra_parts: tuple[str, ...] = (),
) -> Observation:
    """ライブ窓の記録を有効にして 1 コマ回し、結果・記録・読取器の出力・内部状態を取る。"""
    from src.phase_j.live_snapshot import prepare_recognition
    prepare_recognition(pipe)
    result = pipe.update(index, time_sec, frame)
    reader = pipe._reader
    obs = Observation(index)
    obs.sections["result"] = digest_tree(result, "result")
    live = dict(raw=reader._live_raw_boards, hsv=reader._live_hsv_boards,
                pixels=reader._live_hsv_pixels)
    obs.sections["live"] = digest_tree(live, "live")
    skips = SKIP_PATTERNS[index % len(SKIP_PATTERNS)]
    for key, value in reader_section(reader, frame, skips).items():
        obs.sections["reader." + key] = digest_tree(value, key)
    if with_state:
        obs.sections["state"] = digest_tree(pipe, "pipe", STATE_NOISE_NAMES, extra_parts)
    return obs


def compare_observations(
    left: Observation, right: Observation,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """2 つの観測を節ごとに比較し、(節ごとの比較した値の数, 違い) を返す。"""
    counts: dict[str, int] = {}
    mismatches: list[dict[str, Any]] = []
    for name in sorted(set(left.sections) | set(right.sections)):
        a, b = left.sections.get(name, DigestMap()), right.sections.get(name, DigestMap())
        counts[name] = max(a.elements, b.elements)
        for path, x, y in diff_maps(a, b):
            mismatches.append(dict(frame=left.index, section=name, path=path, left=x, right=y))
    return counts, mismatches


# ----------------------------------------------------------------------------
# 測定器の自己検査
# ----------------------------------------------------------------------------

def selftest() -> dict[str, bool]:
    """比較器が 1 か所の違いを必ず検出し、同一入力では違いを出さないことを確かめる。"""
    grid = np.zeros((13, 6), dtype=np.uint8)
    probs = np.linspace(0, 1, 28, dtype=np.float32).reshape(4, 7)
    pixels = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    state = dict(history=[1.0, 2.0, 3.0], note="x")

    def make(g: np.ndarray, p: np.ndarray, px: np.ndarray, s: dict) -> dict[str, str]:
        return digest_tree(dict(board=g, probabilities=p, pixels=px, state=s))

    base = make(grid, probs, pixels, state)
    changed_grid = grid.copy()
    changed_grid[3, 2] = 1
    changed_probs = probs.copy()
    changed_probs.view(np.uint32)[1, 1] ^= 1
    changed_pixels = pixels.copy()
    changed_pixels[1, 2, 3] ^= 1
    changed_state = dict(history=[1.0, 2.0, 3.0000000000000004], note="x")
    return dict(
        identical_is_clean=not diff_maps(base, make(grid.copy(), probs.copy(), pixels.copy(), dict(state))),
        detects_cell=bool(diff_maps(base, make(changed_grid, probs, pixels, state))),
        detects_probability_bit=bool(diff_maps(base, make(grid, changed_probs, pixels, state))),
        detects_hsv_pixel=bool(diff_maps(base, make(grid, probs, changed_pixels, state))),
        detects_state_ulp=bool(diff_maps(base, make(grid, probs, pixels, changed_state))),
    )


# ----------------------------------------------------------------------------
# パイプラインの構築と入力コマ
# ----------------------------------------------------------------------------

def build_pipeline(config_path: Path, vectorized: bool, native: bool) -> Any:
    """配布の実配信設定 (recognition_config 経由) で、フラグだけ違うパイプラインを作る。

    cwd は呼び出し側が配布 app にしておく (ONNX の相対パス models/onnx の解決が cwd 依存のため)。
    """
    from src.phase_j import live_speed
    from src.recognition_pipeline import RecognitionPipeline
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("enable_native_hsv_classifier", None)
    board_key = getattr(live_speed, "BOARD_VECTORIZED_KEY", "live_board_vectorized")
    assert vectorized is False or live_board_vectorized is not None, "このコードに一括経路が無い"
    names = {"PUYO_" + board_key.upper(): vectorized,
             "PUYO_" + live_speed.NATIVE_HSV_MAIN_KEY.upper(): native}
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
    assert getattr(pipe, "_live_board_vectorized_active", False) is vectorized
    assert pipe._native_hsv_active is native
    assert getattr(pipe._reader._classifier, "_cnn", None) is not None, "CNN が読み込まれていません"
    return pipe


def obs_normalize(image: np.ndarray) -> np.ndarray:
    """23y の OBS 録画 (ゲーム画面が左上 1449x815) を、実機入力相当の 1080p・BT.709 へ直す。

    録画からの切り出し用 (obs23y_diag2 の measure3.normalized と同じ変換)。
    """
    def matrix(kr: float, kb: float) -> np.ndarray:
        kg = 1 - kr - kb
        return np.array([[1, 0, 2 * (1 - kr)],
                         [1, -2 * (1 - kb) * kb / kg, -2 * (1 - kr) * kr / kg],
                         [1, 2 * (1 - kb), 0]])
    m = (matrix(.2126, .0722) @ np.linalg.inv(matrix(.299, .114)))[::-1, ::-1].astype(np.float32)
    image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
    image = cv2.resize(image[:815, :1449], (1920, 1080), interpolation=cv2.INTER_AREA)
    return cv2.transform(image, m)


def iter_frames(options: argparse.Namespace) -> Iterator[tuple[int, float, np.ndarray]]:
    """動画の連続コマ、または PNG の一覧を (番号, 秒, BGR 1080p) で返す。"""
    if options.png:
        for i, path in enumerate(options.png):
            image = cv2.imread(str(path))
            assert image is not None and image.shape == (1080, 1920, 3), f"1080p の PNG ではない: {path}"
            yield i, i / 30.0, image
        return
    capture = cv2.VideoCapture(str(options.video), cv2.CAP_FFMPEG)
    assert capture.isOpened(), f"動画を開けません: {options.video}"
    capture.set(cv2.CAP_PROP_POS_FRAMES, options.start)
    for i in range(options.count):
        ok, image = capture.read()
        if not ok:
            break
        if options.obs_normalize:
            image = obs_normalize(image)
        number = options.start + i
        yield number, number / 30.0, image
    capture.release()


# ----------------------------------------------------------------------------
# 実行モード
# ----------------------------------------------------------------------------

def _accumulate(totals: dict[str, int], counts: dict[str, int]) -> None:
    for name, count in counts.items():
        totals[name] = totals.get(name, 0) + count


def run_lockstep(options: argparse.Namespace) -> dict[str, Any]:
    """同一プロセスで 2 本を同じコマで回して比較する (--control は OFF 同士)。"""
    if live_board_vectorized is not None:
        live_board_vectorized.reset_stats()
    pipes = [build_pipeline(options.config, vec, options.native)
             for vec in (False, not options.control)]
    totals: dict[str, int] = {}
    mismatches: list[dict[str, Any]] = []
    frames = 0
    for index, time_sec, frame in iter_frames(options):
        with_state = frames % STATE_EVERY_FRAMES == 0
        left, right = (observe(p, index, time_sec, frame, with_state, tuple(options.exclude))
                       for p in pipes)
        counts, diffs = compare_observations(left, right)
        _accumulate(totals, counts)
        mismatches.extend(diffs)
        frames += 1
    return summarize(options, frames, totals, mismatches)


def to_plain(obs: Observation) -> dict[str, Any]:
    """保存用に標準の型だけへ直す (実行形態によってクラスの pickle が通らないため)。"""
    return dict(index=obs.index, sections={name: (dict(m), m.elements) for name, m in obs.sections.items()})


def from_plain(row: dict[str, Any]) -> Observation:
    obs = Observation(row["index"])
    for name, (mapping, elements) in row["sections"].items():
        digest = DigestMap(mapping)
        digest.elements = elements
        obs.sections[name] = digest
    return obs


def run_dump(options: argparse.Namespace) -> dict[str, Any]:
    """1 本を回し、観測を保存する (OFF と ON を別プロセスで取るため)。"""
    pipe = build_pipeline(options.config, options.flag == "on", options.native)
    observations = []
    for count, (index, time_sec, frame) in enumerate(iter_frames(options)):
        observations.append(observe(pipe, index, time_sec, frame,
                                    count % STATE_EVERY_FRAMES == 0, tuple(options.exclude)))
    with options.dump.open("wb") as stream:
        pickle.dump([to_plain(o) for o in observations], stream)
    return dict(mode="dump", flag=options.flag, frames=len(observations), dump=str(options.dump),
                mismatch_count=0, selftest=selftest(), code_dir=code_dir(),
                has_vectorized_module=live_board_vectorized is not None)


def run_compare(options: argparse.Namespace) -> dict[str, Any]:
    """dump 2 つ (別プロセスで取った OFF と ON) を比較する。"""
    left, right = ([from_plain(row) for row in pickle.loads(path.read_bytes())]
                   for path in options.dumps)
    assert len(left) == len(right) and left, "コマ数が一致しません"
    totals: dict[str, int] = {}
    mismatches: list[dict[str, Any]] = []
    for a, b in zip(left, right):
        assert a.index == b.index
        counts, diffs = compare_observations(a, b)
        _accumulate(totals, counts)
        mismatches.extend(diffs)
    return summarize(options, len(left), totals, mismatches)


def summarize(
    options: argparse.Namespace, frames: int, totals: dict[str, int],
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    """母数 (比較した値の数) と違いの件数を必ず並べて出す。"""
    return dict(
        mode=options.mode, control=bool(getattr(options, "control", False)),
        native=bool(getattr(options, "native", True)), frames=frames,
        compared_values=totals, compared_total=sum(totals.values()),
        mismatch_count=len(mismatches), mismatches=mismatches[:REPORT_MISMATCH_LIMIT],
        selftest=selftest(),
        path_usage=dict(live_board_vectorized.STATS) if live_board_vectorized is not None else {},
        code_dir=code_dir(), cnn_model=cnn_model_info() if options.mode != "compare" else None,
    )


def code_dir() -> str | None:
    """この実行が使った本体コードの置き場 (基点コードとの比較でどちらを使ったかの記録)。"""
    module = sys.modules.get("src.recognition_pipeline")
    return str(Path(module.__file__).resolve().parents[1]) if module is not None else None


def cnn_model_info() -> dict[str, str]:
    """実際に読み込まれる既定 CNN (cwd 相対) の所在と SHA256。検証がどの重みで行われたかの記録。"""
    from src.recognition_pipeline import RecognitionPipeline
    path = RecognitionPipeline.DEFAULT_CNN_MODEL_PATH
    if not path.exists():
        return dict(path=str(path), sha256="missing")
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run_crops(options: argparse.Namespace) -> dict[str, Any]:
    """盤面だけを切り出した連続 PNG (例: 720x384) を read_board へ直接与えて ON/OFF を比べる。

    背景 FP は先頭 5 枚から作り、tier1/tier2・ガード・CNN・HSV の各分岐を踏ませる。
    切り出し画像は 1080p でないので read_board_hsv_only は従来経路へ戻る (その戻りも比べる)。
    """
    from src.background_fingerprint import capture_patch_robust_fingerprint
    from src.image_reader import BoardRegion, ImageReader
    crops = [cv2.imread(str(p)) for p in options.png]
    assert crops and all(c is not None for c in crops), "切り出し PNG を読めません"
    height, width = crops[0].shape[:2]
    region = BoardRegion(0, 0, width, height)
    background = capture_patch_robust_fingerprint(crops[:5], 0, 0, width, height)
    if live_board_vectorized is not None:
        live_board_vectorized.reset_stats()
    readers = []
    for vectorized in (False, not options.control):
        pipe = build_pipeline(options.config, vectorized, options.native)
        reader = ImageReader(
            classifier=pipe._reader._classifier, p1_region=region, p2_region=region,
            bg_fingerprint_p1=background, bg_fingerprint_p2=background, use_ui_mask=False,
            enable_patch_fp_hsv_guard=True)
        if vectorized:
            reader.enable_live_board_vectorized(True)
        readers.append(reader)
    totals: dict[str, int] = {}
    mismatches: list[dict[str, Any]] = []
    for index, crop in enumerate(crops):
        skip = bool(index % 3 == 1)
        observed = []
        for reader in readers:
            obs = Observation(index)
            board = reader.read_board(crop, region, skip_tier1=skip)
            hsv_only = reader.read_board_hsv_only(crop, region)
            patches = cell_patches_for(region, crop)
            section = dict(board=board._grid, hsv_only=hsv_only._grid,
                           colors=np.asarray(reader._classifier.classify_batch(patches), dtype=np.int64),
                           probabilities=reader._classifier._cnn.predict_proba_batch(patches))
            for key, value in section.items():
                obs.sections["crop." + key] = digest_tree(value, key)
            observed.append(obs)
        counts, diffs = compare_observations(*observed)
        _accumulate(totals, counts)
        mismatches.extend(diffs)
    return summarize(options, len(crops), totals, mismatches)


def cell_patches_for(region: Any, image: np.ndarray) -> list[np.ndarray]:
    """検証用の独立した切り出し (単一領域・可視行の行優先)。"""
    from src.board import BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS
    patches = []
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        for col in range(BOARD_COLS):
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            patches.append(image[y1:y2, x1:x2])
    return patches


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("mode", choices=("lockstep", "dump", "compare", "crops"))
    parser.add_argument("--assets", type=Path, help="配布 app (models/ を含む)。実行中の cwd にする")
    parser.add_argument("--config", type=Path, help="recognition_config.json")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--png", type=Path, nargs="*", default=[])
    parser.add_argument("--obs-normalize", action="store_true")
    parser.add_argument("--control", action="store_true", help="OFF 同士を比べる (測定器の雑音)")
    parser.add_argument("--no-native", dest="native", action="store_false")
    parser.add_argument("--exclude", nargs="*", default=[], help="状態の比較から除く属性名の断片")
    parser.add_argument("--flag", choices=("on", "off"))
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--dumps", type=Path, nargs=2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    cv2.setNumThreads(1)
    for name in ("config", "video", "dump", "output"):
        value = getattr(options, name)
        if value is not None:
            setattr(options, name, value.resolve())
    options.png = [p.resolve() for p in options.png]
    if options.dumps:
        options.dumps = [p.resolve() for p in options.dumps]
    if options.assets is not None:
        os.chdir(options.assets)
    runner = dict(lockstep=run_lockstep, dump=run_dump, compare=run_compare, crops=run_crops)[options.mode]
    report = runner(options)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if options.output:
        options.output.write_text(text, encoding="utf-8")
    print(text if len(text) < 6000 else text[:6000] + "\n...")
    return 0 if report["mismatch_count"] == 0 and all(report["selftest"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
