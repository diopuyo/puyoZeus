"""read_board の 1st pass の Rust 経路 (live_native_read_board) が、一括経路 (Python) と同じ判定を返すことの検証。

方針:
- 既定 OFF: 旗が全経路で False。OFF では Rust 経路の関数を 1 度も呼ばない (呼べば例外で落ちる)。
- ON: 合成フレーム (背景の模様 + 色ぷよ + ハイライト + ノイズ) で、盤面・セル一覧・bg 距離 (ビット単位)・
  read_both_boards の結果が OFF と生バイトで一致。NCC は numpy との差の最大と判定の一致を確かめる。
- 使えない入力 (旧形式の背景・静的背景マスク・非連続の画像・例外) は従来経路へ戻り、結果は同じ。
- 比較器そのものが違いを検出できること (陽性対照)。
実コマでの検証は scripts/verify_live_native_read_board.py (母数つきの報告を出す)。
"""
from __future__ import annotations

import inspect
from typing import Any

import cv2
import numpy as np
import pytest

from src import live_board_vectorized as lv
from src import live_native_read_board as native
from src.background_fingerprint import capture_patch_pair_robust
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, HIDDEN_ROWS, Board
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION, ImageReader
from tests.test_live_board_vectorized import (FRAME_SHAPE, _assert_reads_identical, _classifier,
                                              _draw_puyos, _native_available)

# 合成データで NCC (numpy の内積と最終桁がずれうる) を許す差。実コマの最大差は 2e-14 (検証台の報告)。
NCC_TOLERANCE = 1e-12
SEEDS = (1, 2, 3)
SKIP_PATTERNS = ((False, False), (True, False), (False, True), (True, True))
HEIGHT, WIDTH = FRAME_SHAPE[:2]
# 背景の質感: 1/4 解像度の乱数を拡大した、セル内に濃淡のある暗い茶系 (HSV 単独分類は空になる)。
TEXTURE_SCALE = 4
TEXTURE_RANGE = (50, 150)
BGR_GAIN = (0.75, 0.85, 1.0)
EMPTY_NOISE = 1                      # 空セルの画素ノイズ (±)
NOISY_CELL_AMPLITUDE = 25            # NCC を下げるセルのノイズ (±)
NOISY_CELL_RATE, TINT_CELL_RATE, HIGHLIGHT_CELL_RATE = 0.15, 0.12, 0.08
# 色かぶりの乗算ゲイン (BGR)。質感を保ったまま HSV 単独分類がぷよ色になる。
TINT_GAINS = ((0.3, 0.5, 1.8), (1.8, 0.6, 0.3), (0.4, 1.7, 0.5), (0.2, 1.2, 1.8))
HIGHLIGHT_BOX = (5, 11, 8, 16)       # セル内の (y 開始, y 終了, x 開始, x 終了)。42 画素 = セルの 4.4%
HIGHLIGHT_VALUE = 252
PUYO_FILL = 0.4
LOW_NCC_THRESHOLD = 0.30             # 色かぶりセルまで tier1 の候補にして、ガード・ハイライトの分岐を踏ませる


def _rust_available() -> bool:
    try:
        import puyo_core
        return callable(getattr(puyo_core, native.NATIVE_FUNCTION_NAME, None))
    except (ImportError, OSError):
        return False


needs_rust = pytest.mark.skipif(not (_native_available() and _rust_available()),
                                reason="puyo_core の read_board_first_pass が使えない環境")


def _texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low = rng.integers(*TEXTURE_RANGE, (HEIGHT // TEXTURE_SCALE, WIDTH // TEXTURE_SCALE), dtype=np.uint8)
    field = cv2.resize(low, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    return np.stack([field * gain for gain in BGR_GAIN], axis=2)


BACKGROUND = _texture(100)


def _decorate_cell(frame: np.ndarray, rng: np.random.Generator, rect: tuple) -> None:
    """空セルを、NCC が下がるノイズ・色かぶり・白ハイライトのいずれかにする (確率で)。"""
    x1, y1, x2, y2 = rect
    roll = rng.random()
    if roll < NOISY_CELL_RATE:
        frame[y1:y2, x1:x2] += rng.integers(-NOISY_CELL_AMPLITUDE, NOISY_CELL_AMPLITUDE + 1,
                                            (y2 - y1, x2 - x1, 3))
    elif roll < NOISY_CELL_RATE + TINT_CELL_RATE:
        frame[y1:y2, x1:x2] *= np.array(TINT_GAINS[int(rng.integers(0, len(TINT_GAINS)))], dtype=np.float32)
    elif roll < NOISY_CELL_RATE + TINT_CELL_RATE + HIGHLIGHT_CELL_RATE:
        top, bottom, left, right = HIGHLIGHT_BOX
        frame[y1 + top:y1 + bottom, x1 + left:x1 + right] = HIGHLIGHT_VALUE


def _frames(count: int, fill: float = PUYO_FILL, seed: int = 0) -> list[np.ndarray]:
    """質感のある背景 + セルごとの加工 + 色ぷよ。セルの外のノイズは小さい (背景と同じ実体)。"""
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(count):
        frame = BACKGROUND + rng.integers(-EMPTY_NOISE, EMPTY_NOISE + 1, FRAME_SHAPE)
        for region in (DEFAULT_P1_REGION, DEFAULT_P2_REGION):
            table = lv.cell_rect_table(region, WIDTH, HEIGHT)
            for rect in table.rects:
                _decorate_cell(frame, rng, rect)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        _draw_puyos(frame, rng, fill)
        frames.append(frame)
    return frames


def _bg_fingerprints() -> tuple:
    rng = np.random.default_rng(7)
    empties = [np.clip(BACKGROUND + rng.integers(-EMPTY_NOISE, EMPTY_NOISE + 1, FRAME_SHAPE), 0, 255).astype(np.uint8)
               for _ in range(5)]
    p1, p2 = DEFAULT_P1_REGION, DEFAULT_P2_REGION
    return capture_patch_pair_robust(empties, (p1.x, p1.y, p1.width, p1.height), (p2.x, p2.y, p2.width, p2.height))


# 比較するシナリオ (読取器の引数)。ガード・ハイライト・NCC 閾値・bg 閾値を振る。
SCENARIOS = {
    "default": dict(),
    "guard": dict(enable_patch_fp_hsv_guard=True),
    "low_threshold": dict(patch_ncc_threshold=LOW_NCC_THRESHOLD),
    "low_threshold_guard": dict(patch_ncc_threshold=LOW_NCC_THRESHOLD, enable_patch_fp_hsv_guard=True),
    "low_threshold_guard_highlight": dict(patch_ncc_threshold=LOW_NCC_THRESHOLD, enable_patch_fp_hsv_guard=True,
                                          use_highlight_override=True),
    "wide_background_threshold": dict(enable_patch_fp_hsv_guard=True, use_highlight_override=True,
                                      bg_empty_threshold=70.0),
    "strict_threshold": dict(patch_ncc_threshold=0.999, enable_patch_fp_hsv_guard=True),
}


def _native_reader(enable: bool, native_hsv: bool = True, **kwargs: Any) -> ImageReader:
    """一括経路 (+ ネイティブ HSV) の読取器。Rust 経路だけ enable で切り替える。"""
    bg1, bg2 = _bg_fingerprints()
    reader = ImageReader(classifier=_classifier(native_hsv), bg_fingerprint_p1=bg1, bg_fingerprint_p2=bg2,
                         use_ui_mask=False, **kwargs)
    assert reader.enable_live_board_vectorized(True)
    if enable:
        assert reader.enable_live_native_read_board(True)
    return reader


def _first_pass_inputs(reader: ImageReader, frame: np.ndarray, region: Any, skip: bool) -> tuple:
    """read_board が first_pass へ渡す引数 (背景指紋・HSV 画像) を同じ作りで用意する。"""
    hsv_full = reader._board_hsv_frame(frame, (DEFAULT_P1_REGION, DEFAULT_P2_REGION))
    return reader, frame, region, reader._bg_fp_for_region(region), hsv_full, skip


def _same_view(a: np.ndarray, b: np.ndarray) -> bool:
    return (a.__array_interface__["data"][0] == b.__array_interface__["data"][0]
            and a.shape == b.shape and a.strides == b.strides)


def _assert_cells_identical(expected: list, actual: list) -> None:
    assert len(expected) == len(actual)
    for left, right in zip(expected, actual):
        assert (left[0], left[1]) == (right[0], right[1])
        assert _same_view(left[2], right[2]) and _same_view(left[4], right[4])
        assert np.float64(left[3]).tobytes() == np.float64(right[3]).tobytes()


# ----------------------------------------------------------------------------
# 既定 OFF と配線
# ----------------------------------------------------------------------------

def test_default_is_off_everywhere() -> None:
    from src.phase_j.live_speed import ARM_TRIAL_KEYS, NATIVE_READ_BOARD_KEY, SPEED_DEFAULTS
    from src.production_config import LIVE_NATIVE_READ_BOARD_DEFAULT
    assert LIVE_NATIVE_READ_BOARD_DEFAULT is SPEED_DEFAULTS[NATIVE_READ_BOARD_KEY] is False
    assert NATIVE_READ_BOARD_KEY not in ARM_TRIAL_KEYS     # OFF/X/ALL の分類は既存の項目のまま
    assert ImageReader._live_native_read_board is False
    assert _native_reader(False)._live_native_read_board is False


def test_pipeline_and_launcher_defaults() -> None:
    from src.phase_j.launcher import UserConfig
    from src.recognition_pipeline import RecognitionPipeline
    for function in (RecognitionPipeline.__init__, RecognitionPipeline.load_default):
        assert inspect.signature(function).parameters["enable_live_native_read_board"].default is False
    assert UserConfig.__dataclass_fields__["live_native_read_board"].default is False


def test_environment_delivery_to_recognition_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_speed
    env = "PUYO_" + live_speed.NATIVE_READ_BOARD_KEY.upper()
    monkeypatch.delenv(env, raising=False)
    assert "enable_live_native_read_board" not in live_speed.recognition_config({}, True)
    monkeypatch.setenv(env, "0")
    assert "enable_live_native_read_board" not in live_speed.recognition_config({}, True)
    monkeypatch.setenv(env, "1")
    assert live_speed.recognition_config({}, True)["enable_live_native_read_board"] is True


def test_cli_to_environment_to_recognition_config(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI の旗 → 環境変数 (子 process へ) → 認識設定。OFF は何も足さない。"""
    import argparse
    import os

    from src.phase_j import live_speed
    env = "PUYO_" + live_speed.NATIVE_READ_BOARD_KEY.upper()
    for enabled in (True, False):
        monkeypatch.setattr("os.environ", {})
        parser = argparse.ArgumentParser()
        live_speed.add_arguments(parser)
        options = parser.parse_args(["--live-native-read-board"] if enabled else [])
        options.output, options.source, options.cpu_threads = tmp_path, "dshow", 1
        live_speed.configure(options, parser)
        assert os.environ[env] == str(int(enabled))
        assert ("enable_live_native_read_board" in live_speed.recognition_config({}, True)) is enabled


@pytest.mark.parametrize("enabled", [True, False])
def test_json_launcher_forwards_flag(enabled: bool, tmp_path: Any) -> None:
    from src.phase_j import launcher
    data = dict(source="dshow", device_name="OBS", live_native_read_board=enabled)
    config = launcher.build_pipeline_config(launcher.parse_user_config(data, tmp_path))
    assert config.get("live_native_read_board", False) is enabled


def test_default_config_does_not_gain_the_key(tmp_path: Any) -> None:
    """既定 OFF の設定は、従来の pipeline_config.json と同じキー集合のまま (新しい欄を書かない)。"""
    from src.phase_j import launcher
    config = launcher.build_pipeline_config(
        launcher.parse_user_config(dict(source="dshow", device_name="OBS"), tmp_path))
    assert "live_native_read_board" not in config
    with pytest.raises(launcher.LauncherConfigError):
        launcher.parse_user_config(dict(source="dshow", device_name="OBS", live_native_read_board="yes"), tmp_path)


def test_off_never_calls_native_first_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """旗が OFF なら Rust 経路の入口を 1 度も呼ばない (一括経路は通常どおり働く)。"""
    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("OFF で Rust 経路が呼ばれた")
    monkeypatch.setattr(native, "native_first_pass", boom)
    reader = _native_reader(False)
    for frame in _frames(2):
        reader.read_both_boards(frame)


def test_enable_reports_false_without_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native, "native_available", lambda: False)
    reader = _native_reader(False)
    assert reader.enable_live_native_read_board(True) is False
    assert reader._live_native_read_board is False
    monkeypatch.setattr(native, "native_available", lambda: True)
    assert reader.enable_live_native_read_board(True) is True
    assert reader.enable_live_native_read_board(False) is False


def test_runtime_status_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from src.phase_j import live_native_read_board_runtime as runtime
    from src.phase_j.live_speed import NATIVE_READ_BOARD_KEY
    env = "PUYO_" + NATIVE_READ_BOARD_KEY.upper()

    def pipe(native_rb: bool, vectorized: bool, hsv: bool) -> Any:
        reader = SimpleNamespace(_live_native_read_board=native_rb, _live_board_vectorized=vectorized,
                                 _get_hsv_classifier=lambda: SimpleNamespace(_enable_native_hsv=hsv))
        return SimpleNamespace(_reader=reader)
    monkeypatch.delenv(env, raising=False)
    assert runtime.runtime_status(pipe(True, True, True))["reason"] == runtime.REASON_DISABLED
    monkeypatch.setenv(env, "1")
    assert runtime.runtime_status(pipe(False, True, True))["reason"] == runtime.REASON_ACTIVATION_FAILED
    assert runtime.runtime_status(pipe(True, False, True))["reason"] == runtime.REASON_NEEDS_VECTORIZED
    assert runtime.runtime_status(pipe(True, True, False))["reason"] == runtime.REASON_NEEDS_NATIVE_HSV
    assert runtime.runtime_status(pipe(True, True, True)) == dict(active=True, reason=runtime.REASON_ENABLED)


# ----------------------------------------------------------------------------
# numpy との総和の一致 (NCC の平均・分散の順序)
# ----------------------------------------------------------------------------

@needs_rust
def test_pairwise_sum_is_bit_identical_to_numpy() -> None:
    import puyo_core
    rng = np.random.default_rng(0)
    sizes = list(range(1, 400)) + [1728, 1727, 1729, 5184, 100000]
    for n in sizes:
        for values in (rng.normal(size=n) * 100, rng.integers(0, 256, size=n).astype(np.float64) * 0.37):
            assert puyo_core.pairwise_sum_f64(values) == float(np.add.reduce(values)), n


# ----------------------------------------------------------------------------
# first_pass の一致 (合成フレーム)
# ----------------------------------------------------------------------------

def _compare_first_pass(reader_kwargs: dict, seed: int, frames: int = 3) -> int:
    """基準 (Rust 経路 OFF) と ON の first_pass を同じ入力で比べ、比べたセルの数を返す。"""
    reference = _native_reader(False, **reader_kwargs)
    candidate = _native_reader(True, **reader_kwargs)
    compared = 0
    for index, frame in enumerate(_frames(frames, fill=0.5, seed=seed)):
        for region in (DEFAULT_P1_REGION, DEFAULT_P2_REGION):
            skip = SKIP_PATTERNS[index % 4][0]
            board_a, board_b = Board(), Board()
            _reader_unused, frame_, region_, bg_fp, hsv_full, skip_ = _first_pass_inputs(candidate, frame, region, skip)
            # 同じ背景指紋・同じ HSV 画像を両方へ渡す (セル一覧の view のアドレスまで比べるため)
            expected = lv.first_pass(reference, frame_, region_, bg_fp, hsv_full, skip_, board_a)
            actual = lv.first_pass(candidate, frame_, region_, bg_fp, hsv_full, skip_, board_b)
            assert expected is not None and actual is not None
            assert board_a._grid.tobytes() == board_b._grid.tobytes()
            _assert_cells_identical(expected, actual)
            compared += 72
    return compared


@needs_rust
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_first_pass_matches_reference(scenario: str, seed: int) -> None:
    before = lv.STATS["first_pass_native"]
    assert _compare_first_pass(SCENARIOS[scenario], seed) == 3 * 2 * 72
    assert lv.STATS["first_pass_native"] > before, "Rust 経路が一度も使われていない (検査として弱い)"


def _status_histogram(reader_kwargs: dict, seed: int = 6) -> np.ndarray:
    """Rust の出力の状態 (0 分類・1 tier1 空・2 tier2 空) の度数。"""
    reader = _native_reader(True, **reader_kwargs)
    seen = np.zeros(3, dtype=np.int64)
    for frame in _frames(4, seed=seed):
        for region in (DEFAULT_P1_REGION, DEFAULT_P2_REGION):
            reader_, frame_, region_, bg_fp, hsv_full, skip = _first_pass_inputs(reader, frame, region, False)
            table = lv.cell_rect_table(region_, frame_.shape[1], frame_.shape[0])
            arrays = native.table_arrays(reader_, table)
            status, _distance, _ncc = native.call_native(reader_, frame_, hsv_full, arrays,
                                                         native.bg_pack(reader_, bg_fp, arrays), skip)
            seen += np.bincount(status, minlength=3)
    return seen


@needs_rust
def test_scenarios_exercise_every_branch() -> None:
    """合成フレームが、tier1 空・tier2 空・分類のすべてと、ガード・ハイライトの取り消しを踏む (検査の被覆)。"""
    default = _status_histogram(SCENARIOS["default"])
    assert (default > 0).all(), default.tolist()
    no_guard = _status_histogram(SCENARIOS["low_threshold"])
    guarded = _status_histogram(SCENARIOS["low_threshold_guard"])
    highlighted = _status_histogram(SCENARIOS["low_threshold_guard_highlight"])
    assert guarded[1] < no_guard[1], "ガードが tier1 空を取り消したセルが無い"
    assert highlighted[1] < guarded[1], "ハイライトが tier1 空を取り消したセルが無い"


@needs_rust
def test_ncc_matches_numpy_within_tolerance_and_decisions_agree() -> None:
    """Rust の NCC が従来の CellPatchFingerprint.ncc_to と NCC_TOLERANCE 以内、閾値の判定が全セルで同じ。"""
    from src.background_fingerprint import CellPatchFingerprint
    from src.board import HIDDEN_ROWS
    reader = _native_reader(True)
    worst, checked = 0.0, 0
    for frame in _frames(3, fill=0.5, seed=8):
        reader_, frame_, region, bg_fp, hsv_full, skip = _first_pass_inputs(reader, frame, DEFAULT_P2_REGION, False)
        table = lv.cell_rect_table(region, frame_.shape[1], frame_.shape[0])
        arrays = native.table_arrays(reader_, table)
        _status, _distance, ncc = native.call_native(reader_, frame_, hsv_full, arrays,
                                                     native.bg_pack(reader_, bg_fp, arrays), skip)
        for index, value in enumerate(ncc.tolist()):
            x1, y1, x2, y2 = arrays.rects[index].tolist()
            row, col = table.positions[index]
            cell = bg_fp.cell_at_patch(row - HIDDEN_ROWS, col)
            expected = float(CellPatchFingerprint(patch_hsv=hsv_full[y1:y2, x1:x2].astype(np.float32)).ncc_to(cell))
            worst = max(worst, abs(value - expected))
            assert (value >= 0.92) == (expected >= 0.92)
            checked += 1
    assert checked == 3 * 72 and worst < NCC_TOLERANCE


# ----------------------------------------------------------------------------
# read_both_boards 全体 (一括経路 + ネイティブ HSV が基準)
# ----------------------------------------------------------------------------

@needs_rust
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_read_boards_identical(scenario: str) -> None:
    off = _native_reader(False, **SCENARIOS[scenario])
    on = _native_reader(True, **SCENARIOS[scenario])
    for index, frame in enumerate(_frames(8)):
        _assert_reads_identical(off, on, frame, SKIP_PATTERNS[index % 4])


@needs_rust
def test_read_boards_identical_with_roi_offsets_and_side_sat() -> None:
    """ROI 補正 (矩形・背景の形が変わる)・側別彩度較正でも一致する。"""
    off, on = _native_reader(False), _native_reader(True)
    for reader in (off, on):
        reader.set_side_sat_calibration(True)
    for index, frame in enumerate(_frames(14, fill=0.5, seed=4)):
        moving = index >= 10
        offsets = ((float(index % 3), -float(index % 2)), (-2.0, float(index % 4))) if moving else ((0.0, 0.0),) * 2
        left = off.read_both_boards(frame, p1_roi_offset=offsets[0], p2_roi_offset=offsets[1])
        right = on.read_both_boards(frame, p1_roi_offset=offsets[0], p2_roi_offset=offsets[1])
        for a, b in zip(left, right):
            assert a._grid.tobytes() == b._grid.tobytes()
    assert off._side_sat_scale == on._side_sat_scale


@needs_rust
def test_live_window_records_identical() -> None:
    off, on = _native_reader(False), _native_reader(True)
    for frame in _frames(3, fill=0.5, seed=9):
        for reader in (off, on):
            reader._live_raw_boards, reader._live_hsv_boards = {}, {}
            reader._live_hsv_pixels, reader._live_raw_board_sec = {}, 0.0
            reader.read_both_boards(frame)
        for key in off._live_raw_boards:
            assert off._live_raw_boards[key]._grid.tobytes() == on._live_raw_boards[key]._grid.tobytes()
        for key in off._live_hsv_pixels:
            assert off._live_hsv_pixels[key].tobytes() == on._live_hsv_pixels[key].tobytes()


# ----------------------------------------------------------------------------
# 使えない入力は従来経路へ戻る (結果は同じ・理由が記録される)
# ----------------------------------------------------------------------------

def _reads_equal_to_reference(on: ImageReader, frame: np.ndarray) -> None:
    off = _native_reader(False)
    for left, right in zip(off.read_both_boards(frame), on.read_both_boards(frame)):
        assert left._grid.tobytes() == right._grid.tobytes()


@needs_rust
def test_falls_back_when_native_call_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(*args: Any, **kwargs: Any) -> None:
        raise ValueError("拡張側の検査に失敗")
    monkeypatch.setattr(native, "call_native", failing)
    before = lv.STATS["native_first_pass_fallback_error"]
    on = _native_reader(True)
    _reads_equal_to_reference(on, _frames(1, fill=0.5)[0])
    assert lv.STATS["native_first_pass_fallback_error"] > before
    assert native.LAST_FALLBACK_REASON["reason"].startswith("error: ValueError")


@needs_rust
def test_falls_back_for_static_mask_and_non_contiguous_hsv(monkeypatch: pytest.MonkeyPatch) -> None:
    on = _native_reader(True)
    frame = _frames(1, fill=0.5)[0]
    reader, frame_, region, bg_fp, hsv_full, _skip = _first_pass_inputs(on, frame, DEFAULT_P1_REGION, False)
    assert native.unsupported_reason(reader, frame_, bg_fp, hsv_full, region) is None
    assert native.unsupported_reason(reader, frame_, bg_fp, hsv_full[:, ::-1], region) == "image_layout"
    assert native.unsupported_reason(reader, frame_, bg_fp, hsv_full.astype(np.float32), region) == "image_layout"
    assert native.unsupported_reason(reader, frame_, bg_fp, None, region) == "no_patch_background"
    assert native.unsupported_reason(reader, frame_, None, hsv_full, region) == "no_patch_background"
    monkeypatch.setattr(reader, "_get_static_mask_for_region", lambda _region: object())
    assert native.unsupported_reason(reader, frame_, bg_fp, hsv_full, region) == "static_mask"


@needs_rust
def test_falls_back_when_hsv_classifier_is_not_native() -> None:
    reader = _native_reader(False, native_hsv=False)     # ネイティブ HSV が無効
    assert reader.enable_live_native_read_board(True)
    frame = _frames(1, fill=0.5)[0]
    _r, frame_, region, bg_fp, hsv_full, _s = _first_pass_inputs(reader, frame, DEFAULT_P1_REGION, False)
    assert native.unsupported_reason(reader, frame_, bg_fp, hsv_full, region) == "hsv_classifier"
    _reads_equal_to_reference(reader, frame)


@needs_rust
def test_bg_pack_is_rebuilt_when_background_changes() -> None:
    on = _native_reader(True)
    frame = _frames(1, fill=0.5)[0]
    reader, frame_, region, bg_fp, hsv_full, _s = _first_pass_inputs(on, frame, DEFAULT_P1_REGION, False)
    table = lv.cell_rect_table(region, frame_.shape[1], frame_.shape[0])
    arrays = native.table_arrays(reader, table)
    first = native.bg_pack(reader, bg_fp, arrays)
    assert native.bg_pack(reader, bg_fp, arrays) is first
    other = _native_reader(True)._bg_fp_for_region(region)   # 別に採取した指紋 (別の object)
    assert native.bg_pack(reader, other, arrays) is not first


# ----------------------------------------------------------------------------
# 比較器の陽性対照: 判定の入力をずらせば、同じ比較が必ず違いを検出する
# ----------------------------------------------------------------------------

@needs_rust
def test_comparison_detects_perturbed_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    original = native.first_pass_params

    def shifted(reader: Any, skip_tier1: bool) -> dict[str, Any]:
        values = original(reader, skip_tier1)
        values.update(ncc_threshold=-1.0, bg_threshold=1e9)    # 全セルが空になる
        return values
    monkeypatch.setattr(native, "first_pass_params", shifted)
    with pytest.raises(AssertionError):
        _compare_first_pass(dict(enable_patch_fp_hsv_guard=False), seed=1)


@needs_rust
def test_board_cells_set_to_empty_in_dirty_board() -> None:
    """渡された盤に既に色が入っていても、空と確定したセルは COLOR_EMPTY で上書きされる (従来経路と同じ)。"""
    reader = _native_reader(True)
    frame = _frames(1, fill=0.2, seed=2)[0]
    dirty_a, dirty_b = Board(), Board()
    dirty_a._grid[:] = 3
    dirty_b._grid[:] = 3
    inputs = _first_pass_inputs(reader, frame, DEFAULT_P1_REGION, False)
    reader._live_native_read_board = False
    expected = lv.first_pass(*inputs, dirty_a)
    reader._live_native_read_board = True
    actual = lv.first_pass(*inputs, dirty_b)
    assert dirty_a._grid.tobytes() == dirty_b._grid.tobytes()
    assert (dirty_b._grid == COLOR_EMPTY).any()
    _assert_cells_identical(expected, actual)


# ----------------------------------------------------------------------------
# 投票方式 (vote_mode) のネイティブ分類 (LiveImages の盤面読み)
# ----------------------------------------------------------------------------

def _vote_available() -> bool:
    try:
        import puyo_core
        return callable(getattr(puyo_core, "classify_cells_vote", None))
    except (ImportError, OSError):
        return False


needs_vote = pytest.mark.skipif(not _vote_available(), reason="puyo_core の classify_cells_vote が使えない環境")
VOTE_SCALES = (1.0, 0.7)


def _vote_patches(seed: int) -> list[np.ndarray]:
    """色・ノイズ・ハイライト・赤の拡張範囲・半分だけの色・暗所・空・実セルの寸法のパッチ。"""
    from tests.test_live_board_vectorized import _color_patches
    rng = np.random.default_rng(seed)
    patches = _color_patches(seed)
    # 赤の拡張範囲 (H 11-18) の境界: R-G 差が閾値の上下になる色を並べる
    for red_minus_green in (60, 79, 80, 81, 120):
        patch = np.zeros((30, 32, 3), dtype=np.uint8)
        patch[...] = (30, 120 - red_minus_green // 2 + 40, 160 + min(red_minus_green // 2, 90))
        patches.append(patch)
    patches += [frame[y:y + 30, x:x + 32] for frame in _frames(1, seed=seed)
                for y, x in ((200, 300), (400, 330), (600, 300), (170, 1300), (500, 1330))]
    patches.append(rng.integers(0, 256, (30, 32, 3), dtype=np.uint8))
    return patches


@needs_vote
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scale", VOTE_SCALES)
def test_vote_native_matches_python(seed: int, scale: float) -> None:
    from src.image_reader import ColorClassifier
    python_side = ColorClassifier(vote_mode=True)
    native_side = ColorClassifier(vote_mode=True)
    assert native_side.enable_native_vote(True)
    for classifier in (python_side, native_side):
        classifier.set_s_min_scale(scale)
    patches = _vote_patches(seed)
    expected = [python_side.classify(p) for p in patches]
    assert [native_side.classify(p) for p in patches] == expected
    assert len(set(expected)) >= 3, "投票の結果が偏っている (検査として弱い)"


@needs_vote
def test_vote_comparison_detects_a_perturbed_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    """陽性対照: 投票比率をずらした Rust 経路は、同じ比較で必ず違いを出す。"""
    from src.image_reader import ColorClassifier
    python_side = ColorClassifier(vote_mode=True)
    native_side = ColorClassifier(vote_mode=True)
    assert native_side.enable_native_vote(True)
    original = native_side._native_vote_params
    monkeypatch.setattr(native_side, "_native_vote_params",
                        lambda: dict(original(), vote_puyo_min_ratio=0.9, vote_ojama_min_ratio=0.99))
    patches = [p for seed in SEEDS for p in _vote_patches(seed)]
    assert [native_side.classify(p) for p in patches] != [python_side.classify(p) for p in patches]


def test_vote_native_is_off_by_default_and_only_for_vote_mode() -> None:
    from src.image_reader import ColorClassifier
    assert ColorClassifier._enable_native_vote is False
    assert ColorClassifier(vote_mode=True)._enable_native_vote is False
    assert ColorClassifier().enable_native_vote(True) is False       # 中央値方式には効かない
    vote = ColorClassifier(vote_mode=True)
    assert vote.enable_native_vote(False) is False and vote._enable_native_vote is False
    assert vote.enable_native_hsv(True) is False                      # 従来どおり vote_mode は対象外


def test_vote_native_unavailable_keeps_python_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    from src.image_reader import ColorClassifier
    real_import = builtins.__import__

    def deny(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "puyo_core":
            raise ImportError("テスト用に拡張を隠す")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", deny)
    assert ColorClassifier(vote_mode=True).enable_native_vote(True) is False


def test_live_images_flag_env_matches_live_speed_and_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_speed
    from src.image_reader import ColorClassifier
    assert native.FLAG_ENV == "PUYO_" + live_speed.NATIVE_READ_BOARD_KEY.upper()
    monkeypatch.delenv(native.FLAG_ENV, raising=False)
    classifiers = [ColorClassifier(), ColorClassifier(vote_mode=True)]
    assert native.enable_for_live_images(classifiers) == [False, False]
    assert not classifiers[0]._enable_native_hsv and not classifiers[1]._enable_native_vote
    monkeypatch.setenv(native.FLAG_ENV, "0")
    assert native.enable_for_live_images(classifiers) == [False, False]


@needs_rust
@needs_vote
def test_live_images_boards_identical_with_and_without_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """LiveImages.classify の盤 (中央値・投票の 2 枚) が、旗 OFF/ON で全セル同じ。時間の欄は比べない。"""
    from src.phase_j.live_v49f_images import LiveImages
    frames = _frames(3, seed=12)
    boards: dict[str, list] = {}
    for label, value in (("off", "0"), ("on", "1")):
        monkeypatch.setenv(native.FLAG_ENV, value)
        images = LiveImages()
        assert [c._enable_native_hsv for c in images.classifiers[:1]] == [label == "on"]
        assert images.classifiers[1]._enable_native_vote is (label == "on")
        boards[label] = [[b["board"] for b in images.classify(frame, 0.0, side)]
                         for frame in frames for side in (0, 1)]
    assert boards["off"] == boards["on"]
    assert any(any(cell for cell in board) for group in boards["on"] for board in group), "全セル空 (検査として弱い)"
