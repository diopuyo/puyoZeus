"""盤面読取の一括経路 (live_board_vectorized) が従来経路と 1 ビットも違わないことの検証。

方針:
- 既定 OFF: フラグが全経路で False、OFF では一括経路の関数が一度も呼ばれない。
- ON: 合成フレーム (背景の模様 + 色ぷよ + ハイライト + ノイズ) で、盤面・分類色・
  確率・HSV 画素・ライブ窓の記録が OFF と生バイトで一致。
- 一致を確認する比較器そのものが違いを検出できること (scripts の selftest)。
実コマでの検証は scripts/verify_live_board_vectorized.py (母数つきの報告を出す)。
"""
from __future__ import annotations

import zlib
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from src import hybrid_classifier as hybrid_module
from src import live_board_vectorized as lv
from src.background_fingerprint import capture_patch_pair_robust
from src.board import (BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN, COLOR_OJAMA,
                       COLOR_PURPLE, COLOR_RED, COLOR_UNKNOWN, COLOR_YELLOW, HIDDEN_ROWS)
from src.hybrid_classifier import HybridClassifier
from src.image_reader import (DEFAULT_P1_REGION, DEFAULT_P2_REGION, BoardRegion, ColorClassifier,
                              ImageReader)
from src.patch_classifier import NUM_CLASSES

FRAME_SHAPE = (1080, 1920, 3)
PUYO_COLORS = (COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA)
HSV_OF = {COLOR_RED: (4, 225, 205), COLOR_BLUE: (115, 210, 200), COLOR_GREEN: (65, 205, 200),
          COLOR_YELLOW: (28, 220, 225), COLOR_PURPLE: (148, 200, 195), COLOR_OJAMA: (0, 12, 200)}
SEEDS = tuple(range(6))


def _native_available() -> bool:
    try:
        import puyo_core
        return hasattr(puyo_core, "classify_cells_hsv")
    except (ImportError, OSError):
        return False


needs_native = pytest.mark.skipif(not _native_available(), reason="puyo_core (Rust) が使えない環境")


# ----------------------------------------------------------------------------
# 合成フレーム
# ----------------------------------------------------------------------------

def _background(seed: int = 100) -> np.ndarray:
    """盤面領域に空間的な模様を持つ静的な背景 (NCC が働く程度に非一様)。"""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(30, 200, FRAME_SHAPE, dtype=np.uint8), (0, 0), 4)
    return base


def _draw_puyos(frame: np.ndarray, rng: np.random.Generator, fill: float) -> None:
    """可視セルへ、色つきの円 + 白ハイライト + 目を描く (一部のセルは半分だけ・暗め)。"""
    for region in (DEFAULT_P1_REGION, DEFAULT_P2_REGION):
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            for col in range(BOARD_COLS):
                if rng.random() > fill:
                    continue
                color = PUYO_COLORS[int(rng.integers(0, len(PUYO_COLORS)))]
                h, s, v = HSV_OF[color]
                v = int(np.clip(v + rng.integers(-60, 20), 40, 255))
                s = int(np.clip(s + rng.integers(-60, 30), 5, 255))
                bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
                cx, cy = region.cell_center(row, col)
                radius = int(region.cell_width * rng.uniform(0.25, 0.46))
                cv2.circle(frame, (cx, cy), radius, tuple(int(c) for c in bgr), -1)
                cv2.circle(frame, (cx - radius // 3, cy - radius // 3), max(2, radius // 5),
                           (250, 250, 250), -1)


def _frames(count: int = 6, fill: float = 0.45, seed: int = 0) -> list[np.ndarray]:
    background = _background()
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(count):
        noise = rng.integers(-2, 3, FRAME_SHAPE, dtype=np.int16)
        frame = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        _draw_puyos(frame, rng, fill)
        frames.append(frame)
    return frames


def _empty_backgrounds(count: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    background = _background()
    return [np.clip(background.astype(np.int16) + rng.integers(-2, 3, FRAME_SHAPE, dtype=np.int16),
                    0, 255).astype(np.uint8) for _ in range(count)]


def _bg_fingerprints() -> tuple:
    p1, p2 = DEFAULT_P1_REGION, DEFAULT_P2_REGION
    return capture_patch_pair_robust(
        _empty_backgrounds(),
        (p1.x, p1.y, p1.width, p1.height), (p2.x, p2.y, p2.width, p2.height))


# ----------------------------------------------------------------------------
# 偽 CNN (パッチ内容から決まる確率。ON/OFF で同じ入力に同じ出力を返す)
# ----------------------------------------------------------------------------

class FakeCnn:
    """確率の山を 0.3〜0.99 に散らし、override 閾値の上下・同率・UNKNOWN 境界を踏ませる。"""

    def __init__(self, sharpness: float = 2.5) -> None:
        self.sharpness = sharpness
        self.vectorized = False
        self.batch_calls = 0

    def set_live_board_vectorized(self, enable: bool) -> None:
        self.vectorized = bool(enable)

    def predict_proba_batch(self, patches: list[np.ndarray]) -> np.ndarray:
        self.batch_calls += 1
        rows = []
        for patch in patches:
            seed = zlib.crc32(np.ascontiguousarray(patch).tobytes()) if patch.size else 12345
            logits = np.random.default_rng(seed).normal(size=NUM_CLASSES) * self.sharpness
            exp = np.exp(logits - logits.max())
            rows.append((exp / exp.sum()).astype(np.float32))
        return np.stack(rows)

    def predict_proba(self, patch: np.ndarray) -> np.ndarray:
        return self.predict_proba_batch([patch])[0]

    def predict_proba_grid(self, frame: np.ndarray, region: BoardRegion) -> np.ndarray:
        """OnlineHsvCalibrator 用 (各セルの最大確率)。HybridClassifier.predict_proba_and_hsv_grid が使う。"""
        table = lv.cell_rect_table(region, frame.shape[1], frame.shape[0])
        probs = self.predict_proba_batch(lv.slice_patches(frame, table))
        return probs.max(axis=1).reshape(12, BOARD_COLS).astype(np.float32)


def _classifier(native: bool, cnn: Any = "fake", **kwargs: Any) -> HybridClassifier:
    hsv = ColorClassifier(enable_specular_robust_saturation=True)
    if native:
        assert hsv.enable_native_hsv(True)
    return HybridClassifier(
        hsv_classifier=hsv, cnn_classifier=FakeCnn() if cnn == "fake" else cnn,
        ui_mask_cells=frozenset({(HIDDEN_ROWS + 2, 3)}), **kwargs)


def _reader(vectorized: bool, native: bool, *, with_bg: bool, **reader_kwargs: Any) -> ImageReader:
    bg1, bg2 = _bg_fingerprints() if with_bg else (None, None)
    reader = ImageReader(
        classifier=_classifier(native), bg_fingerprint_p1=bg1, bg_fingerprint_p2=bg2,
        use_ui_mask=False, **reader_kwargs)
    if vectorized:
        assert reader.enable_live_board_vectorized(True)
    return reader


# ----------------------------------------------------------------------------
# 既定 OFF
# ----------------------------------------------------------------------------

def test_default_is_off_everywhere() -> None:
    from src.phase_j.live_speed import SPEED_DEFAULTS, TRIAL_FLAG_KEYS
    from src.production_config import LIVE_BOARD_VECTORIZED_DEFAULT
    assert LIVE_BOARD_VECTORIZED_DEFAULT is SPEED_DEFAULTS["live_board_vectorized"] is False
    # 既存の鍵 (24b・1fd4b1cf まで) の後ろへの末尾追加。その後ろに別の鍵が続いてよい
    assert TRIAL_FLAG_KEYS.index("live_board_vectorized") > TRIAL_FLAG_KEYS.index("match_end_yatta_text_template")
    assert list(SPEED_DEFAULTS).index("live_board_vectorized") > list(SPEED_DEFAULTS).index("match_end_yatta_text_template")
    reader = _reader(False, False, with_bg=False)
    assert reader._live_board_vectorized is False
    assert reader._classifier._live_board_vectorized is False
    assert ColorClassifier()._live_board_vectorized is False


def test_off_never_calls_vectorized_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    """OFF では一括経路の関数を 1 度も呼ばない (呼べば例外で落ちる)。"""
    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("OFF で一括経路が呼ばれた")
    for name in ("first_pass", "hsv_only_colors", "classify_patches", "preprocess_patches",
                 "board_hsv_frame", "cell_rect_table"):
        monkeypatch.setattr(lv, name, boom)
    reader = _reader(False, False, with_bg=True)
    frame = _frames(1)[0]
    reader.read_both_boards(frame)
    reader.read_board_hsv_only(frame, DEFAULT_P1_REGION)
    reader._classifier.classify_batch([frame[10:30, 10:30]])


def test_pipeline_flag_defaults_and_active_state() -> None:
    import inspect

    from src.recognition_pipeline import RecognitionPipeline
    for function in (RecognitionPipeline.__init__, RecognitionPipeline.load_default):
        assert inspect.signature(function).parameters["enable_live_board_vectorized"].default is False


# ----------------------------------------------------------------------------
# 矩形表・前処理・HSV 画像
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(1080, 1920), (720, 1280), (200, 300), (1, 1)])
@pytest.mark.parametrize("offset", [(0, 0), (5, -3), (-400, 700)])
def test_cell_rect_table_matches_legacy_clamp(shape: tuple, offset: tuple) -> None:
    height, width = shape
    region = BoardRegion(DEFAULT_P1_REGION.x + offset[0], DEFAULT_P1_REGION.y + offset[1], 384, 720)
    table = lv.cell_rect_table(region, width, height)
    expected = []
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        for col in range(BOARD_COLS):
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1 = max(0, min(x1, width - 1))
            x2 = max(x1 + 1, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(y1 + 1, min(y2, height))
            expected.append((x1, y1, x2, y2))
    assert list(table.rects) == expected
    assert len(table.positions) == lv.VISIBLE_CELL_COUNT


@pytest.mark.parametrize("circle", [False, True])
def test_cnn_preprocess_matches_patch_to_tensor(circle: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    from src import patch_classifier as pc
    monkeypatch.setattr(pc, "USE_CIRCLE_MASK", circle)
    rng = np.random.default_rng(3)
    shapes = [(30, 32), (36, 38), (20, 20), (5, 7), (1, 1), (8, 8), (3, 40), (64, 64)]
    patches = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in shapes * 8]
    patches.insert(4, np.zeros((0, 5, 3), dtype=np.uint8))
    classifier = pc.CnnPatchClassifier()
    legacy = torch.stack([classifier._patch_to_tensor(p)[0] for p in patches]).numpy()
    mask = pc._get_circle_mask(pc.PATCH_RESIZE_H, pc.PATCH_RESIZE_W) if circle else None
    fast = lv.preprocess_patches(patches, pc.PATCH_RESIZE_H, pc.PATCH_RESIZE_W, mask)
    assert fast.dtype == legacy.dtype == np.float32 and fast.shape == legacy.shape
    assert fast.tobytes() == legacy.tobytes()


def test_cnn_batch_probabilities_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """CnnPatchClassifier 全体 (前処理 → 推論 → softmax) の出力が生バイトで一致 (torch 経路)。"""
    pytest.importorskip("torch")
    monkeypatch.delenv("PUYO_CNN_BACKEND", raising=False)
    from src.patch_classifier import CnnPatchClassifierLarge
    classifier = CnnPatchClassifierLarge()
    rng = np.random.default_rng(5)
    patches = [rng.integers(0, 256, (30, 32, 3), dtype=np.uint8) for _ in range(40)]
    baseline = classifier.predict_proba_batch(patches)
    classifier.set_live_board_vectorized(True)
    assert classifier.predict_proba_batch(patches).tobytes() == baseline.tobytes()


@pytest.mark.parametrize("offsets", [((0, 0), (0, 0)), ((7, -4), (-9, 11)), ((300, 200), (0, 0))])
def test_hsv_buffer_pool_matches_legacy_all_pixels(offsets: tuple) -> None:
    """全画素 (領域の外のゼロも) が従来の np.zeros + 領域ごとの cvtColor と同じ。バッファは回転する。"""
    frame = _frames(1)[0]
    reader = _reader(False, False, with_bg=False)
    pooled = _reader(True, False, with_bg=False)
    for step in range(lv.HSV_BUFFER_COUNT * 3):
        regions = tuple(
            BoardRegion(base.x + dx * (step % 3), base.y + dy * (step % 2), base.width, base.height)
            for base, (dx, dy) in zip((DEFAULT_P1_REGION, DEFAULT_P2_REGION), offsets))
        moved = np.ascontiguousarray(np.roll(frame, step * 3, axis=1))
        expected = ImageReader._board_hsv_frame(reader, moved, regions)
        actual = ImageReader._board_hsv_frame(pooled, moved, regions)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()


def test_hsv_buffer_pool_keeps_recent_frames_intact() -> None:
    """直近の数コマ分の view (ライブ窓の観測) が、次のコマの書込みで壊れない。"""
    frames = _frames(lv.HSV_BUFFER_COUNT)
    pooled = _reader(True, False, with_bg=False)
    regions = (DEFAULT_P1_REGION, DEFAULT_P2_REGION)
    kept = [ImageReader._board_hsv_frame(pooled, f, regions) for f in frames]
    legacy = _reader(False, False, with_bg=False)
    for frame, view in zip(frames[1:], kept[1:]):
        assert view.tobytes() == ImageReader._board_hsv_frame(legacy, frame, regions).tobytes()


# ----------------------------------------------------------------------------
# HSV 分類の一括 (Rust)
# ----------------------------------------------------------------------------

def _color_patches(seed: int) -> list[np.ndarray]:
    """色・ノイズ・ハイライト・二峰の赤・半分だけの色・暗所・空パッチを含む。"""
    rng = np.random.default_rng(seed)
    patches: list[np.ndarray] = []
    for color, (h, s, v) in HSV_OF.items():
        for _ in range(6):
            size = (int(rng.integers(8, 40)), int(rng.integers(8, 40)))
            hsv = np.zeros((*size, 3), dtype=np.uint8)
            hsv[...] = (h + rng.integers(-3, 4), np.clip(s + rng.integers(-30, 30), 0, 255),
                        np.clip(v + rng.integers(-40, 30), 0, 255))
            hsv[: size[0] // 4, : size[1] // 3] = (0, 0, 250)
            patches.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    wrap = np.zeros((20, 20, 3), dtype=np.uint8)
    wrap[:, :10] = (2, 220, 200)
    wrap[:, 10:] = (172, 220, 200)
    patches.append(cv2.cvtColor(wrap, cv2.COLOR_HSV2BGR))
    quarter = np.zeros((24, 24, 3), dtype=np.uint8)
    quarter[:12, :12] = cv2.cvtColor(np.uint8([[[115, 200, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
    patches += [quarter, np.zeros((16, 16, 3), dtype=np.uint8), np.zeros((0, 9, 3), dtype=np.uint8),
                rng.integers(0, 256, (30, 32, 3), dtype=np.uint8), rng.integers(0, 256, (3, 3, 3), dtype=np.uint8)]
    return patches


@needs_native
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scale", [1.0, 0.7])
def test_classify_patches_matches_per_patch(seed: int, scale: float) -> None:
    clf = ColorClassifier(enable_specular_robust_saturation=True)
    assert clf.enable_native_hsv(True)
    clf.set_s_min_scale(scale)
    patches = _color_patches(seed)
    expected = [clf.classify(p) for p in patches]
    assert lv.classify_patches(clf, patches) == expected
    clf.set_live_board_vectorized(True)
    assert clf.classify_batch(patches) == expected


@needs_native
def test_classify_patches_falls_back_when_not_applicable() -> None:
    clf = ColorClassifier()  # ネイティブ無効
    patches = _color_patches(1)
    assert lv.classify_patches(clf, patches) == [clf.classify(p) for p in patches]
    vote = ColorClassifier(vote_mode=True)
    assert vote.enable_native_hsv(True) is False
    assert lv.classify_patches(vote, patches) == [vote.classify(p) for p in patches]
    assert lv.classify_patches(clf, []) == []


# ----------------------------------------------------------------------------
# HybridClassifier.classify_batch
# ----------------------------------------------------------------------------

class FakeUiMatcher:
    """平均輝度が高いパッチだけ UI とみなす (is_ui の呼出を数える)。"""

    def __init__(self) -> None:
        self.calls = 0

    def is_ui(self, patch: np.ndarray) -> bool:
        self.calls += 1
        return bool(patch.mean() > 120)


def _batch_inputs(seed: int) -> tuple[list[np.ndarray], list, list]:
    patches = _color_patches(seed)
    rng = np.random.default_rng(seed)
    distances = [None if i % 4 == 0 else float(rng.uniform(0, 80)) for i in range(len(patches))]
    positions = [(HIDDEN_ROWS + (i // BOARD_COLS) % 12, i % BOARD_COLS) for i in range(len(patches))]
    return patches, distances, positions


@pytest.mark.parametrize("native", [False, pytest.param(True, marks=needs_native)])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("variant", ["plain", "mask_ojama", "unknown_branch", "no_cnn", "all_ui"])
def test_hybrid_classify_batch_matches_legacy(native: bool, seed: int, variant: str,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    patches, distances, positions = _batch_inputs(seed)
    kwargs: dict[str, Any] = dict(mask_ojama_logit=variant == "mask_ojama")
    if variant == "unknown_branch":
        monkeypatch.setattr(HybridClassifier, "LOW_CONFIDENCE_UNKNOWN_THRESHOLD", 0.8)
    results = []
    for vectorized in (False, True):
        clf = _classifier(native, cnn=None if variant == "no_cnn" else "fake", **kwargs)
        clf._ui_matcher = FakeUiMatcher()
        if variant == "all_ui":
            clf._ui_mask_cells = None
        clf.set_live_board_vectorized(vectorized)
        results.append([
            clf.classify_batch(patches, bg_distances=distances, cell_positions=positions),
            clf.classify_batch(patches),
            clf.classify_batch(patches, bg_distances=distances),
        ])
    assert results[0] == results[1]
    assert set(results[0][0]) - {COLOR_UNKNOWN, *PUYO_COLORS, COLOR_EMPTY} == set()


@pytest.mark.parametrize("native", [False, pytest.param(True, marks=needs_native)])
def test_hsv_grid_for_online_calibration_identical(native: bool) -> None:
    """OnlineHsvCalibrator へ渡す (確率格子, HSV 格子) が従来と同じ (72 セルの一括分類)。"""
    frame = _frames(1, fill=0.6, seed=11)[0]
    grids = []
    for vectorized in (False, True):
        clf = _classifier(native)
        clf.set_live_board_vectorized(vectorized)
        grids.append([clf.predict_proba_and_hsv_grid(frame, region)
                      for region in (DEFAULT_P1_REGION, DEFAULT_P2_REGION)])
    for (proba_off, hsv_off), (proba_on, hsv_on) in zip(*grids):
        assert proba_off.tobytes() == proba_on.tobytes()
        assert hsv_off.dtype == hsv_on.dtype and hsv_off.tobytes() == hsv_on.tobytes()
    assert grids[0][0][1].any(), "HSV 格子が全て 0 (検査として弱い)"


@pytest.mark.parametrize("override", [0.0, 0.5, 0.70, 0.99, 1.0])
def test_hybrid_override_threshold_boundaries(override: float) -> None:
    """override 閾値 (確信度比較の float64 化) が境界でも従来と同じ。"""
    patches, distances, positions = _batch_inputs(2)
    outs = []
    for vectorized in (False, True):
        clf = _classifier(False, cnn_override_prob=override)
        clf.set_live_board_vectorized(vectorized)
        outs.append(clf.classify_batch(patches, bg_distances=distances, cell_positions=positions))
    assert outs[0] == outs[1]


def test_hybrid_exact_float32_threshold_tie() -> None:
    """確信度がちょうど float32(0.7) 付近でも float64 で比べる従来の結果と同じ。"""
    class Pinned(FakeCnn):
        def predict_proba_batch(self, patches: list[np.ndarray]) -> np.ndarray:
            below, above = np.float32(0.69999999), np.float32(0.70000005)
            rows = []
            for i, _ in enumerate(patches):
                p = np.full(NUM_CLASSES, (1 - below) / (NUM_CLASSES - 1), dtype=np.float32)
                p[i % NUM_CLASSES] = np.float32(0.7) if i % 3 == 0 else (below if i % 3 == 1 else above)
                rows.append(p)
            return np.stack(rows)
    patches = _color_patches(0)
    outs = []
    for vectorized in (False, True):
        clf = _classifier(False, cnn=Pinned())
        clf.set_live_board_vectorized(vectorized)
        outs.append(clf.classify_batch(patches))
    assert outs[0] == outs[1]


def test_hybrid_falls_back_when_soft_prior_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """距離の soft prior が有効 (BOOST_MAX != 0) なら一括経路を使わず従来経路。"""
    monkeypatch.setattr(hybrid_module, "BG_DIST_BOOST_MAX", 0.3)
    patches, distances, positions = _batch_inputs(1)
    outs = []
    for vectorized in (False, True):
        clf = _classifier(False)
        clf.set_live_board_vectorized(vectorized)
        outs.append(clf.classify_batch(patches, bg_distances=distances, cell_positions=positions))
    assert outs[0] == outs[1]
    clf = _classifier(False)
    clf.set_live_board_vectorized(True)
    assert clf._classify_batch_vectorized(patches, distances, positions) is None


def test_hybrid_rejects_non_finite_distance() -> None:
    patches, distances, positions = _batch_inputs(1)
    distances[1] = float("nan")
    outs = []
    for vectorized in (False, True):
        clf = _classifier(False)
        clf.set_live_board_vectorized(vectorized)
        outs.append(clf.classify_batch(patches, bg_distances=distances, cell_positions=positions))
    assert [o for o in outs[0]] == [o for o in outs[1]]


# ----------------------------------------------------------------------------
# ImageReader (read_board / read_board_hsv_only / read_both_boards)
# ----------------------------------------------------------------------------

def _assert_reads_identical(off: ImageReader, on: ImageReader, frame: np.ndarray,
                            skips: tuple[bool, bool] = (False, False)) -> None:
    for left, right in zip(off.read_both_boards(frame, skip_tier1_1p=skips[0], skip_tier1_2p=skips[1]),
                           on.read_both_boards(frame, skip_tier1_1p=skips[0], skip_tier1_2p=skips[1])):
        assert left._grid.tobytes() == right._grid.tobytes()
    for region in (DEFAULT_P1_REGION, DEFAULT_P2_REGION):
        assert (off.read_board_hsv_only(frame, region)._grid.tobytes()
                == on.read_board_hsv_only(frame, region)._grid.tobytes())


@pytest.mark.parametrize("native", [False, pytest.param(True, marks=needs_native)])
@pytest.mark.parametrize("with_bg", [False, True])
@pytest.mark.parametrize("guard", [False, True])
def test_read_boards_identical(native: bool, with_bg: bool, guard: bool) -> None:
    off = _reader(False, native, with_bg=with_bg, enable_patch_fp_hsv_guard=guard)
    on = _reader(True, native, with_bg=with_bg, enable_patch_fp_hsv_guard=guard)
    skips = [(False, False), (True, False), (False, True), (True, True)]
    for i, frame in enumerate(_frames(8, fill=0.45)):
        _assert_reads_identical(off, on, frame, skips[i % 4])


@needs_native
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_read_boards_identical_with_highlight_override_and_threshold(seed: int) -> None:
    kwargs: dict[str, Any] = dict(use_highlight_override=True, patch_ncc_threshold=0.80,
                                  enable_patch_fp_hsv_guard=True, bg_empty_threshold=70.0)
    off = _reader(False, True, with_bg=True, **kwargs)
    on = _reader(True, True, with_bg=True, **kwargs)
    for i, frame in enumerate(_frames(4, fill=0.5, seed=seed)):
        _assert_reads_identical(off, on, frame, (bool(i % 2), False))


@needs_native
def test_read_boards_identical_with_roi_offsets_and_side_sat() -> None:
    """ROI 補正・側別彩度較正 (状態を持つ) でも一致し、較正の履歴も同じ。"""
    off = _reader(False, True, with_bg=True)
    on = _reader(True, True, with_bg=True)
    for reader in (off, on):
        reader.set_side_sat_calibration(True)
    for i, frame in enumerate(_frames(14, fill=0.5, seed=4)):
        # 較正は領域の幾何ごとに 8 コマ貯める。最初の 10 コマは固定、その後で ROI 補正を動かす。
        moving = i >= 10
        offsets = ((float(i % 3), -float(i % 2)), (-2.0, float(i % 4))) if moving else ((0.0, 0.0), (0.0, 0.0))
        left = off.read_both_boards(frame, p1_roi_offset=offsets[0], p2_roi_offset=offsets[1])
        right = on.read_both_boards(frame, p1_roi_offset=offsets[0], p2_roi_offset=offsets[1])
        for a, b in zip(left, right):
            assert a._grid.tobytes() == b._grid.tobytes()
    assert off._side_sat_samples == on._side_sat_samples
    assert off._side_sat_scale == on._side_sat_scale
    assert off._side_sat_scale, "較正が一度も確定しなかった (検査として弱い)"


@needs_native
def test_live_window_records_identical() -> None:
    """ライブ窓の記録 (浮遊除去前の盤面・HSV 盤面・HSV 画素) も一致する。"""
    off = _reader(False, True, with_bg=True)
    on = _reader(True, True, with_bg=True)
    for frame in _frames(3, fill=0.5, seed=9):
        for reader in (off, on):
            reader._live_raw_boards, reader._live_hsv_boards = {}, {}
            reader._live_hsv_pixels, reader._live_raw_board_sec = {}, 0.0
            reader.read_both_boards(frame)
            for region in (DEFAULT_P1_REGION, DEFAULT_P2_REGION):
                reader.read_board_hsv_only(frame, region)
        for name in ("_live_raw_boards", "_live_hsv_boards"):
            left, right = getattr(off, name), getattr(on, name)
            assert left.keys() == right.keys() == {0, 1}
            assert all(left[k]._grid.tobytes() == right[k]._grid.tobytes() for k in left)
        assert off._live_hsv_pixels.keys() == on._live_hsv_pixels.keys() == {0, 1}
        assert all(off._live_hsv_pixels[k].tobytes() == on._live_hsv_pixels[k].tobytes() for k in (0, 1))


@needs_native
def test_vectorized_path_is_actually_used() -> None:
    """ON が本当に一括経路を通ったこと (通っていなければ一致の検証が空振りになる)。"""
    lv.reset_stats()
    on = _reader(True, True, with_bg=True, enable_patch_fp_hsv_guard=True)
    for frame in _frames(3, fill=0.5):
        on.read_both_boards(frame)
        on.read_board_hsv_only(frame, DEFAULT_P1_REGION)
    stats = lv.STATS
    assert stats["first_pass_patch_fp"] == 6
    assert stats["hsv_frame_pooled"] == 3
    assert stats["hsv_only_batch"] == 3 and stats["hsv_only_fallback"] == 0
    assert stats["hybrid_vectorized_calls"] >= 1
    assert stats["classify_native_calls"] >= 1 and stats["classify_per_patch_cells"] == 0
    assert stats["tier1_ncc_empty_candidates"] > 0


def test_static_mask_and_distance_fingerprint_use_legacy_path() -> None:
    """対応外 (距離型の背景 FP) は一括経路が None を返し、従来のループが同じ結果を出す。"""
    from src.background_fingerprint import capture_pair_fingerprint
    p1, p2 = DEFAULT_P1_REGION, DEFAULT_P2_REGION
    bg = capture_pair_fingerprint(_empty_backgrounds(1)[0], (p1.x, p1.y, p1.width, p1.height),
                                  (p2.x, p2.y, p2.width, p2.height))
    outs = []
    lv.reset_stats()
    for vectorized in (False, True):
        reader = ImageReader(classifier=_classifier(False), bg_fingerprint_p1=bg[0],
                             bg_fingerprint_p2=bg[1], use_ui_mask=False)
        reader.enable_live_board_vectorized(vectorized)
        outs.append([b._grid.tobytes() for b in reader.read_both_boards(_frames(1, fill=0.5)[0])])
    assert outs[0] == outs[1]
    assert lv.STATS["first_pass_fallback"] == 2 and lv.STATS["first_pass_patch_fp"] == 0


def test_non_1080p_hsv_only_uses_legacy_path() -> None:
    frame = cv2.resize(_frames(1)[0], (1280, 720), interpolation=cv2.INTER_AREA)
    off, on = (_reader(v, False, with_bg=False) for v in (False, True))
    assert (off.read_board_hsv_only(frame, DEFAULT_P1_REGION)._grid.tobytes()
            == on.read_board_hsv_only(frame, DEFAULT_P1_REGION)._grid.tobytes())


# ----------------------------------------------------------------------------
# 配線 (SPEED_DEFAULTS → 環境変数 → 子 process → recognition_config → 実効値の記録)
# ----------------------------------------------------------------------------

def test_recognition_config_adds_key_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_speed import BOARD_VECTORIZED_KEY, recognition_config
    env = "PUYO_" + BOARD_VECTORIZED_KEY.upper()
    base = {"enable_native_hsv_classifier": False}
    monkeypatch.delenv(env, raising=False)
    assert "enable_live_board_vectorized" not in recognition_config(base, True)
    monkeypatch.setenv(env, "0")
    assert "enable_live_board_vectorized" not in recognition_config(base, True)
    monkeypatch.setenv(env, "1")
    assert recognition_config(base, True)["enable_live_board_vectorized"] is True


def test_launcher_user_config_default_and_env_delivery() -> None:
    from src.phase_j.launcher import UserConfig
    from src.phase_j.live_speed import SPEED_DEFAULTS, validate
    assert UserConfig.__dataclass_fields__["live_board_vectorized"].default is False
    assert validate({"live_board_vectorized": True})["live_board_vectorized"] is True
    assert validate({})["live_board_vectorized"] is SPEED_DEFAULTS["live_board_vectorized"]
    with pytest.raises(ValueError):
        validate({"live_board_vectorized": "yes"})


def test_runtime_status_reports_effective_value(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_board_runtime
    from src.phase_j.live_speed import BOARD_VECTORIZED_KEY
    env = "PUYO_" + BOARD_VECTORIZED_KEY.upper()
    reader = _reader(False, False, with_bg=False)
    pipe = SimpleNamespace(_reader=reader)
    monkeypatch.delenv(env, raising=False)
    assert live_board_runtime.runtime_status(pipe) == dict(active=False, reason="disabled")
    monkeypatch.setenv(env, "1")
    assert live_board_runtime.runtime_status(pipe) == dict(active=False, reason="activation_failed")
    reader.enable_live_board_vectorized(True)
    status = live_board_runtime.runtime_status(pipe)
    assert status["active"] is True and status["native_hsv"] is False
    assert status["reason"] == live_board_runtime.REASON_ENABLED_WITHOUT_NATIVE_HSV


def test_trial_flags_record_has_runtime_entry(tmp_path: Any) -> None:
    import argparse
    import json

    from src.phase_j import live_speed
    options = argparse.Namespace(output=tmp_path, **{k: v for k, v in live_speed.SPEED_DEFAULTS.items()})
    options.live_board_vectorized = True
    record = live_speed.trial_flags_record(options)
    assert list(record["flags"]) == list(live_speed.TRIAL_FLAG_KEYS) and record["flags"]["live_board_vectorized"]
    assert record["runtime"]["live_board_vectorized"] == dict(active=False, reason="recognition_not_started")
    assert list(record["env"]) == ["PUYO_"+key.upper() for key in live_speed.TRIAL_FLAG_KEYS]
    json.dumps(record)


# ----------------------------------------------------------------------------
# 検証台 (比較器) の健全性
# ----------------------------------------------------------------------------

def test_verifier_selftest_detects_injected_differences() -> None:
    from scripts.verify_live_board_vectorized import selftest
    result = selftest()
    assert result and all(result.values()), result


def test_verifier_digest_ignores_flag_names_and_noise_only() -> None:
    from scripts.verify_live_board_vectorized import STATE_NOISE_NAMES, diff_maps, digest_tree
    left = SimpleNamespace(value=1.5, _live_board_vectorized=False, _live_raw_board_sec=1.0)
    right = SimpleNamespace(value=1.5, _live_board_vectorized=True, _live_raw_board_sec=2.0)
    assert not diff_maps(digest_tree(left, "x", STATE_NOISE_NAMES), digest_tree(right, "x", STATE_NOISE_NAMES))
    right.value = 1.5000000000000002
    assert diff_maps(digest_tree(left, "x", STATE_NOISE_NAMES), digest_tree(right, "x", STATE_NOISE_NAMES))
    # 結果の節 (除外名なし) では計時名を持つ値も比べる
    assert diff_maps(digest_tree(SimpleNamespace(time_sec=1.0), "r"), digest_tree(SimpleNamespace(time_sec=2.0), "r"))
