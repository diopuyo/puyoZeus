"""live_recog_speed_py (認識ループの Python 側の出力不変の高速化) の単体試験。

どの項目も「直す前と直した後の出力が同じ」ことを、同じ入力を従来経路と新経路へ通して比べる。
旗が OFF のときは従来の経路と完全に同じ (呼出の回数まで) ことも確かめる。
"""
from __future__ import annotations

import argparse
from threading import Event
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src import recog_speed_py
from src.image_reader import (DEFAULT_P1_REGION, DEFAULT_P2_REGION, ColorClassifier, ImageReader)
from src.next_slide_detector import NextSlideDetector
from src.recog_speed_py import (ALL_ITEMS, ENV_AUDIT, ENV_FLAG, ENV_ITEMS, ENV_PAIR_LAZY, ITEM_DISPLAY_SCORES,
                                ITEM_HSV_MEMO, ITEM_IMAGE_BATCH, ITEM_IMAGE_CROPS, ITEM_PAIR_LAZY,
                                ITEM_SLIDE_GRAY, STATS, STRICT_ITEMS, enabled, read_hsv_only)

FRAME_SHAPE = (1080, 1920, 3)
PALETTE = ((0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (30, 30, 30))
SEED = 20261025
NOISE = 8


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """各試験は旗 OFF から始める。件数の記録も 0 に戻す。"""
    for name in (ENV_FLAG, ENV_ITEMS, ENV_AUDIT, ENV_PAIR_LAZY):
        monkeypatch.delenv(name, raising=False)
    STATS.clear()


def turn_on(monkeypatch: pytest.MonkeyPatch, *items: str, audit: bool = False) -> None:
    monkeypatch.setenv(ENV_FLAG, '1')
    if items:
        monkeypatch.setenv(ENV_ITEMS, ','.join(items))
    if audit:
        monkeypatch.setenv(ENV_AUDIT, '1')


def board_image(rng: np.random.Generator, shape: tuple = FRAME_SHAPE) -> np.ndarray:
    """両盤面のセルへ 5 色・空をばらまいた画像。分類の結果が自明にならないよう画素ノイズも入れる。"""
    image = rng.integers(0, 40, shape, dtype=np.uint8)
    height, width = shape[:2]
    for region in (DEFAULT_P1_REGION, DEFAULT_P2_REGION):
        for row in range(1, 13):
            for col in range(6):
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                color = np.array(PALETTE[int(rng.integers(0, len(PALETTE)))], dtype=np.int16)
                noise = rng.integers(-NOISE, NOISE + 1, (max(0, min(y2, height) - y1), max(0, min(x2, width) - x1), 3))
                if noise.size:
                    image[y1:y2, x1:x2] = np.clip(color + noise, 0, 255).astype(np.uint8)
    return image


# ----------------------------------------------------------------------------
# 旗・配線
# ----------------------------------------------------------------------------

def test_flag_is_off_by_default_and_items_are_listed() -> None:
    assert not any(enabled(item) for item in ALL_ITEMS)


def test_items_filter_only_works_when_flag_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_ITEMS, ITEM_HSV_MEMO)
    assert not enabled(ITEM_HSV_MEMO)            # 旗が OFF なら項目の指定は無視される
    monkeypatch.setenv(ENV_FLAG, '1')
    assert enabled(ITEM_HSV_MEMO) and not enabled(ITEM_SLIDE_GRAY)
    monkeypatch.delenv(ENV_ITEMS)
    assert all(enabled(item) for item in STRICT_ITEMS)
    assert not enabled(ITEM_PAIR_LAZY)           # 壁時計に依存する項目は、この旗では働かない


def test_wall_clock_item_has_its_own_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_PAIR_LAZY, '1')
    assert enabled(ITEM_PAIR_LAZY) and not any(enabled(item) for item in STRICT_ITEMS)
    monkeypatch.setenv(ENV_FLAG, '1')
    monkeypatch.setenv(ENV_ITEMS, ITEM_HSV_MEMO)  # 項目の絞り込みは専用の旗の項目には効かない
    assert enabled(ITEM_PAIR_LAZY)


def test_wiring_single_source_and_launcher_field(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import launcher, live_speed
    from src.production_config import LIVE_RECOG_SPEED_PY_DEFAULT
    assert LIVE_RECOG_SPEED_PY_DEFAULT is live_speed.SPEED_DEFAULTS['live_recog_speed_py'] is False
    assert launcher.UserConfig.live_recog_speed_py is False
    assert launcher.UserConfig.live_recog_speed_py_pair_lazy is False
    assert live_speed.SPEED_DEFAULTS['live_recog_speed_py_pair_lazy'] is False
    assert live_speed.validate({})['live_recog_speed_py'] is False
    with pytest.raises(ValueError):
        live_speed.validate({'live_recog_speed_py': 1})
    # 実効値の記録の鍵 (24d の組立・検収が全項目で腕を組む) には足さない
    assert 'live_recog_speed_py' not in live_speed.TRIAL_FLAG_KEYS
    assert 'live_recog_speed_py_pair_lazy' not in live_speed.TRIAL_FLAG_KEYS


def test_runtime_report_is_silent_when_off_and_records_effective_items_when_on(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """実効値 (どの項目が働く設定か・使い回した件数) を run.log へ 1 行残す。全 OFF なら何も出さない。"""
    from src.phase_j.live_process import report_recog_speed
    from src.recog_speed_py import runtime_report
    assert runtime_report() is None
    report_recog_speed()
    assert capsys.readouterr().out == ''
    turn_on(monkeypatch, ITEM_HSV_MEMO)
    STATS['hsv_only_hit'] = 3
    report = runtime_report()
    assert report['items'][ITEM_HSV_MEMO] and not report['items'][ITEM_SLIDE_GRAY]
    assert report['stats'] == {'hsv_only_hit': 3}
    report_recog_speed()
    line = capsys.readouterr().out
    assert line.startswith('RECOG_SPEED_PY ') and '"hsv_only_hit": 3' in line


@pytest.mark.parametrize('value', [True, False])
def test_configure_delivers_env_to_children(value: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_speed import SPEED_DEFAULTS, configure
    options = argparse.Namespace(**dict(SPEED_DEFAULTS, live_recog_speed_py=value), cpu_threads=1,
        source='dshow', b3_judgment=True, split_evaluation=True, worker_mode='process', compare=False)
    environment: dict[str, str] = {}
    monkeypatch.setattr('os.environ', environment)
    configure(options, argparse.ArgumentParser())
    assert environment[ENV_FLAG] == str(int(value))


# ----------------------------------------------------------------------------
# A2: read_board_hsv_only の同コマ重複
# ----------------------------------------------------------------------------

class CountingReader:
    """read_board_hsv_only の呼出を数える実物の ImageReader。"""

    def __init__(self) -> None:
        self.reader = ImageReader(classifier=ColorClassifier())
        self.calls = 0
        self.reader._live_hsv_boards = {}      # prepare_recognition が毎コマ作る窓と同じ
        self.reader._live_raw_board_sec = 0.

    def read_board_hsv_only(self, frame: np.ndarray, region: Any) -> Any:
        self.calls += 1
        return self.reader.read_board_hsv_only(frame, region)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.reader, name)


def test_hsv_memo_off_calls_reader_every_time() -> None:
    counting = CountingReader()
    image = board_image(np.random.default_rng(SEED))
    for _ in range(3):
        read_hsv_only(counting, image, DEFAULT_P1_REGION)
    assert counting.calls == 3 and not STATS


def test_hsv_memo_same_frame_is_computed_once_and_equal(monkeypatch: pytest.MonkeyPatch) -> None:
    turn_on(monkeypatch, ITEM_HSV_MEMO, audit=True)
    rng = np.random.default_rng(SEED)
    plain = ImageReader(classifier=ColorClassifier())
    memo = ImageReader(classifier=ColorClassifier())
    hits = 0
    for _ in range(6):
        image = board_image(rng)
        for region in (DEFAULT_P1_REGION, DEFAULT_P2_REGION, DEFAULT_P1_REGION, DEFAULT_P2_REGION):
            expected = plain.read_board_hsv_only(image, region)
            actual = read_hsv_only(memo, image, region)
            assert actual._grid.tobytes() == expected._grid.tobytes()
        hits += 2
    assert STATS['hsv_only_hit'] == hits and STATS['hsv_only_miss'] == 12
    assert STATS['audit_checked'] == hits and STATS['audit_mismatch'] == 0


def test_hsv_memo_returns_independent_boards(monkeypatch: pytest.MonkeyPatch) -> None:
    turn_on(monkeypatch, ITEM_HSV_MEMO)
    reader = ImageReader(classifier=ColorClassifier())
    image = board_image(np.random.default_rng(SEED))
    first = read_hsv_only(reader, image, DEFAULT_P1_REGION)
    expected = first._grid.copy()
    first._grid[:] = 7                      # 呼出元が書き換えても
    second = read_hsv_only(reader, image, DEFAULT_P1_REGION)
    assert (second._grid == expected).all()
    second._grid[:] = 9
    third = read_hsv_only(reader, image, DEFAULT_P1_REGION)
    assert (third._grid == expected).all()


def test_hsv_memo_recomputes_for_new_frame_object_even_if_same_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    turn_on(monkeypatch, ITEM_HSV_MEMO)
    counting = CountingReader()
    image = board_image(np.random.default_rng(SEED))
    read_hsv_only(counting, image, DEFAULT_P1_REGION)
    read_hsv_only(counting, image.copy(), DEFAULT_P1_REGION)
    assert counting.calls == 2


def test_hsv_memo_invalidated_by_classifier_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """色範囲の注入・彩度スケール・モードが変われば、同じコマでも結果を使い回さない。"""
    turn_on(monkeypatch, ITEM_HSV_MEMO)
    counting = CountingReader()
    image = board_image(np.random.default_rng(SEED))
    classifier = counting.reader._get_hsv_classifier()
    read_hsv_only(counting, image, DEFAULT_P1_REGION)
    classifier.set_color_ranges_from_simple({1: (0, 180, 0, 255, 0, 255)})   # 赤の範囲を広げる
    changed = read_hsv_only(counting, image, DEFAULT_P1_REGION)
    assert counting.calls == 2
    assert changed._grid.tobytes() == ImageReader(classifier=classifier).read_board_hsv_only(
        image, DEFAULT_P1_REGION)._grid.tobytes()
    classifier.set_s_min_scale(0.5)
    read_hsv_only(counting, image, DEFAULT_P1_REGION)
    assert counting.calls == 3
    read_hsv_only(counting, image, DEFAULT_P1_REGION)   # 状態が同じなら使い回す
    assert counting.calls == 3 and STATS['hsv_only_hit'] == 1


def test_hsv_memo_keeps_live_observation_side_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓 (_live_hsv_boards) はコマごとに作り直される。同じ画像が次のコマに渡されても記録を残す。"""
    turn_on(monkeypatch, ITEM_HSV_MEMO)
    reader = ImageReader(classifier=ColorClassifier())
    image = board_image(np.random.default_rng(SEED))
    reader._live_hsv_boards = {}
    reader._live_raw_board_sec = 0.
    first = read_hsv_only(reader, image, DEFAULT_P2_REGION)
    reader._live_hsv_boards = {}            # 次のコマ (prepare_recognition の作り直し)
    read_hsv_only(reader, image, DEFAULT_P2_REGION)
    assert (reader._live_hsv_boards[1]._grid == first._grid).all()


def test_hsv_memo_audit_counts_a_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """測定器の健全性: 同じ画像オブジェクトが書き換わって結果が変わる場合を、監査が違いとして数えること。"""
    turn_on(monkeypatch, ITEM_HSV_MEMO, audit=True)
    reader = ImageReader(classifier=ColorClassifier())
    image = board_image(np.random.default_rng(SEED))
    cached = read_hsv_only(reader, image, DEFAULT_P1_REGION)
    x1, y1, x2, y2 = DEFAULT_P1_REGION.cell_sample_rect(12, 0)
    for color in PALETTE:                            # 分類が変わる色を見つけて in-place で書き込む
        image[y1:y2, x1:x2] = color
        if reader.read_board_hsv_only(image, DEFAULT_P1_REGION)._grid.tobytes() != cached._grid.tobytes():
            break
    STATS.clear()
    read_hsv_only(reader, image, DEFAULT_P1_REGION)
    assert STATS['audit_checked'] == 1 and STATS['audit_mismatch'] == 1


def test_hsv_memo_bypasses_when_state_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    turn_on(monkeypatch, ITEM_HSV_MEMO)
    calls: list[int] = []
    stub = SimpleNamespace(read_board_hsv_only=lambda frame, region: calls.append(1) or 'board')
    assert read_hsv_only(stub, object(), DEFAULT_P1_REGION) == 'board'
    assert read_hsv_only(stub, object(), DEFAULT_P1_REGION) == 'board' and len(calls) == 2


# ----------------------------------------------------------------------------
# A7: NextSlideDetector の前コマのグレー化
# ----------------------------------------------------------------------------

def slide_sequence(rng: np.random.Generator, steps: int) -> list[np.ndarray | None]:
    """小さな揺れ・スライド (NEXT 領域の大きな差)・None・解像度不足を混ぜたコマ列。"""
    base = rng.integers(0, 255, FRAME_SHAPE, dtype=np.uint8)
    frames: list[np.ndarray | None] = []
    for step in range(steps):
        frame = base.copy()
        for left, right in ((700, 830), (1090, 1220)):          # 1P と 2P の NEXT 領域
            frame[100:400, left:right] = rng.integers(0, 255 if step % 7 == 0 else 6, (300, right - left, 3),
                                                      dtype=np.uint8)
        frames.append(frame)
    frames.insert(5, None)
    frames.insert(9, np.zeros((100, 100, 3), dtype=np.uint8))
    return frames


@pytest.mark.parametrize('side', ['1P', '2P'])
def test_slide_gray_cache_gives_identical_results(side: str, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = slide_sequence(np.random.default_rng(SEED), 60)
    legacy, cached = NextSlideDetector(side=side), NextSlideDetector(side=side)
    expected = [legacy.update(prev, curr) for prev, curr in zip(frames, frames[1:])]
    turn_on(monkeypatch, ITEM_SLIDE_GRAY)
    actual = [cached.update(prev, curr) for prev, curr in zip(frames, frames[1:])]
    assert actual == expected                       # diff_score の浮動小数まで厳密に同じ
    assert any(r.slide_motion for r in expected) and not all(r.slide_motion for r in expected)
    assert STATS['slide_gray_reused'] > 0


def test_slide_gray_cache_only_used_for_same_object_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    turn_on(monkeypatch, ITEM_SLIDE_GRAY)
    rng = np.random.default_rng(SEED)
    a, b, c = (rng.integers(0, 255, FRAME_SHAPE, dtype=np.uint8) for _ in range(3))
    detector = NextSlideDetector(side='1P')
    detector.update(a, b)
    assert STATS['slide_gray_reused'] == 0           # 最初は何も持っていない
    detector.update(b, c)                            # prev が前回の現コマ b と同じ物 → 4 ROI ぶん使い回す
    assert STATS['slide_gray_reused'] == 4
    detector.update(a, b)                            # prev=a は前回の現コマ c ではない → 使い回さない
    assert STATS['slide_gray_reused'] == 4
    detector.update(b.copy(), c)                     # 同じ画素でも別オブジェクトは使い回さない
    assert STATS['slide_gray_reused'] == 4
    detector.update(c, a)                            # 前回の現コマ c と同じ物
    assert STATS['slide_gray_reused'] == 8
    detector.reset()
    detector.update(a, b)                            # reset 後は何も持っていない
    assert STATS['slide_gray_reused'] == 8


def test_slide_gray_audit_counts_zero_for_untouched_frames_and_detects_inplace_change(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """測定器の健全性: 画像が触られていなければ監査の違いは 0、同じ画像を書き換えれば検出する。"""
    turn_on(monkeypatch, ITEM_SLIDE_GRAY, audit=True)
    frames = slide_sequence(np.random.default_rng(SEED), 20)
    detector = NextSlideDetector(side='1P')
    for prev, curr in zip(frames, frames[1:]):
        detector.update(prev, curr)
    assert STATS['audit_checked'] > 0 and STATS['audit_mismatch'] == 0
    a, b = frames[0], frames[1]
    detector = NextSlideDetector(side='1P')
    detector.update(a, b)
    b[200:300, 720:780] = 255 - b[200:300, 720:780]              # b は次の update の prev になる
    detector.update(b, a)
    assert STATS['audit_mismatch'] > 0


def test_slide_gray_off_keeps_no_cache() -> None:
    frames = slide_sequence(np.random.default_rng(SEED), 5)
    detector = NextSlideDetector(side='1P')
    for prev, curr in zip(frames, frames[1:]):
        detector.update(prev, curr)
    assert detector._gray_cache is None and not STATS


# ----------------------------------------------------------------------------
# A3: 発火前画像リング
# ----------------------------------------------------------------------------

def strip(boards: list[dict]) -> list[dict]:
    return [{k: v for k, v in b.items() if k != 'image_compute_ms'} for b in boards]


@pytest.mark.parametrize('shape', [FRAME_SHAPE, (900, 1700, 3), (1080, 1000, 3)])
def test_board_crops_patch_equals_full_image_slice(shape: tuple) -> None:
    """切り出しは全画面の image[y1:y2, x1:x2] と同じ画素・同じ形 (画像の端で切れる寸法も)。"""
    from src.phase_j.live_v49f_images import COLS, RING_CELLS, BoardCrops, _region_boxes
    image = board_image(np.random.default_rng(SEED), shape)
    boxes = _region_boxes(shape[0], shape[1])
    assert boxes is not None
    crops = BoardCrops(image, boxes)
    for side, region in enumerate((DEFAULT_P1_REGION, DEFAULT_P2_REGION)):
        for cell in RING_CELLS:
            x1, y1, x2, y2 = region.cell_sample_rect(*divmod(cell, COLS))
            expected = image[y1:y2, x1:x2]
            actual = crops.patch(side, x1, y1, x2, y2)
            assert actual.shape == expected.shape and np.array_equal(actual, expected)


def test_board_crops_are_much_smaller_than_full_frame() -> None:
    from src.phase_j.live_v49f_images import BoardCrops, _region_boxes
    image = board_image(np.random.default_rng(SEED))
    crops = BoardCrops(image, _region_boxes(*FRAME_SHAPE[:2]))
    assert sum(p.nbytes for p in crops.pixels) < image.nbytes * 0.3


def test_board_crops_fall_back_when_region_is_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    """負の座標は numpy で末尾から数える別の意味になるので、盤面だけの保持にはしない (全画面のまま)。"""
    from src.phase_j import live_v49f_images as module
    from src.image_reader import BoardRegion
    monkeypatch.setattr('src.image_reader.DEFAULT_P1_REGION', BoardRegion(x=-200, y=160, width=384, height=720))
    assert module._region_boxes(1080, 1920) is None


@pytest.mark.parametrize('items', [(), (ITEM_IMAGE_CROPS,), (ITEM_IMAGE_BATCH,), (ITEM_IMAGE_CROPS, ITEM_IMAGE_BATCH)])
def test_live_images_boards_identical_for_every_item_combination(items: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_v49f_images import LiveImages
    rng = np.random.default_rng(SEED)
    images = [board_image(rng) for _ in range(3)]
    legacy = LiveImages()
    for stamp, image in enumerate(images):
        legacy.append(float(stamp), image)
    expected = [strip(legacy.boards(2.0, 2.0, side, (-1.0, 0.0))) for side in (0, 1)]
    if items:
        turn_on(monkeypatch, *items)
    fast = LiveImages()
    for stamp, image in enumerate(images):
        fast.append(float(stamp), image)
    actual = [strip(fast.boards(2.0, 2.0, side, (-1.0, 0.0))) for side in (0, 1)]
    assert actual == expected
    assert any(any(b['board']) for pair in actual for b in pair)       # 全部 0 の自明な一致ではない
    retained = STATS['image_crops_retained']
    assert (retained == 3) == (ITEM_IMAGE_CROPS in items)


def test_live_images_retained_frame_is_not_aliased(monkeypatch: pytest.MonkeyPatch) -> None:
    """画像を呼出元が後で書き換えても、リングの中身は append した時点のまま (従来の .copy() と同じ)。"""
    from src.phase_j.live_v49f_images import LiveImages
    turn_on(monkeypatch, ITEM_IMAGE_CROPS)
    image = board_image(np.random.default_rng(SEED))
    ring = LiveImages()
    ring.append(0.0, image)
    before = strip(ring.boards(0.0, 0.0, 0, (0.0,)))
    image[:] = 0
    assert strip(ring.boards(0.0, 0.0, 0, (0.0,))) == before


def test_live_images_batch_masks_ui_cell_like_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """×印 UI の判定セル (1, 2) は、従来と同じく 0 になる (一括でも対象セルだけマスクが効く)。"""
    from src.phase_j.live_v49f_images import LiveImages
    image = board_image(np.random.default_rng(SEED))
    ring = LiveImages()
    monkeypatch.setattr(ring.ui, 'is_ui', lambda patch: True)
    legacy = [dict(board=b) for b in ring._classify_per_cell(image, DEFAULT_P1_REGION, 0)]
    batched = [dict(board=b) for b in ring._classify_batched(image, DEFAULT_P1_REGION, 0)]
    assert legacy == batched
    assert all(entry['board'][6 + 2] == 0 for entry in batched)       # 6 = 隠し段ぶん、(1, 2) は 8 番目


# ----------------------------------------------------------------------------
# A6: 表示得点
# ----------------------------------------------------------------------------

class StubOcr:
    def __init__(self, scores: dict[str, int | None]) -> None:
        self.scores, self.calls = scores, 0

    def read_side(self, frame: Any, side: str) -> tuple:
        self.calls += 1
        return self.scores[side], 1.0


def display_fixture(raw: tuple | None, scores: dict[str, int | None]) -> tuple:
    from src.phase_j import live_score_share
    ocr = StubOcr(scores)
    tracker = SimpleNamespace(_ocr=ocr)      # 両側の ScoreTracker が owner._score_ocr で読んでいる状態
    owner = SimpleNamespace(_score_ocr=ocr, _live_share_scores=True, _live_raw_scores=raw,
                            _score_tracker_1p=tracker, _score_tracker_2p=tracker)
    frame = SimpleNamespace(index=12, media_sec=0.4, image=np.zeros((2, 2, 3), dtype=np.uint8))
    return live_score_share, owner, ocr, frame


def test_display_scores_off_rereads(monkeypatch: pytest.MonkeyPatch) -> None:
    module, owner, ocr, frame = display_fixture((12, 0.4, 700, None), {'1P': 700, '2P': None})
    assert module.displayed_scores(owner, frame) == (700, None) and ocr.calls == 2


def test_display_scores_on_uses_shared_raw_values_and_matches_reread(monkeypatch: pytest.MonkeyPatch) -> None:
    turn_on(monkeypatch, ITEM_DISPLAY_SCORES)
    for scores in ({'1P': 700, '2P': None}, {'1P': 0, '2P': 123456}, {'1P': None, '2P': None}):
        module, owner, ocr, frame = display_fixture((12, 0.4, scores['1P'], scores['2P']), scores)
        shared = module.displayed_scores(owner, frame)
        assert ocr.calls == 0 and shared == (scores['1P'], scores['2P'])
        from src.exchange_event_overlay import displayed_scores_from_pipeline
        assert shared == displayed_scores_from_pipeline(owner, frame.image)


@pytest.mark.parametrize('raw', [None, (11, 0.4, 700, 5), (12, 0.5, 700, 5)])
def test_display_scores_rereads_when_not_shared_for_this_frame(raw: tuple | None, monkeypatch: pytest.MonkeyPatch) -> None:
    turn_on(monkeypatch, ITEM_DISPLAY_SCORES)
    module, owner, ocr, frame = display_fixture(raw, {'1P': 700, '2P': 5})
    assert module.displayed_scores(owner, frame) == (700, 5) and ocr.calls == 2


def test_display_scores_audit_counts_a_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """測定器の健全性: 共有した値が再読と食い違えば監査が数える (一致なら 0)。"""
    turn_on(monkeypatch, ITEM_DISPLAY_SCORES, audit=True)
    module, owner, ocr, frame = display_fixture((12, 0.4, 700, 5), {'1P': 700, '2P': 5})
    assert module.displayed_scores(owner, frame) == (700, 5)
    assert STATS['audit_checked'] == 1 and STATS['audit_mismatch'] == 0
    module, owner, ocr, frame = display_fixture((12, 0.4, 700, 5), {'1P': 700, '2P': 6})   # 再読すると 2P が違う
    assert module.displayed_scores(owner, frame) == (700, 5)
    assert STATS['audit_mismatch'] == 1


def test_display_scores_rereads_without_share_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """b3_share_score_read が OFF なら生得点は共有されない (_live_share_scores=False) ので再読する。"""
    turn_on(monkeypatch, ITEM_DISPLAY_SCORES)
    module, owner, ocr, frame = display_fixture((12, 0.4, 700, 5), {'1P': 700, '2P': 5})
    owner._live_share_scores = False
    assert module.displayed_scores(owner, frame) == (700, 5) and ocr.calls == 2


def test_display_scores_rereads_when_tracker_ocr_was_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    """tracker が読んだ OCR が owner._score_ocr と別物なら、同じ読取りとは言えないので再読する。"""
    turn_on(monkeypatch, ITEM_DISPLAY_SCORES)
    module, owner, ocr, frame = display_fixture((12, 0.4, 700, 5), {'1P': 700, '2P': 5})
    owner._score_tracker_2p = SimpleNamespace(_ocr=StubOcr({'1P': 0, '2P': 0}))
    assert module.displayed_scores(owner, frame) == (700, 5) and ocr.calls == 2


# ----------------------------------------------------------------------------
# A1: 終了の組の crop (壁時計に依存するので、全行一致でなく決定的な単体で確かめる)
# ----------------------------------------------------------------------------

WAIT_SEC = 3.0


def _blocked_detector(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list, Event, Event]:
    """worker が評価の途中で止まる終了の組 detector。評価に渡った (時刻, crop) を seen へ残す。"""
    from src.async_match_end_pair import AsyncMatchEndPairDetector
    detector = AsyncMatchEndPairDetector({})
    entered, release = Event(), Event()
    seen: list[tuple[float, dict]] = []

    def evaluate(crops: dict, stamp: float) -> Any:
        seen.append((stamp, crops))
        entered.set()
        assert release.wait(WAIT_SEC)
        return detector._result({}, {})

    monkeypatch.setattr(detector, '_evaluate', evaluate)
    monkeypatch.setattr('src.async_match_end_pair.PAIR_SCAN_INTERVAL_SEC', 0.0)
    return detector, seen, entered, release


@pytest.mark.parametrize('lazy', [False, True])
def test_pair_crops_are_equal_copies_when_worker_takes_them(lazy: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.async_match_end_pair import LazyCrops
    from src.match_end_pair import PAIR_ROIS
    if lazy:
        monkeypatch.setenv(ENV_PAIR_LAZY, '1')
    detector, seen, entered, release = _blocked_detector(monkeypatch)
    rng = np.random.default_rng(SEED)
    first, second = (rng.integers(0, 255, FRAME_SHAPE, dtype=np.uint8) for _ in range(2))
    try:
        detector.update(first, 1.0)
        assert entered.wait(WAIT_SEC)                          # worker が 1 件目を取り出した
        detector.update(second, 2.0)                           # 待機の最新 1 件 (worker は 1 件目で止まっている)
        pending = detector._pending[2]
        assert isinstance(pending, LazyCrops) == lazy
        for key, (x, y, w, h) in PAIR_ROIS.items():
            assert np.shares_memory(pending[key], second) == lazy     # OFF は呼出時に複製済み
        release.set()
        with detector._condition:
            assert detector._condition.wait_for(lambda: len(seen) == 2, timeout=WAIT_SEC)
    finally:
        release.set()
        detector.close()
    for (stamp, crops), frame in zip(seen, (first, second)):
        for key, (x, y, w, h) in PAIR_ROIS.items():
            assert type(crops) is dict and np.array_equal(crops[key], frame[y:y+h, x:x+w])
            assert not np.shares_memory(crops[key], frame)         # 評価に渡るのはどちらも複製
    assert [stamp for stamp, _ in seen] == [1.0, 2.0]


def test_pair_update_returns_same_values_lazy_or_not(monkeypatch: pytest.MonkeyPatch) -> None:
    """同じコマ列・同じ待ち方で、ロック・照合元の時刻・勝者が同じ (完了を待ってから次のコマを入れる)。"""
    from src.async_match_end_pair import AsyncMatchEndPairDetector
    no_pair, pair = [0.1] * 4, [0.9, 0.1, 0.1, 0.9]      # PAIR_ROIS の順 (yatta_1p, yatta_2p, batan_1p, batan_2p)

    def run(lazy: bool) -> list[tuple]:
        if lazy:
            monkeypatch.setenv(ENV_PAIR_LAZY, '1')
        else:
            monkeypatch.delenv(ENV_PAIR_LAZY, raising=False)
        rng = np.random.default_rng(SEED)
        detector = AsyncMatchEndPairDetector({})
        scripted = iter(no_pair + pair + no_pair * 6)
        monkeypatch.setattr(detector, '_score_crop', lambda crop, key: next(scripted))
        monkeypatch.setattr('src.async_match_end_pair.PAIR_SCAN_INTERVAL_SEC', 0.0)
        out = []
        try:
            for step in range(5):
                frame = rng.integers(0, 255, FRAME_SHAPE, dtype=np.uint8)
                out.append((detector.update(frame, float(step)), detector.lock_started_t,
                            getattr(detector.last_detection, 'winner', None)))
                with detector._condition:
                    assert detector._condition.wait_for(lambda: detector._completed is not None, timeout=WAIT_SEC)
        finally:
            detector.close()
        return out

    expected, actual = run(False), run(True)
    assert actual == expected
    assert expected[1][0] is False                   # 2 コマ目 (組あり) の結果は次の update で採用される
    assert expected[2] == (True, 1.0, '1P')


# ----------------------------------------------------------------------------
# 検証台の比較器 (測定器の健全性)
# ----------------------------------------------------------------------------

def test_gate_selftest_detects_differences() -> None:
    from scripts.verify_recog_speed_py import selftest
    assert all(selftest().values())
