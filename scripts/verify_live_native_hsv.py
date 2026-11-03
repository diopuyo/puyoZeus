"""保存 PNG だけで配信本体の HSV 同一性と read_both_boards の時間を測る。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import cv2
import numpy as np

from src.board import BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS
from src.image_reader import ImageReader

NATIVE_ENV = 'PUYO_LIVE_NATIVE_HSV_MAIN'
REPEATS = 5
PERCENTILES = (50, 95)


def load_frames(manifest: Path) -> list[np.ndarray]:
    """micro 計測に使った frames3 の実コマを選ぶ。プレビューは除外する。"""
    paths = [path for path in json.loads(manifest.read_text(encoding='utf-8'))
             if '/frames3/' in path.replace('\\', '/') and 'preview' not in path]
    frames = [cv2.imread(path) for path in paths]
    assert frames and all(frame is not None for frame in frames), '保存 PNG が不足しています'
    return frames


@contextmanager
def reader_pair(assets: Path, config_path: Path) -> Iterator[tuple[ImageReader, ImageReader]]:
    """同じ配布重みと実配信設定で、スイッチだけ違う本体を作る。"""
    from src.phase_j.live_speed import recognition_config
    from src.recognition_pipeline import RecognitionPipeline
    directory, previous = Path.cwd(), os.environ.get(NATIVE_ENV)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config.pop('enable_native_hsv_classifier', None)
    try:
        os.chdir(assets)
        readers = []
        for enabled in (False, True):
            os.environ[NATIVE_ENV] = str(int(enabled))
            pipe = RecognitionPipeline.load_default(**recognition_config(config, True))
            assert pipe._native_hsv_active is enabled
            assert pipe._reader._classifier._cnn is not None, '配布 CNN が読み込まれていません'
            readers.append(pipe._reader)
        yield readers[0], readers[1]
    finally:
        os.chdir(directory)
        if previous is None:
            os.environ.pop(NATIVE_ENV, None)
        else:
            os.environ[NATIVE_ENV] = previous


def cell_outputs(reader: ImageReader, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """全可視セルの分類色と確率分布を、推論や隠し段補完の前に取得する。"""
    patches = []
    for region in (reader._p1_region, reader._p2_region):
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            for col in range(BOARD_COLS):
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                patches.append(frame[y1:y2, x1:x2])
    classifier = reader._classifier
    colors = np.asarray(classifier.classify_batch(patches), dtype=np.uint8)
    probabilities = classifier.predict_proba_batch(patches)
    return colors, probabilities


def assert_bytes_equal(left: np.ndarray, right: np.ndarray) -> None:
    """許容誤差なし。浮動小数点も生バイトで照合する。"""
    assert left.dtype == right.dtype and left.shape == right.shape
    assert left.tobytes() == right.tobytes()


def verify_pair(readers: tuple[ImageReader, ImageReader], frames: list[np.ndarray]) -> dict[str, int]:
    """分類色・各クラス確率・補完後の両盤面が全コマでビット一致するか検証する。"""
    cells = board_cells = probabilities = 0
    for frame in frames:
        off, on = (cell_outputs(reader, frame) for reader in readers)
        for left, right in zip(off, on):
            assert_bytes_equal(left, right)
        cells += off[0].size
        probabilities += off[1].size
        boards_off, boards_on = (reader.read_both_boards(frame) for reader in readers)
        for left, right in zip(boards_off, boards_on):
            assert_bytes_equal(left._grid, right._grid)
            board_cells += left._grid.size
    return dict(frames=len(frames), visible_cells=cells, probability_values=probabilities,
                boards=len(frames)*len(readers), board_cells=board_cells, mismatches=0)


def measure_pair(readers: tuple[ImageReader, ImageReader], frames: list[np.ndarray]) -> dict[str, Any]:
    """各14コマで暖機後、順番を交互にして同じコマを ON/OFF で測る。"""
    samples: list[list[float]] = [[], []]
    for reader in readers:
        for frame in frames:
            reader.read_both_boards(frame)
    for index in range(len(frames)*REPEATS):
        for mode in ((0, 1) if index % 2 == 0 else (1, 0)):
            started = perf_counter()
            readers[mode].read_both_boards(frames[index % len(frames)])
            samples[mode].append((perf_counter()-started)*1000)
    return {name: dict(n=len(values), p50_ms=float(np.percentile(values, PERCENTILES[0])),
                       p95_ms=float(np.percentile(values, PERCENTILES[1])), samples_ms=values)
            for name, values in zip(('off', 'on'), samples)}


def main() -> None:
    """配信や動画デコーダを起動せず、明示された PNG とモデルだけを読む。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'assets', 'config', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    options = parser.parse_args()
    import torch
    import puyo_core
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    frames = load_frames(options.manifest)
    with reader_pair(options.assets, options.config) as readers:
        parity = verify_pair(readers, frames)
        timing = measure_pair(readers, frames)
    result = dict(parity=parity, timing=timing, native_module=puyo_core.__file__)
    options.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(parity=parity, timing={key: {k: v for k, v in value.items()
                     if k != 'samples_ms'} for key, value in timing.items()})))


if __name__ == '__main__':
    main()
