"""追加保存専用。動画序盤の式だけで左右の得点枠を一度較正する。"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from src.score_ocr import ScoreOcr, _crop_score_roi, FORMULA_MULT_NCC_MIN, parse_formula_cells
from src.phase_j.dshow_color import correct_601_as_709

SEARCH_RADIUS = 6
SAMPLE_HZ = 5
EARLY_SECONDS = 60
FORMULA_SAMPLES = 12
MIN_FORMULA_SAMPLES = 3
TIE_ATOL = 1e-6  # OpenCVの浮動小数点丸めだけを同点扱いする。
FRAME_SIZE = (1920, 1080)
SIDES = ('1P', '2P')
CANDIDATES = tuple((dx, dy) for dy in range(-SEARCH_RADIUS, SEARCH_RADIUS+1)
                   for dx in range(-SEARCH_RADIUS, SEARCH_RADIUS+1))


def candidate_scores(frame: np.ndarray, side: str, ocr: ScoreOcr) -> list[float]:
    """本番と同じ×照合を全候補に適用する。合格線は変更しない。"""
    return [ocr._match_mult_cell(_crop_score_roi(frame, side, offset), side)
            for offset in CANDIDATES]


def select_offset(scores: list[float]) -> tuple[int, int]:
    """最大一致の同点では移動量最小、横移動最小を優先する。"""
    best = max(scores)
    tied = [offset for offset, score in zip(CANDIDATES, scores)
            if best-score <= TIE_ATOL]
    return min(tied, key=lambda p: (abs(p[0])+abs(p[1]), abs(p[0]), abs(p[1]), p))


def summarize(samples: list[dict]) -> dict:
    """探索原票と各候補の平均を残し、証拠不足では移動しない。"""
    means = (np.mean([s['scores'] for s in samples], axis=0).tolist()
             if samples else [0.] * len(CANDIDATES))
    offset = select_offset(means) if len(samples) >= MIN_FORMULA_SAMPLES else (0, 0)
    return dict(offset=list(offset), formula_frames=len(samples), samples=samples,
                candidates=[dict(dx=x, dy=y, mean_ncc=value)
                            for (x, y), value in zip(CANDIDATES, means)])


def calibrate(video: Path, templates: Path, start: float, end: float,
              live_color: bool = True) -> dict:
    """最初の試合の先頭60秒・各側最初の12式コマで固定する。"""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise IOError(f'動画を開けません: {video}')
    ocr = ScoreOcr.load_default(templates, enable_matmul_ncc=True)
    samples: dict[str, list[dict]] = {side: [] for side in SIDES}
    scanned = {side: 0 for side in SIDES}
    stop = min(end, start+EARLY_SECONDS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start*fps))
    try:
        for index in range(int(start*fps), int(stop*fps)+1):
            ok, frame = cap.read()
            if not ok:
                raise IOError(f'較正画像が欠測: {index}')
            if index % max(1, round(fps/SAMPLE_HZ)):
                continue
            frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
            if live_color:
                frame = correct_601_as_709(frame)
            for side in SIDES:
                if len(samples[side]) < FORMULA_SAMPLES:
                    scanned[side] += 1
                    add_sample(frame, side, ocr, index, samples[side])
            if all(len(s) == FORMULA_SAMPLES for s in samples.values()):
                break
    finally:
        cap.release()
    return dict(start=start, search_end=stop, radius=SEARCH_RADIUS, sample_hz=SAMPLE_HZ,
                scanned_frames=scanned,
                mult_threshold=FORMULA_MULT_NCC_MIN,
                sides={side: summarize(rows) for side, rows in samples.items()})


def add_sample(frame: np.ndarray, side: str, ocr: ScoreOcr, index: int,
               samples: list[dict]) -> None:
    """候補内で有効な式を読めたコマだけを較正に使う。"""
    scores = candidate_scores(frame, side, ocr)
    if max(scores) < FORMULA_MULT_NCC_MIN:
        return
    dx, dy = select_offset(scores)
    _, _, labels, confs = ocr.read_side_detail_at_offset(frame, side, dx, dy)
    if parse_formula_cells(labels, confs, max(scores)).valid:
        samples.append(dict(frame=index, scores=scores))


def offsets(receipt: dict) -> dict[str, tuple[int, int]]:
    """受領票の固定座標を既存OCRの引数へ変換する。"""
    return {side: tuple(data['offset']) for side, data in receipt['sides'].items()}
