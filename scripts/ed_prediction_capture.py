"""ed追加保存用。レビューと同じ画素・分類器・式読取りを原票化する。"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile, ZIP_STORED
import cv2

from scripts.exchange_counts_v33_collect import ImageSources, OFFSETS
from scripts.exchange_v35_capture import SequentialCapture
from src.image_reader import ColorClassifier, DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.score_ocr import ScoreOcr, FormulaStepAccumulator, SCORE_1P_REGION, SCORE_2P_REGION
from src.ui_mask import UiMaskMatcher
from src.phase_j.dshow_color import correct_601_as_709

FRAME_SIZE = (1920, 1080)
OCR_HZ = 30
PNG_COMPRESSION = 3
REGIONS = (DEFAULT_P1_REGION, DEFAULT_P2_REGION)


class LiveColorCapture(SequentialCapture):
    """liveのVideoFileSourceと同じ順序で拡大後に601→709を補正する。"""

    def read(self) -> tuple[bool, Any]:
        ok, frame = super().read()
        if ok:
            frame = correct_601_as_709(cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA))
        return ok, frame


class CropArchive:
    """同じフレーム・側の切出しを一度だけ可逆保存する。"""

    def __init__(self, path: Path) -> None:
        self.archive = ZipFile(path, 'w', compression=ZIP_STORED)
        self.seen: set[str] = set()

    def put(self, name: str, pixels: Any) -> str:
        if name not in self.seen:
            ok, encoded = cv2.imencode('.png', pixels, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
            if not ok:
                raise IOError('画像の可逆保存に失敗')
            self.archive.writestr(name, encoded.tobytes())
            self.seen.add(name)
        return name


class SavedImages(ImageSources):
    """既存ImageSourcesの分類結果を変えず、その入力画素を添付する。"""

    def __init__(self, video: Path, templates: Path, archive: CropArchive,
                 live_color: bool = True) -> None:
        capture_type = LiveColorCapture if live_color else SequentialCapture
        self.cap = capture_type(cv2.VideoCapture(str(video)))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            raise IOError(f'動画を開けません: {video}')
        self.classifiers = [ColorClassifier(), ColorClassifier(vote_mode=True)]
        self.ui = UiMaskMatcher.load_default(templates.parent)
        self.archive = archive

    def boards(self, trigger: float, stamp: float, side: int,
               offsets: tuple[float, ...] = OFFSETS) -> list[dict]:
        sources = super().boards(trigger, stamp, side, offsets)
        region = REGIONS[side]
        for source in sources:
            index = round(source['t'] * self.fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self.cap.read()
            if not ok:
                raise IOError(f'保存画素が欠測: {index}')
            frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
            crop = frame[region.y:region.y+region.height, region.x:region.x+region.width]
            source['crop'] = self.archive.put(f'board/{index}_{side}.png', crop)
        return sources


def capture_ocr(video: Path, start: float, end: float, templates: Path,
                archive: CropArchive, live_color: bool = True,
                region_offsets: dict[str, tuple[int, int]] | None = None) -> dict:
    """通常得点と式は同じROI。読取失敗も保存し、2コマ確認を再現する。"""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ocr = ScoreOcr.load_default(templates, enable_matmul_ncc=True, region_offsets=region_offsets)
    accumulators = [FormulaStepAccumulator(), FormulaStepAccumulator()]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start*fps))
    records, steps = [], []
    try:
        for index in range(int(start*fps), int(end*fps)+1):
            ok, frame = cap.read()
            if not ok:
                raise IOError(f'式画像が欠測: {index}')
            if index % max(1, round(fps/OCR_HZ)):
                continue
            frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
            if live_color:
                frame = correct_601_as_709(frame)
            scores, stamp = ocr.read(frame), index/fps
            for side, name in enumerate(('1P', '2P')):
                formula = ocr.read_formula_side(frame, name)
                score = (scores.score_1p, scores.score_2p)[side]
                step = accumulators[side].update(stamp, formula, score_displayed=score is not None)
                y1, y2, x1, x2 = (SCORE_1P_REGION, SCORE_2P_REGION)[side]
                dx, dy = (region_offsets or {}).get(name, (0, 0))
                y1, y2, x1, x2 = y1+dy, y2+dy, x1+dx, x2+dx
                crop = archive.put(f'score/{index}_{side}.png', frame[y1:y2, x1:x2])
                records.append(dict(t=stamp, side=side, score=score, crop=crop, **asdict(formula)))
                if step:
                    steps.append(dict(side=side, **asdict(step)))
    finally:
        cap.release()
    return dict(records=records, steps=steps, start=start, end=end, fps=fps)
