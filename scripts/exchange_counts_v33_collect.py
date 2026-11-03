"""他10動画の設定用入力採取。画像とCNNは発火通知以前に限定する。"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from scripts.audit_exchange_counts_v33 import OUT, TEMPLATES
from scripts.exchange_review_v32_data import BASE, VIDEOS, source, video_path
from scripts.exchange_review_io import save
from scripts.replay_exchange_projection_shadow import updates
from src.image_reader import ColorClassifier, DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.ui_mask import UiMaskMatcher, UI_MASK_TARGET_CELLS

OFFSETS, FPS, COLS, CELLS = (-.3, -.15, 0.), 60, 6, 78


class ImageSources:
    def __init__(self, path: Path | None) -> None:
        self.cap = cv2.VideoCapture(str(path)) if path else None
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else FPS
        self.classifiers = [ColorClassifier(), ColorClassifier(vote_mode=True)]
        self.ui = UiMaskMatcher.load_default(TEMPLATES.parent)

    def boards(self, trigger: float, stamp: float, side: int,
               offsets: tuple[float, ...] = OFFSETS) -> list[dict]:
        if self.cap is None:
            return []
        sources = []
        region = (DEFAULT_P1_REGION, DEFAULT_P2_REGION)[side]
        for offset in offsets:
            index = int((trigger+offset)*self.fps)
            time = index/self.fps
            assert time <= stamp
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError(f'画像入力不足: t={time}, 総コマ={self.cap.get(cv2.CAP_PROP_FRAME_COUNT)}')
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            boards = [[0]*COLS for _ in self.classifiers]
            began = perf_counter()
            for cell in range(COLS, CELLS):
                row, col = divmod(cell, COLS)
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                patch = frame[y1:y2, x1:x2]
                masked = (row, col) in UI_MASK_TARGET_CELLS and self.ui.is_ui(patch)
                for classifier, board in zip(self.classifiers, boards):
                    board.append(0 if masked else int(classifier.classify(patch)))
            for board in boards:
                sources.append(dict(kind='image', t=time, board=board,
                                    image_compute_ms=(perf_counter()-began)*1000/len(boards)))
        return sources


def extract_sources(raw: object, stamp: float, trigger: float, side: int,
                    images: ImageSources) -> list[dict]:
    snapshot = getattr(raw, 'prefire_snapshot', None) or {}
    sources = []
    if snapshot.get('board') is not None:
        assert snapshot['end_sec'] <= stamp
        sources.append(dict(kind='cnn', t=snapshot['end_sec'],
                            board=np.asarray(snapshot['board']).ravel().tolist()))
    sources.extend(images.boards(trigger, stamp, side))
    return sources


def collect(video: int) -> None:
    assert video in VIDEOS and video != 80
    rows = json.loads((BASE/f'video_80_g7_v3_2/calibration/train_{video}.json').read_text())
    targets = {(r['game'], r['input_sec']): r for r in rows}
    images = ImageSources(video_path(video))
    for args in updates(source(video)/'inputs.jsonl.gz'):
        row = targets.get((args[4], args[3]))
        if row is None:
            continue
        raw = (args[0].p1, args[0].p2)[row['side']]
        event = raw.chain_event
        row['sources'] = extract_sources(raw, args[3], event.trigger_sec, row['side'], images)
        row['observed'] = dict(count=event.chain_count, score=event.total_score, t=args[3])
        row['colors'] = sorted({c for v in row['candidates'] for c in v['origin'] if 1 <= c <= 5})
    if images.cap:
        images.cap.release()
    save(OUT/f'calibration/input_{video}.json', rows)
    print('採取完了', video, len(rows), flush=True)


def main() -> None:
    for video in VIDEOS:
        if not (OUT/f'calibration/input_{video}.json').exists():
            collect(video)


if __name__ == '__main__':
    import sys
    if '--metadata' in sys.argv:
        for video in (*VIDEOS, 80):
            path = video_path(video)
            cap = cv2.VideoCapture(str(path)) if path else None
            print(video, cap.get(cv2.CAP_PROP_FPS) if cap else None)
    else:
        main()
