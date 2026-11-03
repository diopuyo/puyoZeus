"""v3.2の保存入力アダプター。画像デコードとlive用推定関数を分離する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


from pathlib import Path
from typing import Any
import math
import cv2
import numpy as np
from scripts.exchange_review_io import jsonl, fires
from src.exchange_origin_evidence import MAX_FRAMES, WINDOW_SEC

BASE = Path(_public_dev_text('<work>/exch_review_render'))
OUT = BASE/'video_80_g7_v3_2'
POLICY = Path('docs/verification/exch_review_v32/policy.json')
VIDEOS = (29, 32, 34, 39, 51, 61, 65, 79, 92, 94)
RUNS = Path('/mnt/d/puyo_analyzer/share')
RAW_VIDEO_REQUIRED = (32, 34, 51, 61, 80, 92)


def source(video: int) -> Path:
    return BASE/'video_80_g7_v3'/('input' if video == 80 else f'fit_input/video_{video}')


def fire_rows(path: Path, game: int | None = None) -> list[dict]:
    return [r for r in fires(list(jsonl(path/'exchange_projection_shadow.jsonl')))
            if r['game_idx'] > 0 and r.get('exchange_input') and
            (game is None or r['game_idx'] == game)]


def video_path(video: int) -> Path | None:
    for folder in ('tasks/done/ei_ed_parallel_after_bo5/artifacts', 'exchange_runs',
                   'tasks/done/ed_exchange_shadow_12videos_recompute/artifacts',
                   'tasks/done/eh_ed_resume_from_3rd/artifacts'):
        path = RUNS/folder/f'ed_video_{video}/video.mp4'
        if path.exists():
            return path
    return None


class VideoWindow:
    """採取用seek。推定関数には通知時刻以前の最大6コマだけを渡す。"""

    def __init__(self, video: int) -> None:
        self.path = video_path(video)
        if self.path is None and video in RAW_VIDEO_REQUIRED:
            raise FileNotFoundError(f'局所画像採取用の動画{video}が見えません。共有ドライブを確認してください')
        self.cap = cv2.VideoCapture(str(self.path)) if self.path else None
        if self.cap is not None and not self.cap.isOpened():
            raise RuntimeError(f'動画を開けません: {self.path}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else None

    def frames(self, trigger: float, stamp: float) -> list[tuple]:
        if self.cap is None:
            return []
        first = max(0, math.ceil((trigger-WINDOW_SEC)*self.fps))
        last = math.floor(stamp*self.fps)
        desired = set(np.linspace(first, last, min(MAX_FRAMES, last-first+1), dtype=int))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        frames = []
        for index in range(first, last+1):
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError('起点照合用動画が途中で終了しました')
            if index in desired:
                frames.append((index/self.fps, frame))
        return frames

    def close(self) -> None:
        if self.cap:
            self.cap.release()


def fired_pair(seeds: Any, fire: dict, origin: Any) -> tuple | None:
    seed = seeds.at(fire['t_sec'], fire['game_idx'])
    side = fire['exchange_input']['attacker']
    index = seed['next_indices'][side]-1 if seed.get('next_indices') else None
    pair = tuple(seed['tape'][index]) if index is not None and index >= 0 else None
    active = seeds.tape.active_pair(side)
    aligned = pair is not None and active is not None and sorted(pair) == sorted(active)
    origin.pair_metadata = dict(index=index, pair=pair, identified=aligned,
        seed=seed.get('seed'), observed_sec=seed.get('input_sec'))
    return pair if aligned else None
