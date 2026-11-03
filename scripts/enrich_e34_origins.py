"""元認識と一致した場合だけ、E32固定入力へ収集側W48b保持印を追記する。"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import re
import random
from typing import Any

import cv2
import torch
import numpy as np

from scripts.enrich_e16_records_20260928 import ZENCHI_VIDEO, FRAME_SIZE
from scripts.enrich_e26_midchain import frame_at
from scripts.run_e3_exchange_eval_20260926 import SOURCES, VIDEO_ROOT, SEED, save_json, digest
from src.exchange_event_record import read_records, encode, SIDE_FIELDS, CHAIN_FIELDS
from src.recognition_pipeline import RecognitionPipeline

OUT = Path('logs/e34/records')
PROGRESS_FRAMES = 3000


class ConfigurationCaptured(Exception):
    """描画開始前に既存認識設定だけを取り出す。"""


def pipeline(video: Path) -> RecognitionPipeline:
    """既存描画の認識設定を取得し、保存可否の印だけを有効にする。"""
    from scripts import visualize_advantage_overlay as viz
    original, acquire, config = RecognitionPipeline.load_default, viz._acquire_model, {}
    def capture(**kwargs: Any) -> None:
        config.update(kwargs)
        raise ConfigurationCaptured()
    RecognitionPipeline.load_default = capture
    viz._acquire_model = lambda *a, **k: None
    try:
        viz.generate(video=video, out=OUT/'unused.mp4', max_sec=0, sample_interval=0, render=False)
    except ConfigurationCaptured:
        pass
    finally:
        RecognitionPipeline.load_default = original
        viz._acquire_model = acquire
    config.update(enable_landing_chain_record_hold=True, enable_chain_active_record_hold=True)
    pipe = original(**config)
    match = re.search(r'(v\d+|video_\d+)', video.name)
    if match:
        pipe.set_video_id(match.group(1))
    return pipe


def comparable(side: Any) -> dict:
    """補完値以外の保存認識入力を全項目照合する。"""
    result = {key: encode(getattr(side, key)) for key in SIDE_FIELDS}
    event = side.chain_event
    result['chain_event'] = ({key: encode(getattr(event, key, None))
        for key in (*CHAIN_FIELDS, 'before_board')} if event is not None else None)
    return result


def attach(original: Any, replayed: Any, stamp: float) -> None:
    """時刻が同じだけの別盤面へ保持印を転用せず、一項目の差でも停止する。"""
    for index, (left, right) in enumerate(zip((original.p1, original.p2), (replayed.p1, replayed.p2))):
        before, after = comparable(left), comparable(right)
        differences = [key for key in before if before[key] != after[key]]
        if differences:
            raise ValueError(f'{stamp:.9f}秒 {index+1}P: 認識再生不一致 {differences}')
        left.prefire_origin_hold = right.landing_chain_started


def enrich(source: str) -> None:
    """全行の厳密一致と完了マーカーを確認するまで正式入力を公開しない。"""
    OUT.mkdir(parents=True, exist_ok=True)
    record = Path('logs/e31/records')/f'{source}.jsonl.gz'
    dest = OUT/record.name
    video = VIDEO_ROOT/f'{source}_first_0_900_20260925_v1.mp4' if source in SOURCES else ZENCHI_VIDEO
    pipe, cap = pipeline(video), cv2.VideoCapture(str(video))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), 0
    assert fps > 0
    status = dict(source=str(record), source_sha256=digest(record), video=str(video))
    try:
        with gzip.open(dest.with_suffix('.tmp'), 'wt', encoding='utf-8') as stream:
            for row in read_records(record):
                if row['kind'] == 'update':
                    result, stamp = row['args'][0], row['args'][3]
                    index = round(stamp*fps)
                    if frames == 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    frame = frame_at(cap, index)
                    if (frame.shape[1], frame.shape[0]) != FRAME_SIZE:
                        frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
                    attach(result, pipe.update(index, stamp, frame), stamp)
                    frames += 1
                    if frames % PROGRESS_FRAMES == 0:
                        print(source, frames, stamp, flush=True)
                stream.write(json.dumps(encode(row), ensure_ascii=False, separators=(',', ':'))+'\n')
        dest.with_suffix('.tmp').replace(dest)
        status.update(state='completed', frames=frames, sha256=digest(dest))
    except ValueError as error:
        status.update(state='blocked', frames_verified=frames, error=str(error))
        raise
    finally:
        cap.release()
        save_json(dest.with_suffix('.json'), status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=(*SOURCES, 'review', 'zenchi'), required=True)
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    enrich(args.source)
