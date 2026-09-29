"""E22固定入力を保存したまま、原映像の重力待ち盤面だけを追記する。"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import cv2
import numpy as np

from src.exchange_event_record import read_records, encode
from src.midchain_board_reader import MidchainBoardReader, MODEL
from src.board_state_machine import BoardState
from scripts.run_e3_exchange_eval_20260926 import SOURCES, VIDEO_ROOT, save_json, digest
from scripts.enrich_e16_records_20260928 import ZENCHI_VIDEO

OUT = Path('logs/e26/records')
FRAME_SIZE = (1920, 1080)
PROGRESS_FRAMES = 3000


def frame_at(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    """正規化前の原動画のフレーム番号を厳密に合わせる。"""
    position = round(capture.get(cv2.CAP_PROP_POS_FRAMES))
    if position > index or index-position > PROGRESS_FRAMES:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    while round(capture.get(cv2.CAP_PROP_POS_FRAMES)) < index:
        assert capture.grab()
    success, frame = capture.read()
    assert success and round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == index+1
    if (frame.shape[1], frame.shape[0]) != FRAME_SIZE:
        frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    return frame


def enrich(source: str) -> None:
    """元のstate・得点・確定盤面を変更せず、追加入力の出所を保存する。"""
    dest, manifest = OUT/f'{source}.jsonl.gz', OUT/f'{source}.json'
    if dest.exists() and manifest.exists():
        return
    video = VIDEO_ROOT/f'{source}_first_0_900_20260925_v1.mp4' if source in SOURCES else ZENCHI_VIDEO
    record = Path('logs/e16/records')/f'{source}.jsonl.gz'
    reader, capture = MidchainBoardReader(), cv2.VideoCapture(str(video))
    fps, frames, recognized = capture.get(cv2.CAP_PROP_FPS), 0, 0
    assert fps > 0
    with gzip.open(dest.with_suffix('.tmp'), 'wt', encoding='utf-8') as stream:
        for row in read_records(record):
            if row['kind'] == 'update':
                result, stamp = row['args'][0], row['args'][3]
                sides = (result.p1, result.p2)
                boards = (None, None)
                if any(s.state == BoardState.GRAVITY_SETTLE for s in sides):
                    boards = reader.read(frame_at(capture, round(stamp*fps)), sides)
                for side, board in zip(sides, boards):
                    side.midchain_board = board
                    recognized += board is not None
                frames += 1
                if frames % PROGRESS_FRAMES == 0:
                    print(source, frames, recognized, flush=True)
            stream.write(json.dumps(encode(row), ensure_ascii=False, separators=(',', ':'))+'\n')
    capture.release()
    dest.with_suffix('.tmp').replace(dest)
    save_json(manifest, dict(source=str(record), source_sha256=digest(record), video=str(video),
        fps=fps, model=str(MODEL), model_sha256=digest(MODEL), frames=frames,
        recognized=recognized, resize_interpolation='INTER_AREA', sha256=digest(dest)))


def main() -> None:
    """GPU読取は一プロセスで順次行い、再生二プロセスとの合計を三以内に保つ。"""
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    for source in ('review', *SOURCES, 'zenchi'):
        enrich(source)


if __name__ == '__main__':
    main()
