"""mia idx7の観測系列と元映像を診断先へ抜き出す。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts._diag_mia7_inspect import BASE, OUT, START, END, guard, save
from src.exchange_event_record import read_records, encode

VIDEO = Path('/mnt/d/puyo_analyzer/videos/source/mia8KCjr52g_first_0_900_20260925_v1.mp4')
FRAMES = Path('/mnt/d/puyo_analyzer/local_frames/mia7_2026-10-02')
SHEET_COLS, SHEET_ROWS = 3, 3
THUMB_WIDTH, THUMB_HEIGHT = 640, 360
FRAME_STEP = 0.5


def extract_records() -> None:
    """STABLE盤面、入力変化、照合ログを抽出する。"""
    changes, boards, raw = [], [], []
    previous: Any = None
    board_previous = [None, None]
    for row in read_records(BASE / 'collect/records/mia8KCjr52g.jsonl.gz'):
        if row['kind'] != 'update' or not START <= row['args'][3] <= END:
            continue
        result, snap, final, stamp, game, totals, scores, visible = row['args']
        raw.append(encode(row))
        sides = []
        for idx, side in enumerate((result.p1, result.p2)):
            grid = side.confirmed_board._grid.tolist() if side.confirmed_board else None
            sides.append(dict(state=side.state.name, score=side.score, next=side.next_pair,
                              dnext=side.dnext_pair, motion=side.next_slide_motion,
                              chain=encode(vars(side.chain_event)) if side.chain_event else None))
            if side.state.name == 'STABLE' and grid != board_previous[idx]:
                boards.append(dict(t=stamp, side=idx+1, board=grid))
                board_previous[idx] = grid
        value = dict(sides=sides, snapshot=vars(snap), finalization=vars(final),
                     formula=totals, displayed=scores, visible=visible)
        if value != previous:
            changes.append(dict(t=stamp, **value))
            previous = value
    save('input_changes.json', changes)
    save('stable_boards.json', boards)
    save('raw_window.json', raw)
    path = BASE / 'collect/capture/mia8KCjr52g/placement_signal_reconcile.jsonl'
    save('placement_signal_reconcile.json', [r for line in path.open() if START <= (r := json.loads(line))['t_sec'] <= END])


def extract_frames() -> None:
    """動画位置を保存し、全画面の時系列シートを作る。"""
    FRAMES.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(VIDEO))
    assert capture.isOpened(), VIDEO
    fps = capture.get(cv2.CAP_PROP_FPS)
    manifest, thumbs = [], []
    for stamp in np.arange(START, END + FRAME_STEP, FRAME_STEP):
        guard()
        index = round(float(stamp)*fps)
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        assert ok, stamp
        filename = f'frame_{stamp:.3f}.jpg'
        cv2.imwrite(str(FRAMES / filename), frame)
        thumb = cv2.resize(frame, (THUMB_WIDTH, THUMB_HEIGHT))
        cv2.putText(thumb, f'{stamp:.3f}s', (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
        thumbs.append(thumb)
        manifest.append(dict(t=float(stamp), frame=index, path=filename))
    capture.release()
    per_sheet = SHEET_COLS*SHEET_ROWS
    for start in range(0, len(thumbs), per_sheet):
        sheet = np.zeros((THUMB_HEIGHT*SHEET_ROWS, THUMB_WIDTH*SHEET_COLS, 3), np.uint8)
        for offset, thumb in enumerate(thumbs[start:start+per_sheet]):
            y, x = divmod(offset, SHEET_COLS)
            sheet[y*THUMB_HEIGHT:(y+1)*THUMB_HEIGHT, x*THUMB_WIDTH:(x+1)*THUMB_WIDTH] = thumb
        cv2.imwrite(str(FRAMES / f'sheet_{start//per_sheet}.jpg'), sheet)
    save('video_manifest.json', dict(video=str(VIDEO), fps=fps, frames=manifest))


def main() -> None:
    """資源条件を記録して観測抽出する。"""
    save('resources.json', dict(start_available_bytes=guard(), nice=10))
    extract_records()
    extract_frames()
    save('resources_after_extract.json', dict(available_bytes=guard(), nice=10))


if __name__ == '__main__':
    main()
