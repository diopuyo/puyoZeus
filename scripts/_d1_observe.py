"""差分がある発火と保存盤面書換え前後だけを画像で再確認する。"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import cv2
import torch
from src.midchain_board_reader import MidchainBoardReader
from scripts._d1_inventory import OUT, SOURCES
from scripts.enrich_e26_midchain import frame_at

FPS, OFFSET, SEEK_GAP = 30, 1/30, 60
VIDEO_ROOT = Path('/mnt/d/puyo_analyzer/videos/source')


def requests(source: str, rows: list[dict]) -> dict[tuple[int, int], set[str]]:
    """セルごとの最後の書換えと発火直前を重複なく選ぶ。"""
    points: dict[tuple[int, int], set[str]] = {}
    for row in rows:
        if row['source'] != source or not row['cells']:
            continue
        times = {row['trigger']-OFFSET, row['trigger']-.5, row['trigger']-1.2}
        for cell in row['cells']:
            if not cell['final_pair'] and cell['last_change_sec'] is not None:
                times.update(cell['last_change_sec']+delta for delta in (-OFFSET, 0, .1))
        for stamp in times:
            points.setdefault((round(stamp*FPS),row['side']),set()).add(str(row['trigger']))
    return points


def run(source: str, reader: Any, rows: list[dict]) -> None:
    """実動画のCNN融合とHSV単独を保存する。"""
    points = requests(source, rows)
    cap = cv2.VideoCapture(str(VIDEO_ROOT/f'{source}_first_0_900_20260925_v1.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps == FPS
    with (OUT/f'{source}_observations.jsonl').open('w') as stream:
        for count, ((index,side), fires) in enumerate(sorted(points.items())):
            if index-round(cap.get(cv2.CAP_PROP_POS_FRAMES)) > SEEK_GAP:
                cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            frame = frame_at(cap,index)
            region = (reader._p1_region,reader._p2_region)[side]
            cnn = reader.read_board(frame,region,skip_tier1=True)._grid.tolist()
            hsv = reader.read_board_hsv_only(frame,region)._grid.tolist()
            stream.write(json.dumps(dict(frame=index,t=index/fps,side=side,fires=sorted(fires),cnn=cnn,hsv=hsv))+'\n')
            if any(abs(index/fps-float(f)+OFFSET) < .01 for f in fires):
                cv2.imwrite(str(OUT/f'{source}_{index}_p{side+1}.png'),frame[region.y:region.y+region.height,region.x:region.x+region.width])
            if count % FPS == 0:
                print(source,count,len(points),flush=True)
    cap.release()


def main() -> None:
    """既存窓を優先し、起点の変化時刻だけ画像を補う。"""
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    rows = json.loads((OUT/'differences.json').read_text())
    reader = MidchainBoardReader().reader
    for source in SOURCES[:-1]:
        run(source,reader,rows)


if __name__ == '__main__':
    main()
