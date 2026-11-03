"""E31段0: 発火前の全フレームと保存起点の差分を保存する。"""
from pathlib import Path
from typing import Any
import json
import cv2
import numpy as np
from src.exchange_event_record import read_records
from src.midchain_board_reader import MidchainBoardReader
from src.chain import ChainSimulator
from src.scoring import calculate_chain_score
from src.image_reader import ImageReader
from scripts.enrich_e26_midchain import frame_at
from scripts.enrich_e16_records_20260928 import ZENCHI_VIDEO

OUT = Path('logs/e31/scene')
TRIGGER = 2751.383333333333
WINDOW = 1.2
TIME_TOLERANCE = .01


def saved_origin() -> Any:
    """対象連鎖の保存起点を固定記録から取得する。"""
    for row in read_records(Path('logs/e26/records/review.jsonl.gz')):
        if row['kind'] != 'update':
            continue
        side, stamp = row['args'][0].p2, row['args'][3]
        if stamp > TRIGGER + TIME_TOLERANCE:
            break
        if side.chain_event and abs(side.chain_event.trigger_sec-TRIGGER) < TIME_TOLERANCE:
            return side.chain_event.before_board
    raise ValueError('対象の起点が存在しない')


def main() -> None:
    """未来フレームを使用せずCNN融合・HSV単独を同一画像で比較する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    origin = saved_origin()
    reader = MidchainBoardReader().reader
    cnn_only = ImageReader(classifier=reader._classifier._cnn)
    cnn_only._apply_inference = False
    region = reader._p2_region
    cap = cv2.VideoCapture(str(ZENCHI_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    simulator, rows = ChainSimulator(), []
    for index in range(round((TRIGGER-WINDOW)*fps), round(TRIGGER*fps)+1):
        frame = frame_at(cap, index)
        cnn = reader.read_board(frame, region, skip_tier1=True)
        hsv = reader.read_board_hsv_only(frame, region)
        name = f'{index}_{index/fps:.6f}.png'
        cv2.imwrite(str(OUT/name), frame[region.y:region.y+region.height, region.x:region.x+region.width])
        values = dict(frame=index, t_sec=index/fps, image=name)
        for label, board in (('cnn', cnn), ('hsv', hsv), ('cnn_only', cnn_only.read_board(frame, region))):
            chain = simulator.simulate(board)
            score = calculate_chain_score(chain)
            values[label] = dict(board=board._grid.tolist(), diff=[dict(row=int(r), col=int(c),
                saved=int(origin._grid[r,c]), observed=int(board._grid[r,c]))
                for r,c in np.argwhere(board._grid != origin._grid)],
                steps=[s.score for s in score.steps], total=score.total_score)
        rows.append(values)
    cap.release()
    (OUT/'frames.json').write_text(json.dumps(dict(origin=origin._grid.tolist(), frames=rows), indent=2))
    print([(r['t_sec'],len(r['cnn']['diff']),r['cnn']['total']) for r in rows], flush=True)


if __name__ == '__main__':
    main()
