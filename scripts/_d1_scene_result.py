"""対象4セルの元記録と計装再生を照合し、再現誤差も保存する。"""
from __future__ import annotations
import json
from collections import Counter
from typing import Any
import cv2
from scripts._d1_inventory import OUT, read
from scripts.enrich_e26_midchain import frame_at
from scripts.enrich_e16_records_20260928 import ZENCHI_VIDEO
from src.image_reader import DEFAULT_P2_REGION

CELLS = ((1,3),(1,4),(1,5),(2,5))
LOW, HIGH, FPS = 2746.,2751.35,60
IMAGES = (2749.75,2749.85,2750.016667,2750.083333,2750.55)


def main() -> None:
    """元記録への完全一致と4セル一致を別々に数える。"""
    originals = {round(row['args'][3]*FPS):row['args'][0]['p2'] for row in read('review') if LOW<=row['args'][3]<=HIGH}
    trace = [json.loads(line) for line in (OUT/'scene_trace.jsonl').read_text().splitlines()]
    sites = json.loads((OUT/'scene_sites.json').read_text())
    rows, counts = [], Counter()
    for row in trace:
        if row['side']!='2P' or row['frame'] not in originals:
            continue
        original = originals[row['frame']]
        grid = original['confirmed_board']
        same = grid == row['board']
        target_same = row['board'] is not None and all(grid[r][c]==row['board'][r][c] for r,c in CELLS)
        counts.update(total=1,full_equal=int(same),four_equal=int(target_same))
        details = []
        for r,c in CELLS:
            writer = row['writers'][r*6+c] if row['writers'] else 0
            details.append(dict(row=r,col=c,original=grid[r][c],replay=row['board'][r][c] if row['board'] else None,
                cnn=row['cnn'][r][c],hsv=row['hsv'][r][c],path=sites[writer],
                written_sec=row['written_frames'][r*6+c]/FPS if row['written_frames'] else None))
        rows.append(dict(t=row['t'],state_original=original['state']['state'],state_replay=row['state'],full_equal=same,four_equal=target_same,cells=details))
    (OUT/'scene_comparison.json').write_text(json.dumps(dict(counts=counts,rows=rows),ensure_ascii=False,indent=2))
    cap = cv2.VideoCapture(str(ZENCHI_VIDEO))
    reg = DEFAULT_P2_REGION
    for stamp in IMAGES:
        frame = frame_at(cap,round(stamp*FPS))
        cv2.imwrite(str(OUT/f'scene_{stamp:.6f}.png'),frame[reg.y:reg.y+reg.height,reg.x:reg.x+reg.width])
    cap.release()
    print(dict(counts))


if __name__ == '__main__':
    main()
