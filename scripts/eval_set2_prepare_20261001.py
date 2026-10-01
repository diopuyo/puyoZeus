"""セット2の既存勝者観測とWINパネルの照合用画像を作る。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

DATA = Path('/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data')
OUT = Path('logs/eval_set/set2')
VIDEO = DATA/'frames/video_zenchi_c0BQoMJwwQU.mp4'
SOURCE = DATA/'verify/zenchi_two_sets_review_source_2026-08-31'
MS_PER_SEC = 1000
TILE_WIDTH, TILE_HEIGHT = 480, 125
COLS, PAGE_ROWS = 3, 10


def observations() -> list[dict]:
    """既存観測を候補として保存する。目視確認前は公式扱いしない。"""
    rows = []
    for path in SOURCE.glob('runs/**/attempt=zenchi-review-set2-20260831/events/*.jsonl'):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event.get('event_type') != 'winner_observed':
                continue
            timing = event['timing']
            rows.append(dict(game=event['payload']['game_index_unverified'],
                             winner=event['payload']['winner_side'],
                             start=timing['occurred_earliest_ms']/MS_PER_SEC,
                             end=timing['occurred_latest_ms']/MS_PER_SEC))
    return sorted(rows, key=lambda row: row['start'])


def panels(times: list[tuple[str, float]], prefix: str) -> None:
    """1920×1080の上部WINパネルを時刻入り一覧にする。"""
    cap = cv2.VideoCapture(str(VIDEO))
    for page in range(0, len(times), COLS*PAGE_ROWS):
        canvas = Image.new('RGB', (COLS*TILE_WIDTH, PAGE_ROWS*TILE_HEIGHT), 'white')
        draw = ImageDraw.Draw(canvas)
        for index, (label, sec) in enumerate(times[page:page+COLS*PAGE_ROWS]):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec*MS_PER_SEC)
            ok, frame = cap.read()
            assert ok, sec
            frame = cv2.resize(frame, (1920, 1080))
            tile = Image.fromarray(cv2.cvtColor(frame[948:1028, 765:1155], cv2.COLOR_BGR2RGB))
            tile = tile.resize((TILE_WIDTH, 95))
            x, y = index % COLS*TILE_WIDTH, index // COLS*TILE_HEIGHT
            canvas.paste(tile, (x, y+25))
            draw.text((x+5, y+5), f'{label} t={sec:.3f}', fill='black')
        canvas.save(OUT/f'{prefix}_{page//(COLS*PAGE_ROWS)}.jpg')
    cap.release()


def main() -> None:
    """候補表・全試合の境界直後・セット間と末尾の画像を出す。"""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = observations()
    (OUT/'candidates.json').write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows), flush=True)
    panels([(f"g{row['game']} start", row['start']+5) for row in rows]
           + [('last end', rows[-1]['end']-1)], 'panels')
    panels([(f'gap {sec}', sec) for sec in range(3415, 3676, 10)]
           + [('tail', 7030)], 'gap')
    panels([(f'tail {sec}', sec) for sec in range(7005, 7030)], 'tail')


if __name__ == '__main__':
    main()
