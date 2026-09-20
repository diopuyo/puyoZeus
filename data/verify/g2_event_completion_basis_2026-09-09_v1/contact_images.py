"""無加工原PNGの盤面上部/NEXTを並べる目視用補助。認識入力に使わない。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
COLUMNS, LABEL_HEIGHT = 4, 24
CASES = {'normal_motion_v1': (170, 60, 568, 340), 'chain_exit_motion_v1': (714, 60, 1115, 340)}


def render(name: str, rectangle: tuple[int, int, int, int], *, source_name: str | None = None,
           selected: tuple[int, ...] | None = None) -> None:
    source = ROOT / (source_name or name)
    frames = json.loads((source / 'EXTRACTION.json').read_text())['frames']
    if selected is not None:
        frames = [frame for frame in frames if frame['frame'] in selected]
    width, height = rectangle[2] - rectangle[0], rectangle[3] - rectangle[1]
    rows = (len(frames) + COLUMNS - 1) // COLUMNS
    sheet = Image.new('RGB', (width * COLUMNS, (height + LABEL_HEIGHT) * rows), '#222222')
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        x, y = index % COLUMNS * width, index // COLUMNS * (height + LABEL_HEIGHT)
        with Image.open(source / frame['file']) as image:
            sheet.paste(image.crop(rectangle), (x, y + LABEL_HEIGHT))
        draw.text((x + 4, y + 4), f"frame {frame['frame']} / {frame['time_sec']:.4f}s", fill='white')
    output = ROOT / (name + '_contact.png')
    with output.open('xb') as stream:
        sheet.save(stream, format='PNG')
    print({'output': str(output), 'crops_only_no_resize': True, 'frames': len(frames)})


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == 'negative':
        render('false_slide_2p_v1', (714, 60, 1115, 340), source_name='normal_motion_v1',
               selected=(34910, 34914, 34918, 34922, 34926))
    elif len(sys.argv) == 2 and sys.argv[1] == 'samepair':
        render('same_pair_motion_v1', (170, 60, 568, 340))
    elif len(sys.argv) == 2 and sys.argv[1] == 'endpair':
        render('same_pair_end_33794_v1', (170, 60, 568, 340))
    else:
        for name, rectangle in CASES.items():
            render(name, rectangle)
