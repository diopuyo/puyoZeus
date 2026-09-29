"""D3候補窓を原動画からCNN単独・HSV単独で読み取る。"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from scripts._d3_inventory import OUT, SOURCES

NICE = 19
VISIBLE_CELLS = 72
FPS = 30
OFFSETS = (-8, -4, -3, -2, -1, 0, 1)
PROGRESS = 500
SEEK_GAP_SEC = 2
VIDEO_ROOT = Path('/mnt/d/puyo_analyzer/videos/source')
ZENCHI = Path('/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/frames/video_zenchi_c0BQoMJwwQU.mp4')


def requests(sources: tuple[str, ...], fps: float) -> dict[int, set[int]]:
    """全合図と対応する書込み前後を、読取前に固定する。"""
    from scripts._d3_candidates import load
    points: dict[int, set[int]] = {}
    for source in sources:
        rows = load(source)
        data = json.loads((OUT/f'{source}_candidates.json').read_text())
        for side, group in enumerate(data):
            for event in group['events']:
                indices = {event['i']+delta for delta in OFFSETS}
                native = round(event['t']*fps)
                for delta in range(-8, 2):
                    points.setdefault(native+delta, set()).add(side)
                for key in ('cycle_start', 'cycle_end'):
                    if event.get(key) is not None:
                        points.setdefault(round(event[key]*fps), set()).add(side)
                for wi in event['write_indices']:
                    write = group['writes'][wi]
                    if write['additions']:
                        indices.update((write['i']-1, write['i']))
                for index in indices:
                    if 0 <= index < len(rows):
                        frame = round(rows[index]['t']*fps)
                        points.setdefault(frame, set()).add(side)
                # 式は元動画の1〜4フレーム前も必ず保存する。
                if event['type'] == 'formula':
                    frame = round(event['t']*fps)
                    for delta in range(1, 5):
                        points.setdefault(frame-delta, set()).add(side)
        if source in SOURCES[:3]:
            residual = json.loads((OUT.parent/'d1/classified.json').read_text())['cells']
            for cell in residual:
                if cell['source'] != source or not cell['category'].startswith('経路未確定'):
                    continue
                stamp = cell['last_change_sec']
                if stamp is not None:
                    for delta in (-1, 0, 1):
                        points.setdefault(max(0, round(stamp*fps)+delta), set()).add(cell['side'])
    return points


def readers() -> tuple[Any, Any, tuple]:
    """診断器を独立に作成し、本番パイプラインは呼び出さない。"""
    import cv2
    import torch
    from src.patch_classifier import CnnPatchClassifierLarge
    from src.image_reader import ColorClassifier, DEFAULT_P1_REGION, DEFAULT_P2_REGION
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    cnn = CnnPatchClassifierLarge.load(Path('models/cnn_phase_b_large_v2.pt'))
    cnn.to_device('cuda' if torch.cuda.is_available() else 'cpu')
    return cnn, ColorClassifier(), (DEFAULT_P1_REGION, DEFAULT_P2_REGION)


def observe(frame: Any, sides: set[int], cnn: Any, hsv: Any, regions: tuple) -> dict:
    """同一パッチを独立分類し、融合・推論・盤面補正を一切しない。"""
    patches, order = [], sorted(sides)
    for side in order:
        for row in range(1, 13):
            for col in range(6):
                x1, y1, x2, y2 = regions[side].cell_sample_rect(row, col)
                patches.append(frame[y1:y2, x1:x2])
    colors = cnn.classify_batch(patches)
    hsv_colors = [hsv.classify(patch) for patch in patches]
    return {str(side): dict(cnn=colors[n*VISIBLE_CELLS:(n+1)*VISIBLE_CELLS],
        hsv=hsv_colors[n*VISIBLE_CELLS:(n+1)*VISIBLE_CELLS]) for n, side in enumerate(order)}


def run(name: str, sources: tuple[str, ...], classifiers: tuple) -> None:
    """同じ元動画の観測は共有し、元記録の盤面は混ぜない。"""
    import cv2
    from scripts.enrich_e26_midchain import frame_at
    video = ZENCHI if name == 'zenchi' else VIDEO_ROOT/f'{name}_first_0_900_20260925_v1.mp4'
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    assert capture.isOpened(), video
    fps = capture.get(cv2.CAP_PROP_FPS)
    points = requests(sources, fps)
    done, start = {}, time.monotonic()
    dest = OUT/f'{name}_observations.jsonl'
    if dest.exists():
        with dest.open() as stream:
            for line in stream:
                row = json.loads(line)
                done.setdefault(row['frame'], set()).update(int(s) for s in row['sides'])
    with dest.open('a', buffering=1) as stream:
        for number, (index, sides) in enumerate(sorted(points.items())):
            missing = sides-done.get(index, set())
            if not missing:
                continue
            if index-round(capture.get(cv2.CAP_PROP_POS_FRAMES)) > SEEK_GAP_SEC*fps:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            frame = frame_at(capture, index)
            values = observe(frame, missing, *classifiers)
            stream.write(json.dumps(dict(frame=index, fps=fps, t=index/fps, sides=values), separators=(',', ':'))+'\n')
            if number % PROGRESS == 0:
                print(name, number, len(points), round(time.monotonic()-start, 1), flush=True)
    capture.release()
    (OUT/f'{name}_observations_complete.json').write_text(json.dumps(dict(video=str(video), fps=fps, frames=len(points))))


def main() -> None:
    """重い観測をnice 19・直列1プロセスで継続する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    options = parser.parse_args()
    if not options.worker:
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (OUT/'observe.log').open('a') as stream:
            child = subprocess.Popen([sys.executable, '-B', '-m', 'scripts._d3_observe', '--worker'],
                env=environment, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        print(child.pid)
        return
    os.nice(NICE)
    classifiers = readers()
    for source in (*SOURCES[:3], 'zenchi'):
        run(source, ('zenchi', 'review') if source == 'zenchi' else (source,), classifiers)


if __name__ == '__main__':
    main()
