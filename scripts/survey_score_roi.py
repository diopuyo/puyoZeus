"""既知10動画は保存OCRだけを読み、追加区間だけ式枠を探索する。"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.score_ocr import ScoreOcr
from src.score_roi_alignment import ScoreRoiAlignment, SIDES, SAMPLE_HZ
from src.phase_j.dshow_color import correct_601_as_709

KNOWN_VIDEOS = (29, 32, 34, 39, 51, 61, 65, 79, 92, 94)
DURATION_SEC = 180
FRAME_SIZE = (1920, 1080)


def read_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def saved_summary(folder: Path) -> dict:
    """先頭試合から3分以内だけを集計。未探索のずれは推測しない。"""
    games = sorted(folder.glob('game_*'), key=lambda p: int(p.name.split('_')[-1]))
    start = read_json(games[0]/'complete.json')['start']
    rows = []
    for game in games:
        if read_json(game/'complete.json')['start'] > start+DURATION_SEC:
            break
        rows.extend(r for r in read_json(game/'ocr.json.gz')['records']
                    if start <= r['t'] < start+DURATION_SEC)
    result = dict(start=start, end=start+DURATION_SEC, sides={})
    for side, label in enumerate(SIDES):
        values = [r for r in rows if r['side'] == side]
        result['sides'][label] = dict(n=len(values), valid=sum(r['valid'] for r in values),
                                    max_ncc=max(r['mult_ncc'] for r in values))
    alignment = folder/'score_alignment.json'
    if alignment.exists():
        result['offsets'] = {s: r['offset'] for s, r in read_json(alignment)['sides'].items()}
    return result


def saved_survey(root: Path, index: Path) -> list[dict]:
    with index.open(encoding='utf-8') as stream:
        metadata = {int(r['video_idx']): r for r in csv.DictReader(stream, delimiter='\t')}
    results = []
    for video in KNOWN_VIDEOS:
        folder = next(root.glob(f'ed_inputs_rebuild_20261015/lane_*/video_{video}/saved'))
        data = dict(video=video, metadata=metadata[video], fixed=saved_summary(folder))
        info = read_json(folder.parent/'download/video.info.json')
        data['channel'] = info.get('channel')
        corrected = root/f'ed_inputs_rebuild_20261015_v{video}fix/saved'
        if corrected.exists():
            data['aligned'] = saved_summary(corrected)
        results.append(data)
    return results


def scan(video: Path, templates: Path) -> dict:
    """追加区間を5Hzで測り、固定枠最大NCCと初期12コマの最適枠を返す。"""
    ocr = ScoreOcr.load_default(templates, enable_matmul_ncc=True)
    alignment = ScoreRoiAlignment(ocr)
    fixed = ScoreOcr.load_default(templates, enable_matmul_ncc=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise OSError(f'動画を開けません: {video}')
    maxima, aligned_max = {s: 0. for s in SIDES}, {s: 0. for s in SIDES}
    valid, scanned = {s: 0 for s in SIDES}, 0
    try:
        for index in range(round(fps*DURATION_SEC)):
            ok, image = cap.read()
            if not ok:
                break
            if index % max(1, round(fps/SAMPLE_HZ)):
                continue
            image = correct_601_as_709(cv2.resize(image, FRAME_SIZE, interpolation=cv2.INTER_AREA))
            alignment.observe(image, index/fps)
            scanned += 1
            for side in SIDES:
                raw = fixed.read_formula_side(image, side)
                maxima[side] = max(maxima[side], raw.mult_ncc)
                raw = ocr.read_formula_side(image, side)
                aligned_max[side] = max(aligned_max[side], raw.mult_ncc)
                valid[side] += int(raw.valid)
    finally:
        cap.release()
    return dict(scanned=scanned, fixed_max=maxima, aligned_max=aligned_max,
                aligned_valid=valid, status=alignment.status())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-root', type=Path)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--templates', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = (saved_survey(args.saved_root, Path('data/phase_e_dl_index.tsv')) if args.saved_root
              else scan(args.video, args.templates))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
