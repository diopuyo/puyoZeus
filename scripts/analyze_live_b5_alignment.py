"""STABLE盤面列を内容の完全一致で整列し、時刻差と残る色差を分離する。"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.collect_live_b5_recognition import OUTPUT
from scripts.verify_live_b4_degradation import compare, START, FPS, DEFAULT_VIDEO, transcode
from scripts.run_live_pipeline_20260928 import save_json

MAX_PAIR_SHIFT_SEC = 2.0
REPRESENTATIVES = 5
CELL_SCALE = 6
LABEL_HEIGHT = 48


def load(name: str) -> dict:
    with np.load(OUTPUT / f'{name}.npz') as data:
        return {key: data[key] for key in data.files}


def events(data: dict, side: int) -> tuple[list[dict], dict[int, int]]:
    """同じ確定盤面の繰り返しを一件へ圧縮する。試合内外を跨がせない。"""
    output, mapping, epoch, was_active = [], {}, 0, None
    for frame, available in enumerate(data['stable'][:, side]):
        active = bool(data['active'][frame])
        if was_active is not None and active != was_active:
            epoch += 1
        was_active = active
        if not available:
            continue
        board = data['boards'][frame, side]
        if not output or output[-1]['epoch'] != epoch or not np.array_equal(output[-1]['board'], board):
            output.append(dict(board=board, frames=[], epoch=epoch))
        output[-1]['frames'].append(frame)
        mapping[frame] = len(output)-1
    return output, mapping


def exact_pairs(left: list[dict], right: list[dict], allow_color_error: bool = False) -> list[tuple[int, int]]:
    """時間近傍内の完全一致だけをアンカーにした最長共通部分列。"""
    size = (len(left)+1, len(right)+1)
    score = np.zeros(size, dtype=int)
    equal = np.zeros((len(left), len(right)), dtype=bool)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            shift = abs(np.median(a['frames'])-np.median(b['frames']))/FPS
            av, bv = a['board'][1:], b['board'][1:]
            matches = np.array_equal(av, bv)
            if allow_color_error:
                matches |= np.array_equal(av != 0, bv != 0) and int((av != bv).sum()) == 1
            equal[i, j] = (a['epoch'] == b['epoch'] and shift <= MAX_PAIR_SHIFT_SEC
                           and matches)
            score[i+1, j+1] = score[i, j]+1 if equal[i, j] else max(score[i, j+1], score[i+1, j])
    pairs, i, j = [], len(left), len(right)
    while i and j:
        if equal[i-1, j-1]:
            pairs.append((i-1, j-1))
            i, j = i-1, j-1
        elif score[i-1, j] >= score[i, j-1]:
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def pair_events(left: list[dict], right: list[dict]) -> list[tuple[int, int, str]]:
    """一致アンカー間の一対一候補だけ、全セル占有位置一致で追加対応する。"""
    anchors = exact_pairs(left, right)
    output = [(i, j, 'visible_exact') for i, j in anchors]
    bounds = [(-1, -1), *anchors, (len(left), len(right))]
    for (before_a, before_b), (after_a, after_b) in zip(bounds, bounds[1:]):
        indices_a, indices_b = list(range(before_a+1, after_a)), list(range(before_b+1, after_b))
        candidates = exact_pairs([left[i] for i in indices_a], [right[j] for j in indices_b], True)
        for ii, jj in candidates:
            i, j = indices_a[ii], indices_b[jj]
            a, b = left[i], right[j]
            shift = abs(np.median(a['frames'])-np.median(b['frames']))/FPS
            if (a['epoch'] == b['epoch'] and shift <= MAX_PAIR_SHIFT_SEC
                    and np.array_equal(a['board'][1:] != 0, b['board'][1:] != 0)):
                output.append((i, j, 'same_visible_occupancy_one_color_between_anchors'))
    return sorted(output)


def align(left: dict, right: dict) -> dict:
    """元の同時刻差分を排他的に分類し、手番相当の重複除去集計も出す。"""
    original_diff = ((left['boards'] != right['boards'])
                     & (left['stable'] & right['stable'])[:, :, None, None])
    counts = Counter(timing_cells=0, persistent_color_cells=0, hidden_inference_cells=0, unresolved_cells=0)
    details, residuals, matched_cells, aligned_diffs, event_counts = [], [], 0, 0, []
    for side in range(2):
        a, frame_map = events(left, side)
        b, _ = events(right, side)
        event_counts.append(dict(side=side+1, original=len(a), converted=len(b)))
        paired = {i: (j, kind) for i, j, kind in pair_events(a, b)}
        for frame in np.flatnonzero(original_diff[:, side].any(axis=(1, 2))):
            mask = original_diff[frame, side]
            match = paired.get(frame_map[frame])
            if match is None:
                counts['unresolved_cells'] += int(mask.sum())
                continue
            aligned = left['boards'][frame, side] != b[match[0]]['board']
            counts['timing_cells'] += int((mask & ~aligned).sum())
            counts['persistent_color_cells'] += int((mask & aligned)[1:].sum())
            counts['hidden_inference_cells'] += int((mask & aligned)[0].sum())
        for i, (j, kind) in paired.items():
            diffs = np.argwhere(a[i]['board'] != b[j]['board'])
            matched_cells += a[i]['board'].size
            aligned_diffs += len(diffs)
            fa, fb = a[i]['frames'][len(a[i]['frames'])//2], b[j]['frames'][len(b[j]['frames'])//2]
            details.append(dict(side=side+1, left_event=i, right_event=j, kind=kind,
                                left_frame=fa, right_frame=fb, different_cells=len(diffs)))
            for row, col in diffs:
                residuals.append(dict(side=side, row=int(row), col=int(col), left_frame=fa, right_frame=fb,
                    original=int(a[i]['board'][row, col]), converted=int(b[j]['board'][row, col])))
    return dict(original_different_cells=int(original_diff.sum()), **counts,
        paired_events=len(details), paired_cells=matched_cells, aligned_different_cells=aligned_diffs,
        pairs=details, residuals=residuals, event_counts=event_counts, max_pair_shift_sec=MAX_PAIR_SHIFT_SEC)


def image_pair(case: dict, converted: Path, destination: Path) -> None:
    """動画から対応セルを切り出し、最近傍拡大で元と変換後を横に並べる。"""
    from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
    region = (DEFAULT_P1_REGION, DEFAULT_P2_REGION)[case['side']]
    images = []
    for path, frame, offset in ((DEFAULT_VIDEO, case['left_frame'], START),
                                (converted, case['right_frame'], 1.0)):
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, round((offset+frame/FPS)*cap.get(cv2.CAP_PROP_FPS)))
        ok, image = cap.read()
        cap.release()
        if not ok:
            raise ValueError('比較画像の対応フレームを取得できません')
        image = cv2.resize(image, (1920, 1080))
        x1, y1, x2, y2 = region.cell_sample_rect(case['row'], case['col'])
        images.append(cv2.resize(image[y1:y2, x1:x2], None, fx=CELL_SCALE, fy=CELL_SCALE,
                                 interpolation=cv2.INTER_NEAREST))
    combined = np.concatenate(images, axis=1)
    labeled = cv2.copyMakeBorder(combined, LABEL_HEIGHT, 0, 0, 0, cv2.BORDER_CONSTANT)
    text = f"original {START+case['left_frame']/FPS:.3f}s | converted {START+case['right_frame']/FPS:.3f}s"
    cv2.putText(labeled, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    cv2.putText(labeled, f"P{case['side']+1} row{case['row']} col{case['col']} : {case['original']} -> {case['converted']}",
                (5, 38), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    cv2.imwrite(str(destination), labeled)


def representative_cases(left: dict, right: dict, result: dict) -> list[dict]:
    """対応した盤面内の異なるフレームから5例を選ぶ（独立誤読数ではない）。"""
    cases = []
    for side in range(2):
        a, _ = events(left, side)
        b, _ = events(right, side)
        for pair in result['pairs']:
            if pair['side'] != side+1:
                continue
            aa, bb = a[pair['left_event']], b[pair['right_event']]
            for row, col in np.argwhere(aa['board'][1:] != bb['board'][1:]):
                row = int(row)+1
                for fa in aa['frames']:
                    fb = min(bb['frames'], key=lambda frame: abs(frame-fa))
                    cases.append(dict(side=side, row=row, col=int(col), left_frame=fa, right_frame=fb,
                        original=int(aa['board'][row, col]), converted=int(bb['board'][row, col])))
    if not cases:
        return []
    indices = np.linspace(0, len(cases)-1, min(REPRESENTATIVES, len(cases)), dtype=int)
    return [cases[index] for index in indices]


def main() -> None:
    a, b = load('original_a'), load('original_b')
    floor = compare(a, b)
    save_json(OUTPUT / 'floor.json', floor)
    report = dict(floor={k: v for k, v in floor.items() if k != 'breakdown'}, qualities={})
    for crf in (23, 35):
        data = load(f'crf{crf}')
        result = align(a, data)
        result['same_time'] = {k: v for k, v in compare(a, data).items() if k != 'breakdown'}
        visible = representative_cases(a, data, result)
        result['images'] = []
        if visible:
            transcode(DEFAULT_VIDEO, OUTPUT / f'crf{crf}.mp4', crf)
        for index, case in enumerate(visible[:REPRESENTATIVES]):
            path = OUTPUT / f'crf{crf}_residual_{index+1}.png'
            image_pair(case, OUTPUT / f'crf{crf}.mp4', path)
            result['images'].append(dict(path=str(path), **case))
        report['qualities'][str(crf)] = result
    save_json(OUTPUT / 'alignment.json', report)
    for crf in (23, 35):
        (OUTPUT / f'crf{crf}.mp4').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
