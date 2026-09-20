"""既存PNGの方向付き局所追跡。物理進行の判定器ではない。"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASIS = ROOT / 'data/verify/g2_event_completion_basis_2026-09-09_v1'
ROI_SOURCE = ROOT / 'src/next_detector.py'
ROI_SHA = '6f57af41704c8438ce172ea5dacc3c34d804014d0c88665c0860dc73d5ffb434'
VIDEO_SHA = 'b3728078cc2e8282065e5dd78ca1c13e6a443ffdf1dc4333e3da06f0852757d3'
MANIFESTS = {
    'normal_motion_v1': 'af22ebea84cd67ba7cd9e773d00859fbbd3825993f3cf2ca89cc7401f44f9219',
    'chain_exit_motion_v1': 'faaab0f23685000fafeba962535bfb626ccf397fd8a85ef1888a2522651ea58c',
}
FRAME_IDS = {
    'normal_motion_v1': (34876, 34884, 34888, 34892, 34896, 34898, 34902,
        34906, 34910, 34914, 34918, 34922, 34926, 34930, 34934, 34938, 34940, 34944, 34948),
    'chain_exit_motion_v1': (35772, 35776, 35780, 35784, 35786, 35790, 35794, 35798, 35802),
}
SIDES = ('1P', '2P')
SLOTS = ('NEXT_TOP', 'NEXT_BOT', 'DNEXT_TOP', 'DNEXT_BOT')
SOURCE_SIZE, TARGET_SIZE, FPS = (1920, 1080), (1280, 720), 60
FEATURE = dict(maxCorners=64, qualityLevel=0.01, minDistance=3,
               blockSize=3, useHarrisDetector=False, k=0.04)
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
          flags=0, minEigThreshold=1e-4)
QUANTILES = (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)
EXPECTED_RECORDS = 208


@dataclass(frozen=True)
class SavedImage:
    """元PNGの内容整合とgray入力。live物体identityではない。"""
    group: str
    frame: int
    sha256: str
    gray: np.ndarray


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_read(path: Path, expected: str) -> bytes:
    value = path.read_bytes()
    require(sha(value) == expected, 'input_sha:' + str(path))
    return value


def configure() -> None:
    """モデルを読み込まず画像処理だけCPUへ限定。"""
    cv2.setNumThreads(2)
    cv2.ocl.setUseOpenCL(False)
    cv2.setRNGSeed(0)


def scaled_roi(roi: tuple[int, ...]) -> tuple[int, int, int, int]:
    sy, sx = TARGET_SIZE[1] / SOURCE_SIZE[1], TARGET_SIZE[0] / SOURCE_SIZE[0]
    y1, y2, x1, x2 = roi
    return math.floor(y1 * sy), math.ceil(y2 * sy), math.floor(x1 * sx), math.ceil(x2 * sx)


def read_rois() -> dict[str, dict[str, tuple[int, int, int, int]]]:
    """原moduleをimportせず固定定数だけ抽出する。"""
    tree = ast.parse(checked_read(ROI_SOURCE, ROI_SHA))
    names = {f'ROI_{side}_{slot}' for side in SIDES for slot in SLOTS}
    found: dict[str, tuple[int, ...]] = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in names:
                value = ast.literal_eval(node.value)
                require(node.target.id not in found, 'duplicate_roi')
                require(type(value) is tuple and len(value) == 4, 'roi_shape')
                require(all(type(item) is int for item in value), 'roi_type')
                found[node.target.id] = value
    require(set(found) == names, 'roi_names')
    return {side: {slot: scaled_roi(found[f'ROI_{side}_{slot}']) for slot in SLOTS}
            for side in SIDES}


def manifest(group: str) -> dict[str, Any]:
    path = BASIS / group / 'EXTRACTION.json'
    result = json.loads(checked_read(path, MANIFESTS[group]))
    require(result['source_sha256'] == VIDEO_SHA, 'source_metadata')
    for key in ('drawn_or_resized', 'recognition_reexecuted', 'quality_gate_clear'):
        require(result[key] is False, key)
    require(tuple(row['frame'] for row in result['frames']) == FRAME_IDS[group], 'frames')
    return result


def load_images() -> tuple[dict[str, list[SavedImage]], dict[str, str]]:
    """PNGを一度だけ読んでSHA照合後にdecode。元動画は読まない。"""
    images: dict[str, list[SavedImage]] = {}
    inputs = {str(ROI_SOURCE): ROI_SHA}
    for group in MANIFESTS:
        data = manifest(group)
        inputs[str(BASIS / group / 'EXTRACTION.json')] = MANIFESTS[group]
        images[group] = []
        for row in data['frames']:
            frame = row['frame']
            require(type(frame) is int and row['file'] == f'source_{frame}.png', 'file_frame')
            require(row['time_sec'] == frame / FPS, 'frame_time')
            require(row['shape'] == [720, 1280, 3], 'saved_shape')
            path = BASIS / group / row['file']
            raw = checked_read(path, row['sha256'])
            color = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            require(color is not None and color.shape == (720, 1280, 3), 'decoded_shape')
            require(color.dtype == np.uint8, 'decoded_dtype')
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            images[group].append(SavedImage(group, frame, row['sha256'], gray))
            inputs[str(path)] = row['sha256']
    return images, inputs


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {'n': 0, 'mean': None, 'std': None, 'quantiles': None}
    array = np.asarray(values, dtype=np.float64)
    require(bool(np.isfinite(array).all()), 'nonfinite_stat')
    return {'n': len(values), 'mean': float(array.mean()), 'std': float(array.std()),
            'quantiles': {str(q): float(np.quantile(array, q)) for q in QUANTILES}}


def inside(point: np.ndarray, roi: tuple[int, ...]) -> bool:
    y1, y2, x1, x2 = roi
    return bool(x1 <= point[0] < x2 and y1 <= point[1] < y2)


def finite_xy(point: np.ndarray) -> list[float] | None:
    return [float(x) for x in point] if np.isfinite(point).all() else None


def scalar(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def starting_points(gray: np.ndarray, roi: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros_like(gray)
    y1, y2, x1, x2 = roi
    mask[y1:y2, x1:x2] = 255
    points = cv2.goodFeaturesToTrack(gray, mask=mask, **FEATURE)
    return np.empty((0, 1, 2), np.float32) if points is None else points


def trace_points(before: np.ndarray, after: np.ndarray, points: np.ndarray) -> list[dict[str, Any]]:
    """statusだけで対応を保存しFB誤差の閾値は設けない。"""
    if not len(points):
        return []
    following, status, error = cv2.calcOpticalFlowPyrLK(before, after, points, None, **LK)
    require(following is not None and status is not None, 'lk_forward_missing')
    target = following.reshape(-1, 2)
    bounds = (0, after.shape[0], 0, after.shape[1])
    usable = [i for i, point in enumerate(target)
              if status[i, 0] == 1 and finite_xy(point) is not None and inside(point, bounds)]
    backwards: dict[int, tuple[np.ndarray, int, Any]] = {}
    if usable:
        selected = following[usable].copy()
        back, back_status, back_error = cv2.calcOpticalFlowPyrLK(after, before, selected, None, **LK)
        require(back is not None and back_status is not None, 'lk_backward_missing')
        backwards = {i: (back[j, 0], int(back_status[j, 0]), back_error[j, 0])
                     for j, i in enumerate(usable)}
    return [point_record(i, p[0], target[i], int(status[i, 0]), error[i, 0],
                         backwards.get(i), bounds) for i, p in enumerate(points)]


def point_record(index: int, start: np.ndarray, end: np.ndarray, status: int,
                 error: Any, back: tuple[np.ndarray, int, Any] | None,
                 bounds: tuple[int, ...]) -> dict[str, Any]:
    valid_end = status == 1 and finite_xy(end) is not None and inside(end, bounds)
    valid_back = back is not None and back[1] == 1 and finite_xy(back[0]) is not None
    valid = bool(valid_end and valid_back and inside(back[0], bounds)) if valid_back else False
    return {'index': index, 'start': finite_xy(start), 'end': finite_xy(end),
            'forward_status': status, 'forward_in_image': bool(valid_end),
            'forward_l1_error': scalar(error) if status == 1 else None,
            'back': finite_xy(back[0]) if back else None,
            'backward_status': back[1] if back else None,
            'backward_l1_error': scalar(back[2]) if valid_back else None,
            'valid_both_status_in_image': valid,
            'dx': float(end[0] - start[0]) if valid else None,
            'dy': float(end[1] - start[1]) if valid else None,
            'fb_distance': float(np.linalg.norm(back[0] - start)) if valid else None}


def summarize(points: list[dict[str, Any]], roi: tuple[int, ...], gap: int) -> dict[str, Any]:
    valid = [p for p in points if p['valid_both_status_in_image']]
    retained = sum(inside(np.asarray(p['end']), roi) for p in valid)
    return {'seed_count': len(points),
            'forward_status_count': sum(p['forward_status'] == 1 for p in points),
            'forward_in_image_count': sum(p['forward_in_image'] for p in points),
            'backward_status_count': sum(p['backward_status'] == 1 for p in points),
            'valid_count': len(valid), 'tracking_unavailable_count': len(points) - len(valid),
            'retained_origin_slot_count': retained,
            'left_origin_slot_count': len(valid) - retained,
            'retained_fraction_of_valid': retained / len(valid) if valid else None,
            'dx_px': stats([p['dx'] for p in valid]),
            'dy_px': stats([p['dy'] for p in valid]),
            'dx_px_per_source_frame': stats([p['dx'] / gap for p in valid]),
            'dy_px_per_source_frame': stats([p['dy'] / gap for p in valid]),
            'fb_distance_px': stats([p['fb_distance'] for p in valid]),
            'forward_l1_error': stats([p['forward_l1_error'] for p in valid])}


def measure_slot(before: SavedImage, after: SavedImage, side: str, slot: str,
                 rois: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    require(before.group == after.group and before.frame < after.frame, 'pair_order')
    roi = rois[slot]
    points = trace_points(before.gray, after.gray, starting_points(before.gray, roi))
    for point in points:
        end = point['end']
        point['end_slots'] = ([name for name, box in rois.items()
                               if inside(np.asarray(end), box)] if end is not None else [])
    gap = after.frame - before.frame
    return {'group': before.group, 'side': side, 'slot': slot, 'roi_y1_y2_x1_x2': list(roi),
            'frame_before': before.frame, 'frame_after': after.frame, 'frame_gap': gap,
            'time_before': before.frame / FPS, 'time_after': after.frame / FPS,
            'png_before_sha256': before.sha256, 'png_after_sha256': after.sha256,
            'summary': summarize(points, roi, gap), 'points': points,
            'physical_progress_certified': False, 'color_classification_used': False}


def measure_all(images: dict[str, list[SavedImage]], rois: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for values in images.values():
        for before, after in zip(values, values[1:]):
            for side in SIDES:
                for slot in SLOTS:
                    result.append(measure_slot(before, after, side, slot, rois[side]))
    require(len(result) == EXPECTED_RECORDS, 'coverage_count')
    return result


def environment() -> dict[str, Any]:
    return {'opencv': cv2.__version__, 'numpy': np.__version__,
            'opencv_threads': cv2.getNumThreads(), 'opencl_used': cv2.ocl.useOpenCL(),
            'feature': FEATURE, 'lk': LK, 'quantiles': QUANTILES,
            'gray': 'COLOR_BGR2GRAY', 'rounding': 'lower_floor_upper_ceil',
            'fb_threshold': None, 'retention_denominator': 'valid_both_status_in_image',
            'point_identity': 'numerical_correspondence_not_physical_object_identity',
            'roi_overlap': 'preserved; do_not_sum_slots_as_disjoint_objects'}
