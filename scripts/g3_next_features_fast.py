"""固定NEXT観測器の特徴点検出を領域周辺へ限定する任意候補。"""
from __future__ import annotations

from contextlib import contextmanager
import functools
import hashlib
from pathlib import Path
from types import CodeType, FunctionType, ModuleType
from typing import Any, Callable, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/verify/g2_next_motion_observables_2026-09-09_v1/observables.py'
SOURCE_SHA = '31b29d44669cc62727c18dac0bbd235a7382eba25139ed224995edba50a6606c'
BUILD_SHA = '9e0b6b5d3c457d7794d56f55acd3afdadd8fd4b5a8a562e78bcab4fadaf1e604'
HALO = 8
SHAPE = (720, 1280)
FEATURE = dict(maxCorners=64, qualityLevel=0.01, minDistance=3,
               blockSize=3, useHarrisDetector=False, k=0.04)


def require(condition: bool, reason: str) -> None:
    """対象が不明な接続は黙って最適化しない。"""
    if not condition:
        raise ValueError('g3_next_features_fast:' + reason)


def verified_original(module: ModuleType) -> Callable:
    """原fileと実関数codeの両方を認証する。原guardは置換しない。"""
    require(Path(module.__file__).resolve() == SOURCE.resolve(), 'source_path')
    raw = SOURCE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SOURCE_SHA, 'source_sha')
    original = vars(module).get('starting_points')
    require(type(original) is FunctionType and original.__globals__ is vars(module), 'function_owner')
    compiled = compile(raw, original.__code__.co_filename, 'exec')
    expected = next(c for c in compiled.co_consts
                    if isinstance(c, CodeType) and c.co_name == 'starting_points')
    require(original.__code__ == expected, 'function_code')
    require(module.FEATURE == FEATURE, 'feature_contract')
    return original


def crop_points(module: ModuleType, gray: np.ndarray, roi: tuple[int, ...]) -> np.ndarray:
    """検証済みの正規入力に使用。原配列・原座標系を維持する。"""
    y1, y2, x1, x2 = roi
    top, bottom = max(0, y1 - HALO), min(gray.shape[0], y2 + HALO)
    left, right = max(0, x1 - HALO), min(gray.shape[1], x2 + HALO)
    cropped = gray[top:bottom, left:right].copy()
    mask = np.zeros_like(cropped)
    mask[y1 - top:y2 - top, x1 - left:x2 - left] = 255
    points = module.cv2.goodFeaturesToTrack(cropped, mask=mask, **FEATURE)
    if points is None:
        return np.empty((0, 1, 2), np.float32)
    points[:, :, 0] += left
    points[:, :, 1] += top
    return points


def make_wrapper(module: ModuleType, original: Callable, receipt: dict) -> Callable:
    """条件外は原関数へそのまま渡し、例外や出力の仕様を保持する。"""
    rois = frozenset(roi for side in module.read_rois().values() for roi in side.values())
    detector = module.cv2.goodFeaturesToTrack
    build_supported = receipt['build_supported']
    @functools.wraps(original)
    def starting_points(gray: Any, roi: Any) -> np.ndarray:
        valid_roi = type(roi) is tuple and all(type(v) is int for v in roi) and roi in rois
        valid = (build_supported and module.FEATURE == FEATURE
                 and module.cv2.goodFeaturesToTrack is detector and valid_roi
                 and type(gray) is np.ndarray and gray.dtype == np.uint8
                 and gray.shape == SHAPE and gray.flags.c_contiguous)
        receipt['fast_calls' if valid else 'fallback_calls'] += 1
        return crop_points(module, gray, roi) if valid else original(gray, roi)
    return starting_points


@contextmanager
def installed(module: ModuleType) -> Iterator[dict]:
    """呼出側が認証済みmoduleへ明示接続する。稼働中processには使用しない。"""
    original = verified_original(module)
    build = hashlib.sha256(module.cv2.getBuildInformation().encode()).hexdigest()
    receipt = dict(source_sha=SOURCE_SHA, build_sha=build, build_supported=build == BUILD_SHA,
                   halo=HALO, fast_calls=0, fallback_calls=0, restored=False, owner_conflict=False,
                   quality_gate_clear=False, original_guard_preserved=True)
    wrapper = make_wrapper(module, original, receipt)
    module.starting_points = wrapper
    failed = False
    try:
        yield receipt
    except BaseException:
        failed = True
        raise
    finally:
        if vars(module).get('starting_points') is wrapper:
            module.starting_points = original
            receipt['restored'] = True
        else:
            receipt['owner_conflict'] = True
            if not failed:
                raise ValueError('g3_next_features_fast:restore_owner_conflict')
