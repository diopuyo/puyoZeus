"""原NEXT点列との一致と、限定置換の所有/例外/復元を確認する。"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from scripts import g3_next_features_fast as F


@pytest.fixture
def original() -> ModuleType:
    """実原moduleを専用名でloadし、稼働G3のmoduleには触れない。"""
    spec = importlib.util.spec_from_file_location('_g3_fast_test_original', F.SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.configure()
    return module


def test_saved_png_points_and_lk_match(original: ModuleType) -> None:
    """固定PNG28枚の全8ROIと、同点列からの追跡結果が一致する。"""
    images, _ = original.load_images()
    rois = [r for slots in original.read_rois().values() for r in slots.values()]
    old = original.starting_points
    count = 0
    with F.installed(original) as receipt:
        for frames in images.values():
            for image in frames:
                before = hashlib.sha256(image.gray.tobytes()).hexdigest()
                for roi in rois:
                    a, b = old(image.gray, roi), original.starting_points(image.gray, roi)
                    np.testing.assert_array_equal(a, b)
                    count += 1
                assert hashlib.sha256(image.gray.tobytes()).hexdigest() == before
            first, second = frames[:2]
            a = original.trace_points(first.gray, second.gray, old(first.gray, rois[0]))
            b = original.trace_points(first.gray, second.gray,
                                      original.starting_points(first.gray, rois[0]))
            assert a == b
    assert count == 224 and receipt['restored'] and original.starting_points is old
    if receipt['build_supported']:
        assert receipt['fast_calls'] == 226 and receipt['fallback_calls'] == 0


@pytest.mark.parametrize('kind', ['flat', 'checker', 'noise', 'low_contrast'])
def test_crop_border_ties_and_contrast(original: ModuleType, kind: str) -> None:
    """境界と同点付近の選別を、点列順序込みで原関数と比較する。"""
    size = 128
    rng = np.random.default_rng(13)
    image = {'flat': np.zeros((size, size), np.uint8),
             'checker': (np.indices((size, size)).sum(axis=0)//4 % 2 * 255).astype(np.uint8),
             'noise': rng.integers(0, 256, (size, size), dtype=np.uint8),
             'low_contrast': rng.integers(100, 105, (size, size), dtype=np.uint8)}[kind]
    for roi in ((0, 30, 0, 30), (98, 128, 98, 128), (31, 67, 35, 69), (0, 128, 0, 128)):
        np.testing.assert_array_equal(original.starting_points(image, roi), F.crop_points(original, image, roi))


def test_fallback_and_original_exception(original: ModuleType, monkeypatch: Any) -> None:
    """条件外の小画像/不正ROI/変更FEATUREを元関数に渡す。"""
    old = original.starting_points
    image = np.zeros((128, 128), np.uint8)
    with F.installed(original) as receipt:
        np.testing.assert_array_equal(old(image, (0, 30, 0, 30)),
                                      original.starting_points(image, (0, 30, 0, 30)))
        with pytest.raises(ValueError):
            original.starting_points(image, ())
        monkeypatch.setitem(original.FEATURE, 'maxCorners', 1)
        roi = next(iter(original.read_rois()['1P'].values()))
        frame = np.zeros(F.SHAPE, np.uint8)
        np.testing.assert_array_equal(old(frame, roi), original.starting_points(frame, roi))
    assert receipt['fallback_calls'] == 3 and receipt['restored']


def test_unverified_build_uses_original(original: ModuleType, monkeypatch: Any) -> None:
    """未検証ビルドは高速化せず、fallbackを明記する。"""
    monkeypatch.setattr(F, 'BUILD_SHA', 'not-this-build')
    with F.installed(original) as receipt:
        roi = next(iter(original.read_rois()['1P'].values()))
        original.starting_points(np.zeros(F.SHAPE, np.uint8), roi)
    assert not receipt['build_supported'] and receipt['fallback_calls'] == 1


def test_changed_function_and_duplicate_install_rejected(original: ModuleType, monkeypatch: Any) -> None:
    """ファイルだけ同じで実関数が違う接続と二重設置を拒否する。"""
    with F.installed(original):
        with pytest.raises(ValueError, match='function_owner'):
            with F.installed(original):
                pytest.fail('二重設置')
    monkeypatch.setattr(original, 'starting_points', lambda *args: None)
    with pytest.raises(ValueError, match='function_owner'):
        F.verified_original(original)


def test_restore_preserves_body_and_foreign_owner(original: ModuleType) -> None:
    """原例外は維持し、外部による関数置換を上書きしない。"""
    old = original.starting_points
    with pytest.raises(RuntimeError, match='body'):
        with F.installed(original) as receipt:
            raise RuntimeError('body')
    assert receipt['restored'] and original.starting_points is old
    foreign = lambda *args: None
    with pytest.raises(ValueError, match='restore_owner_conflict'):
        with F.installed(original) as receipt:
            original.starting_points = foreign
    assert receipt['owner_conflict'] and original.starting_points is foreign
    original.starting_points = old
    with pytest.raises(RuntimeError, match='body'):
        with F.installed(original) as receipt:
            original.starting_points = foreign
            raise RuntimeError('body')
    assert receipt['owner_conflict'] and original.starting_points is foreign


def test_actual_private_dependency_loader(monkeypatch: Any) -> None:
    """原dependencyのSHA/alias認証を通したmoduleと同じ実関数を置換する。"""
    source = F.ROOT / 'data/verify/g2_directional_next_provider_2026-09-09_v2/dependencies.py'
    spec = importlib.util.spec_from_file_location('_g3_fast_private_dependencies', source)
    dependency = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dependency)
    alias = '_parent_motion_provider_observables'
    monkeypatch.delitem(sys.modules, alias, raising=False)
    # 原loaderが作成するaliasも試験終了時に復元する。
    monkeypatch.setitem(sys.modules, alias, None)
    del sys.modules[alias]
    module = dependency.load('observables')
    old = module.starting_points
    with F.installed(module) as receipt:
        assert dependency.load('observables') is module
        roi = next(iter(module.read_rois()['1P'].values()))
        frame = np.random.default_rng(12).integers(0, 256, F.SHAPE, dtype=np.uint8)
        np.testing.assert_array_equal(old(frame, roi), dependency.load('observables').starting_points(frame, roi))
    assert dependency.load('observables').starting_points is old and receipt['restored']
