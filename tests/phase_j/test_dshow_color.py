"""DirectShow 色行列補正 (src/phase_j/dshow_color.py) の試験。既定 OFF は従来と bit 一致であることを固定する。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.run_live_pipeline_20260928 import parse_args
from src.phase_j import dshow_color, launcher
from src.phase_j.dshow_color import (BT601_KR_KB, BT709_KR_KB, COLOR_CORRECTION_MODES, YUV_SUBTYPES,
                                     bt601_to_709_matrix_bgr, bt601_to_709_matrix_rgb, correct_601_as_709,
                                     fourcc_to_str, resolve_correction)
from src.phase_j.launcher import (build_pipeline_argv, build_pipeline_config, parse_user_config,
                                  LauncherConfigError)
from src.phase_j.live_device import DeviceConfig, DirectShowSource

ROOT = Path(__file__).resolve().parents[2]
NV12_FOURCC = int.from_bytes(b'NV12', 'little')
MJPG_FOURCC = int.from_bytes(b'MJPG', 'little')
RANDOM_SEED = 20260930
PIXEL_TOLERANCE = 2  # 8bit 丸めによる往復誤差の許容 (クリップされない画素)
DSHOW = dict(device_name='OBS Virtual Camera')


def _encode(rgb: np.ndarray, kr_kb: tuple[float, float]) -> np.ndarray:
    """独立実装の RGB(0-255)→(Y,Cb,Cr) limited range (試験用の参照式)。"""
    kr, kb = kr_kb
    kg = 1 - kr - kb
    r, g, b = (rgb[..., i] / 255.0 for i in range(3))
    y = kr * r + kg * g + kb * b
    return np.stack([16 + 219 * y, 128 + 224 * (b - y) / (2 * (1 - kb)),
                     128 + 224 * (r - y) / (2 * (1 - kr))], -1)


def _decode(ycc: np.ndarray, kr_kb: tuple[float, float]) -> np.ndarray:
    kr, kb = kr_kb
    kg = 1 - kr - kb
    y, cb, cr = (ycc[..., 0] - 16) / 219, (ycc[..., 1] - 128) / 224, (ycc[..., 2] - 128) / 224
    r, b = y + 2 * (1 - kr) * cr, y + 2 * (1 - kb) * cb
    g = (y - kr * r - kb * b) / kg
    return np.stack([r, g, b], -1) * 255.0


def test_matrix_values_match_measured_reference() -> None:
    """decode_diag が実測で使った fixC と同じ行列 (RGB 順、小数 4 桁)。"""
    expected = np.array([[1.0864, -0.0723, -0.0141], [0.0965, 0.8451, 0.0584], [-0.0141, -0.0277, 1.0418]])
    assert np.allclose(bt601_to_709_matrix_rgb(), expected, atol=5e-5)


def test_matrix_rows_sum_to_one_so_grays_are_invariant() -> None:
    assert np.allclose(bt601_to_709_matrix_rgb().sum(axis=1), 1.0, atol=1e-9)
    gray = np.repeat(np.arange(256, dtype=np.uint8)[:, None, None], 3, axis=2)
    assert np.abs(correct_601_as_709(gray).astype(int) - gray.astype(int)).max() <= 1


def test_bgr_matrix_is_channel_reversed_rgb_matrix() -> None:
    assert np.array_equal(bt601_to_709_matrix_bgr(), bt601_to_709_matrix_rgb()[::-1, ::-1].astype(np.float32))


def test_correction_inverts_709_encode_601_decode_mismatch() -> None:
    """709 で符号化 → 601 で復号 (実測の不具合) した画素へ補正を掛けると 709 復号値へ戻る。"""
    rng = np.random.default_rng(RANDOM_SEED)
    rgb = rng.uniform(40, 215, size=(64, 64, 3))  # 中間域はクリップしない
    ycc = np.rint(_encode(rgb, BT709_KR_KB))
    wrong = np.clip(np.rint(_decode(ycc, BT601_KR_KB)), 0, 255).astype(np.uint8)
    right = np.clip(np.rint(_decode(ycc, BT709_KR_KB)), 0, 255)
    fixed = correct_601_as_709(wrong[..., ::-1].copy())[..., ::-1].astype(float)  # RGB→BGR 経由
    assert np.abs(fixed - right).max() <= PIXEL_TOLERANCE
    assert np.abs(wrong.astype(float) - right).mean() > 2 * np.abs(fixed - right).mean()  # 補正が効いている


def test_correction_does_not_mutate_input_and_keeps_dtype_shape() -> None:
    image = np.random.default_rng(RANDOM_SEED).integers(0, 256, (36, 64, 3), dtype=np.uint8)
    before = image.copy()
    out = correct_601_as_709(image)
    assert np.array_equal(image, before) and out.dtype == np.uint8 and out.shape == image.shape
    assert out is not image


def test_extremes_are_clipped_not_wrapped() -> None:
    image = np.array([[[0, 0, 0], [255, 255, 255], [0, 0, 255], [255, 0, 0]]], dtype=np.uint8)
    out = correct_601_as_709(image)
    assert out.dtype == np.uint8 and out.min() >= 0 and out.max() <= 255
    assert np.array_equal(out[0, 0], [0, 0, 0]) and out[0, 1].min() >= 254


@pytest.mark.parametrize('mode,subtype,expected', [
    ('off', 'NV12', False), ('off', None, False), ('601to709', None, True), ('601to709', 'MJPG', True),
    ('auto', 'NV12', True), ('auto', 'YUY2', True), ('auto', 'MJPG', False), ('auto', 'RGB3', False),
    ('auto', None, False)])
def test_resolve_correction(mode: str, subtype: str | None, expected: bool) -> None:
    assert resolve_correction(mode, subtype) is expected


def test_resolve_correction_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        resolve_correction('bt2020', 'NV12')


def test_fourcc_to_str() -> None:
    assert fourcc_to_str(NV12_FOURCC) == 'NV12' and fourcc_to_str(0) is None and fourcc_to_str(-1) is None
    assert 'NV12' in YUV_SUBTYPES and 'MJPG' not in YUV_SUBTYPES


def test_modes_are_off_auto_601to709_and_launcher_copy_matches() -> None:
    """launcher は cv2 を避けるため定数を複製している。値の食い違いを固定する。"""
    assert COLOR_CORRECTION_MODES == ('off', 'auto', '601to709') == launcher.COLOR_CORRECTION_MODES
    assert dshow_color.DEFAULT_COLOR_CORRECTION == launcher.DEFAULT_COLOR_CORRECTION == 'off'


# ---- DeviceConfig ----

def test_device_config_default_off_and_backward_compatible(tmp_path: Path) -> None:
    assert DeviceConfig('OBS', 0).color_correction == 'off'
    path = tmp_path / 'i.json'
    path.write_text('{"name":"OBS","index":2}')
    assert DeviceConfig.load(path) == DeviceConfig('OBS', 2, True, 'off')
    path.write_text('{"name":"OBS","index":2,"color_correction":"auto"}')
    assert DeviceConfig.load(path).color_correction == 'auto'


def test_device_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        DeviceConfig('OBS', 0, True, 'bt2020')


# ---- DirectShowSource (実機なし) ----

class FakeDevice:
    """read は固定画像。fourcc=None なら get を持たない (BufferedCapture 相当)。"""

    def __init__(self, image: np.ndarray, fourcc: int | None = None) -> None:
        self.image, self.released = image, False
        if fourcc is not None:
            self.get = lambda key: float(fourcc) if key == cv2.CAP_PROP_FOURCC else 0.0

    def isOpened(self) -> bool:
        return True

    def set(self, key: int, value: float) -> bool:
        return True

    def read(self) -> tuple:
        return True, self.image.copy()

    def release(self) -> None:
        self.released = True


def _frames(config: DeviceConfig, device: FakeDevice) -> list[np.ndarray]:
    clock = [0.0]
    source = DirectShowSource(config, 0.2, lambda: None, lambda *args: device, lambda image: True,
                              lambda: clock[0], lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    return [frame.image for frame in source]


def _test_image() -> np.ndarray:
    return np.random.default_rng(RANDOM_SEED).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)


def test_off_is_byte_identical_to_legacy_construction() -> None:
    """既定 (引数なし) と明示 off は、補正なしの生画素と bit 一致 (get を呼ばない機器でも動く)。"""
    image = _test_image()
    legacy = _frames(DeviceConfig('OBS', 0), FakeDevice(image))
    explicit = _frames(DeviceConfig('OBS', 0, True, 'off'), FakeDevice(image, NV12_FOURCC))
    assert legacy and explicit and all(np.array_equal(f, image) for f in legacy + explicit)


def test_forced_correction_applies_matrix() -> None:
    image = _test_image()
    frames = _frames(DeviceConfig('OBS', 0, True, '601to709'), FakeDevice(image))
    assert frames and all(np.array_equal(f, correct_601_as_709(image)) for f in frames)
    assert not np.array_equal(frames[0], image)


@pytest.mark.parametrize('fourcc,corrected', [(NV12_FOURCC, True), (MJPG_FOURCC, False), (None, False)])
def test_auto_follows_negotiated_subtype(fourcc: int | None, corrected: bool) -> None:
    image = _test_image()
    frames = _frames(DeviceConfig('OBS', 0, True, 'auto'), FakeDevice(image, fourcc))
    expected = correct_601_as_709(image) if corrected else image
    assert frames and all(np.array_equal(f, expected) for f in frames)


# ---- launcher / live_config ----

def test_launcher_default_omits_key_and_accepts_modes(tmp_path: Path) -> None:
    user = parse_user_config(DSHOW, tmp_path)
    assert user.dshow_color_correction == 'off' and 'color_correction' not in build_pipeline_config(user)
    for mode in ('auto', '601to709'):
        user = parse_user_config(dict(DSHOW, dshow_color_correction=mode), tmp_path)
        assert build_pipeline_config(user)['color_correction'] == mode


def test_launcher_rejects_unknown_mode(tmp_path: Path) -> None:
    with pytest.raises(LauncherConfigError, match='dshow_color_correction'):
        parse_user_config(dict(DSHOW, dshow_color_correction='bt2020'), tmp_path)


def test_real_pipeline_parser_accepts_color_correction_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """パイプライン CLI (apply_config) が color_correction を未知項目として拒否しない。"""
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    user = parse_user_config(dict(DSHOW, dshow_color_correction='601to709', output_dir=str(tmp_path / 'o')), tmp_path)
    path = tmp_path / 'p.json'
    path.write_text(json.dumps(build_pipeline_config(user)), encoding='utf-8')
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr('sys.argv', ['live', *build_pipeline_argv(user, path)])
    options = parse_args()
    assert options.source == 'dshow' and options.input_config == path
    assert DeviceConfig.load(path).color_correction == '601to709'
    # 配布既定の画面確認の緩和 (掛け算式許容・上辺免除・連続 3 回不合格) が入力設定まで届く
    loaded = DeviceConfig.load(path)
    assert loaded.relaxed_verify is True and loaded.verify_fail_streak == 3
