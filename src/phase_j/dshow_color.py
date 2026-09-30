"""DirectShow 入力の色行列補正 (既定 OFF)。

OBS 仮想カメラは RGB を BT.709 (limited) の NV12 へ変換して送るが、OpenCV の DirectShow
バックエンドは行列指定なしの YUV→RGB 変換を Windows 側に任せ、BT.601 として復号する
(実測: D:/puyo_analyzer/packaging/equiv/dshow_color/RESULT.md)。認識器は BT.709 復号の画素で
学習されているため、601 として復号された画素へ 3x3 行列を後掛けして 709 復号相当へ戻す。

補正は 8bit 丸め・0/255 クリップ後の画素に掛けるため厳密な逆変換ではない。
"""
from __future__ import annotations

import cv2
import numpy as np

MODE_OFF, MODE_AUTO, MODE_601_TO_709 = 'off', 'auto', '601to709'
COLOR_CORRECTION_MODES = (MODE_OFF, MODE_AUTO, MODE_601_TO_709)
DEFAULT_COLOR_CORRECTION = MODE_OFF

# 輝度係数 (Kr, Kb)。Kg = 1 - Kr - Kb。
BT601_KR_KB = (0.299, 0.114)
BT709_KR_KB = (0.2126, 0.0722)

# auto で補正対象にする DirectShow のメディアサブタイプ (YUV = Windows 側の色変換が入るもの)。
# OpenCV の CAP_PROP_FOURCC が返す fourcc (リトルエンディアン 4 文字)。
YUV_SUBTYPES = frozenset({'NV12', 'YUY2', 'YUYV', 'UYVY', 'I420', 'IYUV', 'YV12'})
FOURCC_BYTES = 4


def _decode_matrix(kr_kb: tuple[float, float]) -> np.ndarray:
    """正規化 (Y, Cb, Cr) → RGB の 3x3 行列 (列順 Y, Cb, Cr)。"""
    kr, kb = kr_kb
    kg = 1.0 - kr - kb
    cr_r, cb_b = 2.0 * (1.0 - kr), 2.0 * (1.0 - kb)
    return np.array([[1.0, 0.0, cr_r],
                     [1.0, -cb_b * kb / kg, -cr_r * kr / kg],
                     [1.0, cb_b, 0.0]], dtype=np.float64)


def bt601_to_709_matrix_rgb() -> np.ndarray:
    """601 として復号された RGB → 709 として復号された RGB の 3x3 行列 (RGB 順、オフセットなし)。
    limited range の Y/Cb/Cr 正規化係数は両行列で共通のため、レンジ係数は相殺される。"""
    return _decode_matrix(BT709_KR_KB) @ np.linalg.inv(_decode_matrix(BT601_KR_KB))


def bt601_to_709_matrix_bgr() -> np.ndarray:
    """OpenCV の BGR 画素へ cv2.transform で掛ける float32 行列。"""
    return bt601_to_709_matrix_rgb()[::-1, ::-1].astype(np.float32)


_MATRIX_BGR = bt601_to_709_matrix_bgr()


def correct_601_as_709(image: np.ndarray) -> np.ndarray:
    """BGR uint8 画像へ 601→709 後補正を掛けて新しい配列を返す (入力は変更しない)。"""
    return cv2.transform(image, _MATRIX_BGR)


def fourcc_to_str(value: float) -> str | None:
    """CAP_PROP_FOURCC の数値を 4 文字へ。不明 (0 以下) は None。"""
    code = int(value)
    if code <= 0:
        return None
    return code.to_bytes(FOURCC_BYTES, 'little').decode('latin1')


def resolve_correction(mode: str, subtype: str | None) -> bool:
    """補正を掛けるか。auto は YUV サブタイプが確認できたときだけ True (不明は掛けない)。"""
    if mode not in COLOR_CORRECTION_MODES:
        raise ValueError(f'color_correction は {COLOR_CORRECTION_MODES} のいずれかです: {mode!r}')
    if mode == MODE_601_TO_709:
        return True
    return mode == MODE_AUTO and subtype in YUV_SUBTYPES
