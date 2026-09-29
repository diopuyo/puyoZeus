"""D3で用いたNEXT実移動・掛け算式の画像検出。"""
from __future__ import annotations
import cv2
import numpy as np
from src.score_ocr import SCORE_1P_REGION, SCORE_2P_REGION

PATCH_HALF, SEARCH_X, SEARCH_Y = 16, 12, 48
MIN_NCC, MIN_GAIN, MIN_SHIFT = .80, .08, 2
NEXT_CENTERS = ((747, 199), (1172, 199))
PAD_Y, PAD_X, MULT_LEFT, MULT_RIGHT = 16, 8, 160, 200
NEXT_DEBOUNCE_SEC = .20


def next_translation(previous: np.ndarray, current: np.ndarray, side: int) -> dict:
    """発光だけを除き、NEXT内部の絵柄の平行移動を確認する。"""
    cx, cy = NEXT_CENTERS[side]
    patch = previous[cy-PATCH_HALF:cy+PATCH_HALF, cx-PATCH_HALF:cx+PATCH_HALF]
    search = current[cy-PATCH_HALF-SEARCH_Y:cy+PATCH_HALF+SEARCH_Y,
                     cx-PATCH_HALF-SEARCH_X:cx+PATCH_HALF+SEARCH_X]
    scores = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
    _, high, _, location = cv2.minMaxLoc(scores)
    dx, dy = location[0]-SEARCH_X, location[1]-SEARCH_Y
    gain = high-float(scores[SEARCH_Y, SEARCH_X])
    return dict(moving=high >= MIN_NCC and gain >= MIN_GAIN and abs(dy) >= MIN_SHIFT,
                ncc=high, gain=gain, dx=dx, dy=dy)


def multiply(frame: np.ndarray, template: np.ndarray) -> list[float]:
    """既存×テンプレートの固定列を上下左右の余白付きで探索する。"""
    values = []
    for y1, y2, x1, _ in (SCORE_1P_REGION, SCORE_2P_REGION):
        roi = frame[y1-PAD_Y:y2+PAD_Y, x1+MULT_LEFT-PAD_X:x1+MULT_RIGHT+PAD_X]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        values.append(float(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED).max()))
    return values
