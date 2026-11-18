"""映像の形と移動から連鎖終了の候補を検証する。"""
from __future__ import annotations
import cv2
import numpy as np

FRAME_SHAPE = (1080, 1920)
MATCH_MIN = 0.80
MOVE_MIN = 3
MOVE_TOLERANCE = 4
PATCH_HALF = 18
MIN_PATCH_STD = 10.0
SEARCH_UP = 30
SEARCH_DOWN = 10
SEARCH_X = 18
DNEXT_CENTERS = {'1P': ((790, 318), (790, 365)), '2P': ((1130, 318), (1130, 365))}
FIELD_LEFT = {'1P': 272, '2P': 1252}
FIELD_TOP = 160
CELL_WIDTH = 64
COLUMNS = 6
EYE_RADIUS = 12
EYE_INNER_RADIUS = 8
EYE_TEMPLATE_SHAPE = (28, 52)
EYE_MATCH_MIN = 0.65
EYE_MAX_SATURATION = 50
ENTRY_MAX_Y = 12
EYE_SEARCH_HEIGHT = 58
EYE_BACKGROUND = 140
EYE_RING = 240
EYE_PUPIL = 80
EYE_CENTERS = ((13, 13), (38, 13))


def valid_frames(prev: np.ndarray | None, curr: np.ndarray | None) -> bool:
    """固定座標の対象外は信号を出さない。"""
    return all(im is not None and im.shape[:2] == FRAME_SHAPE for im in (prev, curr))


def verified_next_motion(prev: np.ndarray | None, curr: np.ndarray | None, side: str) -> bool:
    """下側NEXTの2個が同じ向きに上昇した場合だけ移動を採用する。"""
    if not valid_frames(prev, curr):
        return False
    moves = []
    for x, y in DNEXT_CENTERS[side]:
        patch = prev[y-PATCH_HALF:y+PATCH_HALF, x-PATCH_HALF:x+PATCH_HALF]
        if cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).std() < MIN_PATCH_STD:
            return False
        area = curr[y-PATCH_HALF-SEARCH_UP:y+PATCH_HALF+SEARCH_DOWN,
                    x-PATCH_HALF-SEARCH_X:x+PATCH_HALF+SEARCH_X]
        scores = cv2.matchTemplate(area, patch, cv2.TM_CCOEFF_NORMED)
        _, score, _, (mx, my) = cv2.minMaxLoc(scores)
        dx, dy = mx-SEARCH_X, my-SEARCH_UP
        if not np.isfinite(score) or score < MATCH_MIN or dy > -MOVE_MIN:
            return False
        moves.append((dx, dy))
    return max(abs(a-b) for a, b in zip(*moves)) <= MOVE_TOLERANCE


def eye_template() -> np.ndarray:
    """おじゃまの横並びの白い環を作る（攻撃演出は縦並び）。"""
    template = np.full(EYE_TEMPLATE_SHAPE, EYE_BACKGROUND, dtype=np.uint8)
    for center in EYE_CENTERS:
        cv2.circle(template, center, EYE_RADIUS, EYE_RING, -1, cv2.LINE_AA)
        cv2.circle(template, center, EYE_INNER_RADIUS, EYE_PUPIL, -1, cv2.LINE_AA)
    return template


EYE_TEMPLATE = eye_template()


def top_eyes(frame: np.ndarray, side: str) -> list[int | None]:
    """列ごとに低彩度で横並びの目がある高さを返す。"""
    found = []
    height, width = EYE_TEMPLATE_SHAPE
    for col in range(COLUMNS):
        left = FIELD_LEFT[side] + CELL_WIDTH * col
        roi = frame[FIELD_TOP:FIELD_TOP+EYE_SEARCH_HEIGHT, left:left+CELL_WIDTH]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        scores = cv2.matchTemplate(gray, EYE_TEMPLATE, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        saturation = cv2.cvtColor(roi[y:y+height, x:x+width], cv2.COLOR_BGR2HSV)[:, :, 1]
        found.append(y if score >= EYE_MATCH_MIN and saturation.mean() < EYE_MAX_SATURATION else None)
    return found


def incoming_ojama(prev: np.ndarray | None, curr: np.ndarray | None, side: str) -> bool:
    """上端に新しく入ったおじゃまだけを拾い、既存の上段を除く。"""
    if not valid_frames(prev, curr):
        return False
    before, after = top_eyes(prev, side), top_eyes(curr, side)
    return any(old is None and new is not None and new <= ENTRY_MAX_Y
               for old, new in zip(before, after))
