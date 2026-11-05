"""仮想カメラの色変換と色差間引きを受けた目印を事前生成する。"""
from __future__ import annotations

from functools import lru_cache
import math
import cv2
import numpy as np

from .dshow_color import BT709_KR_KB, correct_601_as_709

SCALES = (1.0, .8, .7, .5)
CHROMA_BLOCK = 2
PADDING = 4
Y_OFFSET, Y_RANGE = 16, 219
CHROMA_OFFSET, CHROMA_RANGE = 128, 224
REFERENCES_PER_SCALE = 1 + CHROMA_BLOCK ** 2 * 2
CORRELATION_ROUNDING_MARGIN = 1e-5


def camera_image(image: np.ndarray) -> np.ndarray:
    """709 limited を4:2:0へ平均し、DirectShowと同じ601で復号する。"""
    kr, kb = BT709_KR_KB
    blue, green, red = (image[..., i].astype(np.float32) / 255 for i in range(3))
    y = kr * red + (1 - kr - kb) * green + kb * blue
    u = CHROMA_OFFSET + CHROMA_RANGE * (blue - y) / (2 * (1 - kb))
    v = CHROMA_OFFSET + CHROMA_RANGE * (red - y) / (2 * (1 - kr))
    height, width = image.shape[:2]
    planes = [Y_OFFSET + Y_RANGE * y] + [cv2.resize(
        c, (width // CHROMA_BLOCK, height // CHROMA_BLOCK), interpolation=cv2.INTER_AREA)
        for c in (u, v)]
    packed = np.concatenate([np.clip(np.rint(c), 0, 255).astype(np.uint8).ravel() for c in planes])
    return cv2.cvtColor(packed.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_I420)


@lru_cache(maxsize=1)
def templates() -> tuple[np.ndarray, ...]:
    """縮尺と縦横の偶奇位相ごとに、補正前・補正後の参照を一度だけ作る。"""
    from .distribution_input import marker_image
    result = []
    for scale in SCALES:
        marker = cv2.resize(marker_image(), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        result.append(marker)
        height, width = marker.shape[:2]
        for dy in range(CHROMA_BLOCK):
            for dx in range(CHROMA_BLOCK):
                canvas = np.zeros(((height + PADDING * 2 + 1) // 2 * 2,
                                   (width + PADDING * 2 + 1) // 2 * 2, 3), dtype=np.uint8)
                y, x = PADDING + dy, PADDING + dx
                canvas[y:y + height, x:x + width] = marker
                raw = camera_image(canvas)
                for image in (raw, correct_601_as_709(raw)):
                    result.append(image[y:y + height, x:x + width].copy())
    for template in result:
        template.flags.writeable = False
    return tuple(result)


@lru_cache(maxsize=1)
def reference_groups(threshold: float) -> tuple[tuple, ...]:
    """相関の幾何学的上限から、合格し得ない参照だけを省く条件を作る。"""
    bank, groups = templates(), []
    for start in range(0, len(bank), REFERENCES_PER_SCALE):
        original, variants = bank[start], []
        for template in bank[start + 1:start + REFERENCES_PER_SCALE]:
            rho = float(cv2.matchTemplate(original, template, cv2.TM_CCOEFF_NORMED)[0, 0])
            # 正規化したベクトルの内積。候補と参照がthreshold以上なら、
            # 候補とoriginalの相関は必ずこの下限以上（Cauchy–Schwarz）。
            lower = threshold * rho - math.sqrt(max(0, (1 - threshold ** 2) * (1 - rho ** 2)))
            variants.append((template, lower - CORRELATION_ROUNDING_MARGIN))
        groups.append((original, tuple(variants)))
    return tuple(groups)
