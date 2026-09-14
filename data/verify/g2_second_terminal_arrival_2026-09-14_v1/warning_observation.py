"""既存CNNとNCCが支持する予告陽性だけを記録。無検出を予告ゼロにしない。"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
from typing import Any

DROP_CAP = 30


@dataclass(frozen=True)
class Observation:
    scope: tuple
    frame: int
    call_token: str
    status: str
    cells: tuple[dict, ...]
    image_sha256: str | None
    conditional_lower_bound: int | None = None
    calibrated: bool = False
    future_landing_guaranteed: bool = False


class Sampler:
    def __init__(self, module: Any, detector: Any) -> None:
        self.module, self.detector = module, detector

    def sample(self, scope: tuple, frame: int, call: str, image: Any) -> Observation:
        if type(frame) is not int or frame < 0 or not call or scope[-1] not in ('1P', '2P'):
            raise ValueError('warning_observation:scope_clock')
        m, model = self.module, self.detector._cnn
        if model is None:
            return Observation(scope, frame, call, 'UNKNOWN_MODEL', (), None)
        if image is None or image.shape != (m.FRAME_HEIGHT, m.FRAME_WIDTH, 3):
            return Observation(scope, frame, call, 'UNKNOWN_IMAGE', (), None)
        x = m.P1_BOARD_X if scope[-1] == '1P' else m.P2_BOARD_X
        cells = tuple(self.cell(image, x, index) for index in range(m.CELL_COUNT))
        positive = any(cell['positive'] for cell in cells)
        return Observation(scope, frame, call, 'POSITIVE' if positive else 'NOT_POSITIVE', cells,
            hashlib.sha256(image.tobytes()).hexdigest(), DROP_CAP if positive else None)

    def cell(self, image: Any, x: int, index: int) -> dict:
        d, m = self.detector, self.module
        cell = d._extract_cell(image, x, index)
        result = dict(slot=index, positive=False, cnn=None, confidence=None, ncc=None)
        if cell.size == 0 or d._patch_features(cell)['v_std'] < m.PRESENCE_V_STD_MIN:
            return result | dict(route='presence_rejected')
        patch = d._extract_center_patch(cell)
        if patch is None:
            return result | dict(route='missing_patch')
        kind, confidence = d._cnn.predict_class(patch)
        ncc = d._match_templates(cell)
        direct = math.isfinite(confidence) and confidence >= d._cnn_confidence_min
        positive = (direct and m.COUNT_TABLE[kind] >= DROP_CAP
                    and ncc is not None and m.COUNT_TABLE[ncc] >= DROP_CAP)
        return result | dict(positive=positive, cnn=kind, confidence=confidence,
            ncc=ncc, route='cnn_and_ncc' if direct else 'cnn_low_confidence')
