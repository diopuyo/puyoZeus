"""値と信頼度の差を区別して、認識から表示まで同一フレームで採取する。"""
from __future__ import annotations

from dataclasses import asdict
from enum import Enum
import math
from typing import Any

import numpy as np


def plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def observe(pipe: Any, result: Any, packet: tuple, features: dict | None,
            display: Any, frame: np.ndarray, probes: dict) -> dict:
    ocr = pipe._score_ocr
    details = [ocr.read_side_detail(frame, side) for side in ('1P', '2P')]
    formulas = [asdict(ocr.read_formula_side(frame, side, d[2], d[3]))
                for side, d in zip(('1P', '2P'), details)]
    values = dict(scores=[d[0] for d in details], digits=[d[2] for d in details],
        formulas=[{k: f[k] for k in ('valid', 'left', 'right', 'product')} for f in formulas],
        confidence=[d[3] for d in details], formula_detail=formulas,
        game=dict(index=packet[0]['game_idx'], active=bool(packet[0]['match_active']),
                  adopted=packet[0].get('adopted_game')),
        winner=dict(dead=result.confirmed_dead_sides, locked=result.match_end_locked),
        all_clear=[getattr(pipe, f'_all_clear_pending_{s}') for s in ('1p', '2p')],
        death=[s.confirmed_board.is_dead() if s.confirmed_board is not None else None
               for s in (result.p1, result.p2)],
        states=[s.state.name for s in (result.p1, result.p2)],
        boards=[s['confirmed_board'] for s in packet[0]['sides']],
        warning=[s['icon'] for s in packet[0]['sides']],
        b3_features=features, probability=display.shown[1],
        displayed_probability=display.shown[1] if display.raw is not None else None,
        display_text=math.floor(display.shown[1]*100+.5) if display.raw is not None else None,
        display_source=display.source, raw_probability=display.raw,
        ui=ui_values(probes, frame))
    return plain(values)


def ui_values(probes: dict, frame: np.ndarray) -> dict:
    """盤面へ採用されなかったロゴ・テロップの差も記録する。"""
    from src.review_terminal_outcome import _template_score
    result = {key: asdict(probes[key].detect(frame)) for key in ('match_end', 'telop', 'zero')}
    panel = probes['panel'].detect(frame)
    result['panel'] = dict(present=panel.present, score=panel.score)
    result['chain_count'] = [asdict(probes['chain'].read_side(frame, side)) for side in ('1P', '2P')]
    # 確定後の勝敗だけでなく、ラッチの原因となる両側の観測値も保存する。
    result.update({f'death_{idx}': probes['death']._score(frame, None, idx)
                   for idx in range(2)})
    result['review'] = {f'{mode}_{name}_{side}': _template_score(frame, probes[mode], name, side)
        for mode in ('review', 'review_crop') for name in ('match_end_batan', 'match_end_yatta')
        for side in ('p1', 'p2')}
    return result
