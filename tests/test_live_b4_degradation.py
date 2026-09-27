"""劣化差分の母数・状態・HSV循環の集計を検証する。"""
import numpy as np
from scripts.verify_live_b4_degradation import breakdown


def observations() -> dict:
    shape = (2, 2, 13, 6)
    return dict(boards=np.zeros(shape, dtype=int), stable=np.ones((2, 2), dtype=bool),
                states=np.full((2, 2), 'STABLE'), raw=np.zeros(shape, dtype=int),
                hsv=np.zeros((*shape, 3)), hsv_colors=np.zeros(shape, dtype=int))


def test_breakdown_uses_common_mask_and_keeps_hidden_cells() -> None:
    left, right = observations(), observations()
    right['boards'][0, 0, 0, 0] = 1
    right['boards'][0, 0, 1, 0] = 2
    right['boards'][1, 1, 1, 0] = 3
    mask = left['stable'].copy()
    mask[1, 1] = False
    report = breakdown(left, right, mask)
    assert report['cells'] == 234 and report['different_cells'] == 2
    assert report['hidden_differences'] == report['visible_differences'] == 1
    assert np.array(report['confusion']).sum() == 2
    assert np.array(report['by_side_row_col']).sum() == 2
    assert sum(report['frame_types'].values()) == 2


def test_hue_wrap_and_instantaneous_observation_are_separate() -> None:
    left, right = observations(), observations()
    right['boards'][0, 0, 1, 0] = 1
    left['hsv'][0, 0, 1, 0, 0] = 179
    right['hsv'][0, 0, 1, 0, 0] = 1
    report = breakdown(left, right)
    assert report['absolute_hsv_delta_percentiles'][0][0] == 2
    assert report['raw_disagreements_at_differences'] == 0
    assert report['hsv_disagreements_at_visible_differences'] == 0
