"""公開時刻の換算で未来への丸めと不正値の見逃しが起きないこと。"""
import math

import pytest

from src.phase_j.live_ex_hold import LiveExHold
from src.phase_j._live_ex_frozen.src.dl1k_columns import COLUMNS, LEARNING_COLUMNS

MEDIA_TIME = 339.03787907101827
CUTOFF = 1095.4666666666667


@pytest.mark.parametrize('media_fresh', [False, True])
@pytest.mark.parametrize('age', [0., .125])
def test_available_time_never_rounds_into_future(media_fresh: bool, age: float) -> None:
    assert MEDIA_TIME - CUTOFF + CUTOFF == math.nextafter(MEDIA_TIME, math.inf)
    hold = LiveExHold(media_fresh=media_fresh)
    key = dict(key=(1, 1, (0., None)), fight=True)
    source = dict(extra=dict.fromkeys(COLUMNS, 1.), input_t=MEDIA_TIME,
                  eligible_at=CUTOFF-age)
    value, reason = hold.available(CUTOFF, MEDIA_TIME, CUTOFF, key, source, 'available', 1)
    assert value is not None and reason == 'available'
    for side in ('p1', 'p2'):
        assert value['extra']['av_ex_' + side] == MEDIA_TIME-age
    assert all(value['extra'][name] == 1. for name in LEARNING_COLUMNS)
    assert source['extra']['av_ex_p1'] == 1.


@pytest.mark.parametrize('eligible_at', [math.nan, math.inf, -math.inf, CUTOFF+1.])
def test_invalid_available_time_is_still_rejected(eligible_at: float) -> None:
    hold = LiveExHold()
    source = dict(extra=dict.fromkeys(COLUMNS, 1.), input_t=MEDIA_TIME,
                  eligible_at=eligible_at)
    source['extra'].update(av_ex_p1=eligible_at, av_ex_p2=eligible_at)
    event = dict(group=(1, 1), t=MEDIA_TIME, row_key='strict', fight=True, reset={})
    with pytest.raises(ValueError, match='公開時刻が未来または不明'):
        hold.adopt(event, source, 'ok')
