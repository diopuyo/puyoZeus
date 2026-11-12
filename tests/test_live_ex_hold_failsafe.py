"""配信の検査失敗だけを回復し、保持期限と厳密な検査を維持する。"""
import logging
import json
import math
from queue import Queue
from types import SimpleNamespace

import pytest

from src.phase_j import live_ex_hold_diagnostics as diagnostics
from src.phase_j.live_death_retr import CURRENT, capture
from src.phase_j.live_ex_hold import LiveExHold
from src.phase_j.live_telemetry import RuntimeTelemetry
from tests.test_live_24h_hold import source


@pytest.mark.parametrize('bad_time', [math.nan, math.inf, -math.inf, 11.])
@pytest.mark.parametrize('seed', [False, True])
def test_capture_finishes_and_hold_expires(monkeypatch: pytest.MonkeyPatch,
                                         bad_time: float, seed: bool) -> None:
    monkeypatch.setenv('PUYO_LIVE_DEATH_CHECK_RETR', '1')
    item = source()
    item.update(input_t=2., eligible_at=10.)
    item['extra'].update(retR_r_p2=-1., retR_c_p2=17.)
    worker = SimpleNamespace(epoch=1, fresh_sec=2.,
        key=dict(key=('board', 1), fight=True, input_t=2., resets=[]),
        available=lambda *a, **k: (item, 'available'))
    unified = SimpleNamespace(epoch=1, game=1, worker=worker, clock=lambda: 10.)
    b3 = SimpleNamespace(live24h=unified, epoch=1, generation=1, adopted_game=1)
    frame = dict(t_sec=2., game_idx=1, adopted_game=1)
    notice = SimpleNamespace(captured_at=10., b3=(frame, [], [], 1))
    if seed:
        capture(b3, notice)
    item['eligible_at'] = bad_time
    frame['t_sec'] = 2.25
    capture(b3, notice)
    value = CURRENT.get()['values'][1][0]
    assert value == -1. if seed else math.isnan(value)
    frame['t_sec'] = 2.51
    capture(b3, notice)
    assert math.isnan(CURRENT.get()['values'][1][0])
    item.update(input_t=3., eligible_at=10.)
    frame['t_sec'] = 3.
    capture(b3, notice)
    assert CURRENT.get()['values'][1] == (-1., 17.)


def test_rejection_counts_reach_runtime_and_warnings_are_thinned(
        monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(diagnostics, '_COUNTS', diagnostics.Counter())
    hold = LiveExHold(media_fresh=True)
    key = dict(key=(1, 1), fight=True)
    item = source()
    item['eligible_at'] = math.nan
    with caplog.at_level(logging.WARNING):
        for index in range(9):
            hold.available(10., index / 100, 10., key, item, 'available', 1)
    bridge = SimpleNamespace(queue=Queue(), batch_remaining=0)
    sink = SimpleNamespace(publisher=SimpleNamespace(input_calibration='ok'))
    saved = RuntimeTelemetry(bridge, sink, None).snapshot()
    counts = saved['ex_hold_validation_rejections']
    assert counts['total'] == 9
    assert list(counts['by_reason'].values()) == [9]
    assert len(caplog.records) == 4  # 1、2、4、8 件目だけ警告する。


def test_unrelated_value_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    hold = LiveExHold()
    def fail(row: dict) -> dict:
        raise ValueError('検査以外の失敗')
    monkeypatch.setattr(hold.policy, 'apply', fail)
    with pytest.raises(ValueError, match='検査以外の失敗'):
        hold.available(10.2, .2, 10.2, dict(key=(1, 1), fight=True), source(), 'available', 1)


@pytest.mark.parametrize('bad_input', [None, math.nan, math.inf, -math.inf, -10., 10.])
def test_invalid_freshness_skips_adoption(bad_input: float | None) -> None:
    item = dict(source(), input_t=bad_input)
    value, _ = LiveExHold().available(10.2, .2, 10.2,
        dict(key=(1, 1), fight=True), item, 'available', 1)
    assert value is None


@pytest.mark.parametrize('media_fresh', [False, True])
@pytest.mark.parametrize('reset', ['none', 'p1', 'game', 'outside'])
def test_rejected_row_matches_transport_missing(media_fresh: bool, reset: str) -> None:
    """再採用で期限を延ばさず、解除も従来の欠測と同じように適用する。"""
    actual, expected = LiveExHold(media_fresh), LiveExHold(media_fresh)
    key = dict(key=(1, 1), fight=True, resets=[])
    for hold in (actual, expected):
        hold.available(10.2, .2, 10.2, key, source(), 'available', 1)
    if reset == 'p1':
        key['resets'] = [dict(side='p1')]
    elif reset == 'game':
        key['key'] = (1, 2)
    elif reset == 'outside':
        key['fight'] = False
    bad = source()
    bad['eligible_at'] = math.inf
    for t in (.3, .5, .7, .71):
        for hold in (actual, expected):
            hold.observe(key)
        got = actual.available(t + 10, t, t + 10, key, bad, 'available', 1)
        want = expected.available(t + 10, t, t + 10, key, None, 'not_yet', 1)
        # 未表示の理由文字列を除き、公開値・監査・世代を丸ごと比較する。
        assert json.dumps(got[0], sort_keys=True) == json.dumps(want[0], sort_keys=True)
    assert got[0] is None
