"""非同期結果の世代・境界・期限と、死亡表示/SSE時刻の接続を確認する。"""
from io import StringIO
from types import SimpleNamespace

import pytest

from src.b3_live_display import B3Display
from src.phase_j.live_b3 import LiveB3
from src.phase_j.live_publish import initial_snapshot, result_snapshot
from src.phase_j.validator import validate_snapshot
from tests.phase_j.test_live_publish import ASSETS, row


def wrapper() -> LiveB3:
    live = LiveB3.__new__(LiveB3)
    live.display = B3Display()
    live.display.boundary(1)
    live.generation, live.result_frame = 2, -1
    live.audit = StringIO()
    return live


def result(**changes: object) -> dict:
    return dict(dict(epoch=2, game=1, frame=30, t_sec=1., mode=0, error=None,
        raw_probability=.8, timings={}, state_timings={}, completed_at=0.,
        trigger=1, state_wait_ms=0., compute_wait_ms=0., captured_at=0.), **changes)


@pytest.mark.parametrize('changes', [dict(epoch=1), dict(game=0), dict(frame=61),
    dict(t_sec=-1.), dict(t_sec=3.)])
def test_invalid_result_is_not_displayed(changes: dict) -> None:
    live = wrapper()
    live._accept_result(result(**changes), dict(game_idx=1, frame=60, t_sec=2.))
    assert live.display.probability is None
    assert '"accepted": false' in live.audit.getvalue()


def test_reversed_result_is_rejected() -> None:
    live = wrapper()
    frame = dict(game_idx=1, frame=60, t_sec=2.)
    live._accept_result(result(frame=40, raw_probability=.7), frame)
    live._accept_result(result(frame=30), frame)
    assert live.display.probability == .7


def test_failed_result_expires_after_half_second() -> None:
    live = wrapper()
    frame = dict(game_idx=1, frame=60, t_sec=1.4)
    live._accept_result(result(), frame)
    live._accept_result(result(frame=40, error='失敗'), frame)
    assert live.display.update(1.4)[1] == .8
    assert live.display.update(1.6) == (0., .5)


def test_pending_does_not_stop_reference_death() -> None:
    live = wrapper()
    live.physical, live.physical_count = set(), 0
    overlay = SimpleNamespace(b3_reference=dict(source='confirmed_death', probability=1., physical=[]))
    live._display(dict(t_sec=1., match_active=True), overlay)
    assert live.display.shown == (100., 1.)


def test_source_time_is_published_and_dto_valid() -> None:
    live = wrapper()
    live._accept_result(result(frame=1), dict(game_idx=1, frame=2, t_sec=1.5))
    live.display.update(1.5)
    data = dict(row(), frame=2, t_sec=1.5)
    live.decorate(data)
    snapshot = result_snapshot(initial_snapshot(ASSETS), data, 2., 2, None)
    assert validate_snapshot(snapshot).is_valid
    assert snapshot.display['message'] == 'B3 基準 1.000秒'
    assert snapshot.evaluations['practical']['input_generation'] == 1
    assert snapshot.evaluations['practical']['calculation_age_ms'] == 500.


def test_update_only_submits_and_polls() -> None:
    live = wrapper()
    calls = []
    live.async_failed = False
    live.async_engine = SimpleNamespace(poll=lambda: [], submit=lambda *a, **k: calls.append(a))
    live._async_update(dict(frame=1), (), (), SimpleNamespace(captured_at=0.))
    assert len(calls) == 1


def test_worker_failure_stops_submission_without_stopping_display() -> None:
    live = wrapper()
    live.async_failed = False
    live.async_engine = SimpleNamespace(poll=lambda: [{'fatal': '停止'}])
    live._async_update({}, (), (), SimpleNamespace(captured_at=0.))
    assert live.async_failed and live.display.failed


def test_adopted_start_invalidates_old_result_in_same_observed_game() -> None:
    """得点側の番号が変わらなくても、採用開始で旧結果を失効させる。"""
    live = wrapper()
    live.epoch, live.adopted_game = 1, 10
    live.physical, live.physical_count = {0.5}, 1
    live.async_failed = False
    calls = []
    live.async_engine = SimpleNamespace(poll=lambda: [result()],
        submit=lambda *a, **kw: calls.append(a))
    frame = dict(frame=60, t_sec=2., game_idx=1, adopted_game=11, match_active=True,
                 sides=[dict(state='STABLE'), dict(state='STABLE')])
    notice = SimpleNamespace(b3=(frame, (), (), 1), captured_at=0.)
    overlay = SimpleNamespace(b3_reference=dict(source='static', probability=.5, physical=[]))
    live.update(notice, overlay)
    assert live.generation == 3 and live.display.raw is None
    assert live.result_frame == -1 and live.physical_count == 0
    assert calls[0][3] == 3


def test_delayed_prediction_cannot_rewind_screen_stage() -> None:
    """旧フレームの推論が到着しても表示段の時刻と値は最新観測を保つ。"""
    from tests.test_b3_chain_display import raw
    from src.b3_contents import contents_from_row
    from tests.test_b3_contents import feature_row
    live = wrapper()
    frame = dict(game_idx=1, frame=59, t_sec=1.97, match_active=True,
                 sides=[raw(14), raw(5)])
    live.display.chain.observe(frame)
    frame.update(frame=60, t_sec=2.)
    live.display.chain.observe(frame)
    contents = contents_from_row(feature_row())
    live._accept_result(result(contents=contents), frame)
    live.display.update(2.)
    data = {}
    live.decorate(data)
    actual = data['b3_contents']
    assert actual['p1']['chain_display_step'] == 14
    assert actual['chain_display_frame'] == 60
    assert actual['p1']['chain_pred_l'] == contents['p1']['chain_pred_l']
    live.display.boundary(2)
    assert live.display.chain.sides[0].value is None
