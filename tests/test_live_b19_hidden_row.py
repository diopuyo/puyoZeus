"""終了済み撃ち合いの発火入力欠測が通知評価を停止させないことを確認する。"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import exchange_hidden_row_probability as probability
from src.exchange_event_landing import logit_mean
from src.exchange_event_tracker import ExchangeChainRecord, ExchangeRecord
from src.phase_j.live_notification_eval import NotificationExchangeOverlay, latest_display
from tests.test_exchange_event_overlay import Signals, build_static
from tests.test_exchange_event_tracker import Models
from tests.test_live_b16 import inputs, m0


@pytest.mark.parametrize('smoothing', [False, True])
def test_notification_after_exchange_without_firing(
        monkeypatch: pytest.MonkeyPatch, smoothing: bool) -> None:
    """通知入口→着弾評価→加重なしへの復帰→表示更新を実物で通す。"""
    live = NotificationExchangeOverlay(Models(), build_static, Signals, m0,
                                       switch_smoothing=smoothing)
    live.update(*inputs(0.))
    chain = ExchangeChainRecord('1P', 1, 0., 0., score_delta=70.)
    base = dict(source='S3', p1=.8, t_sec=0.)
    record = ExchangeRecord(1, 1, 0., chains=[chain], values=[base])
    projection = live._landing_projection
    projection.death_record = record
    entry = dict(options=[dict(score=70., weight=1.)])
    active = Mock(return_value=entry)
    live.tracker.hidden_row_belief = SimpleNamespace(active=active)
    # 学習モデルの値だけ固定し、問題の評価分岐と通知の状態遷移は置換しない。
    monkeypatch.setattr(projection, '_probability_inputs', Mock(return_value=(.6, {})))
    monkeypatch.setattr(live.tracker, 'ready_for_static', Mock(return_value=False))
    weighted = Mock(wraps=probability.weighted_landing)
    monkeypatch.setattr(probability, 'weighted_landing', weighted)

    assert live.tracker.current is None and live.tracker.firing is None
    live.update(*inputs(.1))

    weighted.assert_called_once()
    active.assert_called_once_with(chain)
    assert record.values[-1]['source'] == 'S3_landing'
    assert record.values[-1]['p1'] == pytest.approx(logit_mean(.8, .6))
    assert live.tracker.probability == record.values[-1]['p1']
    assert live.display is not None
    assert latest_display(live, 0., .5) == live.display
