"""保存再生・通知worker・分離更新でE35の観測時系列とH1′境界を照合する。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

from src.exchange_event_overlay import ExchangeEventOverlay
from src.exchange_event_record import ExchangeEventRecorder, read_records
from src.exchange_history_integrity import MatchHistoryIntegrity
from src.exchange_post_counter_bound import PostCounterDeathBound
from src.phase_j.live_evaluation import SplitExchangeOverlay
from src.phase_j.live_notification_eval import NotificationExchangeOverlay
from tests.test_exchange_event_overlay import Signals, build_static, result
from tests.test_exchange_event_tracker import Models


def make_overlay(cls: type) -> Any:
    """重いモデルだけを固定し、trackerと観測照合は実装を使用する。"""
    overlay = cls(Models(), build_static, Signals, lambda b, q: .5)
    overlay._landing_projection.post_counter_bound = PostCounterDeathBound(
        current_puyo_integrity_v2c=True, completion_recent_integrity=True,
        strict_integrity=True, cap_completed=True)
    overlay._history_integrity = MatchHistoryIntegrity(boundary_score_start=True, strict_integrity=True)
    overlay.tracker.history_integrity = overlay._history_integrity
    return overlay


def notification(stamp: float, stage: int) -> tuple:
    """実読した式と、その段に対応する表示得点を同じ通知へ載せる。"""
    observed = result(stamp)
    total = {0: 0, 1: 40, 2: 360, 3: 1000}[stage]
    if stage:
        observed.p1.chain_event.mechanism = 'formula_read'
        observed.p1.chain_event.chain_count = stage
        observed.p1.chain_event.total_score = total
        observed.p1.score = 100+total
    snapshot = NS(net_balance_capped=0, forecast_p1=0,
                  total_dropped_to_p1=0, total_dropped_to_p2=0)
    final = NS(finalized_count_p1=0, finalized_count_p2=0,
               chain_total_score_p1=0, chain_total_score_p2=0)
    return (observed, snapshot, final, stamp, 0,
            (total if stage else None, None), (observed.p1.score, observed.p2.score),
            (bool(stage), False))


@pytest.mark.parametrize('cls', [NotificationExchangeOverlay, SplitExchangeOverlay])
def test_replay_and_live_observe_same_formula_and_display(cls: type,
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """tracker更新前の照合、result欠落、時刻欠落はどれもこの試験で落ちる。"""
    pair = [make_overlay(ExchangeEventOverlay), make_overlay(cls)]
    seen: dict[int, list] = {id(overlay): [] for overlay in pair}
    original = PostCounterDeathBound.refresh_current_observation

    def refresh(self: Any, overlay: Any, observed: Any = None, stamp: float | None = None) -> None:
        assert observed is not None and stamp is not None
        original(self, overlay, observed, stamp)
        seen[id(overlay)].append((stamp, deepcopy(self.step_observations)))

    monkeypatch.setattr(PostCounterDeathBound, 'refresh_current_observation', refresh)
    live = [notification(stamp, stage) for stamp, stage in
            ((0., 0), (1., 1), (1.1, 1), (1.2, 2), (1.3, 2), (1.4, 3))]
    path = tmp_path/'observations.jsonl.gz'
    recorder = ExchangeEventRecorder(path, '時系列試験', False, Path('models'))
    for args in live:
        recorder.update(*args)
    recorder.close()
    saved = [row['args'] for row in read_records(path) if row['kind'] == 'update']
    for offline_args, live_args in zip(saved, live, strict=True):
        pair[0].update(*offline_args)
        pair[1].update(*live_args)
    assert seen[id(pair[0])] == seen[id(pair[1])]
    entries = pair[1]._landing_projection.post_counter_bound.step_observations
    stages = next(iter(entries.values()))['stages']
    assert stages[1] == dict(formula=40, display=40)
    assert stages[2] == dict(formula=360, display=360)
    assert stages[3] == dict(formula=1000, display=1000)
    assert stages[2]['display']-stages[1]['display'] == 320


@pytest.mark.parametrize('cls', [NotificationExchangeOverlay, SplitExchangeOverlay])
def test_replay_and_live_h1_boundary_after_clear(cls: type) -> None:
    """実updateで試合を切り替え、全消去後も前試合の得点証拠を保持する。"""
    pair = [make_overlay(ExchangeEventOverlay), make_overlay(cls)]
    for game in (0, 1):
        args = list(notification(0., 0))
        args[3], args[4] = float(game), game
        for side in (args[0].p1, args[0].p2):
            side.score = 100 if game == 0 else 0
        args[6] = (args[0].p1.score, args[0].p2.score)
        for overlay in pair:
            overlay.update(*deepcopy(args))
    reference = pair[0]._history_integrity.summary()
    assert pair[1]._history_integrity.summary() == reference
    assert set(pair[1]._history_integrity.games) == {1}
    assert pair[1]._history_integrity.evidence()['history_complete']
