"""再投入台はゲーム境界・時限解除を実writerへ通す。"""
from __future__ import annotations

import pytest

from scripts.replay_terminal_hold_20261012 import replay, durations, safety
from tests.phase_j.test_live_terminal_hold import terminal
from tests.phase_j.test_live_publish import row


@pytest.mark.parametrize('cut', [.5, 2., 4.999, 5., 6.])
def test_replay_integrates_until_boundary_or_deadline(cut: float) -> None:
    rows = [(10., 'evaluation', terminal()),
        (10.1, 'evaluation', dict(row(), match_active=False)),
        (10.+cut, 'evaluation', dict(row(), game=1, frame=2))]
    records = replay(rows, True)
    assert durations(records)[0] == pytest.approx(min(cut, 5.))
    result = safety(records, {0: 10., 1: 10.+cut})
    assert result['next_games'] == 1
    assert result['first_publication_extremes'] == result['early_extremes'] == 0
    assert result['held_wrong_game'] == 0


def test_replay_rescues_null_prefix_without_carrying_previous_terminal() -> None:
    """通常評価のない番号を救済し、その直後の旧確定の持越しは防ぐ。"""
    rows = [(10., 'evaluation', row()),
        (11., 'evaluation', dict(row('null'), game=1, frame=2, raw_probability=None)),
        (12.5, 'evaluation', dict(terminal(game=1), frame=3)),
        (15., 'evaluation', dict(terminal(game=2), frame=4))]
    records = replay(rows, True)
    assert durations(records)[1] == pytest.approx(2.5)
    assert durations(records)[2] == 0.
    starts = {0: 10., 1: 11., 2: 15.}
    result = safety(records, starts, first_seconds=1.)
    assert result['next_games'] == 2
    assert result['first_publication_extremes'] == result['early_extremes'] == 0
    assert safety(records, starts)['early_extreme_games'] == [1]


def test_final_confirmation_is_measured_until_five_second_expiry() -> None:
    """記録末尾に次試合がなくても、5秒全体と期限切れ公開を測る。"""
    records = replay([(10., 'evaluation', terminal())], True)
    assert durations(records)[0] == pytest.approx(5.)
    assert records[-1]['p1'] is None
