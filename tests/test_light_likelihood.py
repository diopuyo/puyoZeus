"""候補集合の下限と旗OFFの不変性を確認する。"""
from types import SimpleNamespace
import numpy as np
import pytest
from src.light_likelihood import CandidateEvidence, LOG_RATIO, probabilities, read_probabilities
from src.light_refind import physical_forecast
from src.fire_board import Forecast


def evidence() -> CandidateEvidence:
    board = np.zeros((13, 6), dtype=np.int8)
    other = board.copy()
    other[12, 0] = 1
    candidates = [SimpleNamespace(board=board, steps=(40, 320), final=board),
                  SimpleNamespace(board=other, steps=(40, 320, 900), final=other)]
    result = CandidateEvidence(candidates)
    result.frames, result.first_t, result.last_t = 10, 0., .3
    return result


def test_all_candidates_within_nine_are_kept() -> None:
    result = evidence()
    result.scores = np.array([-LOG_RATIO*10, 0.])
    result.select((40,))
    assert result.retained == [0, 1] and result.steps == (40, 320)
    assert result.final is None


def test_repeated_observations_do_not_multiply_evidence_margin() -> None:
    result = evidence()
    result.scores = np.array([-2., 0.])*10
    result.select((40,))
    assert result.retained == [0, 1]


def test_formula_filters_before_likelihood() -> None:
    result = evidence()
    result.scores = np.array([100., 0.])
    result.select((40, 320, 900))
    assert result.retained == [1] and result.steps == (40, 320, 900)


def test_existing_higher_forecast_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    lower = evidence()
    lower.select((40,))
    physical = SimpleNamespace(evidence=lower, active=True, rejected=False, chain_id=1.)
    original = Forecast(1., (40,), (40, 320, 900), 'refind', None, 1.)
    assert physical_forecast(SimpleNamespace(_light_physics=physical), original) is original


def test_ambiguous_final_board_is_not_invented(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    lower = evidence()
    lower.select((40,))
    physical = SimpleNamespace(evidence=lower, active=True, rejected=False, chain_id=1.)
    original = Forecast(1., (40,), (40,), 'none', None, 1.)
    result = physical_forecast(SimpleNamespace(_light_physics=physical), original)
    assert result.total == 360 and result.lower_bound and result.final_board is None
    assert original.source == 'none'


def test_off_does_not_read_cnn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('PUYO_LIGHT_ROBUST_CELLS', raising=False)
    assert read_probabilities(object(), object(), '1P') is None


def test_b3_publishes_score_without_an_uncertain_final_board() -> None:
    from src.b3_state import SideState
    episode = SimpleNamespace(steps=[(1.,1,40)])
    owner = SimpleNamespace(episode=episode)
    forecast = Forecast(1.,(40,),(40,320,900),'light_physics_lower',None,1.,lower_bound=True)
    SideState._publish_light_lower(owner,forecast,1.4)
    assert episode.prediction.score == 1260 and episode.prediction.length == 3
    assert episode.prediction.final is None and episode.forecast is forecast


@pytest.mark.parametrize('value', [None, [], np.ones((12,6,7)), np.full((12,6,7), np.nan)])
def test_invalid_probabilities_are_missing(value: object) -> None:
    assert probabilities(value) is None
