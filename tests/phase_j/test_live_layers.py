"""C案は値の転載だけで判定を変更しない。"""
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from src.phase_j.contracts import DisplayLayers
from src.phase_j.live_layers import evaluation_layers
from src.phase_j.live_publish import initial_snapshot
from src.phase_j.validator import validate_snapshot
from tests.phase_j.test_live_publish import ASSETS


@pytest.mark.parametrize('value', [-0.1, 1.1, float('nan'), float('inf'), True, '0.5'])
def test_invalid_probability(value: object) -> None:
    with pytest.raises(ValueError):
        DisplayLayers(value, None, 0.5, False)


@pytest.mark.parametrize('reason', ['unknown_cells', 'observed_exceeds_prediction', 'confirmed_board_mismatch'])
def test_discard_reason(reason: str) -> None:
    assert DisplayLayers(0.4, None, 0.4, False, reason).prediction_discard_reason == reason
    with pytest.raises(ValueError):
        DisplayLayers(0.4, 0.8, 0.8, True, reason)


@pytest.mark.parametrize('source', ['G_fe', 'S1', 'S3', 'S3_landing', 'unavoidable_death'])
def test_extract_does_not_mutate(source: str) -> None:
    tracker = SimpleNamespace(source=source, probability=0.8, _static_probability=0.4)
    before = dict(vars(tracker))
    result = evaluation_layers(SimpleNamespace(tracker=tracker), 0.7)
    assert result['current_p1'] == 0.4 and result['displayed_p1'] == 0.7
    assert result['includes_prediction'] == (source in {'S3_landing', 'unavoidable_death'})
    assert result['prediction_discard_reason'] is None
    assert vars(tracker) == before


def test_validator_optional_and_correlated() -> None:
    snapshot = initial_snapshot(ASSETS).to_mapping()
    snapshot['evaluations']['display_layers'] = asdict(DisplayLayers(None, None, None, False))
    assert not any(issue.rule_id == 'R21' for issue in validate_snapshot(snapshot).issues)
    snapshot['evaluations']['display_layers']['displayed_p1'] = 0.8
    assert any(issue.rule_id == 'R21' for issue in validate_snapshot(snapshot).issues)


def test_unknown_reason_and_missing_prediction() -> None:
    with pytest.raises(ValueError):
        DisplayLayers(0.4, None, 0.8, True)
    with pytest.raises(ValueError):
        DisplayLayers(0.4, None, 0.4, False, 'invented')
