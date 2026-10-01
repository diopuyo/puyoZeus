"""案Cの全列挙境界・手数・欠測・側交換を検証する。"""
from dataclasses import replace

import numpy as np
import pytest

from src import prefire_threat_features as features
from src import prefire_v5_search as base
from src.indicators_v2 import SEC_PER_HAND


def position() -> base.Position:
    """有効な探索前入力。"""
    return base.Position(bytes(features.CAPACITY), (1, 2, 3, 4, 1, 2))


def terminal(score: int, hands: int, chains: int = 1) -> base.Position:
    """物理列挙器の出力を手数別に作る。"""
    return replace(position(), score=score, consumed=hands, chains=chains)


def test_disabled_does_not_enumerate(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object) -> None:
        raise AssertionError('OFFでは計算しない')
    monkeypatch.setattr(features, '_options', forbidden)
    assert features.threat_features((), np.nan) is None


@pytest.mark.parametrize('queue', [(), (0,)*6, (1, 2, 3, 4), (1, 2, 3, 4, 0, 0)])
def test_missing_is_nan_not_zero(queue: tuple[int, ...]) -> None:
    p = replace(position(), queue=queue)
    result = features.threat_features((p, p), 0., enabled=True)
    assert result is not None
    for side in result:
        assert np.isnan(side.raw()[:len(features.RAW_COLUMNS)]).all()
        assert side.attack_missing and side.counter_missing


def test_maximum_then_shortest_and_response_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    options = ((terminal(700, 3), terminal(700, 2), terminal(70, 1)),
               (terminal(420, 1), terminal(1400, 3)))
    monkeypatch.setattr(features, '_options', lambda p, s: options[s])
    monkeypatch.setattr(features, 'remaining_hands', lambda *args: 1)
    result = features.threat_features((position(), position()), 0., True)
    assert result[0] == features.Threat(10., 2., 6., 4., 4/6, False, False, False)


def test_counter_larger_than_attack_has_zero_unreturned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(features, '_options', lambda p, s: (terminal(70*(s+1), 1),))
    result = features.threat_features((position(), position()), 0., True)
    assert result[0].difference == 0
    assert result[1].difference == 1


def test_missing_receiver_does_not_erase_known_attack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(features, '_options', lambda p, s: None if s else (terminal(700, 1),))
    result = features.threat_features((position(), position()), 0., True)
    assert result[0].attack == 10 and not result[0].attack_missing
    assert result[0].counter_missing and np.isnan(result[0].difference)


def test_no_fire_is_measured_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(features, '_options', lambda p, s: (terminal(0, 3, 0),))
    result = features.threat_features((position(), position()), 0., True)
    assert np.array_equal(result[0].raw(), np.zeros(len(features.SIDE_COLUMNS)))


def test_horizon_beyond_three_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(features, '_options', lambda p, s: (terminal(700, 3, 10),))
    result = features.threat_features((position(), position()), 0., True)
    assert all(t.beyond_known_horizon for t in result)


@pytest.mark.parametrize('elapsed', [0., 95., 96., 120., 240.])
def test_margin_matches_existing_table(monkeypatch: pytest.MonkeyPatch, elapsed: float) -> None:
    candidate = terminal(7000, 3)
    monkeypatch.setattr(features, '_options', lambda p, s: (candidate,))
    result = features.threat_features((position(), position()), elapsed, True)
    expected = base.sim.send_ojama(candidate.score, elapsed+2*SEC_PER_HAND)
    assert result[0].attack == result[1].counter == expected


@pytest.mark.parametrize('rows', [0., .5, 1., 5., 13., 100.])
def test_encoding_range_bins_and_swapping(rows: float) -> None:
    a = features.Threat(rows*6, 2., 0., rows*6, rows, False, False, False)
    b = features.Threat(12., 1., 6., 6., 1., False, False, True)
    x = features.encode((a, b)).reshape(3, -1)
    y = features.encode((a, b), 1).reshape(3, -1)
    assert np.all((x[:2] >= 0) & (x[:2] <= 1))
    np.testing.assert_array_equal(x[0], y[1])
    np.testing.assert_array_equal(x[1], y[0])
    np.testing.assert_array_equal(x[2], -y[2])
    np.testing.assert_array_equal(x[0, len(features.SIDE_COLUMNS):], rows >= np.array(features.ROW_BOUNDS))
    assert x.size == len(features.COLUMNS)


@pytest.mark.parametrize('elapsed', [-1., np.nan, np.inf])
def test_reject_invalid_elapsed(elapsed: float) -> None:
    with pytest.raises(ValueError):
        features.threat_features((position(), position()), elapsed, True)


@pytest.mark.parametrize('side', [-1, 2, 3])
def test_reject_invalid_side(side: int) -> None:
    with pytest.raises(ValueError):
        features.encode((features.Threat(), features.Threat()), side)


def test_encoding_preserves_missing() -> None:
    x = features.encode((features.Threat(), features.Threat())).reshape(3, -1)
    assert np.isnan(x[:, :len(features.RAW_COLUMNS)]).all()
    assert np.all(x[:2, len(features.RAW_COLUMNS):len(features.RAW_COLUMNS)+2] == 1)


@pytest.mark.parametrize('changes', [{'board': b''}, {'score': 1}, {'consumed': 1}])
def test_reject_noninitial_state(changes: dict) -> None:
    with pytest.raises(ValueError):
        features.threat_features((replace(position(), **changes), position()), 0., True)


@pytest.mark.parametrize('kind', ['unknown', 'dead', 'already_firing'])
def test_invalid_stable_board_is_missing(kind: str) -> None:
    grid = np.zeros((13, 6), np.int8)
    if kind == 'unknown':
        grid[-1, 0] = 10
    elif kind == 'dead':
        grid[1:, 2] = 9
    else:
        grid[-1, :4] = 1
    p = replace(position(), board=grid.tobytes())
    result = features.threat_features((p, p), 0., True)
    assert all(t.attack_missing for t in result)


def test_real_enumeration_matches_terminal_reference() -> None:
    grid = np.zeros((13, 6), np.int8)
    grid[-1, :3] = 1
    p = replace(position(), board=grid.tobytes())
    # 重複統合前の全終端を直接比較し、最小手数と最大送り量を検算する。
    options = features.exact.candidates(p, 3, side=0)
    fires = [t for t in options if t.chains]
    expected = max((base.sim.send_ojama(t.score, (t.consumed-1)*SEC_PER_HAND),
                    -t.consumed) for t in fires)
    result = features.threat_features((p, p), 0., True)
    assert result[0].attack == expected[0]
    assert result[0].attack_hands == -expected[1]
    assert result[0] == result[1]


@pytest.mark.parametrize('queue', [(1, 2, 3, 4, 1, 2), (1,)*6, (2, 1, 4, 3, 2, 1)])
def test_compact_native_matches_full_enumeration_maxima(queue: tuple[int, ...]) -> None:
    engine = base.native._native
    if engine is None or not hasattr(engine, 'prefire_threat_maxima_py'):
        pytest.skip('新しいnative拡張未ビルド')
    grid = np.zeros((13, 6), np.int8)
    grid[-1, :3] = 1
    p = replace(position(), board=grid.tobytes(), queue=queue)
    complete = features.exact.candidates(p, 3)
    expected = []
    for hands in range(1, 4):
        first = next((t for t in complete if t.chains and t.consumed == hands), None)
        if first is not None:
            expected.append((first.score, first.chains, first.consumed))
    compact = engine.prefire_threat_maxima_py(list(p.board),
        [queue[i:i+2] for i in range(0, 6, 2)], base.GHOST_CHAIN_RULE_ENABLED)
    assert compact == expected
