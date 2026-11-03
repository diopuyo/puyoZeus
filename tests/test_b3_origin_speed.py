"""発火起点の最適化が候補順・重み・全段・同点採用を変えないことを保証する。"""
import json
from pathlib import Path

import numpy as np
import pytest

from src import production_config, puyo_core_bridge as native
from src.b3_chain_prediction import origin_candidates
from src.board import Board
from src.chain import ChainSimulator
from src.exchange_prefire_candidates import enumerate_candidates, pairs_for, statistics

FIXTURES = json.loads((Path(__file__).parent / 'fixtures/b3_origin_speed.json').read_text())
pytestmark = pytest.mark.skipif(not native.NATIVE_AVAILABLE or
    not hasattr(native._native, 'b3_origin_completions_py'), reason='新しいpuyo_coreが必要')


@pytest.mark.parametrize('case', FIXTURES)
def test_complete_ordered_options_equal_reference(case: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    grid = np.asarray(case['board'], dtype=np.int8)
    colors, queue = tuple(case['colors']), tuple(case['queue'])
    monkeypatch.setattr(production_config, 'B3_ORIGIN_NATIVE_ENABLED', False)
    expected = origin_candidates(grid, colors, queue)
    monkeypatch.setattr(production_config, 'B3_ORIGIN_NATIVE_ENABLED', True)
    actual = origin_candidates(grid, colors, queue)
    assert actual == expected
    for max_chain in (False, True):
        assert statistics(actual[0], 0., max_chain=max_chain) == statistics(expected[0], 0., max_chain=max_chain)


@pytest.mark.parametrize('hidden', (False, True))
def test_parallel_serial_and_reference_trials(hidden: bool) -> None:
    board = Board.from_list(FIXTURES[0]['board'])
    colors, queue = tuple(FIXTURES[0]['colors']), tuple(FIXTURES[0]['queue'])
    pairs = pairs_for(colors, queue)
    args = (board._grid.ravel().tolist(), pairs, hidden)
    serial = native._native.b3_origin_candidates_py(*args, False)
    assert serial == native._native.b3_origin_candidates_py(*args, True)
    expected, trials = enumerate_candidates(board, colors, queue,
        ChainSimulator(exclude_hidden_row_from_pop=hidden, fast_groups=True))
    assert trials == serial[1]
    assert [weight for _, weight in serial[0]] == [v['weight'] for v in expected]


def test_result_mutation_does_not_affect_next_call() -> None:
    case = FIXTURES[0]
    args = (np.asarray(case['board'], dtype=np.int8), tuple(case['colors']), tuple(case['queue']))
    first, reason = origin_candidates(*args)
    expected = json.dumps(first)
    first[0]['weight'] = -1
    second, second_reason = origin_candidates(*args)
    assert json.dumps(second) == expected and reason == second_reason


@pytest.mark.parametrize('seed', range(3))
def test_gaps_hidden_row_and_ojama_preserve_completion(seed: int) -> None:
    """未整地盤面でも、候補判定と段別完走の異なる旧規則をそのまま維持する。"""
    rng = np.random.default_rng(seed)
    grid = np.zeros((13, 6), dtype=np.int8)
    grid[-5:] = rng.choice((0, 1, 2, 3, 4, 9), size=(5, 6))
    grid[:3, 0] = (9, 1, 1)
    board = Board.from_list(grid.tolist())
    colors, queue = (1, 2, 3, 4), (1, 5, 0, 0)
    for hidden in (False, True):
        expected = enumerate_candidates(board, colors, queue,
            ChainSimulator(exclude_hidden_row_from_pop=hidden, fast_groups=True))
        actual = native.b3_origin_candidates(board, pairs_for(colors, queue), hidden)
        assert actual == expected
