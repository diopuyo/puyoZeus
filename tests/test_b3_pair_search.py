"""B3専用2個組探索の契約。既存指標の意味論から独立して検証する。"""
from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from src import indicators_v2 as iv
from src import puyo_core_bridge as native
from src.b3_search import _SIM, clear_cache, ideal_fire_search, SEARCH_VERSION, search_manifest
from src.board import Board

COLORS = (1, 2, 3, 4)
OJAMA = 9
DEATH_COL = 2
RESCUE_SCORE = 100


@pytest.mark.parametrize('resolve', [False, True])
@pytest.mark.parametrize('color, score, done', [(1, RESCUE_SCORE, 1), (2, 0, 0)])
def test_death_is_checked_after_resolving_chain(resolve: bool, color: int, score: int, done: int) -> None:
    """配置直後は全手が窒息。赤赤だけ連鎖で生存し、その得点をQ候補に残す。"""
    grid = np.full((13, 6), OJAMA, dtype=np.int8)
    grid[:3, DEATH_COL] = 0
    grid[3:6, DEATH_COL] = 1
    result = ideal_fire_search(grid, (color, color), (1, 2), hmax=1,
                               resolve_before_death=resolve)
    assert result.scores == (score,)
    assert result.done_hands == done
    assert result.beam_died == (done == 0)


def test_rules_manifest_v2() -> None:
    manifest = search_manifest()
    assert manifest['version'] == 'b3_pair_potential_v2'
    assert manifest['death_check'] == 'after_chain'
    assert manifest['hidden_row_ojama_pop'] is False
    assert (manifest['beam_width'], manifest['potential_shortlist'], manifest['hmax']) == (4, 12, 20)


@pytest.mark.parametrize('queue', [(), (1, 2), (1, 1), (0, 2), (3, 0)])
@pytest.mark.parametrize('seed', [11, 21, 31])
def test_one_hand_matches_exhaustive_legal_pairs(queue: tuple[int, ...], seed: int) -> None:
    """未知・部分既知・同色・異色の全合法配置との照合。"""
    rng = np.random.default_rng(seed)
    grid = np.zeros((13, 6), dtype=np.int8)
    grid[-3:, :] = rng.choice(COLORS, size=(3, 6))
    board = Board.from_list(grid.tolist())
    q = (queue + (0, 0))[:2]
    scores = []
    for pair in product(COLORS, repeat=2):
        if any(known and known != value for known, value in zip(q, pair)):
            continue
        scores.extend(s for s, _, _ in iv._near_future_known_expand(
            [(0, board)], pair, _SIM, use_exact_score=True))
    result = ideal_fire_search(grid, queue, COLORS, hmax=1)
    assert result.scores == (max(scores),)


def test_potential_is_not_counted_as_fired_score() -> None:
    grid = np.zeros((13, 6), dtype=np.int8)
    grid[-1, 0] = 1
    result = ideal_fire_search(grid, (1, 1), (1,), hmax=2)
    assert result.scores[0] == 0
    assert result.scores[1] > 0


@pytest.mark.parametrize('shortlist', [0, 1, 4])
def test_unknown_is_pair_after_next2(shortlist: int) -> None:
    grid = np.zeros((13, 6), dtype=np.int8)
    # 異なる色の2組を独立列へ積み、3手目の未知組で4個を揃えられる。
    result = ideal_fire_search(grid, (1, 1, 2, 2), (1, 2), hmax=3,
                               potential_shortlist=shortlist)
    assert result.scores[:2] == (0, 0)
    assert result.scores[2] >= 40


@pytest.mark.parametrize('width', [1, 8, 16])
def test_determinism_and_prefix(width: int) -> None:
    grid = np.zeros((13, 6), dtype=np.int8)
    grid[-2:] = [[1, 2, 1, 3, 2, 3], [2, 1, 2, 3, 3, 1]]
    original = grid.copy()
    a = ideal_fire_search(grid, (), COLORS, hmax=4, beam_width=width, potential_shortlist=4)
    clear_cache()
    b = ideal_fire_search(grid, (), COLORS, hmax=4, beam_width=width, potential_shortlist=4)
    prefix = ideal_fire_search(grid, (), COLORS, hmax=2, beam_width=width, potential_shortlist=4)
    assert a == b and prefix.scores == a.scores[:2]
    np.testing.assert_array_equal(grid, original)


def test_observed_colors_constrain_unresolved_color_set() -> None:
    grid = np.zeros((13, 6), dtype=np.int8)
    grid[-3:, 0] = 5
    # 当時観測済み4色には紫を含む。他の候補色は使えない。
    a = ideal_fire_search(grid, (), observed_colors=(1, 2, 3, 5), hmax=3)
    b = ideal_fire_search(grid, (), (1, 2, 3, 5), hmax=3)
    assert a.scores == b.scores


def test_search_version_is_metadata_and_side_invariant() -> None:
    from src.b3_column_spec import COLUMN_NAMES, Kind, SPEC_BY_NAME, swap_sides
    from src.b3_train_model import feature_columns
    assert COLUMN_NAMES.index('search_version') < COLUMN_NAMES.index('ta_bound_p1')
    assert SPEC_BY_NAME['search_version'].kind is Kind.META
    assert 'search_version' not in feature_columns(COLUMN_NAMES)
    assert swap_sides({'search_version': SEARCH_VERSION}) == {'search_version': SEARCH_VERSION}
    assert search_manifest()['version'] == SEARCH_VERSION


def test_old_native_does_not_silently_use_old_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native, 'NATIVE_AVAILABLE', False)
    clear_cache()
    with pytest.raises(RuntimeError, match='更新したpuyo_core'):
        ideal_fire_search(np.zeros((13, 6), dtype=np.int8), (), COLORS, hmax=1)


def test_five_color_conflict_is_not_physical_full() -> None:
    grid = np.zeros((13, 6), dtype=np.int8)
    grid[-1, :5] = (1, 2, 3, 4, 5)
    result = ideal_fire_search(grid, (), hmax=1)
    assert result.color_conflict and result.beam_died and not result.full
    assert result.at(1) == (0, 0, True)


def test_conflicting_colors_are_missing_in_learning_features() -> None:
    from src.b3_column_spec import Mask
    from src.b3_features import board_group
    from src.b3_state import Episode, SideState
    from src.b3_timeaxis import time_axis_features
    from src.b3_timing import TimingTable
    side, attacker = SideState('2P'), SideState('1P')
    side.board = np.zeros((13, 6), dtype=np.int8)
    side.board[-1, :5] = (1, 2, 3, 4, 5)
    row = {}
    board_group(row, side, 'p2', 0, 0, COLORS)
    assert row['mk_k_p2'] == Mask.RECOGNITION_CONFLICT and np.isnan(row['k1_ojama_p2'])
    attacker.episode = Episode(0, 0, pred_l=3, pred_score=1000)
    ta = time_axis_features(attacker, side, 0, 0, COLORS, TimingTable(), 0)
    assert ta['__mk'] == Mask.RECOGNITION_CONFLICT and np.isnan(ta['ta_q_med'])
