"""完全一致の候補・順序・特定状態を元の全探索と比較する。"""
import numpy as np
import pytest
from src.ama_tsumo_seed_v5 import identify, excluded_mask, zero_mismatch_hits
from src.phase_j.live_ama_seed import ExactTable, identify_exact
from src.tsumo_seed_identifier import SeedTable
from src.tsumo_seed_generator import PROJECT_COLOR_OF_LETTER, WIKI_COLOR_LETTERS


@pytest.fixture(scope='module')
def table() -> SeedTable:
    rng = np.random.default_rng(20261020)
    cmap = np.stack([rng.permutation(5) for _ in range(64)]).astype(np.uint8)
    pool = rng.integers(0, 4, (64, 256), dtype=np.uint8)
    # 同一候補の重複を含め、順序と曖昧判定も検査する。
    cmap[1], pool[1] = cmap[0], pool[0]
    return SeedTable(cmap, pool)


@pytest.mark.parametrize('seed,start,length', [(0, 0, 8), (3, 5, 8), (7, 20, 24), (9, 117, 8)])
def test_matches_and_identification(table: SeedTable, seed: int, start: int, length: int) -> None:
    colors = [PROJECT_COLOR_OF_LETTER[WIKI_COLOR_LETTERS[i]] for i in table.cmap[seed, :4]]
    pairs = [tuple(map(int, table.abs_colors[seed, table.tape_to_absolute_index(seed, i)][::-1]))
             for i in range(start, start+length)]
    mask, fast = excluded_mask(table, colors), ExactTable(table, colors)
    assert zero_mismatch_hits(fast, pairs, mask) == zero_mismatch_hits(table, pairs, mask)
    for observed in (pairs, pairs[:3], [p[::-1] for p in pairs], [(9, 9), *pairs[1:]]):
        for first in (0, start, start+1):
            history = dict(counters=[first, first], tapes=[observed, observed])
            assert identify_exact(table, history, colors) == identify(table, history, colors)


def test_gaps_and_conflicts(table: SeedTable) -> None:
    for history in (dict(counters=[None, 0], tapes=[[], []]),
                    dict(counters=[0, 0], tapes=[[(1, 1)]*8, [(2, 2)]*8]),
                    dict(counters=[0, 10], tapes=[[(1, 1)]*3, [(1, 1)]*3])):
        assert identify_exact(table, history, [1, 2, 3, 4]) == identify(table, history, [1, 2, 3, 4])
