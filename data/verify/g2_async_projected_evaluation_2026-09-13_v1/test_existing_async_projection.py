"""既存正式adapterの片側連鎖予測＋相手最新確定盤面を人工因果prefixで確認する。"""
import copy
import importlib.util
from pathlib import Path
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('_existing_projected_fixture',
    ROOT / 'tests/test_projected_state_observation_adapter_v1.py')
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)


def fixture() -> tuple[Any, Any, list[list[int]], int]:
    board = copy.deepcopy(F.EMPTY_GRID)
    board[-1][:4] = [1] * 4
    adapter = F.ProjectedStateObservationAdapter(F._context())
    seq = F._apply_stable(adapter, p1_grid=board)
    first, fire, _ = F._fire_active(adapter, side='p1', seq=seq, p1_grid=board)
    assert first is not None and first.post_chain_state.p1.provenance == 'physics_projected'
    assert first.post_chain_state.p1.grid == tuple(tuple(row) for row in F.EMPTY_GRID)
    return adapter, first, board, fire.events[-1]['seq'] + 1


@pytest.mark.parametrize('missing', [False, True])
def test_other_side_changes_without_both_stable(missing: bool) -> None:
    adapter, first, board, seq = fixture()
    other = copy.deepcopy(F.EMPTY_GRID)
    other[-1][5] = 2
    batch = F._batch('opponent-new-placement', F._event(seq, 'stable_board_observed', 'p2', grid=other))
    state = F._state(batch, p1_active=True, p1_grid=board, p2_grid=other)
    if missing:
        state.pop('a_p2_unknown_mask')
    result = adapter.apply_batch(batch, state)
    assert result is not None and result.post_chain_state.p1 == first.post_chain_state.p1
    assert result.through_event_seq > first.through_event_seq
    assert result.causal_cutoff_digest != first.causal_cutoff_digest
    if missing:
        assert not result.post_chain_state.p2.present and result.post_chain_state.p2.grid is None
    else:
        assert result.post_chain_state.p2.provenance == 'observed'
        assert result.post_chain_state.p2.grid == tuple(tuple(row) for row in other)
        assert result.post_chain_state.p2 != first.post_chain_state.p2
    assert first.post_chain_state.p2.grid == tuple(tuple(row) for row in F.EMPTY_GRID)
