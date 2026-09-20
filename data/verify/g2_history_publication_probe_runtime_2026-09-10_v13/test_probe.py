"""新goal/合成差分だけの検査。原全suiteは反復しない。"""
from __future__ import annotations
import ast
from copy import deepcopy
from pathlib import Path
from typing import Any
import pytest
import common as K
import goals as G
import closure as Q
import session as S


def selected(path: Path, name: str) -> Any:
    return ast.dump(next(n for n in ast.parse(path.read_bytes()).body
        if isinstance(n, ast.FunctionDef) and n.name == name))


def test_original_collect_and_prepare_boundaries_preserved() -> None:
    import derivation_check as D
    for file, names in {'live_cli.py': ('collect', 'constructor_guard', 'refs'),
            'session.py': ('prepare_output', 'live_scopes', 'preflight'),
            'closure.py': ('saved_closures', 'seal', 'verify_engine')}.items():
        for name in names:
            if name == 'seal':
                continue
            if file == 'live_cli.py' and name in ('collect', 'constructor_guard'):
                D.verify_function(file, name)
                continue
            assert selected(K.ROOT/file, name) == selected(K.OLD/file, name)
    assert (K.ROOT/'recording.py').read_bytes() == (K.OLD/'recording.py').read_bytes()


def test_prepare_exclusive_shared_boundary(tmp_path: Path, monkeypatch: Any) -> None:
    output, owned = tmp_path/'new', []
    def prepare(env: Any, path: Path) -> Any:
        assert not path.exists() and path == output and not owned
        return {}, {}
    monkeypatch.setattr(S, 'prepare', prepare)
    receipt, _ = S.prepare_output({}, output, owned)
    assert owned == [output] and receipt['output_boundary']['shared_live_preflight'] is True


def current_row() -> Any:
    grid = [[0] * 6 for _ in range(13)]
    placement = {'artificial': True}
    proof = dict(kind='completed_history_current', frame=34796, clock=34796/60,
        scope={'side': '1P'}, grid=grid, placement=placement,
        channels={n: deepcopy(grid) for n in ('raw', 'sm', 'returned', 'probability')})
    return dict(scope={'frame_idx': 34796, 'time_sec': 34796/60, 'side': '1P'},
        decision={'current_permission': True, 'history_consumed': True, 'current_proof': proof},
        prepared=placement, grid_after=grid)


def test_current_proof_channels_and_clock() -> None:
    row = current_row()
    assert G.current_proof(row) is row['decision']['current_proof']
    row['decision']['current_proof']['channels']['raw'][12][0] = 1
    with pytest.raises(ValueError, match='current_channel_mismatch'):
        G.current_proof(row)


def test_no_event_is_not_current_success(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(K, 'FRAMES', (34796,))
    K.write(tmp_path/G.PUB_STATUS, dict(receiver_closed=True, receiver_errors=[], issued=0, released=0))
    K.write(tmp_path/G.OUTER_ROWS, [dict(frame_idx=34796, time_sec=34796/60,
        old_projection_join_verified=True, comparison_completed=True, tickets_this_update=0,
        released_this_update=0, changed_sides=[])])
    K.write(tmp_path/G.OUTER_COMPARISON, dict(all_consumers_ready=True, equal={'grid': True}))
    value = G.publication(tmp_path, 0)
    assert value['outer_publication_observed'] is value['no_event_is_current_success'] is False
    with pytest.raises(ValueError, match='publication_event_count'):
        G.publication(tmp_path, 1)


def test_failure_never_writes_complete(tmp_path: Path) -> None:
    result = Q.finalize(tmp_path, 1, 0, tmp_path/'unused')
    assert result['status'] == 'child_or_resource_failed' and not (tmp_path/'COMPLETE').exists()


def test_no_old_history_goal_and_raw_outer_order() -> None:
    assert not hasattr(Q, 'historical_goal') and 'history_goal_met' not in (K.ROOT/'closure.py').read_text()
    text = (K.ROOT/'live_cli.py').read_text()
    assert text.index('R.install(stack') < text.index('raw.install(stack')
    assert 'comparison.consume' not in text and 'video38_history_publication_probe_' in text


def test_connected_adoption_metadata(monkeypatch: Any) -> None:
    monkeypatch.setattr(G,'history',lambda rows,legal:dict(current_proofs=[],current_event_observed=False))
    monkeypatch.setattr(G,'publication',lambda output,count:dict(outer_publication_observed=False))
    result = G.evaluate([],None,None)
    assert result['baseline_adoption_repair_not_connected'] is False
    assert result['quality_gate_clear'] is False and result['physical_certified'] is False
