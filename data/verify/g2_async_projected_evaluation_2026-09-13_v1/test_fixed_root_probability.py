"""原J/consumer/rootを使い、PB履歴だけ人工にした確率1接続の限定CPU検査。"""
from dataclasses import asdict
import json
from types import SimpleNamespace as N
from typing import Any
import pytest
import fixed_root_probability as P
from test_fixed_origin_reference import source


def history_for(capture: Any) -> tuple[Any, dict]:
    snapshot = capture.ledger.snapshot(capture.handles['2P'])
    scope = capture.scopes['2P']
    generation = asdict(snapshot.generation)
    cells = [[[[color, 1.0]] for color in row] for row in snapshot.origin_before_grid]
    proof = dict(side='2P', frame=100, clock=100 / P.FPS, journal_token='artificial_PB_source', state='stable',
        scope=[scope['source_id'], scope['run_id'], scope['software_reset'], id(capture.pipe),
               id(capture.pipe._sm_2p), snapshot.generation.reset_epoch, '2P'],
        journal_scope=dict(scope, frame_idx=100, time_sec=100 / P.FPS, side='2P'),
        generation=generation, generation_after=generation.copy(), pb_holds=[], pb_errors=[],
        in_step_generation_changed=False, basis_registered=False, quality_gate_clear=False,
        probability=dict(present=True, type_valid=True, errors=[], cells=cells, sha256=P.digest(cells)))
    history = N(journal=capture.journal, pipe=capture.pipe, last_frame=100,
        closed=False, sealed=True, transferred=True, error=None, probability=N(closed=True),
        probability_inputs=lambda: (json.dumps(proof),))
    return history, proof


def test_existing_prediction_reused_without_simulation_or_ledger_mutation(source: Any) -> None:
    capture, _, _, _, _ = source
    history, proof = history_for(capture)
    original = capture.ledger.snapshot(capture.handles['2P'])
    def forbidden(*args: Any) -> None: raise AssertionError('extra simulation')
    capture.simulator = N(simulate=forbidden)
    result = P.read(history, capture, 100)
    assert result['status'] == 'READY' and result['prediction']['weight'] == 1.0
    assert result['probability_frame'] == result['history_last_frame'] == result['cutoff_frame'] == 100
    assert result['prediction']['grid'] == original.predictions[0].final_grid
    assert not any(result[key] for key in ('accounting_permission', 'future_fire_power_supply_authorized',
        'physical_identity_verified', 'calibrated_win_probability', 'quality_gate_clear', 'additional_simulation'))
    result['probability_source']['state'] = 'changed_returned_copy'
    assert proof['state'] == 'stable' and capture.ledger.snapshot(capture.handles['2P']) == original


def test_original_J_identity_scope_metadata_is_not_a_generation_difference(source: Any) -> None:
    capture, _, _, _, _ = source
    history, proof = history_for(capture)
    for key in ('generation', 'generation_after'):
        proof[key]['identity_scope'] = 'software_observation_only_not_physical_identity'
    result = P.read(history, capture, 100)
    assert result['status'] == 'READY'


@pytest.mark.parametrize('fault,reason', [('before', 'before_origin_trigger'), ('generation', 'generation_mismatch'),
    ('uncertain', 'joint_not_certified'), ('grid', 'root_grid_mismatch'), ('missing', 'probability_missing')])
def test_unqualified_candidate_is_explicit_hold(source: Any, fault: str, reason: str) -> None:
    capture, _, _, _, _ = source
    history, proof = history_for(capture)
    if fault == 'before':
        proof.update(frame=98, clock=98 / P.FPS)
        proof['journal_scope']['frame_idx'] = 98
        proof['journal_scope']['time_sec'] = 98 / P.FPS
    elif fault == 'generation': proof['scope'][5] += 1
    elif fault == 'missing': history.probability_inputs = lambda: ()
    else:
        cell = proof['probability']['cells'][0][0]
        replacement = 1 if cell[0][0] == 0 else 0
        if fault == 'uncertain': cell[:] = [[cell[0][0], 0.5], [replacement, 0.5]]
        else: cell[:] = [[replacement, 1.0]]
        proof['probability']['sha256'] = P.digest(proof['probability']['cells'])
    result = P.read(history, capture, 100)
    assert result['status'] == 'HOLD' and reason in result['reason'] and result['prediction'] is None


@pytest.mark.parametrize('fault', ['owner', 'hash', 'future', 'closed', 'unsealed'])
def test_corrupted_source_or_lifetime_stays_strict(source: Any, fault: str) -> None:
    capture, _, _, _, _ = source
    history, proof = history_for(capture)
    if fault == 'owner': proof['journal_scope']['run_id'] = 'foreign'
    elif fault == 'hash': proof['probability']['sha256'] = 'wrong'
    elif fault == 'future':
        proof.update(frame=102, clock=102 / P.FPS)
        proof['journal_scope']['frame_idx'] = 102
        proof['journal_scope']['time_sec'] = 102 / P.FPS
    elif fault == 'closed': history.closed = True
    else: history.sealed = False
    with pytest.raises(ValueError, match='fixed_root_probability:'): P.read(history, capture, 100)
