"""限定CPU検査。実保存v12の拒否と人工正常対照の受理、及び権限・非干渉の契約。"""
from __future__ import annotations
from dataclasses import replace
from typing import Any
import pytest
import saved_contrast as S
import settled_basis_gate as G


def observations(repaired: bool) -> list[G.CallObservation]:
    return S.repaired_observations() if repaired else S.saved_observations()


def gate() -> G.SettledBasisGate:
    return G.SettledBasisGate(S.SCOPE, S.RESET_FRAME, S.DEADLINE)


def feed(items: list[G.CallObservation]) -> G.SettledBasisGate:
    value = gate()
    for obs in items: value.observe(obs)
    return value


def reasons(value: G.SettledBasisGate) -> dict[int, str | None]:
    return {row['frame']: row['reason'] for row in value.rows}


def test_saved_v12_is_rejected_by_hidden_distribution_loss() -> None:
    value = feed(observations(False))
    assert value.candidate is None
    assert reasons(value)[35172] == 'hidden_distribution_lost'


def test_initial_stable_after_reset_is_not_adopted() -> None:
    value = feed(observations(False))
    assert reasons(value)[35164] == 'awaiting_tsumo_fall_after_reset'
    assert reasons(value)[35166] == 'awaiting_tsumo_fall_after_reset'


def test_repaired_input_is_an_artificial_control_that_passes() -> None:
    value = feed(observations(True))
    assert value.state == 'ISSUED' and value.candidate is not None
    assert value.candidate.frame == 35172 and value.candidate.tsumo_fall_frames == (35168, 35170)


def test_candidate_holds_no_permission_and_no_consumption() -> None:
    candidate = feed(observations(True)).candidate
    assert candidate is not None
    assert not any((candidate.quality_gate_clear, candidate.physical_certified,
                    candidate.integer_current_permission, candidate.accounting_permission,
                    candidate.production_permission, candidate.current_permission,
                    candidate.display_update_permission, candidate.both_sides_stable_confirmed))
    assert candidate.consumed_next_token is None


def test_single_side_candidate_does_not_grant_display_update() -> None:
    """片側基準の成立だけでは有利不利表示を更新しない。bothSTABLEは評価側の追加条件。"""
    candidate = feed(observations(True)).candidate
    assert candidate is not None and candidate.scope[6] == '1P'
    assert candidate.display_update_permission is False


def test_source_call_contract_is_the_issuing_call() -> None:
    items = observations(True)
    candidate = feed(items).candidate
    issuing = [obs for obs in items if obs.frame == 35172][0]
    assert candidate is not None
    assert candidate.source_call_token == issuing.call_token
    assert candidate.source_observation_stage == G.ACTUAL_CALL_STAGE == issuing.observation_stage


def test_reissue_is_refused() -> None:
    items = observations(True)
    value = feed(items)
    first = value.candidate
    again = value.observe(replace(items[-1], call_token='retry', frame=35176))
    assert again is None and value.candidate is first
    assert value.rows[-1]['reason'] == 'candidate_already_issued'


def test_core_records_only_and_touches_nothing() -> None:
    value = feed(observations(True))
    assert all(not row['mutated_fifo'] and not row['mutated_current']
               and not row['mutated_counter'] and row['consumed_next_token'] is None
               for row in value.rows)
    assert not hasattr(value, 'history') and not hasattr(value, 'inventory')


@pytest.mark.parametrize('field,value,reason', [
    ('exception', 'RuntimeError()', 'exception_present'),
    ('observation_stage', 'publication_after', 'not_actual_J_call_stage'),
    ('epoch', 99, 'epoch_or_generation_changed'),
    ('generation', 99, 'epoch_or_generation_changed'),
    ('scope', ('video_38', 'pipe_1', 4, 0, 0, 1, '1P'), 'scope_changed'),
    ('frame', 35173, 'stride_broken'),
    ('match_active', False, 'inactive_or_window'),
    ('effect_gate_window_active', True, 'inactive_or_window'),
    ('origin_present', True, 'origin_present'),
    ('landing_grace_expired', False, 'landing_grace_pending'),
])
def test_each_qualification_blocks_issue(field: str, value: Any, reason: str) -> None:
    items = observations(True)
    items[-2] = replace(items[-2], **{field: value})
    result = feed(items)
    assert result.candidate is None and reasons(result)[items[-2].frame] == reason


def test_raw_cnn_sm_returned_must_agree() -> None:
    items = observations(True)
    last = items[-2]
    assert last.cnn_grid is not None
    broken = tuple(tuple(0 for _ in row) for row in last.cnn_grid)
    for name, expected in (('cnn_grid', 'raw_SM_mismatch'), ('sm_confirmed_grid', 'raw_SM_mismatch'),
                           ('returned_grid', 'returned_mismatch')):
        items[-2] = replace(last, **{name: broken})
        assert reasons(feed(items))[last.frame] == expected


def test_visible_must_be_pointmass_and_known() -> None:
    items = observations(True)
    last = items[-2]
    assert last.visible_probability is not None and last.raw_grid is not None
    spread = ((0, 0.5), (1, 0.5))
    items[-2] = replace(last, visible_probability=(spread,) + last.visible_probability[1:])
    assert reasons(feed(items))[last.frame] == 'probability_nonpointmass'
    row = tuple(G.COLOR_UNKNOWN for _ in last.raw_grid[G.HIDDEN_ROWS])
    unknown = last.raw_grid[:G.HIDDEN_ROWS] + (row,) + last.raw_grid[G.HIDDEN_ROWS + 1:]
    items[-2] = replace(last, raw_grid=unknown, cnn_grid=unknown,
                        sm_confirmed_grid=unknown, returned_grid=unknown)
    assert reasons(feed(items))[last.frame] == 'visible_unknown'


def test_unknown_is_never_converted_to_empty() -> None:
    """UNKNOWNの隠しは分布を伴う場合だけ受理し、0へ潰した入力は拒否する。"""
    items = observations(True)
    last = items[-2]
    assert last.hidden_probability is not None
    collapsed = tuple(((0, 1.0),) for _ in last.hidden_probability)
    items[-2] = replace(last, hidden_probability=collapsed)
    assert reasons(feed(items))[last.frame] == 'hidden_unknown_without_distribution'


def test_deadline_is_not_relaxed() -> None:
    items = observations(True)
    assert items[-1].frame == S.DEADLINE + 2
    assert reasons(feed(observations(False)))[items[-1].frame] == 'deadline_exceeded'


def test_call_token_reuse_breaks_the_gate() -> None:
    items = observations(True)
    items[-2] = replace(items[-2], call_token=items[-3].call_token)
    result = feed(items)
    assert result.state == 'BROKEN' and result.candidate is None


def test_board_constants_match_the_original_module() -> None:
    from src.board import BOARD_COLS, BOARD_ROWS, COLOR_UNKNOWN, HIDDEN_ROWS
    assert (G.BOARD_ROWS, G.BOARD_COLS, G.HIDDEN_ROWS, G.COLOR_UNKNOWN) == (
        BOARD_ROWS, BOARD_COLS, HIDDEN_ROWS, COLOR_UNKNOWN)
