"""M1 ledger V3 の finalized/provisional 分離と tensor 契約を検査する。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from src import advantage_m1_causal_ledger_v3 as subject
from src.advantage_m1_causal_ledger_v3 import (
    AVAILABILITY_ORDER,
    AdvantageM1CausalLedgerCNNV3,
    AdvantageM1PrimaryV3,
    LEDGER_FIELD_COUNT,
    LEDGER_SIDE_FIELDS,
    M1_INPUT_SCHEMA_VERSION,
    advantage_m1_inputs_for_serving,
    advantage_m1_inputs_for_training,
    advantage_m1_inputs_from_materialized_state,
    advantage_m1_primary_from_canonical,
    advantage_m1_primary_from_materialized_state,
    materialized_ledger_is_usable,
    tensorize_primary,
)
from src.canonical_observation_adapter_v3 import canonical_observation_v3_from_v2
from src.canonical_observation_adapter_v2 import (
    CanonicalCutoff,
    canonical_observation_for_training as canonical_v2_for_training,
)
from src.canonical_observation_v2 import AvailabilityState, AvailableInt
from src.canonical_observation_v3 import CanonicalObservationError, CanonicalObservationV3
from src.event_learning_tables_v1 import build_learning_tables
from tests.test_canonical_observation_adapter_v2 import _base_batches
from tests.test_canonical_observation_v2 import _known_provenance, _observation


KNOWN_ZERO = AVAILABILITY_ORDER.index(AvailabilityState.KNOWN_ZERO)
UNKNOWN = AVAILABILITY_ORDER.index(AvailabilityState.UNKNOWN)
UNSUPPORTED = AVAILABILITY_ORDER.index(AvailabilityState.UNSUPPORTED)
FAULT = AVAILABILITY_ORDER.index(AvailabilityState.INTEGRITY_FAULT)


def _state() -> dict[str, object]:
    state: dict[str, object] = {
        "a_p1_grid": [0] * 78, "a_p2_grid": [0] * 78,
        "b_causal_exchange_usable": True, "b_input_usable": True,
        "b_causal_observed_attack_balance": 999,
    }
    for side in ("p1", "p2"):
        state.update({
            f"a_{side}_next_first": 1, f"a_{side}_next_second": 2,
            f"a_{side}_double_next_first": 3, f"a_{side}_double_next_second": 4,
            f"b_{side}_causal_pending_garbage": 0 if side == "p1" else 35,
            f"b_{side}_causal_effective_rate": 70,
            f"b_{side}_chain_active": side == "p1",
            f"b_{side}_provisional_generated": 4 if side == "p1" else None,
            f"b_{side}_provisional_score": 280 if side == "p1" else None,
            f"b_{side}_provisional_chain_count": 2 if side == "p1" else None,
        })
    return state


def _c76_saved_state_1729133ms() -> dict[str, object]:
    """保存済みV6 states.parquetのvideo_c76/segment26実値。"""

    state = _state()
    state.update({
        "source_video_id": "video_c76", "available_ms": 1_729_133,
        "online_segment_index": 26, "b_causal_observed_attack_balance": 24,
        "b_p1_causal_pending_garbage": 0, "b_p2_causal_pending_garbage": 0,
        "b_p1_chain_active": True, "b_p2_chain_active": False,
        "b_p1_provisional_generated": 32, "b_p2_provisional_generated": None,
        "b_p1_provisional_score": 2240, "b_p2_provisional_score": None,
        "b_p1_provisional_chain_count": 5,
        "b_p2_provisional_chain_count": None,
    })
    return state


def _tensor_batch(observation: CanonicalObservationV3) -> tuple[torch.Tensor, ...]:
    inputs = advantage_m1_inputs_for_training(observation)
    return (
        torch.from_numpy(inputs.boards[None].astype(np.int64)),
        torch.from_numpy(inputs.queues[None].astype(np.int64)),
        torch.from_numpy(inputs.ledger_values[None]),
        torch.from_numpy(inputs.ledger_availability[None]),
    )


def test_six_primary_fields_and_schema_are_fixed() -> None:
    assert LEDGER_SIDE_FIELDS == (
        "pending_garbage", "effective_rate", "chain_active",
        "provisional_generated", "provisional_score", "provisional_chain_count",
    )
    assert LEDGER_FIELD_COUNT == 6
    inputs = advantage_m1_inputs_from_materialized_state(_state())
    assert inputs.schema_version == M1_INPUT_SCHEMA_VERSION
    assert M1_INPUT_SCHEMA_VERSION == "m1-causal-ledger-input/v2"


def test_materialized_pending_uses_finalized_source_only() -> None:
    first = advantage_m1_inputs_from_materialized_state(_state())
    changed = _state()
    changed["b_causal_observed_attack_balance"] = -5000
    changed["b_p1_provisional_generated"] = 72
    changed["b_p1_provisional_score"] = 9000
    second = advantage_m1_inputs_from_materialized_state(changed)

    np.testing.assert_array_equal(first.ledger_values[:, 0], second.ledger_values[:, 0])
    assert first.ledger_values[0, 3] != second.ledger_values[0, 3]
    assert first.ledger_values[0, 4] != second.ledger_values[0, 4]


def test_c76_saved_1729133ms_keeps_balance_24_out_of_finalized_pending() -> None:
    inputs = advantage_m1_inputs_from_materialized_state(_c76_saved_state_1729133ms())

    np.testing.assert_array_equal(inputs.ledger_values[:, 0], np.zeros(2))
    np.testing.assert_array_equal(inputs.ledger_availability[:, 0, KNOWN_ZERO], 1.0)
    assert inputs.ledger_values[0, 3] == pytest.approx(32.0 / (32.0 + 72.0))
    assert inputs.ledger_values[0, 4] == pytest.approx(2240.0 / (2240.0 + 1000.0))


def test_inactive_provisional_is_known_zero_not_unknown() -> None:
    inputs = advantage_m1_inputs_from_materialized_state(_state())
    for index in (3, 4, 5):
        assert inputs.ledger_values[1, index] == 0.0
        assert inputs.ledger_availability[1, index, KNOWN_ZERO] == 1.0


def test_active_missing_provisional_is_unknown() -> None:
    state = _state()
    state["b_p1_provisional_score"] = None
    inputs = advantage_m1_inputs_from_materialized_state(state)
    assert inputs.ledger_values[0, 4] == 0.0
    assert inputs.ledger_availability[0, 4, UNKNOWN] == 1.0


def test_known_zero_unsupported_and_fault_have_distinct_masks() -> None:
    known_zero = advantage_m1_inputs_from_materialized_state(_state())
    unsupported_state = _state() | {"b_input_usable": False}
    fault_state = _state() | {"b_integrity_fault": True}
    unsupported = advantage_m1_inputs_from_materialized_state(unsupported_state)
    fault = advantage_m1_inputs_from_materialized_state(fault_state)

    assert known_zero.ledger_availability[0, 0, KNOWN_ZERO] == 1.0
    assert unsupported.ledger_availability[0, 0, UNSUPPORTED] == 1.0
    assert fault.ledger_availability[0, 0, FAULT] == 1.0


def test_physical_accounting_unsupported_segment_disables_materialized_ledger() -> None:
    state = _state() | {"quality_physical_accounting_unsupported_segment": True}

    assert not materialized_ledger_is_usable(state)
    inputs = advantage_m1_inputs_from_materialized_state(state)
    np.testing.assert_array_equal(inputs.ledger_values[:, 0], np.zeros(2))
    np.testing.assert_array_equal(inputs.ledger_availability[:, 0, UNSUPPORTED], 1.0)


@pytest.mark.parametrize("flag", [False, None])
def test_non_true_physical_accounting_flag_keeps_existing_usability(
    flag: bool | None,
) -> None:
    state = _state() | {"quality_physical_accounting_unsupported_segment": flag}

    assert materialized_ledger_is_usable(state)


def test_missing_physical_accounting_flag_keeps_existing_usability() -> None:
    assert materialized_ledger_is_usable(_state())


def test_invalid_finalized_pending_is_integrity_fault() -> None:
    state = _state()
    state["b_p2_causal_pending_garbage"] = -1
    inputs = advantage_m1_inputs_from_materialized_state(state)
    assert inputs.ledger_availability[1, 0, FAULT] == 1.0


def test_train_serving_alias_and_canonical_side_swap() -> None:
    assert advantage_m1_inputs_for_training is advantage_m1_inputs_for_serving
    observation = canonical_observation_v3_from_v2(_observation())
    direct = advantage_m1_inputs_for_training(observation)
    swapped = advantage_m1_inputs_for_training(observation.swap_sides())
    for name in ("boards", "queues", "ledger_values", "ledger_availability"):
        np.testing.assert_array_equal(getattr(swapped, name), getattr(direct, name)[::-1])


def test_both_sources_extract_typed_primary_and_call_one_public_tensorizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = canonical_observation_v3_from_v2(_observation())
    canonical = advantage_m1_primary_from_canonical(observation)
    materialized = advantage_m1_primary_from_materialized_state(_state())
    assert isinstance(canonical, AdvantageM1PrimaryV3)
    assert isinstance(materialized, AdvantageM1PrimaryV3)
    np.testing.assert_array_equal(
        tensorize_primary(canonical).ledger_values,
        advantage_m1_inputs_for_training(observation).ledger_values,
    )
    calls: list[AdvantageM1PrimaryV3] = []
    original = subject.tensorize_primary
    monkeypatch.setattr(subject, "tensorize_primary", lambda value: (
        calls.append(value) or original(value)
    ))
    subject.advantage_m1_inputs_from_canonical(observation)
    subject.advantage_m1_inputs_from_materialized_state(_state())
    assert len(calls) == 2
    assert all(isinstance(value, AdvantageM1PrimaryV3) for value in calls)


def test_canonical_provisional_change_does_not_change_finalized_pending() -> None:
    observation = canonical_observation_v3_from_v2(_observation())
    changed_side = replace(
        observation.ledger.p1,
        provisional_generated=AvailableInt.known(72, _known_provenance("new:provisional")),
    )
    changed = replace(observation, ledger=replace(observation.ledger, p1=changed_side))
    before = advantage_m1_inputs_for_training(observation)
    after = advantage_m1_inputs_for_training(changed)
    assert before.ledger_values[0, 0] == after.ledger_values[0, 0]
    assert before.ledger_values[0, 3] != after.ledger_values[0, 3]


@pytest.mark.parametrize("mode", ["values", "masks", "values_and_masks"])
def test_model_probability_is_side_swap_complement(mode: str) -> None:
    torch.manual_seed(41)
    model = AdvantageM1CausalLedgerCNNV3(mode)  # type: ignore[arg-type]
    observation = canonical_observation_v3_from_v2(_observation())
    direct = model(*_tensor_batch(observation)).raw_probability
    swapped = model(*_tensor_batch(observation.swap_sides())).raw_probability
    torch.testing.assert_close(swapped, 1.0 - direct, atol=1e-7, rtol=0.0)


def test_materialized_values_are_float32_and_normalized() -> None:
    inputs = advantage_m1_inputs_from_materialized_state(_state())
    assert inputs.ledger_values.dtype == np.float32
    assert inputs.ledger_availability.dtype == np.float32
    assert 0.0 <= float(inputs.ledger_values.min())
    assert float(inputs.ledger_values.max()) <= 1.0
    np.testing.assert_array_equal(inputs.ledger_availability.sum(axis=-1), 1.0)


def test_v2_observation_is_rejected_at_public_boundary() -> None:
    with pytest.raises(CanonicalObservationError, match="V3の型"):
        advantage_m1_inputs_for_training(_observation())  # type: ignore[arg-type]


def test_materialized_tensor_is_bit_identical_to_canonical_v3() -> None:
    observation = canonical_observation_v3_from_v2(_observation())
    state = _state()
    for side in ("p1", "p2"):
        canonical_side = getattr(observation, side)
        state[f"a_{side}_grid"] = np.asarray(canonical_side.board.grid).reshape(-1)
    direct = advantage_m1_inputs_for_training(observation)
    materialized = advantage_m1_inputs_from_materialized_state(state)
    for name in ("boards", "queues", "ledger_values", "ledger_availability"):
        np.testing.assert_array_equal(getattr(direct, name), getattr(materialized, name))


def test_event_prefix_materialization_matches_canonical_v3() -> None:
    batches = _base_batches()
    legacy = canonical_v2_for_training(batches, CanonicalCutoff(1, 10))
    assert legacy is not None
    canonical = canonical_observation_v3_from_v2(legacy)
    tables = build_learning_tables(
        batches, fold=1, tier="test", source_group_id="source-test",
    )
    direct = advantage_m1_inputs_for_training(canonical)
    materialized = advantage_m1_inputs_from_materialized_state(tables.states[-1])
    for name in ("boards", "queues", "ledger_values", "ledger_availability"):
        np.testing.assert_array_equal(getattr(direct, name), getattr(materialized, name))
