"""finalized ledger と provisional 観測を分離した M1 入力契約 V2。"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal, Mapping

import numpy as np
import torch
import torch.nn as nn

from src.advantage_m0_current_cnn_v1 import (
    AdvantageM0OutputV1,
    COLOR_INVARIANT_EMBED_DIM,
    ColorInvariantSideEncoderV1,
    board_categories_from_raw,
    queue_categories_from_raw,
)
from src.canonical_observation_v3 import (
    AvailabilityState,
    AvailableBool,
    AvailableInt,
    CanonicalObservationV3,
    validate_canonical_observation,
)


M1_INPUT_SCHEMA_VERSION = "m1-causal-ledger-input/v2"
M1_MODEL_VERSION = "advantage-m1-causal-ledger-cnn/v3"
AVAILABILITY_ORDER = tuple(AvailabilityState)
AVAILABILITY_COUNT = len(AVAILABILITY_ORDER)
LEDGER_MODE = Literal["values", "masks", "values_and_masks"]
LEDGER_SIDE_FIELDS = (
    "pending_garbage",
    "effective_rate",
    "chain_active",
    "provisional_generated",
    "provisional_score",
    "provisional_chain_count",
)
LEDGER_FIELD_COUNT = len(LEDGER_SIDE_FIELDS)
LEDGER_EMBED_DIM = 24
PAIR_HIDDEN_DIM = 64
NORMALIZATION_SCALES = {
    "pending_garbage": 72.0,
    "effective_rate": 70.0,
    "provisional_generated": 72.0,
    "provisional_score": 1000.0,
    "provisional_chain_count": 5.0,
}
SIDES = ("p1", "p2")
INTEGRITY_STATUS = "integrity_fault"


class AdvantageM1InputError(ValueError):
    """canonical 観測または ledger tensor が M1 契約に違反した。"""


@dataclass(frozen=True, slots=True)
class AdvantageM1InputsV3:
    """一観測分の学習・本番共通 M1 tensor 原本。"""

    boards: np.ndarray
    queues: np.ndarray
    ledger_values: np.ndarray
    ledger_availability: np.ndarray
    schema_version: str = M1_INPUT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class PrimaryIntV3:
    """provenance検証済み入力から抽出したraw整数とavailability。"""

    value: int | None
    availability: AvailabilityState

    def __post_init__(self) -> None:
        if self.value is not None and not _is_nonnegative_int(self.value):
            raise AdvantageM1InputError("primary integerが非負整数ではありません")
        _validate_primary_availability(self.value, self.availability, 0)


@dataclass(frozen=True, slots=True)
class PrimaryBoolV3:
    """provenance検証済み入力から抽出したraw boolとavailability。"""

    value: bool | None
    availability: AvailabilityState

    def __post_init__(self) -> None:
        if self.value is not None and type(self.value) is not bool:
            raise AdvantageM1InputError("primary boolの型が不正です")
        _validate_primary_availability(self.value, self.availability, False)


@dataclass(frozen=True, slots=True)
class AdvantageM1PrimarySideV3:
    """正規化前の6個のtyped ledger一次観測。"""

    pending_garbage: PrimaryIntV3
    effective_rate: PrimaryIntV3
    chain_active: PrimaryBoolV3
    provisional_generated: PrimaryIntV3
    provisional_score: PrimaryIntV3
    provisional_chain_count: PrimaryIntV3

    def __post_init__(self) -> None:
        expected = (PrimaryIntV3, PrimaryIntV3, PrimaryBoolV3) + (PrimaryIntV3,) * 3
        if any(not isinstance(value, kind) for value, kind in zip(
            _primary_items(self), expected, strict=True,
        )):
            raise AdvantageM1InputError("primary ledger sideの型が不正です")
        if self.effective_rate.availability == AvailabilityState.KNOWN_ZERO:
            raise AdvantageM1InputError("known effective rateは正整数必須です")


@dataclass(frozen=True, slots=True)
class AdvantageM1PrimaryV3:
    """canonical/materializedが共通tensorizerへ渡すraw一次DTO。"""

    boards: np.ndarray
    queues: np.ndarray
    p1: AdvantageM1PrimarySideV3
    p2: AdvantageM1PrimarySideV3


class CausalLedgerEncoderV3(nn.Module):
    """6 個の一次 ledger 値と availability を左右共有で encode する。"""

    def __init__(self, mode: LEDGER_MODE = "values_and_masks") -> None:
        super().__init__()
        if mode not in {"values", "masks", "values_and_masks"}:
            raise AdvantageM1InputError(f"未対応 ledger mode です: {mode}")
        self.mode = mode
        self.network = nn.Sequential(
            nn.Linear(_ledger_input_width(mode), 48), nn.ReLU(),
            nn.Linear(48, LEDGER_EMBED_DIM), nn.ReLU(),
        )

    def forward(self, values: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        _validate_side_ledger(values, availability)
        if self.mode == "values":
            inputs = values
        elif self.mode == "masks":
            inputs = availability.flatten(start_dim=1)
        else:
            inputs = torch.cat((values, availability.flatten(start_dim=1)), dim=1)
        return self.network(inputs)


class AdvantageM1CausalLedgerCNNV3(nn.Module):
    """盤面と重複を除いた因果 ledger を読む左右交換反対称モデル。"""

    model_version = M1_MODEL_VERSION

    def __init__(self, ledger_mode: LEDGER_MODE = "values_and_masks") -> None:
        super().__init__()
        self.side_encoder = ColorInvariantSideEncoderV1()
        self.ledger_encoder = CausalLedgerEncoderV3(ledger_mode)
        width = (COLOR_INVARIANT_EMBED_DIM + LEDGER_EMBED_DIM) * 3
        self.scorer = nn.Sequential(
            nn.Linear(width, PAIR_HIDDEN_DIM), nn.ReLU(),
            nn.Linear(PAIR_HIDDEN_DIM, 1),
        )

    def forward(
        self, boards: torch.Tensor, queues: torch.Tensor,
        ledger_values: torch.Tensor, ledger_availability: torch.Tensor,
    ) -> AdvantageM0OutputV1:
        batch_size = boards.shape[0]
        _validate_batched_ledger(ledger_values, ledger_availability, batch_size)
        board_side = self.side_encoder(
            boards.flatten(0, 1), queues.flatten(0, 1),
        ).reshape(batch_size, 2, COLOR_INVARIANT_EMBED_DIM)
        ledger_side = self.ledger_encoder(
            ledger_values.flatten(0, 1), ledger_availability.flatten(0, 1),
        ).reshape(batch_size, 2, LEDGER_EMBED_DIM)
        sides = torch.cat((board_side, ledger_side), dim=2)
        direct = self._score(sides[:, 0], sides[:, 1])
        swapped = self._score(sides[:, 1], sides[:, 0])
        logit = 0.5 * (direct - swapped)
        return AdvantageM0OutputV1(logit, torch.sigmoid(logit))

    def _score(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.scorer(torch.cat((left, right, left - right), dim=1)).squeeze(1)


def advantage_m1_inputs_from_canonical(
    observation: CanonicalObservationV3,
) -> AdvantageM1InputsV3:
    """CanonicalObservationV3 から固定順序の M1 入力を作る。"""

    return tensorize_primary(advantage_m1_primary_from_canonical(observation))


def advantage_m1_primary_from_canonical(
    observation: CanonicalObservationV3,
) -> AdvantageM1PrimaryV3:
    """CanonicalObservationV3から正規化前の共通一次DTOを抽出する。"""

    validate_canonical_observation(observation)
    boards = np.stack([_canonical_board(observation, side) for side in SIDES])
    queues = np.stack([_canonical_queue(observation, side) for side in SIDES])
    sides = [_canonical_primary_side(observation, side) for side in SIDES]
    return AdvantageM1PrimaryV3(boards, queues, *sides)


def advantage_m1_inputs_from_materialized_state(
    state: Mapping[str, Any],
) -> AdvantageM1InputsV3:
    """保存済み event state の finalized 列だけから同一 tensor を作る。"""

    return tensorize_primary(advantage_m1_primary_from_materialized_state(state))


def advantage_m1_primary_from_materialized_state(
    state: Mapping[str, Any],
) -> AdvantageM1PrimaryV3:
    """保存済みevent stateから正規化前の共通一次DTOを抽出する。"""

    boards = np.stack([_materialized_board(state, side) for side in SIDES])
    queues = np.stack([_materialized_queue(state, side) for side in SIDES])
    sides = [_materialized_primary_side(state, side) for side in SIDES]
    return AdvantageM1PrimaryV3(boards, queues, *sides)


def tensorize_primary(primary: AdvantageM1PrimaryV3) -> AdvantageM1InputsV3:
    """raw一次DTOを唯一の6列正規化・availability生成経路でtensor化する。"""

    if not isinstance(primary, AdvantageM1PrimaryV3):
        raise AdvantageM1InputError("primary DTOの型が不正です")
    rows = [_tensorize_primary_side(side) for side in (primary.p1, primary.p2)]
    values = np.asarray([row[0] for row in rows], dtype=np.float32)
    masks = np.asarray([row[1] for row in rows], dtype=np.float32)
    return AdvantageM1InputsV3(primary.boards, primary.queues, values, masks)


def _canonical_board(observation: CanonicalObservationV3, side: str) -> np.ndarray:
    board = getattr(observation, side).board
    if _state_name(board.availability) != AvailabilityState.KNOWN.value or board.grid is None:
        raise AdvantageM1InputError(f"{side} STABLE 盤面が known ではありません")
    return board_categories_from_raw(np.asarray(board.grid, dtype=np.int8))


def _canonical_queue(observation: CanonicalObservationV3, side: str) -> np.ndarray:
    pieces = getattr(observation, side).pieces
    values = (
        pieces.next.axis.value, pieces.next.child.value,
        pieces.double_next.axis.value, pieces.double_next.child.value,
    )
    return queue_categories_from_raw(values)


def _canonical_primary_side(
    observation: CanonicalObservationV3, side: str,
) -> AdvantageM1PrimarySideV3:
    ledger_side = getattr(observation.ledger, side)
    return AdvantageM1PrimarySideV3(
        _primary_int(ledger_side.pending_garbage),
        _primary_int(ledger_side.effective_rate),
        _primary_bool(ledger_side.chain_active),
        _primary_int(ledger_side.provisional_generated),
        _primary_int(ledger_side.provisional_score),
        _primary_int(ledger_side.provisional_chain_count),
    )


def _materialized_board(state: Mapping[str, Any], side: str) -> np.ndarray:
    grid = np.asarray(state.get(f"a_{side}_grid"), dtype=np.int8)
    if grid.size != 78:
        raise AdvantageM1InputError(f"{side} materialized 盤面が 78 セルではありません")
    return board_categories_from_raw(grid.reshape(13, 6))


def _materialized_queue(state: Mapping[str, Any], side: str) -> np.ndarray:
    values = tuple(
        state.get(f"a_{side}_{name}") for name in (
            "next_first", "next_second", "double_next_first", "double_next_second",
        )
    )
    return queue_categories_from_raw(values)


def _materialized_primary_side(
    state: Mapping[str, Any], side: str,
) -> AdvantageM1PrimarySideV3:
    entries = _materialized_entries(state, side)
    return AdvantageM1PrimarySideV3(*(entries[name] for name in LEDGER_SIDE_FIELDS))


def _tensorize_primary_side(
    side: AdvantageM1PrimarySideV3,
) -> tuple[list[float], list[list[float]]]:
    values, masks = [], []
    for name, item in zip(LEDGER_SIDE_FIELDS, _primary_items(side), strict=True):
        values.append(_available_value(item, name))
        masks.append(_availability_mask(item.availability))
    return values, masks


def _primary_items(
    side: AdvantageM1PrimarySideV3,
) -> tuple[PrimaryIntV3 | PrimaryBoolV3, ...]:
    return tuple(getattr(side, name) for name in LEDGER_SIDE_FIELDS)


def _primary_int(item: AvailableInt) -> PrimaryIntV3:
    return PrimaryIntV3(item.value, item.availability)


def _primary_bool(item: AvailableBool) -> PrimaryBoolV3:
    return PrimaryBoolV3(item.value, item.availability)


def _available_int(
    value: int | None, availability: AvailabilityState,
) -> PrimaryIntV3:
    return PrimaryIntV3(value, availability)


def _available_bool(
    value: bool | None, availability: AvailabilityState,
) -> PrimaryBoolV3:
    return PrimaryBoolV3(value, availability)


def _materialized_entries(
    state: Mapping[str, Any], side: str,
) -> dict[str, PrimaryIntV3 | PrimaryBoolV3]:
    fault = _materialized_integrity_fault(state)
    usable = materialized_ledger_is_usable(state)
    active = _materialized_bool(state.get(f"b_{side}_chain_active"), fault)
    return {
        "pending_garbage": _available_int(*_materialized_pending(state, side, usable, fault)),
        "effective_rate": _available_int(*_materialized_optional_int(
            state.get(f"b_{side}_causal_effective_rate"), fault,
        )),
        "chain_active": _available_bool(*active),
        "provisional_generated": _available_int(*_materialized_provisional(
            state, side, "generated", active,
        )),
        "provisional_score": _available_int(*_materialized_provisional(
            state, side, "score", active,
        )),
        "provisional_chain_count": _available_int(*_materialized_provisional(
            state, side, "chain_count", active,
        )),
    }


def materialized_ledger_is_usable(state: Mapping[str, Any]) -> bool:
    """finalized pending を M1 に入力できる学習表行かを返す。"""

    return bool(
        state.get("b_causal_exchange_usable") is True
        and state.get("b_input_usable") is True
        and state.get("quality_physical_accounting_unsupported_segment") is not True
        and not _materialized_integrity_fault(state)
    )


def _materialized_integrity_fault(state: Mapping[str, Any]) -> bool:
    status = state.get("b_observation_status", state.get("observation_status"))
    if status == INTEGRITY_STATUS or state.get("b_integrity_fault") is True:
        return True
    reasons = state.get("b_reason_codes", ())
    return isinstance(reasons, (list, tuple)) and INTEGRITY_STATUS in reasons


def _materialized_pending(
    state: Mapping[str, Any], side: str, usable: bool, fault: bool,
) -> tuple[int | None, AvailabilityState]:
    if fault:
        return None, AvailabilityState.INTEGRITY_FAULT
    if not usable:
        return None, AvailabilityState.UNSUPPORTED
    return _materialized_required_int(state.get(f"b_{side}_causal_pending_garbage"))


def _materialized_required_int(value: Any) -> tuple[int | None, AvailabilityState]:
    if not _is_nonnegative_int(value):
        return None, AvailabilityState.INTEGRITY_FAULT
    parsed = int(value)
    state = AvailabilityState.KNOWN_ZERO if parsed == 0 else AvailabilityState.KNOWN
    return parsed, state


def _materialized_optional_int(
    value: Any, fault: bool,
) -> tuple[int | None, AvailabilityState]:
    if fault:
        return None, AvailabilityState.INTEGRITY_FAULT
    if value is None:
        return None, AvailabilityState.UNKNOWN
    return _materialized_required_int(value)


def _materialized_bool(value: Any, fault: bool) -> tuple[bool | None, AvailabilityState]:
    if fault:
        return None, AvailabilityState.INTEGRITY_FAULT
    if type(value) is not bool:
        return None, AvailabilityState.INTEGRITY_FAULT
    state = AvailabilityState.KNOWN if value else AvailabilityState.KNOWN_ZERO
    return value, state


def _materialized_provisional(
    state: Mapping[str, Any], side: str, name: str,
    active: tuple[bool | None, AvailabilityState],
) -> tuple[int | None, AvailabilityState]:
    active_value, active_state = active
    if active_state == AvailabilityState.INTEGRITY_FAULT:
        return None, active_state
    if active_value is False:
        return 0, AvailabilityState.KNOWN_ZERO
    if active_value is None:
        return None, AvailabilityState.UNKNOWN
    return _materialized_optional_int(state.get(f"b_{side}_provisional_{name}"), False)


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value >= 0


def _validate_primary_availability(
    value: int | bool | None, state: AvailabilityState, zero: int | bool,
) -> None:
    if not isinstance(state, AvailabilityState):
        raise AdvantageM1InputError("primary availabilityが固定Enumではありません")
    known = state in {AvailabilityState.KNOWN, AvailabilityState.KNOWN_ZERO}
    if known != (value is not None):
        raise AdvantageM1InputError("primary valueとavailabilityが不整合です")
    if state == AvailabilityState.KNOWN_ZERO and value != zero:
        raise AdvantageM1InputError("primary known zeroの値が不正です")
    if state == AvailabilityState.KNOWN and value == zero:
        raise AdvantageM1InputError("primary known値がzeroです")


def _normalized_value(value: int | bool | None, name: str) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    scale = NORMALIZATION_SCALES[name]
    return float(value) / (float(value) + scale)


def _available_value(item: PrimaryIntV3 | PrimaryBoolV3, name: str) -> float:
    return _normalized_value(item.value, name)


def _state_name(state: object) -> str:
    return str(getattr(state, "value", state))


def _availability_mask(state: object) -> list[float]:
    name = _state_name(state)
    candidates = tuple(item.value for item in AVAILABILITY_ORDER)
    if name not in candidates:
        raise AdvantageM1InputError(f"未登録 availability です: {name}")
    return [float(name == candidate) for candidate in candidates]


def _ledger_input_width(mode: LEDGER_MODE) -> int:
    if mode == "values":
        return LEDGER_FIELD_COUNT
    if mode == "masks":
        return LEDGER_FIELD_COUNT * AVAILABILITY_COUNT
    return LEDGER_FIELD_COUNT * (1 + AVAILABILITY_COUNT)


def _validate_batched_ledger(
    values: torch.Tensor, availability: torch.Tensor, batch_size: int,
) -> None:
    if tuple(values.shape) != (batch_size, 2, LEDGER_FIELD_COUNT):
        raise AdvantageM1InputError("batch ledger values shape が不正です")
    expected = (batch_size, 2, LEDGER_FIELD_COUNT, AVAILABILITY_COUNT)
    if tuple(availability.shape) != expected:
        raise AdvantageM1InputError("batch ledger availability shape が不正です")
    if values.dtype != torch.float32 or availability.dtype != torch.float32:
        raise AdvantageM1InputError("ledger tensor は float32 必須です")


def _validate_side_ledger(values: torch.Tensor, availability: torch.Tensor) -> None:
    batch_size = values.shape[0]
    if tuple(values.shape) != (batch_size, LEDGER_FIELD_COUNT):
        raise AdvantageM1InputError("side ledger values shape が不正です")
    if tuple(availability.shape) != (batch_size, LEDGER_FIELD_COUNT, AVAILABILITY_COUNT):
        raise AdvantageM1InputError("side ledger availability shape が不正です")
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(availability).all()):
        raise AdvantageM1InputError("ledger tensor は有限値必須です")
    if values.numel() and (values.min().item() < 0.0 or values.max().item() > 1.0):
        raise AdvantageM1InputError("ledger 値は 0〜1 必須です")
    if not bool(((availability == 0.0) | (availability == 1.0)).all()):
        raise AdvantageM1InputError("availability は 0/1 の厳密 one-hot 必須です")
    expected = torch.ones_like(availability.sum(dim=-1))
    if not bool(torch.equal(availability.sum(dim=-1), expected)):
        raise AdvantageM1InputError("availability は厳密 one-hot 必須です")


# 学習と本番は必ず同一 pure tensorizer を呼ぶ。
advantage_m1_inputs_for_training = advantage_m1_inputs_from_canonical
advantage_m1_inputs_for_serving = advantage_m1_inputs_from_canonical


__all__ = [
    "AVAILABILITY_COUNT", "AVAILABILITY_ORDER", "AdvantageM1CausalLedgerCNNV3",
    "AdvantageM1InputError", "AdvantageM1InputsV3", "AdvantageM1PrimarySideV3",
    "AdvantageM1PrimaryV3", "CausalLedgerEncoderV3",
    "LEDGER_EMBED_DIM", "LEDGER_FIELD_COUNT", "LEDGER_SIDE_FIELDS",
    "M1_INPUT_SCHEMA_VERSION", "M1_MODEL_VERSION", "PAIR_HIDDEN_DIM",
    "PrimaryBoolV3", "PrimaryIntV3",
    "advantage_m1_primary_from_canonical",
    "advantage_m1_primary_from_materialized_state",
    "advantage_m1_inputs_for_serving", "advantage_m1_inputs_for_training",
    "advantage_m1_inputs_from_canonical", "advantage_m1_inputs_from_materialized_state",
    "materialized_ledger_is_usable", "tensorize_primary",
]
