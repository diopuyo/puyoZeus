"""M0盤面CNNへraw causal ledgerを単一branchで加えるM1。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.advantage_m0_current_cnn_v1 import (
    AdvantageM0CurrentCNNV2,
    AdvantageM0InputError,
    AdvantageM0OutputV1,
    COLOR_INVARIANT_EMBED_DIM,
    ColorInvariantSideEncoderV1,
    board_categories_from_raw,
    queue_categories_from_raw,
)
from src.canonical_observation_v2 import (
    AvailabilityState,
    AvailableBool,
    AvailableInt,
    AvailableSide,
    CanonicalObservationV2,
)


M1_MODEL_VERSION = "advantage-m1-causal-ledger-cnn/v1"
M1_FROZEN_RESIDUAL_MODEL_VERSION = "advantage-m1-frozen-ledger-residual/v1"
AVAILABILITY_ORDER = tuple(AvailabilityState)
AVAILABILITY_COUNT = len(AVAILABILITY_ORDER)
LEDGER_MODE = Literal["values", "masks", "values_and_masks"]
LEDGER_SIDE_FIELDS = (
    "pending_garbage", "effective_rate", "chain_active", "chain_step",
    "provisional_generated", "provisional_score", "provisional_chain_count",
    "finalized_unsettled_attack", "post_cancel_residual", "send_waiting",
    "placement_waiting", "chain_end_waiting", "garbage_falling",
    "first_drop_amount", "leftover_after_first_drop",
)
LEDGER_CONTEXT_FIELDS = ("active_chain_side", "recipient")
LEDGER_FIELD_COUNT = len(LEDGER_SIDE_FIELDS) + len(LEDGER_CONTEXT_FIELDS)
LEDGER_EMBED_DIM = 24
PAIR_HIDDEN_DIM = 64
NORMALIZATION_SCALES = {
    "pending_garbage": 72.0, "effective_rate": 70.0, "chain_step": 5.0,
    "provisional_generated": 72.0, "provisional_score": 1000.0,
    "provisional_chain_count": 5.0, "finalized_unsettled_attack": 72.0,
    "post_cancel_residual": 72.0, "first_drop_amount": 30.0,
    "leftover_after_first_drop": 72.0,
}


class AdvantageM1InputError(ValueError):
    """canonical観測またはledger tensorがM1契約を満たさない。"""


@dataclass(frozen=True, slots=True)
class AdvantageM1InputsV1:
    """一観測分の学習・本番共通M1 tensor原本。"""

    boards: np.ndarray
    queues: np.ndarray
    ledger_values: np.ndarray
    ledger_availability: np.ndarray


class CausalLedgerEncoderV1(nn.Module):
    """raw値と欠測状態を、左右共有の一branchでencodeする。"""

    def __init__(self, mode: LEDGER_MODE = "values_and_masks") -> None:
        super().__init__()
        if mode not in {"values", "masks", "values_and_masks"}:
            raise AdvantageM1InputError(f"未対応ledger modeです: {mode}")
        self.mode = mode
        width = _ledger_input_width(mode)
        self.network = nn.Sequential(
            nn.Linear(width, 48), nn.ReLU(),
            nn.Linear(48, LEDGER_EMBED_DIM), nn.ReLU(),
        )

    def forward(self, values: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        _validate_ledger_tensors(values, availability)
        if self.mode == "values":
            inputs = values
        elif self.mode == "masks":
            inputs = availability.flatten(start_dim=1)
        else:
            inputs = torch.cat((values, availability.flatten(start_dim=1)), dim=1)
        return self.network(inputs)


class AdvantageM1CausalLedgerCNNV1(nn.Module):
    """current盤面と因果台帳を読む、左右交換反対称モデル。"""

    model_version = M1_MODEL_VERSION

    def __init__(self, ledger_mode: LEDGER_MODE = "values_and_masks") -> None:
        super().__init__()
        self.side_encoder = ColorInvariantSideEncoderV1()
        self.ledger_encoder = CausalLedgerEncoderV1(ledger_mode)
        side_width = COLOR_INVARIANT_EMBED_DIM + LEDGER_EMBED_DIM
        self.scorer = nn.Sequential(
            nn.Linear(side_width * 3, PAIR_HIDDEN_DIM), nn.ReLU(),
            nn.Linear(PAIR_HIDDEN_DIM, 1),
        )

    def forward(
        self, boards: torch.Tensor, queues: torch.Tensor,
        ledger_values: torch.Tensor, ledger_availability: torch.Tensor,
    ) -> AdvantageM0OutputV1:
        """4入力を左右共有encoderへ通し、反対称確率を返す。"""

        batch_size = boards.shape[0]
        board_side = self.side_encoder(
            boards.flatten(0, 1), queues.flatten(0, 1),
        ).reshape(batch_size, 2, COLOR_INVARIANT_EMBED_DIM)
        _validate_batched_ledger(ledger_values, ledger_availability, batch_size)
        ledger_side = self.ledger_encoder(
            ledger_values.flatten(0, 1), ledger_availability.flatten(0, 1),
        ).reshape(batch_size, 2, LEDGER_EMBED_DIM)
        sides = torch.cat((board_side, ledger_side), dim=2)
        direct = self._score_pair(sides[:, 0], sides[:, 1])
        swapped = self._score_pair(sides[:, 1], sides[:, 0])
        logit = 0.5 * (direct - swapped)
        return AdvantageM0OutputV1(logit, torch.sigmoid(logit))

    def _score_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        features = torch.cat((left, right, left - right), dim=1)
        return self.scorer(features).squeeze(1)


class AdvantageM1FrozenResidualCNNV1(nn.Module):
    """同fold M0を固定し、因果ledgerの反対称残差だけを学習する。"""

    model_version = M1_FROZEN_RESIDUAL_MODEL_VERSION

    def __init__(
        self, m0: AdvantageM0CurrentCNNV2,
        ledger_mode: LEDGER_MODE = "values_and_masks",
    ) -> None:
        super().__init__()
        self.m0 = m0
        self.m0.requires_grad_(False)
        self.m0.eval()
        self.ledger_encoder = CausalLedgerEncoderV1(ledger_mode)
        side_width = COLOR_INVARIANT_EMBED_DIM + LEDGER_EMBED_DIM
        self.residual_scorer = nn.Sequential(
            nn.Linear(side_width * 3, PAIR_HIDDEN_DIM), nn.ReLU(),
            nn.Linear(PAIR_HIDDEN_DIM, 1),
        )
        nn.init.zeros_(self.residual_scorer[-1].weight)
        nn.init.zeros_(self.residual_scorer[-1].bias)

    def train(self, mode: bool = True) -> AdvantageM1FrozenResidualCNNV1:
        super().train(mode)
        self.m0.eval()
        return self

    def forward(
        self, boards: torch.Tensor, queues: torch.Tensor,
        ledger_values: torch.Tensor, ledger_availability: torch.Tensor,
    ) -> AdvantageM0OutputV1:
        batch_size = boards.shape[0]
        _validate_batched_ledger(ledger_values, ledger_availability, batch_size)
        with torch.no_grad():
            base = self.m0(boards, queues)
            board_side = self.m0.side_encoder(
                boards.flatten(0, 1), queues.flatten(0, 1),
            ).reshape(batch_size, 2, COLOR_INVARIANT_EMBED_DIM)
        ledger_side = self.ledger_encoder(
            ledger_values.flatten(0, 1), ledger_availability.flatten(0, 1),
        ).reshape(batch_size, 2, LEDGER_EMBED_DIM)
        sides = torch.cat((board_side, ledger_side), dim=2)
        residual = 0.5 * (
            self._score_pair(sides[:, 0], sides[:, 1])
            - self._score_pair(sides[:, 1], sides[:, 0])
        )
        gated = torch.where(_ledger_rows_usable(ledger_availability), residual, 0.0)
        logit = base.logit + gated
        return AdvantageM0OutputV1(logit, torch.sigmoid(logit))

    def _score_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        features = torch.cat((left, right, left - right), dim=1)
        return self.residual_scorer(features).squeeze(1)


def _ledger_rows_usable(availability: torch.Tensor) -> torch.Tensor:
    """両側pendingが既知の行だけ残差を有効にする。"""

    known = AVAILABILITY_ORDER.index(AvailabilityState.KNOWN)
    known_zero = AVAILABILITY_ORDER.index(AvailabilityState.KNOWN_ZERO)
    pending = availability[:, :, 0]
    usable_side = (pending[:, :, known] + pending[:, :, known_zero]) > 0.5
    return usable_side.all(dim=1)


def advantage_m1_inputs_from_canonical(
    observation: CanonicalObservationV2,
) -> AdvantageM1InputsV1:
    """CanonicalObservationV2だけからM1入力を決定論的に作る。"""

    boards = np.stack([_board(observation, side) for side in ("p1", "p2")])
    queues = np.stack([_queue(observation, side) for side in ("p1", "p2")])
    values, masks = _ledger_arrays(observation)
    return AdvantageM1InputsV1(boards, queues, values, masks)


def advantage_m1_inputs_from_materialized_state(
    state: Mapping[str, Any],
) -> AdvantageM1InputsV1:
    """検査済みevent state表をcanonical tensorとbit-identicalに復元する。"""

    boards = np.stack([_materialized_board(state, side) for side in ("p1", "p2")])
    queues = np.stack([_materialized_queue(state, side) for side in ("p1", "p2")])
    values, masks = _materialized_ledger_arrays(state)
    return AdvantageM1InputsV1(boards, queues, values, masks)


def _board(observation: CanonicalObservationV2, side: str) -> np.ndarray:
    board = getattr(observation, side).board
    if board.availability != AvailabilityState.KNOWN or board.grid is None:
        raise AdvantageM1InputError(f"{side} STABLE盤面がknownではありません")
    return board_categories_from_raw(np.asarray(board.grid, dtype=np.int8))


def _queue(observation: CanonicalObservationV2, side: str) -> np.ndarray:
    pieces = getattr(observation, side).pieces
    values = (
        pieces.next.axis.value, pieces.next.child.value,
        pieces.double_next.axis.value, pieces.double_next.child.value,
    )
    return queue_categories_from_raw(values)


def _materialized_board(state: Mapping[str, Any], side: str) -> np.ndarray:
    grid = np.asarray(state.get(f"a_{side}_grid"), dtype=np.int8)
    if grid.size != 78:
        raise AdvantageM1InputError(f"{side} materialized盤面が78セルではありません")
    return board_categories_from_raw(grid.reshape(13, 6))


def _materialized_queue(state: Mapping[str, Any], side: str) -> np.ndarray:
    values = tuple(
        state.get(f"a_{side}_{name}") for name in (
            "next_first", "next_second", "double_next_first", "double_next_second",
        )
    )
    return queue_categories_from_raw(values)


def _ledger_arrays(observation: CanonicalObservationV2) -> tuple[np.ndarray, np.ndarray]:
    values, masks = [], []
    for side in ("p1", "p2"):
        side_values, side_masks = _ledger_side_arrays(observation, side)
        values.append(side_values)
        masks.append(side_masks)
    return np.asarray(values, dtype=np.float32), np.asarray(masks, dtype=np.float32)


def _ledger_side_arrays(
    observation: CanonicalObservationV2, side: str,
) -> tuple[list[float], list[list[float]]]:
    ledger_side = getattr(observation.ledger, side)
    values, masks = [], []
    for name in LEDGER_SIDE_FIELDS:
        item = getattr(ledger_side, name)
        values.append(_available_value(item, name))
        masks.append(_availability_mask(item.availability))
    for name in LEDGER_CONTEXT_FIELDS:
        item = getattr(observation.ledger, name)
        values.append(_side_membership(item, side))
        masks.append(_availability_mask(item.availability))
    return values, masks


def _materialized_ledger_arrays(
    state: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    output = [_materialized_ledger_side(state, side) for side in ("p1", "p2")]
    values = np.asarray([item[0] for item in output], dtype=np.float32)
    masks = np.asarray([item[1] for item in output], dtype=np.float32)
    return values, masks


def _materialized_ledger_side(
    state: Mapping[str, Any], side: str,
) -> tuple[list[float], list[list[float]]]:
    usable = materialized_ledger_is_usable(state)
    raw = _materialized_raw_fields(state, side)
    values, masks = [], []
    for name in LEDGER_SIDE_FIELDS:
        value, availability = _materialized_availability(name, raw[name], usable)
        values.append(_normalized_raw_value(value, name))
        masks.append(_availability_mask(availability))
    for name, item in _materialized_context_fields(state, side, usable).items():
        value, availability = item
        values.append(float(value))
        masks.append(_availability_mask(availability))
    return values, masks


def materialized_ledger_is_usable(state: Mapping[str, Any]) -> bool:
    """学習表の品質gateを含め、因果ledgerを入力可能か判定する。"""

    return bool(
        state.get("b_causal_exchange_usable") is True
        and state.get("b_input_usable") is True
    )


def _materialized_raw_fields(
    state: Mapping[str, Any], side: str,
) -> dict[str, int | bool | None]:
    balance = state.get("b_causal_observed_attack_balance")
    residual = _materialized_residual(balance, side)
    pending = state.get(f"b_{side}_causal_pending_garbage")
    chain_active = bool(state.get(f"b_{side}_chain_active"))
    chain_count = _provisional_value(state, side, "chain_count", chain_active)
    return {
        "pending_garbage": pending,
        "effective_rate": state.get(f"b_{side}_causal_effective_rate"),
        "chain_active": chain_active,
        "chain_step": chain_count,
        "provisional_generated": _provisional_value(
            state, side, "generated", chain_active,
        ),
        "provisional_score": _provisional_value(state, side, "score", chain_active),
        "provisional_chain_count": chain_count,
        "finalized_unsettled_attack": pending,
        "post_cancel_residual": residual,
        "send_waiting": None, "placement_waiting": None,
        "chain_end_waiting": None, "garbage_falling": None,
        "first_drop_amount": None if residual is None else min(residual, 30),
        "leftover_after_first_drop": None if residual is None else max(0, residual - 30),
    }


def _provisional_value(
    state: Mapping[str, Any], side: str, name: str, chain_active: bool,
) -> int | None:
    if not chain_active:
        return 0
    value = state.get(f"b_{side}_provisional_{name}")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _materialized_residual(value: Any, side: str) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return max(0, -value) if side == "p1" else max(0, value)


def _materialized_availability(
    name: str, value: int | bool | None, causal_usable: bool,
) -> tuple[int | bool | None, AvailabilityState]:
    causal = {
        "pending_garbage", "finalized_unsettled_attack", "post_cancel_residual",
        "first_drop_amount", "leftover_after_first_drop",
    }
    if name in causal and not causal_usable:
        return None, AvailabilityState.UNSUPPORTED
    if value is None:
        return None, AvailabilityState.UNKNOWN
    zero = value is False if isinstance(value, bool) else value == 0
    return value, AvailabilityState.KNOWN_ZERO if zero else AvailabilityState.KNOWN


def _materialized_context_fields(
    state: Mapping[str, Any], side: str, usable: bool,
) -> dict[str, tuple[float, AvailabilityState]]:
    p1_active, p2_active = bool(state.get("b_p1_chain_active")), bool(
        state.get("b_p2_chain_active")
    )
    active = p1_active if side == "p1" else p2_active
    active_state = AvailabilityState.KNOWN if active else AvailabilityState.KNOWN_ZERO
    balance = state.get("b_causal_observed_attack_balance")
    if not usable or not isinstance(balance, int) or isinstance(balance, bool):
        recipient = (0.0, AvailabilityState.UNSUPPORTED)
    else:
        is_recipient = (balance < 0 and side == "p1") or (balance > 0 and side == "p2")
        state_value = AvailabilityState.KNOWN if is_recipient else AvailabilityState.KNOWN_ZERO
        recipient = float(is_recipient), state_value
    return {"active_chain_side": (float(active), active_state), "recipient": recipient}


def _normalized_raw_value(value: int | bool | None, name: str) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    scale = NORMALIZATION_SCALES.get(name, 1.0)
    return float(value) / (float(value) + scale)


def _available_value(item: AvailableInt | AvailableBool, name: str) -> float:
    if item.value is None:
        return 0.0
    if isinstance(item, AvailableBool):
        return float(item.value)
    scale = NORMALIZATION_SCALES.get(name, 1.0)
    return float(item.value) / (float(item.value) + scale)


def _side_membership(item: AvailableSide, side: str) -> float:
    if item.value is None:
        return 0.0
    return float(item.value in {side, "both"})


def _availability_mask(state: AvailabilityState) -> list[float]:
    return [float(state == candidate) for candidate in AVAILABILITY_ORDER]


def _ledger_input_width(mode: LEDGER_MODE) -> int:
    if mode == "values":
        return LEDGER_FIELD_COUNT
    if mode == "masks":
        return LEDGER_FIELD_COUNT * AVAILABILITY_COUNT
    return LEDGER_FIELD_COUNT * (1 + AVAILABILITY_COUNT)


def _validate_batched_ledger(
    values: torch.Tensor, availability: torch.Tensor, batch_size: int,
) -> None:
    expected_values = (batch_size, 2, LEDGER_FIELD_COUNT)
    expected_masks = (batch_size, 2, LEDGER_FIELD_COUNT, AVAILABILITY_COUNT)
    if tuple(values.shape) != expected_values or tuple(availability.shape) != expected_masks:
        raise AdvantageM1InputError("batch ledger tensor shapeが不正です")
    if values.dtype != torch.float32 or availability.dtype != torch.float32:
        raise AdvantageM1InputError("ledger tensorはfloat32必須です")


def _validate_ledger_tensors(values: torch.Tensor, availability: torch.Tensor) -> None:
    expected_values = (values.shape[0], LEDGER_FIELD_COUNT)
    expected_masks = (values.shape[0], LEDGER_FIELD_COUNT, AVAILABILITY_COUNT)
    if tuple(values.shape) != expected_values or tuple(availability.shape) != expected_masks:
        raise AdvantageM1InputError("side ledger tensor shapeが不正です")
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(availability).all()):
        raise AdvantageM1InputError("ledger tensorは有限値必須です")
    if values.numel() and (values.min().item() < 0.0 or values.max().item() > 1.0):
        raise AdvantageM1InputError("ledger値は0〜1必須です")


advantage_m1_inputs_for_training = advantage_m1_inputs_from_canonical
advantage_m1_inputs_for_serving = advantage_m1_inputs_from_canonical


__all__ = [
    "AVAILABILITY_COUNT", "AdvantageM1CausalLedgerCNNV1",
    "AdvantageM1FrozenResidualCNNV1", "AdvantageM1InputError",
    "AdvantageM1InputsV1", "CausalLedgerEncoderV1", "LEDGER_FIELD_COUNT",
    "LEDGER_SIDE_FIELDS", "M1_FROZEN_RESIDUAL_MODEL_VERSION", "M1_MODEL_VERSION",
    "advantage_m1_inputs_for_serving",
    "advantage_m1_inputs_for_training", "advantage_m1_inputs_from_canonical",
    "advantage_m1_inputs_from_materialized_state", "materialized_ledger_is_usable",
]
