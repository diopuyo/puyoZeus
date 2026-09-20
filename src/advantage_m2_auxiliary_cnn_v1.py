"""M1因果ledgerへ材料・空間8補助教師を加えるM2本命候補。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.advantage_m0_current_cnn_v1 import (
    COLOR_INVARIANT_EMBED_DIM,
    ColorInvariantSideEncoderV1,
)
from src.advantage_m1_causal_ledger_v3 import (
    AVAILABILITY_COUNT,
    AVAILABILITY_ORDER,
    LEDGER_EMBED_DIM,
    LEDGER_FIELD_COUNT,
    AdvantageM1PrimaryV3,
    CausalLedgerEncoderV3,
    PrimaryBoolV3,
    advantage_m1_primary_from_canonical,
    advantage_m1_primary_from_materialized_state,
    tensorize_primary as tensorize_m1_primary,
)
from src.advantage_m1_zero_counterfactual_v3 import (
    AdvantageM1ZeroCounterfactualV3,
    zero_counterfactual_ledger_v3,
)
from src.board import Board
from src.canonical_observation_v3 import AvailabilityState, CanonicalObservationV3
from src.projected_state_tensorizer_v1 import (
    AUXILIARY_FEATURE_NAMES,
    tensorize_board_auxiliary_targets_v1,
)


M2_INPUT_SCHEMA_VERSION = "m2-auxiliary-ledger-input/v1"
M2_MODEL_VERSION = "advantage-m2-auxiliary-ledger-cnn/v1"
M2_ANCHORED_MODEL_VERSION = "advantage-m2-anchored-auxiliary-residual/v2"
M2_TWO_STAGE_MODEL_VERSION = "advantage-m2-two-stage-auxiliary-linear/v3"
M2_AUXILIARY_LOSS_COEFFICIENT = 0.3
ALL_CLEAR_EMBED_DIM = 8
BASE_SIDE_DIM = COLOR_INVARIANT_EMBED_DIM + ALL_CLEAR_EMBED_DIM
PAIR_HIDDEN_DIM = 64
RAW_VALUES_BY_CATEGORY = (0, 1, 2, 3, 4, 5, 9, 10)
_KNOWN_INDEX = AVAILABILITY_ORDER.index(AvailabilityState.KNOWN)
_KNOWN_ZERO_INDEX = AVAILABILITY_ORDER.index(AvailabilityState.KNOWN_ZERO)
_FAULT_INDEX = AVAILABILITY_ORDER.index(AvailabilityState.INTEGRITY_FAULT)


class AdvantageM2InputError(ValueError):
    """M2入力または補助教師が固定契約に違反した。"""


class AdvantageM2IntegrityError(RuntimeError):
    """fallbackで隠してはならないM2入力の完全性異常。"""


@dataclass(frozen=True, slots=True)
class AdvantageM2InputsV1:
    """学習・本番が共有するM2 model入力。"""

    boards: np.ndarray
    queues: np.ndarray
    all_clear_values: np.ndarray
    all_clear_availability: np.ndarray
    ledger_values: np.ndarray
    ledger_availability: np.ndarray
    schema_version: str = M2_INPUT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class AdvantageM2AuxiliaryTargetsV1:
    """盤面単体から作る材料・空間8補助教師。"""

    values: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True, slots=True)
class AdvantageM2PrimaryV1:
    """canonical/materialized共通tensorizerへ渡す一次DTO。"""

    m1: AdvantageM1PrimaryV3
    p1_all_clear: PrimaryBoolV3
    p2_all_clear: PrimaryBoolV3


@dataclass(frozen=True, slots=True)
class AdvantageM2OutputV1:
    """反対称勝率と、勝率headへ戻さないside別補助出力。"""

    logit: torch.Tensor
    raw_probability: torch.Tensor
    auxiliary: torch.Tensor


@dataclass(frozen=True, slots=True)
class AdvantageM2LossV1:
    """M2学習lossの分離記録。"""

    total: torch.Tensor
    binary_cross_entropy: torch.Tensor
    auxiliary_mse: torch.Tensor


class AdvantageM2AuxiliaryCNNV1(nn.Module):
    """current board、全消し、typed ledgerを補助学習つきで評価する。"""

    model_version = M2_MODEL_VERSION

    def __init__(self) -> None:
        super().__init__()
        self.side_encoder = ColorInvariantSideEncoderV1()
        self.all_clear_encoder = nn.Sequential(
            nn.Linear(1 + AVAILABILITY_COUNT, ALL_CLEAR_EMBED_DIM), nn.ReLU(),
        )
        self.base_scorer = _pair_scorer(BASE_SIDE_DIM)
        self.ledger_encoder = CausalLedgerEncoderV3("values_and_masks")
        self.effect_scorer = _pair_scorer(BASE_SIDE_DIM + LEDGER_EMBED_DIM)
        self.auxiliary_head = nn.Sequential(
            nn.Linear(COLOR_INVARIANT_EMBED_DIM, len(AUXILIARY_FEATURE_NAMES)),
            nn.Sigmoid(),
        )

    def forward(
        self, boards: torch.Tensor, queues: torch.Tensor,
        all_clear_values: torch.Tensor, all_clear_availability: torch.Tensor,
        ledger_values: torch.Tensor, ledger_availability: torch.Tensor,
        supported: torch.Tensor | None = None,
    ) -> AdvantageM2OutputV1:
        """unsupported ledgerはcurrent-state基礎値へbit-identicalに戻す。"""

        _validate_model_inputs(
            boards, queues, all_clear_values, all_clear_availability,
            ledger_values, ledger_availability,
        )
        _raise_on_integrity_fault(all_clear_availability, ledger_availability)
        active = _supported_rows(ledger_availability, supported)
        board_side = self._encode_board(boards, queues)
        base_side = self._encode_base(board_side, all_clear_values, all_clear_availability)
        base_logit = self._antisymmetric_score(self.base_scorer, base_side)
        residual = self._supported_residual(base_side, ledger_values, ledger_availability, active)
        logit = base_logit + residual
        return AdvantageM2OutputV1(
            logit, torch.sigmoid(logit), self.auxiliary_head(board_side),
        )

    def _encode_board(self, boards: torch.Tensor, queues: torch.Tensor) -> torch.Tensor:
        batch_size = boards.shape[0]
        return self.side_encoder(
            boards.flatten(0, 1), queues.flatten(0, 1),
        ).reshape(batch_size, 2, COLOR_INVARIANT_EMBED_DIM)

    def _encode_base(
        self, board: torch.Tensor, values: torch.Tensor, availability: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.all_clear_encoder(torch.cat((values.unsqueeze(-1), availability), dim=-1))
        return torch.cat((board, encoded), dim=-1)

    def _supported_residual(
        self, base: torch.Tensor, values: torch.Tensor,
        availability: torch.Tensor, active: torch.Tensor,
    ) -> torch.Tensor:
        indices = active.nonzero(as_tuple=False).flatten()
        output = base.new_zeros((base.shape[0],))
        if not indices.numel():
            return output
        selected_base = base.index_select(0, indices)
        selected_values = values.index_select(0, indices)
        selected_masks = availability.index_select(0, indices)
        residual = self._ledger_residual(selected_base, selected_values, selected_masks)
        return output.index_copy(0, indices, residual)

    def _ledger_residual(
        self, base: torch.Tensor, values: torch.Tensor, availability: torch.Tensor,
    ) -> torch.Tensor:
        zero_values, zero_masks = zero_counterfactual_ledger_v3(
            values, availability, "values_and_masks",
        )
        actual = self._encode_ledger(values, availability)
        zero = self._encode_ledger(zero_values, zero_masks)
        direct = self._effect_difference(base, actual, zero)
        swapped = self._effect_difference(base.flip(1), actual.flip(1), zero.flip(1))
        return 0.5 * (direct - swapped)

    def _encode_ledger(
        self, values: torch.Tensor, availability: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = values.shape[0]
        return self.ledger_encoder(
            values.flatten(0, 1), availability.flatten(0, 1),
        ).reshape(batch_size, 2, LEDGER_EMBED_DIM)

    def _effect_difference(
        self, base: torch.Tensor, actual: torch.Tensor, zero: torch.Tensor,
    ) -> torch.Tensor:
        return self._score_effect(base, actual) - self._score_effect(base, zero)

    def _score_effect(self, base: torch.Tensor, ledger: torch.Tensor) -> torch.Tensor:
        return _pair_score(self.effect_scorer, torch.cat((base, ledger), dim=-1))

    @staticmethod
    def _antisymmetric_score(scorer: nn.Module, sides: torch.Tensor) -> torch.Tensor:
        direct = _pair_score(scorer, sides)
        swapped = _pair_score(scorer, sides.flip(1))
        return 0.5 * (direct - swapped)


class AdvantageM2AnchoredAuxiliaryCNNV2(nn.Module):
    """凍結M1を保持し、補助共有encoderの改善分だけを上乗せする。"""

    model_version = M2_ANCHORED_MODEL_VERSION

    def __init__(self, anchor: AdvantageM1ZeroCounterfactualV3) -> None:
        super().__init__()
        if not isinstance(anchor, AdvantageM1ZeroCounterfactualV3):
            raise AdvantageM2InputError("anchorはM1 zero-counterfactual V3必須です")
        self.anchor = anchor
        self.anchor.requires_grad_(False)
        self.anchor.eval()
        self.adapted_side_encoder = copy.deepcopy(anchor.m0.side_encoder)
        self.adapted_side_encoder.requires_grad_(True)
        self.all_clear_encoder = nn.Sequential(
            nn.Linear(1 + AVAILABILITY_COUNT, ALL_CLEAR_EMBED_DIM), nn.ReLU(),
        )
        self.residual_scorer = _pair_scorer(BASE_SIDE_DIM)
        nn.init.zeros_(self.residual_scorer[-1].weight)
        nn.init.zeros_(self.residual_scorer[-1].bias)
        self.auxiliary_head = nn.Sequential(
            nn.Linear(COLOR_INVARIANT_EMBED_DIM, len(AUXILIARY_FEATURE_NAMES)),
            nn.Sigmoid(),
        )

    def train(self, mode: bool = True) -> "AdvantageM2AnchoredAuxiliaryCNNV2":
        super().train(mode)
        self.anchor.eval()
        return self

    def forward(
        self, boards: torch.Tensor, queues: torch.Tensor,
        all_clear_values: torch.Tensor, all_clear_availability: torch.Tensor,
        ledger_values: torch.Tensor, ledger_availability: torch.Tensor,
        supported: torch.Tensor | None = None,
    ) -> AdvantageM2OutputV1:
        """epoch 0はM1とbit-identicalで、残差は常に反対称とする。"""

        _validate_model_inputs(
            boards, queues, all_clear_values, all_clear_availability,
            ledger_values, ledger_availability,
        )
        _raise_on_integrity_fault(all_clear_availability, ledger_availability)
        baseline = self.anchor(
            boards, queues, ledger_values, ledger_availability, supported,
        )
        board_side = self._encode_board(boards, queues)
        clear_side = self.all_clear_encoder(torch.cat(
            (all_clear_values.unsqueeze(-1), all_clear_availability), dim=-1,
        ))
        sides = torch.cat((board_side, clear_side), dim=-1)
        residual = AdvantageM2AuxiliaryCNNV1._antisymmetric_score(
            self.residual_scorer, sides,
        )
        logit = baseline.logit + residual
        return AdvantageM2OutputV1(
            logit, torch.sigmoid(logit), self.auxiliary_head(board_side),
        )

    def _encode_board(self, boards: torch.Tensor, queues: torch.Tensor) -> torch.Tensor:
        batch_size = boards.shape[0]
        return self.adapted_side_encoder(
            boards.flatten(0, 1), queues.flatten(0, 1),
        ).reshape(batch_size, 2, COLOR_INVARIANT_EMBED_DIM)


class AdvantageM2TwoStageAuxiliaryLinearV3(nn.Module):
    """補助事前学習済みencoderを固定し、小容量の勝率残差だけを学ぶ。"""

    model_version = M2_TWO_STAGE_MODEL_VERSION

    def __init__(self, anchor: AdvantageM1ZeroCounterfactualV3) -> None:
        super().__init__()
        if not isinstance(anchor, AdvantageM1ZeroCounterfactualV3):
            raise AdvantageM2InputError("anchorはM1 zero-counterfactual V3必須です")
        self.anchor = anchor
        self.anchor.requires_grad_(False)
        self.anchor.eval()
        self.adapted_side_encoder = copy.deepcopy(anchor.m0.side_encoder)
        self.adapted_side_encoder.requires_grad_(True)
        self.all_clear_encoder = nn.Sequential(
            nn.Linear(1 + AVAILABILITY_COUNT, ALL_CLEAR_EMBED_DIM), nn.ReLU(),
        )
        self.residual_utility = nn.Linear(BASE_SIDE_DIM, 1)
        nn.init.zeros_(self.residual_utility.weight)
        nn.init.zeros_(self.residual_utility.bias)
        self.auxiliary_head = nn.Sequential(
            nn.Linear(COLOR_INVARIANT_EMBED_DIM, len(AUXILIARY_FEATURE_NAMES)),
            nn.Sigmoid(),
        )

    def train(self, mode: bool = True) -> "AdvantageM2TwoStageAuxiliaryLinearV3":
        super().train(mode)
        self.anchor.eval()
        return self

    def predict_auxiliary(
        self, boards: torch.Tensor, queues: torch.Tensor,
    ) -> torch.Tensor:
        """補助事前学習では勝率branchを計算しない。"""

        _validate_board_queue_inputs(boards, queues)
        return self.auxiliary_head(self._encode_board(boards, queues))

    def freeze_auxiliary_encoder(self) -> None:
        """補助学習後の表現を勝率labelから隔離して固定する。"""

        self.adapted_side_encoder.requires_grad_(False)
        self.auxiliary_head.requires_grad_(False)

    def forward(
        self, boards: torch.Tensor, queues: torch.Tensor,
        all_clear_values: torch.Tensor, all_clear_availability: torch.Tensor,
        ledger_values: torch.Tensor, ledger_availability: torch.Tensor,
        supported: torch.Tensor | None = None,
    ) -> AdvantageM2OutputV1:
        _validate_model_inputs(
            boards, queues, all_clear_values, all_clear_availability,
            ledger_values, ledger_availability,
        )
        _raise_on_integrity_fault(all_clear_availability, ledger_availability)
        baseline = self.anchor(
            boards, queues, ledger_values, ledger_availability, supported,
        )
        board_side = self._encode_board(boards, queues)
        clear_side = self.all_clear_encoder(torch.cat(
            (all_clear_values.unsqueeze(-1), all_clear_availability), dim=-1,
        ))
        sides = torch.cat((board_side, clear_side), dim=-1)
        utility = self.residual_utility(sides).squeeze(-1)
        logit = baseline.logit + utility[:, 0] - utility[:, 1]
        return AdvantageM2OutputV1(
            logit, torch.sigmoid(logit), self.auxiliary_head(board_side),
        )

    def _encode_board(self, boards: torch.Tensor, queues: torch.Tensor) -> torch.Tensor:
        batch_size = boards.shape[0]
        return self.adapted_side_encoder(
            boards.flatten(0, 1), queues.flatten(0, 1),
        ).reshape(batch_size, 2, COLOR_INVARIANT_EMBED_DIM)


def advantage_m2_inputs_from_canonical(
    observation: CanonicalObservationV3,
) -> AdvantageM2InputsV1:
    """canonical観測を共通一次DTO経由でM2入力へ変換する。"""

    primary = advantage_m1_primary_from_canonical(observation)
    all_clear = tuple(
        PrimaryBoolV3(getattr(observation, side).all_clear_pending.value,
                      getattr(observation, side).all_clear_pending.availability)
        for side in ("p1", "p2")
    )
    return tensorize_m2_primary(AdvantageM2PrimaryV1(primary, *all_clear))


def advantage_m2_inputs_from_materialized_state(
    state: Mapping[str, Any],
) -> AdvantageM2InputsV1:
    """event学習表をcanonicalと同じM2 tensorizerへ通す。"""

    primary = advantage_m1_primary_from_materialized_state(state)
    all_clear = _materialized_all_clear_primary(state)
    return tensorize_m2_primary(AdvantageM2PrimaryV1(primary, *all_clear))


def tensorize_materialized_all_clear_v1(
    state: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """event表の全消し2値をcanonical共通規約でtensor化する。"""

    rows = tuple(_tensorize_all_clear(item) for item in _materialized_all_clear_primary(state))
    values = np.asarray([row[0] for row in rows], dtype=np.float32)
    availability = np.asarray([row[1] for row in rows], dtype=np.float32)
    return values, availability


def tensorize_m2_primary(primary: AdvantageM2PrimaryV1) -> AdvantageM2InputsV1:
    """一次DTOを唯一のM2正規化・availability生成経路でtensor化する。"""

    if not isinstance(primary, AdvantageM2PrimaryV1):
        raise AdvantageM2InputError("primary DTOの型が不正です")
    m1 = tensorize_m1_primary(primary.m1)
    all_clear = tuple(_tensorize_all_clear(value) for value in (
        primary.p1_all_clear, primary.p2_all_clear,
    ))
    values = np.asarray([item[0] for item in all_clear], dtype=np.float32)
    availability = np.asarray([item[1] for item in all_clear], dtype=np.float32)
    return AdvantageM2InputsV1(
        m1.boards, m1.queues, values, availability,
        m1.ledger_values, m1.ledger_availability,
    )


def auxiliary_targets_from_m2_inputs(
    inputs: AdvantageM2InputsV1,
) -> AdvantageM2AuxiliaryTargetsV1:
    """M2 category盤面から凍結済み8補助教師を作る。"""

    if not isinstance(inputs, AdvantageM2InputsV1):
        raise AdvantageM2InputError("M2入力の型が不正です")
    pairs = [auxiliary_target_from_category_board_v1(grid) for grid in inputs.boards]
    values = np.stack([pair[0] for pair in pairs]).astype(np.float32)
    masks = np.stack([pair[1] for pair in pairs]).astype(np.bool_)
    return AdvantageM2AuxiliaryTargetsV1(values, masks)


def auxiliary_target_from_category_board_v1(
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """M2 category盤面1面を凍結済み8補助教師へ変換する。"""

    return tensorize_board_auxiliary_targets_v1(_board_from_categories(grid))


def advantage_m2_weighted_loss(
    output: AdvantageM2OutputV1, labels: torch.Tensor, weights: torch.Tensor,
    targets: torch.Tensor, target_mask: torch.Tensor, *, auxiliary_coefficient: float,
) -> AdvantageM2LossV1:
    """試合均等BCEとside/指標平均の補助MSEを分離して返す。"""

    _validate_loss_inputs(output, labels, weights, targets, target_mask)
    if auxiliary_coefficient not in {0.0, M2_AUXILIARY_LOSS_COEFFICIENT}:
        raise AdvantageM2InputError("補助係数は事前固定した0.0または0.3だけです")
    binary_rows = F.binary_cross_entropy_with_logits(output.logit, labels, reduction="none")
    binary = (binary_rows * weights).sum() / weights.sum()
    safe_targets = torch.where(target_mask, targets, output.auxiliary.detach())
    squared = (output.auxiliary - safe_targets).square()
    present = target_mask.sum(dim=(1, 2)).clamp_min(1).to(squared.dtype)
    auxiliary_rows = torch.where(target_mask, squared, torch.zeros_like(squared)).sum((1, 2)) / present
    auxiliary = (auxiliary_rows * weights).sum() / weights.sum()
    return AdvantageM2LossV1(binary + auxiliary_coefficient * auxiliary, binary, auxiliary)


def _pair_scorer(side_width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(side_width * 3, PAIR_HIDDEN_DIM), nn.ReLU(),
        nn.Linear(PAIR_HIDDEN_DIM, 1),
    )


def _pair_score(scorer: nn.Module, sides: torch.Tensor) -> torch.Tensor:
    features = torch.cat((sides[:, 0], sides[:, 1], sides[:, 0] - sides[:, 1]), dim=1)
    return scorer(features).squeeze(1)


def _materialized_all_clear(value: Any) -> PrimaryBoolV3:
    if type(value) is bool:
        state = AvailabilityState.KNOWN if value else AvailabilityState.KNOWN_ZERO
        return PrimaryBoolV3(value, state)
    if value is None:
        return PrimaryBoolV3(None, AvailabilityState.UNKNOWN)
    return PrimaryBoolV3(None, AvailabilityState.INTEGRITY_FAULT)


def _materialized_all_clear_primary(
    state: Mapping[str, Any],
) -> tuple[PrimaryBoolV3, PrimaryBoolV3]:
    return tuple(
        _materialized_all_clear(state.get(f"a_{side}_all_clear_pending"))
        for side in ("p1", "p2")
    )


def _tensorize_all_clear(value: PrimaryBoolV3) -> tuple[float, np.ndarray]:
    encoded = np.zeros(AVAILABILITY_COUNT, dtype=np.float32)
    encoded[AVAILABILITY_ORDER.index(value.availability)] = 1.0
    return float(value.value is True), encoded


def _board_from_categories(grid: np.ndarray) -> Board:
    if grid.shape != (13, 6) or grid.dtype != np.int8:
        raise AdvantageM2InputError("category盤面はint8[13,6]必須です")
    if grid.size and (grid.min() < 0 or grid.max() >= len(RAW_VALUES_BY_CATEGORY)):
        raise AdvantageM2InputError("category盤面値が0..7ではありません")
    raw = np.asarray(RAW_VALUES_BY_CATEGORY, dtype=np.int8)[grid]
    return Board.from_list(raw.tolist())


def _validate_model_inputs(
    boards: torch.Tensor, queues: torch.Tensor, all_clear_values: torch.Tensor,
    all_clear_availability: torch.Tensor, ledger_values: torch.Tensor,
    ledger_availability: torch.Tensor,
) -> None:
    _validate_board_queue_inputs(boards, queues)
    batch = boards.shape[0]
    expected = ((batch, 2, 13, 6), (batch, 2, 4), (batch, 2),
                (batch, 2, AVAILABILITY_COUNT), (batch, 2, LEDGER_FIELD_COUNT),
                (batch, 2, LEDGER_FIELD_COUNT, AVAILABILITY_COUNT))
    actual = tuple(value.shape for value in (
        boards, queues, all_clear_values, all_clear_availability,
        ledger_values, ledger_availability,
    ))
    dtypes = (torch.int64, torch.int64) + (torch.float32,) * 4
    values = (boards, queues, all_clear_values, all_clear_availability,
              ledger_values, ledger_availability)
    if actual != expected or any(value.dtype != dtype for value, dtype in zip(values, dtypes)):
        raise AdvantageM2InputError("M2 tensorのshapeまたはdtypeが不正です")
    _validate_devices(values)
    _validate_values(boards, queues, all_clear_values, ledger_values)
    _validate_one_hot(all_clear_availability, "全消しavailability")
    _validate_one_hot(ledger_availability, "ledger availability")


def _validate_board_queue_inputs(boards: torch.Tensor, queues: torch.Tensor) -> None:
    batch = boards.shape[0]
    if boards.dtype != torch.int64 or boards.shape != (batch, 2, 13, 6):
        raise AdvantageM2InputError("boardsはint64[B,2,13,6]必須です")
    if queues.dtype != torch.int64 or queues.shape != (batch, 2, 4):
        raise AdvantageM2InputError("queuesはint64[B,2,4]必須です")
    if boards.device != queues.device:
        raise AdvantageM2InputError("board/queueのdeviceが一致しません")
    if boards.numel() and (boards.min() < 0 or boards.max() > 7):
        raise AdvantageM2InputError("board categoryは0..7必須です")
    if queues.numel() and (queues.min() < 0 or queues.max() > 5):
        raise AdvantageM2InputError("queue categoryは0..5必須です")


def _validate_devices(values: tuple[torch.Tensor, ...]) -> None:
    if len({value.device for value in values}) != 1:
        raise AdvantageM2InputError("M2 tensorのdeviceが一致しません")


def _validate_values(
    boards: torch.Tensor, queues: torch.Tensor,
    all_clear: torch.Tensor, ledger: torch.Tensor,
) -> None:
    if boards.numel() and (boards.min() < 0 or boards.max() > 7):
        raise AdvantageM2InputError("board categoryは0..7必須です")
    if queues.numel() and (queues.min() < 0 or queues.max() > 5):
        raise AdvantageM2InputError("queue categoryは0..5必須です")
    for value, label in ((all_clear, "全消し値"), (ledger, "ledger値")):
        if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
            raise AdvantageM2InputError(f"{label}は有限な0..1必須です")


def _validate_one_hot(value: torch.Tensor, label: str) -> None:
    if (not bool(torch.isfinite(value).all())
            or not bool(((value == 0.0) | (value == 1.0)).all())
            or not bool(torch.equal(value.sum(-1), torch.ones_like(value.sum(-1))))):
        raise AdvantageM2InputError(f"{label}は厳密one-hot必須です")


def _raise_on_integrity_fault(
    all_clear: torch.Tensor, ledger: torch.Tensor,
) -> None:
    if bool((all_clear[..., _FAULT_INDEX] > 0.5).any()):
        raise AdvantageM2IntegrityError("全消し状態のintegrity faultです")
    if bool((ledger[..., _FAULT_INDEX] > 0.5).any()):
        raise AdvantageM2IntegrityError("ledgerのintegrity faultです")


def _supported_rows(
    availability: torch.Tensor, supplied: torch.Tensor | None,
) -> torch.Tensor:
    pending = availability[:, :, 0]
    inferred = (pending[..., _KNOWN_INDEX] + pending[..., _KNOWN_ZERO_INDEX] > 0.5).all(1)
    if supplied is None:
        return inferred
    if supplied.dtype != torch.bool or supplied.shape != inferred.shape:
        raise AdvantageM2InputError("supportedはbool[B]必須です")
    if supplied.device != availability.device or not bool((~supplied | inferred).all()):
        raise AdvantageM2InputError("supportedがledger availabilityと不整合です")
    return supplied


def _validate_loss_inputs(
    output: AdvantageM2OutputV1, labels: torch.Tensor, weights: torch.Tensor,
    targets: torch.Tensor, mask: torch.Tensor,
) -> None:
    batch = output.logit.shape[0]
    if labels.dtype != torch.float32 or labels.shape != (batch,):
        raise AdvantageM2InputError("labelはfloat32[B]必須です")
    if weights.dtype != torch.float32 or weights.shape != (batch,):
        raise AdvantageM2InputError("weightはfloat32[B]必須です")
    if targets.dtype != torch.float32 or targets.shape != output.auxiliary.shape:
        raise AdvantageM2InputError("補助target shape/dtypeが不正です")
    if mask.dtype != torch.bool or mask.shape != targets.shape:
        raise AdvantageM2InputError("補助mask shape/dtypeが不正です")
    tensors = (output.logit, output.auxiliary, labels, weights, targets, mask)
    if len({value.device for value in tensors}) != 1:
        raise AdvantageM2InputError("loss tensorのdeviceが一致しません")
    if not bool(torch.isfinite(output.logit).all()):
        raise AdvantageM2InputError("logitは有限値必須です")
    if not bool(torch.isfinite(labels).all()) or bool(((labels != 0) & (labels != 1)).any()):
        raise AdvantageM2InputError("labelは0/1必須です")
    if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise AdvantageM2InputError("weightは有限な正値必須です")
    if mask.any() and not bool(torch.isfinite(targets[mask]).all()):
        raise AdvantageM2InputError("補助targetは有限値必須です")
    if mask.any() and bool(((targets[mask] < 0) | (targets[mask] > 1)).any()):
        raise AdvantageM2InputError("補助targetは0..1必須です")


advantage_m2_inputs_for_training = advantage_m2_inputs_from_canonical
advantage_m2_inputs_for_serving = advantage_m2_inputs_from_canonical


__all__ = [
    "AdvantageM2AnchoredAuxiliaryCNNV2", "AdvantageM2AuxiliaryCNNV1",
    "AdvantageM2AuxiliaryTargetsV1",
    "AdvantageM2InputError", "AdvantageM2InputsV1", "AdvantageM2IntegrityError",
    "AdvantageM2LossV1", "AdvantageM2OutputV1", "AdvantageM2PrimaryV1",
    "AdvantageM2TwoStageAuxiliaryLinearV3", "M2_ANCHORED_MODEL_VERSION",
    "M2_AUXILIARY_LOSS_COEFFICIENT",
    "M2_INPUT_SCHEMA_VERSION", "M2_MODEL_VERSION",
    "M2_TWO_STAGE_MODEL_VERSION",
    "advantage_m2_inputs_for_serving", "advantage_m2_inputs_for_training",
    "advantage_m2_inputs_from_canonical", "advantage_m2_inputs_from_materialized_state",
    "advantage_m2_weighted_loss", "auxiliary_targets_from_m2_inputs",
    "auxiliary_target_from_category_board_v1", "tensorize_m2_primary",
    "tensorize_materialized_all_clear_v1",
]
