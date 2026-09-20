"""6 列の M1 ledger 契約へ zero-counterfactual 残差を適用する V3。"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from src.advantage_m0_current_cnn_v1 import (
    AdvantageM0CurrentCNNV2,
    AdvantageM0OutputV1,
    COLOR_INVARIANT_EMBED_DIM,
)
from src.advantage_m1_causal_ledger_v3 import (
    AVAILABILITY_COUNT,
    AVAILABILITY_ORDER,
    LEDGER_EMBED_DIM,
    LEDGER_FIELD_COUNT,
    PAIR_HIDDEN_DIM,
    CausalLedgerEncoderV3,
)
from src.canonical_observation_v3 import AvailabilityState


ZERO_COUNTERFACTUAL_MODEL_VERSION = "advantage-m1-zero-counterfactual/v3"
ZERO_VARIANT = Literal["values", "masks", "values_and_masks", "random_control"]
ZERO_VARIANTS = ("values", "masks", "values_and_masks", "random_control")
_KNOWN_INDEX = AVAILABILITY_ORDER.index(AvailabilityState.KNOWN)
_KNOWN_ZERO_INDEX = AVAILABILITY_ORDER.index(AvailabilityState.KNOWN_ZERO)
_FAULT_INDEX = AVAILABILITY_ORDER.index(AvailabilityState.INTEGRITY_FAULT)


class AdvantageM1ZeroCounterfactualError(ValueError):
    """V3 tensor・variant・推論制御が固定契約に違反した。"""


class AdvantageM1ZeroCounterfactualIntegrityError(RuntimeError):
    """baseline fallback で隠してはならない完全性異常。"""

    disposition = "HOLD/FAULT"


def zero_counterfactual_ledger_v3(
    values: torch.Tensor, availability: torch.Tensor, variant: ZERO_VARIANT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ablation が読む成分だけを明示的な無信号 tensor へ置換する。"""

    _validate_variant(variant)
    known_zero = torch.zeros_like(availability)
    known_zero[..., _KNOWN_ZERO_INDEX] = 1.0
    zero_values = values.clone() if variant == "masks" else torch.zeros_like(values)
    zero_masks = known_zero if variant == "masks" else availability.clone()
    return zero_values, zero_masks


class AdvantageM1ZeroCounterfactualV3(nn.Module):
    """actual-minus-zero の反対称差だけを凍結 M0 へ加える。"""

    model_version = ZERO_COUNTERFACTUAL_MODEL_VERSION

    def __init__(self, m0: AdvantageM0CurrentCNNV2, variant: ZERO_VARIANT) -> None:
        super().__init__()
        _validate_variant(variant)
        self.variant = variant
        self.m0 = m0
        self.m0.requires_grad_(False)
        self.m0.eval()
        mode = "values_and_masks" if variant == "random_control" else variant
        self.ledger_encoder = CausalLedgerEncoderV3(mode)
        width = (COLOR_INVARIANT_EMBED_DIM + LEDGER_EMBED_DIM) * 3
        self.effect_scorer = nn.Sequential(
            nn.Linear(width, PAIR_HIDDEN_DIM), nn.ReLU(),
            nn.Linear(PAIR_HIDDEN_DIM, 1),
        )
        nn.init.zeros_(self.effect_scorer[-1].weight)
        nn.init.zeros_(self.effect_scorer[-1].bias)

    def train(self, mode: bool = True) -> AdvantageM1ZeroCounterfactualV3:
        super().train(mode)
        self.m0.eval()
        return self

    def forward(
        self, boards: torch.Tensor, queues: torch.Tensor,
        ledger_values: torch.Tensor, ledger_availability: torch.Tensor,
        supported: torch.Tensor | None = None,
    ) -> AdvantageM0OutputV1:
        """supported 行だけ residual を評価し、他行は M0 を bit 複写する。"""

        return self._evaluate_supported(
            boards, queues, ledger_values, ledger_availability, supported,
        )

    def infer(
        self, boards: torch.Tensor, queues: torch.Tensor,
        ledger_values: torch.Tensor, ledger_availability: torch.Tensor,
        supported: torch.Tensor, integrity_valid: torch.Tensor,
    ) -> AdvantageM0OutputV1:
        """本番 hard control を検査してから共通評価経路を呼ぶ。"""

        _validate_control(supported, integrity_valid, boards.shape[0], boards.device)
        if not bool(integrity_valid.all().item()):
            raise AdvantageM1ZeroCounterfactualIntegrityError(
                "integrity fault は baseline fallback せず HOLD/FAULT です"
            )
        return self._evaluate_supported(
            boards, queues, ledger_values, ledger_availability, supported,
        )

    def _evaluate_supported(
        self, boards: torch.Tensor, queues: torch.Tensor,
        ledger_values: torch.Tensor, ledger_availability: torch.Tensor,
        supported: torch.Tensor | None,
    ) -> AdvantageM0OutputV1:
        _validate_ledger_outer(ledger_values, ledger_availability, boards)
        if _declares_integrity_fault(ledger_availability):
            raise AdvantageM1ZeroCounterfactualIntegrityError(
                "integrity fault は baseline fallback せず HOLD/FAULT です"
            )
        active = _resolve_supported(ledger_availability, supported)
        if active.device != boards.device:
            raise AdvantageM1ZeroCounterfactualError("supported の device が一致しません")
        base, board_side = self._frozen_board(boards, queues)
        residual = self._supported_residual(
            board_side, ledger_values, ledger_availability, active,
        )
        return _gated_output(base, residual, active)

    def _supported_residual(
        self, board_side: torch.Tensor, values: torch.Tensor,
        availability: torch.Tensor, active: torch.Tensor,
    ) -> torch.Tensor:
        indices = active.nonzero(as_tuple=False).flatten()
        residual = board_side.new_zeros((board_side.shape[0],))
        if not indices.numel():
            return residual
        selected_values = values.index_select(0, indices)
        selected_masks = availability.index_select(0, indices)
        _validate_ledger(selected_values, selected_masks, len(indices))
        if _has_integrity_fault(selected_masks):
            raise AdvantageM1ZeroCounterfactualIntegrityError("selected ledger が FAULT です")
        if not bool(_ledger_rows_supported(selected_masks).all().item()):
            raise AdvantageM1ZeroCounterfactualError("unsupported ledger を有効化できません")
        selected = self._counterfactual_residual(
            board_side.index_select(0, indices), selected_values, selected_masks,
        )
        return residual.index_copy(0, indices, selected)

    def _frozen_board(
        self, boards: torch.Tensor, queues: torch.Tensor,
    ) -> tuple[AdvantageM0OutputV1, torch.Tensor]:
        with torch.no_grad():
            base = self.m0(boards, queues)
            batch_size = boards.shape[0]
            encoded = self.m0.side_encoder(
                boards.flatten(0, 1), queues.flatten(0, 1),
            ).reshape(batch_size, 2, COLOR_INVARIANT_EMBED_DIM)
        return base, encoded.detach()

    def _counterfactual_residual(
        self, board_side: torch.Tensor, values: torch.Tensor,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        zero_values, zero_masks = zero_counterfactual_ledger_v3(
            values, availability, self.variant,
        )
        actual = self._encode_ledger(values, availability)
        zero = self._encode_ledger(zero_values, zero_masks)
        q_original = self._score(board_side, actual) - self._score(board_side, zero)
        swapped_board, swapped_actual = board_side.flip(1), actual.flip(1)
        q_swapped = self._score(swapped_board, swapped_actual) - self._score(
            swapped_board, zero.flip(1),
        )
        return 0.5 * (q_original - q_swapped)

    def _encode_ledger(
        self, values: torch.Tensor, availability: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = values.shape[0]
        return self.ledger_encoder(
            values.flatten(0, 1), availability.flatten(0, 1),
        ).reshape(batch_size, 2, LEDGER_EMBED_DIM)

    def _score(self, board_side: torch.Tensor, ledger_side: torch.Tensor) -> torch.Tensor:
        sides = torch.cat((board_side, ledger_side), dim=2)
        features = torch.cat((sides[:, 0], sides[:, 1], sides[:, 0] - sides[:, 1]), dim=1)
        return self.effect_scorer(features).squeeze(1)


def _gated_output(
    base: AdvantageM0OutputV1, residual: torch.Tensor, supported: torch.Tensor,
) -> AdvantageM0OutputV1:
    candidate_logit = base.logit + residual
    logit = torch.where(supported, candidate_logit, base.logit)
    probability = torch.where(supported, torch.sigmoid(candidate_logit), base.raw_probability)
    return AdvantageM0OutputV1(logit, probability)


def _resolve_supported(
    availability: torch.Tensor, supported: torch.Tensor | None,
) -> torch.Tensor:
    if supported is None:
        _validate_availability_details(availability)
        return _ledger_rows_supported(availability)
    _validate_bool_vector(supported, availability.shape[0], "supported")
    return supported


def _ledger_rows_supported(availability: torch.Tensor) -> torch.Tensor:
    pending = availability[:, :, 0]
    available = pending[:, :, _KNOWN_INDEX] + pending[:, :, _KNOWN_ZERO_INDEX]
    return (available > 0.5).all(dim=1)


def _has_integrity_fault(availability: torch.Tensor) -> bool:
    return bool((availability[..., _FAULT_INDEX] > 0.5).any().item())


def _declares_integrity_fault(availability: torch.Tensor) -> bool:
    if availability.ndim != 4 or availability.shape[-1] <= _FAULT_INDEX:
        return False
    fault = availability[..., _FAULT_INDEX]
    return bool((torch.isfinite(fault) & (fault > 0.5)).any().item())


def _validate_variant(variant: str) -> None:
    if variant not in ZERO_VARIANTS:
        raise AdvantageM1ZeroCounterfactualError(f"未対応 variant です: {variant}")


def _validate_ledger(
    values: torch.Tensor, availability: torch.Tensor, batch_size: int,
) -> None:
    expected_values = (batch_size, 2, LEDGER_FIELD_COUNT)
    expected_masks = (batch_size, 2, LEDGER_FIELD_COUNT, AVAILABILITY_COUNT)
    if values.dtype != torch.float32 or tuple(values.shape) != expected_values:
        raise AdvantageM1ZeroCounterfactualError("ledger values shape/dtype が不正です")
    if availability.dtype != torch.float32 or tuple(availability.shape) != expected_masks:
        raise AdvantageM1ZeroCounterfactualError("availability shape/dtype が不正です")
    if not bool(torch.isfinite(values).all()):
        raise AdvantageM1ZeroCounterfactualError("ledger tensor は有限値必須です")
    if values.numel() and (values.min().item() < 0.0 or values.max().item() > 1.0):
        raise AdvantageM1ZeroCounterfactualError("ledger values は 0..1 必須です")
    _validate_availability_details(availability)


def _validate_availability_details(availability: torch.Tensor) -> None:
    if not bool(torch.isfinite(availability).all()):
        raise AdvantageM1ZeroCounterfactualError("availability は有限値必須です")
    if not bool(((availability == 0.0) | (availability == 1.0)).all()):
        raise AdvantageM1ZeroCounterfactualError("availability は 0/1 の厳密 one-hot 必須です")
    sums = availability.sum(dim=-1)
    if not bool(torch.allclose(sums, torch.ones_like(sums), atol=0.0, rtol=0.0)):
        raise AdvantageM1ZeroCounterfactualError("availability は厳密 one-hot 必須です")


def _validate_control(
    supported: torch.Tensor, integrity_valid: torch.Tensor,
    batch_size: int, device: torch.device,
) -> None:
    _validate_bool_vector(supported, batch_size, "supported")
    _validate_bool_vector(integrity_valid, batch_size, "integrity_valid")
    if supported.device != integrity_valid.device or supported.device != device:
        raise AdvantageM1ZeroCounterfactualError("推論 control の device が一致しません")


def _validate_bool_vector(value: torch.Tensor, batch_size: int, label: str) -> None:
    if value.dtype != torch.bool or tuple(value.shape) != (batch_size,):
        raise AdvantageM1ZeroCounterfactualError(f"{label} は bool[B] 必須です")


def _validate_ledger_outer(
    values: torch.Tensor, availability: torch.Tensor, boards: torch.Tensor,
) -> None:
    batch_size = boards.shape[0]
    if values.dtype != torch.float32:
        raise AdvantageM1ZeroCounterfactualError("ledger values dtype が不正です")
    if availability.dtype != torch.float32:
        raise AdvantageM1ZeroCounterfactualError("availability dtype が不正です")
    if tuple(values.shape) != (batch_size, 2, LEDGER_FIELD_COUNT):
        raise AdvantageM1ZeroCounterfactualError("ledger values shape が不正です")
    expected = (batch_size, 2, LEDGER_FIELD_COUNT, AVAILABILITY_COUNT)
    if tuple(availability.shape) != expected:
        raise AdvantageM1ZeroCounterfactualError("ledger availability shape が不正です")
    if values.device != availability.device or values.device != boards.device:
        raise AdvantageM1ZeroCounterfactualError("board/ledger の device が一致しません")


__all__ = [
    "AdvantageM1ZeroCounterfactualError",
    "AdvantageM1ZeroCounterfactualIntegrityError",
    "AdvantageM1ZeroCounterfactualV3",
    "ZERO_COUNTERFACTUAL_MODEL_VERSION",
    "ZERO_VARIANTS",
    "zero_counterfactual_ledger_v3",
]
