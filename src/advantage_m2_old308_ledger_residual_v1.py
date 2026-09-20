"""診断用 old308 tree logit へ因果 ledger 残差だけを加える M2。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from src.advantage_m1_causal_ledger_v3 import (
    AVAILABILITY_COUNT,
    AVAILABILITY_ORDER,
    LEDGER_EMBED_DIM,
    LEDGER_FIELD_COUNT,
    PAIR_HIDDEN_DIM,
    CausalLedgerEncoderV3,
)
from src.advantage_m1_zero_counterfactual_v3 import zero_counterfactual_ledger_v3
from src.canonical_observation_v3 import AvailabilityState


MODEL_VERSION = "advantage-m2-old308-ledger-residual/v1"
LedgerVariant = Literal["values", "masks", "values_and_masks", "random_control"]
LEDGER_VARIANTS = ("values", "masks", "values_and_masks", "random_control")
_FAULT_INDEX = AVAILABILITY_ORDER.index(AvailabilityState.INTEGRITY_FAULT)
_KNOWN_INDEX = AVAILABILITY_ORDER.index(AvailabilityState.KNOWN)
_KNOWN_ZERO_INDEX = AVAILABILITY_ORDER.index(AvailabilityState.KNOWN_ZERO)


class Old308LedgerResidualError(ValueError):
    """診断モデルの入力契約違反。"""


class Old308LedgerResidualIntegrityError(RuntimeError):
    """fallback で隠してはならない完全性異常。"""

    disposition = "HOLD/FAULT"


@dataclass(frozen=True, slots=True)
class Old308LedgerResidualOutputV1:
    """anchor と残差を分離して監査できるモデル出力。"""

    logit: torch.Tensor
    raw_probability: torch.Tensor
    residual: torch.Tensor


class AdvantageM2Old308LedgerResidualV1(nn.Module):
    """凍結 old308 tree logit に反対称 actual-minus-zero 残差を足す。"""

    model_version = MODEL_VERSION

    def __init__(self, variant: LedgerVariant = "values_and_masks") -> None:
        super().__init__()
        _validate_variant(variant)
        self.variant = variant
        mode = "values_and_masks" if variant == "random_control" else variant
        self.ledger_encoder = CausalLedgerEncoderV3(mode)
        self.effect_scorer = nn.Sequential(
            nn.Linear(LEDGER_EMBED_DIM * 3, PAIR_HIDDEN_DIM), nn.ReLU(),
            nn.Linear(PAIR_HIDDEN_DIM, 1),
        )
        nn.init.zeros_(self.effect_scorer[-1].weight)
        nn.init.zeros_(self.effect_scorer[-1].bias)

    def forward(
        self, anchor_logit: torch.Tensor, ledger_values: torch.Tensor,
        ledger_availability: torch.Tensor, supported: torch.Tensor | None = None,
    ) -> Old308LedgerResidualOutputV1:
        """unsupported は anchor を厳密複写し、FAULT は停止する。"""

        _validate_inputs(anchor_logit, ledger_values, ledger_availability)
        if _has_integrity_fault(ledger_availability):
            raise Old308LedgerResidualIntegrityError("integrity fault は HOLD/FAULT です")
        active = _resolve_supported(ledger_availability, supported)
        residual = self._supported_residual(ledger_values, ledger_availability, active)
        logit = torch.where(active, anchor_logit + residual, anchor_logit)
        return Old308LedgerResidualOutputV1(logit, torch.sigmoid(logit), residual)

    def _supported_residual(
        self, values: torch.Tensor, availability: torch.Tensor, active: torch.Tensor,
    ) -> torch.Tensor:
        indices = active.nonzero(as_tuple=False).flatten()
        residual = values.new_zeros((values.shape[0],))
        if not indices.numel():
            return residual
        selected_values = values.index_select(0, indices)
        selected_masks = availability.index_select(0, indices)
        if not bool(_ledger_rows_supported(selected_masks).all().item()):
            raise Old308LedgerResidualError("unsupported ledger を有効化できません")
        selected = self._counterfactual_residual(selected_values, selected_masks)
        return residual.index_copy(0, indices, selected)

    def _counterfactual_residual(
        self, values: torch.Tensor, availability: torch.Tensor,
    ) -> torch.Tensor:
        zero_values, zero_masks = zero_counterfactual_ledger_v3(
            values, availability, self.variant,
        )
        actual = self._encode(values, availability)
        zero = self._encode(zero_values, zero_masks)
        direct = self._pair_score(actual) - self._pair_score(zero)
        swapped = self._pair_score(actual.flip(1)) - self._pair_score(zero.flip(1))
        return 0.5 * (direct - swapped)

    def _encode(self, values: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        count = values.shape[0]
        return self.ledger_encoder(
            values.flatten(0, 1), availability.flatten(0, 1),
        ).reshape(count, 2, LEDGER_EMBED_DIM)

    def _pair_score(self, sides: torch.Tensor) -> torch.Tensor:
        features = torch.cat((sides[:, 0], sides[:, 1], sides[:, 0] - sides[:, 1]), dim=1)
        return self.effect_scorer(features).squeeze(1)


def _resolve_supported(
    availability: torch.Tensor, supported: torch.Tensor | None,
) -> torch.Tensor:
    inferred = _ledger_rows_supported(availability)
    if supported is None:
        return inferred
    if supported.dtype != torch.bool or tuple(supported.shape) != (availability.shape[0],):
        raise Old308LedgerResidualError("supported は bool[B] 必須です")
    if supported.device != availability.device:
        raise Old308LedgerResidualError("supported の device が一致しません")
    return supported


def _ledger_rows_supported(availability: torch.Tensor) -> torch.Tensor:
    pending = availability[:, :, 0]
    known = pending[:, :, _KNOWN_INDEX] + pending[:, :, _KNOWN_ZERO_INDEX]
    return (known > 0.5).all(dim=1)


def _has_integrity_fault(availability: torch.Tensor) -> bool:
    return bool((availability[..., _FAULT_INDEX] > 0.5).any().item())


def _validate_variant(variant: str) -> None:
    if variant not in LEDGER_VARIANTS:
        raise Old308LedgerResidualError(f"未対応 variant です: {variant}")


def _validate_inputs(
    anchor_logit: torch.Tensor, values: torch.Tensor, availability: torch.Tensor,
) -> None:
    count = anchor_logit.shape[0]
    if anchor_logit.dtype != torch.float32 or tuple(anchor_logit.shape) != (count,):
        raise Old308LedgerResidualError("anchor_logit は float32[B] 必須です")
    if values.dtype != torch.float32 or tuple(values.shape) != (count, 2, LEDGER_FIELD_COUNT):
        raise Old308LedgerResidualError("ledger_values の shape/dtype が不正です")
    expected = (count, 2, LEDGER_FIELD_COUNT, AVAILABILITY_COUNT)
    if availability.dtype != torch.float32 or tuple(availability.shape) != expected:
        raise Old308LedgerResidualError("ledger_availability の shape/dtype が不正です")
    if anchor_logit.device != values.device or values.device != availability.device:
        raise Old308LedgerResidualError("入力 device が一致しません")
    _validate_numeric(anchor_logit, values, availability)


def _validate_numeric(
    anchor_logit: torch.Tensor, values: torch.Tensor, availability: torch.Tensor,
) -> None:
    if not bool(torch.isfinite(anchor_logit).all()) or not bool(torch.isfinite(values).all()):
        raise Old308LedgerResidualError("anchor/ledger は有限値必須です")
    if values.numel() and (values.min().item() < 0.0 or values.max().item() > 1.0):
        raise Old308LedgerResidualError("ledger_values は 0..1 必須です")
    if not bool(torch.isfinite(availability).all()):
        raise Old308LedgerResidualError("availability は有限値必須です")
    if not bool(((availability == 0.0) | (availability == 1.0)).all()):
        raise Old308LedgerResidualError("availability は 0/1 必須です")
    sums = availability.sum(dim=-1)
    if not bool(torch.allclose(sums, torch.ones_like(sums), atol=0.0, rtol=0.0)):
        raise Old308LedgerResidualError("availability は厳密 one-hot 必須です")
