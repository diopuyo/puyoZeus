"""M1確信度を拡大せず、選択的に縮約するM2補助学習モデル。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import advantage_m2_auxiliary_cnn_v1 as base
from src.advantage_m1_zero_counterfactual_v3 import AdvantageM1ZeroCounterfactualV3


MODEL_VERSION = "advantage-m2-bounded-auxiliary-shrink/v4"
MAX_SHRINK_FRACTION = 0.5
MATERIAL_SPATIAL_AUXILIARY_NAMES = tuple(base.AUXILIARY_FEATURE_NAMES)
FIREPOWER_AUXILIARY_NAMES = (
    "current_max_chain", "ignition_point_count", "immediate_fire_power",
    "min_puyos_to_ignite", "saturation_chain_upper", "second_chain_potential",
)
CONNECTIVITY_AUXILIARY_NAMES = (
    "chain_articulation_point_count", "chain_efficiency",
    "color_diversity_evenness", "column_bumpiness", "isolated_pair_count",
    "main_linked_pair_count", "main_linked_ratio", "multi_color_ignition",
    "simultaneous_pop_richness", "sub_chain_count",
)
_NEW_CONNECTIVITY_NAMES = tuple(
    name for name in CONNECTIVITY_AUXILIARY_NAMES
    if name not in MATERIAL_SPATIAL_AUXILIARY_NAMES
)
AUXILIARY_FEATURE_NAMES_V4 = (
    MATERIAL_SPATIAL_AUXILIARY_NAMES
    + FIREPOWER_AUXILIARY_NAMES
    + _NEW_CONNECTIVITY_NAMES
)
AUXILIARY_FAMILY_NAMES = {
    "material_spatial8": MATERIAL_SPATIAL_AUXILIARY_NAMES,
    "firepower_and_ignition": FIREPOWER_AUXILIARY_NAMES,
    "connectivity_and_shape": CONNECTIVITY_AUXILIARY_NAMES,
    "firepower_and_connectivity": tuple(dict.fromkeys(
        FIREPOWER_AUXILIARY_NAMES + CONNECTIVITY_AUXILIARY_NAMES
    )),
}
SYMMETRIC_HIDDEN_DIM = 64


@dataclass(frozen=True, slots=True)
class AdvantageM2BoundedOutputV4:
    """M1基準、縮約率、補助出力を分離した監査可能出力。"""

    logit: torch.Tensor
    raw_probability: torch.Tensor
    auxiliary: torch.Tensor
    shrink_fraction: torch.Tensor
    anchor_logit: torch.Tensor


class AdvantageM2BoundedAuxiliaryV4(nn.Module):
    """補助事前学習表現でM1確信度だけを安全に縮約する。"""

    model_version = MODEL_VERSION

    def __init__(self, anchor: AdvantageM1ZeroCounterfactualV3) -> None:
        super().__init__()
        if not isinstance(anchor, AdvantageM1ZeroCounterfactualV3):
            raise base.AdvantageM2InputError("anchorはM1 zero-counterfactual V3必須です")
        self.anchor = anchor
        self.anchor.requires_grad_(False)
        self.anchor.eval()
        self.adapted_side_encoder = _copy_side_encoder(anchor)
        self.all_clear_encoder = nn.Sequential(
            nn.Linear(1 + base.AVAILABILITY_COUNT, base.ALL_CLEAR_EMBED_DIM), nn.ReLU(),
        )
        side_width = base.BASE_SIDE_DIM
        self.shrink_scorer = nn.Sequential(
            nn.Linear(side_width * 2 + 1, SYMMETRIC_HIDDEN_DIM), nn.ReLU(),
            nn.Linear(SYMMETRIC_HIDDEN_DIM, 1),
        )
        nn.init.zeros_(self.shrink_scorer[-1].weight)
        nn.init.zeros_(self.shrink_scorer[-1].bias)
        self.auxiliary_head = nn.Sequential(
            nn.Linear(base.COLOR_INVARIANT_EMBED_DIM, len(AUXILIARY_FEATURE_NAMES_V4)),
            nn.Sigmoid(),
        )

    def train(self, mode: bool = True) -> "AdvantageM2BoundedAuxiliaryV4":
        super().train(mode)
        self.anchor.eval()
        return self

    def predict_auxiliary(
        self, boards: torch.Tensor, queues: torch.Tensor,
    ) -> torch.Tensor:
        """勝率labelを使わない補助事前学習出力。"""

        base._validate_board_queue_inputs(boards, queues)
        return self.auxiliary_head(self._encode_board(boards, queues))

    def freeze_auxiliary_encoder(self) -> None:
        """補助表現を勝率labelから隔離する。"""

        self.adapted_side_encoder.requires_grad_(False)
        self.auxiliary_head.requires_grad_(False)

    def forward(
        self, boards: torch.Tensor, queues: torch.Tensor,
        all_clear_values: torch.Tensor, all_clear_availability: torch.Tensor,
        ledger_values: torch.Tensor, ledger_availability: torch.Tensor,
        supported: torch.Tensor | None = None,
    ) -> AdvantageM2BoundedOutputV4:
        """ledger未対応行はM1を厳密複写し、対応行も確信度を拡大しない。"""

        base._validate_model_inputs(
            boards, queues, all_clear_values, all_clear_availability,
            ledger_values, ledger_availability,
        )
        base._raise_on_integrity_fault(all_clear_availability, ledger_availability)
        active = base._supported_rows(ledger_availability, supported)
        anchor = self.anchor(
            boards, queues, ledger_values, ledger_availability, supported,
        )
        sides = self._side_features(
            boards, queues, all_clear_values, all_clear_availability,
        )
        shrink = self._bounded_shrink(sides, anchor.logit, active)
        logit = torch.where(active, anchor.logit * (1.0 - shrink), anchor.logit)
        probability = torch.where(active, torch.sigmoid(logit), anchor.raw_probability)
        return AdvantageM2BoundedOutputV4(
            logit, probability, self.auxiliary_head(sides[..., :base.COLOR_INVARIANT_EMBED_DIM]),
            shrink, anchor.logit,
        )

    def _encode_board(self, boards: torch.Tensor, queues: torch.Tensor) -> torch.Tensor:
        count = boards.shape[0]
        return self.adapted_side_encoder(
            boards.flatten(0, 1), queues.flatten(0, 1),
        ).reshape(count, 2, base.COLOR_INVARIANT_EMBED_DIM)

    def _side_features(
        self, boards: torch.Tensor, queues: torch.Tensor,
        values: torch.Tensor, availability: torch.Tensor,
    ) -> torch.Tensor:
        board = self._encode_board(boards, queues)
        clear = self.all_clear_encoder(torch.cat((values.unsqueeze(-1), availability), -1))
        return torch.cat((board, clear), dim=-1)

    def _bounded_shrink(
        self, sides: torch.Tensor, anchor_logit: torch.Tensor, active: torch.Tensor,
    ) -> torch.Tensor:
        symmetric = torch.cat((
            sides[:, 0] + sides[:, 1],
            torch.abs(sides[:, 0] - sides[:, 1]),
            torch.abs(anchor_logit).unsqueeze(1),
        ), dim=1)
        raw = self.shrink_scorer(symmetric).squeeze(1)
        zero = F.softplus(torch.zeros_like(raw))
        shrink = torch.clamp(F.softplus(raw) - zero, 0.0, MAX_SHRINK_FRACTION)
        return torch.where(active, shrink, torch.zeros_like(shrink))


def _copy_side_encoder(anchor: AdvantageM1ZeroCounterfactualV3) -> nn.Module:
    import copy

    encoder = copy.deepcopy(anchor.m0.side_encoder)
    encoder.requires_grad_(True)
    return encoder


def family_indices(name: str) -> tuple[int, ...]:
    """family名を固定22列上の位置へ変換する。"""

    if name not in AUXILIARY_FAMILY_NAMES:
        raise base.AdvantageM2InputError(f"未対応auxiliary familyです: {name}")
    selected = set(AUXILIARY_FAMILY_NAMES[name])
    return tuple(
        index for index, feature in enumerate(AUXILIARY_FEATURE_NAMES_V4)
        if feature in selected
    )


__all__ = [
    "AUXILIARY_FAMILY_NAMES", "AUXILIARY_FEATURE_NAMES_V4",
    "AdvantageM2BoundedAuxiliaryV4", "AdvantageM2BoundedOutputV4",
    "CONNECTIVITY_AUXILIARY_NAMES", "FIREPOWER_AUXILIARY_NAMES",
    "MATERIAL_SPATIAL_AUXILIARY_NAMES", "MAX_SHRINK_FRACTION",
    "MODEL_VERSION", "family_indices",
]
