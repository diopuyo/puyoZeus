"""M1の確信をprojected保証局面だけで安全側へ縮めるCNN。"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.projected_set_residual_cnn_v1 import (
    ProjectedSetForwardOutputV1,
    ProjectedSetResidualCNNV1,
)


MAX_PROBABILITY_SHRINK: float = 0.5


class ProjectedSetBoundedShrinkError(ValueError):
    """bounded shrink入力が固定契約を満たさない。"""


class ProjectedSetBoundedShrinkCNNV2(ProjectedSetResidualCNNV1):
    """反対称projected deltaを非拡大の縮約率へ変換する。"""

    def __init__(self) -> None:
        super().__init__()
        final = self.scorer[-1]
        if not isinstance(final, nn.Linear):
            raise ProjectedSetBoundedShrinkError("scorer最終層がLinearではありません")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        current: torch.Tensor,
        post_chain: torch.Tensor,
        landing: torch.Tensor,
        branch_mask: torch.Tensor,
        scalar: torch.Tensor,
        quantity_present_mask: torch.Tensor,
        baseline_raw: torch.Tensor,
    ) -> ProjectedSetForwardOutputV1:
        """projected提案をbaselineと0.5の間へ固定して返す。"""

        unconstrained = super().forward(
            current, post_chain, landing, branch_mask, scalar,
            quantity_present_mask, baseline_raw,
        )
        probability, branch_probability = bounded_projected_probability_v2(
            baseline_raw, unconstrained.delta, branch_mask,
        )
        return ProjectedSetForwardOutputV1(
            raw_probability=probability,
            branch_probability=branch_probability,
            delta=unconstrained.delta,
            branch_mask=unconstrained.branch_mask,
            auxiliary_current=unconstrained.auxiliary_current,
            auxiliary_post_chain=unconstrained.auxiliary_post_chain,
            auxiliary_landing=unconstrained.auxiliary_landing,
        )


def bounded_projected_probability_v2(
    baseline: torch.Tensor,
    delta: torch.Tensor,
    branch_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """baseline方向と反対のdeltaだけを最大50%の確率縮約へ写像する。"""

    _validate_bounded_inputs(baseline, delta, branch_mask)
    direction = torch.sign(baseline - 0.5).unsqueeze(1)
    toward_even = -direction * delta.to(torch.float64)
    fraction = MAX_PROBABILITY_SHRINK * torch.clamp(toward_even, 0.0, 1.0)
    active = branch_mask.to(torch.float64)
    mean_fraction = (fraction * active).sum(1) / active.sum(1)
    probability = baseline + mean_fraction * (0.5 - baseline)
    expanded = baseline.unsqueeze(1) + fraction * (0.5 - baseline).unsqueeze(1)
    branch_probability = torch.where(branch_mask, expanded, torch.zeros_like(expanded))
    return probability, branch_probability


def _validate_bounded_inputs(
    baseline: torch.Tensor,
    delta: torch.Tensor,
    branch_mask: torch.Tensor,
) -> None:
    batch_size = baseline.shape[0] if baseline.ndim == 1 else -1
    valid = (
        baseline.dtype == torch.float64
        and delta.dtype == torch.float32
        and branch_mask.dtype == torch.bool
        and delta.ndim == 2
        and branch_mask.shape == delta.shape
        and delta.shape[0] == batch_size
        and baseline.device == delta.device == branch_mask.device
    )
    if not valid:
        raise ProjectedSetBoundedShrinkError("bounded shrinkのshape/dtype/deviceが不正です")
    finite = torch.isfinite(baseline).all() and torch.isfinite(delta).all()
    in_range = ((baseline >= 0.0) & (baseline <= 1.0)).all()
    branch_count = branch_mask.sum(1)
    if not bool(finite.item()) or not bool(in_range.item()):
        raise ProjectedSetBoundedShrinkError("baselineまたはdeltaが非有限・範囲外です")
    if not bool((branch_count > 0).all().item()):
        raise ProjectedSetBoundedShrinkError("有効landing branchがありません")


__all__ = [
    "MAX_PROBABILITY_SHRINK",
    "ProjectedSetBoundedShrinkCNNV2",
    "ProjectedSetBoundedShrinkError",
    "bounded_projected_probability_v2",
]

