"""ProjectedStateTensorV1を順序保持でPyTorch batchへ束ねる。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np
import torch

from src.projected_set_residual_cnn_v1 import ProjectedSetAuxiliaryTargetsV1
from src.projected_state_tensorizer_v1 import (
    AUXILIARY_FEATURE_COUNT,
    AUXILIARY_MANIFEST_SHA256,
    CHANNEL_COUNT,
    SCALAR_COUNT,
    ProjectedStateTensorV1,
    projected_state_tensor_content_digest_v1,
)
from src.projected_state_observation_v1 import MAX_LANDING_BRANCHES
from src.board import BOARD_COLS, BOARD_ROWS


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ProjectedStateBatchError(ValueError):
    """collate対象が凍結tensor契約を満たさない。"""


@dataclass(frozen=True, slots=True)
class ProjectedStateBatchV1:
    """勝敗labelとbaselineを含まないprojected-state入力batch。"""

    current: torch.Tensor
    post_chain: torch.Tensor
    landing: torch.Tensor
    branch_mask: torch.Tensor
    scalar: torch.Tensor
    quantity_present_mask: torch.Tensor
    candidate_count: torch.Tensor
    auxiliary_targets: ProjectedSetAuxiliaryTargetsV1
    observation_input_digests: tuple[str, ...]
    auxiliary_manifest_sha256: str
    tensor_content_digests: tuple[str, ...]


def collate_projected_state_tensors_v1(
    values: Sequence[ProjectedStateTensorV1],
    *,
    device: str | torch.device = "cpu",
) -> ProjectedStateBatchV1:
    """入力順を変えず、NumPy単観測列を凍結dtypeのtorch batchへ変換する。"""

    if not values:
        raise ProjectedStateBatchError("空のprojected-state batchは作れません")
    for value in values:
        _validate_tensor(value)
    target_device = torch.device(device)
    auxiliary = ProjectedSetAuxiliaryTargetsV1(
        current=_stack(values, "auxiliary_current", target_device),
        post_chain=_stack(values, "auxiliary_post_chain", target_device),
        landing=_stack(values, "auxiliary_landing", target_device),
        current_mask=_stack(values, "auxiliary_current_mask", target_device),
        post_chain_mask=_stack(values, "auxiliary_post_chain_mask", target_device),
        landing_mask=_stack(values, "auxiliary_landing_mask", target_device),
    )
    return ProjectedStateBatchV1(
        current=_stack(values, "current", target_device),
        post_chain=_stack(values, "post_chain", target_device),
        landing=_stack(values, "landing", target_device),
        branch_mask=_stack(values, "branch_mask", target_device),
        scalar=_stack(values, "scalar", target_device),
        quantity_present_mask=_stack(values, "quantity_present_mask", target_device),
        candidate_count=_candidate_count(values, target_device),
        auxiliary_targets=auxiliary,
        observation_input_digests=tuple(v.observation_input_digest for v in values),
        auxiliary_manifest_sha256=AUXILIARY_MANIFEST_SHA256,
        tensor_content_digests=tuple(v.tensor_content_digest for v in values),
    )


def _stack(
    values: Sequence[ProjectedStateTensorV1], name: str, device: torch.device,
) -> torch.Tensor:
    array = np.stack([getattr(value, name) for value in values], axis=0)
    return torch.from_numpy(np.ascontiguousarray(array)).to(device=device)


def _candidate_count(
    values: Sequence[ProjectedStateTensorV1], device: torch.device,
) -> torch.Tensor:
    array = np.asarray([value.candidate_count for value in values], dtype=np.int8)
    return torch.from_numpy(array).to(device=device)


def _validate_tensor(value: object) -> None:
    if not isinstance(value, ProjectedStateTensorV1):
        raise ProjectedStateBatchError("batch要素はProjectedStateTensorV1必須です")
    if value.auxiliary_manifest_sha256 != AUXILIARY_MANIFEST_SHA256:
        raise ProjectedStateBatchError("補助indicator manifest SHA-256が一致しません")
    specifications = _array_specifications(value)
    for array, shape, dtype, label in specifications:
        if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype != dtype:
            raise ProjectedStateBatchError(f"{label}のshape/dtypeが不正です")
    count = int(value.candidate_count)
    if type(value.candidate_count) is not np.int8 or not 0 <= count <= MAX_LANDING_BRANCHES:
        raise ProjectedStateBatchError("candidate countが0..20範囲外です")
    expected_mask = np.arange(MAX_LANDING_BRANCHES) < count
    if not np.array_equal(value.branch_mask, expected_mask):
        raise ProjectedStateBatchError("candidate countとbranch maskが一致しません")
    if SHA256_PATTERN.fullmatch(value.observation_input_digest) is None:
        raise ProjectedStateBatchError("observation input digestが不正です")
    if SHA256_PATTERN.fullmatch(value.tensor_content_digest) is None:
        raise ProjectedStateBatchError("tensor content digestが不正です")
    actual_digest = projected_state_tensor_content_digest_v1(value)
    if value.tensor_content_digest != actual_digest:
        raise ProjectedStateBatchError("tensor内容とcontent digestが一致しません")


def _array_specifications(
    value: ProjectedStateTensorV1,
) -> tuple[tuple[np.ndarray, tuple[int, ...], np.dtype[np.generic], str], ...]:
    board = (2, CHANNEL_COUNT, BOARD_ROWS, BOARD_COLS)
    landing = (MAX_LANDING_BRANCHES, *board)
    auxiliary = (2, AUXILIARY_FEATURE_COUNT)
    auxiliary_landing = (MAX_LANDING_BRANCHES, *auxiliary)
    return (
        (value.current, board, np.dtype(np.float32), "current"),
        (value.post_chain, board, np.dtype(np.float32), "post-chain"),
        (value.landing, landing, np.dtype(np.float32), "landing"),
        (value.branch_mask, (MAX_LANDING_BRANCHES,), np.dtype(np.bool_), "branch mask"),
        (value.scalar, (SCALAR_COUNT,), np.dtype(np.float32), "scalar"),
        (value.quantity_present_mask, (SCALAR_COUNT,), np.dtype(np.bool_), "quantity mask"),
        (value.auxiliary_current, auxiliary, np.dtype(np.float32), "aux current"),
        (value.auxiliary_post_chain, auxiliary, np.dtype(np.float32), "aux post"),
        (value.auxiliary_landing, auxiliary_landing, np.dtype(np.float32), "aux landing"),
        (value.auxiliary_current_mask, auxiliary, np.dtype(np.bool_), "aux current mask"),
        (value.auxiliary_post_chain_mask, auxiliary, np.dtype(np.bool_), "aux post mask"),
        (value.auxiliary_landing_mask, auxiliary_landing, np.dtype(np.bool_), "aux landing mask"),
    )


__all__ = [
    "ProjectedStateBatchError", "ProjectedStateBatchV1",
    "collate_projected_state_tensors_v1",
]
