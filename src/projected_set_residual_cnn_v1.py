"""Formal100 projected-set residual CNN の凍結済みモデル本体。

DTO や動画処理には依存せず、tensorizer が作った Tensor だけを受け取る。
既存 BoardCNN / 学習済み重み / 本番設定とは独立した診断用モデルである。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


BOARD_ROWS: int = 13
BOARD_COLS: int = 6
COLOR_CHANNELS: int = 8
MAX_BRANCHES: int = 20
SCALAR_DIM: int = 6
EMBED_DIM: int = 32
PAIR_DIM: int = EMBED_DIM * 3
SCORER_INPUT_DIM: int = PAIR_DIM * 3 + SCALAR_DIM
AUXILIARY_DIM: int = 8
SCORER_HIDDEN_DIM: int = 64
EPS: float = 1e-6
AUXILIARY_LOSS_COEFFICIENT: float = 0.3
SCALAR_SWAP_ORDER: tuple[int, ...] = (1, 0, 3, 2, 5, 4)


class ProjectedSetModelInputError(ValueError):
    """モデル入力またはbaselineが凍結契約を満たさない。"""


class ProjectedSetModelIntegrityError(RuntimeError):
    """因果・資産integrityが不正で、fallbackしてはいけない。"""


class ProjectedSetModelOutputError(RuntimeError):
    """モデル出力が非有限などの安全契約違反になった。"""


@dataclass(frozen=True, slots=True)
class ProjectedSetForwardOutputV1:
    """projected-supported観測に対する学習用出力。"""

    raw_probability: torch.Tensor
    branch_probability: torch.Tensor
    delta: torch.Tensor
    branch_mask: torch.Tensor
    auxiliary_current: torch.Tensor
    auxiliary_post_chain: torch.Tensor
    auxiliary_landing: torch.Tensor


@dataclass(frozen=True, slots=True)
class ProjectedSetInferenceOutputV1:
    """fallbackを含む推論用raw/calibrated出力。"""

    raw_probability: torch.Tensor
    calibrated_probability: torch.Tensor
    projected_model_used: torch.Tensor


@dataclass(frozen=True, slots=True)
class ProjectedSetAuxiliaryTargetsV1:
    """8補助指標のtargetとpresent mask。"""

    current: torch.Tensor
    post_chain: torch.Tensor
    landing: torch.Tensor
    current_mask: torch.Tensor
    post_chain_mask: torch.Tensor
    landing_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class ProjectedSetLossV1:
    """ゲーム均等重みを適用した損失内訳。"""

    total: torch.Tensor
    binary_cross_entropy: torch.Tensor
    auxiliary_mse: torch.Tensor


@dataclass(frozen=True, slots=True)
class ProjectedSetUnreducedLossV1:
    """logical batch全体で一度だけ正規化するための観測別損失。"""

    binary_cross_entropy: torch.Tensor
    auxiliary_mse: torch.Tensor
    auxiliary_present: torch.Tensor


class PositionPreservingBoardEncoderV1(nn.Module):
    """global poolingを使わず13行6列の位置を保持する共有encoder。"""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(COLOR_CHANNELS, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
        )
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(32 * BOARD_ROWS * BOARD_COLS, EMBED_DIM),
            nn.ReLU(),
        )

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        """float32[N,8,13,6]をfloat32[N,32]へ変換する。"""
        return self.projection(self.features(board))


class ProjectedSetResidualCNNV1(nn.Module):
    """等確率landing集合をbaseline残差として評価するCNN。"""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = PositionPreservingBoardEncoderV1()
        self.scorer = nn.Sequential(
            nn.Linear(SCORER_INPUT_DIM, SCORER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(SCORER_HIDDEN_DIM, 1),
        )
        self.auxiliary_head = nn.Sequential(
            nn.Linear(EMBED_DIM, AUXILIARY_DIM), nn.Sigmoid(),
        )

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
        """全行projected-supportedのbatchを学習用に評価する。"""
        _validate_layout(
            current, post_chain, landing, branch_mask, scalar,
            quantity_present_mask, baseline_raw,
        )
        _validate_supported_values(
            current, post_chain, landing, branch_mask, scalar,
            quantity_present_mask, baseline_raw,
        )
        current_emb = self._encode_pair_boards(current)
        post_emb = self._encode_pair_boards(post_chain)
        landing_emb = self._encode_landing(landing, branch_mask)
        delta = self._antisymmetric_delta(current_emb, post_emb, landing_emb, scalar)
        branch_probability = _branch_probabilities(baseline_raw, delta, branch_mask)
        raw_probability = _equal_branch_mean(branch_probability, branch_mask)
        output = self._make_output(
            raw_probability, branch_probability, delta, branch_mask,
            current_emb, post_emb, landing_emb,
        )
        _validate_model_output(output)
        return output

    def infer(
        self,
        current: torch.Tensor,
        post_chain: torch.Tensor,
        landing: torch.Tensor,
        branch_mask: torch.Tensor,
        scalar: torch.Tensor,
        quantity_present_mask: torch.Tensor,
        baseline_raw: torch.Tensor,
        baseline_calibrated: torch.Tensor,
        projected_supported: torch.Tensor,
        integrity_valid: torch.Tensor,
        calibration_slope: torch.Tensor | None = None,
    ) -> ProjectedSetInferenceOutputV1:
        """supportedだけを評価し、projected固有unsupportedはbaselineを複写する。"""
        _validate_layout(
            current, post_chain, landing, branch_mask, scalar,
            quantity_present_mask, baseline_raw,
        )
        _validate_inference_controls(
            baseline_raw, baseline_calibrated, projected_supported, integrity_valid,
        )
        if not bool(integrity_valid.all().item()):
            raise ProjectedSetModelIntegrityError("integrity不一致はfallbackできません")
        indices = projected_supported.nonzero(as_tuple=False).flatten()
        raw = baseline_raw.clone()
        calibrated = baseline_calibrated.clone()
        if indices.numel() > 0:
            supported = self._forward_selected(
                indices, current, post_chain, landing, branch_mask, scalar,
                quantity_present_mask, baseline_raw,
            )
            raw = raw.index_copy(0, indices, supported.raw_probability)
            fitted = symmetric_platt_probability(
                supported.raw_probability, calibration_slope,
            )
            calibrated = calibrated.index_copy(0, indices, fitted)
        return ProjectedSetInferenceOutputV1(raw, calibrated, projected_supported.clone())

    def _forward_selected(
        self,
        indices: torch.Tensor,
        current: torch.Tensor,
        post_chain: torch.Tensor,
        landing: torch.Tensor,
        branch_mask: torch.Tensor,
        scalar: torch.Tensor,
        quantity_present_mask: torch.Tensor,
        baseline_raw: torch.Tensor,
    ) -> ProjectedSetForwardOutputV1:
        """推論batchからsupported行だけを抽出し、無効値へ触れず評価する。"""
        selected = [
            value.index_select(0, indices) for value in (
                current, post_chain, landing, branch_mask, scalar,
                quantity_present_mask, baseline_raw,
            )
        ]
        return self(*selected)

    def _encode_pair_boards(self, boards: torch.Tensor) -> torch.Tensor:
        """両side盤面を同じencoderへ通す。"""
        batch_size = boards.shape[0]
        flat = boards.reshape(batch_size * 2, COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS)
        return self.encoder(flat).reshape(batch_size, 2, EMBED_DIM)

    def _encode_landing(
        self, landing: torch.Tensor, branch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """有効branchだけをencodeし、paddingを計算・gradientから外す。"""
        batch_size, branch_count = branch_mask.shape
        flat = landing.reshape(
            batch_size * branch_count * 2, COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS,
        )
        valid = branch_mask.unsqueeze(-1).expand(-1, -1, 2).reshape(-1)
        indices = valid.nonzero(as_tuple=False).flatten()
        selected = self.encoder(flat.index_select(0, indices))
        encoded = flat.new_zeros((flat.shape[0], EMBED_DIM))
        encoded = encoded.index_copy(0, indices, selected)
        return encoded.reshape(batch_size, branch_count, 2, EMBED_DIM)

    def _antisymmetric_delta(
        self,
        current: torch.Tensor,
        post_chain: torch.Tensor,
        landing: torch.Tensor,
        scalar: torch.Tensor,
    ) -> torch.Tensor:
        """zero counterfactualを引き、side-swap反対称残差を作る。"""
        actual = self._score_branches(current, post_chain, landing, scalar)
        zero = self._score_zero(current)
        swapped_scalar = scalar[:, SCALAR_SWAP_ORDER]
        swapped = self._score_branches(
            current.flip(1), post_chain.flip(1), landing.flip(2), swapped_scalar,
        )
        swapped_zero = self._score_zero(current.flip(1))
        residual = actual - zero.unsqueeze(1)
        swapped_residual = swapped - swapped_zero.unsqueeze(1)
        return 0.5 * (residual - swapped_residual)

    def _score_branches(
        self,
        current: torch.Tensor,
        post_chain: torch.Tensor,
        landing: torch.Tensor,
        scalar: torch.Tensor,
    ) -> torch.Tensor:
        """current/post/landing pairと6物理量からbranch別logit残差を返す。"""
        branch_count = landing.shape[1]
        current_pair = _pair_embedding(current).unsqueeze(1).expand(-1, branch_count, -1)
        post_pair = _pair_embedding(post_chain).unsqueeze(1).expand(-1, branch_count, -1)
        landing_pair = _pair_embedding(landing)
        expanded_scalar = scalar.unsqueeze(1).expand(-1, branch_count, -1)
        features = torch.cat(
            (current_pair, post_pair, landing_pair, expanded_scalar), dim=-1,
        )
        return self.scorer(features).squeeze(-1)

    def _score_zero(self, current: torch.Tensor) -> torch.Tensor:
        """物理的新事実なしのz=(current,current,current,0)を採点する。"""
        pair = _pair_embedding(current)
        zero_scalar = pair.new_zeros((pair.shape[0], SCALAR_DIM))
        return self.scorer(torch.cat((pair, pair, pair, zero_scalar), dim=-1)).squeeze(-1)

    def _make_output(
        self,
        raw_probability: torch.Tensor,
        branch_probability: torch.Tensor,
        delta: torch.Tensor,
        branch_mask: torch.Tensor,
        current: torch.Tensor,
        post_chain: torch.Tensor,
        landing: torch.Tensor,
    ) -> ProjectedSetForwardOutputV1:
        """共有embeddingから補助headを計算し、padding出力を0に固定する。"""
        auxiliary_landing = self.auxiliary_head(landing)
        auxiliary_landing = torch.where(
            branch_mask[:, :, None, None], auxiliary_landing,
            torch.zeros_like(auxiliary_landing),
        )
        return ProjectedSetForwardOutputV1(
            raw_probability=raw_probability,
            branch_probability=branch_probability,
            delta=delta,
            branch_mask=branch_mask.clone(),
            auxiliary_current=self.auxiliary_head(current),
            auxiliary_post_chain=self.auxiliary_head(post_chain),
            auxiliary_landing=auxiliary_landing,
        )


def symmetric_platt_probability(
    raw_probability: torch.Tensor,
    slope: torch.Tensor | None = None,
) -> torch.Tensor:
    """切片0・傾き非負の対称Platt較正をfloat64で適用する。"""
    if raw_probability.dtype != torch.float64 or raw_probability.ndim != 1:
        raise ProjectedSetModelInputError("Platt入力はfloat64[B]必須です")
    if not _is_finite_in_unit_interval(raw_probability):
        raise ProjectedSetModelInputError("Platt入力確率が非有限または範囲外です")
    actual_slope = raw_probability.new_tensor(1.0) if slope is None else slope
    if (actual_slope.ndim != 0 or actual_slope.device != raw_probability.device
            or actual_slope.dtype != torch.float64):
        raise ProjectedSetModelInputError("Platt slopeは同deviceのfloat64 scalar必須です")
    if not bool(torch.isfinite(actual_slope).item()) or actual_slope.item() < 0.0:
        raise ProjectedSetModelInputError("Platt slopeは有限な非負値必須です")
    clipped = raw_probability.clamp(EPS, 1.0 - EPS)
    calibrated = torch.sigmoid(actual_slope * torch.logit(clipped))
    if not bool(torch.isfinite(calibrated).all().item()):
        raise ProjectedSetModelOutputError("較正後確率が非有限です")
    return calibrated


def equal_game_observation_weights(game_ids: torch.Tensor) -> torch.Tensor:
    """同一gameの観測重み合計が1になるfloat64重みを返す。

    合成性試験専用。正式学習では母集団全体から事前計算した
    ``precompute_equal_game_weights_v1``だけを使い、batch内で再計算しない。
    """
    if game_ids.ndim != 1 or game_ids.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64,
    ):
        raise ProjectedSetModelInputError("game_idsは整数Tensor[B]必須です")
    _, inverse, counts = torch.unique(
        game_ids, sorted=True, return_inverse=True, return_counts=True,
    )
    return counts.index_select(0, inverse).to(torch.float64).reciprocal()


def projected_set_weighted_loss(
    output: ProjectedSetForwardOutputV1,
    winner_label: torch.Tensor,
    observation_weight: torch.Tensor,
    auxiliary_targets: ProjectedSetAuxiliaryTargetsV1 | None = None,
) -> ProjectedSetLossV1:
    """branch数非依存の観測BCEと係数0.3の補助MSEを返す。"""
    terms = projected_set_unreduced_loss(output, winner_label, auxiliary_targets)
    _validate_observation_weight(output, observation_weight)
    binary_loss = _weighted_mean(
        terms.binary_cross_entropy, observation_weight,
    )
    auxiliary_loss = _weighted_present_mean(
        terms.auxiliary_mse, observation_weight, terms.auxiliary_present,
    )
    total = binary_loss + AUXILIARY_LOSS_COEFFICIENT * auxiliary_loss
    return ProjectedSetLossV1(total, binary_loss, auxiliary_loss)


def projected_set_unreduced_loss(
    output: ProjectedSetForwardOutputV1,
    winner_label: torch.Tensor,
    auxiliary_targets: ProjectedSetAuxiliaryTargetsV1 | None = None,
) -> ProjectedSetUnreducedLossV1:
    """microbatchで割らず、観測別BCE・補助MSE・presentを返す。"""

    _validate_model_output(output)
    _validate_winner_label(output, winner_label)
    clipped = output.raw_probability.clamp(EPS, 1.0 - EPS)
    binary = F.binary_cross_entropy(clipped, winner_label, reduction="none")
    if auxiliary_targets is None:
        auxiliary, present = _zero_auxiliary_terms(output)
    else:
        auxiliary, present = _auxiliary_loss_per_observation(
            output, auxiliary_targets,
        )
    return ProjectedSetUnreducedLossV1(
        binary, auxiliary.to(torch.float64), present,
    )


def _validate_layout(
    current: torch.Tensor,
    post_chain: torch.Tensor,
    landing: torch.Tensor,
    branch_mask: torch.Tensor,
    scalar: torch.Tensor,
    quantity_present_mask: torch.Tensor,
    baseline_raw: torch.Tensor,
) -> None:
    """固定shape/dtype/deviceだけを検査し、unsupported値には触れない。"""
    batch_size = current.shape[0] if current.ndim > 0 else -1
    expected_board = (batch_size, 2, COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS)
    expected_landing = (
        batch_size, MAX_BRANCHES, 2, COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS,
    )
    typed_shapes = (
        (current, torch.float32, expected_board, "current"),
        (post_chain, torch.float32, expected_board, "post-chain"),
        (landing, torch.float32, expected_landing, "landing"),
        (branch_mask, torch.bool, (batch_size, MAX_BRANCHES), "branch mask"),
        (scalar, torch.float32, (batch_size, SCALAR_DIM), "scalar"),
        (quantity_present_mask, torch.bool, (batch_size, SCALAR_DIM), "quantity mask"),
        (baseline_raw, torch.float64, (batch_size,), "baseline raw"),
    )
    if any(value.dtype != dtype or tuple(value.shape) != shape
           for value, dtype, shape, _ in typed_shapes):
        detail = ", ".join(
            f"{name}={tuple(value.shape)}/{value.dtype}"
            for value, _, _, name in typed_shapes
        )
        raise ProjectedSetModelInputError(f"tensor shape/dtype不一致: {detail}")
    if any(value.device != current.device for value, _, _, _ in typed_shapes):
        raise ProjectedSetModelInputError("入力Tensorのdeviceが一致しません")


def _validate_supported_values(
    current: torch.Tensor,
    post_chain: torch.Tensor,
    landing: torch.Tensor,
    branch_mask: torch.Tensor,
    scalar: torch.Tensor,
    quantity_present_mask: torch.Tensor,
    baseline_raw: torch.Tensor,
) -> None:
    """supported観測のone-hot、mask、数量、baselineを厳格検査する。"""
    counts = branch_mask.sum(dim=1)
    if not bool(((counts >= 1) & (counts <= MAX_BRANCHES)).all().item()):
        raise ProjectedSetModelInputError("supported観測のbranch数は1..20必須です")
    if not bool(quantity_present_mask.all().item()):
        raise ProjectedSetModelInputError("supported観測にquantity欠測があります")
    if not _is_finite_in_unit_interval(scalar):
        raise ProjectedSetModelInputError("scalarが非有限または0..1範囲外です")
    if not _is_finite_in_unit_interval(baseline_raw):
        raise ProjectedSetModelInputError("baseline rawが非有限または範囲外です")
    _require_onehot(current, "current")
    _require_onehot(post_chain, "post-chain")
    active = branch_mask[:, :, None, None, None, None].expand_as(landing)
    selected = landing[active].reshape(-1, COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS)
    _require_onehot(selected, "landing")
    if not bool((landing[~active] == 0.0).all().item()):
        raise ProjectedSetModelInputError("landing paddingは全0必須です")


def _validate_inference_controls(
    baseline_raw: torch.Tensor,
    baseline_calibrated: torch.Tensor,
    projected_supported: torch.Tensor,
    integrity_valid: torch.Tensor,
) -> None:
    """fallbackより先にbaselineとintegrity controlを検査する。"""
    batch_size = baseline_raw.shape[0]
    for value, label in (
        (baseline_calibrated, "baseline calibrated"),
    ):
        if value.dtype != torch.float64 or value.shape != (batch_size,):
            raise ProjectedSetModelInputError(f"{label}はfloat64[B]必須です")
        if value.device != baseline_raw.device or not _is_finite_in_unit_interval(value):
            raise ProjectedSetModelInputError(f"{label}が非有限・範囲外・device不一致です")
    if not _is_finite_in_unit_interval(baseline_raw):
        raise ProjectedSetModelInputError("baseline rawが非有限または範囲外です")
    for value, label in (
        (projected_supported, "projected supported"),
        (integrity_valid, "integrity valid"),
    ):
        if value.dtype != torch.bool or value.shape != (batch_size,):
            raise ProjectedSetModelInputError(f"{label}はbool[B]必須です")
        if value.device != baseline_raw.device:
            raise ProjectedSetModelInputError(f"{label}のdeviceが不一致です")


def _require_onehot(value: torch.Tensor, label: str) -> None:
    """盤面の各cellが8channelの厳密one-hotであることを要求する。"""
    if not bool(torch.isfinite(value).all().item()):
        raise ProjectedSetModelInputError(f"{label}盤面に非有限値があります")
    if not bool(((value == 0.0) | (value == 1.0)).all().item()):
        raise ProjectedSetModelInputError(f"{label}盤面は0/1 one-hot必須です")
    if not bool((value.sum(dim=-3) == 1.0).all().item()):
        raise ProjectedSetModelInputError(f"{label}盤面のchannel和が1ではありません")


def _pair_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """末尾2sideを[p1,p2,p1-p2]のpair表現へ変換する。"""
    p1 = embedding[..., 0, :]
    p2 = embedding[..., 1, :]
    return torch.cat((p1, p2, p1 - p2), dim=-1)


def _branch_probabilities(
    baseline_raw: torch.Tensor,
    delta: torch.Tensor,
    branch_mask: torch.Tensor,
) -> torch.Tensor:
    """EPS logitとfloat32 residualからbranch別float32勝率を返す。"""
    clipped = baseline_raw.to(torch.float32).clamp(EPS, 1.0 - EPS)
    probability = torch.sigmoid(torch.logit(clipped).unsqueeze(1) + delta)
    return torch.where(branch_mask, probability, torch.zeros_like(probability))


def _equal_branch_mean(
    branch_probability: torch.Tensor, branch_mask: torch.Tensor,
) -> torch.Tensor:
    """float32候補をfloat64へcast後、等確率masked平均する。"""
    masked = branch_probability.to(torch.float64) * branch_mask.to(torch.float64)
    count = branch_mask.sum(dim=1).to(torch.float64)
    return masked.sum(dim=1) / count


def _validate_model_output(output: ProjectedSetForwardOutputV1) -> None:
    """非有限・範囲外モデル出力をfallbackせず例外にする。"""
    finite_values = (
        output.raw_probability, output.branch_probability, output.delta,
        output.auxiliary_current, output.auxiliary_post_chain,
        output.auxiliary_landing,
    )
    if any(not bool(torch.isfinite(value).all().item()) for value in finite_values):
        raise ProjectedSetModelOutputError("projectedモデル出力に非有限値があります")
    active_probability = output.branch_probability[output.branch_mask]
    if not _is_finite_in_unit_interval(active_probability):
        raise ProjectedSetModelOutputError("branch確率が0..1範囲外です")
    if not _is_finite_in_unit_interval(output.raw_probability):
        raise ProjectedSetModelOutputError("集約確率が0..1範囲外です")


def _is_finite_in_unit_interval(value: torch.Tensor) -> bool:
    """Tensor全要素が有限かつ0..1かを返す。"""
    valid = torch.isfinite(value) & (value >= 0.0) & (value <= 1.0)
    return bool(valid.all().item())


def _validate_main_loss_inputs(
    output: ProjectedSetForwardOutputV1,
    winner_label: torch.Tensor,
    observation_weight: torch.Tensor,
) -> None:
    """主損失のlabel・観測重み契約を検査する。"""
    _validate_winner_label(output, winner_label)
    _validate_observation_weight(output, observation_weight)


def _validate_winner_label(
    output: ProjectedSetForwardOutputV1, winner_label: torch.Tensor,
) -> None:
    """winner labelのshape・dtype・device・0/1を検査する。"""
    batch_size = output.raw_probability.shape[0]
    if (winner_label.dtype != torch.float64
            or winner_label.shape != (batch_size,)
            or winner_label.device != output.raw_probability.device
            or not _is_finite_in_unit_interval(winner_label)
            or not bool(((winner_label == 0.0) | (winner_label == 1.0)).all().item())):
        raise ProjectedSetModelInputError("winner labelは同deviceのfloat64 0/1[B]必須です")


def _validate_observation_weight(
    output: ProjectedSetForwardOutputV1, observation_weight: torch.Tensor,
) -> None:
    """観測重みのshape・dtype・device・正値を検査する。"""
    batch_size = output.raw_probability.shape[0]
    if (observation_weight.dtype != torch.float64
            or observation_weight.shape != (batch_size,)
            or observation_weight.device != output.raw_probability.device
            or not bool(torch.isfinite(observation_weight).all().item())
            or not bool((observation_weight > 0.0).all().item())):
        raise ProjectedSetModelInputError("observation weightは正のfloat64[B]必須です")


def _auxiliary_loss_per_observation(
    output: ProjectedSetForwardOutputV1,
    targets: ProjectedSetAuxiliaryTargetsV1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """current/post/landingを各1票にした観測別補助MSEを返す。"""
    _validate_auxiliary_targets(output, targets)
    current, current_valid = _auxiliary_group_loss(
        output.auxiliary_current, targets.current, targets.current_mask, side_dim=1,
    )
    post, post_valid = _auxiliary_group_loss(
        output.auxiliary_post_chain, targets.post_chain,
        targets.post_chain_mask, side_dim=1,
    )
    landing_board, landing_board_valid = _board_auxiliary_mse(
        output.auxiliary_landing, targets.landing, targets.landing_mask,
    )
    landing_side, landing_side_valid = _masked_mean(
        landing_board, landing_board_valid, dim=2,
    )
    landing, landing_valid = _masked_mean(
        landing_side, landing_side_valid & output.branch_mask, dim=1,
    )
    group_values = torch.stack((current, post, landing), dim=1)
    group_valid = torch.stack((current_valid, post_valid, landing_valid), dim=1)
    return _masked_mean(group_values, group_valid, dim=1)


def _auxiliary_group_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    side_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """盤面別8列平均の後、side間を単純平均する。"""
    board_loss, board_valid = _board_auxiliary_mse(prediction, target, mask)
    return _masked_mean(board_loss, board_valid, dim=side_dim)


def _board_auxiliary_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """validな8指標だけを盤面内平均し、missingのgradientを0にする。"""
    safe_target = torch.where(mask, target, prediction.detach())
    squared = (prediction - safe_target).square()
    return _masked_mean(squared, mask, dim=-1)


def _masked_mean(
    value: torch.Tensor, mask: torch.Tensor, dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """全欠測を0値+present=falseとして返すmasked平均。"""
    count = mask.sum(dim=dim)
    present = count > 0
    numerator = torch.where(mask, value, torch.zeros_like(value)).sum(dim=dim)
    safe_count = count.clamp_min(1).to(value.dtype)
    return numerator / safe_count, present


def _validate_auxiliary_targets(
    output: ProjectedSetForwardOutputV1,
    targets: ProjectedSetAuxiliaryTargetsV1,
) -> None:
    """補助targetのshape/dtype/range/padding maskを検査する。"""
    entries = (
        (targets.current, targets.current_mask, output.auxiliary_current, "current"),
        (targets.post_chain, targets.post_chain_mask,
         output.auxiliary_post_chain, "post-chain"),
        (targets.landing, targets.landing_mask, output.auxiliary_landing, "landing"),
    )
    for target, mask, prediction, label in entries:
        if target.dtype != torch.float32 or target.shape != prediction.shape:
            raise ProjectedSetModelInputError(f"{label}補助target shape/dtype不一致です")
        if mask.dtype != torch.bool or mask.shape != prediction.shape:
            raise ProjectedSetModelInputError(f"{label}補助mask shape/dtype不一致です")
        if target.device != prediction.device or mask.device != prediction.device:
            raise ProjectedSetModelInputError(f"{label}補助target device不一致です")
        if mask.any() and not _is_finite_in_unit_interval(target[mask]):
            raise ProjectedSetModelInputError(f"{label}補助targetが0..1範囲外です")
    active = output.branch_mask[:, :, None, None].expand_as(targets.landing_mask)
    if bool((targets.landing_mask & ~active).any().item()):
        raise ProjectedSetModelInputError("padding branchに補助targetがあります")


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """正の観測重みで平均する。"""
    return (value * weight).sum() / weight.sum()


def _weighted_present_mean(
    value: torch.Tensor, weight: torch.Tensor, present: torch.Tensor,
) -> torch.Tensor:
    """補助正解を一つ以上持つ観測だけで重み付き平均する。"""
    active_weight = weight * present.to(weight.dtype)
    if not bool(present.any().item()):
        return value.sum() * 0.0
    return (value.to(torch.float64) * active_weight).sum() / active_weight.sum()


def _zero_auxiliary_loss(output: ProjectedSetForwardOutputV1) -> torch.Tensor:
    """補助targetなしでもheadへ0 gradientを接続する。"""
    zero = output.auxiliary_current.sum() + output.auxiliary_post_chain.sum()
    zero = zero + output.auxiliary_landing.sum()
    return zero.to(torch.float64) * 0.0


def _zero_auxiliary_terms(
    output: ProjectedSetForwardOutputV1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """全補助headへ0 gradientを接続した観測別0とpresent=falseを返す。"""

    current = output.auxiliary_current.sum(dim=(1, 2))
    post = output.auxiliary_post_chain.sum(dim=(1, 2))
    landing = output.auxiliary_landing.sum(dim=(1, 2, 3))
    zero = (current + post + landing).to(torch.float64) * 0.0
    return zero, torch.zeros_like(zero, dtype=torch.bool)


__all__ = [
    "AUXILIARY_DIM", "AUXILIARY_LOSS_COEFFICIENT", "EPS", "MAX_BRANCHES",
    "PositionPreservingBoardEncoderV1", "ProjectedSetAuxiliaryTargetsV1",
    "ProjectedSetForwardOutputV1", "ProjectedSetInferenceOutputV1",
    "ProjectedSetLossV1", "ProjectedSetModelInputError",
    "ProjectedSetModelIntegrityError", "ProjectedSetModelOutputError",
    "ProjectedSetResidualCNNV1", "ProjectedSetUnreducedLossV1",
    "equal_game_observation_weights", "projected_set_unreduced_loss",
    "projected_set_weighted_loss", "symmetric_platt_probability",
]
