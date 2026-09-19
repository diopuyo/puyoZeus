"""Formal100 projected-setモデルのlabel開封前学習基盤。

動画処理や本番設定から独立し、固定fold、試合均等weight、決定論的な
microbatch学習と実行環境receiptだけを提供する。
"""

from __future__ import annotations

import hashlib
import math
import os
import platform
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
import torch

from src.event_provisional_oof_v1 import FoldPlan, fixed_fold_plan
from src.projected_set_residual_cnn_v1 import (
    AUXILIARY_LOSS_COEFFICIENT,
    ProjectedSetAuxiliaryTargetsV1,
    ProjectedSetResidualCNNV1,
    projected_set_unreduced_loss,
)
from src.projected_state_batch_v1 import ProjectedStateBatchV1


FOLD_IDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
LOGICAL_BATCH_SIZE: int = 32
MICROBATCH_SIZE: int = 8
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
WEIGHT_SUM_ABS_TOLERANCE: float = 1e-12
GRADIENT_CLIP_NORM: float = 1.0
SCHEDULER_T_MAX: int = 50
SCHEDULER_ETA_MIN: float = 1e-5
EPOCH_COUNT: int = 50
DATA_LOADER_WORKERS: int = 0
TRAINING_SEEDS: tuple[int, ...] = (20260831, 20260832, 20260833)
CUBLAS_WORKSPACE_CONFIG: str = ":4096:8"
CHAIN_BITBOARD_PATH: str = "src/chain_bitboard.py"
UNKNOWN_VALUE: str = "unknown"
_CONFIGURED_TRAINING_SEED: int | None = None

Float64Array = npt.NDArray[np.float64]
GameKey = tuple[str, int]


class ProjectedSetTrainingError(ValueError):
    """固定学習契約を満たさない入力を拒否する。"""


@dataclass(frozen=True, slots=True)
class ProjectedSetTrainStepV1:
    """一logical batchにつき一回だけ更新した損失とclip前norm。"""

    total: float
    binary_cross_entropy: float
    auxiliary_mse: float
    gradient_norm_before_clip: float
    observation_count: int
    microbatch_count: int


def validate_fixed_fold_assignment_v1(
    source_video_ids: Sequence[str],
    source_group_ids: Sequence[str],
    game_indices: Sequence[int],
    folds: Sequence[int],
) -> tuple[FoldPlan, ...]:
    """source/gameがfoldを跨がず1〜6が全てある固定計画を返す。"""

    sources, groups, games, fold_values = _validated_assignment_arrays(
        source_video_ids, source_group_ids, game_indices, folds,
    )
    violations = []
    if _keys_cross_folds(sources, fold_values):
        violations.append("source_video_idが複数foldへ跨いでいます")
    if _keys_cross_folds(groups, fold_values):
        violations.append("source_group_idが複数foldへ跨いでいます")
    composite = tuple(zip(sources, games, strict=True))
    if _keys_cross_folds(composite, fold_values):
        violations.append("複合game keyが複数foldへ跨いでいます")
    observed = set(fold_values)
    if observed != set(FOLD_IDS):
        violations.append(f"fold 1〜6が全て必要です: {sorted(observed)}")
    if violations:
        raise ProjectedSetTrainingError("; ".join(violations))
    return fixed_fold_plan(fold_values)


def precompute_equal_game_weights_v1(
    source_video_ids: Sequence[str], game_indices: Sequence[int],
) -> Float64Array:
    """全fit対象で複合gameごとの観測重み合計を1に固定する。"""

    sources, games = _validated_game_keys(source_video_ids, game_indices)
    keys = tuple(zip(sources, games, strict=True))
    counts: dict[GameKey, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    weights = np.asarray([1.0 / counts[key] for key in keys], dtype=np.float64)
    _validate_precomputed_weights(keys, weights)
    weights.setflags(write=False)
    return weights


def deterministic_epoch_order_v1(
    seed: int, epoch: int, observation_input_digests: Sequence[str],
) -> tuple[int, ...]:
    """仕様固定hashの昇順でfit行indexを返し、重複digestを拒否する。"""

    _validate_order_integer(seed, "seed")
    _validate_order_integer(epoch, "epoch")
    digests = tuple(observation_input_digests)
    if not digests:
        raise ProjectedSetTrainingError("epoch order対象が空です")
    for digest in digests:
        _validate_observation_digest(digest)
    if len(set(digests)) != len(digests):
        raise ProjectedSetTrainingError("observation_input_digestが重複しています")
    keyed: list[tuple[bytes, str, int]] = []
    for index, digest in enumerate(digests):
        payload = f"{seed}|{epoch}|{digest}".encode("ascii")
        keyed.append((hashlib.sha256(payload).digest(), digest, index))
    return tuple(value[2] for value in sorted(keyed))


def make_projected_optimizer_v1(
    model: ProjectedSetResidualCNNV1,
) -> torch.optim.AdamW:
    """凍結済みAdamW設定を生成する。"""

    if not isinstance(model, ProjectedSetResidualCNNV1):
        raise ProjectedSetTrainingError("model型がProjectedSetResidualCNNV1ではありません")
    return torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        foreach=False,
    )


def make_projected_scheduler_v1(
    optimizer: torch.optim.AdamW,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """50 epoch固定のcosine schedulerを生成する。"""

    _validate_optimizer(optimizer, require_initial_rate=True)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX, eta_min=SCHEDULER_ETA_MIN,
    )


def train_projected_logical_batch_v1(
    model: ProjectedSetResidualCNNV1,
    optimizer: torch.optim.AdamW,
    batch: ProjectedStateBatchV1,
    baseline_raw: torch.Tensor,
    winner_label: torch.Tensor,
    observation_weight: torch.Tensor,
) -> ProjectedSetTrainStepV1:
    """最大32件を8件ずつ処理し、一括損失と等価な一更新を行う。"""

    count = _validate_training_batch(
        model, batch, baseline_raw, winner_label, observation_weight,
    )
    _validate_optimizer(optimizer)
    _validate_optimizer_model_binding(model, optimizer)
    optimizer.zero_grad(set_to_none=True)
    binary_value, auxiliary_value = _accumulate_logical_batch_gradients(
        model, batch, baseline_raw, winner_label, observation_weight, count,
    )
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
    optimizer.step()
    return _step_result(binary_value, auxiliary_value, norm, count)


def _accumulate_logical_batch_gradients(
    model: ProjectedSetResidualCNNV1,
    batch: ProjectedStateBatchV1,
    baseline_raw: torch.Tensor,
    winner_label: torch.Tensor,
    observation_weight: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """logical分母を固定し、step前gradientをmicrobatchで蓄積する。"""

    auxiliary_denominator = _auxiliary_weight_denominator(
        batch.auxiliary_targets, observation_weight,
    )
    binary_denominator = observation_weight.sum()
    model.zero_grad(set_to_none=True)
    binary_value = baseline_raw.new_zeros(())
    auxiliary_value = baseline_raw.new_zeros(())
    for start in range(0, count, MICROBATCH_SIZE):
        stop = min(start + MICROBATCH_SIZE, count)
        contributions = _microbatch_contributions(
            model, batch, baseline_raw, winner_label, observation_weight,
            start, stop, binary_denominator, auxiliary_denominator,
        )
        (contributions[0] + AUXILIARY_LOSS_COEFFICIENT * contributions[1]).backward()
        binary_value = binary_value + contributions[0].detach()
        auxiliary_value = auxiliary_value + contributions[1].detach()
    return binary_value, auxiliary_value


def configure_deterministic_training_v1(seed: int) -> None:
    """Formal100学習の乱数源とPyTorch決定性を固定する。"""

    global _CONFIGURED_TRAINING_SEED
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ProjectedSetTrainingError("seedは0以上の整数必須です")
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if torch.cuda.is_initialized() and configured != CUBLAS_WORKSPACE_CONFIG:
        raise ProjectedSetTrainingError(
            "CUDA初期化後のCUBLAS_WORKSPACE_CONFIG不一致です",
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _CONFIGURED_TRAINING_SEED = seed


def build_runtime_receipt_v1(
    asset_paths: Sequence[str | Path],
    *,
    training_seed: int,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """環境情報と指定資産のSHA-256をJSON互換objectで返す。"""

    if training_seed not in TRAINING_SEEDS:
        raise ProjectedSetTrainingError("receiptのseedが凍結3seedに含まれません")
    if _CONFIGURED_TRAINING_SEED != training_seed:
        raise ProjectedSetTrainingError("receiptのseedと決定性設定seedが一致しません")
    _validate_runtime_determinism()
    root = Path(project_root) if project_root is not None else Path(__file__).parents[1]
    assets = _asset_sha256(root.resolve(), asset_paths)
    gpu_names, capabilities = _gpu_information()
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda or UNKNOWN_VALUE,
        "cuda_driver_version": _cuda_driver_version(),
        "cudnn_version": torch.backends.cudnn.version() or UNKNOWN_VALUE,
        "gpu_names": gpu_names,
        "gpu_capabilities": capabilities,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG", UNKNOWN_VALUE,
        ),
        "training_config": _fixed_training_config(training_seed),
        "asset_sha256": assets,
    }


def _validated_assignment_arrays(
    source_video_ids: Sequence[str],
    source_group_ids: Sequence[str],
    game_indices: Sequence[int],
    folds: Sequence[int],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    sources, games = _validated_game_keys(source_video_ids, game_indices)
    groups = _validated_source_ids(source_group_ids, "source_group_id")
    if len(groups) != len(sources) or len(folds) != len(sources):
        raise ProjectedSetTrainingError("source groupまたはfoldの行数が一致しません")
    fold_values = tuple(_strict_integer(value, "fold") for value in folds)
    return sources, groups, games, fold_values


def _validated_game_keys(
    source_video_ids: Sequence[str], game_indices: Sequence[int],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if len(source_video_ids) == 0 or len(source_video_ids) != len(game_indices):
        raise ProjectedSetTrainingError("fit対象のsource/game行数が不正です")
    sources = _validated_source_ids(source_video_ids, "source_video_id")
    games = tuple(_strict_integer(value, "game_idx") for value in game_indices)
    if any(value < 0 for value in games):
        raise ProjectedSetTrainingError("game_idxは0以上必須です")
    return sources, games


def _validated_source_ids(values: Sequence[str], label: str) -> tuple[str, ...]:
    if any(not isinstance(value, str) or not value or value != value.strip()
           for value in values):
        raise ProjectedSetTrainingError(f"{label}は空白なしの非空文字列必須です")
    return tuple(values)


def _strict_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ProjectedSetTrainingError(f"{label}は整数必須です")
    return int(value)


def _validate_order_integer(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProjectedSetTrainingError(f"{label}は0以上の整数必須です")


def _validate_observation_digest(value: object) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ProjectedSetTrainingError("observation_input_digestは64桁hex必須です")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ProjectedSetTrainingError("observation_input_digestは64桁hex必須です") from exc
    if len(decoded) != 32 or value != value.lower():
        raise ProjectedSetTrainingError("observation_input_digestは小文字64桁hex必須です")


def _keys_cross_folds(keys: Sequence[object], folds: Sequence[int]) -> bool:
    assignments: dict[object, int] = {}
    for key, fold in zip(keys, folds, strict=True):
        previous = assignments.setdefault(key, fold)
        if previous != fold:
            return True
    return False


def _validate_precomputed_weights(
    keys: Sequence[GameKey], weights: Float64Array,
) -> None:
    grouped: dict[GameKey, list[float]] = {}
    for key, weight in zip(keys, weights, strict=True):
        grouped.setdefault(key, []).append(float(weight))
    if any(not math.isclose(
        math.fsum(values), 1.0, rel_tol=0.0,
        abs_tol=WEIGHT_SUM_ABS_TOLERANCE,
    ) for values in grouped.values()):
        raise ProjectedSetTrainingError("game均等weightの事前計算に失敗しました")


def _validate_optimizer(
    optimizer: torch.optim.Optimizer, *, require_initial_rate: bool = False,
) -> None:
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise ProjectedSetTrainingError("optimizerは固定AdamW単一group必須です")
    group = optimizer.param_groups[0]
    rate_valid = (
        group["lr"] == LEARNING_RATE if require_initial_rate
        else SCHEDULER_ETA_MIN <= group["lr"] <= LEARNING_RATE
    )
    if (not rate_valid or group["weight_decay"] != WEIGHT_DECAY
            or group.get("foreach") is not False):
        raise ProjectedSetTrainingError("AdamW固定設定が一致しません")


def _validate_training_batch(
    model: ProjectedSetResidualCNNV1,
    batch: ProjectedStateBatchV1,
    baseline_raw: torch.Tensor,
    winner_label: torch.Tensor,
    observation_weight: torch.Tensor,
) -> int:
    if not isinstance(model, ProjectedSetResidualCNNV1):
        raise ProjectedSetTrainingError("model型が不正です")
    if not isinstance(batch, ProjectedStateBatchV1):
        raise ProjectedSetTrainingError("batch型が不正です")
    count = batch.current.shape[0]
    if count < 1 or count > LOGICAL_BATCH_SIZE:
        raise ProjectedSetTrainingError("logical batchは1..32件必須です")
    device = batch.current.device
    entries = (baseline_raw, winner_label, observation_weight)
    if any(value.shape != (count,) or value.dtype != torch.float64 for value in entries):
        raise ProjectedSetTrainingError("baseline/label/weightはfloat64[B]必須です")
    if any(value.device != device for value in entries):
        raise ProjectedSetTrainingError("学習Tensorのdeviceが一致しません")
    if not bool(torch.isfinite(observation_weight).all().item()) \
            or not bool((observation_weight > 0.0).all().item()):
        raise ProjectedSetTrainingError("事前計算weightは有限な正値必須です")
    weight_sum = observation_weight.sum()
    if not bool(torch.isfinite(weight_sum).item()) or weight_sum.item() <= 0.0:
        raise ProjectedSetTrainingError("logical batchのweight合計が非有限です")
    return count


def _validate_optimizer_model_binding(
    model: ProjectedSetResidualCNNV1, optimizer: torch.optim.AdamW,
) -> None:
    model_parameters = [id(parameter) for parameter in model.parameters()]
    optimizer_parameters = [
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    ]
    unique_optimizer = set(optimizer_parameters)
    if (len(optimizer_parameters) != len(unique_optimizer)
            or len(model_parameters) != len(set(model_parameters))
            or unique_optimizer != set(model_parameters)):
        raise ProjectedSetTrainingError(
            "optimizerとmodelのparameterが1対1で一致しません",
        )


def _auxiliary_weight_denominator(
    targets: ProjectedSetAuxiliaryTargetsV1, weights: torch.Tensor,
) -> torch.Tensor:
    present = (
        targets.current_mask.flatten(1).any(1)
        | targets.post_chain_mask.flatten(1).any(1)
        | targets.landing_mask.flatten(1).any(1)
    )
    return (weights * present.to(weights.dtype)).sum()


def _microbatch_contributions(
    model: ProjectedSetResidualCNNV1,
    batch: ProjectedStateBatchV1,
    baseline: torch.Tensor,
    label: torch.Tensor,
    weight: torch.Tensor,
    start: int,
    stop: int,
    binary_denominator: torch.Tensor,
    auxiliary_denominator: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    section = slice(start, stop)
    output = model(
        batch.current[section], batch.post_chain[section], batch.landing[section],
        batch.branch_mask[section], batch.scalar[section],
        batch.quantity_present_mask[section], baseline[section],
    )
    targets = _slice_auxiliary(batch.auxiliary_targets, section)
    terms = projected_set_unreduced_loss(output, label[section], targets)
    selected_weight = weight[section]
    binary = (terms.binary_cross_entropy * selected_weight).sum() / binary_denominator
    auxiliary = _normalized_auxiliary_contribution(
        terms.auxiliary_mse, terms.auxiliary_present, selected_weight,
        auxiliary_denominator,
    )
    return binary, auxiliary


def _slice_auxiliary(
    targets: ProjectedSetAuxiliaryTargetsV1, section: slice,
) -> ProjectedSetAuxiliaryTargetsV1:
    return ProjectedSetAuxiliaryTargetsV1(
        current=targets.current[section], post_chain=targets.post_chain[section],
        landing=targets.landing[section],
        current_mask=targets.current_mask[section],
        post_chain_mask=targets.post_chain_mask[section],
        landing_mask=targets.landing_mask[section],
    )


def _normalized_auxiliary_contribution(
    loss: torch.Tensor,
    present: torch.Tensor,
    weight: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    numerator = (loss * weight * present.to(weight.dtype)).sum()
    if denominator.item() == 0.0:
        return numerator * 0.0
    return numerator / denominator


def _step_result(
    binary: torch.Tensor,
    auxiliary: torch.Tensor,
    norm: torch.Tensor,
    count: int,
) -> ProjectedSetTrainStepV1:
    binary_value = float(binary.item())
    auxiliary_value = float(auxiliary.item())
    return ProjectedSetTrainStepV1(
        total=binary_value + AUXILIARY_LOSS_COEFFICIENT * auxiliary_value,
        binary_cross_entropy=binary_value,
        auxiliary_mse=auxiliary_value,
        gradient_norm_before_clip=float(norm.item()),
        observation_count=count,
        microbatch_count=(count + MICROBATCH_SIZE - 1) // MICROBATCH_SIZE,
    )


def _asset_sha256(
    root: Path, asset_paths: Sequence[str | Path],
) -> dict[str, str]:
    requested = [Path(value) for value in asset_paths]
    required = Path(CHAIN_BITBOARD_PATH)
    if required not in requested:
        requested.append(required)
    result: dict[str, str] = {}
    for requested_path in requested:
        path = requested_path if requested_path.is_absolute() else root / requested_path
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ProjectedSetTrainingError(f"学習資産を読めません: {requested_path}") from exc
        key = _receipt_asset_key(root, path.resolve())
        result[key] = hashlib.sha256(payload).hexdigest()
    return dict(sorted(result.items()))


def _receipt_asset_key(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _cuda_driver_version() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10, shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_VALUE
    values = sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})
    return ",".join(values) if values else UNKNOWN_VALUE


def _gpu_information() -> tuple[list[str], list[str]]:
    if not torch.cuda.is_available():
        return [UNKNOWN_VALUE], [UNKNOWN_VALUE]
    try:
        count = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(index) for index in range(count)]
        capabilities = [
            ".".join(map(str, torch.cuda.get_device_capability(index)))
            for index in range(count)
        ]
    except (RuntimeError, AssertionError):
        return [UNKNOWN_VALUE], [UNKNOWN_VALUE]
    return names or [UNKNOWN_VALUE], capabilities or [UNKNOWN_VALUE]


def _validate_runtime_determinism() -> None:
    valid = (
        torch.are_deterministic_algorithms_enabled()
        and torch.backends.cudnn.deterministic
        and not torch.backends.cudnn.benchmark
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_WORKSPACE_CONFIG
    )
    if not valid:
        raise ProjectedSetTrainingError("決定性設定が未完了のためreceiptを発行できません")


def _fixed_training_config(training_seed: int) -> dict[str, Any]:
    return {
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "foreach": False,
        "scheduler": "CosineAnnealingLR", "scheduler_t_max": SCHEDULER_T_MAX,
        "scheduler_eta_min": SCHEDULER_ETA_MIN, "epochs": EPOCH_COUNT,
        "logical_batch_size": LOGICAL_BATCH_SIZE,
        "microbatch_size": MICROBATCH_SIZE, "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "data_loader_workers": DATA_LOADER_WORKERS, "dtype": "float32", "amp": False,
        "seed": training_seed, "allowed_seeds": list(TRAINING_SEEDS),
    }


__all__ = [
    "CUBLAS_WORKSPACE_CONFIG", "EPOCH_COUNT", "GRADIENT_CLIP_NORM", "LEARNING_RATE",
    "LOGICAL_BATCH_SIZE", "MICROBATCH_SIZE", "ProjectedSetTrainStepV1",
    "ProjectedSetTrainingError", "SCHEDULER_ETA_MIN", "SCHEDULER_T_MAX",
    "TRAINING_SEEDS", "WEIGHT_DECAY", "build_runtime_receipt_v1",
    "configure_deterministic_training_v1", "deterministic_epoch_order_v1",
    "make_projected_optimizer_v1",
    "make_projected_scheduler_v1", "precompute_equal_game_weights_v1",
    "train_projected_logical_batch_v1", "validate_fixed_fold_assignment_v1",
]
