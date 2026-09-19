"""ProjectedStateObservationV1を凍結済み学習tensorへ変換する。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src import indicators_v2
from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
    Board,
)
from src.projected_state_observation_v1 import (
    DualBoardState,
    IntegerQuantity,
    MAX_LANDING_BRANCHES,
    ProjectedBoardState,
    ProjectedStateObservationV1,
)


# 2026-09-19 再固定 (user 承認)。manifest の dependency_sha256 が
# src/production_config.py を 2026-09-04 の作業コピーの値で止めており、
# その後のフラグ採用のたびに不一致になっていた。manifest が実際に依存する定数は
# GHOST_CHAIN_RULE_ENABLED だけで、値は True のまま変わっていない。
AUXILIARY_MANIFEST_SHA256 = (
    "d4fdfc986c3543529fcccc9e78aca377f3df194f23c3ebdf41c0d232b61bf0aa"
)
AUXILIARY_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/manifests/FORMAL100_PROJECTED_AUX_INDICATORS_V1.json"
)
BOARD_CHANNEL_VALUES = (
    COLOR_EMPTY,
    COLOR_RED,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_PURPLE,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
)
AUXILIARY_FEATURE_NAMES = (
    "board_color_puyo_total",
    "board_ojama_count",
    "max_column_height",
    "death_margin",
    "column_bumpiness",
    "center_bulge_color",
    "center_bulge_ojama",
    "color_diversity_evenness",
)
SIDE_ORDER = ("p1", "p2")
CHANNEL_COUNT = len(BOARD_CHANNEL_VALUES)
AUXILIARY_FEATURE_COUNT = len(AUXILIARY_FEATURE_NAMES)
SCALAR_COUNT = 6
OJAMA_NORMALIZER = 30.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TENSOR_CONTENT_DIGEST_DOMAIN = b"projected-state-tensor-v1\0"
TENSOR_ARRAY_FIELDS = (
    "current", "post_chain", "landing", "branch_mask", "scalar",
    "quantity_present_mask", "auxiliary_current", "auxiliary_post_chain",
    "auxiliary_landing", "auxiliary_current_mask", "auxiliary_post_chain_mask",
    "auxiliary_landing_mask",
)

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
IndicatorFunction = Callable[[Board], Any]


class ProjectedStateTensorizationError(ValueError):
    """tensor化入力または凍結資産が正式契約を満たさない。"""


@dataclass(frozen=True, slots=True)
class ProjectedStateTensorV1:
    """単一projected-state観測の入力tensorと補助教師。"""

    current: FloatArray
    post_chain: FloatArray
    landing: FloatArray
    branch_mask: BoolArray
    scalar: FloatArray
    quantity_present_mask: BoolArray
    candidate_count: np.int8
    auxiliary_current: FloatArray
    auxiliary_post_chain: FloatArray
    auxiliary_landing: FloatArray
    auxiliary_current_mask: BoolArray
    auxiliary_post_chain_mask: BoolArray
    auxiliary_landing_mask: BoolArray
    observation_input_digest: str
    auxiliary_manifest_sha256: str
    tensor_content_digest: str


def tensorize_projected_state_observation_v1(
    observation: ProjectedStateObservationV1,
    auxiliary_manifest_path: str | Path = AUXILIARY_MANIFEST_PATH,
) -> ProjectedStateTensorV1:
    """DTOをモデル入力と8補助indicator教師へ決定論的に変換する。"""
    if not isinstance(observation, ProjectedStateObservationV1):
        raise ProjectedStateTensorizationError("入力はProjectedStateObservationV1必須です")
    manifest = _load_frozen_auxiliary_manifest(Path(auxiliary_manifest_path))
    functions = _resolve_indicator_functions(manifest)
    current, current_aux, current_aux_mask = _tensorize_dual(
        observation.current_state, functions,
    )
    post, post_aux, post_aux_mask = _tensorize_dual(
        observation.post_chain_state, functions,
    )
    landing = _tensorize_landing(observation, functions)
    scalar, quantity_mask = _tensorize_scalars(observation)
    value = ProjectedStateTensorV1(
        current=current, post_chain=post, landing=landing[0],
        branch_mask=landing[1], scalar=scalar,
        quantity_present_mask=quantity_mask,
        candidate_count=np.int8(len(observation.landing_branches)),
        auxiliary_current=current_aux,
        auxiliary_post_chain=post_aux,
        auxiliary_landing=landing[2],
        auxiliary_current_mask=current_aux_mask,
        auxiliary_post_chain_mask=post_aux_mask,
        auxiliary_landing_mask=landing[3],
        observation_input_digest=observation.input_digest,
        auxiliary_manifest_sha256=AUXILIARY_MANIFEST_SHA256,
        tensor_content_digest="",
    )
    _make_tensor_arrays_read_only(value)
    return replace(
        value, tensor_content_digest=projected_state_tensor_content_digest_v1(value),
    )


def tensorize_board_auxiliary_targets_v1(
    board: Board,
    auxiliary_manifest_path: str | Path = AUXILIARY_MANIFEST_PATH,
) -> tuple[FloatArray, BoolArray]:
    """単一の確定盤面を凍結済み8補助indicator教師へ変換する。"""

    if not isinstance(board, Board):
        raise ProjectedStateTensorizationError("入力はBoard必須です")
    manifest = _load_frozen_auxiliary_manifest(Path(auxiliary_manifest_path))
    functions = _resolve_indicator_functions(manifest)
    targets, mask = _board_auxiliary_targets(board, functions)
    targets.setflags(write=False)
    mask.setflags(write=False)
    return targets, mask


def projected_state_tensor_content_digest_v1(
    value: ProjectedStateTensorV1,
) -> str:
    """DTO digestと全tensor内容を固定順で結合したSHA-256を返す。"""

    if not isinstance(value, ProjectedStateTensorV1):
        raise ProjectedStateTensorizationError("tensor content digest対象が不正です")
    digest = hashlib.sha256(TENSOR_CONTENT_DIGEST_DOMAIN)
    _update_digest_text(digest, value.observation_input_digest)
    _update_digest_text(digest, value.auxiliary_manifest_sha256)
    _update_digest_text(digest, str(int(value.candidate_count)))
    for name in TENSOR_ARRAY_FIELDS:
        array = getattr(value, name)
        if not isinstance(array, np.ndarray):
            raise ProjectedStateTensorizationError(f"tensor配列が不正です: {name}")
        _update_digest_array(digest, name, array)
    return digest.hexdigest()


def _update_digest_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _update_digest_array(digest: Any, name: str, array: np.ndarray) -> None:
    _update_digest_text(digest, name)
    _update_digest_text(digest, array.dtype.str)
    _update_digest_text(digest, ",".join(str(size) for size in array.shape))
    contiguous = np.ascontiguousarray(array)
    digest.update(contiguous.nbytes.to_bytes(8, "big"))
    digest.update(contiguous.tobytes(order="C"))


def _make_tensor_arrays_read_only(value: ProjectedStateTensorV1) -> None:
    """通常経路のtensor配列をread-onlyにして偶発的な変更を防ぐ。"""

    for name in TENSOR_ARRAY_FIELDS:
        getattr(value, name).setflags(write=False)


@lru_cache(maxsize=8)
def _load_frozen_auxiliary_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProjectedStateTensorizationError(
            f"補助indicator manifestを読めません: {path}"
        ) from exc
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != AUXILIARY_MANIFEST_SHA256:
        raise ProjectedStateTensorizationError(
            f"補助indicator manifest SHA-256不一致: {actual_hash}"
        )
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectedStateTensorizationError("補助indicator manifestが不正です") from exc
    if not isinstance(manifest, dict):
        raise ProjectedStateTensorizationError("補助indicator manifestはobject必須です")
    _verify_dependency_sha256(manifest)
    return manifest


def _verify_dependency_sha256(manifest: dict[str, Any]) -> None:
    dependencies = manifest.get("dependency_sha256")
    if not isinstance(dependencies, dict) or not dependencies:
        raise ProjectedStateTensorizationError("補助indicator依存SHA-256がありません")
    for relative_path, expected in dependencies.items():
        if not isinstance(relative_path, str) or not isinstance(expected, str):
            raise ProjectedStateTensorizationError("補助indicator依存SHA-256が不正です")
        path = PROJECT_ROOT / relative_path
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ProjectedStateTensorizationError(
                f"補助indicator依存資産を読めません: {relative_path}"
            ) from exc
        if actual != expected:
            raise ProjectedStateTensorizationError(
                f"補助indicator依存SHA-256不一致: {relative_path}"
            )


def _resolve_indicator_functions(
    manifest: dict[str, Any],
) -> tuple[IndicatorFunction, ...]:
    if tuple(manifest.get("feature_order", ())) != AUXILIARY_FEATURE_NAMES:
        raise ProjectedStateTensorizationError("補助indicator列順が凍結仕様と不一致です")
    entries = manifest.get("features")
    if not isinstance(entries, list) or len(entries) != AUXILIARY_FEATURE_COUNT:
        raise ProjectedStateTensorizationError("補助indicator定義数が不正です")
    functions: list[IndicatorFunction] = []
    for name, entry in zip(AUXILIARY_FEATURE_NAMES, entries, strict=True):
        expected_callable = f"src.indicators_v2.{name}"
        if not isinstance(entry, dict) or entry.get("callable") != expected_callable:
            raise ProjectedStateTensorizationError("補助indicator callableが不正です")
        function = getattr(indicators_v2, name, None)
        if not callable(function):
            raise ProjectedStateTensorizationError(f"補助indicatorが呼出不能です: {name}")
        functions.append(function)
    return tuple(functions)


def _tensorize_dual(
    state: DualBoardState,
    functions: tuple[IndicatorFunction, ...],
) -> tuple[FloatArray, FloatArray, BoolArray]:
    boards = (state.p1, state.p2)
    tensors = np.stack([_one_hot(board) for board in boards]).astype(np.float32)
    auxiliary_pairs = [_auxiliary_targets(board, functions) for board in boards]
    auxiliary = np.stack([pair[0] for pair in auxiliary_pairs]).astype(np.float32)
    masks = np.stack([pair[1] for pair in auxiliary_pairs]).astype(np.bool_)
    return tensors, auxiliary, masks


def _one_hot(state: ProjectedBoardState) -> FloatArray:
    shape = (CHANNEL_COUNT, BOARD_ROWS, BOARD_COLS)
    encoded = np.zeros(shape, dtype=np.float32)
    if not state.present:
        return encoded
    grid = np.asarray(state.grid, dtype=np.int64)
    for channel, value in enumerate(BOARD_CHANNEL_VALUES):
        encoded[channel] = grid == value
    if not np.all(encoded.sum(axis=0) == 1.0):
        raise ProjectedStateTensorizationError("盤面に未登録cell値があります")
    return encoded


def _auxiliary_targets(
    state: ProjectedBoardState,
    functions: tuple[IndicatorFunction, ...],
) -> tuple[FloatArray, BoolArray]:
    targets = np.zeros(AUXILIARY_FEATURE_COUNT, dtype=np.float32)
    mask = np.zeros(AUXILIARY_FEATURE_COUNT, dtype=np.bool_)
    if not state.present:
        return targets, mask
    board = Board.from_list([list(row) for row in state.grid or ()])
    return _board_auxiliary_targets(board, functions)


def _board_auxiliary_targets(
    board: Board, functions: tuple[IndicatorFunction, ...],
) -> tuple[FloatArray, BoolArray]:
    """検証済みcallableだけから単一盤面の補助教師を作る。"""

    targets = np.zeros(AUXILIARY_FEATURE_COUNT, dtype=np.float32)
    mask = np.ones(AUXILIARY_FEATURE_COUNT, dtype=np.bool_)
    for index, function in enumerate(functions):
        result = function(board)
        score = getattr(result, "score", None)
        _validate_auxiliary_score(score, AUXILIARY_FEATURE_NAMES[index])
        targets[index] = np.float32(score)
    return targets, mask


def _validate_auxiliary_score(score: object, name: str) -> None:
    if isinstance(score, bool) or not isinstance(score, (int, float, np.number)):
        raise ProjectedStateTensorizationError(f"補助indicator値が数値ではありません: {name}")
    value = float(score)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ProjectedStateTensorizationError(
            f"補助indicator値が0..1の有限値ではありません: {name}={value}"
        )


def _tensorize_landing(
    observation: ProjectedStateObservationV1,
    functions: tuple[IndicatorFunction, ...],
) -> tuple[FloatArray, BoolArray, FloatArray, BoolArray]:
    board_shape = (MAX_LANDING_BRANCHES, 2, CHANNEL_COUNT, BOARD_ROWS, BOARD_COLS)
    aux_shape = (MAX_LANDING_BRANCHES, 2, AUXILIARY_FEATURE_COUNT)
    tensors = np.zeros(board_shape, dtype=np.float32)
    branch_mask = np.zeros(MAX_LANDING_BRANCHES, dtype=np.bool_)
    auxiliary = np.zeros(aux_shape, dtype=np.float32)
    auxiliary_mask = np.zeros(aux_shape, dtype=np.bool_)
    for index, branch in enumerate(observation.landing_branches):
        dual, aux, aux_mask = _tensorize_dual(branch.boards, functions)
        tensors[index], auxiliary[index], auxiliary_mask[index] = dual, aux, aux_mask
        branch_mask[index] = True
    return tensors, branch_mask, auxiliary, auxiliary_mask


def _tensorize_scalars(
    observation: ProjectedStateObservationV1,
) -> tuple[FloatArray, BoolArray]:
    scalar = np.zeros(SCALAR_COUNT, dtype=np.float32)
    mask = np.zeros(SCALAR_COUNT, dtype=np.bool_)
    side = observation.recipient or observation.active_chain_side
    if side not in SIDE_ORDER:
        return scalar, mask
    side_index = SIDE_ORDER.index(side)
    specifications = (
        (observation.prefire_incoming_amount, 0, _normalize_ojama),
        (observation.first_drop_amount, 2, _normalize_first_drop),
        (observation.leftover_after_first_drop, 4, _normalize_ojama),
    )
    for quantity, pair_start, normalizer in specifications:
        _fill_scalar_pair(scalar, mask, quantity, pair_start, side_index, normalizer)
    return scalar, mask


def _fill_scalar_pair(
    scalar: FloatArray,
    mask: BoolArray,
    quantity: IntegerQuantity,
    pair_start: int,
    side_index: int,
    normalizer: Callable[[int], float],
) -> None:
    active_slot = pair_start + side_index
    inactive_slot = pair_start + (1 - side_index)
    mask[inactive_slot] = True
    if not quantity.present:
        return
    scalar[active_slot] = np.float32(normalizer(int(quantity.value)))
    mask[active_slot] = True


def _normalize_ojama(value: int) -> float:
    if value == 0:
        return 0.0
    return 1.0 / (1.0 + OJAMA_NORMALIZER / value)


def _normalize_first_drop(value: int) -> float:
    return value / OJAMA_NORMALIZER


__all__ = [
    "AUXILIARY_FEATURE_NAMES", "AUXILIARY_MANIFEST_PATH",
    "AUXILIARY_MANIFEST_SHA256", "BOARD_CHANNEL_VALUES",
    "ProjectedStateTensorizationError", "ProjectedStateTensorV1",
    "projected_state_tensor_content_digest_v1",
    "tensorize_board_auxiliary_targets_v1",
    "tensorize_projected_state_observation_v1",
]
