"""位置保持・左右交換反対称の current-state CNN M0。"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.projected_set_residual_cnn_v1 import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_CHANNELS,
    PositionPreservingBoardEncoderV1,
)


M0_MODEL_VERSION = "advantage-m0-current-cnn/v1"
QUEUE_SLOT_COUNT = 4
QUEUE_CATEGORY_COUNT = 6
QUEUE_EMBED_DIM = 16
SIDE_EMBED_DIM = 32 + QUEUE_EMBED_DIM
PAIR_FEATURE_DIM = SIDE_EMBED_DIM * 3
HEAD_HIDDEN_DIM = 64
COLOR_COUNT = 5
COLOR_INVARIANT_EMBED_DIM = 32
BOARD_VALUE_TO_CATEGORY = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 9: 6, 10: 7}


class AdvantageM0InputError(ValueError):
    """M0入力が固定契約を満たさない。"""


@dataclass(frozen=True, slots=True)
class AdvantageM0OutputV1:
    """M0の反対称logitとraw確率。"""

    logit: torch.Tensor
    raw_probability: torch.Tensor


class AdvantageM0CurrentCNNV1(nn.Module):
    """raw盤面とNEXT/DNEXTを左右共有encoderで評価する。"""

    def __init__(self) -> None:
        super().__init__()
        self.board_encoder = PositionPreservingBoardEncoderV1()
        self.queue_encoder = nn.Sequential(
            nn.Linear(QUEUE_SLOT_COUNT * QUEUE_CATEGORY_COUNT, QUEUE_EMBED_DIM),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(PAIR_FEATURE_DIM, HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HEAD_HIDDEN_DIM, 1),
        )

    def forward(
        self, boards: torch.Tensor, queues: torch.Tensor,
    ) -> AdvantageM0OutputV1:
        """category化済み盤面・queueから反対称確率を返す。"""

        _validate_inputs(boards, queues)
        sides = self._encode_sides(boards, queues)
        direct = self._score_pair(sides[:, 0], sides[:, 1])
        swapped = self._score_pair(sides[:, 1], sides[:, 0])
        logit = 0.5 * (direct - swapped)
        return AdvantageM0OutputV1(logit, torch.sigmoid(logit))

    def _encode_sides(
        self, boards: torch.Tensor, queues: torch.Tensor,
    ) -> torch.Tensor:
        """左右を同じboard/queue encoderへ通す。"""

        batch_size = boards.shape[0]
        board_one_hot = F.one_hot(
            boards.reshape(-1, BOARD_ROWS, BOARD_COLS),
            num_classes=COLOR_CHANNELS,
        ).permute(0, 3, 1, 2).to(torch.float32)
        board_embedding = self.board_encoder(board_one_hot)
        queue_one_hot = F.one_hot(
            queues.reshape(-1, QUEUE_SLOT_COUNT),
            num_classes=QUEUE_CATEGORY_COUNT,
        ).to(torch.float32).flatten(start_dim=1)
        queue_embedding = self.queue_encoder(queue_one_hot)
        combined = torch.cat((board_embedding, queue_embedding), dim=1)
        return combined.reshape(batch_size, 2, SIDE_EMBED_DIM)

    def _score_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """順序付きside pairを単一scalarへ写す。"""

        features = torch.cat((left, right, left - right), dim=1)
        return self.scorer(features).squeeze(1)


class ColorInvariantSideEncoderV1(nn.Module):
    """色IDを交換しても不変な、位置保持盤面+queue encoder。"""

    def __init__(self) -> None:
        super().__init__()
        self.color_features = nn.Sequential(
            nn.Conv2d(1 + QUEUE_SLOT_COUNT, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16), nn.ReLU(),
        )
        self.context_features = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16), nn.ReLU(),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.ReLU(), nn.Flatten(start_dim=1),
            nn.Linear(32 * BOARD_ROWS * BOARD_COLS, COLOR_INVARIANT_EMBED_DIM),
            nn.ReLU(),
        )

    def forward(self, board: torch.Tensor, queue: torch.Tensor) -> torch.Tensor:
        """category盤面とqueueを色集合の順序に依存せずencodeする。"""

        colors = torch.arange(1, COLOR_COUNT + 1, device=board.device)
        color_mask = board[:, None] == colors[None, :, None, None]
        queue_mask = queue[:, None] == colors[None, :, None]
        queue_planes = queue_mask[:, :, :, None, None].expand(
            -1, -1, -1, BOARD_ROWS, BOARD_COLS,
        )
        per_color = torch.cat((color_mask[:, :, None], queue_planes), dim=2)
        flat = per_color.to(torch.float32).flatten(0, 1)
        color_features = self.color_features(flat).reshape(
            board.shape[0], COLOR_COUNT, 16, BOARD_ROWS, BOARD_COLS,
        ).mean(dim=1)
        context = _board_context_channels(board)
        return self.projection(torch.cat((color_features, self.context_features(context)), dim=1))


class AdvantageM0CurrentCNNV2(nn.Module):
    """色置換不変かつ左右交換反対称のcurrent-state CNN。"""

    model_version = "advantage-m0-current-cnn/v2-color-invariant"

    def __init__(self) -> None:
        super().__init__()
        self.side_encoder = ColorInvariantSideEncoderV1()
        self.scorer = nn.Sequential(
            nn.Linear(COLOR_INVARIANT_EMBED_DIM * 3, HEAD_HIDDEN_DIM),
            nn.ReLU(), nn.Linear(HEAD_HIDDEN_DIM, 1),
        )

    def forward(
        self, boards: torch.Tensor, queues: torch.Tensor,
    ) -> AdvantageM0OutputV1:
        _validate_inputs(boards, queues)
        batch_size = boards.shape[0]
        sides = self.side_encoder(
            boards.flatten(0, 1), queues.flatten(0, 1),
        ).reshape(batch_size, 2, COLOR_INVARIANT_EMBED_DIM)
        direct = self._score_pair(sides[:, 0], sides[:, 1])
        swapped = self._score_pair(sides[:, 1], sides[:, 0])
        logit = 0.5 * (direct - swapped)
        return AdvantageM0OutputV1(logit, torch.sigmoid(logit))

    def _score_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        features = torch.cat((left, right, left - right), dim=1)
        return self.scorer(features).squeeze(1)


def _board_context_channels(board: torch.Tensor) -> torch.Tensor:
    empty = board == 0
    known_occupied = (board >= 1) & (board <= 6)
    garbage = board == 6
    unknown = board == 7
    return torch.stack((empty, known_occupied, garbage, unknown), dim=1).to(torch.float32)


def equal_game_weighted_bce(
    output: AdvantageM0OutputV1, label: torch.Tensor, weight: torch.Tensor,
) -> torch.Tensor:
    """各gameの総重みを等しくしたBCEを返す。"""

    if label.shape != output.logit.shape or weight.shape != label.shape:
        raise AdvantageM0InputError("label/weight shapeがlogitと一致しません")
    if not bool(torch.isfinite(weight).all().item()) or not bool((weight > 0).all().item()):
        raise AdvantageM0InputError("sample weightは有限な正値必須です")
    losses = F.binary_cross_entropy_with_logits(output.logit, label, reduction="none")
    return (losses * weight).sum() / weight.sum()


def board_categories_from_raw(grid: np.ndarray) -> np.ndarray:
    """raw 13x6盤面を学習・推論共通categoryへ変換する。"""

    if grid.shape != (BOARD_ROWS, BOARD_COLS):
        raise AdvantageM0InputError("raw boardは13x6必須です")
    result = np.empty(grid.shape, dtype=np.int8)
    for value in np.unique(grid):
        category = BOARD_VALUE_TO_CATEGORY.get(int(value))
        if category is None:
            raise AdvantageM0InputError(f"未登録board値です: {int(value)}")
        result[grid == value] = category
    return result


def queue_categories_from_raw(values: Sequence[int | None]) -> np.ndarray:
    """NEXT/DNEXT 4セルを色1..5、欠測0へ変換する。"""

    if len(values) != QUEUE_SLOT_COUNT:
        raise AdvantageM0InputError("queueは4要素必須です")
    return np.asarray([
        int(value) if value is not None and 1 <= int(value) <= 5 else 0
        for value in values
    ], dtype=np.int8)


def _validate_inputs(boards: torch.Tensor, queues: torch.Tensor) -> None:
    expected_boards = (boards.shape[0], 2, BOARD_ROWS, BOARD_COLS)
    expected_queues = (boards.shape[0], 2, QUEUE_SLOT_COUNT)
    if boards.dtype != torch.int64 or tuple(boards.shape) != expected_boards:
        raise AdvantageM0InputError("boardsはint64[B,2,13,6]必須です")
    if queues.dtype != torch.int64 or tuple(queues.shape) != expected_queues:
        raise AdvantageM0InputError("queuesはint64[B,2,4]必須です")
    if boards.device != queues.device:
        raise AdvantageM0InputError("boardsとqueuesのdeviceが一致しません")
    if boards.numel() and (boards.min().item() < 0 or boards.max().item() >= COLOR_CHANNELS):
        raise AdvantageM0InputError("board categoryが0..7の範囲外です")
    if queues.numel() and (queues.min().item() < 0 or queues.max().item() >= QUEUE_CATEGORY_COUNT):
        raise AdvantageM0InputError("queue categoryが0..5の範囲外です")


__all__ = [
    "AdvantageM0CurrentCNNV1", "AdvantageM0CurrentCNNV2", "AdvantageM0InputError",
    "AdvantageM0OutputV1", "ColorInvariantSideEncoderV1",
    "M0_MODEL_VERSION", "board_categories_from_raw", "equal_game_weighted_bce",
    "queue_categories_from_raw",
]
