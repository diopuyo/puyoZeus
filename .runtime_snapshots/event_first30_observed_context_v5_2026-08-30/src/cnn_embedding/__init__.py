"""Phase H4.2: 盤面 → CNN embedding パッケージ.

Phase H4.1 (Deep tabular MLP) の next step として、raw 6×13 盤面を
直接 CNN で読み込む End-to-End 構成を提供する。
"""
from __future__ import annotations

from src.cnn_embedding.board_cnn import (
    BOARD_COLS,
    BOARD_ROWS,
    EMBED_DIM,
    N_COLOR_CHANNELS,
    BoardCNN,
    SiameseBoardCNN,
    board_to_onehot,
)
from src.cnn_embedding.pretrain import (
    AugmentConfig,
    SimCLRBoardEncoder,
    augment_board,
    make_two_views,
    nt_xent_loss,
)

__all__ = [
    "BOARD_COLS",
    "BOARD_ROWS",
    "EMBED_DIM",
    "N_COLOR_CHANNELS",
    "BoardCNN",
    "SiameseBoardCNN",
    "board_to_onehot",
    "AugmentConfig",
    "SimCLRBoardEncoder",
    "augment_board",
    "make_two_views",
    "nt_xent_loss",
]
