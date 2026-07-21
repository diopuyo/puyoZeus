"""Phase H4.2: 6×13 盤面 → CNN embedding モデル定義.

設計方針:
    - 入力: (B, n_channel=8, ROWS=13, COLS=6) one-hot encoded board.
    - BoardCNN: 3 段 Conv (チャンネル拡張) + AdaptiveAvgPool + Linear.
    - SiameseBoardCNN: 1P / 2P 盤面を共通重み BoardCNN で encode し、
      [emb_1p ; emb_2p ; emb_1p - emb_2p] を multi-head に渡す.
        * winrate_head: 1P 勝率 (BCE)
        * indicator_head: 45 indicator 回帰 (MSE) 補助損失.

注意:
    - 1 関数 50 行以内.
    - マジックナンバーは定数化.
    - color インデックスは src.board と整合 (0=空, 1=赤, 2=青, 3=緑, 4=黄,
      5=紫, 9=おじゃま) → 7 値 + UNKNOWN を扱う必要があるため,
      実装上は 0..5 (6 色) + 9 (おじゃま) + UNKNOWN/EMPTY を 8 channel に圧縮する.
      具体的には channel index は次表に従う.
        ch0: COLOR_EMPTY (0)
        ch1: COLOR_RED (1)
        ch2: COLOR_BLUE (2)
        ch3: COLOR_GREEN (3)
        ch4: COLOR_YELLOW (4)
        ch5: COLOR_PURPLE (5)
        ch6: COLOR_OJAMA (9)
        ch7: COLOR_UNKNOWN (10) ※隠し段の不確定セル
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# ============================
# 定数
# ============================
BOARD_ROWS: int = 13
BOARD_COLS: int = 6
N_COLOR_CHANNELS: int = 8  # 上記 ch0..ch7

# Conv channel 基底数 (width_mult=1.0 時の値)
CONV1_CHANNELS: int = 16
CONV2_CHANNELS: int = 32
CONV3_CHANNELS: int = 64
EMBED_DIM: int = 32  # BoardCNN の出力次元

# Siamese head 構造
HEAD_HIDDEN_DIM: int = 64
N_INDICATORS: int = 45  # 補助 head の output 次元 (Phase H4.1 と整合)

# width_mult/dropout のデフォルト (後方互換: 既存動作を変えない)
DEFAULT_WIDTH_MULT: float = 1.0
DEFAULT_DROPOUT: float = 0.0

# board セル値 → channel index 対応 (board.py の COLOR_* と整合)
_COLOR_TO_CHANNEL: dict[int, int] = {
    0: 0,   # EMPTY
    1: 1,   # RED
    2: 2,   # BLUE
    3: 3,   # GREEN
    4: 4,   # YELLOW
    5: 5,   # PURPLE
    9: 6,   # OJAMA
    10: 7,  # UNKNOWN
}


def board_to_onehot(grid: np.ndarray) -> np.ndarray:
    """(ROWS, COLS) int 盤面 → (N_COLOR_CHANNELS, ROWS, COLS) float32 one-hot.

    grid のセル値 (0,1,2,3,4,5,9,10) を _COLOR_TO_CHANNEL でマップしてから
    one-hot 化する. 不明な値は EMPTY (ch0) にフォールバック.
    """
    if grid.shape != (BOARD_ROWS, BOARD_COLS):
        raise ValueError(
            f"grid shape が不正: {grid.shape} (期待 ({BOARD_ROWS},{BOARD_COLS}))"
        )
    out = np.zeros(
        (N_COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32
    )
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            v = int(grid[r, c])
            ch = _COLOR_TO_CHANNEL.get(v, 0)
            out[ch, r, c] = 1.0
    return out


def _scaled_ch(base: int, width_mult: float) -> int:
    """チャンネル数を width_mult でスケールし、最小 1 を保証する。"""
    return max(1, int(base * width_mult))


def _conv_block(in_ch: int, out_ch: int, dropout: float = 0.0) -> nn.Sequential:
    """Conv2d 3×3 padding=1 → BatchNorm → ReLU [→ Dropout2d]。

    dropout=0.0 (既定) なら Dropout2d は追加されない (後方互換)。
    """
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0.0:
        # 空間方向のドロップアウト: チャンネル単位でマスク
        layers.append(nn.Dropout2d(p=dropout))
    return nn.Sequential(*layers)


class BoardCNN(nn.Module):
    """6×13×8 → embed_dim 次元 embedding。

    後方互換: BoardCNN() は既存と同一の重み形状。
    width_mult/dropout を指定した場合のみ形状が変わる。
    """

    def __init__(
        self,
        in_channels: int = N_COLOR_CHANNELS,
        embed_dim: int = EMBED_DIM,
        width_mult: float = DEFAULT_WIDTH_MULT,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        # チャンネル数を width_mult でスケール
        c1 = _scaled_ch(CONV1_CHANNELS, width_mult)
        c2 = _scaled_ch(CONV2_CHANNELS, width_mult)
        c3 = _scaled_ch(CONV3_CHANNELS, width_mult)
        # Conv ブロック (空間 dropout は conv 後)
        self.block1 = _conv_block(in_channels, c1, dropout)
        self.block2 = _conv_block(c1, c2, dropout)
        self.block3 = _conv_block(c2, c3, dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c3, embed_dim)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        """(B, in_channels, ROWS, COLS) → (B, embed_dim)."""
        h = self.block1(board)
        h = self.block2(h)
        h = self.block3(h)
        h = self.pool(h)              # (B, c3, 1, 1)
        h = h.flatten(start_dim=1)    # (B, c3)
        return self.fc(h)             # (B, embed_dim)


def _build_head(
    in_dim: int,
    hidden: int,
    out_dim: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    """中間 1 層の head (Linear → ReLU [→ Dropout] → Linear)。

    dropout=0.0 (既定) なら Dropout は追加されない (後方互換)。
    """
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class SiameseBoardCNN(nn.Module):
    """1P + 2P 盤面の siamese encoding + multi-head (winrate + indicators).

    forward 戻り値: (winrate_logit, indicator_pred, embedding).
        winrate_logit: (B,) BCE 用 logit
        indicator_pred: (B, N_INDICATORS) 補助 MSE 回帰 target
        embedding: (B, embed_dim*3) 結合後 embedding (debug 用)

    後方互換:
        SiameseBoardCNN() — 引数なし生成で既存と同一の重み形状。
        dropout=0.0, width_mult=1.0 がデフォルト (= no-op)。
        新引数を指定した場合は形状が変わるため別ファイルで保存すること。
    """

    def __init__(
        self,
        in_channels: int = N_COLOR_CHANNELS,
        embed_dim: int = EMBED_DIM,
        n_indicators: int = N_INDICATORS,
        head_hidden: int = HEAD_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
        width_mult: float = DEFAULT_WIDTH_MULT,
    ) -> None:
        super().__init__()
        # width_mult を head の hidden 次元にも適用
        scaled_head_hidden = _scaled_ch(head_hidden, width_mult)
        self.cnn = BoardCNN(
            in_channels=in_channels,
            embed_dim=embed_dim,
            width_mult=width_mult,
            dropout=dropout,
        )
        concat_dim = embed_dim * 3  # [emb_1p ; emb_2p ; emb_1p - emb_2p]
        self.winrate_head = _build_head(concat_dim, scaled_head_hidden, 1, dropout)
        self.indicator_head = _build_head(
            concat_dim, scaled_head_hidden, n_indicators, dropout
        )

    def forward(
        self, board_1p: torch.Tensor, board_2p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """1P/2P 盤面ペアから (winrate_logit, indicator_pred, embedding)。"""
        emb_1p = self.cnn(board_1p)
        emb_2p = self.cnn(board_2p)
        h = torch.cat([emb_1p, emb_2p, emb_1p - emb_2p], dim=-1)
        winrate_logit = self.winrate_head(h).squeeze(-1)
        indicator_pred = self.indicator_head(h)
        return winrate_logit, indicator_pred, h
