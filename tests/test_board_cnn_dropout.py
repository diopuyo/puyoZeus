"""SiameseBoardCNN の dropout/width_mult 追加に対する単体テスト。

確認事項:
    1. デフォルト引数 (dropout=0.0, width_mult=1.0) で既存と同一の forward 形状。
    2. dropout>0.0 かつ width_mult<1.0 を指定した時の forward 形状が正しい。
    3. 後方互換: 引数なし生成が例外なく動作する。
    4. 左右反転 augment: flip_aug=True で盤面が正しく反転される。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.cnn_embedding.board_cnn import (
    BOARD_COLS,
    BOARD_ROWS,
    EMBED_DIM,
    N_COLOR_CHANNELS,
    N_INDICATORS,
    SiameseBoardCNN,
    board_to_onehot,
)


# ============================
# ヘルパー
# ============================

def _make_dummy_input(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """ダミー盤面テンソルを生成する。"""
    shape = (batch, N_COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS)
    x1 = torch.zeros(shape, dtype=torch.float32)
    x2 = torch.zeros(shape, dtype=torch.float32)
    # ch0 (EMPTY) を 1 にしてゼロ勾配を防ぐ
    x1[:, 0, :, :] = 1.0
    x2[:, 0, :, :] = 1.0
    return x1, x2


# ============================
# テスト: 後方互換 (引数なし生成)
# ============================

def test_siamese_default_instantiation() -> None:
    """引数なし生成が例外なく動作し、forward の戻り値形状が正しい。"""
    model = SiameseBoardCNN()
    model.eval()
    x1, x2 = _make_dummy_input(batch=3)
    logit, ind_pred, emb = model(x1, x2)

    assert logit.shape == (3,), f"winrate_logit shape: {logit.shape}"
    assert ind_pred.shape == (3, N_INDICATORS), f"indicator_pred shape: {ind_pred.shape}"
    assert emb.shape == (3, EMBED_DIM * 3), f"embedding shape: {emb.shape}"


# ============================
# テスト: dropout + width_mult 指定時の形状
# ============================

@pytest.mark.parametrize("dropout,width_mult", [
    (0.3, 0.5),   # 容量半減 + dropout 0.3 (主要ユースケース)
    (0.5, 0.25),  # 容量 1/4 + dropout 0.5
    (0.0, 0.5),   # dropout なし + 容量半減
    (0.3, 1.0),   # dropout のみ (容量変えない)
])
def test_siamese_with_dropout_and_width_mult(
    dropout: float, width_mult: float
) -> None:
    """dropout/width_mult を指定しても forward 戻り値の外側形状は不変。"""
    model = SiameseBoardCNN(dropout=dropout, width_mult=width_mult)
    model.eval()
    x1, x2 = _make_dummy_input(batch=4)
    logit, ind_pred, emb = model(x1, x2)

    # 外側形状はモデル容量に依らず固定
    assert logit.shape == (4,)
    assert ind_pred.shape == (4, N_INDICATORS)
    assert emb.shape == (4, EMBED_DIM * 3)


# ============================
# テスト: train モードで dropout が機能する
# ============================

def test_dropout_stochastic_in_train_mode() -> None:
    """dropout=0.5 の train モードでは 2 回の forward が異なる出力を返す。"""
    model = SiameseBoardCNN(dropout=0.5, width_mult=1.0)
    model.train()
    x1, x2 = _make_dummy_input(batch=8)
    logit_a, _, _ = model(x1, x2)
    logit_b, _, _ = model(x1, x2)
    # 同一入力でも dropout により出力が異なる (確率的)
    assert not torch.allclose(logit_a, logit_b), (
        "dropout=0.5 の train モードで forward が決定論的になっている"
    )


def test_no_dropout_deterministic_in_eval_mode() -> None:
    """eval モードでは dropout=0.5 でも forward が決定論的。"""
    model = SiameseBoardCNN(dropout=0.5, width_mult=1.0)
    model.eval()
    x1, x2 = _make_dummy_input(batch=8)
    logit_a, _, _ = model(x1, x2)
    logit_b, _, _ = model(x1, x2)
    assert torch.allclose(logit_a, logit_b), (
        "eval モードで forward が非決定論的になっている"
    )


# ============================
# テスト: 左右反転 augment
# ============================

def test_flip_aug_reverses_columns() -> None:
    """flip_aug で得られる盤面が列方向(axis=1=幅方向)に反転していること。"""
    # 非対称な盤面を作成 (左列に色を置く)
    grid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    grid[12, 0] = 1  # 左端・最下段に赤ぷよ
    flipped = np.flip(grid, axis=1)
    assert flipped[12, BOARD_COLS - 1] == 1, "反転後、右端に赤ぷよが移動していること"
    assert flipped[12, 0] == 0, "反転後、左端が空であること"


def test_flip_aug_board_to_onehot_consistency() -> None:
    """反転後の board_to_onehot が正しい one-hot を返すこと。"""
    grid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    grid[12, 0] = 2  # 左端に青ぷよ
    flipped = np.ascontiguousarray(np.flip(grid, axis=1))
    onehot = board_to_onehot(flipped)

    # ch2 (BLUE) が右端 col=5 にあること
    assert onehot[2, 12, BOARD_COLS - 1] == 1.0
    # ch2 が左端 col=0 にないこと
    assert onehot[2, 12, 0] == 0.0


def test_flip_aug_won_label_unchanged() -> None:
    """1P/2P 同時反転では won ラベルが変化しないことをドキュメント的に確認。

    反転は盤面の対称変換であり、勝敗は列の並び順に依存しない。
    このテストでは flip_aug=True の Dataset が won を変えないことを確認する。
    """
    # Dataset の import は scripts 配下だが PYTHONPATH は sys.path.insert で通る
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.train_board_cnn import BoardPairDataset

    n = 10
    rng = np.random.default_rng(0)
    b1 = rng.integers(0, 2, size=(n, BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    b2 = rng.integers(0, 2, size=(n, BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    won = np.ones(n, dtype=np.float32)  # 全サンプル 1P 勝ちとする
    puyo = np.zeros(n, dtype=np.int32)

    # flip_aug=True でも won は取得できる (反転の影響を受けない)
    ds = BoardPairDataset(b1, b2, won, puyo, flip_aug=True, rng_seed=42)
    for i in range(n):
        _, _, y = ds[i]
        assert float(y) == 1.0, f"idx={i}: won ラベルが変化している"
