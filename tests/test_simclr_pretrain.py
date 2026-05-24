"""SimCLR Board CNN 事前学習モジュールの単体テスト.

検証対象:
    - SimCLRBoardEncoder の forward 出力 shape
    - nt_xent_loss の対称性 / 値域 / 完全一致時の理論最小値
    - augmentation 各処理 (color permute / hflip / mask / shift) の不変性
    - augment_board と make_two_views の shape 維持
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.cnn_embedding.board_cnn import (
    BOARD_COLS,
    BOARD_ROWS,
    N_COLOR_CHANNELS,
    board_to_onehot,
)
from src.cnn_embedding.pretrain import (
    AugmentConfig,
    DEFAULT_PROJECTION_DIM,
    SimCLRBoardEncoder,
    _aug_color_permute,
    _aug_hflip,
    _aug_partial_mask,
    _aug_row_shift,
    augment_board,
    make_two_views,
    nt_xent_loss,
)


# ============================
# fixture
# ============================
@pytest.fixture()
def sample_grid() -> np.ndarray:
    """6 色 + おじゃま + EMPTY が混在する代表盤面 (13×6)."""
    g = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    # 下段に色を配置
    g[12, 0] = 1   # 赤
    g[12, 1] = 2   # 青
    g[12, 2] = 3   # 緑
    g[12, 3] = 4   # 黄
    g[12, 4] = 5   # 紫
    g[12, 5] = 9   # おじゃま
    g[11, 2] = 1
    g[10, 2] = 2
    g[0, 0] = 10   # UNKNOWN (隠し段)
    return g


@pytest.fixture()
def sample_onehot(sample_grid: np.ndarray) -> np.ndarray:
    """sample_grid の one-hot エンコード."""
    return board_to_onehot(sample_grid)


# ============================
# Encoder forward
# ============================
def test_simclr_encoder_forward_shape() -> None:
    """SimCLRBoardEncoder の出力 (h, z) が想定 shape."""
    model = SimCLRBoardEncoder(embed_dim=32, projection_dim=DEFAULT_PROJECTION_DIM)
    x = torch.randn(4, N_COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS)
    h, z = model(x)
    assert h.shape == (4, 32)
    assert z.shape == (4, DEFAULT_PROJECTION_DIM)


def test_simclr_encoder_no_nan() -> None:
    """forward 出力に NaN/Inf が含まれない."""
    model = SimCLRBoardEncoder(embed_dim=16, projection_dim=8)
    x = torch.randn(2, N_COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS)
    h, z = model(x)
    assert torch.isfinite(h).all()
    assert torch.isfinite(z).all()


# ============================
# NT-Xent loss
# ============================
def test_nt_xent_loss_symmetry() -> None:
    """nt_xent_loss(z1,z2) と (z2,z1) は等しい (対称)."""
    torch.manual_seed(0)
    z1 = torch.randn(8, 16)
    z2 = torch.randn(8, 16)
    a = nt_xent_loss(z1, z2, temperature=0.5)
    b = nt_xent_loss(z2, z1, temperature=0.5)
    assert torch.allclose(a, b, atol=1e-5)


def test_nt_xent_loss_positive() -> None:
    """ランダム embedding に対して loss は正の有限値."""
    torch.manual_seed(1)
    z1 = torch.randn(4, 32)
    z2 = torch.randn(4, 32)
    loss = nt_xent_loss(z1, z2, temperature=0.5)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_nt_xent_loss_perfect_alignment_low() -> None:
    """同一 embedding (z1==z2) で他 sample が直交なら loss は理論最小近傍.

    2B=2N 個中 positive と self を除外 → 残り 2N-2 個が distractor.
    z 同士が完全に揃えば loss は -log(1) = 0 ではなく
    -log(exp(s_pos/T) / sum(exp(.../T))) で計算される.
    ここでは「ランダム pairs より低い」ことを確認する.
    """
    torch.manual_seed(2)
    z = torch.randn(8, 16)
    matched = nt_xent_loss(z, z, temperature=0.5)
    z2 = torch.randn(8, 16)
    random_loss = nt_xent_loss(z, z2, temperature=0.5)
    assert matched.item() < random_loss.item()


def test_nt_xent_loss_shape_mismatch() -> None:
    """z1/z2 shape 不一致なら ValueError."""
    z1 = torch.randn(4, 16)
    z2 = torch.randn(5, 16)
    with pytest.raises(ValueError):
        nt_xent_loss(z1, z2)


# ============================
# Augmentation
# ============================
def test_aug_hflip_invariance(sample_onehot: np.ndarray) -> None:
    """二度 hflip すると元に戻る."""
    once = _aug_hflip(sample_onehot)
    twice = _aug_hflip(once)
    np.testing.assert_array_equal(twice, sample_onehot)


def test_aug_color_permute_preserves_count(sample_onehot: np.ndarray) -> None:
    """色 permutation は色 ch 合計の総和を保つ (色枚数は不変)."""
    rng = np.random.default_rng(0)
    out = _aug_color_permute(sample_onehot, rng)
    src_color_sum = sample_onehot[1:6].sum()
    out_color_sum = out[1:6].sum()
    assert pytest.approx(src_color_sum) == out_color_sum
    # EMPTY/おじゃま/UNKNOWN は不変
    np.testing.assert_array_equal(out[0], sample_onehot[0])
    np.testing.assert_array_equal(out[6], sample_onehot[6])
    np.testing.assert_array_equal(out[7], sample_onehot[7])


def test_aug_partial_mask_increases_empty(sample_onehot: np.ndarray) -> None:
    """partial mask 後は EMPTY ch の合計が同じか増える."""
    rng = np.random.default_rng(0)
    out = _aug_partial_mask(sample_onehot, rng)
    assert out[0].sum() >= sample_onehot[0].sum()
    # 全 channel 合計は cell 数に等しい (one-hot 維持)
    assert pytest.approx(out.sum()) == BOARD_ROWS * BOARD_COLS


def test_aug_row_shift_one_hot_preserved(sample_onehot: np.ndarray) -> None:
    """row shift 後も各 cell で 1 channel だけ立つ (one-hot 維持)."""
    rng = np.random.default_rng(7)
    out = _aug_row_shift(sample_onehot, rng)
    cell_sum = out.sum(axis=0)
    assert np.allclose(cell_sum, 1.0)


def test_augment_board_shape(sample_onehot: np.ndarray) -> None:
    """augment_board は shape を維持する."""
    rng = np.random.default_rng(0)
    out = augment_board(sample_onehot, rng)
    assert out.shape == sample_onehot.shape
    assert out.dtype == np.float32


def test_make_two_views_independence(sample_onehot: np.ndarray) -> None:
    """make_two_views は独立な 2 view を返す (確率的に異なる)."""
    rng = np.random.default_rng(0)
    cfg = AugmentConfig(
        p_color_perm=1.0, p_hflip=1.0, p_mask=1.0, p_shift=1.0,
    )
    v1, v2 = make_two_views(sample_onehot, rng, cfg)
    assert v1.shape == sample_onehot.shape
    assert v2.shape == sample_onehot.shape
    # 全 augmentation を必ず適用すれば 2 view は元盤面と別である可能性が高い
    diff = np.abs(v1 - v2).sum()
    assert diff > 0.0
