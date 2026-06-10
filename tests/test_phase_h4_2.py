"""Phase H4.2 (Raw Board CNN End-to-End) のスモークテスト.

検証項目:
    A. BoardCNN forward shape
    B. SiameseBoardCNN forward shape
    C. board_to_onehot 正確性 (色 → channel index)
    D. Multi-task loss + gradient flow
    E. boards_to_onehot_batch 一括処理
    F. collect script (phase_h2_collect_board) の board NPZ 保存形式

実行: python -m pytest tests/test_phase_h4_2.py -v
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
    BoardCNN,
    SiameseBoardCNN,
    board_to_onehot,
)


# ============================
# board_to_onehot
# ============================
def test_board_to_onehot_shape() -> None:
    """one-hot tensor が (N_COLOR_CHANNELS, ROWS, COLS) になる."""
    grid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.uint8)
    out = board_to_onehot(grid)
    assert out.shape == (N_COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS)
    assert out.dtype == np.float32


def test_board_to_onehot_empty_all_zero_channel() -> None:
    """全 EMPTY の盤面は ch0 が全 1.0、他 channel は全 0.0."""
    grid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.uint8)
    oh = board_to_onehot(grid)
    assert np.allclose(oh[0], 1.0)
    assert np.allclose(oh[1:], 0.0)


def test_board_to_onehot_color_mapping() -> None:
    """各色値が想定 channel に one-hot 化される.

    マッピング: 0→ch0, 1→ch1, 2→ch2, 3→ch3, 4→ch4, 5→ch5, 9→ch6, 10→ch7.
    """
    grid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.uint8)
    grid[0, 0] = 1   # red
    grid[0, 1] = 2   # blue
    grid[0, 2] = 3   # green
    grid[0, 3] = 4   # yellow
    grid[0, 4] = 5   # purple
    grid[0, 5] = 9   # ojama
    grid[1, 0] = 10  # unknown
    oh = board_to_onehot(grid)
    assert oh[1, 0, 0] == 1.0
    assert oh[2, 0, 1] == 1.0
    assert oh[3, 0, 2] == 1.0
    assert oh[4, 0, 3] == 1.0
    assert oh[5, 0, 4] == 1.0
    assert oh[6, 0, 5] == 1.0
    assert oh[7, 1, 0] == 1.0
    # それぞれのセルは exactly 1 channel が 1.0
    assert oh.sum() == BOARD_ROWS * BOARD_COLS  # 全セル合計 = ROWS*COLS


def test_board_to_onehot_invalid_shape_raises() -> None:
    """形状不正なら ValueError."""
    bad = np.zeros((10, 6), dtype=np.uint8)
    with pytest.raises(ValueError):
        board_to_onehot(bad)


# ============================
# BoardCNN
# ============================
def test_board_cnn_forward_shape() -> None:
    """BoardCNN 出力が (B, EMBED_DIM)."""
    model = BoardCNN()
    bs = 4
    x = torch.randn(bs, N_COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS)
    out = model(x)
    assert out.shape == (bs, EMBED_DIM)


def test_board_cnn_handles_single_sample() -> None:
    """B=1 でも問題なく動く."""
    model = BoardCNN()
    model.eval()
    x = torch.zeros(1, N_COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS)
    out = model(x)
    assert out.shape == (1, EMBED_DIM)


def test_board_cnn_custom_embed_dim() -> None:
    """embed_dim を変更しても出力次元が反映される."""
    model = BoardCNN(embed_dim=16)
    x = torch.randn(2, N_COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS)
    out = model(x)
    assert out.shape == (2, 16)


# ============================
# SiameseBoardCNN
# ============================
def test_siamese_forward_shape() -> None:
    """winrate_logit, indicator_pred, embedding の shape を検証."""
    n_ind = 45
    model = SiameseBoardCNN(n_indicators=n_ind)
    bs = 8
    b1 = torch.randn(bs, N_COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS)
    b2 = torch.randn(bs, N_COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS)
    wl, ip, emb = model(b1, b2)
    assert wl.shape == (bs,)
    assert ip.shape == (bs, n_ind)
    # 結合 embedding は emb_1p + emb_2p + diff = 3*EMBED_DIM
    assert emb.shape == (bs, EMBED_DIM * 3)


def test_siamese_shared_weights() -> None:
    """siamese は同一 BoardCNN を 2 回呼ぶ → 同じ盤面なら emb 一致."""
    model = SiameseBoardCNN()
    model.eval()
    b1 = torch.randn(2, N_COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS)
    # 同じ盤面を 1P/2P に渡せば diff embedding = 0.
    with torch.no_grad():
        _, _, emb = model(b1, b1)
    diff_part = emb[:, EMBED_DIM * 2:]  # 最後の 1/3 が emb_1p - emb_2p
    assert torch.allclose(diff_part, torch.zeros_like(diff_part), atol=1e-6)


def test_siamese_gradient_flows_to_both_heads() -> None:
    """winrate / indicator 両 head に勾配が立つ."""
    n_ind = 45
    model = SiameseBoardCNN(n_indicators=n_ind)
    bs = 4
    b1 = torch.randn(bs, N_COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS)
    b2 = torch.randn(bs, N_COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS)
    y = torch.randint(0, 2, (bs,)).float()
    ind_t = torch.randn(bs, n_ind)
    wl, ip, _ = model(b1, b2)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(wl, y)
    mse = torch.nn.functional.mse_loss(ip, ind_t)
    (bce + 0.3 * mse).backward()
    win_grad = model.winrate_head[-1].weight.grad
    ind_grad = model.indicator_head[-1].weight.grad
    assert win_grad is not None and float(win_grad.abs().sum().item()) > 0
    assert ind_grad is not None and float(ind_grad.abs().sum().item()) > 0
    # 共通 BoardCNN の最初の Conv にも勾配が伝わる
    first_conv_w = model.cnn.block1[0].weight
    assert first_conv_w.grad is not None
    assert float(first_conv_w.grad.abs().sum().item()) > 0


# ============================
# scripts.phase_h4_2_train ユーティリティ
# ============================
def test_boards_to_onehot_batch_shape() -> None:
    """(N, ROWS, COLS) → (N, N_COLOR_CHANNELS, ROWS, COLS)."""
    from scripts.old.phase_h4_2_train import boards_to_onehot_batch
    n = 5
    boards = np.zeros((n, BOARD_ROWS, BOARD_COLS), dtype=np.uint8)
    boards[0, 12, 0] = 1  # 1 セル赤
    out = boards_to_onehot_batch(boards)
    assert out.shape == (n, N_COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS)
    assert out[0, 1, 12, 0] == 1.0
    assert out[0, 0, 12, 0] == 0.0


def test_video_holdout_split_no_leak() -> None:
    """train/test に同一動画 ID が混ざらない."""
    from scripts.old.phase_h4_2_train import video_holdout_split
    video_ids = np.array([f"v{i % 8:02d}" for i in range(160)])
    tr, te = video_holdout_split(video_ids, n_test=3, seed=0)
    assert (tr & te).sum() == 0
    train_set = set(video_ids[tr])
    test_set = set(video_ids[te])
    assert train_set.isdisjoint(test_set)
    assert len(test_set) == 3


def test_multi_task_loss_returns_scalar() -> None:
    """合成 batch で loss が scalar tensor + 内訳 dict."""
    from scripts.old.phase_h4_2_train import multi_task_loss
    bs = 6
    wl = torch.randn(bs, requires_grad=True)
    yb = torch.randint(0, 2, (bs,)).float()
    ip = torch.randn(bs, 45, requires_grad=True)
    it = torch.randn(bs, 45)
    total, parts = multi_task_loss(wl, yb, ip, it)
    assert total.dim() == 0
    assert all(k in parts for k in ("bce", "mse"))
    assert all(parts[k] >= 0 for k in ("bce", "mse"))


def test_multi_task_loss_weights_apply_zero() -> None:
    """α=β=0 で total loss = 0."""
    from scripts.old.phase_h4_2_train import multi_task_loss
    bs = 4
    wl = torch.zeros(bs)
    yb = torch.zeros(bs)
    ip = torch.randn(bs, 5)
    it = torch.randn(bs, 5)
    total, _ = multi_task_loss(wl, yb, ip, it, alpha=0.0, beta=0.0)
    assert float(total.item()) == pytest.approx(0.0, abs=1e-6)


def test_standardize_train_mean_zero_std_one() -> None:
    """train で fit すると train の平均≈0、標準偏差≈1."""
    from scripts.old.phase_h4_2_train import standardize
    rng = np.random.RandomState(0)
    X_tr = rng.randn(80, 5).astype(np.float32) * 3 + 7
    X_te = rng.randn(20, 5).astype(np.float32)
    Xt, Xv = standardize(X_tr, X_te)
    assert np.abs(Xt.mean(axis=0)).max() < 1e-3
    assert np.abs(Xt.std(axis=0) - 1.0).max() < 1e-2
    assert Xv.shape == X_te.shape


# ============================
# Collect script の NPZ 保存形式
# ============================
def test_save_board_npz_roundtrip(tmp_path) -> None:
    """save_board_npz で書いた NPZ が想定 key を含み、形状が正しい."""
    from scripts.old.phase_h2_collect_board import save_board_npz, BOARD_GRID_ROWS, BOARD_GRID_COLS
    rows = [
        {"video_id": "01", "match_idx": 1, "frame_idx": 0,
         "timestamp": 1.0, "label": 1},
        {"video_id": "01", "match_idx": 1, "frame_idx": 1,
         "timestamp": 1.6, "label": 1},
    ]
    p1 = [np.zeros((BOARD_GRID_ROWS, BOARD_GRID_COLS), dtype=np.uint8)] * 2
    p2 = [np.ones((BOARD_GRID_ROWS, BOARD_GRID_COLS), dtype=np.uint8)] * 2
    out = tmp_path / "v01.npz"
    save_board_npz(out, rows, p1, p2)
    assert out.exists()
    data = np.load(out, allow_pickle=False)
    keys = set(data.files)
    assert {
        "video_ids", "match_indices", "frame_indices",
        "timestamps", "labels", "p1_boards", "p2_boards",
    }.issubset(keys)
    assert data["p1_boards"].shape == (2, BOARD_GRID_ROWS, BOARD_GRID_COLS)
    assert data["p2_boards"].shape == (2, BOARD_GRID_ROWS, BOARD_GRID_COLS)
    assert data["labels"].tolist() == [1, 1]


def test_grid_or_zero_handles_none() -> None:
    """_grid_or_zero(None) は zero ndarray を返す."""
    from scripts.old.phase_h2_collect_board import _grid_or_zero, BOARD_GRID_ROWS, BOARD_GRID_COLS
    out = _grid_or_zero(None)
    assert out.shape == (BOARD_GRID_ROWS, BOARD_GRID_COLS)
    assert out.dtype == np.uint8
    assert out.sum() == 0
