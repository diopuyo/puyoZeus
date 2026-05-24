"""scripts.phase_h4_1_train のスモークテスト.

DeepTabularMLP の forward / multi-task loss / gradient flow を中心に検証する。
合成データで fit が動くこと、3 head 全てに勾配が伝播することを確認。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.phase_h4_1_train import (
    BACKBONE_DIMS,
    DeepTabularMLP,
    EMBEDDING_DIM,
    HEAD_HIDDEN_DIM,
    PHASES,
    PHASE_TO_IDX,
    build_feature_subset,
    find_indicator_indices,
    multi_task_loss,
    set_global_seed,
    standardize,
    video_holdout_split,
)


# ============================
# DeepTabularMLP forward テスト
# ============================
def test_deep_tabular_mlp_forward_shapes() -> None:
    """forward の各 head の出力形状が正しい."""
    set_global_seed(0)
    n_feat, n_ind, n_phase = 50, 45, len(PHASES)
    bs = 16
    model = DeepTabularMLP(n_feat, n_ind, n_phase)
    x = torch.randn(bs, n_feat)
    wl, ind, pl, embed = model(x)
    assert wl.shape == (bs,)
    assert ind.shape == (bs, n_ind)
    assert pl.shape == (bs, n_phase)
    assert embed.shape == (bs, EMBEDDING_DIM)
    assert embed.shape[1] == BACKBONE_DIMS[-1]


def test_deep_tabular_mlp_handles_small_input() -> None:
    """入力次元が小さい場合 (top_20 想定) でも forward が成功する."""
    set_global_seed(0)
    model = DeepTabularMLP(20, 45, len(PHASES))
    x = torch.randn(4, 20)
    wl, ind, pl, embed = model(x)
    assert wl.shape == (4,)
    assert ind.shape == (4, 45)
    assert pl.shape == (4, len(PHASES))


# ============================
# multi-task loss テスト
# ============================
def test_multi_task_loss_returns_scalar() -> None:
    """合成 batch で loss が scalar tensor + 内訳 dict を返す."""
    bs = 8
    wl = torch.randn(bs, requires_grad=True)
    y_bin = torch.randint(0, 2, (bs,)).float()
    ind_pred = torch.randn(bs, 45, requires_grad=True)
    ind_true = torch.randn(bs, 45)
    pl = torch.randn(bs, len(PHASES), requires_grad=True)
    pt = torch.randint(0, len(PHASES), (bs,))
    loss, parts = multi_task_loss(wl, y_bin, ind_pred, ind_true, pl, pt)
    assert loss.dim() == 0
    assert all(k in parts for k in ("bce", "mse", "ce"))
    assert all(parts[k] >= 0 for k in ("bce", "mse", "ce"))


def test_multi_task_loss_weights_apply() -> None:
    """α/β/γ=0 にすると個別 loss が消える挙動を確認."""
    bs = 4
    wl = torch.zeros(bs)
    y_bin = torch.zeros(bs)
    ind_pred = torch.randn(bs, 5)
    ind_true = torch.randn(bs, 5)
    pl = torch.zeros(bs, 3)
    pt = torch.zeros(bs, dtype=torch.long)
    # α=0, β=0, γ=0 → total loss は 0
    total, _ = multi_task_loss(wl, y_bin, ind_pred, ind_true, pl, pt,
                                alpha=0.0, beta=0.0, gamma=0.0)
    assert float(total.item()) == pytest.approx(0.0, abs=1e-6)


# ============================
# gradient flow テスト (3 head 全てに勾配が流れる)
# ============================
def test_gradient_flows_to_all_heads() -> None:
    """1 step backward で winrate / indicator / phase head 全ての重みに勾配が立つ."""
    set_global_seed(0)
    n_feat, n_ind, n_phase = 30, 45, len(PHASES)
    model = DeepTabularMLP(n_feat, n_ind, n_phase)
    x = torch.randn(8, n_feat)
    y_bin = torch.randint(0, 2, (8,)).float()
    ind_true = torch.randn(8, n_ind)
    phase_true = torch.randint(0, n_phase, (8,))

    wl, ip, pl, _ = model(x)
    loss, _ = multi_task_loss(wl, y_bin, ip, ind_true, pl, phase_true)
    loss.backward()

    # 各 head の最終 Linear に勾配が立っているか確認
    win_grad = model.winrate_head[-1].weight.grad
    ind_grad = model.indicator_head[-1].weight.grad
    phase_grad = model.phase_head[-1].weight.grad
    assert win_grad is not None and float(win_grad.abs().sum().item()) > 0
    assert ind_grad is not None and float(ind_grad.abs().sum().item()) > 0
    assert phase_grad is not None and float(phase_grad.abs().sum().item()) > 0
    # backbone にも勾配が立つ
    assert model.backbone[0].weight.grad is not None
    assert float(model.backbone[0].weight.grad.abs().sum().item()) > 0


# ============================
# Helper 関数のテスト
# ============================
def test_find_indicator_indices_picks_static_only() -> None:
    """'__static' で終わる列だけが auxiliary 対象となる."""
    cols = [
        "main_chain_maturity__static",
        "main_chain_maturity__delta",
        "shape_score__static",
        "self_main_x_opp_threat",
    ]
    idx = find_indicator_indices(cols)
    assert idx == [0, 2]


def test_build_feature_subset_preserves_csv_order() -> None:
    """指定 feature 名から CSV 出現順に index list を返す."""
    cols = ["a__static", "b__static", "c__static", "d__static"]
    names = ("c__static", "a__static")
    idx = build_feature_subset(cols, names)
    # CSV 順 → a (0), c (2)
    assert idx == [0, 2]


def test_standardize_train_mean_zero_std_one() -> None:
    """train で fit すると train の平均≈0、標準偏差≈1 になる."""
    X_tr = np.random.RandomState(0).randn(100, 5).astype(np.float32) * 3 + 7
    X_te = np.random.RandomState(1).randn(30, 5).astype(np.float32)
    Xt, Xv = standardize(X_tr, X_te)
    assert np.abs(Xt.mean(axis=0)).max() < 1e-3
    assert np.abs(Xt.std(axis=0) - 1.0).max() < 1e-2
    assert Xv.shape == X_te.shape


def test_video_holdout_split_no_leak() -> None:
    """train/test に同じ動画 ID が混ざらない."""
    video_ids = np.array([f"v{i % 6:02d}" for i in range(120)])
    tr, te = video_holdout_split(video_ids, n_test=2, seed=0)
    assert (tr & te).sum() == 0
    assert tr.sum() + te.sum() == len(video_ids)
    train_set = set(video_ids[tr])
    test_set = set(video_ids[te])
    assert train_set.isdisjoint(test_set)


def test_phase_to_idx_covers_all_phases() -> None:
    """5 phase 全てが index にマップされる (0..4)."""
    assert set(PHASE_TO_IDX.values()) == {0, 1, 2, 3, 4}
    assert len(PHASES) == 5


# ============================
# 一括 fit smoke test
# ============================
def test_one_fit_step_runs_end_to_end() -> None:
    """合成データで 1 epoch 分の forward+backward が走る (smoke)."""
    set_global_seed(0)
    n, n_feat, n_ind = 64, 16, 45
    X = np.random.randn(n, n_feat).astype(np.float32)
    y_bin = (X[:, 0] > 0).astype(np.float32)
    ind = np.random.randn(n, n_ind).astype(np.float32)
    phase = np.random.randint(0, len(PHASES), size=n).astype(np.int64)

    model = DeepTabularMLP(n_feat, n_ind, len(PHASES))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    xb = torch.from_numpy(X)
    yb = torch.from_numpy(y_bin)
    ib = torch.from_numpy(ind)
    pb = torch.from_numpy(phase)

    optim.zero_grad()
    wl, ip, pl, _ = model(xb)
    loss, parts = multi_task_loss(wl, yb, ip, ib, pl, pb)
    loss.backward()
    optim.step()
    assert loss.item() >= 0
    assert parts["bce"] >= 0
