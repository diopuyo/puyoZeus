"""scripts.multicollinearity_analysis のスモークテスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.eda_features import Dataset
from scripts.multicollinearity_analysis import (
    CORR_CLUSTER_THRESHOLD,
    PCA_VAR_THRESHOLD,
    cluster_by_correlation,
    compute_vif,
    effective_dim,
    pca_explained_variance,
    plot_correlation_heatmap,
    serialize_report,
    write_report,
)


def _make_dataset(seed: int = 0) -> Dataset:
    """3 列のうち 2 列が完全相関するダミーデータ。"""
    rng = np.random.default_rng(seed)
    n = 200
    a = rng.standard_normal(n)
    b = a + 0.001 * rng.standard_normal(n)  # 強相関
    c = rng.standard_normal(n)
    X = np.stack([a, b, c], axis=1)
    y = np.where(a + c > 0, 1, -1)
    return Dataset(
        feature_names=("a", "b", "c"),
        X=X, y=y,
        video_ids=["01"] * n,
        time_phases=["midpoint"] * n,
    )


def test_compute_vif_detects_collinearity() -> None:
    """完全相関ペア (a, b) は VIF が大きく出ること。"""
    ds = _make_dataset()
    vif = compute_vif(ds.X)
    assert vif.shape == (3,)
    # a, b が高 VIF (>=20)。c は VIF≈1。
    assert vif[0] > 20.0 or not np.isfinite(vif[0])
    assert vif[1] > 20.0 or not np.isfinite(vif[1])
    assert vif[2] < 5.0


def test_cluster_by_correlation_groups_collinear_pair() -> None:
    """強相関ペアが 1 クラスタにまとまること。"""
    ds = _make_dataset()
    corr = np.corrcoef(ds.X.T)
    clusters = cluster_by_correlation(
        list(ds.feature_names), corr, threshold=CORR_CLUSTER_THRESHOLD,
    )
    assert len(clusters) >= 1
    members = clusters[0].members
    assert "a" in members and "b" in members
    assert clusters[0].avg_abs_corr >= 0.99


def test_pca_explained_variance_sums_to_one() -> None:
    """PCA 寄与率合計は 1。"""
    ds = _make_dataset()
    explained = pca_explained_variance(ds.X)
    assert explained.shape == (3,)
    assert abs(float(explained.sum()) - 1.0) < 1e-6
    eff = effective_dim(explained, PCA_VAR_THRESHOLD)
    # 3 列のうち実質 2 軸 (a≈b) なので 2 で 95% を超える可能性が高い
    assert 1 <= eff <= 3


def test_serialize_and_write_report(tmp_path: Path) -> None:
    """JSON シリアライズと Markdown レポート書き出しが動くこと。"""
    ds = _make_dataset()
    vif = compute_vif(ds.X)
    corr = np.corrcoef(ds.X.T)
    clusters = cluster_by_correlation(list(ds.feature_names), corr)
    explained = pca_explained_variance(ds.X)
    eff = effective_dim(explained, PCA_VAR_THRESHOLD)

    out_json = serialize_report(
        list(ds.feature_names), vif, corr, clusters, explained, eff,
    )
    assert "vif" in out_json
    assert "pca_effective_dim" in out_json
    assert out_json["pca_effective_dim"] == eff

    md_path = tmp_path / "report.md"
    write_report(
        list(ds.feature_names), vif, clusters, explained, eff, corr, md_path,
    )
    assert md_path.exists()
    text = md_path.read_text(encoding="utf-8")
    assert "VIF" in text
    assert "PCA" in text


@pytest.mark.skipif(
    not Path("/usr/bin/dpkg").exists() and not Path("C:/").exists(),
    reason="matplotlib backend non-disponible",
)
def test_plot_heatmap_writes_png(tmp_path: Path) -> None:
    """ヒートマップ PNG が生成されること。"""
    ds = _make_dataset()
    vif = compute_vif(ds.X)
    corr = np.corrcoef(ds.X.T)
    out = tmp_path / "heatmap.png"
    plot_correlation_heatmap(list(ds.feature_names), corr, vif, out)
    assert out.exists() and out.stat().st_size > 0
