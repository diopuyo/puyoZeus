"""S-7 TopoFilter のテスト.

KMeans + 多数決による擬似ラベル外れ値除去、24 通り color symmetry 集約、
CellColorFineTuner との chain 統合の検証。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.board import (
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_YELLOW,
)
from src.self_supervised.pseudo_label import (
    COMPONENT_CELL,
    PseudoLabelSample,
)
from src.self_supervised.topo_filter import (
    DEFAULT_MIN_AGREEMENT,
    DEFAULT_N_CLUSTERS,
    cluster_pseudo_labels,
    majority_vote_filter,
    topo_filter_with_color_symmetry,
)


# ============================
# helper
# ============================


_BGR_TABLE: dict[int, tuple[int, int, int]] = {
    COLOR_EMPTY: (10, 10, 10),
    COLOR_RED: (40, 40, 220),
    COLOR_BLUE: (220, 80, 40),
    COLOR_GREEN: (60, 200, 60),
    COLOR_YELLOW: (40, 230, 230),
    COLOR_PURPLE: (180, 60, 180),
}


def _make_patch(
    color: int, size: int = 16, seed: int = 0,
) -> np.ndarray:
    """色ごとに代表 BGR + 微小ノイズの合成 patch."""
    rng = np.random.default_rng(seed=seed + color * 13)
    base = _BGR_TABLE.get(color, (128, 128, 128))
    patch = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        patch[:, :, c] = base[c]
    patch += rng.normal(0, 4.0, patch.shape)
    return np.clip(patch, 0, 255).astype(np.uint8)


def _make_sample(
    label_color: int,
    patch_color: int,
    seed: int = 0,
) -> PseudoLabelSample:
    """patch は patch_color だが label は label_color (誤ラベル混入再現用)."""
    return PseudoLabelSample(
        component=COMPONENT_CELL,
        timestamp=float(seed) * 0.1,
        input_data={
            "patch": _make_patch(patch_color, seed=seed),
            "side": "1P",
            "row": 11,
            "col": 0,
        },
        label=int(label_color),
        confidence=0.9,
        metadata={"frame_idx": int(seed)},
    )


# ============================
# cluster_pseudo_labels
# ============================


def test_cluster_pseudo_labels_basic_grouping() -> None:
    """同色 patch 同士は同 cluster, 異色は別 cluster (粗いがほぼ)."""
    samples: list[PseudoLabelSample] = []
    seed = 0
    for color in (COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW):
        for _ in range(5):
            samples.append(_make_sample(color, color, seed=seed))
            seed += 1
    clusters = cluster_pseudo_labels(samples, n_clusters=4)
    assert len(clusters) >= 2  # 少なくとも複数 cluster に分かれる
    total = sum(len(v) for v in clusters.values())
    assert total == 20


def test_cluster_pseudo_labels_skips_non_cell() -> None:
    """cell 以外は cluster に入らない."""
    samples = [
        PseudoLabelSample(
            component="next", timestamp=0.0,
            input_data={"patch_top": _make_patch(COLOR_RED)},
            label=1, confidence=0.9,
        ),
        _make_sample(COLOR_RED, COLOR_RED, seed=0),
        _make_sample(COLOR_BLUE, COLOR_BLUE, seed=1),
    ]
    clusters = cluster_pseudo_labels(samples, n_clusters=2)
    # 0 (next) は含まれない、1 と 2 のみ
    flat = [i for v in clusters.values() for i in v]
    assert 0 not in flat
    assert set(flat) == {1, 2}


def test_cluster_pseudo_labels_empty() -> None:
    """空 list は空 dict を返す."""
    assert cluster_pseudo_labels([], n_clusters=4) == {}


def test_cluster_pseudo_labels_single_sample() -> None:
    """1 件のみは単一 cluster に入る."""
    samples = [_make_sample(COLOR_RED, COLOR_RED, seed=0)]
    clusters = cluster_pseudo_labels(samples, n_clusters=4)
    assert len(clusters) == 1
    assert sum(len(v) for v in clusters.values()) == 1


# ============================
# majority_vote_filter
# ============================


def test_majority_vote_keeps_unanimous_cluster() -> None:
    """全員一致 cluster は除外なし."""
    samples = [_make_sample(COLOR_RED, COLOR_RED, seed=i) for i in range(5)]
    clusters = {0: [0, 1, 2, 3, 4]}
    filtered, stats = majority_vote_filter(samples, clusters, 0.6)
    assert len(filtered) == 5
    assert stats["n_minority_excluded"] == 0
    assert stats["n_low_agreement_clusters"] == 0


def test_majority_vote_excludes_minority() -> None:
    """80% RED + 20% BLUE label の cluster: 多数派 RED keep, BLUE 除外."""
    samples: list[PseudoLabelSample] = []
    for i in range(8):
        samples.append(_make_sample(COLOR_RED, COLOR_RED, seed=i))
    for i in range(2):
        samples.append(_make_sample(COLOR_BLUE, COLOR_RED, seed=100 + i))
    clusters = {0: list(range(10))}
    filtered, stats = majority_vote_filter(samples, clusters, 0.6)
    assert len(filtered) == 8
    assert stats["n_minority_excluded"] == 2
    assert all(int(s.label) == COLOR_RED for s in filtered)


def test_majority_vote_drops_low_agreement_cluster() -> None:
    """合意率 < min_agreement の cluster は全員除外."""
    samples = [
        _make_sample(COLOR_RED, COLOR_RED, seed=0),
        _make_sample(COLOR_BLUE, COLOR_RED, seed=1),
        _make_sample(COLOR_GREEN, COLOR_RED, seed=2),
        _make_sample(COLOR_YELLOW, COLOR_RED, seed=3),
    ]
    clusters = {0: [0, 1, 2, 3]}
    # 各 label 25% で合意率不足
    filtered, stats = majority_vote_filter(samples, clusters, 0.6)
    assert len(filtered) == 0
    assert stats["n_low_agreement_clusters"] == 1
    assert stats["n_minority_excluded"] == 4


def test_majority_vote_invalid_min_agreement_raises() -> None:
    """min_agreement が範囲外で ValueError."""
    with pytest.raises(ValueError):
        majority_vote_filter([], {}, min_agreement=1.5)


def test_majority_vote_stats_structure() -> None:
    """stats dict が想定 key を持つ."""
    samples = [_make_sample(COLOR_RED, COLOR_RED, seed=i) for i in range(3)]
    clusters = {0: [0, 1, 2]}
    _, stats = majority_vote_filter(samples, clusters, 0.6)
    expected = {
        "n_in", "n_out", "n_minority_excluded",
        "n_low_agreement_clusters", "n_clusters", "per_cluster_stats",
    }
    assert expected.issubset(set(stats.keys()))
    assert isinstance(stats["per_cluster_stats"], list)
    assert stats["per_cluster_stats"][0]["majority_label"] == COLOR_RED


# ============================
# topo_filter_with_color_symmetry
# ============================


def test_topo_filter_with_color_symmetry_keeps_clean_data() -> None:
    """全 sample が patch と label 一致 → ほぼ全件 keep."""
    samples: list[PseudoLabelSample] = []
    seed = 0
    for color in (COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW):
        for _ in range(8):
            samples.append(_make_sample(color, color, seed=seed))
            seed += 1
    filtered, stats = topo_filter_with_color_symmetry(
        samples, n_clusters=4, min_agreement=0.5,
        n_permutations=4,
    )
    # クリーンデータなのでほぼ全件 keep
    assert len(filtered) >= int(len(samples) * 0.8)
    assert stats["n_permutations"] == 4
    assert stats["n_in"] == len(samples)
    assert stats["n_out"] == len(filtered)


def test_topo_filter_with_color_symmetry_removes_outlier() -> None:
    """patch は RED なのに label が GREEN の少数派外れ値が除外される."""
    samples: list[PseudoLabelSample] = []
    seed = 0
    # 各色 8 件、計 32 件のクリーン
    for color in (COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW):
        for _ in range(8):
            samples.append(_make_sample(color, color, seed=seed))
            seed += 1
    # RED patch に GREEN label の混入を 2 件
    outlier_indices: list[int] = []
    for _ in range(2):
        outlier_indices.append(len(samples))
        samples.append(_make_sample(COLOR_GREEN, COLOR_RED, seed=seed))
        seed += 1
    filtered, stats = topo_filter_with_color_symmetry(
        samples, n_clusters=4, min_agreement=0.5,
        n_permutations=6,
    )
    kept_ids = {id(s) for s in filtered}
    # 外れ値 2 件は除外されている
    excluded_outliers = sum(
        1 for i in outlier_indices if id(samples[i]) not in kept_ids
    )
    assert excluded_outliers >= 1
    assert stats["n_out"] < stats["n_in"]


def test_topo_filter_zero_permutations() -> None:
    """n_permutations=0 → 全 keep."""
    samples = [_make_sample(COLOR_RED, COLOR_RED, seed=i) for i in range(5)]
    filtered, stats = topo_filter_with_color_symmetry(
        samples, n_permutations=0,
    )
    assert len(filtered) == len(samples)
    assert stats["n_permutations"] == 0


def test_topo_filter_stats_structure() -> None:
    """stats dict が想定 key を持つ."""
    samples = [
        _make_sample(c, c, seed=i)
        for i, c in enumerate(
            [COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW] * 3,
        )
    ]
    _, stats = topo_filter_with_color_symmetry(
        samples, n_clusters=4, n_permutations=2,
    )
    expected = {
        "n_in", "n_out", "n_permutations",
        "per_permutation_stats", "exclude_vote_threshold",
    }
    assert expected.issubset(set(stats.keys()))
    assert len(stats["per_permutation_stats"]) == 2


def test_topo_filter_color_symmetry_equivalence() -> None:
    """color permutation を施した sample 群でも、

    "全員一致クラスタは除外なし" の性質は保たれる (等価性).
    """
    # クリーンな同色 cluster
    samples = [_make_sample(COLOR_RED, COLOR_RED, seed=i) for i in range(8)]
    samples += [_make_sample(COLOR_BLUE, COLOR_BLUE, seed=8 + i) for i in range(8)]
    filtered, _ = topo_filter_with_color_symmetry(
        samples, n_clusters=2, min_agreement=0.6, n_permutations=8,
    )
    # 全員一致 + 同色集合 → ほぼ全件 keep
    assert len(filtered) >= int(len(samples) * 0.9)


# ============================
# 後方互換: defaults
# ============================


def test_defaults_are_sensible() -> None:
    """デフォルト値が docstring と一致."""
    assert DEFAULT_N_CLUSTERS == 8
    assert 0.0 < DEFAULT_MIN_AGREEMENT < 1.0


# ============================
# CellColorFineTuner との chain 統合
# ============================


def test_finetuner_topo_filter_default_off() -> None:
    """CellColorFineTuner の enable_topo_filter は default off (後方互換)."""
    pytest.importorskip("torch")
    from src.self_supervised.cell_color_fine_tuner import CellColorFineTuner
    tuner = CellColorFineTuner(
        base_model_path="nonexistent.pt",
        output_path="/tmp/cell_topo_off.pt",
    )
    assert tuner._enable_topo_filter is False
    assert tuner._last_topo_stats == {}


def test_finetuner_topo_filter_no_op_on_empty() -> None:
    """空 samples で topo filter を有効にしても fine_tune は no-op."""
    pytest.importorskip("torch")
    from src.self_supervised.cell_color_fine_tuner import CellColorFineTuner
    tuner = CellColorFineTuner(
        base_model_path="nonexistent.pt",
        output_path="/tmp/cell_topo_empty.pt",
        enable_topo_filter=True,
    )
    metrics = tuner.fine_tune([])
    assert metrics["n_samples"] == 0


def test_finetuner_topo_filter_chain_runs(tmp_path) -> None:
    """enable_topo_filter=True で fine_tune が完走し、stats が記録される."""
    pytest.importorskip("torch")
    from src.self_supervised.cell_color_fine_tuner import CellColorFineTuner
    samples: list[PseudoLabelSample] = []
    seed = 0
    for color in (COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW):
        for _ in range(6):
            samples.append(_make_sample(color, color, seed=seed))
            seed += 1
    out = tmp_path / "cell_topo_chain.pt"
    tuner = CellColorFineTuner(
        base_model_path="nonexistent.pt",
        output_path=out,
        epochs=1, batch_size=8,
        enable_topo_filter=True,
        topo_n_clusters=4,
        topo_min_agreement=0.5,
        topo_n_permutations=2,
    )
    metrics = tuner.fine_tune(samples)
    # 大半 keep されているはず
    assert metrics["n_samples"] >= int(len(samples) * 0.8)
    # stats が更新されている
    assert tuner._last_topo_stats.get("filter_applied", 0) == 1
    assert "n_in" in tuner._last_topo_stats
    assert "n_out" in tuner._last_topo_stats


# ============================
# B-2 (cell_augment) との連携検証
# ============================


def test_color_permutation_label_is_consistent() -> None:
    """topo filter 内部で patch と label が同じ permutation に従う."""
    from src.cell_augment import (
        DEFAULT_PERMUTABLE_COLORS,
        apply_color_permutation_to_label,
    )
    cmap = {COLOR_RED: COLOR_BLUE, COLOR_BLUE: COLOR_RED,
            COLOR_GREEN: COLOR_YELLOW, COLOR_YELLOW: COLOR_GREEN}
    new_label = apply_color_permutation_to_label(COLOR_RED, cmap)
    assert new_label == COLOR_BLUE
    # PURPLE / EMPTY / OJAMA 等は不変 (cell_augment 側仕様)
    assert apply_color_permutation_to_label(COLOR_EMPTY, cmap) == COLOR_EMPTY


# ============================
# R-3 (physical_consistency) との chain 動作
# ============================


def test_finetuner_chain_r3_then_s7(tmp_path) -> None:
    """board_lookup_fn + enable_topo_filter で R-3 → S-7 chain が動く."""
    pytest.importorskip("torch")
    from src.board import Board
    from src.self_supervised.cell_color_fine_tuner import CellColorFineTuner
    samples: list[PseudoLabelSample] = []
    seed = 0
    for color in (COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW):
        for _ in range(6):
            samples.append(_make_sample(color, color, seed=seed))
            seed += 1
    # board_lookup_fn は常に空 board (= 全部 EMPTY) を返すと、
    # R-3 の color/gravity/4plus check は pass する (空 board 整合)
    board_lookup = lambda ts, side: Board()
    out = tmp_path / "cell_chain_r3_s7.pt"
    tuner = CellColorFineTuner(
        base_model_path="nonexistent.pt",
        output_path=out,
        epochs=1, batch_size=8,
        board_lookup_fn=board_lookup,
        enable_topo_filter=True,
        topo_n_clusters=4,
        topo_min_agreement=0.5,
        topo_n_permutations=2,
    )
    metrics = tuner.fine_tune(samples)
    # R-3 stats は board lookup により filter_applied=1
    assert tuner._last_filter_stats.get("filter_applied", 0) == 1
    # S-7 stats も filter_applied=1
    assert tuner._last_topo_stats.get("filter_applied", 0) == 1
    # 学習自体は進む
    assert metrics["n_samples"] >= 1
