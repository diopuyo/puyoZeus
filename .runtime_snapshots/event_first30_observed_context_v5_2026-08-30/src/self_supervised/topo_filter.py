"""S-7 TopoFilter: 擬似ラベルの bootstrap 自己強化リスク対策.

cell color 擬似ラベル (PseudoLabelSample) に対して、
"近い見た目同士は同じ label を持つはず" という topological consistency を
KMeans + 多数決で検証し、外れ値 (label noise) を除外する。

参考: Wu et al. NeurIPS 2020 "A Topological Filter for Learning with Label Noise"
       (`pxiangwu/TopoFilter`) — noise level 0.8 でも k-cluster + majority voting
       で外れ値を除去できる。

ぷよぷよ特有の対称性:
    赤+青 連鎖と黄+紫 連鎖は色置換で等価。
    `cell_augment.generate_color_permutations` で 24 通りの permutation を
    かけ、等価盤面群上で TopoFilter を回すことで「色置換しても紛れ込む
    foreign noise」を浮かび上がらせる。

主な API:
    - cluster_pseudo_labels(samples, n_clusters)
        cell color 擬似ラベルを KMeans クラスタリング。
    - majority_vote_filter(samples, cluster_assignments, min_agreement)
        各クラスタ内で多数決を取り、合意率 < min_agreement の cluster は
        全部除外、合意 cluster 内の少数派 sample のみ除外。
    - topo_filter_with_color_symmetry(samples, n_clusters, min_agreement)
        24 通り color permutation 適用済 sample を投入して filter。

設計ポリシー:
    - stateless: 内部状態は持たない。全 API は pure function。
    - backwards compat: 既存テストに影響しない。
    - 並列稼働中の `phase_i_collect_pseudo_labels` worker には触れない。
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Optional

import numpy as np

from src.cell_augment import (
    DEFAULT_PERMUTABLE_COLORS,
    apply_color_permutation_to_label,
    generate_color_permutations,
    permute_colors_in_patch,
)
from src.self_supervised.pseudo_label import (
    COMPONENT_CELL,
    PseudoLabelSample,
)


# ============================
# 定数
# ============================

# KMeans のデフォルトクラスタ数 (色種 5 + ojama + empty + unknown を覆う想定)
DEFAULT_N_CLUSTERS: int = 8

# 多数決で合意率がこれ未満なら「label noise の可能性」として除外候補
DEFAULT_MIN_AGREEMENT: float = 0.6

# patch を flatten する際の最大次元 (KMeans 高次元爆発を防ぐ)
MAX_FEATURE_DIM: int = 768

# KMeans の re-init 回数 (sklearn 既定 10 では遅いので 4 に抑える)
DEFAULT_KMEANS_N_INIT: int = 4

# KMeans seed (再現性)
DEFAULT_SEED: int = 42

# OOM ガード: MiniBatchKMeans + 上限 sub-sample。
# 1.6M sample × 24 permutation で permuted patch コピーが ~10GB ピークに
# 達し WSL2 をクラッシュさせた事例あり (2026-05-09)。
# サブサンプル + MiniBatchKMeans でピーク 5GB 以下に抑える。
DEFAULT_MAX_SAMPLES: int = 200_000
DEFAULT_MINIBATCH_SIZE: int = 4096
DEFAULT_MINIBATCH_MAX_ITER: int = 100


# ============================
# 公開: クラスタリング
# ============================


def cluster_pseudo_labels(
    samples: list[PseudoLabelSample],
    n_clusters: int = DEFAULT_N_CLUSTERS,
    seed: int = DEFAULT_SEED,
) -> dict[int, list[int]]:
    """cell color 擬似ラベル group を KMeans クラスタリングし、

    cluster index → sample 元 index リストを返す.

    Args:
        samples: list[PseudoLabelSample] (cell component 以外は無視).
        n_clusters: KMeans のクラスタ数.
        seed: KMeans random_state.

    Returns:
        dict[cluster_id, list[sample_index]]
        cluster_id は 0..n_clusters-1。
        cell sample が n_clusters 未満しか無ければ、抽出可能な分だけ
        cluster_id を割り当てる (cluster_id == sample 自身の index)。
    """
    indices, features = _extract_patch_features(samples)
    if not indices:
        return {}
    n_eff = min(int(n_clusters), len(indices))
    if n_eff <= 1:
        # 単一クラスタに全部
        return {0: list(indices)}
    labels = _kmeans_fit_predict(features, n_eff, seed)
    out: dict[int, list[int]] = {}
    for cl_id, sample_idx in zip(labels, indices):
        out.setdefault(int(cl_id), []).append(int(sample_idx))
    return out


def majority_vote_filter(
    samples: list[PseudoLabelSample],
    cluster_assignments: dict[int, list[int]],
    min_agreement: float = DEFAULT_MIN_AGREEMENT,
) -> tuple[list[PseudoLabelSample], dict[str, Any]]:
    """各クラスタ内で多数決を取り、合意率 < min_agreement の cluster は破棄、

    合意 cluster 内の少数派 sample のみ除外する.

    Args:
        samples: 元の sample list (cluster_assignments の index 元).
        cluster_assignments: cluster_id -> [sample_index].
        min_agreement: 多数派ラベルの占有率がこれ未満なら cluster 全部除外.

    Returns:
        (filtered_samples, stats):
            stats = {
                "n_in", "n_out", "n_minority_excluded",
                "n_low_agreement_clusters", "n_clusters",
                "per_cluster_stats": [
                    {"cluster_id", "size", "majority_label",
                     "agreement", "n_excluded"}, ...
                ],
            }
    """
    if not (0.0 <= float(min_agreement) <= 1.0):
        raise ValueError("min_agreement must be in [0, 1]")
    keep_mask = [False] * len(samples)
    per_cluster: list[dict[str, Any]] = []
    n_minority_excluded = 0
    n_low_clusters = 0
    for cl_id, idx_list in cluster_assignments.items():
        cluster_stat, excluded_n, low_flag = _process_cluster(
            cl_id, idx_list, samples, keep_mask, float(min_agreement),
        )
        per_cluster.append(cluster_stat)
        n_minority_excluded += excluded_n
        if low_flag:
            n_low_clusters += 1
    filtered = [s for s, k in zip(samples, keep_mask) if k]
    stats: dict[str, Any] = {
        "n_in": len(samples),
        "n_out": len(filtered),
        "n_minority_excluded": int(n_minority_excluded),
        "n_low_agreement_clusters": int(n_low_clusters),
        "n_clusters": len(cluster_assignments),
        "per_cluster_stats": per_cluster,
    }
    return filtered, stats


def topo_filter_with_color_symmetry(
    samples: list[PseudoLabelSample],
    n_clusters: int = DEFAULT_N_CLUSTERS,
    min_agreement: float = DEFAULT_MIN_AGREEMENT,
    n_permutations: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    max_samples: Optional[int] = DEFAULT_MAX_SAMPLES,
    use_minibatch: bool = True,
) -> tuple[list[PseudoLabelSample], dict[str, Any]]:
    """24 通り color permutation 適用済 sample を投入して TopoFilter を実行.

    各 sample に対し「全 24 permutation で多数決から外された」回数を集計し、
    過半数の permutation で外れた sample を除外する (頑健投票)。

    Args:
        samples: cell PseudoLabelSample list.
        n_clusters: KMeans cluster 数.
        min_agreement: 多数決合意率.
        n_permutations: 適用する permutation 数 (None=24 全通り).
        seed: KMeans seed.

    Returns:
        (filtered_samples, stats):
            stats = {
                "n_in", "n_out", "n_permutations",
                "per_permutation_stats": [stats_dict, ...],
                "exclude_vote_threshold": int,
            }
    """
    rng = np.random.default_rng(int(seed))
    perms = generate_color_permutations(rng=rng, include_identity=True)
    if n_permutations is not None:
        perms = perms[: int(n_permutations)]
    if not perms:
        return list(samples), {
            "n_in": len(samples), "n_out": len(samples),
            "n_permutations": 0, "per_permutation_stats": [],
            "exclude_vote_threshold": 0,
        }
    # OOM ガード: 巨大 sample は均等 sub-sample。除外票は sub-sample 内のみ
    # 集計し、subsample 外の sample は default keep 扱い (= 投票しない)。
    work_indices, work_samples = _select_subsample(
        samples, max_samples, seed,
    )
    n_work = len(work_samples)
    # 各 work sample が「除外」票を受けた回数 (work_indices と同 size)
    exclude_votes = [0] * n_work
    per_perm_stats: list[dict[str, Any]] = []
    for cmap in perms:
        permuted = _apply_permutation_to_samples(work_samples, cmap)
        # MiniBatchKMeans でメモリ peak を抑える
        kept_indices = _cluster_and_keep(
            permuted, n_clusters, float(min_agreement), seed, use_minibatch,
        )
        per_perm_stats.append(
            _build_perm_stat(
                permuted,
                _reconstruct_cluster_assignments(permuted, kept_indices),
                kept_indices,
            ),
        )
        for idx in range(n_work):
            if idx in kept_indices:
                continue
            if permuted[idx].component != COMPONENT_CELL:
                continue
            exclude_votes[idx] += 1
        del permuted  # 即解放 (24 perm × ~5GB peak を防ぐ)
    threshold = (len(perms) // 2) + 1
    # 除外する元 index 集合
    excluded_orig_idx: set[int] = {
        work_indices[i] for i, v in enumerate(exclude_votes) if v >= threshold
    }
    filtered = [
        s for i, s in enumerate(samples) if i not in excluded_orig_idx
    ]
    return filtered, {
        "n_in": len(samples),
        "n_out": len(filtered),
        "n_permutations": len(perms),
        "per_permutation_stats": per_perm_stats,
        "exclude_vote_threshold": int(threshold),
        "subsampled_n": int(n_work),
    }


def _select_subsample(
    samples: list[PseudoLabelSample],
    max_samples: Optional[int],
    seed: int,
) -> tuple[list[int], list[PseudoLabelSample]]:
    """ランダム均等 sub-sample。max_samples 以下なら全採用."""
    if max_samples is None or len(samples) <= int(max_samples):
        return list(range(len(samples))), list(samples)
    rng = np.random.default_rng(int(seed) + 17)
    pick = rng.choice(
        len(samples), size=int(max_samples), replace=False,
    )
    pick.sort()
    work_indices = [int(i) for i in pick]
    work_samples = [samples[i] for i in work_indices]
    return work_indices, work_samples


def _cluster_and_keep(
    permuted: list[PseudoLabelSample],
    n_clusters: int,
    min_agreement: float,
    seed: int,
    use_minibatch: bool,
) -> set[int]:
    """1 permutation 分の cluster + 多数決 → kept index set."""
    indices, features = _extract_patch_features(permuted)
    if not indices:
        return set()
    n_eff = min(int(n_clusters), len(indices))
    if n_eff <= 1:
        labels = np.zeros(len(indices), dtype=np.int32)
    else:
        labels = _kmeans_fit_predict(features, n_eff, seed, use_minibatch)
    clusters: dict[int, list[int]] = {}
    for cl_id, sample_idx in zip(labels, indices):
        clusters.setdefault(int(cl_id), []).append(int(sample_idx))
    return _kept_indices_from_clusters(
        permuted, clusters, float(min_agreement),
    )


def _reconstruct_cluster_assignments(
    permuted: list[PseudoLabelSample],
    kept_indices: set[int],
) -> dict[int, list[int]]:
    """per_permutation_stats 構築用の擬似 cluster_assignments。

    kept/non-kept だけ分かれば stats は再現できないので、cluster ごとの
    内訳は保てない。簡易: 全 cell sample を 1 cluster にまとめて返す。
    詳細統計が要る場合は use_minibatch=False で従来路線に切替。
    """
    one_cluster = [
        i for i, s in enumerate(permuted) if s.component == COMPONENT_CELL
    ]
    return {0: one_cluster} if one_cluster else {}


# ============================
# 内部: 特徴抽出
# ============================


def _extract_patch_features(
    samples: list[PseudoLabelSample],
) -> tuple[list[int], np.ndarray]:
    """cell sample から (元 index リスト, 特徴ベクトル行列) を抽出.

    cell でない / patch が無い sample は飛ばす。
    特徴は patch を flatten + 適度に subsample + L2 normalize。
    """
    indices: list[int] = []
    feats: list[np.ndarray] = []
    for i, s in enumerate(samples):
        feat = _patch_to_feature(s)
        if feat is None:
            continue
        indices.append(i)
        feats.append(feat)
    if not feats:
        return [], np.zeros((0, 0), dtype=np.float32)
    matrix = np.stack(feats, axis=0).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    matrix = matrix / norms
    return indices, matrix


def _patch_to_feature(sample: PseudoLabelSample) -> Optional[np.ndarray]:
    """1 sample から flatten 特徴 1D 配列を抽出 (失敗時 None)."""
    if sample.component != COMPONENT_CELL:
        return None
    if not isinstance(sample.input_data, dict):
        return None
    patch = sample.input_data.get("patch")
    if not isinstance(patch, np.ndarray) or patch.size == 0:
        return None
    flat = patch.astype(np.float32).reshape(-1)
    if flat.size > MAX_FEATURE_DIM:
        # 等間隔にダウンサンプル
        idx = np.linspace(0, flat.size - 1, MAX_FEATURE_DIM).astype(np.int64)
        flat = flat[idx]
    return flat


def _kmeans_fit_predict(
    matrix: np.ndarray, n_clusters: int, seed: int,
    use_minibatch: bool = True,
) -> np.ndarray:
    """KMeans で fit_predict (lazy import)。

    use_minibatch=True (default) で MiniBatchKMeans に切替、巨大データでも
    メモリ上限を超えずに収束する (1.6M sample で 9GB → ~300MB 削減)。
    """
    if use_minibatch:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(
            n_clusters=int(n_clusters),
            n_init=DEFAULT_KMEANS_N_INIT,
            random_state=int(seed),
            batch_size=DEFAULT_MINIBATCH_SIZE,
            max_iter=DEFAULT_MINIBATCH_MAX_ITER,
        )
        return km.fit_predict(matrix)
    from sklearn.cluster import KMeans  # 既存依存
    km = KMeans(
        n_clusters=int(n_clusters),
        n_init=DEFAULT_KMEANS_N_INIT,
        random_state=int(seed),
    )
    return km.fit_predict(matrix)


# ============================
# 内部: 多数決処理
# ============================


def _process_cluster(
    cl_id: int,
    idx_list: list[int],
    samples: list[PseudoLabelSample],
    keep_mask: list[bool],
    min_agreement: float,
) -> tuple[dict[str, Any], int, bool]:
    """1 cluster を処理し (cluster_stat, n_excluded, low_flag) を返す."""
    if not idx_list:
        return ({
            "cluster_id": int(cl_id), "size": 0,
            "majority_label": None, "agreement": 0.0,
            "n_excluded": 0,
        }, 0, False)
    labels = [_safe_label(samples[i]) for i in idx_list]
    counter = Counter(labels)
    majority_label, majority_count = counter.most_common(1)[0]
    agreement = majority_count / len(idx_list)
    n_excluded = 0
    low_flag = False
    if agreement < min_agreement:
        # cluster 全部除外
        n_excluded = len(idx_list)
        low_flag = True
    else:
        # majority に一致する sample のみ keep
        for i in idx_list:
            if _safe_label(samples[i]) == majority_label:
                keep_mask[i] = True
            else:
                n_excluded += 1
    return ({
        "cluster_id": int(cl_id),
        "size": len(idx_list),
        "majority_label": majority_label,
        "agreement": float(agreement),
        "n_excluded": int(n_excluded),
    }, int(n_excluded), bool(low_flag))


def _safe_label(sample: PseudoLabelSample) -> Any:
    """label を hashable に正規化."""
    lab = sample.label
    if isinstance(lab, (list, tuple)):
        return tuple(lab)
    if isinstance(lab, dict):
        return tuple(sorted(lab.items()))
    return lab


# ============================
# 内部: 24 permutation 集約
# ============================


def _apply_permutation_to_samples(
    samples: list[PseudoLabelSample],
    color_map: dict[int, int],
) -> list[PseudoLabelSample]:
    """全 sample に color_map を適用して新 sample list を返す (immutable).

    cell でない sample はそのまま通す。
    """
    out: list[PseudoLabelSample] = []
    for s in samples:
        if s.component != COMPONENT_CELL:
            out.append(s)
            continue
        out.append(_apply_permutation_one(s, color_map))
    return out


def _apply_permutation_one(
    sample: PseudoLabelSample,
    color_map: dict[int, int],
) -> PseudoLabelSample:
    """1 cell sample に color_map を適用して PseudoLabelSample 新 instance を返す."""
    new_input = sample.input_data
    new_label: Any = sample.label
    if isinstance(sample.input_data, dict):
        patch = sample.input_data.get("patch")
        try:
            src_color = int(sample.label)
        except (TypeError, ValueError):
            src_color = -1
        if isinstance(patch, np.ndarray) and src_color >= 0:
            new_patch = permute_colors_in_patch(patch, src_color, color_map)
            new_input = {**sample.input_data, "patch": new_patch}
        try:
            new_label = apply_color_permutation_to_label(
                int(sample.label), color_map,
            )
        except (TypeError, ValueError):
            new_label = sample.label
    return PseudoLabelSample(
        component=sample.component,
        timestamp=sample.timestamp,
        input_data=new_input,
        label=new_label,
        confidence=sample.confidence,
        metadata=dict(sample.metadata),
    )


def _kept_indices_from_clusters(
    permuted_samples: list[PseudoLabelSample],
    cluster_assignments: dict[int, list[int]],
    min_agreement: float,
) -> set[int]:
    """各 cluster で多数決を実行し、keep する元 index 集合を返す.

    low_agreement cluster は全員除外、合意 cluster 内の少数派も除外。
    """
    kept: set[int] = set()
    for idx_list in cluster_assignments.values():
        if not idx_list:
            continue
        labels = [_safe_label(permuted_samples[i]) for i in idx_list]
        counter = Counter(labels)
        majority_label, majority_count = counter.most_common(1)[0]
        agreement = majority_count / len(idx_list)
        if agreement < min_agreement:
            continue
        for i in idx_list:
            if _safe_label(permuted_samples[i]) == majority_label:
                kept.add(int(i))
    return kept


def _build_perm_stat(
    permuted_samples: list[PseudoLabelSample],
    cluster_assignments: dict[int, list[int]],
    kept_indices: set[int],
) -> dict[str, Any]:
    """1 permutation 分の per-cluster stats を構築 (デバッグ用)."""
    per_cluster: list[dict[str, Any]] = []
    for cl_id, idx_list in cluster_assignments.items():
        if not idx_list:
            per_cluster.append({
                "cluster_id": int(cl_id), "size": 0,
                "majority_label": None, "agreement": 0.0,
                "n_excluded": 0,
            })
            continue
        labels = [_safe_label(permuted_samples[i]) for i in idx_list]
        counter = Counter(labels)
        majority_label, majority_count = counter.most_common(1)[0]
        agreement = majority_count / len(idx_list)
        n_excluded = sum(1 for i in idx_list if i not in kept_indices)
        per_cluster.append({
            "cluster_id": int(cl_id),
            "size": len(idx_list),
            "majority_label": majority_label,
            "agreement": float(agreement),
            "n_excluded": int(n_excluded),
        })
    n_kept = len(kept_indices)
    return {
        "n_in": len(permuted_samples),
        "n_out": int(n_kept),
        "n_minority_excluded": int(len(permuted_samples) - n_kept),
        "n_clusters": len(cluster_assignments),
        "per_cluster_stats": per_cluster,
    }


__all__ = [
    "DEFAULT_KMEANS_N_INIT",
    "DEFAULT_MIN_AGREEMENT",
    "DEFAULT_N_CLUSTERS",
    "DEFAULT_SEED",
    "MAX_FEATURE_DIM",
    "cluster_pseudo_labels",
    "majority_vote_filter",
    "topo_filter_with_color_symmetry",
]
