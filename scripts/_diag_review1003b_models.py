"""木の経路寄与と学習行の特徴範囲を診断する。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
from scripts._diag_review1003b_extract import OUT, resources, save
from src.exchange_event_features import G_COLUMNS, S3_COLUMNS
from src.exchange_event_count_features import NEW_COLUMNS

TIMES = (4240., 4242., 4244., 4246.)


def node_means(nodes: np.ndarray, idx: int, means: dict[int, float]) -> float:
    """学習サンプル数で重み付けした子孫葉平均を計算する。"""
    node = nodes[idx]
    if node['is_leaf']:
        value = float(node['value'])
    else:
        left, right = int(node['left']), int(node['right'])
        value = sum(node_means(nodes, k, means)*nodes[k]['count']
                    for k in (left, right))/node['count']
    means[idx] = value
    return value


def tree_path(nodes: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray]:
    """Saabas型の経路分解。SHAPではなく、分岐順に依存する局所寄与。"""
    means: dict[int, float] = {}
    baseline = node_means(nodes, 0, means)
    result, idx = np.zeros(len(x)), 0
    while not nodes[idx]['is_leaf']:
        node = nodes[idx]
        feature = int(node['feature_idx'])
        assert not node['is_categorical']
        left = bool(node['missing_go_to_left']) if np.isnan(x[feature]) else x[feature] <= node['num_threshold']
        child = int(node['left'] if left else node['right'])
        result[feature] += means[child]-means[idx]
        idx = child
    return baseline, result


def attribution() -> None:
    """各木の経路を足し、decision_functionとの加法一致を検証する。"""
    model = joblib.load('models/exchange_event_v1_r1_hgb/G_fe.joblib')
    entries = json.loads((OUT / 'gfe_same_input.json').read_text())
    result = []
    for stamp in TIMES:
        source = min(entries, key=lambda r: abs(r['t_sec']-stamp))
        x = np.asarray(source['features'])
        bias, contributions = float(model._baseline_prediction[0, 0]), np.zeros(len(x))
        for predictors in model._predictors:
            base, parts = tree_path(predictors[0].nodes, x)
            bias += base
            contributions += parts
        raw = float(model.decision_function(x[None, :])[0])
        assert np.isclose(bias+contributions.sum(), raw, atol=1e-10)
        result.append(dict(t_sec=source['t_sec'], baseline_logit=bias, logit=raw,
            probability=source['new_p1'], features=[dict(name=name, value=float(x[i]),
            contribution_logit=float(contributions[i])) for i, name in enumerate(G_COLUMNS)]))
    save('gfe_tree_path_contributions.json', result)


def training_support() -> None:
    """S3学習特徴の単変量範囲を取得する。分布外の断定とは区別する。"""
    x = np.load('logs/e15/S3_prime.npy', mmap_mode='r')
    names = S3_COLUMNS+NEW_COLUMNS
    result: dict[str, Any] = dict(rows=len(x), columns=len(names), feature_ranges={})
    for i, name in enumerate(names):
        col = x[:, i]
        finite = col[np.isfinite(col)]
        result['feature_ranges'][name] = dict(n=len(finite), min=float(finite.min()),
            max=float(finite.max()), quantiles=np.quantile(finite, [.01, .5, .99]).tolist())
    save('s3_training_support.json', result)


def main() -> None:
    """リソース検証後にモデル診断する。"""
    resources()
    attribution()
    training_support()
    save('models_resources.json', resources())


if __name__ == '__main__':
    main()
