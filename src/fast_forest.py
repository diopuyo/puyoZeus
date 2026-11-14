"""学習済みの決定木 (sklearn) を、同じ答えのまま速く評価する。

画像なし方式の判定器は決定木 48〜64 本の森で、sklearn は木 1 本ごとに Python の手続き (joblib・設定の
出し入れ) を踏むため、1 コマに千回以上呼ぶと手続きだけで重くなる (11/14 実測: 認識 1 コマ 27.6 → 89.8ms)。
ここでは全部の木の節を 1 つの配列に詰め、全部の木を同時にたどる。答えは sklearn と一致させる:
- 特徴は sklearn と同じく float32 に丸めてから、閾値 (float64) と「以下なら左」で比べる。
- 分類の森は、木ごとの確率 (葉の値を合計で割る、合計 0 なら 1) を木の順に足して木の数で割り、最大の類を取る。
- 回帰の森は、木ごとの葉の値を木の順に足して木の数で割る。
- 欠損値 (NaN) が入った時と、想定外の形 (多出力の分類など) は sklearn にそのまま任せる。
同じ特徴を受ける複数の森 (数字の種類と確信度など) は predict_together で 1 回のたどりにまとめる
(木ごとの葉はその木だけで決まるので、別々に評価した答えと同じ)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

LEAF = -1                      # sklearn の葉の印 (children_left == -1)
LEAF_THRESHOLD = np.inf        # 葉は「以下なら左 = 自分」に進ませて止める
FEATURE_DTYPE = np.float32     # sklearn の木は特徴を float32 で受ける
LEAF_CHECK_EVERY = 4           # 全部が葉に着いたかの確かめを何段おきにするか (葉は自分に戻るので遅れても答えは同じ)
ESTIMATOR_ATTRS = ('classifier', 'coordinates', 'labels', 'scores')   # 画像なし方式のモデルの持ち方
ENABLE_ENV = 'PUYO_IMAGE_FREE_FAST_TREES'   # '0' で sklearn のまま (切り分け用。既定は速い版、答えは同じ)


@dataclass(frozen=True)
class _Packed:
    """全部の木の節を 1 つの配列に詰めたもの。"""

    child: np.ndarray          # (節の総数, 2) [右, 左] 「以下なら左」の真偽で引く。葉は自分
    is_leaf: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray          # (節の総数, 出力の数)。分類は木ごとの確率 (葉の値 ÷ 合計) を先に計算しておく
    roots: np.ndarray          # 木ごとの根の位置
    depth: int


def _pack(trees: list[Any], is_classifier: bool) -> _Packed:
    """木 (tree_) の並びを、葉が自分に戻る形で 1 つの配列にする。
    葉の値は、分類なら (節, 類) = value[:, 0, :]、回帰なら (節, 出力) = value[:, :, 0]。"""
    lefts, rights, features, thresholds, values, roots = [], [], [], [], [], []
    offset = 0
    for tree in trees:
        count = tree.node_count
        index = np.arange(count) + offset
        leaf = tree.children_left == LEAF
        lefts.append(np.where(leaf, index, tree.children_left + offset))
        rights.append(np.where(leaf, index, tree.children_right + offset))
        features.append(np.where(leaf, 0, tree.feature))
        thresholds.append(np.where(leaf, LEAF_THRESHOLD, tree.threshold))
        values.append(tree.value[:, 0, :] if is_classifier else tree.value[:, :, 0])
        roots.append(offset)
        offset += count
    left, right, value = np.concatenate(lefts), np.concatenate(rights), np.concatenate(values)
    return _Packed(np.stack([right, left], axis=1), left == np.arange(len(left)), np.concatenate(features),
                   np.concatenate(thresholds), _tree_proba(value) if is_classifier else value,
                   np.array(roots), max(t.max_depth for t in trees))


def _leaves(packed: _Packed, data: np.ndarray) -> np.ndarray:
    """(標本, 木) ごとの葉の位置。全部が葉に着いたら打ち切る。"""
    node = np.broadcast_to(packed.roots, (data.shape[0], len(packed.roots))).copy()
    rows = np.arange(data.shape[0])[:, None]
    for level in range(packed.depth):
        go_left = data[rows, packed.feature[node]] <= packed.threshold[node]
        node = packed.child[node, go_left.view(np.int8)]
        if level % LEAF_CHECK_EVERY == LEAF_CHECK_EVERY - 1 and packed.is_leaf[node].all():
            break
    return node


class FastTreeModel:
    """sklearn の決定木・森 (単一出力の分類、回帰) の predict を同じ答えで速くしたもの。"""

    def __init__(self, estimator: Any) -> None:
        self.original = estimator
        self.is_classifier = hasattr(estimator, 'classes_')
        members = getattr(estimator, 'estimators_', None)
        self.trees = [m.tree_ for m in members] if members is not None else [estimator.tree_]
        self.packed = _pack(self.trees, self.is_classifier)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)       # predict 以外は元の推定器のまま

    def predict(self, data: Any) -> np.ndarray:
        array = _features(data)
        if array is None:
            return self.original.predict(data)
        return self._finish(_leaves(self.packed, array))

    def _finish(self, leaves: np.ndarray) -> np.ndarray:
        """(標本, 木) の葉の位置から答えを作る。"""
        by_tree = self.packed.value[leaves.T]                       # (木, 標本, 出力)
        # sklearn と同じく木の順に足す (accumulate は一つ前までの和に次の木を足す順で決まる)
        total = np.add.accumulate(by_tree, axis=0)[-1] / by_tree.shape[0]
        if self.is_classifier:
            return self.original.classes_.take(np.argmax(total, axis=1), axis=0)
        return total[:, 0] if total.shape[1] == 1 else total


def _features(data: Any) -> np.ndarray | None:
    """sklearn と同じく float32 に丸めた特徴。速い版で扱えない入力 (NaN・2 次元でない) は None。"""
    array = np.asarray(data, dtype=FEATURE_DTYPE)
    if array.ndim != 2 or np.isnan(array).any():
        return None
    return array


@dataclass(frozen=True)
class _Joint:
    """複数の森の木を 1 つにつないだもの。spans は森ごとの (木の始め, 終わり, 節の位置のずれ)。"""

    members: tuple[FastTreeModel, ...]
    packed: _Packed
    spans: tuple[tuple[int, int, int], ...]


def _join(models: tuple[FastTreeModel, ...]) -> _Joint:
    children, leaves, features, thresholds, roots, spans = [], [], [], [], [], []
    nodes = trees = 0
    for model in models:
        packed = model.packed
        children.append(packed.child + nodes)
        leaves.append(packed.is_leaf)
        features.append(packed.feature)
        thresholds.append(packed.threshold)
        roots.append(packed.roots + nodes)
        spans.append((trees, trees + len(packed.roots), nodes))
        nodes += len(packed.feature)
        trees += len(packed.roots)
    packed = _Packed(np.concatenate(children), np.concatenate(leaves), np.concatenate(features),
                     np.concatenate(thresholds), np.empty((0, 0)), np.concatenate(roots),
                     max(model.packed.depth for model in models))
    return _Joint(models, packed, tuple(spans))


_JOINTS: dict[tuple[int, ...], _Joint] = {}


def predict_together(models: list[Any], data: Any) -> list[np.ndarray]:
    """同じ特徴を受ける複数の推定器の predict を、まとめて 1 回のたどりで返す (答えは別々の predict と同じ)。"""
    array = _features(data) if all(isinstance(m, FastTreeModel) for m in models) else None
    if array is None:
        return [model.predict(data) for model in models]
    key = tuple(id(model) for model in models)
    joint = _JOINTS.get(key)
    if joint is None or any(a is not b for a, b in zip(joint.members, models)):
        joint = _JOINTS[key] = _join(tuple(models))
    leaves = _leaves(joint.packed, array)
    return [model._finish(leaves[:, start:end] - offset)
            for model, (start, end, offset) in zip(models, joint.spans)]


def _tree_proba(value: np.ndarray) -> np.ndarray:
    """葉ごとの確率 (sklearn の DecisionTreeClassifier.predict_proba と同じ正規化。葉の値だけで決まる)。"""
    normalizer = value.sum(axis=1)[:, np.newaxis]
    normalizer[normalizer == 0.0] = 1.0
    return value / normalizer


def supported(estimator: Any) -> bool:
    """差し替えてよい推定器か (単一出力の決定木・森だけ)。"""
    trees = getattr(estimator, 'estimators_', None)
    first = trees[0] if trees else estimator
    if not hasattr(first, 'tree_'):
        return False
    if hasattr(estimator, 'classes_') and getattr(estimator, 'n_outputs_', 1) != 1:
        return False                              # 多出力の分類は sklearn に任せる
    return getattr(estimator, 'n_jobs', None) in (None, 1)


def accelerate(model: Any) -> Any:
    """読み込んだモデル (dict・画像なし方式の入れ物) の中の決定木を速い版に差し替えて返す。"""
    import os
    if os.environ.get(ENABLE_ENV, '1') == '0':
        return model
    if isinstance(model, dict):
        for key, value in list(model.items()):
            model[key] = _swap(value)
        return model
    for name in ESTIMATOR_ATTRS:
        if getattr(model, name, None) is not None and name in vars(model):
            setattr(model, name, _swap(getattr(model, name)))
    return model


def _swap(value: Any) -> Any:
    if supported(value):
        return FastTreeModel(value)
    if isinstance(value, dict) or any(name in vars(value) for name in ESTIMATOR_ATTRS
                                      if hasattr(value, '__dict__')):
        return accelerate(value)
    return value
