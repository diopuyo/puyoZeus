"""v4.9e: 固定B3評価の出力不変の高速化 (特徴量の複製の軽量化・まとめ推論・結果の再利用)。

v4.9d は候補盤1つごとに GameState 全体を deepcopy し (試合が長いほど台帳が肥大して遅い)、1行ずつ推論していた。
ここでは次の3つを入れる。いずれも固定B3の出力 (確率) を変えない。

1. 特徴量づくりの複製を、書き換える部分 (両側の SideState と台帳スナップショット) だけの浅い複製にする。
2. 1更新内の複数の候補盤を、同じ mode の行ごとに1回の `predict_proba` にまとめる。
3. 特徴量のバイト列 (+モデルSHA+コードSHA) をキーに確率を再利用する。

抜き取りの実計算照合: 一定間隔で従来経路 (deepcopy・1行推論) を実行して結果を比べ、不一致を件数で記録する。
不一致があれば再利用を捨てて従来経路の値を使う。照合に使った時間は計測時間から差し引く。
"""
from __future__ import annotations

from collections import OrderedDict
from copy import copy, deepcopy
from hashlib import sha256
from time import perf_counter
from typing import Any, Callable
import numpy as np
from src.b3_features import build_features
from src.exchange_native_chain import native_chains
from src.prefire_post_b3 import forecast_features, ROWS, COLS

# 抜き取り照合の間隔 (呼出し・命中の通し番号で決める。乱数は使わない)。
FEATURE_AUDIT_EVERY = 64
HIT_AUDIT_EVERY = 32
BATCH_AUDIT_EVERY = 32
# 確率の再利用表の上限 (超えたら古い順に捨てる)。
PROBABILITY_CACHE_LIMIT = 8192
CODE_FILES = ('src/prefire_b3_v49e.py', 'src/b3_features.py', 'src/b3_predict.py', 'src/b3_timeaxis.py',
              'src/b3_search.py', 'src/b3_state.py', 'src/b3_timing.py', 'src/b3_static.py')
SKIPPED_TYPES = (type(len), type(lambda: 0), type, type(np))


def forecast_features_light(current: Any, boards: list[list[int]], pending: list[float],
                            table: Any, native: Any) -> dict:
    """`forecast_features` と同じ特徴量。deepcopy の代わりに、書き換える属性を持つ物だけを浅く複製する。"""
    state = copy(current)
    state.sides = [copy(side) for side in current.sides]
    state.sl_snap = dict(current.sl_snap)
    for i, side in enumerate(state.sides):
        side.board = np.asarray(boards[i], dtype=np.int8).reshape(ROWS, COLS)
        side.board_t, side.state, side.episode = state.t, 'STABLE', None
        state.sl_snap[f'pending_p{i+1}'] = pending[i]
        state.sl_snap[f'pending_confirmed_p{i+1}'] = pending[i]
    with native_chains(native):
        return build_features(state, table)


def fingerprint(value: Any) -> str:
    """オブジェクトの中身の指紋 (属性・コンテナ・配列を再帰的にたどる)。特徴量づくりが元の状態を書き換えていないかの検査用。"""
    digest, seen = sha256(), set()

    def walk(item: Any) -> None:
        if isinstance(item, np.ndarray):
            digest.update(b'A'+str(item.dtype).encode()+str(item.shape).encode()+item.tobytes())
        elif item is None or isinstance(item, (bool, int, float, str, bytes, np.generic)):
            digest.update(b'S'+repr(item).encode())
        elif isinstance(item, SKIPPED_TYPES) or callable(item):
            digest.update(b'F')
        elif id(item) in seen:
            digest.update(b'R')
        else:
            seen.add(id(item))
            visit(item)

    def visit(item: Any) -> None:
        digest.update(b'O'+type(item).__name__.encode())
        if isinstance(item, dict):
            for key in sorted(item, key=repr):
                walk(key)
                walk(item[key])
        elif isinstance(item, (list, tuple, set, frozenset)) or hasattr(item, 'popleft'):
            members = sorted(item, key=repr) if isinstance(item, (set, frozenset)) else item
            for member in members:
                walk(member)
        else:
            for key in sorted(getattr(item, '__dict__', {})):
                digest.update(key.encode())
                walk(item.__dict__[key])
            for klass in type(item).__mro__:
                for key in getattr(klass, '__slots__', ()):
                    if hasattr(item, key):
                        walk(getattr(item, key))

    walk(value)
    return digest.hexdigest()


def same_features(a: dict, b: dict) -> bool:
    """特徴量の辞書が完全に一致するか (欠測 NaN は同じ欠測として一致)。"""
    if a.keys() != b.keys():
        return False
    for key, left in a.items():
        right = b[key]
        if left != right and not (isinstance(left, float) and isinstance(right, float)
                                  and np.isnan(left) and np.isnan(right)):
            return False
    return True


def predict_rows(predictor: Any, rows: list[dict]) -> list[float]:
    """`RowPredictor.predict` を行ごとに呼んだ結果と同じ値を、mode ごとの1回の推論でまとめて返す。"""
    from src.b3_train_model import MODE_COLUMN, THREADS, hgb_threads
    out: list[float] = [float('nan')]*len(rows)
    groups: dict[int, list[tuple[int, np.ndarray]]] = {}
    for index, row in enumerate(rows):
        mode = float(row[MODE_COLUMN])
        if np.isnan(mode):
            raise ValueError('mode_exchange に欠測 (状態を選べない)')
        if mode not in (0.0, 1.0):
            continue
        if predictor.evaluators[int(mode)] is None:
            raise ValueError(f'mode={int(mode)} の行があるが評価器が無い')
        groups.setdefault(int(mode), []).append((index, np.asarray(predictor.getters[int(mode)](row), dtype=np.float32)))
    for mode, items in groups.items():
        evaluator = predictor.evaluators[mode]
        values = np.stack([vector for _, vector in items]).reshape(len(items), len(evaluator.columns))
        with predictor.pools.limit(limits=THREADS, user_api='blas'), \
                predictor.pools.limit(limits=hgb_threads(), user_api='openmp'):
            probabilities = evaluator.model.predict_proba(values)[:, 1]
        for (index, _), probability in zip(items, probabilities):
            out[index] = float(probability)
    return out


def code_digest(root: Any) -> str:
    """B3経路のソースのSHA (改行差を吸収)。確率の再利用キーに含め、コードが変われば無効化する。"""
    digest = sha256()
    for name in CODE_FILES:
        digest.update((root/name).read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()[:16]


def model_digest(predictor: Any) -> str:
    """固定B3モデルの指紋 (評価器の列とモデルの内容)。"""
    return fingerprint([(ev.columns, ev.model) if ev is not None else None for ev in predictor.evaluators])[:16]


class B3Batch:
    """1更新内の候補盤をまとめて評価し、確率を再利用する。抜き取り照合の件数を持つ。"""

    def __init__(self, predictor: Any, table: Any, native: Any, salt: str,
                 light: bool = True, batch: bool = True, reuse: bool = True) -> None:
        self.predictor, self.table, self.native, self.salt = predictor, table, native, salt
        self.light, self.batch, self.reuse = light, batch, reuse
        self.cache: OrderedDict[tuple, float] = OrderedDict()
        self.stats = dict(requests=0, features=0, predicted=0, hits=0, batches=0, rows_batched=0,
                          audits_feature=0, audits_hit=0, audits_batch=0, mismatches_feature=0,
                          mismatches_hit=0, mismatches_batch=0, state_mutations=0)
        self.audit_seconds = 0.

    def _features(self, state: Any, boards: list, pending: list) -> dict:
        self.stats['features'] += 1
        if not self.light:
            return forecast_features(state, boards, pending, self.table, self.native)
        row = forecast_features_light(state, boards, pending, self.table, self.native)
        if self.stats['features'] % FEATURE_AUDIT_EVERY == 0:
            self._audit_features(state, boards, pending, row)
        return row

    def _audit_features(self, state: Any, boards: list, pending: list, row: dict) -> None:
        began = perf_counter()
        before = fingerprint(state)
        reference = forecast_features(state, boards, pending, self.table, self.native)
        self.stats['audits_feature'] += 1
        self.stats['mismatches_feature'] += not same_features(reference, row)
        self.stats['state_mutations'] += before != fingerprint(state)
        self.audit_seconds += perf_counter()-began

    def _key(self, row: dict) -> tuple:
        from src.b3_train_model import MODE_COLUMN
        mode = float(row[MODE_COLUMN])
        if mode not in (0.0, 1.0):
            return (self.salt, mode, b'')
        vector = np.asarray(self.predictor.getters[int(mode)](row), dtype=np.float32)
        return (self.salt, mode, vector.tobytes())

    def evaluate(self, state: Any, requests: list[tuple[list, list]]) -> list[float]:
        """各要求 (両側の盤面, 両側の予告) の確率。要求の順に返す。"""
        self.stats['requests'] += len(requests)
        rows = [self._features(state, boards, pending) for boards, pending in requests]
        keys = [self._key(row) if self.reuse else None for row in rows]
        values: list[float | None] = [None]*len(rows)
        for index, key in enumerate(keys):
            if key is not None and key in self.cache and key[2]:
                self.cache.move_to_end(key)
                values[index] = self.cache[key]
                self.stats['hits'] += 1
                if self.stats['hits'] % HIT_AUDIT_EVERY == 0:
                    values[index] = self._audit_hit(rows[index], values[index])
        missing = [i for i, v in enumerate(values) if v is None]
        for index, probability in zip(missing, self._predict([rows[i] for i in missing])):
            values[index] = probability
            if keys[index] is not None and keys[index][2]:
                self.cache[keys[index]] = probability
        while len(self.cache) > PROBABILITY_CACHE_LIMIT:
            self.cache.popitem(last=False)
        return [float(v) for v in values]

    def _predict(self, rows: list[dict]) -> list[float]:
        if not rows:
            return []
        self.stats['predicted'] += len(rows)
        if not self.batch:
            return [float(self.predictor.predict(row)) for row in rows]
        self.stats['batches'] += 1
        self.stats['rows_batched'] += len(rows)
        result = predict_rows(self.predictor, rows)
        if self.stats['batches'] % BATCH_AUDIT_EVERY == 0:
            began = perf_counter()
            reference = [float(self.predictor.predict(row)) for row in rows]
            self.stats['audits_batch'] += 1
            same = all(a == b or (np.isnan(a) and np.isnan(b)) for a, b in zip(reference, result))
            self.stats['mismatches_batch'] += not same
            result = reference if not same else result
            self.audit_seconds += perf_counter()-began
        return result

    def _audit_hit(self, row: dict, cached: float) -> float:
        began = perf_counter()
        actual = float(self.predictor.predict(row))
        self.stats['audits_hit'] += 1
        mismatch = not (actual == cached or (np.isnan(actual) and np.isnan(cached)))
        self.stats['mismatches_hit'] += mismatch
        if mismatch:
            self.cache.clear()
        self.audit_seconds += perf_counter()-began
        return actual if mismatch else cached

    def take_audit_seconds(self) -> float:
        """直前の取り出し以降に照合へ使った時間 (計測時間から差し引く)。"""
        seconds, self.audit_seconds = self.audit_seconds, 0.
        return seconds


Evaluator = Callable[[list[tuple[list, list]]], list[float]]
