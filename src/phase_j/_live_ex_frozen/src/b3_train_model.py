"""B3 学習器: F_static / F_exchange の 2 評価器 + 状態選択 + 内側 3fold の設定選択 (事前登録 §1.1, §3.2)。

- 勾配ブースティング木 (sklearn HistGradientBoostingClassifier、NaN をそのまま分割に使う。fillna(0) 禁止)。
- BLAS は 1、HGB の OpenMP は B3_HGB_THREADS (既定 1) で指定する。
- 行重みは試合内合計 1 の重みを、学習時だけ平均 1 に正規化して渡す (相対比は不変。sklearn の
  min_hessian_to_split が極小重みで分割を封じる事故の回避)。
- 状態選択は勝敗を見ず、行の mode_exchange だけで決める。2 評価器は平均しない。
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.phase_j._live_ex_frozen.src.b3_column_spec import COLUMN_SPECS, EXCLUDED_FAST_COLUMNS
from src.phase_j._live_ex_frozen.src.b3_train_matrix import MATRIX_ATTR, available_columns, column_values, matrix_values, prepare_frame
from src.phase_j._live_ex_frozen.src.b3_eval_metrics import game_values, match_equal_mean, row_log_loss
from src.phase_j._live_ex_frozen.src.b3_train_config import HGB_REFERENCE, INNER_TIE_LL, N_INNER_FOLDS, THREADS
from src.phase_j._live_ex_frozen.src.b3_train_folds import assert_no_group_leak, inner_assignment

MODE_COLUMN = "mode_exchange"
LABEL_COLUMN = "won_p1"
WEIGHT_COLUMN = "weight"
# F9 / b3_column_spec.Kind: 絶対量は側交換、差分はいずれも1P優位ほど増える。
# df_cells_color は p1-p2 (交換で -x)、指標差は 0.5+0.5*(p1-p2) (交換で 1-x)。
# 交換後は教師も1-yとなるため、同じ列・同じ符号で新しい1Pの勝率に整合する。
F9_MONOTONIC: dict[str, int] = {
    "cells_color_p1": 1, "cells_color_p2": -1,
    "ind_board_color_puyo_total_p1": 1, "ind_board_color_puyo_total_p2": -1,
    "df_ind_board_color_puyo_total": 1, "df_cells_color": 1,
}


def monotonic_constraints(columns: Sequence[str], enabled: Optional[bool] = None) -> dict[str, int]:
    """実際に使用する列の制約。既定OFF、B3_MONOTONIC=1で候補だけ有効化する。"""
    if enabled is None:
        value = os.environ.get("B3_MONOTONIC", "0")
        if value not in ("0", "1"):
            raise ValueError("B3_MONOTONIC は 0 または 1 で指定する")
        enabled = value == "1"
    return {c: F9_MONOTONIC[c] for c in columns if c in F9_MONOTONIC} if enabled else {}


# 特徴に入れてはならない列 (教師・重み・保持秒・監査時刻・キー)
FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {"won_p1", "weight", "held_sec", "row_key", "video_id", "game_idx", "frame_idx", "t_sec", "orientation",
     "trigger_mask", "av_max", "av_frame", MODE_COLUMN})


@dataclass(frozen=True)
class HgbConfig:
    """HGB の設定 1 点 (内側 CV の候補)。費用は決定的な代理量 max_iter × max_leaf_nodes。"""
    name: str
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float = 0.0

    @property
    def cost(self) -> int:
        return self.max_iter * self.max_leaf_nodes

    def params(self, seed: int) -> dict:
        return dict(max_iter=self.max_iter, learning_rate=self.learning_rate, max_leaf_nodes=self.max_leaf_nodes,
                    min_samples_leaf=self.min_samples_leaf, l2_regularization=self.l2_regularization,
                    early_stopping=False, random_state=seed)


# 事前登録の候補格子 (提案値。内側 CV だけで選ぶ。候補数を外側を見て増やさない)
DEFAULT_GRID: tuple[HgbConfig, ...] = (
    HgbConfig("small", 150, 0.05, 15, 400),
    HgbConfig("ref", HGB_REFERENCE["max_iter"], HGB_REFERENCE["learning_rate"], HGB_REFERENCE["max_leaf_nodes"],
              HGB_REFERENCE["min_samples_leaf"]),
    HgbConfig("large", 400, 0.04, 63, 100),
)


def feature_columns(df_columns: Sequence[str], include_mask_codes: bool = False) -> list[str]:
    """台帳 role=='feature' (+任意で mk_* 欠測理由コード) の列を、台帳順で返す。

    r_ledger_* (R_only)・av_* (監査)・教師/重み/キーは入らない。df に無い列は黙って落とさず例外にする。
    """
    have = set(df_columns)
    cols = [s.name for s in COLUMN_SPECS if s.role == "feature"
            and s.name not in FORBIDDEN_FEATURES and s.name not in EXCLUDED_FAST_COLUMNS]
    if include_mask_codes:
        cols += [s.name for s in COLUMN_SPECS if s.name.startswith("mk_") and s.role == "audit"]
    missing = [c for c in cols if c not in have]
    if missing:
        raise KeyError(f"学習行に台帳の列が無い: {missing[:8]} (計{len(missing)})")
    bad = [c for c in cols if c in FORBIDDEN_FEATURES or c.startswith("av_") or c.startswith("r_ledger_")]
    if bad:
        raise AssertionError(f"禁止列が特徴に混入: {bad}")
    return cols


def to_matrix(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """float32 行列 (pandas 欠測 <NA> は NaN へ)。fillna(0) はしない。"""
    return matrix_values(df, columns)


def train_weights(w: np.ndarray) -> np.ndarray:
    """試合内合計 1 の重みを平均 1 に正規化 (相対比不変)。"""
    w = np.asarray(w, dtype=np.float64)
    m = float(np.mean(w))
    if not np.isfinite(m) or m <= 0:
        raise ValueError("重みが不正 (平均が 0 以下/非有限)")
    return w / m


def hgb_threads() -> int:
    """HGB だけに適用する OpenMP スレッド数を検証して返す。"""
    value = int(os.environ.get("B3_HGB_THREADS", str(THREADS)))
    if value < 1:
        raise ValueError("B3_HGB_THREADS は正の整数で指定する")
    return value


def fit_hgb(X: np.ndarray, y: np.ndarray, w: np.ndarray, cfg: HgbConfig, seed: int,
            monotonic_cst: Optional[Sequence[int]] = None) -> object:
    """単一 HGB の学習。全 NaN の列は呼出側が除外済みである前提。"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    with threadpool_limits(limits=THREADS, user_api="blas"), \
            threadpool_limits(limits=hgb_threads(), user_api="openmp"):
        params = cfg.params(seed)
        if monotonic_cst is not None:
            params["monotonic_cst"] = list(monotonic_cst)
        model = HistGradientBoostingClassifier(**params)
        model.fit(X, y, sample_weight=train_weights(w))
    return model


def predict_hgb(model, X: np.ndarray) -> np.ndarray:
    with threadpool_limits(limits=THREADS, user_api="blas"), \
            threadpool_limits(limits=hgb_threads(), user_api="openmp"):
        return model.predict_proba(X)[:, 1]


@dataclass
class Evaluator:
    """1 つの評価器 (static か exchange)。使った列と除外した全 NaN 列を記録する。"""
    name: str
    model: object
    columns: list[str]
    dropped_all_nan: list[str]
    n_train_rows: int
    cfg: HgbConfig
    monotonic_cst: dict[str, int] = field(default_factory=dict)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return predict_hgb(self.model, to_matrix(df, self.columns))


def fit_evaluator(name: str, df: pd.DataFrame, columns: Sequence[str], cfg: HgbConfig, seed: int,
                  extra: Optional[Mapping[str, np.ndarray]] = None,
                  monotonic: Optional[bool] = None) -> Evaluator:
    """df の行だけで 1 評価器を学習する。全 train で欠測の列は非対応として落として記録する。

    extra は補助器の出力列 (列名 -> 配列、df と同じ長さ。train 側は内側 OOF であること)。
    """
    names = list(columns)
    # 全NaN列を先に調べ、HGB用float64行列の二重確保を避ける。
    dropped = [c for c in names if np.isnan(column_values(df, c)).all()]
    names = [c for c in names if c not in dropped]
    X = matrix_values(df, names, np.float64 if MATRIX_ATTR in df.attrs else np.float32)
    if extra:
        X = np.column_stack([X] + [np.asarray(v, dtype=np.float32) for v in extra.values()])
        names += list(extra)
    keep = np.array([not np.isnan(X[:, j]).all() for j in range(X.shape[1])])
    dropped += [n for n, k in zip(names, keep) if not k]
    if not keep.all():
        X = X[:, keep]
    y = df[LABEL_COLUMN].to_numpy(dtype=np.float64)
    if len(np.unique(y)) < 2:
        raise ValueError(f"{name}: 教師が 1 クラスだけ (行数 {len(y)})")
    names = [n for n, k in zip(names, keep) if k]
    constraints = monotonic_constraints(names, monotonic)
    signs = [constraints.get(n, 0) for n in names] if constraints else None
    model = fit_hgb(X, y, df[WEIGHT_COLUMN].to_numpy(dtype=np.float64), cfg, seed, signs)
    return Evaluator(name, model, names, dropped, len(df), cfg, constraints)


@dataclass
class TwoStateModel:
    """F_static + F_exchange。行の mode_exchange だけで選ぶ。片方の行が無ければその評価器は None。"""
    static: Optional[Evaluator]
    exchange: Optional[Evaluator]

    def predict_row(self, row: Mapping[str, object]) -> float:
        """固定評価器の1行用経路。列順の準備は初回だけ行う（学習・一括APIは従来通り）。"""
        from src.phase_j._live_ex_frozen.src.b3_predict import RowPredictor
        predictor = getattr(self, '_row_predictor', None)
        if predictor is None:
            predictor = self._row_predictor = RowPredictor(self)
        return predictor.predict(row)

    def __getstate__(self) -> dict:
        """実行時だけの変換器は保存せず、従来のモデル保存形式を保つ。"""
        state = self.__dict__.copy()
        state.pop('_row_predictor', None)
        return state

    def predict(self, df: pd.DataFrame, extra: Optional[Mapping[str, np.ndarray]] = None) -> np.ndarray:
        mode = df[MODE_COLUMN].to_numpy(dtype=np.float64, na_value=np.nan)
        if np.isnan(mode).any():
            raise ValueError("mode_exchange に欠測 (状態を選べない)")
        p = np.full(len(df), np.nan)
        for flag, ev in ((0.0, self.static), (1.0, self.exchange)):
            sel = mode == flag
            if not sel.any():
                continue
            if ev is None:
                raise ValueError(f"mode={int(flag)} の行があるが評価器が無い")
            sub = df[sel]
            ex = None if not extra else {k: np.asarray(v)[sel] for k, v in extra.items()}
            names = [c for c in ev.columns if c in available_columns(sub)]
            X = to_matrix(sub, names)
            if ex:
                ex_names = [c for c in ev.columns if c in ex]
                X = np.column_stack([X] + [np.asarray(ex[c], dtype=np.float32) for c in ex_names])
                names += ex_names
            if names != ev.columns:
                raise AssertionError("予測時の列が学習時と一致しない")
            p[sel] = predict_hgb(ev.model, X)
        return p


def save_model(model: TwoStateModel, path: Path) -> None:
    """特徴列を含む最終モデルを保存する。既存ファイルは上書きしない。"""
    import joblib
    with Path(path).open("xb") as stream:
        joblib.dump(model, stream)


def load_model(path: Path) -> TwoStateModel:
    """自分で保存した信頼できる joblib のみを読む。"""
    import joblib
    model = joblib.load(path)
    if not isinstance(model, TwoStateModel):
        raise TypeError("TwoStateModel の保存ファイルではない")
    return model


def model_monotonic_constraints(model: TwoStateModel) -> dict[str, dict[str, int]]:
    """環境変数でなく、保存対象の評価器が実際に使った列と符号を記録する。"""
    return {name: getattr(ev, "monotonic_cst", {}) for name, ev in
            (("static", model.static), ("exchange", model.exchange)) if ev is not None}


def fit_two_state(df: pd.DataFrame, columns: Sequence[str], cfg: HgbConfig, seed: int,
                  extra: Optional[Mapping[str, np.ndarray]] = None,
                  monotonic: Optional[bool] = None) -> TwoStateModel:
    mode = df[MODE_COLUMN].to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isnan(mode).any():
        raise ValueError("mode_exchange に欠測")
    evs = {}
    for flag, name in ((0.0, "static"), (1.0, "exchange")):
        sel = mode == flag
        ex = None if not extra else {k: np.asarray(v)[sel] for k, v in extra.items()}
        evs[name] = fit_evaluator(name, df[sel], columns, cfg, seed, ex, monotonic) if sel.any() else None
    return TwoStateModel(evs["static"], evs["exchange"])


# ---------- 補助器 (内側 OOF) ----------
# aux(train_df, apply_df) -> apply_df の各行の補助出力 (1 本の配列)。外側 train 行には内側 OOF、test 行には
# train 全体で学習した値を入れる。既存の固定 M0 は使わない (§1.1)。
AuxFn = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def aux_columns(train: pd.DataFrame, test: pd.DataFrame, aux: Mapping[str, AuxFn],
                inner_of_group: Mapping[str, int], group_of_video: Mapping[str, str]
                ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """補助器の列を作る。train は内側 OOF (自分の内側 fold を除いて学習)、test は train 全体で学習。"""
    groups = train["video_id"].map(group_of_video).to_numpy()
    fold = np.array([inner_of_group[g] for g in groups])
    tr_cols: dict[str, np.ndarray] = {}
    te_cols: dict[str, np.ndarray] = {}
    for name, fn in aux.items():
        oof = np.full(len(train), np.nan)
        for k in sorted(set(fold)):
            held = fold == k
            assert_no_group_leak(set(groups[~held]), set(groups[held]))
            oof[held] = fn(train[~held], train[held])
        tr_cols[name] = oof
        te_cols[name] = fn(train, test)
    return tr_cols, te_cols


# ---------- 内側 CV による設定選択 ----------
@dataclass
class InnerChoice:
    chosen: HgbConfig
    table: list[dict]
    tie_rule_applied: bool


def inner_ll(df: pd.DataFrame, columns: Sequence[str], cfg: HgbConfig, seed: int,
             inner_of_group: Mapping[str, int], group_of_video: Mapping[str, str]) -> dict:
    """内側 fold 全体での試合等重み LL (検証側の予測を集めた OOF)。2 評価器とも cfg を使う。

    補助器の列 (aux_columns の出力) は df の通常の列として columns に含めて渡す。"""
    groups = df["video_id"].map(group_of_video).to_numpy()
    fold = np.array([inner_of_group[g] for g in groups])
    p = np.full(len(df), np.nan)
    t0 = time.perf_counter()
    for k in sorted(set(fold)):
        held = fold == k
        assert_no_group_leak(set(groups[~held]), set(groups[held]))
        tr, te = df[~held], df[held]
        model = fit_two_state(tr, columns, cfg, seed)
        p[held] = model.predict(te)
    ll = row_log_loss(df[LABEL_COLUMN].to_numpy(dtype=np.float64), p)
    gv = game_values(df, ll, group_of_video)
    return {"ll": match_equal_mean(gv), "seconds": time.perf_counter() - t0, "games": int(len(gv))}


def select_config(df: pd.DataFrame, columns: Sequence[str], grid: Sequence[HgbConfig], seed: int,
                  inner_of_group: Mapping[str, int], group_of_video: Mapping[str, str],
                  tie: float = INNER_TIE_LL) -> InnerChoice:
    """LL 最小を選び、最小から tie 以内の候補があれば費用 (max_iter×leaf) が最小のものを選ぶ。"""
    df = prepare_frame(df, columns)
    table = []
    for cfg in grid:
        r = inner_ll(df, columns, cfg, seed, inner_of_group, group_of_video)
        table.append({"config": cfg.name, "cost": cfg.cost, **r, "params": asdict(cfg)})
    best = min(t["ll"] for t in table)
    near = [t for t in table if t["ll"] - best <= tie]
    pick = min(near, key=lambda t: (t["cost"], t["ll"], t["config"]))
    by_name = {c.name: c for c in grid}
    lowest = min(table, key=lambda t: t["ll"])
    return InnerChoice(by_name[pick["config"]], table, pick["config"] != lowest["config"])


def outer_inner_assignment(games_per_group: Mapping[str, int], outer_fold: int, outer: Mapping[str, int]):
    return inner_assignment(games_per_group, outer_fold, outer)
