"""色ぷよ差の単調性の事後検査 (事前登録 §3.5)。学習の制約にはしない。

静止 mode の行について、自側の色ぷよ数を ±k 個動かした反実仮想の入力を作り、勝率が単調に動くか調べる。
動かす列の一覧 (F9) は学習開始前に固定する (ここでは列と 1 個あたりの変化量を引数で受ける)。
層 = (盤面おじゃま段換算, 予告段換算, 局面)。違反 = 色ぷよを増やしたのに自側勝率が下がる (減らしたのに上がる) 幅。
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.b3_train_config import MONO_STEPS
from src.b3_eval_strata import static_mask

OJAMA_PER_ROW = 6        # 段換算 = 個数 ÷ 6 (切り捨て)
VIOLATION_GAME_WIDTH = 0.05


def make_count_perturb(per_unit: Mapping[str, float]) -> Callable[[pd.DataFrame, int], pd.DataFrame]:
    """列ごとの 1 個あたり加算量で入力を動かす関数を作る (列は学習前に固定する F9)。"""
    def perturb(df: pd.DataFrame, k: int) -> pd.DataFrame:
        out = df.copy()
        for col, unit in per_unit.items():
            out[col] = out[col].astype("float64") + k * unit
        return out
    return perturb


def stage_layers(df: pd.DataFrame, own_ojama_col: str, opp_ojama_col: str, own_forecast_col: str,
                 opp_forecast_col: str) -> np.ndarray:
    """層キー (文字列)。盤面おじゃま・予告は両側の段換算、局面は df.phase。欠測は 'NA'。"""
    def stage(col: str) -> np.ndarray:
        v = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
        return np.where(np.isfinite(v), np.floor(v / OJAMA_PER_ROW), np.nan)
    cols = [stage(own_ojama_col), stage(opp_ojama_col), stage(own_forecast_col), stage(opp_forecast_col),
            df["phase"].to_numpy(dtype=np.float64, na_value=np.nan)]
    keys = []
    for row in zip(*cols):
        keys.append("NA" if any(np.isnan(x) for x in row) else "|".join(str(int(x)) for x in row))
    return np.array(keys)


def check_monotonic(df: pd.DataFrame, predict: Callable[[pd.DataFrame], np.ndarray],
                    perturb: Callable[[pd.DataFrame, int], pd.DataFrame], layer_keys: np.ndarray,
                    steps: Sequence[int] = MONO_STEPS) -> list[dict]:
    """静止 mode 行だけで反実仮想を作り、層ごとの違反幅・保持秒・違反試合数を返す。"""
    sel = static_mask(df)
    sub = df[sel]
    keys = layer_keys[sel]
    base = predict(sub)
    held = np.nan_to_num(sub["held_sec"].to_numpy(dtype=np.float64, na_value=np.nan))
    viol = np.zeros(len(sub))
    for k in steps:
        p = predict(perturb(sub, k))
        delta = p - base
        v = np.where(k > 0, np.maximum(0.0, -delta), np.maximum(0.0, delta))   # 増やして下がる / 減らして上がる
        viol = np.maximum(viol, v)
    games = (sub["video_id"].astype(str) + "|" + sub["game_idx"].astype(str)).to_numpy()
    layers = []
    for key in sorted(set(keys)):
        m = keys == key
        mv = float(viol[m].max())
        over = len(set(games[m & (viol > VIOLATION_GAME_WIDTH)]))
        layers.append({"name": key, "held_sec": float(held[m].sum()),   # 違反のある層は層全体の保持秒で数える (§3.5)
                        "rows": int(m.sum()), "max_violation": mv,
                       "games_over_005": int(over)})
    return layers


def static_total_sec(df: pd.DataFrame) -> float:
    held = np.nan_to_num(df["held_sec"].to_numpy(dtype=np.float64, na_value=np.nan))
    return float(held[static_mask(df)].sum())
