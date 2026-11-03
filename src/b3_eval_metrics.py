"""B3 の採点 (事前登録 §2.4): LL / Brier / ECE / AUC、試合等重み・動画等重み、null 課点、group bootstrap。

重みの扱い: 行の `weight` は試合内合計 1 (原向き 0.5 + 複製 0.5)。試合等重み = 各試合の重み付き平均を
試合間で単純平均。動画等重み = 動画内の試合平均を動画間で単純平均。
層 (行部分集合) の値は、その層に行を持つ試合ごとに層内で重みを再正規化した平均を試合間で平均する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from src.b3_train_config import (BOOTSTRAP_N, BOOTSTRAP_SEED, CI_HIGH_PCT, CI_LOW_PCT, ECE_BINS, NULL_P, PROB_CLIP)

STATUS_OK, STATUS_HOLD, STATUS_NULL = 0, 1, 2   # 提供状態 (推論成功 / 最大500ms保持 / null)


def clip_p(p: np.ndarray) -> np.ndarray:
    """全構成共通の clip [1e-6, 1-1e-6]。"""
    return np.clip(np.asarray(p, dtype=np.float64), PROB_CLIP, 1.0 - PROB_CLIP)


def row_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    pc = clip_p(p)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(pc) + (1.0 - y) * np.log1p(-pc))


def row_brier(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (clip_p(p) - np.asarray(y, dtype=np.float64)) ** 2


def apply_null(p: np.ndarray, status: Optional[np.ndarray]) -> np.ndarray:
    """null 状態の行は p=0.5 で課点する (都合の悪い時間を採点から消さない)。status=None は全行 OK。"""
    p = np.asarray(p, dtype=np.float64).copy()
    if status is None:
        return p
    p[np.asarray(status) == STATUS_NULL] = NULL_P
    return p


def null_report(status: Optional[np.ndarray], held_sec: np.ndarray, weight: np.ndarray) -> dict:
    """保持率・null 率 (行数・保持秒・重みの3種の分母つき)。"""
    n = len(held_sec)
    if status is None:
        return {"rows": n, "ok_rows": n, "hold_rows": 0, "null_rows": 0, "hold_rate_sec": 0.0, "null_rate_sec": 0.0}
    st = np.asarray(status)
    h = np.nan_to_num(np.asarray(held_sec, dtype=np.float64))
    tot = float(h.sum())
    return {"rows": n, "ok_rows": int((st == STATUS_OK).sum()), "hold_rows": int((st == STATUS_HOLD).sum()),
            "null_rows": int((st == STATUS_NULL).sum()), "total_sec": tot,
            "hold_rate_sec": float(h[st == STATUS_HOLD].sum() / tot) if tot else float("nan"),
            "null_rate_sec": float(h[st == STATUS_NULL].sum() / tot) if tot else float("nan")}


def weighted_mean(v: np.ndarray, w: np.ndarray) -> float:
    s = float(np.sum(w))
    return float(np.sum(v * w) / s) if s > 0 else float("nan")


def ece(y: np.ndarray, p: np.ndarray, w: np.ndarray, bins: int = ECE_BINS) -> float:
    """10 等幅 bin の期待較正誤差 (重み付き)。"""
    p = clip_p(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = float(np.sum(w))
    if total <= 0:
        return float("nan")
    err = 0.0
    for b in range(bins):
        m = idx == b
        wb = float(np.sum(w[m]))
        if wb > 0:
            err += wb / total * abs(weighted_mean(y[m], w[m]) - weighted_mean(p[m], w[m]))
    return err


def auc(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """重み付き AUC (報告のみ。採否に使わない)。"""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p, sample_weight=w))


@dataclass
class GameTable:
    """試合単位の値 (試合等重みで集計する単位)。"""
    frame: pd.DataFrame   # columns: video_id, game_idx, group, value, n_rows, weight_sum, held_sec


def game_values(df: pd.DataFrame, values: np.ndarray, group_of_video: Mapping[str, str],
                mask: Optional[np.ndarray] = None) -> pd.DataFrame:
    """行ごとの値を (動画,試合) ごとの重み付き平均にする。mask は層の選択 (層内で重みを再正規化)。"""
    w = df["weight"].to_numpy(dtype=np.float64)
    h = np.nan_to_num(df["held_sec"].to_numpy(dtype=np.float64))
    sel = np.ones(len(df), bool) if mask is None else np.asarray(mask, bool)
    tmp = pd.DataFrame({"video_id": df["video_id"].to_numpy()[sel], "game_idx": df["game_idx"].to_numpy()[sel],
                        "wv": (values * w)[sel], "w": w[sel], "h": h[sel]})
    agg = tmp.groupby(["video_id", "game_idx"], sort=True).agg(wv=("wv", "sum"), w=("w", "sum"),
                                                              n_rows=("w", "size"), held_sec=("h", "sum")).reset_index()
    agg = agg[agg["w"] > 0].copy()
    agg["value"] = agg["wv"] / agg["w"]
    agg["group"] = agg["video_id"].map(group_of_video)
    if agg["group"].isna().any():
        raise KeyError("group 未割当の動画がある")
    return agg[["video_id", "game_idx", "group", "value", "n_rows", "w", "held_sec"]].rename(columns={"w": "weight_sum"})


def match_equal_mean(gv: pd.DataFrame) -> float:
    return float(gv["value"].mean()) if len(gv) else float("nan")


def video_equal_mean(gv: pd.DataFrame) -> float:
    return float(gv.groupby("video_id")["value"].mean().mean()) if len(gv) else float("nan")


def average_over_seeds(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """3 seed の試合別値を試合ごとに平均する (独立な動画として水増ししない)。"""
    base = tables[0].set_index(["video_id", "game_idx"])
    vals = np.mean([t.set_index(["video_id", "game_idx"]).loc[base.index, "value"].to_numpy() for t in tables], axis=0)
    out = base.copy()
    out["value"] = vals
    return out.reset_index()


def population(gv: pd.DataFrame) -> dict:
    """層の母数 (試合数・group 数・動画数・保持秒・行数)。"""
    return {"games": int(len(gv)), "groups": int(gv["group"].nunique()), "videos": int(gv["video_id"].nunique()),
            "rows": int(gv["n_rows"].sum()), "held_sec": float(gv["held_sec"].sum())}


def paired_bootstrap(cand: pd.DataFrame, ref: pd.DataFrame, n: int = BOOTSTRAP_N,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    """候補−比較相手の試合等重み差を元動画 group 単位で再抽出 (対応付き)。

    cand/ref は game_values の出力。共通の (動画,試合) だけを使い、欠ける試合数を報告する。
    group 再抽出は G 個を復元抽出 (多項分布) して Σ(試合差)/Σ(試合数)。
    """
    key = ["video_id", "game_idx"]
    m = cand.merge(ref, on=key, suffixes=("_c", "_r"))
    if m.empty:
        return {"n_games": 0, "n_groups": 0, "point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    m["diff"] = m["value_c"] - m["value_r"]
    g = m.groupby("group_c").agg(s=("diff", "sum"), n=("diff", "size")).sort_index()
    s, cnt = g["s"].to_numpy(), g["n"].to_numpy(dtype=np.float64)
    point = float(s.sum() / cnt.sum())
    rng = np.random.default_rng(seed)
    G = len(g)
    counts = rng.multinomial(G, np.full(G, 1.0 / G), size=n).astype(np.float64)
    boot = (counts @ s) / (counts @ cnt)
    return {"n_games": int(len(m)), "n_groups": int(G), "point": point,
            "lo": float(np.percentile(boot, CI_LOW_PCT)), "hi": float(np.percentile(boot, CI_HIGH_PCT)),
            "games_only_cand": int(len(cand) - len(m)), "games_only_ref": int(len(ref) - len(m)),
            "bootstrap_n": int(n), "bootstrap_seed": int(seed)}


def score_predictions(df: pd.DataFrame, p: np.ndarray, group_of_video: Mapping[str, str],
                      status: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None) -> dict:
    """1構成の採点一式 (LL/Brier/ECE/AUC、試合等重みと動画等重み)。null は 0.5 で課点。"""
    y = df["won_p1"].to_numpy(dtype=np.float64)
    pe = apply_null(p, status)
    ll, br = row_log_loss(y, pe), row_brier(y, pe)
    w = df["weight"].to_numpy(dtype=np.float64)
    sel = np.ones(len(df), bool) if mask is None else np.asarray(mask, bool)
    gl = game_values(df, ll, group_of_video, sel)
    gb = game_values(df, br, group_of_video, sel)
    # ECE は試合等重みの重み (各試合の層内重みを 1 に正規化して試合等重みにする)
    gw = pd.Series(w[sel]).groupby([df["video_id"].to_numpy()[sel], df["game_idx"].to_numpy()[sel]]).transform("sum").to_numpy()
    w_game_equal = np.divide(w[sel], gw, out=np.zeros_like(w[sel]), where=gw > 0)
    return {"ll": match_equal_mean(gl), "brier": match_equal_mean(gb),
            "ece": ece(y[sel], pe[sel], w_game_equal), "auc": auc(y[sel], pe[sel], w_game_equal),
            "ll_video_equal": video_equal_mean(gl), "brier_video_equal": video_equal_mean(gb),
            "population": population(gl), "null": null_report(status, df["held_sec"].to_numpy(), w)}


def paired_scalar_diff(df: pd.DataFrame, y_col: str, p_c: np.ndarray, p_r: np.ndarray,
                       fn, mask: Optional[np.ndarray] = None) -> float:
    """ECE 差などスカラー指標の候補−比較相手 (試合等重み)。fn(y,p,w)->float。"""
    y = df[y_col].to_numpy(dtype=np.float64)
    w = df["weight"].to_numpy(dtype=np.float64)
    sel = np.ones(len(df), bool) if mask is None else np.asarray(mask, bool)
    gw = pd.Series(w[sel]).groupby([df["video_id"].to_numpy()[sel], df["game_idx"].to_numpy()[sel]]).transform("sum").to_numpy()
    wg = np.divide(w[sel], gw, out=np.zeros_like(w[sel]), where=gw > 0)
    return float(fn(y[sel], clip_p(p_c[sel]), wg) - fn(y[sel], clip_p(p_r[sel]), wg))


def oracle_exceeded(candidate_ll: float, oracle_ll: Optional[float], margin: float = 0.0) -> Optional[bool]:
    """候補 LL がオラクル (未来の終点得点を使う参考上限) を下回れば True = リーク疑いで停止。

    oracle_ll が未指定なら None (判定不能)。呼出側は None を合格扱いにしない。
    """
    if oracle_ll is None or not np.isfinite(candidate_ll):
        return None
    return bool(candidate_ll < oracle_ll - margin)
