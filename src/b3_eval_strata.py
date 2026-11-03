"""B3 の層別採点用マスク (事前登録 §2.5)。すべて当時までに観測された情報だけで決める。

本線後・判定不能は (試合, 向き) ごとに frame_idx 順の状態機械で作る (未来の行を使わない)。
前提 (要確認): ep_honsen60_* は「直近に閉じた発火エピソード」の判定で、ep_closed_n が増えた行で確定する。
主経路 = 盤面差分 (ep_honsen60_board_*)、副経路 = 式 (ep_honsen60_formula_*)。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.b3_train_config import OTHER_STRATA, PAIR_STRATA, PHASE_NAMES, REPORT_STRATA

SIDES = ("p1", "p2")
HONSEN_TRUE = 0.5          # honsen60 列は 0/1 (NaN=不明)。0.5 以上を「本線」とみなす


def _col(df: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy(dtype=np.float64, na_value=np.nan)
    return np.full(len(df), default)


def phase_mask(df: pd.DataFrame, idx: int) -> np.ndarray:
    ph = _col(df, "phase")
    return np.isfinite(ph) & (ph == idx)


def exchange_mask(df: pd.DataFrame) -> np.ndarray:
    m = _col(df, "mode_exchange")
    return np.isfinite(m) & (m == 1)


def static_mask(df: pd.DataFrame) -> np.ndarray:
    m = _col(df, "mode_exchange")
    return np.isfinite(m) & (m == 0)


def both_chain_mask(df: pd.DataFrame) -> np.ndarray:
    a, b = _col(df, "chain_active_p1"), _col(df, "chain_active_p2")
    return np.isfinite(a) & np.isfinite(b) & (a == 1) & (b == 1)


def both_build_mask(df: pd.DataFrame) -> np.ndarray:
    """静止 mode で両側の盤面が観測済み (STABLE 確定盤面あり) かつ連鎖中でない。"""
    ok1 = _col(df, "mk_board_p1") == 0
    ok2 = _col(df, "mk_board_p2") == 0
    a, b = _col(df, "chain_active_p1", 0.0), _col(df, "chain_active_p2", 0.0)
    return static_mask(df) & ok1 & ok2 & ~((a == 1) | (b == 1))


def _episode_value(df: pd.DataFrame, side: str, prefer: str = "board") -> np.ndarray:
    """主経路 (盤面差分) を優先し、欠測なら副経路 (式)。両方 NaN は NaN。"""
    b, f = _col(df, f"ep_honsen60_board_{side}"), _col(df, f"ep_honsen60_formula_{side}")
    return np.where(np.isfinite(b), b, f)


def honsen_states(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(本線後, 本線判定不能) の行マスク。行の並びに依存せず (試合,向き) ごとに frame 順で計算する。"""
    n = len(df)
    honsen = np.zeros(n, bool)
    unknown = np.zeros(n, bool)
    ep = {s: _col(df, f"ep_closed_n_{s}", 0.0) for s in SIDES}
    val = {s: _episode_value(df, s) for s in SIDES}
    keys = pd.DataFrame({"v": df["video_id"].to_numpy(), "g": df["game_idx"].to_numpy(),
                         "o": df["orientation"].to_numpy(), "f": df["frame_idx"].to_numpy(),
                         "i": np.arange(n)})
    for _, grp in keys.groupby(["v", "g", "o"], sort=False):
        order = grp.sort_values(["f", "i"])["i"].to_numpy()
        is_honsen = False
        is_unknown = False
        prev = {s: 0.0 for s in SIDES}
        for i in order:
            for s in SIDES:
                cur = ep[s][i]
                if np.isfinite(cur) and cur > prev[s]:
                    v = val[s][i]
                    if np.isfinite(v):
                        if v >= HONSEN_TRUE:
                            is_honsen, is_unknown = True, False
                        elif not is_honsen:
                            is_unknown = False
                    elif not is_honsen:
                        is_unknown = True
                    prev[s] = cur
            honsen[i], unknown[i] = is_honsen, is_unknown and not is_honsen
    return honsen, unknown


def chain7plus_after_mask(df: pd.DataFrame) -> np.ndarray:
    """旧定義 (7連鎖以上の完走後、参考併記のみ)。(試合,向き) 内で観測後は持続。"""
    n = len(df)
    out = np.zeros(n, bool)
    c = np.nan_to_num(np.maximum(_col(df, "ep_chain7plus_p1"), _col(df, "ep_chain7plus_p2")), nan=0.0)
    keys = pd.DataFrame({"v": df["video_id"].to_numpy(), "g": df["game_idx"].to_numpy(),
                         "o": df["orientation"].to_numpy(), "f": df["frame_idx"].to_numpy(), "i": np.arange(n)})
    for _, grp in keys.groupby(["v", "g", "o"], sort=False):
        order = grp.sort_values(["f", "i"])["i"].to_numpy()
        seen = np.maximum.accumulate((c[order] >= HONSEN_TRUE).astype(np.uint8)).astype(bool)
        out[order] = seen
    return out


def build_strata(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """層名 -> 行マスク。名前は b3_train_config の PAIR/OTHER/REPORT と一致。"""
    honsen, unknown = honsen_states(df)
    out: dict[str, np.ndarray] = {"overall": np.ones(len(df), bool)}
    out["exchange"], out["static"] = exchange_mask(df), static_mask(df)
    out["both_build"], out["both_chain"] = both_build_mask(df), both_chain_mask(df)
    for i, name in enumerate(PHASE_NAMES):
        out[f"phase_{name}"] = phase_mask(df, i)
    out["honsen_after"], out["honsen_unknown"] = honsen, unknown
    out["chain7plus_after_old"] = chain7plus_after_mask(df)
    missing = [s for s in (*PAIR_STRATA, *OTHER_STRATA, *REPORT_STRATA) if s not in out]
    if missing:
        raise KeyError(f"未定義の層: {missing}")
    return out


STRATA_BUILDERS: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    "exchange": exchange_mask, "static": static_mask, "both_chain": both_chain_mask, "both_build": both_build_mask}
