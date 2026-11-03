"""G_feのuser定義局面。学習と提供で同じ純粋関数を使用する。"""
from __future__ import annotations

import numpy as np

from src.exchange_event_features import D_COLUMNS, G_COLUMNS, g_features

CELLS, EARLY_MAX, LATE_MIN = 72, 18, 48
USER_COLUMNS = tuple(n.replace("fill", "U") for n in G_COLUMNS if "elapsed" not in n) + (
    "opp_total_missing",)
USER_MODEL_VERSION = "exchange_event_userphase_v1"


def user_phase(own_total: np.ndarray, diff_total: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """整数個数へ復元し、相手欠測には自側段階と欠測フラグを用いる。"""
    own = np.rint(own_total * CELLS)
    opp = np.rint((own_total - diff_total) * CELLS)
    if not np.isfinite(own).all():
        raise ValueError("自側個数の欠測は事前登録の対象外")
    missing = ~np.isfinite(opp)
    maximum = np.where(missing, own, np.maximum(own, opp))
    phase = np.where(maximum <= EARLY_MAX, 0, np.where(maximum < LATE_MIN, 1, 2))
    return phase, missing.astype(float)


def user_features(design: np.ndarray, probability: np.ndarray, sign: np.ndarray,
                  phase: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """段1と同じ54列を作る。絶対量・局面は反転せずM0のlogitだけ視点変換。"""
    dummy_elapsed = np.zeros_like(phase)
    base = g_features(design, probability, sign, np.column_stack((phase, dummy_elapsed)))
    keep = [i for i, name in enumerate(G_COLUMNS) if "elapsed" not in name]
    return np.column_stack((base[:, keep], missing))


def static_user_features(design: np.ndarray, probability: float, side: int) -> np.ndarray:
    """欠測を0補完する前にUを確定し、その後Dだけを段1と同じ0補完する。"""
    phase, missing = user_phase(design[[D_COLUMNS.index("board_puyo_total")]],
                               design[[D_COLUMNS.index("diff_board_puyo_total")]])
    return user_features(np.nan_to_num(design, nan=0.0)[None, :], np.array([probability]),
                         np.array([1-2*side]), phase, missing)[0]
