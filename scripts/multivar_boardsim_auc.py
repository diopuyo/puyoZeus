from __future__ import annotations
import logging, os, sys
from pathlib import Path
from typing import NamedTuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

DATA_CSV: Path = Path("data/indicators_v2/prescreen_boardsim_auc.csv")
OUT_CSV: Path = Path("data/indicators_v2/multivar_boardsim_auc.csv")
LOG_PATH: Path = Path("logs/multivar_boardsim.log")
N_FOLDS: int = 5
MIN_ROWS: int = 50
RANDOM_STATE: int = 42
N_PERM: int = 5
OMP_T: str = "3"
MKL_T: str = "3"
BOARDSIM_NAMES: list[str] = [
    "saturated_chain_count", "ignition_point_count", "multi_color_ignition",
    "sub_chain_count", "simultaneous_pop_richness",
]
BASELINE_COLS: list[str] = [
    "fire_current_max_chain", "fire_death_margin", "fire_absorption_capacity",
    "fire_dig_resistance", "fire_potential_fire_power", "fire_immediate_fire_power",
    "fire_board_ojama_count", "fire_max_column_height", "fire_second_chain_potential",
    "opp_current_max_chain", "opp_death_margin", "opp_absorption_capacity",
    "opp_dig_resistance", "opp_potential_fire_power", "opp_immediate_fire_power",
    "opp_board_ojama_count", "opp_max_column_height", "opp_second_chain_potential",
    "diff_current_max_chain", "diff_death_margin", "diff_absorption_capacity",
    "diff_dig_resistance", "diff_potential_fire_power", "diff_immediate_fire_power",
    "diff_board_ojama_count", "diff_max_column_height", "diff_second_chain_potential",
    "net_ojama",
]
RELATION_COLS: list[str] = [
    "death_margin_ratio", "chain_ratio", "height_ratio",
    "absorption_ratio", "ojama_asymmetry",
]
RELATION_EXCL: frozenset[str] = frozenset([
    "fire_death_margin", "opp_death_margin", "diff_death_margin",
    "fire_current_max_chain", "opp_current_max_chain", "diff_current_max_chain",
    "fire_max_column_height", "opp_max_column_height", "diff_max_column_height",
    "fire_absorption_capacity", "opp_absorption_capacity", "diff_absorption_capacity",
    "fire_board_ojama_count", "opp_board_ojama_count", "diff_board_ojama_count",
])
TARGETS: list[str] = ["won", "opp_buried"]
PHASES: list[str] = ["全体", "中盤", "終盤"]
PHASE_MAP: dict[str, str] = {"中盤": "中", "終盤": "終"}


class OofResult(NamedTuple):
    auc: float
    n: int
    n_feat: int
    nan_rate: float
    top10: list[tuple[str, float]]


def _fill_bsim_nan(df: pd.DataFrame) -> pd.DataFrame:
    """board sim NaNを-1で填充し欠損フラグとしてモデルに渡す。"""
    out = df.copy()
    for nm in BOARDSIM_NAMES:
        for pfx in ("fire_", "opp_", "diff_"):
            c = pfx + nm
            if c in out.columns:
                out[c] = out[c].fillna(-1.0)
    return out


def _add_relation(df: pd.DataFrame) -> pd.DataFrame:
    """比/差の関係化指標列を追加して返す。"""
    eps = 1e-6
    out = df.copy()
    out["death_margin_ratio"] = out["fire_death_margin"] / (out["opp_death_margin"] + eps)
    out["chain_ratio"] = out["fire_current_max_chain"] / (out["opp_current_max_chain"] + eps)
    out["height_ratio"] = out["fire_max_column_height"] / (out["opp_max_column_height"] + eps)
    out["absorption_ratio"] = out["fire_absorption_capacity"] / (out["opp_absorption_capacity"] + eps)
    out["ojama_asymmetry"] = out["fire_board_ojama_count"] - out["opp_board_ojama_count"]
    return out


def _get_bsim_cols(df: pd.DataFrame) -> list[str]:
    return [pfx + nm for nm in BOARDSIM_NAMES for pfx in ("fire_", "opp_", "diff_") if pfx + nm in df.columns]


def _get_phase_df(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """位相フィルタを適用したDataFrameを返す。"""
    if phase == "全体":
        return df
    return df[df["phase"] == PHASE_MAP[phase]]


def _oof(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> OofResult:
    """GroupKFold OOF AUCを計算しOofResultを返す。"""
    feat_names = list(X.columns)
    nan_rate = float(X.isna().mean().mean())
    xa = X.values.astype(np.float32)
    ya = y.values.astype(np.int32)
    ga = groups.values
    if len(np.unique(ya)) < 2:
        return OofResult(float("nan"), len(ya), len(feat_names), nan_rate, [])
    oof = np.full(len(ya), np.nan)
    gkf = GroupKFold(n_splits=N_FOLDS)
    for _, (tr, va) in enumerate(gkf.split(xa, ya, ga)):
        if len(np.unique(ya[tr])) < 2 or len(np.unique(ya[va])) < 2:
            continue
        clf = HistGradientBoostingClassifier(
            max_iter=300, max_leaf_nodes=31, learning_rate=0.05, random_state=RANDOM_STATE,
        )
        clf.fit(xa[tr], ya[tr])
        oof[va] = clf.predict_proba(xa[va])[:, 1]
    valid = ~np.isnan(oof)
    if valid.sum() < MIN_ROWS or len(np.unique(ya[valid])) < 2:
        return OofResult(float("nan"), len(ya), len(feat_names), nan_rate, [])
    auc = float(roc_auc_score(ya[valid], oof[valid]))
    clf_full = HistGradientBoostingClassifier(
        max_iter=300, max_leaf_nodes=31, learning_rate=0.05, random_state=RANDOM_STATE,
    )
    clf_full.fit(xa, ya)
    perm = permutation_importance(
        clf_full, xa, ya, n_repeats=N_PERM, random_state=RANDOM_STATE, scoring="roc_auc",
    )
    top10 = sorted(zip(feat_names, perm.importances_mean.tolist()), key=lambda kv: kv[1], reverse=True)[:10]
    return OofResult(auc, int(valid.sum()), len(feat_names), nan_rate, top10)


def main() -> None:
    """エントリポイント。"""
    os.environ.setdefault("OMP_NUM_THREADS", OMP_T)
    os.environ.setdefault("MKL_NUM_THREADS", MKL_T)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(LOG_PATH), encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("=== multivar_boardsim_auc 開始 ===")
    df_raw = pd.read_csv(str(DATA_CSV))
    logger.info("データ: %d行 %d列 %d動画",
               df_raw.shape[0], df_raw.shape[1], df_raw["video_id"].nunique())
    bsim_raw = [c for c in df_raw.columns if any(nm in c for nm in BOARDSIM_NAMES)]
    for c in bsim_raw:
        logger.info("NaN確認 %s: %.1f%%", c, df_raw[c].isna().mean() * 100)
    df = _fill_bsim_nan(df_raw)
    df = _add_relation(df)
    bsim_cols = _get_bsim_cols(df)
    logger.info("board sim列数: %d", len(bsim_cols))
    base_excl = [c for c in BASELINE_COLS if c not in RELATION_EXCL]
    conditions: dict[str, list[str]] = {
        "baseline": BASELINE_COLS,
        "+board_sim": BASELINE_COLS + bsim_cols,
        "+rel_full": BASELINE_COLS + RELATION_COLS,
        "+rel_only": base_excl + RELATION_COLS,
        "+all": BASELINE_COLS + bsim_cols + RELATION_COLS,
    }
    records: list[dict] = []
    for phase in PHASES:
        df_ph = _get_phase_df(df, phase)
        logger.info("=== 位相: %s (n=%d) ===", phase, len(df_ph))
        if len(df_ph) < MIN_ROWS:
            continue
        for target in TARGETS:
            y = df_ph[target]
            baseline_auc: float | None = None
            for cond, feat_cols in conditions.items():
                valid_cols = [c for c in feat_cols if c in df_ph.columns]
                x_df = df_ph[valid_cols].copy()
                rem = int(x_df.isna().sum().sum())
                if rem > 0:
                    logger.warning("NaN残留 %d: %s", rem, cond)
                    x_df = x_df.fillna(x_df.median())
                logger.info("OOF: %s/%s/%s feat=%d", cond, phase, target, len(valid_cols))
                res = _oof(x_df, y, df_ph["video_id"])
                if cond == "baseline":
                    baseline_auc = res.auc
                if (baseline_auc is not None
                        and not (res.auc != res.auc)
                        and not (baseline_auc != baseline_auc)):
                    delta: float = res.auc - baseline_auc
                else:
                    delta = float("nan")
                if res.top10:
                    logger.info("  Top10 [%s/%s/%s]:", cond, phase, target)
                    for feat, imp in res.top10:
                        logger.info("    %.4f  %s", imp, feat)
                records.append({
                    "condition": cond, "phase": phase, "target": target,
                    "auc": res.auc, "delta_vs_baseline": delta,
                    "n": res.n, "n_feat": res.n_feat, "nan_rate": res.nan_rate,
                })
                logger.info("  AUC=%.4f DELTA=%+.4f n=%d feat=%d",
                            res.auc, delta, res.n, res.n_feat)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rdf = pd.DataFrame(records)
    rdf.to_csv(str(OUT_CSV), index=False)
    logger.info("結果保存: %s", OUT_CSV)
    logger.info("\n=== OOF AUC サマリ ===")
    for phase in PHASES:
        for target in TARGETS:
            sub = rdf[(rdf["phase"] == phase) & (rdf["target"] == target)]
            if sub.empty:
                continue
            logger.info("[%s / %s]", phase, target)
            for _, row in sub.iterrows():
                logger.info("  %-20s AUC=%.4f DELTA=%+.4f feat=%d",
                            row["condition"], row["auc"],
                            row["delta_vs_baseline"], row["n_feat"])
    logger.info("=== multivar_boardsim_auc 完了 ===")


if __name__ == "__main__":
    main()
