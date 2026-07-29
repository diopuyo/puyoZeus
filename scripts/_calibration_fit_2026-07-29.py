"""校正 (calibration) を実際に当てて効果を測る -- Platt / Isotonic を比較する。

## 背景
scripts/_calibration_check_2026-07-29.py の実測で「有利不利判定は系統的に自信過剰」
(予測0.8台の実勝率が終盤で64.1%等) と判明した。本スクリプトはそれに対して
実際に後段校正 (post-hoc calibration) を当て、ECE/Brier がどこまで改善するかを
実測する。手法は Platt scaling (ロジスティック回帰) と Isotonic 回帰の2種。

## 入れ子 (nested) 構成 -- リーク防止の要点
既存 OOF 予測をそのまま校正器の学習に使うと「学習データで校正器を学習し
同じデータで評価する」リークになる (校正器が最終評価データの分布を直接見てしまう)。
これを避けるため、以下の 2 段構成にした:

1. 外側 GroupKFold (video_id 単位, outer_folds分割):
   outer-train 全体で「最終モデル」を学習し、そのモデルで outer-test を予測した
   ものを raw (未校正) 予測とする。
2. 内側 GroupKFold (outer-train 内部のみ, inner_folds分割):
   outer-train を再度 GroupKFold し、モデルの OOF 予測確率 (= "inner-OOF") を作る。
   Platt/Isotonic 校正器はこの inner-OOF (outer-test を一度も見ていない値) と
   正解ラベルのみで学習する。
3. 校正器を outer-test の raw 予測に適用したものを校正後予測とする。

外側 fold ごとに (a) 最終モデルは outer-train 全体で学習・(b) 校正器は
outer-train 内部の inner-OOF のみで学習、という 2 つの独立な学習物を使うため、
校正器も最終モデルも outer-test を一度も参照しない。よって
outer-test 上で評価した ECE/Brier/AUC は校正込みでリークしていない。

位相 (序盤/中盤/終盤) は tsumo_1p の三分位 (--fixed-q33/--fixed-q67、
既存 run_phase_models と同一の手数境界) で決める。「位相別」校正器は
位相ごとに独立学習したモデル (get_phase_oof と同方式) の inner-OOF に対して
位相ごとに個別の Platt/Isotonic を学習する。「全位相共通」校正器は
全体単一モデルの inner-OOF に対して位相を区別せず1つだけ学習し、
それを位相でスライスして評価する (=「一つの校正関数で全位相の歪みを
同時に直せるか」を直接見るための対照群)。

## 制約
- src/ 配下は一切変更しない・import もしない。
- scripts/model_indicator_win.py, scripts/_calibration_check_2026-07-29.py は無改変
  (関数を import して再利用するのみ)。

## 使い方
    nice -n 19 python -m scripts._calibration_fit_2026-07-29 \
        --labeled data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv \
        --fixed-q33 18 --fixed-q67 40 \
        --out-dir data/verify/win_eval_combined66_2026-07-29/calibration_fit_2026-07-29
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from scripts.model_indicator_win import (
    DEFAULT_MAX_TDIFF,
    GBC_PARAMS,
    N_FOLDS,
    _get_indicator_cols,
    build_features,
    load_labeled_csv,
    pair_sides_for_win,
    run_oof_classifier,
)

# 注意: scripts/_calibration_check_2026-07-29.py はファイル名にハイフンを含み
# `from scripts._calibration_check_2026-07-29 import ...` は構文エラーになる
# (import文はハイフンを識別子として解釈できない、-m実行のみ可能)。
# そのため compute_calibration_table/compute_ece/summarize_distribution は
# 同スクリプトの実装と同一ロジックをここに複製する (同スクリプトは無改変)。

# =============================================================================
# 定数
# =============================================================================

# 校正ビン幅 (0.1刻み = 10ビン、 _calibration_check_2026-07-29.py と同一定義)
N_CALIBRATION_BINS: int = 10

# 分布要約で報告する percentile 一覧 (同上)
DIST_PERCENTILES: tuple[int, ...] = (1, 5, 10, 25, 50, 75, 90, 95, 99)

# 位相ラベル一覧 (この順で表示・集計する)
PHASES: tuple[str, ...] = ("序盤", "中盤", "終盤")

# Platt scaling の logit 変換で 0/1 近傍が発散しないためのクリップ幅
LOGIT_EPS: float = 1e-6

# 校正器 (Platt/Isotonic) を学習可能とみなす inner-OOF 最小件数
# (これ未満は校正をスキップし raw のまま通す = 過学習した校正器を避ける安全弁)
MIN_CALIB_FIT_N: int = 200

# 位相別モデルを学習可能とみなす最小サンプル数 (get_phase_oof と同一基準)
MIN_PHASE_N: int = 20

# 校正器の乱数シード (GBC_PARAMS の random_state と揃える)
RANDOM_STATE: int = 42

# 有利不利スコア -100〜+100 換算の代表確率 (report専用の例示値)
ANSWER_PROBS: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)


# =============================================================================
# 校正テーブル計算 (_calibration_check_2026-07-29.py の同名関数の複製、無改変転記)
# =============================================================================

def compute_calibration_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """予測確率を0.1幅ビンに区切り、ビンごとの平均予測確率・実勝率・件数を返す。"""
    bin_edges = np.linspace(0.0, 1.0, N_CALIBRATION_BINS + 1)
    bin_idx = np.clip(
        np.digitize(p, bin_edges[1:-1], right=False), 0, N_CALIBRATION_BINS - 1
    )
    rows: list[dict] = []
    for b in range(N_CALIBRATION_BINS):
        mask = bin_idx == b
        n = int(mask.sum())
        rows.append({
            "bin": f"{bin_edges[b]:.1f}-{bin_edges[b + 1]:.1f}",
            "n": n,
            "mean_pred": float(p[mask].mean()) if n > 0 else float("nan"),
            "actual_rate": float(y[mask].mean()) if n > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def compute_ece(calib_table: pd.DataFrame, n_total: int) -> float:
    """期待校正誤差 (ECE) = Σ (n_bin/n) * |実勝率 - 平均予測|。"""
    ece = 0.0
    for _, row in calib_table.iterrows():
        if row["n"] == 0:
            continue
        ece += (row["n"] / n_total) * abs(row["actual_rate"] - row["mean_pred"])
    return float(ece)


def summarize_distribution(p: np.ndarray) -> dict:
    """予測確率分布の要約統計 (percentile・両端割合) を返す。"""
    pct = {f"p{q}": float(np.percentile(p, q)) for q in DIST_PERCENTILES}
    return {
        "mean": float(p.mean()), "std": float(p.std()),
        "min": float(p.min()), "max": float(p.max()),
        **pct,
        "frac_in_extreme_0.2_0.8": float(((p <= 0.2) | (p >= 0.8)).mean()),
    }


# =============================================================================
# Platt / Isotonic 校正器
# =============================================================================

def _logit(p: np.ndarray) -> np.ndarray:
    """0/1 近傍をクリップしてから logit 変換する。"""
    p_clip = np.clip(p, LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(p_clip / (1.0 - p_clip))


def fit_platt(raw_p: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Platt scaling: logit(raw_p) を1特徴量とする1次元ロジスティック回帰。"""
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(_logit(raw_p).reshape(-1, 1), y)
    return lr


def apply_platt(model: LogisticRegression, raw_p: np.ndarray) -> np.ndarray:
    """学習済 Platt 校正器を適用する。"""
    return model.predict_proba(_logit(raw_p).reshape(-1, 1))[:, 1]


def fit_isotonic(raw_p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Isotonic 回帰 (単調写像) を学習する。"""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_p, y)
    return iso


def apply_isotonic(model: IsotonicRegression, raw_p: np.ndarray) -> np.ndarray:
    """学習済 Isotonic 校正器を適用する。"""
    return model.predict(raw_p)


# =============================================================================
# 入れ子 (nested) GroupKFold -- raw / Platt / Isotonic の OOF 予測を返す
# =============================================================================

def _effective_folds(n_folds: int, n_groups: int) -> int:
    """GroupKFold の fold 数を実際のグループ (video) 数に収まるよう丸める。"""
    return min(n_folds, max(2, n_groups))


def _run_one_outer_fold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    inner_folds: int,
    fold_idx: int,
    n_outer: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1 outer fold 分の raw/Platt/Isotonic 予測を計算する (nested_calibrated_oof の内部処理)。

    最終モデルは outer-train 全体で学習して outer-test に適用 (raw)。
    校正器は outer-train を内側 GroupKFold した inner-OOF のみで学習する
    (outer-test を一度も参照しないためリークしない)。
    """
    X_tr, y_tr, g_tr = X[tr_idx], y[tr_idx], groups[tr_idx]
    final_model = HistGradientBoostingClassifier(**GBC_PARAMS)
    final_model.fit(X_tr, y_tr)
    raw_te = final_model.predict_proba(X[te_idx])[:, 1]

    eff_inner = _effective_folds(inner_folds, len(np.unique(g_tr)))
    inner_oof, _ = run_oof_classifier(X_tr, y_tr, g_tr, eff_inner)
    inner_valid = ~np.isnan(inner_oof[:, 0])
    inner_p, inner_y = inner_oof[inner_valid, 1], y_tr[inner_valid]

    if len(inner_p) < MIN_CALIB_FIT_N or len(np.unique(inner_y)) < 2:
        print(f"    [{label}] fold {fold_idx + 1}: inner-OOF 不足"
              f" (n={len(inner_p)}) -> 校正スキップ (raw のまま)")
        return raw_te, raw_te.copy(), raw_te.copy()

    platt_model = fit_platt(inner_p, inner_y)
    iso_model = fit_isotonic(inner_p, inner_y)
    platt_te = apply_platt(platt_model, raw_te)
    iso_te = apply_isotonic(iso_model, raw_te)
    print(f"    [{label}] fold {fold_idx + 1}/{n_outer}:"
          f" outer_train={len(tr_idx)} outer_test={len(te_idx)}"
          f" inner_oof_n={len(inner_p)}")
    return raw_te, platt_te, iso_te


def nested_calibrated_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_folds: int,
    inner_folds: int,
    label: str,
) -> dict[str, np.ndarray]:
    """入れ子 GroupKFold で raw/Platt/Isotonic の OOF 予測確率を返す (リークなし)。

    外側 fold の学習側 (outer-train) で最終モデルを学習し検証側 (outer-test) に
    適用したものを raw とする。校正器は outer-train を内側 GroupKFold した
    inner-OOF (outer-test 非依存) だけで学習し、raw に適用する。
    """
    n = len(y)
    raw = np.full(n, np.nan)
    platt = np.full(n, np.nan)
    iso = np.full(n, np.nan)

    eff_outer = _effective_folds(outer_folds, len(np.unique(groups)))
    outer_gkf = GroupKFold(n_splits=eff_outer)
    t0 = time.time()

    for fold_idx, (tr_idx, te_idx) in enumerate(outer_gkf.split(X, y, groups=groups)):
        raw_te, platt_te, iso_te = _run_one_outer_fold(
            X, y, groups, tr_idx, te_idx, inner_folds, fold_idx, eff_outer, label
        )
        raw[te_idx], platt[te_idx], iso[te_idx] = raw_te, platt_te, iso_te

    print(f"    [{label}] 完了 ({time.time() - t0:.1f}秒)")
    return {"raw": raw, "platt": platt, "iso": iso}


# =============================================================================
# 位相ラベル付与
# =============================================================================

def assign_phase_labels(tsumo_vals: np.ndarray, q33: float, q67: float) -> np.ndarray:
    """手数三分位境界で 序盤/中盤/終盤 ラベルを付与する (run_phase_models と同一境界)。"""
    labels = np.full(len(tsumo_vals), "中盤", dtype=object)
    labels[tsumo_vals <= q33] = "序盤"
    labels[tsumo_vals > q67] = "終盤"
    return labels


# =============================================================================
# 指標集計
# =============================================================================

def _phase_sliced_metrics(
    y: np.ndarray, p: np.ndarray, phase_labels: np.ndarray
) -> dict[str, float]:
    """1本のスコア配列を位相でスライスして ECE全体/序盤/中盤/終盤 + Brier + AUC を返す。"""
    out: dict[str, float] = {}
    calib_all = compute_calibration_table(y, p)
    out["ece_全体"] = compute_ece(calib_all, len(y))
    for phase in PHASES:
        mask = phase_labels == phase
        if mask.sum() == 0:
            out[f"ece_{phase}"] = float("nan")
            continue
        calib_ph = compute_calibration_table(y[mask], p[mask])
        out[f"ece_{phase}"] = compute_ece(calib_ph, int(mask.sum()))
    out["brier"] = float(brier_score_loss(y, p))
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return out


def _pool_phase_results(
    phase_results: dict[str, dict], key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """位相別モデルの結果 (y, key配列) を1本にプールし phase_labels も返す。"""
    ys, ps, labels = [], [], []
    for phase, res in phase_results.items():
        ys.append(res["y"])
        ps.append(res[key])
        labels.append(np.full(len(res["y"]), phase, dtype=object))
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(labels)


def build_metric_rows(
    y_all: np.ndarray,
    phase_labels: np.ndarray,
    res_all: dict[str, np.ndarray],
    phase_results: dict[str, dict],
) -> pd.DataFrame:
    """5手法 x (ECE全体/序盤/中盤/終盤/Brier/AUC) の比較表を作る。"""
    rows: list[dict] = []

    # 校正なし(現状): ECE全体は単一全体モデル、位相別列は位相別モデル (現状報告と同一定義)
    # (2つの異なる母集団/モデルを1行に混在させている点に注意。参考行で内訳を分離する)
    m_raw_all = _phase_sliced_metrics(y_all, res_all["raw"], phase_labels)
    y_pool_raw, p_pool_raw, lab_pool_raw = _pool_phase_results(phase_results, "raw")
    m_raw_phase = _phase_sliced_metrics(y_pool_raw, p_pool_raw, lab_pool_raw)
    rows.append({
        "手法": "校正なし(現状)",
        "ECE全体": m_raw_all["ece_全体"],
        "ECE序盤": m_raw_phase["ece_序盤"], "ECE中盤": m_raw_phase["ece_中盤"],
        "ECE終盤": m_raw_phase["ece_終盤"],
        "Brier": m_raw_all["brier"], "AUC": m_raw_all["auc"],
    })
    # 参考行: 「位相別」校正行 (下記) と母集団を完全に揃えた無校正版
    # (=位相別モデルの生スコアのみをプールした場合のECE全体/Brier/AUC)
    rows.append({
        "手法": "参考:校正なし(位相別モデルのみ)",
        "ECE全体": m_raw_phase["ece_全体"],
        "ECE序盤": m_raw_phase["ece_序盤"], "ECE中盤": m_raw_phase["ece_中盤"],
        "ECE終盤": m_raw_phase["ece_終盤"],
        "Brier": m_raw_phase["brier"], "AUC": m_raw_phase["auc"],
    })

    for method_name, key in (("Platt", "platt"), ("Isotonic", "iso")):
        m_common = _phase_sliced_metrics(y_all, res_all[key], phase_labels)
        rows.append({
            "手法": f"{method_name}(全位相共通)",
            "ECE全体": m_common["ece_全体"],
            "ECE序盤": m_common["ece_序盤"], "ECE中盤": m_common["ece_中盤"],
            "ECE終盤": m_common["ece_終盤"],
            "Brier": m_common["brier"], "AUC": m_common["auc"],
        })
        y_p, p_p, lab_p = _pool_phase_results(phase_results, key)
        m_phase = _phase_sliced_metrics(y_p, p_p, lab_p)
        rows.append({
            "手法": f"{method_name}(位相別)",
            "ECE全体": m_phase["ece_全体"],
            "ECE序盤": m_phase["ece_序盤"], "ECE中盤": m_phase["ece_中盤"],
            "ECE終盤": m_phase["ece_終盤"],
            "Brier": m_phase["brier"], "AUC": m_phase["auc"],
        })
    return pd.DataFrame(rows)


def build_answer_table(
    res_all: dict[str, np.ndarray],
    phase_results: dict[str, dict],
    y_all: np.ndarray,
) -> pd.DataFrame:
    """「予測0.8台の実勝率」の直接回答表 (校正前後・位相別) を作る。"""
    rows: list[dict] = []
    sources = [("全体", y_all, res_all)]
    for phase, res in phase_results.items():
        sources.append((phase, res["y"], res))
    for scope_label, y, res in sources:
        for score_label, key in (("raw", "raw"), ("Platt", "platt"), ("Isotonic", "iso")):
            calib = compute_calibration_table(y, res[key])
            for target in ANSWER_PROBS:
                bin_label = f"{target:.1f}-{target + 0.1:.1f}"
                row = calib[calib["bin"] == bin_label]
                if row.empty or int(row.iloc[0]["n"]) == 0:
                    continue
                r = row.iloc[0]
                rows.append({
                    "範囲": scope_label, "手法": score_label, "予測bin": bin_label,
                    "n": int(r["n"]), "平均予測": r["mean_pred"], "実勝率": r["actual_rate"],
                })
    return pd.DataFrame(rows)


def report_distribution(
    res_all: dict[str, np.ndarray], phase_results: dict[str, dict]
) -> pd.DataFrame:
    """予測確率分布 (raw/Platt/Isotonic) を要約し ±100スケール換算も添える。"""
    rows: list[dict] = []
    sources = [("全体", res_all)]
    for phase, res in phase_results.items():
        sources.append((phase, res))
    for scope_label, res in sources:
        for score_label, key in (("raw", "raw"), ("Platt", "platt"), ("Isotonic", "iso")):
            dist = summarize_distribution(res[key])
            score_100 = (res[key] - 0.5) * 200.0
            rows.append({
                "範囲": scope_label, "手法": score_label,
                "mean": dist["mean"], "std": dist["std"],
                "p5": dist["p5"], "p95": dist["p95"],
                "0.2-0.8外(=|score|>=60)割合": dist["frac_in_extreme_0.2_0.8"],
                "score100_p5": float(np.percentile(score_100, 5)),
                "score100_p95": float(np.percentile(score_100, 95)),
                "score100_absmean": float(np.mean(np.abs(score_100))),
            })
    return pd.DataFrame(rows)


# =============================================================================
# メイン
# =============================================================================

def _load_and_build(
    labeled_path: str, max_tdiff: float
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """データ読込〜特徴量構築までをまとめる (main を50行以内に保つための分割)。"""
    df = load_labeled_csv(labeled_path)
    paired = pair_sides_for_win(df, max_tdiff)
    y_all = paired["won_1p"].astype(int).values
    groups_all = paired["video_id_1p"].values
    tsumo_vals = paired["tsumo_1p"].astype(float).values
    indicator_cols = _get_indicator_cols(paired)
    feat_df = build_features(paired, indicator_cols)
    X_all = feat_df.fillna(0.0).values.astype(float)
    print(f"  特徴量数: {X_all.shape[1]}  サンプル数: {len(y_all)}")
    return paired, X_all, y_all, groups_all, tsumo_vals


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を定義・解析する (main を50行以内に保つための分割)。"""
    parser = argparse.ArgumentParser(description="後段校正 (Platt/Isotonic) 実測")
    parser.add_argument(
        "--labeled",
        default="data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv",
    )
    parser.add_argument("--max-tdiff", type=float, default=DEFAULT_MAX_TDIFF)
    parser.add_argument("--fixed-q33", type=float, default=18.0)
    parser.add_argument("--fixed-q67", type=float, default=40.0)
    parser.add_argument("--outer-folds", type=int, default=N_FOLDS)
    parser.add_argument("--inner-folds", type=int, default=N_FOLDS)
    parser.add_argument(
        "--out-dir",
        default="data/verify/win_eval_combined66_2026-07-29/calibration_fit_2026-07-29",
    )
    return parser.parse_args()


def _run_phase_calibrations(
    X_all: np.ndarray,
    y_all: np.ndarray,
    groups_all: np.ndarray,
    phase_labels: np.ndarray,
    outer_folds: int,
    inner_folds: int,
) -> dict[str, dict]:
    """序盤/中盤/終盤ごとに位相別モデルで入れ子校正を実行する (main を50行以内に保つための分割)。"""
    phase_results: dict[str, dict] = {}
    for phase in PHASES:
        mask = phase_labels == phase
        if int(mask.sum()) < MIN_PHASE_N:
            print(f"  {phase}: データ不足 ({int(mask.sum())}) -> skip")
            continue
        res_ph = nested_calibrated_oof(
            X_all[mask], y_all[mask], groups_all[mask],
            outer_folds, inner_folds, phase,
        )
        phase_results[phase] = {"y": y_all[mask], **res_ph}
    return phase_results


def _emit_reports(
    y_all: np.ndarray,
    phase_labels: np.ndarray,
    res_all: dict[str, np.ndarray],
    phase_results: dict[str, dict],
    out_dir: Path,
) -> None:
    """比較表・直接回答表・分布表を出力し CSV 保存する (main を50行以内に保つための分割)。"""
    pd.set_option("display.width", 160)
    fmt = lambda v: f"{v:.4f}"  # noqa: E731

    print("\n=== 6. 比較表 ===")
    metric_df = build_metric_rows(y_all, phase_labels, res_all, phase_results)
    print(metric_df.to_string(index=False, float_format=fmt))
    metric_df.to_csv(out_dir / "calibration_comparison.csv", index=False)

    print("\n=== 7. 直接回答 (予測0.6/0.7/0.8/0.9台の実勝率) ===")
    answer_df = build_answer_table(res_all, phase_results, y_all)
    print(answer_df.to_string(index=False, float_format=fmt))
    answer_df.to_csv(out_dir / "calibration_direct_answers.csv", index=False)

    print("\n=== 8. 予測確率分布 (±100スケール換算は (p-0.5)*200 の例示的変換) ===")
    dist_df = report_distribution(res_all, phase_results)
    print(dist_df.to_string(index=False, float_format=fmt))
    dist_df.to_csv(out_dir / "calibration_distribution.csv", index=False)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[_calibration_fit] labeled={args.labeled}")
    print("\n=== 1-3. データ読み込み・ペアリング・特徴量構築 ===")
    paired, X_all, y_all, groups_all, tsumo_vals = _load_and_build(
        args.labeled, args.max_tdiff
    )
    phase_labels = assign_phase_labels(tsumo_vals, args.fixed_q33, args.fixed_q67)
    print("  位相内訳: " + ", ".join(
        f"{ph}={int((phase_labels == ph).sum())}" for ph in PHASES
    ))

    print("\n=== 4. 全体モデルで入れ子校正 (共通校正器の元データ) ===")
    res_all = nested_calibrated_oof(
        X_all, y_all, groups_all, args.outer_folds, args.inner_folds, "全体"
    )

    print("\n=== 5. 位相別モデルで入れ子校正 (位相別校正器) ===")
    phase_results = _run_phase_calibrations(
        X_all, y_all, groups_all, phase_labels, args.outer_folds, args.inner_folds
    )

    _emit_reports(y_all, phase_labels, res_all, phase_results, out_dir)
    print(f"\n出力先: {out_dir}")
    print("=== 完了 ===")


if __name__ == "__main__":
    main()
