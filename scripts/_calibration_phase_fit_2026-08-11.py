"""表示勝率較正の位相別 (序盤/中盤/終盤) 再測定 + 位相別 Platt/Isotonic 比較。

## 背景 (ロードマップ Phase1-2)
memory `project_calibration_overconfident_2026-07-29` で「80%有利表示の実勝率が
64%」「終盤 ECE が最悪 (0.056)」と判明し、全位相共通 Platt scaling を導入した
(scripts/fit_platt_calibration.py)。しかしその後 B-1 (対称化ミラー符号バグ修正)
+ B-2 (進行度列 match_progress 追加、PR #22) で表示用モデル (tier1、
scripts/visualize_advantage_overlay.py の _train_model) 自体が変わったため、
較正は測り直しが必要 (user指示)。

## 本スクリプトの立ち位置
scripts/_calibration_fit_2026-07-29.py と同じ「入れ子 (nested) GroupKFold」で
リークを防ぎつつ校正手法を比較する設計を踏襲するが、以下の2点を変更する:
  1. 位相の決め方: 旧スクリプトは tsumo (手数) の三分位だったが、本スクリプトは
     RT でも使える match_progress (両者の盤面ぷよ総数の平均、B-2 で導入) の
     均等3分割 (src.probability_calibration.PHASE_BOUND_EARLY/LATE) を使う。
  2. 較正対象モデル: 旧スクリプトは model_indicator_win.py の全指標 HistGBC
     (combined66) だったが、本スクリプトは表示に実際使われる tier1 モデル
     (scripts.visualize_advantage_overlay._train_model と同じ特徴量構築
     + 対称化ミラー標本) を re-implement して評価する。

## 制約
- src/ 配下は既存関数を追加のみ (このスクリプトは新規ファイルなので既存の
  probability_calibration.py 関数を import して使うだけで変更しない)。
- scripts/model_indicator_win.py, scripts/visualize_advantage_overlay.py は
  無改変 (関数を import して再利用するのみ、このコミットで generate() 側に
  加えた配線とは独立)。
- ファイル名にハイフンを含む scripts/_calibration_fit_2026-07-29.py は
  import 不可 (import文はハイフンを識別子として解釈できない) のため、
  小さい Platt/Isotonic ヘルパー関数は同一ロジックをここに複製する。

## 使い方
    nice -n 19 python -m scripts._calibration_phase_fit_2026-08-11 \
        --labeled data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv \
        --out-dir data/verify/calibration_phase_2026-08-11
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.probability_calibration import (  # noqa: E402
    PHASE_BOUND_EARLY, PHASE_BOUND_LATE, PHASE_NAMES,
    PlattCalibrationParams, apply_platt_calibration, load_platt_calibration,
    phase_label_for_progress,
)
from scripts.model_indicator_win import (  # noqa: E402
    GBC_PARAMS, build_features, load_labeled_csv, pair_sides_for_win,
)
import scripts.visualize_advantage_overlay as vao  # noqa: E402

# =============================================================================
# 定数
# =============================================================================

N_CALIBRATION_BINS: int = 10                # 校正ビン幅 (0.1刻み)
LOGIT_EPS: float = 1e-6                     # logit変換の発散防止クリップ幅
MIN_CALIB_FIT_N: int = 200                  # 校正器を学習可能とみなす最小件数
RANDOM_STATE: int = 42                      # 校正器の乱数シード
MAX_TDIFF: float = 1.0                      # ペアリング最大時刻差 (_train_model と同一)
DEFAULT_OUTER_FOLDS: int = 5
DEFAULT_INNER_FOLDS: int = 5
ANSWER_PROBS: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)  # 直接回答表の対象bin
DEFAULT_LABELED = (
    "data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv"
)
DEFAULT_PROD_PLATT_PATH = Path("data/indicators_v2/platt_calibration.json")


# =============================================================================
# データ読込 + tier1 特徴量構築 (_train_model と同一構成の再現)
# =============================================================================

def _load_tier1_dataset(labeled_path: str) -> dict[str, np.ndarray]:
    """combined66 CSV から tier1 モデルと同じ特徴量 X・ラベル y 等を作る。"""
    df = load_labeled_csv(labeled_path)
    feat_cols = vao._resolve_features(df)
    paired = pair_sides_for_win(df, MAX_TDIFF)
    feat = build_features(paired, feat_cols)
    feat, cols = vao._add_interaction_columns(feat, feat_cols, paired)
    X = feat[cols].fillna(0.0).values.astype(float)
    y = paired["won_1p"].astype(int).values
    groups = paired["video_id_1p"].values
    progress = vao._match_progress_from_totals(
        paired["board_puyo_total_1p"], paired["board_puyo_total_2p"])
    mirror_sign = vao._mirror_sign(cols)
    print(f"  tier1特徴量数: {len(cols)}  サンプル数: {len(y)}  動画数: {len(np.unique(groups))}")
    return {"X": X, "y": y, "groups": groups, "progress": np.asarray(progress),
            "cols": cols, "mirror_sign": mirror_sign}


def _fit_tier1_gbc(X_tr: np.ndarray, y_tr: np.ndarray,
                    mirror_sign: np.ndarray) -> HistGradientBoostingClassifier:
    """対称化ミラー標本込みで tier1 HistGBC を学習する (_train_model と同一手法)。"""
    X_sym = np.vstack([X_tr, X_tr * mirror_sign])
    y_sym = np.concatenate([y_tr, 1 - y_tr])
    model = HistGradientBoostingClassifier(**GBC_PARAMS)
    model.fit(X_sym, y_sym)
    return model


def assign_phase_labels(progress: np.ndarray) -> np.ndarray:
    """match_progress配列から位相ラベル配列を返す (src.probability_calibration と同一境界)。"""
    return np.array(
        [phase_label_for_progress(float(p), PHASE_BOUND_EARLY, PHASE_BOUND_LATE)
         for p in progress], dtype=object,
    )


# =============================================================================
# Platt / Isotonic 校正器 (_calibration_fit_2026-07-29.py と同一ロジックの複製)
# =============================================================================

def _logit(p: np.ndarray) -> np.ndarray:
    p_clip = np.clip(p, LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(p_clip / (1.0 - p_clip))


def fit_platt(raw_p: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Platt scaling: logit(raw_p) を1特徴量とする1次元ロジスティック回帰。"""
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(_logit(raw_p).reshape(-1, 1), y)
    return lr


def apply_platt(model: LogisticRegression, raw_p: np.ndarray) -> np.ndarray:
    return model.predict_proba(_logit(raw_p).reshape(-1, 1))[:, 1]


def fit_isotonic(raw_p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_p, y)
    return iso


def apply_isotonic(model: IsotonicRegression, raw_p: np.ndarray) -> np.ndarray:
    return model.predict(raw_p)


def apply_current_production_platt(
    raw_p: np.ndarray, prod_params: PlattCalibrationParams | None,
) -> np.ndarray:
    """現行本番 Platt (data/indicators_v2/platt_calibration.json) を近似適用する。

    本番運用で今まさに使われている「全位相共通・近似適用」構成の実力を
    比較の基準として併記するためのもの (未生成なら raw をそのまま返す)。
    """
    if prod_params is None:
        return raw_p.copy()
    return np.array([apply_platt_calibration(float(p), prod_params) for p in raw_p])


# =============================================================================
# 入れ子 (nested) GroupKFold -- tier1版 (対称化込み)
# =============================================================================

def _effective_folds(n_folds: int, n_groups: int) -> int:
    """GroupKFold の fold 数を実際の video 数に収まるよう丸める。"""
    return min(n_folds, max(2, n_groups))


def _run_oof_tier1(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    n_folds: int, mirror_sign: np.ndarray) -> np.ndarray:
    """対称化込み tier1 モデルで GroupKFold OOF 予測確率 (P(y=1)) を返す。"""
    oof = np.full(len(y), np.nan)
    eff = _effective_folds(n_folds, len(np.unique(groups)))
    for tr_idx, te_idx in GroupKFold(n_splits=eff).split(X, y, groups=groups):
        model = _fit_tier1_gbc(X[tr_idx], y[tr_idx], mirror_sign)
        oof[te_idx] = model.predict_proba(X[te_idx])[:, 1]
    return oof


def _fit_common_calibrators(
    inner_p: np.ndarray, inner_y: np.ndarray,
) -> tuple[LogisticRegression, IsotonicRegression] | None:
    """inner-OOF から全位相共通の Platt/Isotonic を学習する (データ不足なら None)。"""
    if len(inner_p) < MIN_CALIB_FIT_N or len(np.unique(inner_y)) < 2:
        return None
    return fit_platt(inner_p, inner_y), fit_isotonic(inner_p, inner_y)


def _fit_phase_calibrators(
    inner_p: np.ndarray, inner_y: np.ndarray, inner_phase: np.ndarray,
) -> dict[str, tuple[LogisticRegression, IsotonicRegression]]:
    """位相ごとに独立な Platt/Isotonic を学習する (データ不足な位相はスキップ)。"""
    models: dict[str, tuple[LogisticRegression, IsotonicRegression]] = {}
    for phase in PHASE_NAMES:
        mask = inner_phase == phase
        fitted = _fit_common_calibrators(inner_p[mask], inner_y[mask])
        if fitted is not None:
            models[phase] = fitted
    return models


def _apply_phase_calibrators(
    raw_te: np.ndarray, te_phase: np.ndarray,
    phase_models: dict[str, tuple[LogisticRegression, IsotonicRegression]],
) -> tuple[np.ndarray, np.ndarray]:
    """位相別モデルを各テスト行の位相に応じて適用する (未学習位相は raw のまま)。"""
    platt_out, iso_out = raw_te.copy(), raw_te.copy()
    for phase, (platt_model, iso_model) in phase_models.items():
        mask = te_phase == phase
        if mask.sum() == 0:
            continue
        platt_out[mask] = apply_platt(platt_model, raw_te[mask])
        iso_out[mask] = apply_isotonic(iso_model, raw_te[mask])
    return platt_out, iso_out


def _run_one_outer_fold(
    data: dict[str, np.ndarray], tr_idx: np.ndarray, te_idx: np.ndarray,
    inner_folds: int, prod_params: PlattCalibrationParams | None,
) -> dict[str, np.ndarray]:
    """1 outer fold 分の raw/共通/位相別 校正済み予測を計算する (リークなし)。"""
    X, y, groups, progress = data["X"], data["y"], data["groups"], data["progress"]
    ms = data["mirror_sign"]
    X_tr, y_tr, g_tr = X[tr_idx], y[tr_idx], groups[tr_idx]

    final_model = _fit_tier1_gbc(X_tr, y_tr, ms)
    raw_te = final_model.predict_proba(X[te_idx])[:, 1]

    eff_inner = _effective_folds(inner_folds, len(np.unique(g_tr)))
    inner_oof = _run_oof_tier1(X_tr, y_tr, g_tr, eff_inner, ms)
    inner_valid = ~np.isnan(inner_oof)
    inner_p, inner_y = inner_oof[inner_valid], y_tr[inner_valid]
    inner_phase = assign_phase_labels(progress[tr_idx][inner_valid])
    te_phase = assign_phase_labels(progress[te_idx])

    common = _fit_common_calibrators(inner_p, inner_y)
    platt_common = raw_te.copy() if common is None else apply_platt(common[0], raw_te)
    iso_common = raw_te.copy() if common is None else apply_isotonic(common[1], raw_te)
    phase_models = _fit_phase_calibrators(inner_p, inner_y, inner_phase)
    platt_phase, iso_phase = _apply_phase_calibrators(raw_te, te_phase, phase_models)

    return {
        "raw": raw_te, "platt_common": platt_common, "iso_common": iso_common,
        "platt_phase": platt_phase, "iso_phase": iso_phase,
        "prod_platt_approx": apply_current_production_platt(raw_te, prod_params),
        "phase_label": te_phase,
    }


def nested_calibrated_oof_tier1(
    data: dict[str, np.ndarray], outer_folds: int, inner_folds: int,
    prod_params: PlattCalibrationParams | None,
) -> dict[str, np.ndarray]:
    """入れ子 GroupKFold で tier1 の raw/各種校正 OOF 予測を返す (リークなし)。"""
    n = len(data["y"])
    keys = ("raw", "platt_common", "iso_common", "platt_phase", "iso_phase",
            "prod_platt_approx")
    out: dict[str, np.ndarray] = {k: np.full(n, np.nan) for k in keys}
    phase_label = np.empty(n, dtype=object)
    eff_outer = _effective_folds(outer_folds, len(np.unique(data["groups"])))
    t0 = time.time()
    splitter = GroupKFold(n_splits=eff_outer).split(
        data["X"], data["y"], groups=data["groups"])
    for fold_idx, (tr_idx, te_idx) in enumerate(splitter):
        res = _run_one_outer_fold(data, tr_idx, te_idx, inner_folds, prod_params)
        for k in keys:
            out[k][te_idx] = res[k]
        phase_label[te_idx] = res["phase_label"]
        print(f"    fold {fold_idx + 1}/{eff_outer} 完了"
              f" (train={len(tr_idx)} test={len(te_idx)}, 累計 {time.time() - t0:.1f}秒)")
    out["phase_label"] = phase_label
    return out


# =============================================================================
# 校正テーブル・ECE・レポート
# =============================================================================

def compute_calibration_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """予測確率を0.1幅ビンに区切り、ビンごとの平均予測確率・実勝率・件数を返す。"""
    bin_edges = np.linspace(0.0, 1.0, N_CALIBRATION_BINS + 1)
    bin_idx = np.clip(
        np.digitize(p, bin_edges[1:-1], right=False), 0, N_CALIBRATION_BINS - 1)
    rows: list[dict] = []
    for b in range(N_CALIBRATION_BINS):
        mask = bin_idx == b
        n = int(mask.sum())
        rows.append({
            "bin": f"{bin_edges[b]:.1f}-{bin_edges[b + 1]:.1f}", "n": n,
            "mean_pred": float(p[mask].mean()) if n > 0 else float("nan"),
            "actual_rate": float(y[mask].mean()) if n > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def compute_ece(y: np.ndarray, p: np.ndarray) -> float:
    """期待校正誤差 (ECE) = Σ (n_bin/n) * |実勝率 - 平均予測|。"""
    table = compute_calibration_table(y, p)
    n_total = len(y)
    ece = 0.0
    for _, row in table.iterrows():
        if row["n"] == 0:
            continue
        ece += (row["n"] / n_total) * abs(row["actual_rate"] - row["mean_pred"])
    return float(ece)


def _auc_safe(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def _phase_sliced_metrics(y: np.ndarray, p: np.ndarray, phase: np.ndarray) -> dict[str, float]:
    """1本のスコア配列を ECE全体/序盤/中盤/終盤 + 位相内AUC(最小/最大) + Brier + 全体AUC に集計する。

    位相内AUC (auc_min_phase/auc_max_phase) は「位相ごとの校正器が独立でも
    各位相の内部順序 (=判別力) は保たれているか」を見る無悪化ゲートの本体。
    全体AUC (auc_pooled) は位相をまたいだプール順序で、位相別校正では
    理論上ズレうる (位相間の順序保証はない設計、詳細は本ファイル docstring)。
    """
    out: dict[str, float] = {"ece_全体": compute_ece(y, p), "brier": float(brier_score_loss(y, p)),
                              "auc_pooled": _auc_safe(y, p)}
    phase_aucs: list[float] = []
    for ph in PHASE_NAMES:
        mask = phase == ph
        if mask.sum() == 0:
            out[f"ece_{ph}"] = float("nan")
            continue
        out[f"ece_{ph}"] = compute_ece(y[mask], p[mask])
        auc_ph = _auc_safe(y[mask], p[mask])
        out[f"auc_{ph}"] = auc_ph
        if not np.isnan(auc_ph):
            phase_aucs.append(auc_ph)
    out["auc_min_phase"] = min(phase_aucs) if phase_aucs else float("nan")
    out["auc_max_phase"] = max(phase_aucs) if phase_aucs else float("nan")
    return out


def build_metric_rows(y: np.ndarray, phase: np.ndarray, res: dict[str, np.ndarray]) -> pd.DataFrame:
    """各手法の ECE全体/序盤/中盤/終盤/Brier/AUC(プール・位相内min-max) 比較表を作る。"""
    method_labels = (
        ("校正なし(raw、B-1+B-2後のtier1)", "raw"),
        ("現行Platt(本番approx適用、旧モデル学習)", "prod_platt_approx"),
        ("Platt(全位相共通、tier1で再学習)", "platt_common"),
        ("Isotonic(全位相共通、tier1で再学習)", "iso_common"),
        ("Platt(位相別)", "platt_phase"),
        ("Isotonic(位相別)", "iso_phase"),
    )
    rows: list[dict] = []
    for label, key in method_labels:
        m = _phase_sliced_metrics(y, res[key], phase)
        rows.append({
            "手法": label, "ECE全体": m["ece_全体"],
            "ECE序盤": m["ece_序盤"], "ECE中盤": m["ece_中盤"], "ECE終盤": m["ece_終盤"],
            "Brier": m["brier"], "AUC(プール)": m["auc_pooled"],
            "AUC(位相内min)": m["auc_min_phase"], "AUC(位相内max)": m["auc_max_phase"],
        })
    return pd.DataFrame(rows)


def build_answer_table(y: np.ndarray, phase: np.ndarray, res: dict[str, np.ndarray]) -> pd.DataFrame:
    """「予測0.6/0.7/0.8/0.9台の実勝率」を全体+位相別+手法別に直接回答する表。"""
    rows: list[dict] = []
    scopes: list[tuple[str, np.ndarray]] = [("全体", np.ones(len(y), dtype=bool))]
    for ph in PHASE_NAMES:
        scopes.append((ph, phase == ph))
    for scope_label, mask in scopes:
        for method_label, key in (("raw", "raw"), ("現行Platt(approx)", "prod_platt_approx"),
                                   ("Platt(位相別)", "platt_phase"), ("Isotonic(位相別)", "iso_phase")):
            calib = compute_calibration_table(y[mask], res[key][mask])
            for target in ANSWER_PROBS:
                bin_label = f"{target:.1f}-{target + 0.1:.1f}"
                row = calib[calib["bin"] == bin_label]
                if row.empty or int(row.iloc[0]["n"]) == 0:
                    continue
                r = row.iloc[0]
                rows.append({"範囲": scope_label, "手法": method_label, "予測bin": bin_label,
                             "n": int(r["n"]), "平均予測": r["mean_pred"], "実勝率": r["actual_rate"]})
    return pd.DataFrame(rows)


# =============================================================================
# メイン
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を定義・解析する (main を50行以内に保つための分割)。"""
    parser = argparse.ArgumentParser(description="位相別 (match_progress) 較正の再測定")
    parser.add_argument("--labeled", default=DEFAULT_LABELED)
    parser.add_argument("--outer-folds", type=int, default=DEFAULT_OUTER_FOLDS)
    parser.add_argument("--inner-folds", type=int, default=DEFAULT_INNER_FOLDS)
    parser.add_argument("--prod-platt", default=str(DEFAULT_PROD_PLATT_PATH),
                         help="現行本番Platt(全位相共通)のJSONパス (比較の基準用)")
    parser.add_argument("--out-dir",
                         default="data/verify/calibration_phase_2026-08-11")
    return parser.parse_args()


def _load_prod_platt(path_str: str) -> PlattCalibrationParams | None:
    """現行本番Plattを読む (未生成なら警告のみでNone、比較行はrawと同一になる)。"""
    return load_platt_calibration(Path(path_str), required=False)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[_calibration_phase_fit] labeled={args.labeled}")
    print("\n=== 1. データ読込 + tier1特徴量構築 (B-1+B-2後のモデルを再現) ===")
    data = _load_tier1_dataset(args.labeled)
    phase_all = assign_phase_labels(data["progress"])
    print("  位相内訳(全体): " + ", ".join(
        f"{ph}={int((phase_all == ph).sum())}" for ph in PHASE_NAMES))

    prod_params = _load_prod_platt(args.prod_platt)
    print(f"  現行本番Platt: {'読込成功 a=%.4f b=%.4f' % (prod_params.a, prod_params.b) if prod_params else '未生成のためraw同値で比較'}")

    print("\n=== 2. 入れ子GroupKFoldで raw/各種校正のOOF予測を計算 ===")
    res = nested_calibrated_oof_tier1(
        data, args.outer_folds, args.inner_folds, prod_params)
    phase_oof = res.pop("phase_label")

    print("\n=== 3. 比較表 (ECE全体/位相別 + Brier + AUC) ===")
    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    metric_df = build_metric_rows(data["y"], phase_oof, res)
    print(metric_df.to_string(index=False, float_format=fmt))
    metric_df.to_csv(out_dir / "calibration_phase_comparison.csv", index=False)

    print("\n=== 4. 直接回答 (予測0.6/0.7/0.8/0.9台の実勝率、全体+位相別) ===")
    answer_df = build_answer_table(data["y"], phase_oof, res)
    print(answer_df.to_string(index=False, float_format=fmt))
    answer_df.to_csv(out_dir / "calibration_phase_direct_answers.csv", index=False)

    print(f"\n出力先: {out_dir}")
    print("=== 完了 ===")


if __name__ == "__main__":
    main()
