"""#24 打ち合い計測器「三つ巴比較」共通評価ハーネス (2026-08-01)。

案D (学習器) / 修正シミュ / 併用 の3案を同一 held-out イベント集合で比較する
ための共通関数群。本モジュール自体はモデルを学習しない (予測値は呼び出し側
=各案のスクリプトが用意する)。stateless な純関数のみで構成する
(CLAUDE.md 「観測指標は stateless 実装を原則」に準拠、ただし本モジュールは
指標そのものではなく評価ツールなので厳密な stateless 制約の対象外だが
副作用のない設計を踏襲する)。

## 主指標 / 副指標の設計 (userとのすり合わせ済み)
- 主指標: net_ojama_after (連続値) の Spearman 順位相関 (主) + MAE (副)。
- 副指標: taiou_success の二値 AUC。**主指標には昇格させない**
  (「二値判定へ戻る設計矛盾を避ける」-- reference_ojama_damage_nonlinear_2026-07-29
  で「返せるか」の二値判定は設計として誤りと確定済みのため)。

## 測定器事故対策 (測定器事故4件目の教訓)
AUC 計算は sklearn.roc_auc_score をそのまま使い (独自の近似実装を作らない)、
DeLong 検定内部の厳密 AUC 計算がそれと一致することを単体テストで担保する
(tests/test_exchange_meter_eval_harness.py)。

## CV 設計
GroupKFold(n_splits=5, groups=video_id)。**game_idx でのグルーピングは厳禁**
(project_game_idx_desync_bug_2026-07-29 の前科: 1P/2P 独立カウンタでズレる
基幹バグがあり、game_idx を跨いだグルーピングは無汚染性を保証できない)。

## ブートストラップ CI の単位
MAE / 順位相関の差は**動画クラスタ単位**でリサンプルする (video_id を復元抽出
し、そのビデオに属する全イベントをまとめてプールする)。イベント単位で
リサンプルすると同一動画内の相関を無視して検定力を過大評価してしまうため
禁止 (同一動画内のイベントは同じ試合展開・同じ認識品質を共有し独立でない)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

# GroupKFold の既定 fold 数 (video_id 単位)
N_SPLITS_GROUP_KFOLD: int = 5

# reliability diagram のビン数 (0.1刻み)
N_RELIABILITY_BINS: int = 10

# 位相別イベント数がこれ未満なら「参考値・検定力不足」と明記する
# (silent cap 禁止: 数値は必ず出力し、警告ラベルを併記する)
MIN_PHASE_N_FOR_POWER: int = 200

# 動画クラスタ単位ブートストラップの既定リサンプル回数
N_BOOTSTRAP_RESAMPLES: int = 2000

# ブートストラップ乱数シード (再現性確保)
BOOTSTRAP_RANDOM_STATE: int = 42

# ブートストラップ CI の有意水準 (95% CI)
BOOTSTRAP_CI_ALPHA: float = 0.05

# 位相ラベル一覧 (exchange_labels.csv の phase 列は単一文字 序/中/終)
EXCHANGE_PHASES: tuple[str, ...] = ("序", "中", "終")

# 検定力不足を示す警告ラベル文言 (出力に必ずこの文言を付す)
POWER_INSUFFICIENT_LABEL: str = "参考値・検定力不足"


# =============================================================================
# 基本指標 (主指標: 順位相関+MAE、副指標: AUC+Brier)
# =============================================================================

def exact_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """厳密 AUC (sklearn.roc_auc_score のラッパー)。

    測定器事故4件目 (train_board_cnn の 224x224 サブサンプル近似バグ) の
    教訓を踏まえ、独自近似は行わず sklearn の実装をそのまま使う。
    正例/負例のどちらかが 0 件の場合は判定不能として NaN を返す。
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def spearman_and_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """主指標: net_ojama_after の Spearman 順位相関 (主) + MAE (副) を返す。"""
    rho = float(spearmanr(y_true, y_pred).statistic)
    mae = float(mean_absolute_error(y_true, y_pred))
    return rho, mae


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (較正の副指標)。"""
    return float(brier_score_loss(y_true, y_prob))


def phase_power_flag(n_events: int, min_n: int = MIN_PHASE_N_FOR_POWER) -> str:
    """位相別イベント数が閾値未満なら警告ラベルを返す (silent cap 禁止)。"""
    return POWER_INSUFFICIENT_LABEL if n_events < min_n else ""


# =============================================================================
# reliability diagram (較正、10 bin)
# =============================================================================

def compute_reliability_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_RELIABILITY_BINS,
) -> pd.DataFrame:
    """予測確率を等幅ビンに区切り、ビンごとの平均予測・実成功率・件数を返す。

    _calibration_fit_2026-07-29.py の compute_calibration_table と同一設計
    (等幅ビン・右閉区間) だが、当該ファイルはハイフンを含みモジュールとして
    import 不可のため独立実装する (同ファイルは Step0 対象外で無改変)。
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=False), 0, n_bins - 1)
    rows: list[dict] = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        rows.append({
            "bin": f"{bin_edges[b]:.1f}-{bin_edges[b + 1]:.1f}",
            "n": n,
            "mean_pred": float(y_prob[mask].mean()) if n > 0 else float("nan"),
            "actual_rate": float(y_true[mask].mean()) if n > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def plot_reliability_diagrams(
    tables_by_label: dict[str, pd.DataFrame], out_path: Path,
) -> None:
    """複数 (予測器×位相) の reliability table を1枚の図にまとめて保存する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 日本語ラベル (序/中/終 等) が文字化けしないよう Meiryo を優先使用する
    # (feedback_terminal_font_mojibake の教訓、collect_chain_stats.py と同一パターン)。
    meiryo_path = "/mnt/c/Windows/Fonts/meiryo.ttc"
    if Path(meiryo_path).exists():
        font_manager.fontManager.addfont(meiryo_path)
        plt.rcParams["font.family"] = "Meiryo"

    n_panels = len(tables_by_label)
    n_cols = min(3, max(1, n_panels))
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows), squeeze=False)
    for i, (label, table) in enumerate(tables_by_label.items()):
        ax = axes[i // n_cols][i % n_cols]
        _plot_single_reliability(ax, table, label)
    for j in range(n_panels, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_single_reliability(ax, table: pd.DataFrame, label: str) -> None:
    """reliability diagram 1パネル分を描画する (plot_reliability_diagrams の内部処理)。"""
    valid = table["n"] > 0
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.plot(table.loc[valid, "mean_pred"], table.loc[valid, "actual_rate"], "o-", color="C0")
    ax.set_title(label, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("予測確率")
    ax.set_ylabel("実成功率")


# =============================================================================
# GroupKFold (video_id 単位、game_idx でのグルーピングは厳禁)
# =============================================================================

def group_kfold_splits(
    n_samples: int, groups: np.ndarray, n_splits: int = N_SPLITS_GROUP_KFOLD,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold(groups=video_id) の (train_idx, test_idx) リストを返す。

    fold 数は動画本数を超えないようクランプする (動画本数が5未満の開発時
    スモークでも動作するように)。
    """
    n_groups = len(np.unique(groups))
    effective_splits = min(n_splits, max(2, n_groups))
    dummy_x = np.zeros((n_samples, 1))
    gkf = GroupKFold(n_splits=effective_splits)
    return list(gkf.split(dummy_x, groups=groups))


# =============================================================================
# DeLong ペアード検定 (AUC差の有意性、副指標 taiou_success の二値AUC比較用)
# =============================================================================
#
# 実装は Sun & Xu (2014) "Fast Implementation of DeLong's Algorithm" の
# O(N log N) 版 (通称 fast DeLong、共分散行列を midrank で構成する手法)。
# 単体テストで sklearn.roc_auc_score と厳密一致することを確認する
# (測定器事故4件目の教訓: 独自AUC実装は必ず突き合わせる)。

@dataclass(frozen=True)
class DeLongTestResult:
    """DeLong ペアード検定の結果。"""
    auc_a: float
    auc_b: float
    auc_diff: float  # auc_a - auc_b
    z_stat: float
    p_value: float


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """同点を平均ランクとして扱う midrank を計算する (fast DeLong の下請け)。"""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    result = np.empty(n, dtype=np.float64)
    result[order] = ranks
    return result


def _fast_delong(
    predictions_sorted: np.ndarray, n_pos: int,
) -> tuple[np.ndarray, np.ndarray]:
    """複数モデル分の AUC と共分散行列を同時に計算する (fast DeLong 本体)。

    Args:
        predictions_sorted: shape (k_models, n_pos + n_neg)、各行が
            「正例を先頭 n_pos 個・負例を残り」に並べたスコア配列。
        n_pos: 正例数。

    Returns:
        (aucs, cov): aucs は shape (k_models,)、cov は共分散行列
        shape (k_models, k_models)。
    """
    k_models, n_total = predictions_sorted.shape
    n_neg = n_total - n_pos
    tx = np.empty((k_models, n_pos))
    ty = np.empty((k_models, n_neg))
    tz = np.empty((k_models, n_total))
    for r in range(k_models):
        tx[r, :] = _compute_midrank(predictions_sorted[r, :n_pos])
        ty[r, :] = _compute_midrank(predictions_sorted[r, n_pos:])
        tz[r, :] = _compute_midrank(predictions_sorted[r, :])
    aucs = tz[:, :n_pos].sum(axis=1) / n_pos / n_neg - float(n_pos + 1) / 2.0 / n_neg
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    cov = np.cov(v01) / n_pos + np.cov(v10) / n_neg
    return aucs, np.atleast_2d(cov)


def delong_paired_test(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
) -> DeLongTestResult:
    """同一 held-out 集合上の2予測器の AUC 差を DeLong ペアード検定で評価する。

    正例/負例のいずれかが0件の場合は判定不能として NaN を返す。
    """
    if len(np.unique(y_true)) < 2:
        return DeLongTestResult(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    pos_mask = y_true == 1
    order = np.concatenate([np.where(pos_mask)[0], np.where(~pos_mask)[0]])
    n_pos = int(pos_mask.sum())
    stacked = np.vstack([score_a[order], score_b[order]])
    aucs, cov = _fast_delong(stacked, n_pos)
    diff = float(aucs[0] - aucs[1])
    var_diff = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var_diff <= 0.0:
        z_stat, p_value = float("nan"), float("nan")
    else:
        z_stat = diff / float(np.sqrt(var_diff))
        p_value = float(2.0 * (1.0 - _std_normal_cdf(abs(z_stat))))
    return DeLongTestResult(float(aucs[0]), float(aucs[1]), diff, z_stat, p_value)


def _std_normal_cdf(z: float) -> float:
    """標準正規分布の累積分布関数 (math.erf ベースの最小実装)。"""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# =============================================================================
# ブートストラップ CI (動画クラスタ単位、MAE/順位相関の差の検定用)
# =============================================================================

@dataclass(frozen=True)
class BootstrapCI:
    """動画クラスタ単位ブートストラップの点推定 + 信頼区間。"""
    point: float
    ci_low: float
    ci_high: float
    n_resamples: int


def _video_index_map(video_ids: np.ndarray) -> dict:
    """動画IDごとの行インデックス配列を事前計算する (リサンプル高速化用)。"""
    return {v: np.where(video_ids == v)[0] for v in np.unique(video_ids)}


def _resample_indices(
    video_to_idx: dict, unique_videos: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """動画IDを復元抽出し、対応する全イベント行インデックスを連結して返す。"""
    sampled = rng.choice(unique_videos, size=len(unique_videos), replace=True)
    return np.concatenate([video_to_idx[v] for v in sampled])


def bootstrap_ci_by_video(
    metric_fn: Callable[..., float],
    video_ids: np.ndarray,
    arrays: dict[str, np.ndarray],
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    random_state: int = BOOTSTRAP_RANDOM_STATE,
    ci_alpha: float = BOOTSTRAP_CI_ALPHA,
) -> BootstrapCI:
    """動画クラスタ単位ブートストラップで metric_fn(**arrays) の点推定+CIを返す。

    video_id を復元抽出し、そのビデオに属する全イベントをまとめてプールする
    (イベント単位リサンプルは同一動画内相関で検定力を過大評価するため禁止)。
    metric_fn は arrays の各キーをキーワード引数として受け取る関数とする。
    """
    unique_videos = np.unique(video_ids)
    video_to_idx = _video_index_map(video_ids)
    rng = np.random.default_rng(random_state)
    point = float(metric_fn(**arrays))
    boot_vals = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = _resample_indices(video_to_idx, unique_videos, rng)
        boot_vals[b] = metric_fn(**{k: v[idx] for k, v in arrays.items()})
    lo, hi = np.percentile(boot_vals, [100 * ci_alpha / 2, 100 * (1 - ci_alpha / 2)])
    return BootstrapCI(point, float(lo), float(hi), n_resamples)


def bootstrap_diff_ci_by_video(
    metric_fn: Callable[..., float],
    video_ids: np.ndarray,
    arrays_a: dict[str, np.ndarray],
    arrays_b: dict[str, np.ndarray],
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    random_state: int = BOOTSTRAP_RANDOM_STATE,
    ci_alpha: float = BOOTSTRAP_CI_ALPHA,
) -> BootstrapCI:
    """2予測器の metric 差 (A-B) を動画クラスタ単位ペアードブートストラップで評価する。

    同一リサンプル (同じ動画集合) を A/B 両方に適用するペアード設計
    (両者とも同一 held-out イベント集合上の予測なので y_true は共通)。
    """
    unique_videos = np.unique(video_ids)
    video_to_idx = _video_index_map(video_ids)
    rng = np.random.default_rng(random_state)
    point = float(metric_fn(**arrays_a)) - float(metric_fn(**arrays_b))
    diffs = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = _resample_indices(video_to_idx, unique_videos, rng)
        val_a = metric_fn(**{k: v[idx] for k, v in arrays_a.items()})
        val_b = metric_fn(**{k: v[idx] for k, v in arrays_b.items()})
        diffs[b] = val_a - val_b
    lo, hi = np.percentile(diffs, [100 * ci_alpha / 2, 100 * (1 - ci_alpha / 2)])
    return BootstrapCI(point, float(lo), float(hi), n_resamples)


# =============================================================================
# 高レベル API: 三つ巴比較 (案D / 修正シミュ / 併用 を同一 held-out で比較)
# =============================================================================

@dataclass(frozen=True)
class PredictorPredictions:
    """1予測器分の held-out 予測値 (df と行順を揃えること)。"""
    name: str
    prob_taiou_success: np.ndarray  # 副指標 (二値AUC) 用
    net_ojama_after_pred: np.ndarray  # 主指標 (連続値 Spearman/MAE) 用


def _metric_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """bootstrap_ci_by_video に渡す MAE ラッパー (キーワード引数統一のため)。"""
    return float(mean_absolute_error(y_true, y_pred))


def _metric_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """bootstrap_ci_by_video に渡す Spearman ラッパー。"""
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def _metric_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """bootstrap_ci_by_video に渡す AUC ラッパー。"""
    return exact_auc(y_true, y_pred)


def _slice_masks(df: pd.DataFrame, phase_col: str) -> dict[str, np.ndarray]:
    """「全体」+ 位相別のブールマスク辞書を返す (compare_predictors の内部処理)。"""
    masks: dict[str, np.ndarray] = {"全体": np.ones(len(df), dtype=bool)}
    for ph in EXCHANGE_PHASES:
        masks[ph] = (df[phase_col].values == ph)
    return masks


def _eval_scope_row(
    scope: str, name: str, mask: np.ndarray,
    y_taiou: np.ndarray, y_net: np.ndarray, pred: PredictorPredictions,
    video_ids: np.ndarray, n_resamples: int,
) -> dict:
    """1 (範囲×予測器) 分の指標行を計算する (compare_predictors の分割、50行制約対応)。"""
    n = int(mask.sum())
    rho_ci = bootstrap_ci_by_video(
        _metric_rho, video_ids[mask],
        {"y_true": y_net[mask], "y_pred": pred.net_ojama_after_pred[mask]},
        n_resamples=n_resamples,
    )
    mae_ci = bootstrap_ci_by_video(
        _metric_mae, video_ids[mask],
        {"y_true": y_net[mask], "y_pred": pred.net_ojama_after_pred[mask]},
        n_resamples=n_resamples,
    )
    auc_val = exact_auc(y_taiou[mask], pred.prob_taiou_success[mask])
    brier_val = brier(y_taiou[mask], pred.prob_taiou_success[mask]) if n > 0 else float("nan")
    return {
        "範囲": scope, "予測器": name, "n": n,
        "spearman_rho(主)": rho_ci.point, "rho_CI_low": rho_ci.ci_low, "rho_CI_high": rho_ci.ci_high,
        "MAE(主副)": mae_ci.point, "MAE_CI_low": mae_ci.ci_low, "MAE_CI_high": mae_ci.ci_high,
        "AUC(副指標)": auc_val, "Brier(副指標)": brier_val,
        "備考": phase_power_flag(n) if scope != "全体" else "",
    }


def build_scope_comparison_table(
    df: pd.DataFrame,
    predictors: list[PredictorPredictions],
    video_col: str = "video_id",
    phase_col: str = "phase",
    y_taiou_col: str = "taiou_success",
    y_net_col: str = "net_ojama_after",
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
) -> pd.DataFrame:
    """全体+位相別の (範囲×予測器) 指標一覧表を作る (主指標/副指標を1表に併記)。"""
    video_ids = df[video_col].values
    y_taiou = df[y_taiou_col].values
    y_net = df[y_net_col].values
    masks = _slice_masks(df, phase_col)
    rows = [
        _eval_scope_row(scope, pred.name, mask, y_taiou, y_net, pred, video_ids, n_resamples)
        for scope, mask in masks.items()
        for pred in predictors
    ]
    return pd.DataFrame(rows)


def build_pairwise_delong_table(
    df: pd.DataFrame,
    predictors: list[PredictorPredictions],
    phase_col: str = "phase",
    y_taiou_col: str = "taiou_success",
) -> pd.DataFrame:
    """予測器の全ペアについて taiou_success の DeLong AUC 差検定を行う (副指標)。"""
    masks = _slice_masks(df, phase_col)
    y_taiou = df[y_taiou_col].values
    rows: list[dict] = []
    for scope, mask in masks.items():
        for i in range(len(predictors)):
            for j in range(i + 1, len(predictors)):
                a, b = predictors[i], predictors[j]
                res = delong_paired_test(
                    y_taiou[mask], a.prob_taiou_success[mask], b.prob_taiou_success[mask],
                )
                rows.append({
                    "範囲": scope, "予測器A": a.name, "予測器B": b.name,
                    "AUC_A": res.auc_a, "AUC_B": res.auc_b, "差(A-B)": res.auc_diff,
                    "z": res.z_stat, "p値": res.p_value,
                    "備考": phase_power_flag(int(mask.sum())) if scope != "全体" else "",
                })
    return pd.DataFrame(rows)


def build_pairwise_bootstrap_table(
    df: pd.DataFrame,
    predictors: list[PredictorPredictions],
    video_col: str = "video_id",
    phase_col: str = "phase",
    y_net_col: str = "net_ojama_after",
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
) -> pd.DataFrame:
    """予測器の全ペアについて主指標 (Spearman/MAE) の差を動画単位ブートストラップで検定する。"""
    masks = _slice_masks(df, phase_col)
    video_ids = df[video_col].values
    y_net = df[y_net_col].values
    rows: list[dict] = []
    for scope, mask in masks.items():
        for i in range(len(predictors)):
            for j in range(i + 1, len(predictors)):
                a, b = predictors[i], predictors[j]
                arrays_a = {"y_true": y_net[mask], "y_pred": a.net_ojama_after_pred[mask]}
                arrays_b = {"y_true": y_net[mask], "y_pred": b.net_ojama_after_pred[mask]}
                rho_diff = bootstrap_diff_ci_by_video(
                    _metric_rho, video_ids[mask], arrays_a, arrays_b, n_resamples=n_resamples,
                )
                mae_diff = bootstrap_diff_ci_by_video(
                    _metric_mae, video_ids[mask], arrays_a, arrays_b, n_resamples=n_resamples,
                )
                rows.append({
                    "範囲": scope, "予測器A": a.name, "予測器B": b.name,
                    "rho差(A-B)": rho_diff.point, "rho差CI_low": rho_diff.ci_low, "rho差CI_high": rho_diff.ci_high,
                    "MAE差(A-B)": mae_diff.point, "MAE差CI_low": mae_diff.ci_low, "MAE差CI_high": mae_diff.ci_high,
                    "備考": phase_power_flag(int(mask.sum())) if scope != "全体" else "",
                })
    return pd.DataFrame(rows)


def build_reliability_tables(
    df: pd.DataFrame,
    predictors: list[PredictorPredictions],
    phase_col: str = "phase",
    y_taiou_col: str = "taiou_success",
    n_bins: int = N_RELIABILITY_BINS,
) -> dict[str, pd.DataFrame]:
    """予測器×(全体+位相別) の reliability table 辞書を返す (副指標較正確認用)。"""
    masks = _slice_masks(df, phase_col)
    y_taiou = df[y_taiou_col].values
    tables: dict[str, pd.DataFrame] = {}
    for pred in predictors:
        for scope, mask in masks.items():
            label = f"{pred.name} / {scope}"
            tables[label] = compute_reliability_table(
                y_taiou[mask], pred.prob_taiou_success[mask], n_bins,
            )
    return tables


def _df_to_markdown(df: pd.DataFrame) -> str:
    """tabulate 非依存の最小 markdown テーブル変換 (環境に tabulate 未導入のため自前実装)。"""
    if df.empty:
        return "(データなし)\n"
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    )
    return f"{header}\n{sep}\n{body}\n"


def _write_markdown_report(
    out_dir: Path,
    scope_table: pd.DataFrame,
    delong_table: pd.DataFrame,
    bootstrap_table: pd.DataFrame,
) -> Path:
    """比較表3種を1枚の markdown レポートにまとめて書き出す。"""
    lines = [
        "# #24 打ち合い計測器 三つ巴比較レポート",
        "",
        "主指標 = net_ojama_after の Spearman順位相関(主)+MAE(副)。",
        "副指標 = taiou_success の二値AUC (主指標には昇格させない)。",
        f"位相別イベント数が{MIN_PHASE_N_FOR_POWER}件未満の層は「{POWER_INSUFFICIENT_LABEL}」と明記。",
        "",
        "## 範囲別 (全体/序/中/終) × 予測器 指標一覧",
        _df_to_markdown(scope_table),
        "## ペアード DeLong 検定 (taiou_success AUC差、副指標)",
        _df_to_markdown(delong_table),
        "## ペアード ブートストラップ CI (net_ojama_after の rho/MAE差、主指標、動画クラスタ単位)",
        _df_to_markdown(bootstrap_table),
    ]
    report_path = out_dir / "comparison_report.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def compare_predictors(
    df: pd.DataFrame,
    predictors: list[PredictorPredictions],
    out_dir: Path,
    video_col: str = "video_id",
    phase_col: str = "phase",
    y_taiou_col: str = "taiou_success",
    y_net_col: str = "net_ojama_after",
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
) -> dict[str, pd.DataFrame]:
    """三つ巴比較の全処理 (表3種+reliability図+markdown) を実行するトップレベルAPI。

    predictors は1件以上であれば良い (1件なら単一予測器の評価レポート、
    2〜3件なら案D/修正シミュ/併用のペアード比較になる)。
    held-out 集合は呼び出し側で同一に揃えること (df の行順=各 predictor の
    予測配列の行順が一致している前提)。
    """
    scope_table = build_scope_comparison_table(
        df, predictors, video_col, phase_col, y_taiou_col, y_net_col, n_resamples,
    )
    delong_table = build_pairwise_delong_table(df, predictors, phase_col, y_taiou_col)
    bootstrap_table = build_pairwise_bootstrap_table(
        df, predictors, video_col, phase_col, y_net_col, n_resamples,
    )
    reliability_tables = build_reliability_tables(df, predictors, phase_col, y_taiou_col)
    plot_reliability_diagrams(reliability_tables, out_dir / "reliability_diagrams.png")
    _write_markdown_report(out_dir, scope_table, delong_table, bootstrap_table)
    return {
        "scope": scope_table, "delong_pairs": delong_table,
        "bootstrap_pairs": bootstrap_table,
    }
