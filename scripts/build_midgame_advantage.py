"""中盤有利不利スコア 2段構成モデル (B-1/B-2/B-3)。

## 目的
exchange_labels.csv(発火イベント 7960 件)を使って以下を実行:
  B-1: 近い地平アウトカムの条件付き勝率 / ΔWinProb 表(位相別)
  B-2: 盤面特徴 → 段1予測(P(taiou_success)/P(opp_buried))を合成した
       対称な中盤有利不利スコア adv_mid を構築
  B-3: video 単位 holdout で「近い地平の 2 段が中盤勝率予測に足すか」を検証

## 設計方針
- オフライン・CPU 節度(スレッド制限 3)
- 既存ファイル破壊なし(新規 CSV/テキスト出力のみ)
- 1 関数 50 行以内・型ヒント・日本語コメント
- 解釈可能な形を優先(線形合成 / 少数特徴)

## 入出力
- 入力: data/indicators_v2/exchange_labels.csv
- 出力: data/indicators_v2/midgame_advantage_report.txt
        data/indicators_v2/midgame_advantage_adv_mid.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# スレッド制限(CPU 節度)
# --------------------------------------------------------------------------
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJ_ROOT / "data" / "indicators_v2" / "exchange_labels.csv"
REPORT_PATH = PROJ_ROOT / "data" / "indicators_v2" / "midgame_advantage_report.txt"
ADV_MID_CSV = PROJ_ROOT / "data" / "indicators_v2" / "midgame_advantage_adv_mid.csv"

# --------------------------------------------------------------------------
# 定数
# --------------------------------------------------------------------------
# net_ojama_after の分位数境界(B-1 分位別集計用)
NET_OJAMA_QUANTILES: tuple[float, float] = (0.33, 0.67)

# GBM パラメータ(段1予測・B-3 用)
GBM_PARAMS: dict[str, Any] = {
    "max_iter": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_samples_leaf": 10,
    "random_state": 42,
    "early_stopping": False,
}

# adv_mid の合成重み: opp_buried 有効度 vs taiou_difficulty(1-taiou_success)
# 両者同等に扱う簡単な線形合成
W_OPP_BURIED: float = 0.5
W_TAIOU_DIFFICULTY: float = 0.5

# 段1モデルの特徴量(発火側盤面 + diff)
STAGE1_FEATURES: list[str] = [
    "fire_absorption_capacity",
    "fire_dig_resistance",
    "fire_current_max_chain",
    "fire_death_margin",
    "fire_board_ojama_count",
    "opp_absorption_capacity",
    "opp_dig_resistance",
    "opp_death_margin",
    "opp_board_ojama_count",
    "diff_absorption_capacity",
    "diff_dig_resistance",
    "diff_current_max_chain",
    "diff_death_margin",
]

# B-3 の tier1 指標(発火側 + opp + diff を全列使用)
TIER1_FEATURE_PREFIXES: tuple[str, ...] = ("fire_", "opp_", "diff_")


# --------------------------------------------------------------------------
# データ読み込み
# --------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    """exchange_labels.csv を読み込んで基本確認を行う。"""
    df = pd.read_csv(path)
    required = {"video_id", "phase", "won", "taiou_success", "opp_buried",
                "net_ojama_after", "fire_side"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"必須列が不足: {missing}")
    df["won"] = df["won"].astype(float)
    print(f"[読み込み] {len(df)} 行 / {df['video_id'].nunique()} 動画")
    return df


# --------------------------------------------------------------------------
# B-1: 条件付き勝率 / ΔWinProb 表
# --------------------------------------------------------------------------

def _win_rate_table(
    df: pd.DataFrame,
    col: str,
    label_pos: str,
    label_neg: str,
) -> dict[str, float]:
    """2値列の条件付き勝率と ΔWinProb を返す辞書。

    won=1 は fire_side(攻撃側)が試合に勝ったことを意味する。
    """
    pos = df[df[col] == 1]
    neg = df[df[col] == 0]
    p_pos = float(pos["won"].mean()) if len(pos) > 0 else float("nan")
    p_neg = float(neg["won"].mean()) if len(neg) > 0 else float("nan")
    delta = p_pos - p_neg if not (np.isnan(p_pos) or np.isnan(p_neg)) else float("nan")
    return {
        f"P(won|{label_pos})": p_pos,
        f"n_{label_pos}": len(pos),
        f"P(won|{label_neg})": p_neg,
        f"n_{label_neg}": len(neg),
        "ΔWinProb": delta,
    }


def compute_b1_conditional_winprob(df: pd.DataFrame) -> list[str]:
    """B-1: 近い地平アウトカム別の条件付き勝率 / ΔWinProb 表を構築して行リストで返す。"""
    lines: list[str] = ["=== B-1: 条件付き勝率 / ΔWinProb ===", ""]
    phases = ["全体", "序", "中", "終"]

    for phase in phases:
        sub = df if phase == "全体" else df[df["phase"] == phase]
        if len(sub) == 0:
            continue
        lines.append(f"--- 位相: {phase} (n={len(sub)}) ---")

        # taiou_success: 受け手が対応成功した場合の攻撃側勝率
        ts = _win_rate_table(sub, "taiou_success", "対応成功", "対応失敗")
        lines.append(
            f"  taiou_success: P(won|対応成功)={ts['P(won|対応成功)']:.3f} (n={ts['n_対応成功']})"
            f" / P(won|対応失敗)={ts['P(won|対応失敗)']:.3f} (n={ts['n_対応失敗']})"
            f"  ΔWinProb={ts['ΔWinProb']:+.3f}"
        )

        # opp_buried: 相手が埋没した場合の攻撃側勝率
        ob = _win_rate_table(sub, "opp_buried", "相手埋没", "相手生存")
        lines.append(
            f"  opp_buried:    P(won|相手埋没)={ob['P(won|相手埋没)']:.3f} (n={ob['n_相手埋没']})"
            f" / P(won|相手生存)={ob['P(won|相手生存)']:.3f} (n={ob['n_相手生存']})"
            f"  ΔWinProb={ob['ΔWinProb']:+.3f}"
        )

        # net_ojama_after 分位別の勝率
        lines.append(_net_ojama_winprob_line(sub))
        lines.append("")

    return lines


def _net_ojama_winprob_line(sub: pd.DataFrame) -> str:
    """net_ojama_after を 3 分位に分けた勝率を 1 行で返す。"""
    q_lo = float(sub["net_ojama_after"].quantile(NET_OJAMA_QUANTILES[0]))
    q_hi = float(sub["net_ojama_after"].quantile(NET_OJAMA_QUANTILES[1]))
    low = sub[sub["net_ojama_after"] <= q_lo]
    mid = sub[(sub["net_ojama_after"] > q_lo) & (sub["net_ojama_after"] <= q_hi)]
    hi = sub[sub["net_ojama_after"] > q_hi]
    p_low = float(low["won"].mean()) if len(low) > 0 else float("nan")
    p_mid = float(mid["won"].mean()) if len(mid) > 0 else float("nan")
    p_hi = float(hi["won"].mean()) if len(hi) > 0 else float("nan")
    return (
        f"  net_ojama_after 分位別: 低(<={q_lo:.0f})={p_low:.3f}(n={len(low)})"
        f" / 中({q_lo:.0f}-{q_hi:.0f})={p_mid:.3f}(n={len(mid)})"
        f" / 高(>{q_hi:.0f})={p_hi:.3f}(n={len(hi)})"
        f"  ΔWinProb(高-低)={p_hi - p_low:+.3f}"
    )


# --------------------------------------------------------------------------
# 段1モデル: video 単位 LeaveOneOut OOF
# --------------------------------------------------------------------------

def _get_stage1_features(df: pd.DataFrame) -> list[str]:
    """STAGE1_FEATURES のうち DataFrame に存在する列を返す。"""
    return [c for c in STAGE1_FEATURES if c in df.columns]


def run_logo_gbm(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """LeaveOneGroupOut (video 単位) で OOF 確率を返す。shape=(n,)。

    グループ数が 2 未満の場合は全 NaN を返す。
    """
    oof = np.full(len(y), np.nan)
    logo = LeaveOneGroupOut()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return oof
    for tr_idx, te_idx in logo.split(X, y, groups=groups):
        if len(np.unique(y[tr_idx])) < 2:
            continue
        mdl = HistGradientBoostingClassifier(**GBM_PARAMS)
        mdl.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = mdl.predict_proba(X[te_idx])[:, 1]
    return oof


def build_stage1_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """段1: taiou_success / opp_buried の OOF 予測確率を追加した DataFrame を返す。

    中盤行のみ対象(B-2/B-3 の中盤評価用)。全体行には NaN を返す。
    """
    result = df.copy()
    result["pred_taiou"] = np.nan
    result["pred_opp_buried"] = np.nan

    # 中盤行のみで段1モデルを学習
    mid_mask = result["phase"] == "中"
    mid = result[mid_mask].copy()
    if len(mid) == 0:
        return result

    feat_cols = _get_stage1_features(mid)
    X = mid[feat_cols].fillna(0.0).values.astype(float)
    groups = mid["video_id"].values

    # taiou_success 予測
    y_ts = mid["taiou_success"].values.astype(int)
    if len(np.unique(y_ts)) >= 2:
        oof_ts = run_logo_gbm(X, y_ts, groups)
        result.loc[mid_mask, "pred_taiou"] = oof_ts
    else:
        print("[WARN] taiou_success が単一クラス → 予測スキップ")

    # opp_buried 予測
    y_ob = mid["opp_buried"].values.astype(int)
    if len(np.unique(y_ob)) >= 2:
        oof_ob = run_logo_gbm(X, y_ob, groups)
        result.loc[mid_mask, "pred_opp_buried"] = oof_ob
    else:
        print("[WARN] opp_buried が単一クラス → 予測スキップ")

    return result


# --------------------------------------------------------------------------
# B-2: adv_mid 合成スコア
# --------------------------------------------------------------------------

def compute_adv_mid(df: pd.DataFrame) -> pd.DataFrame:
    """B-2: 1P 視点の中盤有利不利スコア adv_mid を計算して返す。

    adv_mid の直感:
      - 「攻撃側として相手を埋めやすい(pred_opp_buried 高)」
      - 「かつ相手は攻撃に対応できない(pred_taiou 低 → 困難度高)」
      ほど adv_mid が高い(攻撃側有利)。

    fire_side=1P 行: adv_mid = raw_score
    fire_side=2P 行: adv_mid = -raw_score (2P が攻撃側 → 1P 視点では不利)
    最終的に各ゲームフレーム相当行に 1P 視点スコアを持つ。
    """
    df = df.copy()
    # 生スコア: opp_buried 有効度 - taiou_success 困難度 の逆(攻撃側有利)
    # taiou 困難度 = 1 - pred_taiou(受け手が対応しにくいほど攻撃有利)
    df["raw_adv"] = (
        W_OPP_BURIED * df["pred_opp_buried"]
        + W_TAIOU_DIFFICULTY * (1.0 - df["pred_taiou"])
    ) * 2.0 - 1.0  # 0〜1 を -1〜+1 に変換

    # 1P 視点への変換
    df["adv_mid"] = np.where(df["fire_side"] == "1P", df["raw_adv"], -df["raw_adv"])
    return df


# --------------------------------------------------------------------------
# adv_mid 分布確認
# --------------------------------------------------------------------------

def summarize_adv_mid(df: pd.DataFrame) -> list[str]:
    """adv_mid の分布・対称性・アウトカム相関を確認して行リストで返す。"""
    lines: list[str] = ["=== B-2: adv_mid 分布・健全性 ===", ""]
    mid = df[df["phase"] == "中"].dropna(subset=["adv_mid"])

    lines.append(f"  中盤行数(adv_mid 有効): {len(mid)}")
    desc = mid["adv_mid"].describe()
    lines.append(f"  mean={desc['mean']:.4f}  std={desc['std']:.4f}"
                 f"  min={desc['min']:.4f}  max={desc['max']:.4f}")

    # 1P/2P 対称性確認
    p1 = mid[mid["fire_side"] == "1P"]["adv_mid"]
    p2 = mid[mid["fire_side"] == "2P"]["adv_mid"]
    lines.append(f"  1P発火時 adv_mid mean={p1.mean():.4f} / 2P発火時 adv_mid mean={p2.mean():.4f}")
    lines.append(f"  ※ 対称性健全なら 両者の符号が逆向き・絶対値が近い")
    lines.append("")

    # アウトカム相関
    corr_ts = float(mid["adv_mid"].corr(mid["pred_taiou"]))
    corr_ob = float(mid["adv_mid"].corr(mid["pred_opp_buried"]))
    corr_won = float(mid["adv_mid"].corr(mid["won"]))
    lines.append(f"  corr(adv_mid, pred_taiou)   = {corr_ts:+.4f}")
    lines.append(f"    ※ fire_side 混在で相殺: 攻撃側行のみで確認すること")
    lines.append(f"  corr(adv_mid, pred_opp_buried)= {corr_ob:+.4f}")
    lines.append(f"    ※ 同上(2P 攻撃行では adv_mid が負・pred_opp_buried が正で相殺)")
    lines.append(f"  corr(adv_mid, won)            = {corr_won:+.4f}  (正なら adv_mid が勝率方向と一致)")

    # 攻撃側行(fire_side=1P)のみで確認
    p1_mid = mid[mid["fire_side"] == "1P"]
    if len(p1_mid) > 0:
        corr_ts_p1 = float(p1_mid["adv_mid"].corr(p1_mid["pred_taiou"]))
        corr_ob_p1 = float(p1_mid["adv_mid"].corr(p1_mid["pred_opp_buried"]))
        lines.append(f"  [1P攻撃行のみ] corr(adv_mid, pred_taiou)={corr_ts_p1:+.4f}"
                     f"  (正が健全: 1P攻撃有利→pred_taiou高い)")
        lines.append(f"  [1P攻撃行のみ] corr(adv_mid, pred_opp_buried)={corr_ob_p1:+.4f}"
                     f"  (正が健全: 1P攻撃有利→pred_opp_buried高い)")
    lines.append("")
    return lines


# --------------------------------------------------------------------------
# B-3: video 単位 holdout AUC 比較
# --------------------------------------------------------------------------

def _tier1_feature_cols(df: pd.DataFrame) -> list[str]:
    """B-3 baseline 用の tier1 特徴量列名を返す。"""
    return [
        c for c in df.columns
        if any(c.startswith(pfx) for pfx in TIER1_FEATURE_PREFIXES)
        and c not in {"fire_side", "phase", "won", "video_id", "game_idx",
                      "t_sec", "taiou_success", "opp_buried", "survived",
                      "net_ojama_after", "net_ojama", "returned",
                      "returned_competitive", "return_window_sec", "approx_fire_chains"}
    ]


def _logo_auc(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label: str,
) -> float:
    """LeaveOneGroupOut OOF AUC を計算して返す。エラー時は NaN。"""
    oof = np.full(len(y), np.nan)
    logo = LeaveOneGroupOut()
    for tr_idx, te_idx in logo.split(X, y, groups=groups):
        if len(np.unique(y[tr_idx])) < 2:
            continue
        mdl = HistGradientBoostingClassifier(**GBM_PARAMS)
        mdl.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = mdl.predict_proba(X[te_idx])[:, 1]
    valid = ~np.isnan(oof)
    if valid.sum() < 10 or len(np.unique(y[valid])) < 2:
        return float("nan")
    auc = float(roc_auc_score(y[valid], oof[valid]))
    n_vid = len(np.unique(groups))
    print(f"    [{label}] n={valid.sum()}  video={n_vid}  OOF AUC={auc:.4f}")
    return auc


def run_b3_comparison(df: pd.DataFrame) -> list[str]:
    """B-3: 中盤 won 予測 AUC 3条件比較(baseline/+exchange/adv_mid単体)。"""
    lines: list[str] = ["=== B-3: 中盤 won 予測 AUC 比較 (video 単位 LeaveOneOut) ===", ""]

    mid = df[df["phase"] == "中"].dropna(subset=["won"]).copy()
    y = mid["won"].values.astype(int)
    groups = mid["video_id"].values
    lines.append(f"  中盤サンプル: {len(mid)} / 動画: {len(np.unique(groups))}")
    lines.append("")

    # (1) baseline: tier1 指標のみ
    tier1_cols = _tier1_feature_cols(mid)
    X_base = mid[tier1_cols].fillna(0.0).values.astype(float)
    auc_base = _logo_auc(X_base, y, groups, "baseline:tier1のみ")
    lines.append(f"  (1) baseline (tier1指標のみ, n_feat={len(tier1_cols)}):  AUC={auc_base:.4f}")

    # (2) +exchange: baseline + 段1近い地平予測(pred_taiou/pred_opp_buried/net_ojama_after)
    exchange_cols = ["pred_taiou", "pred_opp_buried", "net_ojama_after"]
    valid_ex = [c for c in exchange_cols if c in mid.columns and mid[c].notna().any()]
    X_ex = mid[tier1_cols + valid_ex].fillna(0.0).values.astype(float)
    auc_ex = _logo_auc(X_ex, y, groups, "+exchange予測")
    lines.append(f"  (2) +exchange (段1予測追加, n_feat={len(tier1_cols)+len(valid_ex)}):  AUC={auc_ex:.4f}")
    lines.append(f"      ΔAUC vs baseline: {auc_ex - auc_base:+.4f}")
    lines.append("")

    # (3) adv_mid 単体
    if mid["adv_mid"].notna().any():
        X_adv = mid[["adv_mid"]].fillna(0.0).values.astype(float)
        auc_adv = _logo_auc(X_adv, y, groups, "adv_mid単体")
        lines.append(f"  (3) adv_mid 単体:  AUC={auc_adv:.4f}")
    else:
        auc_adv = float("nan")
        lines.append("  (3) adv_mid 単体:  データ不足 -> NaN")
    lines.append("")

    # 判定コメント
    lines += _b3_verdict(auc_base, auc_ex, auc_adv)
    return lines


def _b3_verdict(
    auc_base: float,
    auc_ex: float,
    auc_adv: float,
) -> list[str]:
    """B-3 判定: 2段が勝率予測に足すか/中盤有利不利として表示に足るか。"""
    lines: list[str] = ["  --- 判定 ---"]
    delta = auc_ex - auc_base

    if np.isnan(delta):
        lines.append("  ΔAUCが計算不能。データ確認が必要。")
    elif delta > 0.01:
        lines.append(f"  [YES] 2段 +exchange が baseline を +{delta:.4f} 上回る。")
        lines.append("  近い地平アウトカム予測は中盤勝率予測に明確な追加価値あり。")
    elif delta > 0.002:
        lines.append(f"  [WEAK] 2段 +exchange が baseline を +{delta:.4f} 上回るが軽微。")
        lines.append("  近い地平の追加価値は小さいが adv_mid の解釈的意味は保持。")
    else:
        lines.append(f"  [NO] 2段の追加は baseline を {delta:+.4f} しか動かさない。")
        lines.append("  中盤勝率予測はやはり頭打ち。")
        lines.append("  ただし adv_mid 自体は「対応力差」という表示可能な意味を持つ。")
    lines.append("")

    if not np.isnan(auc_adv):
        if auc_adv >= 0.55:
            lines.append(f"  adv_mid 単体 AUC={auc_adv:.4f}: 単独でも予測力あり。オーバーレイ表示に足る。")
        elif auc_adv >= 0.51:
            lines.append(f"  adv_mid 単体 AUC={auc_adv:.4f}: 弱い予測力。表示指標として参考程度。")
        else:
            lines.append(f"  adv_mid 単体 AUC={auc_adv:.4f}: ランダムと区別困難。表示は要注意。")
    return lines


# --------------------------------------------------------------------------
# 段1 OOF 精度確認(補助)
# --------------------------------------------------------------------------

def evaluate_stage1_auc(df: pd.DataFrame) -> list[str]:
    """段1予測(taiou/opp_buried)の OOF AUC を中盤で確認して行リストで返す。"""
    lines: list[str] = ["=== 段1モデル OOF AUC (中盤) ===", ""]
    mid = df[df["phase"] == "中"].dropna(subset=["pred_taiou", "pred_opp_buried"])
    if len(mid) == 0:
        lines.append("  中盤データなし")
        return lines

    y_ts = mid["taiou_success"].values.astype(int)
    y_ob = mid["opp_buried"].values.astype(int)
    groups = mid["video_id"].values

    for label, y_true, pred_col in [
        ("taiou_success", y_ts, "pred_taiou"),
        ("opp_buried",    y_ob, "pred_opp_buried"),
    ]:
        p = mid[pred_col].values
        valid = ~np.isnan(p)
        if valid.sum() < 5 or len(np.unique(y_true[valid])) < 2:
            lines.append(f"  {label}: 評価不能 (有効={valid.sum()})")
            continue
        auc = float(roc_auc_score(y_true[valid], p[valid]))
        n_pos = int(y_true[valid].sum())
        lines.append(f"  {label}: AUC={auc:.4f}  n={valid.sum()}  n_pos={n_pos}")
    lines.append("")
    return lines


# --------------------------------------------------------------------------
# メイン
# --------------------------------------------------------------------------

def main() -> None:
    """メイン処理。"""
    import warnings
    warnings.filterwarnings("ignore")

    print(f"[build_midgame_advantage] 入力: {INPUT_CSV}")
    df = load_data(INPUT_CSV)

    # B-1: 条件付き勝率
    print("\n--- B-1: 条件付き勝率計算中 ---")
    b1_lines = compute_b1_conditional_winprob(df)
    for line in b1_lines:
        print(line)

    # 段1モデル構築(B-2/B-3 の前提)
    print("\n--- 段1モデル(OOF) 構築中 (video 単位 LeaveOneOut) ---")
    print("    ※ 89 動画 × LeaveOneOut = 89 fold。数分かかります。")
    df_pred = build_stage1_predictions(df)

    # 段1 AUC 確認
    stage1_lines = evaluate_stage1_auc(df_pred)
    for line in stage1_lines:
        print(line)

    # B-2: adv_mid 合成
    print("\n--- B-2: adv_mid 合成中 ---")
    df_adv = compute_adv_mid(df_pred)
    adv_lines = summarize_adv_mid(df_adv)
    for line in adv_lines:
        print(line)

    # B-3: AUC 比較
    print("\n--- B-3: 中盤 AUC 比較 ---")
    b3_lines = run_b3_comparison(df_adv)
    for line in b3_lines:
        print(line)

    # レポート保存
    all_lines = (
        ["=== 中盤有利不利スコア 2段構成 (B) レポート ===", ""]
        + b1_lines
        + [""]
        + stage1_lines
        + [""]
        + adv_lines
        + [""]
        + b3_lines
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(all_lines), encoding="utf-8")
    print(f"\n[保存] レポート: {REPORT_PATH}")

    # adv_mid CSV 保存(中盤行のみ)
    out_cols = ["video_id", "game_idx", "t_sec", "fire_side", "phase",
                "won", "pred_taiou", "pred_opp_buried", "adv_mid",
                "taiou_success", "opp_buried", "net_ojama_after"]
    out_cols_valid = [c for c in out_cols if c in df_adv.columns]
    mid_out = df_adv[df_adv["phase"] == "中"][out_cols_valid].reset_index(drop=True)
    mid_out.to_csv(ADV_MID_CSV, index=False)
    print(f"[保存] adv_mid CSV: {ADV_MID_CSV}  ({len(mid_out)} 行)")
    print("\n[完了]")


if __name__ == "__main__":
    main()
