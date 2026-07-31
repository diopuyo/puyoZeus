"""「中盤は動的な流れ(時系列)で決まる」仮説の安いプローブ検証。

## 背景
静止盤面 1 枚 (現在値のみ) の指標では中盤 win-AUC が 0.51〜0.56 で頭打ちに
なることが確定している (scripts/model_indicator_win.py, prescreen_candidates.py)。
本スクリプトは「時系列情報 (指標の軌跡・変化速度・傾き・接戦度) を足すと
中盤 AUC が静止 baseline を明確に超えるか」を検証する。

## データと時系列の組み方 (重要な注意)
labeled_win.csv の `game_idx` は実際の 1 試合と 1:1 対応しない
(同一 (video_id, game_idx) バケツ内に tsumo リセットを挟んで複数の
実試合が混在する。model_indicator_win.pair_sides_for_win の docstring
が警告する「窓内相対インデックス」問題と同根)。
そのため本スクリプトは game_idx を信用せず、(video_id, side) 単位で
t_sec 昇順に全行を並べ、tsumo が減少した地点を「新しい試合の開始」と
みなして match_uid を独自に割り当て直す (assign_match_segments)。
これにより試合境界をまたいだ誤った差分/傾き計算を防ぐ。

## 時系列特徴
各 diff 指標 (1P-2P) について、直近 K 手窓 (K=3,5,8) から:
  - momentum: (現在値 - K手前の値) / K  (全 diff 指標)
  - slope:    窓内 OLS 傾き (手数を x とする線形回帰の傾き) (指標カテゴリ厳選)
  - sign_flip: 窓内の一手ごとの符号反転回数 (接戦度・厳選)
  - var:      窓内分散 (揺れ幅・厳選)
を作り、静止 baseline (現在値=diff のみ) と比較する。
比較は HistGBC (video 単位 GroupKFold OOF) + 小型 LSTM (任意)。

## 使い方
    PYTHONPATH=. python -m scripts.proto_temporal_winprob
    PYTHONPATH=. python -m scripts.proto_temporal_winprob --skip-lstm
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

# スレッド制限 (CLAUDE.md 熱暴走対策)
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from scripts.model_indicator_win import (  # noqa: E402
    DEFAULT_MAX_TDIFF,
    FIRE_INDICATORS,
    DANGER_INDICATORS,
    DENSITY_INDICATORS,
    build_features,
    load_labeled_csv,
    pair_sides_for_win,
    run_oof_classifier,
    _get_indicator_cols,
)

warnings.filterwarnings("ignore")

# =============================================================================
# 定数
# =============================================================================

LABELED_CSV: Path = PROJ_ROOT / "data" / "indicators_v2" / "study" / "labeled_win.csv"
OUT_DIR: Path = PROJ_ROOT / "data" / "indicators_v2" / "study"
RESULT_CSV: Path = OUT_DIR / "proto_temporal_winprob_auc.csv"

N_FOLDS: int = 5

# 時系列窓長候補 (手数単位)
K_LIST: list[int] = [3, 5, 8]
K_MAX: int = max(K_LIST)

# 試合境界とみなす tsumo 減少の許容差 (0 = わずかでも減少したら新試合とみなす)
TSUMO_RESET_TOL: float = 0.0

# 位相境界 (tsumo_count_rate 基準。既存 prescreen_candidates.py と同一定義)
TSUMO_EARLY_MAX: float = 0.33
TSUMO_LATE_MIN: float = 0.67
PHASE_ALL: str = "全体"
PHASE_EARLY: str = "序盤"
PHASE_MID: str = "中盤"
PHASE_LATE: str = "終盤"

# board sim 本命 5 指標 (commit b4228c8) - slope/sign_flip/var 厳選対象に含める
BOARD_SIM_INDICATORS: frozenset[str] = frozenset([
    "saturated_chain_count", "ignition_point_count", "multi_color_ignition",
    "sub_chain_count", "simultaneous_pop_richness",
])

# slope/sign_flip/var (計算コストがやや高い特徴) の厳選対象指標
CURATED_INDICATORS: frozenset[str] = (
    FIRE_INDICATORS | DANGER_INDICATORS | DENSITY_INDICATORS | BOARD_SIM_INDICATORS
)

# LSTM ハイパーパラメータ (小型・軽量)
LSTM_HIDDEN: int = 32
LSTM_EPOCHS: int = 40
LSTM_LR: float = 1e-3
LSTM_WEIGHT_DECAY: float = 1e-4


# =============================================================================
# 1. データ読み込み + ペアリング + 試合境界の再構築
# =============================================================================

def load_and_pair(labeled_path: Path, max_tdiff: float) -> pd.DataFrame:
    """labeled_win.csv を読み込み 1P/2P ペアを構成する (既存関数を再利用)。"""
    df = load_labeled_csv(str(labeled_path))
    paired = pair_sides_for_win(df, max_tdiff)
    return paired


def assign_match_segments(paired: pd.DataFrame) -> pd.DataFrame:
    """(video_id, t_sec) 昇順に並べ直し、tsumo 減少地点を試合境界として
    match_uid (試合固有ID) と seg_pos (試合内の相対手順位置, 0始まり) を付与する。

    game_idx は窓内相対インデックスで実試合と 1:1 対応しないため使わない
    (docstring 冒頭の注意を参照)。
    """
    df = paired.sort_values(["video_id_1p", "t_sec_1p"]).reset_index(drop=True)
    vid = df["video_id_1p"].values
    tsumo = df["tsumo_1p"].astype(float).values
    reset = np.zeros(len(df), dtype=bool)
    reset[0] = True
    is_new_video = vid[1:] != vid[:-1]
    is_tsumo_drop = tsumo[1:] < (tsumo[:-1] - TSUMO_RESET_TOL)
    reset[1:] = is_new_video | is_tsumo_drop
    match_local_id = np.cumsum(reset)
    df["match_uid"] = [f"{v}_g{m}" for v, m in zip(vid, match_local_id)]
    df["seg_pos"] = df.groupby("match_uid").cumcount()
    n_matches = df["match_uid"].nunique()
    seg_len = df.groupby("match_uid").size()
    print(f"  試合境界再構築: {n_matches} 試合セグメント "
          f"(平均長={seg_len.mean():.1f}手, 中央値={seg_len.median():.0f}手, "
          f"最小={seg_len.min()}, 最大={seg_len.max()})")
    return df


def drop_inconsistent_segments(df: pd.DataFrame) -> pd.DataFrame:
    """セグメント内で won_1p が揺れる (=境界誤検出の疑い) セグメントを除外する。"""
    won_nunique = df.groupby("match_uid")["won_1p"].nunique()
    bad_uids = won_nunique[won_nunique > 1].index
    if len(bad_uids) > 0:
        n_bad_rows = df["match_uid"].isin(bad_uids).sum()
        print(f"  [WARN] won 不整合セグメント {len(bad_uids)} 件 "
              f"({n_bad_rows} 行) を除外")
        df = df[~df["match_uid"].isin(bad_uids)].reset_index(drop=True)
    return df


# =============================================================================
# 2. 時系列特徴量の構築
# =============================================================================

def compute_momentum_features(
    df: pd.DataFrame, diff_cols: list[str], k_list: list[int],
) -> pd.DataFrame:
    """全 diff 指標について momentum_K = (現在値-K手前)/K を計算する。"""
    g = df.groupby("match_uid")
    for col in diff_cols:
        for k in k_list:
            shifted = g[col].shift(k)
            df[f"{col}_mom{k}"] = (df[col] - shifted) / float(k)
    return df


def _slope_of_window(values: np.ndarray) -> float:
    """窓内 OLS 傾き (x=0..len-1 に対する線形回帰の傾き)。"""
    n = len(values)
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = values.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom < 1e-12:
        return 0.0
    return float(((x - x_mean) * (values - y_mean)).sum() / denom)


def compute_slope_var_features(
    df: pd.DataFrame, curated_cols: list[str], k_list: list[int],
) -> pd.DataFrame:
    """厳選指標について slope_K (傾き) / var_K (窓内分散) を計算する。"""
    for col in curated_cols:
        for k in k_list:
            w = k + 1
            roll = df.groupby("match_uid")[col].rolling(window=w, min_periods=w)
            df[f"{col}_slope{k}"] = roll.apply(
                _slope_of_window, raw=True
            ).reset_index(level=0, drop=True)
            df[f"{col}_var{k}"] = roll.std().reset_index(level=0, drop=True)
    return df


def compute_sign_flip_features(
    df: pd.DataFrame, curated_cols: list[str], k_list: list[int],
) -> pd.DataFrame:
    """厳選指標について sign_flip_K (窓内の一手ごとの符号反転回数) を計算する。"""
    g = df.groupby("match_uid")
    for col in curated_cols:
        step_diff = g[col].diff()
        sign = np.sign(step_diff.fillna(0.0))
        flip = (sign.groupby(df["match_uid"]).diff().abs() > 0).astype(float)
        for k in k_list:
            df[f"{col}_signflip{k}"] = (
                flip.groupby(df["match_uid"])
                .rolling(window=k, min_periods=k).sum()
                .reset_index(level=0, drop=True)
            )
    return df


def filter_min_history(df: pd.DataFrame, k_max: int) -> pd.DataFrame:
    """全 K 条件で共通のフェアな比較ができるよう、seg_pos>=k_max の行のみ残す。"""
    filtered = df[df["seg_pos"] >= k_max].reset_index(drop=True)
    print(f"  履歴フィルタ (seg_pos>={k_max}): {len(df)} -> {len(filtered)} 行")
    return filtered


# =============================================================================
# 3. 特徴量セット定義 (variant)
# =============================================================================

def build_variant_column_sets(
    diff_cols: list[str], curated_cols: list[str], k_list: list[int],
) -> dict[str, list[str]]:
    """比較する variant ごとの特徴量列名リストを返す。"""
    variants: dict[str, list[str]] = {}
    variants["static_diff"] = list(diff_cols)
    for k in k_list:
        cols = list(diff_cols)
        cols += [f"{c}_mom{k}" for c in diff_cols]
        cols += [f"{c}_slope{k}" for c in curated_cols]
        cols += [f"{c}_var{k}" for c in curated_cols]
        cols += [f"{c}_signflip{k}" for c in curated_cols]
        variants[f"temporal_K{k}"] = cols
    multi_cols = list(diff_cols)
    for k in k_list:
        multi_cols += [f"{c}_mom{k}" for c in diff_cols]
    multi_cols += [f"{c}_slope{K_MAX}" for c in curated_cols]
    multi_cols += [f"{c}_var{K_MAX}" for c in curated_cols]
    multi_cols += [f"{c}_signflip{K_MAX}" for c in curated_cols]
    variants["temporal_multiK"] = multi_cols
    return variants


# =============================================================================
# 4. HistGBC 位相別 OOF AUC
# =============================================================================

def _phase_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    tcr = df["tsumo_count_rate_1p"].astype(float)
    return {
        PHASE_ALL: pd.Series(True, index=df.index),
        PHASE_EARLY: tcr <= TSUMO_EARLY_MAX,
        PHASE_MID: (tcr > TSUMO_EARLY_MAX) & (tcr <= TSUMO_LATE_MIN),
        PHASE_LATE: tcr > TSUMO_LATE_MIN,
    }


def run_variant_phase_aucs(
    df: pd.DataFrame,
    cols: list[str],
    y: np.ndarray,
    groups: np.ndarray,
    phase_masks: dict[str, pd.Series],
    n_folds: int,
    label: str,
) -> dict[str, float]:
    """指定 variant の列セットで位相別 (video 単位 GroupKFold) OOF AUC を返す。"""
    valid_cols = [c for c in cols if c in df.columns]
    X_all = df[valid_cols].fillna(0.0).values.astype(float)
    result: dict[str, float] = {}
    for phase, mask in phase_masks.items():
        idx = mask.values
        X_ph, y_ph, g_ph = X_all[idx], y[idx], groups[idx]
        n_uniq = len(np.unique(g_ph))
        if len(X_ph) < 30 or n_uniq < 2 or len(np.unique(y_ph)) < 2:
            result[phase] = float("nan")
            continue
        folds = min(n_folds, n_uniq)
        oof, _ = run_oof_classifier(X_ph, y_ph, g_ph, folds)
        valid = ~np.isnan(oof[:, 0])
        auc = float(roc_auc_score(y_ph[valid], oof[valid, 1]))
        result[phase] = auc
    print(f"  [{label}] n_features={len(valid_cols)}  "
          + "  ".join(f"{p}={result.get(p, float('nan')):.4f}" for p in phase_masks))
    return result


# =============================================================================
# 5. 小型 LSTM (任意)
# =============================================================================

def build_lstm_sequences(
    df: pd.DataFrame, diff_cols: list[str], k_max: int,
) -> np.ndarray:
    """各行について直近 k_max+1 手の diff 指標系列を (n, k_max+1, n_feat) で返す。

    df は filter_min_history 適用前の (match_uid ごとの) 全履歴 df を渡すこと。
    呼び出し側で最終的に filter_min_history 後の行に合わせて添字選択する。
    """
    g = df.groupby("match_uid")
    steps: list[np.ndarray] = []
    for offset in range(k_max, -1, -1):
        shifted = pd.concat(
            [g[c].shift(offset) for c in diff_cols], axis=1
        ).values
        steps.append(shifted)
    seq = np.stack(steps, axis=1)  # (n, k_max+1, n_feat)
    return seq


class _TemporalLSTM:
    """torch LSTM 分類器の薄いラッパー (stateless インターフェース維持)。"""

    def __init__(self, n_features: int, device: str) -> None:
        import torch.nn as nn
        self.device = device
        self.net = nn.Sequential()
        self.lstm = nn.LSTM(n_features, LSTM_HIDDEN, batch_first=True).to(device)
        self.head = nn.Linear(LSTM_HIDDEN, 1).to(device)

    def parameters(self):
        return list(self.lstm.parameters()) + list(self.head.parameters())

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


def _train_lstm_fold(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, device: str,
) -> np.ndarray:
    """1 fold 分の LSTM を学習し、held-out の予測確率を返す。"""
    import torch
    import torch.nn as nn

    model = _TemporalLSTM(x_tr.shape[2], device)
    opt = torch.optim.Adam(model.parameters(), lr=LSTM_LR, weight_decay=LSTM_WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.tensor(x_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    for _ in range(LSTM_EPOCHS):
        opt.zero_grad()
        logit = model.forward(xt)
        loss = loss_fn(logit, yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        xe = torch.tensor(x_te, dtype=torch.float32, device=device)
        pred = torch.sigmoid(model.forward(xe)).cpu().numpy()
    return pred


def run_lstm_oof(
    seq: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int,
) -> np.ndarray:
    """video 単位 GroupKFold で LSTM OOF 確率を返す。"""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [LSTM] device={device}  seq_shape={seq.shape}")
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_folds)
    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(seq, y, groups=groups)):
        pred = _train_lstm_fold(seq[tr_idx], y[tr_idx].astype(np.float32),
                                 seq[te_idx], device)
        oof[te_idx] = pred
        print(f"    fold {fold_idx + 1}/{n_folds} 完了 "
              f"(train={len(tr_idx)} test={len(te_idx)})")
    return oof


def run_lstm_phase_aucs(
    seq: np.ndarray, y: np.ndarray, groups: np.ndarray,
    phase_masks: dict[str, pd.Series], n_folds: int,
) -> dict[str, float]:
    """LSTM を 1 回だけ全体で OOF 学習し、位相ごとに AUC を切り出す。"""
    oof = run_lstm_oof(seq, y, groups, n_folds)
    result: dict[str, float] = {}
    for phase, mask in phase_masks.items():
        idx = mask.values
        valid = idx & ~np.isnan(oof)
        if valid.sum() < 30 or len(np.unique(y[valid])) < 2:
            result[phase] = float("nan")
            continue
        result[phase] = float(roc_auc_score(y[valid], oof[valid]))
    print("  [LSTM] " + "  ".join(
        f"{p}={result.get(p, float('nan')):.4f}" for p in phase_masks))
    return result


# =============================================================================
# 6. レポート
# =============================================================================

def print_and_save_report(
    all_results: dict[str, dict[str, float]], out_path: Path,
) -> None:
    """variant x phase の AUC 表をコンソール出力し CSV 保存する。"""
    rows = []
    for variant, phase_auc in all_results.items():
        row = {"variant": variant, **phase_auc}
        rows.append(row)
    result_df = pd.DataFrame(rows)
    print()
    print("=" * 78)
    print("  中盤動的仮説プローブ: variant x 位相 win-AUC 一覧")
    print("=" * 78)
    print(result_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"\n  CSV 保存: {out_path}")

    base_mid = all_results.get("static_diff", {}).get(PHASE_MID, float("nan"))
    print("\n  --- 中盤 (核心) ΔAUC vs static_diff baseline ---")
    for variant, phase_auc in all_results.items():
        if variant == "static_diff":
            continue
        mid = phase_auc.get(PHASE_MID, float("nan"))
        if np.isnan(mid) or np.isnan(base_mid):
            continue
        print(f"    {variant:<16}: 中盤AUC={mid:.4f}  Δ={mid - base_mid:+.4f}")


# =============================================================================
# メイン
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="中盤動的仮説プローブ検証")
    parser.add_argument("--labeled", default=str(LABELED_CSV))
    parser.add_argument("--max-tdiff", type=float, default=DEFAULT_MAX_TDIFF)
    parser.add_argument("--skip-lstm", action="store_true", help="LSTM をスキップ")
    parser.add_argument("--out", default=str(RESULT_CSV))
    args = parser.parse_args()

    print("=== 1. データ読み込み + ペアリング ===")
    paired = load_and_pair(Path(args.labeled), args.max_tdiff)

    print("\n=== 2. 試合境界の再構築 (game_idx を信用しない) ===")
    df = assign_match_segments(paired)
    df = drop_inconsistent_segments(df)

    indicator_cols = _get_indicator_cols(df)
    diff_cols = [f"{c}_diff" for c in indicator_cols]
    curated_cols = [f"{c}_diff" for c in indicator_cols if c in CURATED_INDICATORS]
    print(f"  diff指標数={len(diff_cols)}  厳選指標数(slope/var/signflip対象)={len(curated_cols)}")

    # diff (1P-2P) 列を付与 (build_features は 1p/2p も返すが既存重複列は捨てる)
    feat_df = build_features(df, indicator_cols)
    diff_only = feat_df[[c for c in feat_df.columns if c.endswith("_diff")]]
    df = pd.concat([df.reset_index(drop=True), diff_only.reset_index(drop=True)], axis=1)

    print("\n=== 3. 時系列特徴量の構築 ===")
    df = compute_momentum_features(df, diff_cols, K_LIST)
    df = compute_slope_var_features(df, curated_cols, K_LIST)
    df = compute_sign_flip_features(df, curated_cols, K_LIST)

    if not args.skip_lstm:
        print("  LSTM 用シーケンス構築中...")
        seq_full = build_lstm_sequences(df, diff_cols, K_MAX)

    print(f"\n=== 4. 履歴フィルタ (K_MAX={K_MAX} 共通行セット) ===")
    keep_mask = (df["seg_pos"] >= K_MAX).values
    dff = filter_min_history(df, K_MAX)
    y = dff["won_1p"].astype(int).values
    groups = dff["video_id_1p"].values
    print(f"  最終サンプル数: {len(dff)}  動画数: {len(np.unique(groups))}")

    phase_masks = _phase_masks(dff)
    for phase, mask in phase_masks.items():
        print(f"  位相 {phase}: {int(mask.sum())} 行")

    print("\n=== 5. HistGBC variant 比較 (video単位GroupKFold OOF) ===")
    variants = build_variant_column_sets(diff_cols, curated_cols, K_LIST)
    all_results: dict[str, dict[str, float]] = {}
    for name, cols in variants.items():
        all_results[name] = run_variant_phase_aucs(
            dff, cols, y, groups, phase_masks, N_FOLDS, name)

    if not args.skip_lstm:
        print("\n=== 6. 小型 LSTM (直接系列入力) ===")
        seq = seq_full[keep_mask]
        all_results["lstm_raw_seq"] = run_lstm_phase_aucs(
            seq, y, groups, phase_masks, N_FOLDS)

    print_and_save_report(all_results, Path(args.out))
    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
