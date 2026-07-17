"""打ち合い収支 (honsen_output / honsen_tempo_output) の位相別単変量 AUC 測定。

CSV から current_max_chain_raw / reach_fire_power_max_chain を使って
honsen_output および時間窓つき honsen_tempo_output を再現し、
両サイドのペアリングで「打ち合い収支 = 1P出力 - 2P出力」を計算する。
既存強指標 (board_ojama_count, current_max_chain) と比較する。

★ 動画認識/重い処理はゼロ。CSV と軽量計算のみ。
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# =============================================================================
# 定数 (indicators_v2.py と同値で管理)
# =============================================================================

# 11 本較正: 連鎖数 → お邪魔数 ≈ A * exp(B * n)
CHAIN_OJAMA_A: float = 30.13
CHAIN_OJAMA_B: float = 0.297

# 正規化分母 (=144=ON_FIELD_CAP*2)
HONSEN_OUTPUT_NORM: float = 144.0

# テンポ核定数 (indicators_v2.py の較正値と同値)
# 1手あたり秒 (labeled_win.csv 実測中央値)
SEC_PER_HAND: float = 0.733
# 1連鎖ギャップを埋めるのに要する手数
HANDS_PER_CHAIN_GAP: float = 2.0
# achievable 不明時のフォールバック加算連鎖数
FALLBACK_CHAIN_ADD: float = 2.0
# 連鎖発火アニメ時間 (chain_to_time 用)
TIME_PER_CHAIN_SEC: float = 0.30

# ペアリング最大時刻差 (秒)
DEFAULT_MAX_TDIFF: float = 1.0

# 手数率 三分位 (序盤/中盤/終盤)
EARLY_THRESHOLD: float = 0.33
LATE_THRESHOLD: float = 0.67

# =============================================================================
# honsen_output / honsen_tempo_output 計算 (CSV 行から再現)
# =============================================================================


def _chain_to_ojama(n: float) -> float:
    """連鎖数 n から推定お邪魔個数を返す (n<=0 で 0.0)。"""
    if n <= 0.0:
        return 0.0
    return CHAIN_OJAMA_A * math.exp(CHAIN_OJAMA_B * n)


def _chain_to_time(n: float) -> float:
    """連鎖数 n の発火所要推定秒数。"""
    return max(0.0, TIME_PER_CHAIN_SEC * n)


def compute_honsen_raw(chain_raw: float) -> float:
    """current_max_chain_raw から honsen_output の raw 値を計算する。

    副砲連鎖数は現状取得不可のため current_max_chain 単体を使用。
    """
    return _chain_to_ojama(float(chain_raw))


def compute_tempo_raw(
    current_chain: float,
    achievable_chain: float,
    opp_chain: float,
) -> float:
    """テンポ核: 相手本線の窓内で自分が伸ばせる打ち合い出力 (raw)。

    honsen_tempo_output と同ロジック (CSV 再現用スタンドアロン版)。
    """
    ach = achievable_chain if achievable_chain > 0.0 else (current_chain + FALLBACK_CHAIN_ADD)
    window = _chain_to_time(opp_chain)
    hands = window / SEC_PER_HAND
    chain_gap = max(0.0, ach - current_chain)
    hands_needed = chain_gap * HANDS_PER_CHAIN_GAP
    frac = min(1.0, hands / hands_needed) if hands_needed > 0.0 else 1.0
    my_built = current_chain + frac * chain_gap
    return _chain_to_ojama(my_built)


# =============================================================================
# ペアリング
# =============================================================================


def pair_sides(df: pd.DataFrame, max_tdiff: float) -> pd.DataFrame:
    """1P/2P を (video_id, t_sec 近傍) でペアリングする。

    model_indicator_win.py の pair_sides_for_win と同ロジック。
    reach_fire_power_max_chain も保持 (テンポ核の achievable として使用)。
    """
    p1 = df[df["side"] == "1P"].reset_index(drop=True)
    p2 = df[df["side"] == "2P"].reset_index(drop=True)
    rows: list[dict] = []
    for vid, g1 in p1.groupby("video_id"):
        g2 = p2[p2["video_id"] == vid].reset_index(drop=True)
        if len(g2) == 0:
            continue
        t2 = g2["t_sec"].values
        for _, r1 in g1.iterrows():
            diffs = np.abs(t2 - float(r1["t_sec"]))
            idx_min = int(diffs.argmin())
            if diffs[idx_min] > max_tdiff:
                continue
            r2 = g2.iloc[idx_min]
            won1 = r1["won"]
            won2 = r2["won"]
            if pd.isna(won1) or pd.isna(won2):
                continue
            if abs(float(won1) + float(won2) - 1.0) > 0.01:
                continue
            # reach_fire_power_max_chain が列にない場合は 0 にフォールバック
            reach_1p = float(r1.get("reach_fire_power_max_chain", 0) or 0)
            reach_2p = float(r2.get("reach_fire_power_max_chain", 0) or 0)
            rows.append({
                "video_id": vid,
                "t_sec_1p": r1["t_sec"],
                "t_sec_2p": r2["t_sec"],
                "tsumo_count_rate_1p": r1["tsumo_count_rate"],
                "chain_raw_1p": r1["current_max_chain_raw"],
                "chain_raw_2p": r2["current_max_chain_raw"],
                "reach_chain_1p": reach_1p,
                "reach_chain_2p": reach_2p,
                "board_ojama_raw_1p": r1["board_ojama_count_raw"],
                "board_ojama_raw_2p": r2["board_ojama_count_raw"],
                "won_1p": float(won1),
            })
    paired = pd.DataFrame(rows)
    total_1p = len(p1)
    pair_rate = len(paired) / total_1p if total_1p > 0 else 0.0
    print(f"  ペア成立 (won整合後): {len(paired)} / 1P行 {total_1p}"
          f" (成立率 {pair_rate:.1%})")
    return paired


# =============================================================================
# AUC 計算
# =============================================================================


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """クラス数 < 2 のとき NaN を返す安全版 AUC。"""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def phase_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    phase_mask_early: np.ndarray,
    phase_mask_mid: np.ndarray,
    phase_mask_late: np.ndarray,
    name: str,
) -> dict[str, float]:
    """位相別 (全体/序盤/中盤/終盤) の単変量 AUC を計算して出力する。"""
    result = {
        "全体": _safe_auc(y_true, y_score),
        "序盤": _safe_auc(y_true[phase_mask_early], y_score[phase_mask_early]),
        "中盤": _safe_auc(y_true[phase_mask_mid], y_score[phase_mask_mid]),
        "終盤": _safe_auc(y_true[phase_mask_late], y_score[phase_mask_late]),
    }
    n_all = int(y_true.sum())
    print(f"\n  [{name}] n={len(y_true)}, 勝利={n_all}")
    for phase, auc in result.items():
        n_phase = {
            "全体": len(y_true),
            "序盤": int(phase_mask_early.sum()),
            "中盤": int(phase_mask_mid.sum()),
            "終盤": int(phase_mask_late.sum()),
        }[phase]
        auc_str = f"{auc:.4f}" if not math.isnan(auc) else "N/A"
        print(f"    {phase:3s}: AUC={auc_str}  (n={n_phase})")
    return result


# =============================================================================
# メイン
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="打ち合い収支 AUC 測定")
    parser.add_argument(
        "--csv",
        default="data/indicators_v2/study/labeled_win.csv",
        help="labeled_win.csv のパス",
    )
    parser.add_argument("--max-tdiff", type=float, default=DEFAULT_MAX_TDIFF)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/5] CSV 読み込み: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  行数={len(df)}, 列数={len(df.columns)}")

    print("\n[2/5] 1P/2P ペアリング")
    paired = pair_sides(df, args.max_tdiff)
    if len(paired) == 0:
        print("[ERROR] ペア 0 件: 分析不能", file=sys.stderr)
        sys.exit(1)

    print("\n[3/5] honsen_output および honsen_tempo_output 計算 (CSV 再現)")
    paired["honsen_1p"] = paired["chain_raw_1p"].apply(compute_honsen_raw)
    paired["honsen_2p"] = paired["chain_raw_2p"].apply(compute_honsen_raw)
    # 打ち合い収支 (単純版) = 1P出力 - 2P出力 (正→1P有利)
    paired["honsen_balance"] = paired["honsen_1p"] - paired["honsen_2p"]
    # テンポ核: 1P側が2Pの窓で伸ばせる出力、2P側が1Pの窓で伸ばせる出力
    paired["tempo_1p"] = [
        compute_tempo_raw(r["chain_raw_1p"], r["reach_chain_1p"], r["chain_raw_2p"])
        for _, r in paired.iterrows()
    ]
    paired["tempo_2p"] = [
        compute_tempo_raw(r["chain_raw_2p"], r["reach_chain_2p"], r["chain_raw_1p"])
        for _, r in paired.iterrows()
    ]
    # テンポ打ち合い収支 = 1P窓内出力 - 2P窓内出力
    paired["tempo_balance"] = paired["tempo_1p"] - paired["tempo_2p"]
    # 既存指標の差分 (比較用)
    paired["chain_diff"] = paired["chain_raw_1p"] - paired["chain_raw_2p"]
    paired["ojama_diff"] = paired["board_ojama_raw_1p"] - paired["board_ojama_raw_2p"]

    print(f"  honsen_balance 統計: "
          f"mean={paired['honsen_balance'].mean():.2f}, "
          f"std={paired['honsen_balance'].std():.2f}")
    print(f"  tempo_balance  統計: "
          f"mean={paired['tempo_balance'].mean():.2f}, "
          f"std={paired['tempo_balance'].std():.2f}")
    reach_avail = float((paired["reach_chain_1p"] > 0).mean())
    print(f"  reach_chain_1p > 0 (achievable 有効率): {reach_avail:.1%}")

    print("\n[4/5] 位相マスク作成 (tsumo_count_rate 三分位)")
    rate = paired["tsumo_count_rate_1p"].values
    early_q = np.quantile(rate, EARLY_THRESHOLD)
    late_q = np.quantile(rate, LATE_THRESHOLD)
    mask_early = rate <= early_q
    mask_mid = (rate > early_q) & (rate <= late_q)
    mask_late = rate > late_q
    print(f"  序盤n={mask_early.sum()}, 中盤n={mask_mid.sum()}, 終盤n={mask_late.sum()}")

    print("\n[5/5] 単変量 AUC (ペア差分、1P有利=高スコアが1P勝利に対応)")
    y = paired["won_1p"].values.astype(float)

    auc_honsen = phase_auc(y, paired["honsen_balance"].values, mask_early, mask_mid, mask_late,
                           "honsen_balance (単純版)")
    auc_tempo = phase_auc(y, paired["tempo_balance"].values, mask_early, mask_mid, mask_late,
                          "honsen_tempo_balance (時間窓)")
    auc_chain = phase_auc(y, paired["chain_diff"].values, mask_early, mask_mid, mask_late,
                          "current_max_chain 差分")
    auc_ojama = phase_auc(y, paired["ojama_diff"].values, mask_early, mask_mid, mask_late,
                          "board_ojama_count 差分")

    print("\n" + "=" * 70)
    print("比較サマリ:")
    print(f"{'指標':40s} {'全体':>6s} {'序盤':>6s} {'中盤':>6s} {'終盤':>6s}")
    print("-" * 70)
    for name, auc_dict in [
        ("honsen_balance (単純版)", auc_honsen),
        ("honsen_tempo_balance (時間窓)", auc_tempo),
        ("current_max_chain 差分", auc_chain),
        ("board_ojama_count 差分", auc_ojama),
    ]:
        row = f"{name:40s}"
        for k in ["全体", "序盤", "中盤", "終盤"]:
            v = auc_dict[k]
            row += f"  {v:.4f}" if not math.isnan(v) else "    N/A"
        print(row)
    print("=" * 70)
    mid_simple = auc_honsen.get("中盤", float("nan"))
    mid_tempo = auc_tempo.get("中盤", float("nan"))
    delta = mid_tempo - mid_simple if not (math.isnan(mid_tempo) or math.isnan(mid_simple)) else float("nan")
    delta_str = f"{delta:+.4f}" if not math.isnan(delta) else "N/A"
    print(f"\nテンポ核 中盤 ΔAUC = {delta_str}"
          f" (simple={mid_simple:.4f}, tempo={mid_tempo:.4f})")
    print("注: AUC > 0.5 = 1P有利側が1P勝利と対応。中盤目標 conn_pair_count=0.562。")
    print(f"テンポ核定数: SEC_PER_HAND={SEC_PER_HAND}秒 (実測中央値),"
          f" HANDS_PER_CHAIN_GAP={HANDS_PER_CHAIN_GAP}手/連鎖")


if __name__ == "__main__":
    main()
