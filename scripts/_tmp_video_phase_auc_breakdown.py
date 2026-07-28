"""終盤/中盤AUC下落の診断: video別(LeaveOneGroupOut)でAUC内訳を出す(read-only分析)。

目的(2026-07-23 コーディネーター依頼):
  「終盤AUC 0.676(07-05基準)→0.614(本日)」の下落が
    (a) 全動画で一様に下がっている(=真の劣化)のか
    (b) 特定1-2動画に起因する(=サンプル変動)のか
  を切り分ける。中盤も同様に内訳を出す。

手法:
  - model_indicator_win.py の既存関数(load_labeled_csv/pair_sides_for_win/
    build_features/_get_indicator_cols)をそのまま再利用。新規学習ロジックは
    LeaveOneGroupOut(video単位)で OOF 確率を得るのみ(GroupKFold 5-foldでは
    1fold に video 2本入るため単一video別の厳密なAUCが出せないための変更)。
  - HistGBC パラメータは model_indicator_win.GBC_PARAMS を完全流用(変更なし)。
  - 認識・ラベル生成スクリプトは一切変更しない。read-only 分析用の使い捨てスクリプト。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts._tmp_video_phase_auc_breakdown
"""
from __future__ import annotations

import sys
from pathlib import Path

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.model_indicator_win as miw  # noqa: E402

# デフォルトは従来通り(旧データ)。--labeled で新データ等への切替に対応(後方互換)。
LABELED_WIN_CSV = "data/indicators_v2/study/labeled_win.csv"


def _phase_masks(paired: pd.DataFrame) -> dict[str, np.ndarray]:
    """model_indicator_win と同じ手数三分位境界で序盤/中盤/終盤マスクを返す。"""
    tsumo = paired["tsumo_1p"].astype(float).values
    q33 = float(np.quantile(tsumo, miw.TSUMO_EARLY_RATIO))
    q67 = float(np.quantile(tsumo, miw.TSUMO_LATE_RATIO))
    return {
        "序盤": tsumo <= q33,
        "中盤": (tsumo > q33) & (tsumo <= q67),
        "終盤": tsumo > q67,
    }


def _run_logo_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """LeaveOneGroupOut(video単位)でOOF確率(1P勝ち確率)を返す。"""
    logo = LeaveOneGroupOut()
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        model = HistGradientBoostingClassifier(**miw.GBC_PARAMS)
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return oof


def _video_row(vid: str, y_v: np.ndarray, p_v: np.ndarray) -> dict:
    """1動画分の集計行(n, 勝敗内訳, AUC, 予測確率分布)を作る。"""
    n = len(y_v)
    n_win = int(y_v.sum())
    n_lose = n - n_win
    row: dict = {
        "video": vid, "n": n, "n_1p勝ち": n_win, "n_2p勝ち": n_lose,
        "pred_mean": float(np.mean(p_v)) if n else float("nan"),
        "pred_min": float(np.min(p_v)) if n else float("nan"),
        "pred_max": float(np.max(p_v)) if n else float("nan"),
    }
    if n >= 5 and len(np.unique(y_v)) > 1:
        row["auc"] = float(roc_auc_score(y_v, p_v))
        row["note"] = ""
    else:
        row["auc"] = float("nan")
        row["note"] = "単一クラスのみ(AUC計算不能)" if n > 0 else "データなし"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="video別/位相別AUC内訳診断")
    parser.add_argument(
        "--labeled", default=LABELED_WIN_CSV,
        help="labeled_win.csv パス (デフォルト: 旧データ、後方互換維持)",
    )
    parser.add_argument(
        "--out-csv", default=None,
        help="スコープ別プールAUC/video別統計サマリのCSV出力先 (省略時は出力しない、既存動作維持)",
    )
    args = parser.parse_args()

    print("=== データ読み込み・ペアリング ===")
    print(f"  labeled={args.labeled}")
    df = miw.load_labeled_csv(args.labeled)
    paired = miw.pair_sides_for_win(df, miw.DEFAULT_MAX_TDIFF)
    y = paired["won_1p"].astype(int).values
    groups = paired["video_id_1p"].values
    videos = sorted(np.unique(groups).tolist())
    print(f"  video数={len(videos)}: {videos}")

    indicator_cols = miw._get_indicator_cols(paired)
    feat_df = miw.build_features(paired, indicator_cols)
    X = feat_df.fillna(0.0).values.astype(float)
    print(f"  特徴量数={X.shape[1]} (indicator base={len(indicator_cols)})")

    print("\n=== LeaveOneGroupOut OOF 学習(video数分 = "
          f"{len(videos)} fold) ===")
    oof = _run_logo_oof(X, y, groups)

    masks = _phase_masks(paired)
    all_scope = {"全体": np.ones(len(y), dtype=bool), **masks}

    results: dict[str, list[dict]] = {}
    scope_summary_rows: list[dict] = []
    for scope_name, mask in all_scope.items():
        print(f"\n{'=' * 90}")
        print(f"  スコープ: {scope_name}  (全体n={int(mask.sum())})")
        print(f"{'=' * 90}")
        rows = []
        for vid in videos:
            vmask = mask & (groups == vid)
            row = _video_row(vid, y[vmask], oof[vmask])
            rows.append(row)
        rdf = pd.DataFrame(rows)
        results[scope_name] = rows
        print(rdf.to_string(index=False,
                             formatters={
                                 "pred_mean": "{:.3f}".format,
                                 "pred_min": "{:.3f}".format,
                                 "pred_max": "{:.3f}".format,
                                 "auc": lambda v: "  n/a " if np.isnan(v) else f"{v:.4f}",
                             }))
        # プールしたAUC(全video合算=フェーズ全体のAUC、既存レポートと同じ定義)
        pooled_auc = float("nan")
        valid = ~np.isnan(oof[mask])
        if valid.sum() >= 5 and len(np.unique(y[mask][valid])) > 1:
            pooled_auc = roc_auc_score(y[mask][valid], oof[mask][valid])
            print(f"  [プール全video合算AUC(LOGO)] = {pooled_auc:.4f}"
                  f"  (n={int(valid.sum())})")
        # AUC算出可能だったvideoのみでの分散
        aucs = [r["auc"] for r in rows if not np.isnan(r["auc"])]
        auc_mean = float(np.mean(aucs)) if aucs else float("nan")
        auc_std = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
        auc_min = float(min(aucs)) if aucs else float("nan")
        auc_max = float(max(aucs)) if aucs else float("nan")
        if aucs:
            print(f"  [AUC算出可能video={len(aucs)}本] "
                  f"平均={auc_mean:.4f}  std={auc_std:.4f}"
                  f"  最小={auc_min:.4f}  最大={auc_max:.4f}")
        scope_summary_rows.append({
            "scope": scope_name, "n": int(mask.sum()), "pooled_auc": pooled_auc,
            "n_videos_with_auc": len(aucs), "auc_mean": auc_mean, "auc_std": auc_std,
            "auc_min": auc_min, "auc_max": auc_max,
        })

    print("\n=== 完了 ===")

    # --out-csv 指定時のみ構造化サマリを保存 (省略時は従来通りstdoutのみ、後方互換)
    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(scope_summary_rows).to_csv(out_path, index=False)
        print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
