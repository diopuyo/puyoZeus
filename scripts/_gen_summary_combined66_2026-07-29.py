"""#43段階3: combined66 (m30+m20+c20) 評価結果の summary.md 自動生成 (2026-07-29)。

m20評価(data/verify/win_eval_m20_2026-07-28/summary.md)のプロトコルを踏襲し、
v10 / c20 / m20 / combined40 の既存結果と combined66 を横並び比較する表を
自動生成する。数値は全て既存/新規CSVから読み込む(手打ち転記はしない)。

fail-silent 防止: 参照する各CSVが存在しない場合は例外を投げて非ゼロ終了する
(欠損データのまま summary.md を生成しない)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._gen_summary_combined66_2026-07-29 \\
        --combined66-dir data/verify/win_eval_combined66_2026-07-29
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 既存結果の参照元 (2026-07-26/07-28 に確定済み、read-only参照のみ)
V10_RELPHASE_CSV = "data/verify/win_eval_regen_2026-07-26/relphase_auc_summary.csv"
C20_RELPHASE_CSV = "data/verify/win_eval_c20_2026-07-28/relphase_c20/relphase_auc_summary.csv"
M20_RELPHASE_CSV = "data/verify/win_eval_m20_2026-07-28/relphase_m20/relphase_auc_summary.csv"
COMBINED40_RELPHASE_CSV = "data/verify/win_eval_m20_2026-07-28/relphase_combined40/relphase_auc_summary.csv"

# 判定基準値 (user指定の比較基準、project_m20_eval_tier_settled_2026-07-28実績値)
M20_MID_BASELINE: float = 0.614
COMBINED40_MID_BASELINE: float = 0.674

PHASES = ["全体", "序盤", "中盤", "終盤"]


def _load_relphase_row(csv_path: str, condition: str = "relphase_all") -> dict[str, float]:
    """relphase_auc_summary.csv から指定条件行を読み込む(存在確認込み)。"""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"[FATAL] 参照CSVが存在しない: {path}")
    df = pd.read_csv(path)
    row = df[df["condition"] == condition]
    if len(row) == 0:
        raise ValueError(f"[FATAL] {path}: condition={condition} の行が無い")
    r = row.iloc[0]
    return {ph: float(r[f"auc_{ph}"]) for ph in PHASES}


def _load_logo_summary(csv_path: str) -> pd.DataFrame:
    """video別LOGO内訳サマリCSV(scope,pooled_auc,auc_mean,auc_std,auc_min,auc_max)を読み込む。"""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"[FATAL] LOGOサマリCSVが存在しない: {path}")
    return pd.read_csv(path)


def _load_midphase_importance(csv_path: str, top_n: int = 10) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"[FATAL] 中盤importance CSVが存在しない: {path}")
    df = pd.read_csv(path)
    return df.sort_values("importance_mean", ascending=False).head(top_n)


def _fmt_row(name: str, n_videos: int, vals: dict[str, float]) -> str:
    return f"| {name} | {n_videos} | " + " | ".join(f"{vals[ph]:.3f}" for ph in PHASES) + " |"


def build_summary(combined66_dir: Path) -> str:
    v10 = _load_relphase_row(V10_RELPHASE_CSV)
    c20 = _load_relphase_row(C20_RELPHASE_CSV)
    m20 = _load_relphase_row(M20_RELPHASE_CSV)
    combined40 = _load_relphase_row(COMBINED40_RELPHASE_CSV)
    combined66 = _load_relphase_row(str(combined66_dir / "relphase_combined66" / "relphase_auc_summary.csv"))

    logo_df = _load_logo_summary(str(combined66_dir / "combined66_video_breakdown_summary.csv"))
    mid_logo = logo_df[logo_df["scope"] == "中盤"].iloc[0]

    importance_df = _load_midphase_importance(str(combined66_dir / "combined66_midphase_importance.csv"))

    mid66 = combined66["中盤"]
    delta_vs_m20 = mid66 - M20_MID_BASELINE
    delta_vs_combined40 = mid66 - COMBINED40_MID_BASELINE

    lines: list[str] = []
    lines.append("# #43段階3: combined66(m30+m20+c20) 評価結果 (2026-07-29 自動生成)")
    lines.append("")
    lines.append("## 位相別AUC(相対位相=セグメント内進行率、主指標)比較")
    lines.append("")
    lines.append("| データセット | 動画数 | 全体 | 序盤 | 中盤 | 終盤 |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(_fmt_row("v10(旧マスター)", 10, v10))
    lines.append(_fmt_row("c20(チャレンジャー中心)", 20, c20))
    lines.append(_fmt_row("m20(純マスター)", 20, m20))
    lines.append(_fmt_row("combined40(m20+c20)", 40, combined40))
    lines.append(_fmt_row("**combined66(m30+m20+c20)**", 66, combined66))
    lines.append("")
    lines.append(f"- 中盤 combined66 vs m20基準({M20_MID_BASELINE:.3f}): "
                 f"{'+' if delta_vs_m20 >= 0 else ''}{delta_vs_m20:.3f}")
    lines.append(f"- 中盤 combined66 vs combined40基準({COMBINED40_MID_BASELINE:.3f}): "
                 f"{'+' if delta_vs_combined40 >= 0 else ''}{delta_vs_combined40:.3f}")
    if delta_vs_combined40 > 0:
        lines.append("- **判定: combined40からさらに改善、動画数増加の効果が継続している。**")
    elif abs(delta_vs_combined40) < 0.01:
        lines.append("- **判定: combined40と同水準、頭打ちの兆候(要注意)。**")
    else:
        lines.append("- **判定: combined40から悪化、動画数増加以外の要因(新動画の質等)を要確認。**")
    lines.append("")
    lines.append("## video別LOGO内訳(combined66、中盤)")
    lines.append("")
    lines.append("| 位相 | プールAUC | 平均 | std | 最小 | 最大 | AUC算出可能video数 |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in logo_df.iterrows():
        lines.append(
            f"| {r['scope']} | {r['pooled_auc']:.4f} | {r['auc_mean']:.4f} | "
            f"{r['auc_std']:.4f} | {r['auc_min']:.4f} | {r['auc_max']:.4f} | "
            f"{int(r['n_videos_with_auc'])} |"
        )
    lines.append("")
    lines.append(f"- 中盤 LOGO プールAUC = {mid_logo['pooled_auc']:.4f} "
                 f"(relphase版 {mid66:.4f} との差分は評価手法差=参考値)")
    lines.append("")
    lines.append("## 中盤限定 Permutation Importance top10(combined66)")
    lines.append("")
    for i, (_, r) in enumerate(importance_df.iterrows(), start=1):
        lines.append(f"{i}. {r['feature']} ({r['importance_mean']:+.4f})")
    lines.append("")
    lines.append("## ファイル")
    lines.append("")
    lines.append("- 相対位相: `relphase_combined66/relphase_auc_summary.csv`")
    lines.append("- 絶対境界: `combined66_run.log`, `combined66_importance*.csv`")
    lines.append("- LOGO: `combined66_video_breakdown.log`, `combined66_video_breakdown_summary.csv`")
    lines.append("- 中盤importance: `combined66_midphase_importance.csv`")
    lines.append("- 結合データ: `labeled_win_combined66.csv`"
                 "(m30+m20+c20 単純concat・スキーマ一致確認済、"
                 "`scripts/_build_labeled_win_combined66_2026-07-29.py`)")
    lines.append("")
    lines.append("## 限界")
    lines.append("")
    lines.append("- GroupKFold(video_id) / LOGO でリーク防止済。")
    lines.append("- 本summaryは完全自動生成(m30完走検知→自動学習チェイン)。人によるviz目視レビューは"
                 "別途 feedback_human_review_at_steps に基づき実施すること。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="combined66 summary.md 自動生成")
    parser.add_argument(
        "--combined66-dir", default="data/verify/win_eval_combined66_2026-07-29",
        help="combined66 評価出力ディレクトリ",
    )
    args = parser.parse_args()

    combined66_dir = Path(args.combined66_dir)
    try:
        summary_text = build_summary(combined66_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    out_path = combined66_dir / "summary.md"
    out_path.write_text(summary_text, encoding="utf-8")
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
