"""66動画版 vs 79動画版 の位相別サンプル数+三つ巴主要数値の対比報告 (2026-08-04)。

既存の comparison_report.md (旧66動画=step3、新79動画=synth79) をそのまま
読み込んで主要数値(全体/序/中/終のrho・AUC、併用列のみ)を抜粋し、
位相別サンプル数の変化を併記する (再計算しない、既存資産の突合のみ)。
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

OLD_AUG_CSV = Path("data/indicators_v2/exchange_labels_regen_synth_aug_2026-08-03.csv")
NEW_AUG_CSV = Path("data/indicators_v2/exchange_labels_regen_synth79_aug_2026-08-04.csv")
OLD_REPORT_MD = Path("data/verify/exchange_triple_comparison_step3_2026-08-02/comparison_report.md")
NEW_REPORT_MD = Path("data/verify/exchange_triple_comparison_synth79_2026-08-04/comparison_report.md")


def phase_counts(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    return df["phase"].value_counts()


def extract_stacking_rows(report_md: Path) -> pd.DataFrame:
    """comparison_report.md 冒頭の「範囲別×予測器 指標一覧」表から「併用
    (スタッキング)」行のみを抜粋する (DeLong/bootstrap の別表と列数で区別)。
    """
    text = report_md.read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 12 or cells[1] != "併用(スタッキング)":
            continue
        rows.append(cells)
    cols = ["範囲", "予測器", "n", "rho", "rho_lo", "rho_hi", "MAE", "MAE_lo", "MAE_hi", "AUC", "Brier", "備考"]
    return pd.DataFrame(rows, columns=cols)


def main() -> None:
    print("=== 位相別サンプル数 (66動画版 vs 79動画版) ===")
    old_counts = phase_counts(OLD_AUG_CSV)
    new_counts = phase_counts(NEW_AUG_CSV)
    for phase in ["序", "中", "終"]:
        old_n = int(old_counts.get(phase, 0))
        new_n = int(new_counts.get(phase, 0))
        delta_pct = (new_n - old_n) / old_n * 100 if old_n > 0 else float("nan")
        print(f"  {phase}: 66動画={old_n} -> 79動画={new_n} ({delta_pct:+.1f}%)")
    print(f"  合計: 66動画={int(old_counts.sum())} -> 79動画={int(new_counts.sum())}")

    print("\n=== 三つ巴主要数値 (併用(スタッキング)、66動画版 vs 79動画版) ===")
    if OLD_REPORT_MD.exists() and NEW_REPORT_MD.exists():
        old_rows = extract_stacking_rows(OLD_REPORT_MD)
        new_rows = extract_stacking_rows(NEW_REPORT_MD)
        print("--- 66動画版(旧) ---")
        print(old_rows.to_string(index=False))
        print("--- 79動画版(新) ---")
        print(new_rows.to_string(index=False))
    else:
        print(f"  [警告] レポートが見つかりません: old={OLD_REPORT_MD.exists()} new={NEW_REPORT_MD.exists()}")


if __name__ == "__main__":
    main()
