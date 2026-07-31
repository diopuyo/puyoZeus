"""相対位相(セグメント内進行率) win-AUC 評価の汎用版 (#43 段階1 c20/結合30本用)。

## 背景
_tmp_relphase_win_auc_2026-07-26.py は v10 (regen) 専用に STUDY_DIR/LABELED_WIN_CSV
をハードコードしていた。本スクリプトは同一ロジックを --study-dir (複数可) /
--labeled / --out-dir で外部化し、c20単独・v10+c20結合30本の評価に流用する。

既存 _tmp_relphase_win_auc_2026-07-26.py は変更しない (read-only 分析の使い捨て
スクリプトを追加するのみ、既存動作に影響なし)。

## 使い方
    PYTHONPATH=. ./venv/bin/python -m scripts._tmp_relphase_win_auc_generic_2026-07-28 \
        --study-dir data/verify/labeled_win_c20_2026-07-26/study \
        --labeled data/verify/labeled_win_c20_2026-07-26/labeled_win_c20.csv \
        --out-dir data/verify/win_eval_c20_2026-07-28/relphase_c20
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.model_indicator_win as miw  # noqa: E402

# ファイル名がハイフンを含み `import` 文の識別子構文では読めないため
# importlib.import_module (文字列引数、識別子制約なし) 経由で読み込む。
base = importlib.import_module("scripts._tmp_relphase_win_auc_2026-07-26")


def main() -> None:
    parser = argparse.ArgumentParser(description="相対位相 win-AUC 評価 (汎用版)")
    parser.add_argument("--study-dir", nargs="+", required=True, help="study CSV ディレクトリ (複数可)")
    parser.add_argument("--labeled", required=True, help="labeled_win.csv パス")
    parser.add_argument("--out-dir", required=True, help="出力先ディレクトリ")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. phase_map 構築 (study CSV からセグメント検出) ===")
    phase_map_parts = []
    for sd in args.study_dir:
        part = base.build_phase_map(Path(sd))
        phase_map_parts.append(part)
    phase_map = pd.concat(phase_map_parts, ignore_index=True)
    print(f"[phase_map] 結合後合計: {len(phase_map)} 行")
    phase_map.to_csv(out_dir / "relphase_phase_map.csv", index=False)

    print("\n=== 2. labeled_win.csv 読み込み・ペアリング ===")
    df = miw.load_labeled_csv(args.labeled)
    paired = miw.pair_sides_for_win(df, miw.DEFAULT_MAX_TDIFF)
    paired = base.merge_phase(paired, phase_map)

    results: dict[str, dict[str, float]] = {}
    results["relphase_all"] = base.evaluate(paired, exclude_contaminated=False)
    results["relphase_clean"] = base.evaluate(paired, exclude_contaminated=True)

    print("\n" + "=" * 70)
    print("  相対位相 win-AUC 結果")
    print("=" * 70)
    print(f"  {'条件':<20}  {'全体':>8}  {'序盤':>8}  {'中盤':>8}  {'終盤':>8}")
    for cond, res in results.items():
        print(
            f"  {cond:<20}  "
            + "  ".join(f"{res.get(ph, float('nan')):>8.4f}" for ph in ["全体", "序盤", "中盤", "終盤"])
        )

    rows = []
    for cond, res in results.items():
        row = {"condition": cond}
        row.update({f"auc_{k}": v for k, v in res.items()})
        rows.append(row)
    out_csv = out_dir / "relphase_auc_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[save] {out_csv}")


if __name__ == "__main__":
    main()
