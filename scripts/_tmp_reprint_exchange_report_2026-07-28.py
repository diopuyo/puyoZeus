"""既存 exchange_dynamics_stats.csv からレポートのみ再出力する (再計算なし)。

#24 Step0 前提ゲート再検証 (2026-07-28) の一手目: 既存558イベントの
レポートを永続化するための一時スクリプト。measure_exchange_dynamics.py の
_print_report をそのまま再利用し、保存済み CSV を読み込んで出力するだけ。
既存スクリプト本体は変更しない (import のみ)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from scripts.measure_exchange_dynamics import OUTPUT_CSV, _print_report  # noqa: E402


def main() -> None:
    df = pd.read_csv(OUTPUT_CSV)
    print(f"[INFO] {OUTPUT_CSV} を読み込み: {len(df)} 行 (2026-07-22 生成、再計算なし)")
    _print_report(df)


if __name__ == "__main__":
    main()
