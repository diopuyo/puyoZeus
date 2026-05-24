"""next_acceptance 統計を確認するための一時スクリプト。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/training/match_features_v2.csv")


def main() -> int:
    df = pd.read_csv(CSV_PATH)
    print(f"rows={len(df)}, cols={len(df.columns)}")
    col = "next_acceptance"
    print(
        f"std={df[col].std():.6f}, "
        f"mean={df[col].mean():.6f}, "
        f"unique={df[col].nunique()}, "
        f"min={df[col].min():.4f}, max={df[col].max():.4f}",
    )
    print("video_ids:", sorted(df.video_id.unique().tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
