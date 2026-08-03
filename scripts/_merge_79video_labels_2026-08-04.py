"""66動画分(合成込み)+13動画分(合成込み)を統合し、79動画統合CSVを作る (2026-08-04)。

既存資産の再計算はしない (66動画分はそのまま読み込むだけ、13動画分のみ
新規計算した結果をマージする)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

EXISTING_66_AUG_CSV = Path("data/indicators_v2/exchange_labels_regen_synth_aug_2026-08-03.csv")
NEW_13_AUG_CSV = Path("data/indicators_v2/exchange_labels_expand13_aug_2026-08-04.csv")
OUT_79_AUG_CSV = Path("data/indicators_v2/exchange_labels_regen_synth79_aug_2026-08-04.csv")

EXISTING_66_BASE_CSV = Path("data/indicators_v2/exchange_labels_regen_synth_2026-08-03.csv")
NEW_13_BASE_CSV = Path("data/indicators_v2/exchange_labels_expand13_2026-08-04.csv")
OUT_79_BASE_CSV = Path("data/indicators_v2/exchange_labels_regen_synth79_2026-08-04.csv")


def merge_and_save(existing_path: Path, new_path: Path, out_path: Path, label: str) -> None:
    """既存CSV+新規CSVを結合して保存する (重複video_idが無いことを確認)。"""
    existing = pd.read_csv(existing_path)
    new = pd.read_csv(new_path)
    overlap = set(existing["video_id"].unique()) & set(new["video_id"].unique())
    if overlap:
        print(f"[ERROR] {label}: 既存と新規でvideo_idが重複しています: {overlap}", file=sys.stderr)
        sys.exit(1)
    merged = pd.concat([existing, new], ignore_index=True)
    merged.to_csv(out_path, index=False)
    print(f"[{label}] 既存{len(existing)}行(動画{existing['video_id'].nunique()}本)"
          f" + 新規{len(new)}行(動画{new['video_id'].nunique()}本)"
          f" = 統合{len(merged)}行(動画{merged['video_id'].nunique()}本) -> {out_path}")


def main() -> None:
    merge_and_save(EXISTING_66_BASE_CSV, NEW_13_BASE_CSV, OUT_79_BASE_CSV, "base")
    merge_and_save(EXISTING_66_AUG_CSV, NEW_13_AUG_CSV, OUT_79_AUG_CSV, "aug")


if __name__ == "__main__":
    main()
