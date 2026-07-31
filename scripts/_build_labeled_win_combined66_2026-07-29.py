"""#43段階3: combined66 (m30+m20+c20) labeled_win CSV結合スクリプト (2026-07-29)。

## 背景
labeled_win_combined40.csv (m20+c20) は 2026-07-28 に単純concat(スキーマ一致確認済)
で作られた (project_m20_eval_tier_settled_2026-07-28)。本スクリプトはそのプロトコルを
m30(マスター級残り26本)を加えた66本版に一般化する。m30 収集完走後の自動学習
チェイン (_wait_and_train_combined66_2026-07-29.sh) から呼ばれる想定。

label_win_from_winners.py 等は変更しない (read-only 結合のみ)。

## fail-silent 防止方針
各ソースCSVについて以下を検証し、いずれか不成立なら例外を投げて非ゼロ終了する
(黙って空/不整合データで学習に進めない):
  - ファイルが存在するか
  - 行数が 0 でないか
  - video_id 列が存在し video 数が 0 でないか
  - 動画あたり行数が m20実績(2026-07-28, 58547行/20本)の半分以上あるか
    (収集ジョブの一部failで大量欠損したケースを検出する)
  - 全ソース間で列名が完全一致するか (スキーマ不一致は結合できない)

## 使い方
    PYTHONPATH=. ./venv/bin/python -m scripts._build_labeled_win_combined66_2026-07-29 \\
        --sources data/verify/labeled_win_m30_2026-07-28/labeled_win_m30.csv \\
                  data/verify/labeled_win_m20_2026-07-28/labeled_win_m20.csv \\
                  data/verify/labeled_win_c20_2026-07-26/labeled_win_c20.csv \\
        --out data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# m20実績(2026-07-28, project_m20_eval_tier_settled_2026-07-28)より、
# 動画1本あたり平均行数の基準値。この比の半分未満なら収集不完全の疑いとして拒否する。
M20_ROWS_PER_VIDEO_BASELINE: float = 58547.0 / 20.0
MIN_ROWS_PER_VIDEO_RATIO: float = 0.5


def _load_and_check(path: Path) -> pd.DataFrame:
    """1ソースCSVを読み込み、存在/非空/収集完全性を検証する(不成立なら例外)。"""
    if not path.exists():
        raise FileNotFoundError(f"[FATAL] ソースCSVが存在しない: {path}")
    df = pd.read_csv(path)
    if len(df) == 0:
        raise ValueError(f"[FATAL] ソースCSVが空: {path}")
    if "video_id" not in df.columns:
        raise ValueError(f"[FATAL] video_id 列が無い: {path}")
    n_videos = df["video_id"].nunique()
    if n_videos == 0:
        raise ValueError(f"[FATAL] video_id が空: {path}")
    per_video = len(df) / n_videos
    min_required = M20_ROWS_PER_VIDEO_BASELINE * MIN_ROWS_PER_VIDEO_RATIO
    if per_video < min_required:
        raise ValueError(
            f"[FATAL] {path}: 動画あたり行数 {per_video:.0f} が基準 {min_required:.0f} "
            "未満 (収集不完全の疑い、黙って続行しない)"
        )
    print(f"  [OK] {path}: {len(df)} 行, video数={n_videos}, 動画あたり={per_video:.0f}")
    return df


def build_combined(sources: list[str]) -> pd.DataFrame:
    """複数ソースCSVをスキーマ一致確認の上で結合する。"""
    dfs: list[pd.DataFrame] = []
    ref_cols: list[str] | None = None
    ref_path: Path | None = None
    for s in sources:
        path = Path(s)
        df = _load_and_check(path)
        if ref_cols is None:
            ref_cols = list(df.columns)
            ref_path = path
        elif list(df.columns) != ref_cols:
            raise ValueError(
                f"[FATAL] スキーマ不一致: {ref_path} (列数={len(ref_cols)}) vs "
                f"{path} (列数={len(df.columns)})"
            )
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="combined66 labeled_win CSV結合")
    parser.add_argument("--sources", nargs="+", required=True, help="結合元CSVパス(複数)")
    parser.add_argument("--out", required=True, help="出力CSVパス")
    args = parser.parse_args()

    print("=== combined66 結合開始 ===")
    try:
        combined = build_combined(args.sources)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    n_videos_total = combined["video_id"].nunique()
    print(f"[結合完了] 合計 {len(combined)} 行, video数={n_videos_total}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
