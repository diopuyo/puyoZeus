"""subset42 (2026-08-19) 用: 旧CSVから対応42動画分だけを抽出する。

`scripts/_ab_stage_compare_2026-08-18.py` の `extract_old_csv_subset` と同じ
ロジック (チャンク読み込みで該当video_idのみ抽出、旧CSVはビルドし直さない)。
ファイル名にハイフンを含むモジュールは `import` できないため、ロジックを
直接複製する (薄い複製、2026-08-19)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

OLD_CSV_CHUNK_SIZE: int = 200_000


def extract_old_csv_subset(old_csv: Path, target_ids: list[str], out_csv: Path) -> int:
    """旧CSVから target_ids 該当行だけをチャンク読み込みで抽出する (CSVビルド不要)。"""
    wanted = {f"video_{tid}" for tid in target_ids}
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    first = True
    for chunk in pd.read_csv(old_csv, chunksize=OLD_CSV_CHUNK_SIZE):
        sub = chunk[chunk["video_id"].isin(wanted)]
        if len(sub) == 0:
            continue
        sub.to_csv(out_csv, mode="w" if first else "a", header=first, index=False)
        first = False
        n_rows += len(sub)
    found: set = set()
    if not first:
        found = set(pd.read_csv(out_csv, usecols=["video_id"])["video_id"].unique())
    missing = wanted - found
    if missing:
        raise RuntimeError(f"旧CSVに見つからない video_id があります: {sorted(missing)}")
    print(f"[extract_old] {out_csv} に {n_rows}行 ({len(target_ids)}本) を書き出し", flush=True)
    return n_rows

TARGET_IDS = [
    "29", "36", "39", "52", "c100", "c101", "c102", "c103", "c104", "c105",
    "c106", "c107", "c108", "c109", "c11", "c110", "c111", "c112", "c113",
    "c114", "c115", "c116", "c117", "c118", "c119", "c125", "c126", "c127",
    "c128", "c129", "c13", "c130", "c131", "c132", "c133", "c134", "c135",
    "c136", "c137", "c96s1", "c96s2", "c96s3",
]

OLD_CSV = _PROJ_ROOT / "data/verify/labeled_win_full148_2026-08-14/labeled_win_full148.csv"
OUT_CSV = _PROJ_ROOT / "data/verify/retrain_subset42_2026-08-19/old/labeled_win_old.csv"


def main() -> int:
    n = extract_old_csv_subset(OLD_CSV, TARGET_IDS, OUT_CSV)
    print(f"[done] {n}行 -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
