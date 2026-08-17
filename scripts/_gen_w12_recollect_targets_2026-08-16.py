"""W12根治P2: forecast真値列が欠損している63本npzを特定し、再収集対象
targets.tsv を生成する (2026-08-16)。

判定基準: `data/indicators_v2/boards_lean_phase_l_2026-08-11/{target_id}.npz`
に "ojama_net_balance" と "ojama_forecast" の両キーが揃っているかどうか
(`scripts/build_labeled_win_from_npz.py` の `_extract_ojama_truth_arrays`
と同一の判定式)。揃っていない = 収集時点で OjamaAccountingTracker 未配線
(npz収集64本目より前) の旧npz。

再収集対象の video_filename/video_id/tier/origin は既存の
`data/verify/regen_2026-08-11_manifest.tsv` (148本、同一動画セット) から
そのまま流用する (新規DL元調査は不要、元々同じ148本の一部)。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SRC_MANIFEST = PROJECT_ROOT / "data" / "verify" / "regen_2026-08-11_manifest.tsv"
SRC_NPZ_DIR = PROJECT_ROOT / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-11"
OUT_TSV = PROJECT_ROOT / "data" / "verify" / "w12_recollect_2026-08-16" / "targets.tsv"


def has_ojama_truth(npz_path: Path) -> bool:
    """build_labeled_win_from_npz.py の has_truth 判定式と完全一致させる。"""
    with np.load(npz_path) as d:
        return "ojama_net_balance" in d.files and "ojama_forecast" in d.files


def main() -> int:
    with SRC_MANIFEST.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    missing_rows = []
    for r in rows:
        npz_path = SRC_NPZ_DIR / f"{r['target_id']}.npz"
        if not npz_path.exists():
            print(f"[WARN] npz不在、判定不能: {npz_path}", file=sys.stderr)
            continue
        if not has_ojama_truth(npz_path):
            missing_rows.append(r)

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", encoding="utf-8", newline="\n") as f:
        f.write("target_id\tvideo_filename\tvideo_id\ttier\torigin\n")
        for r in missing_rows:
            f.write(
                f"{r['target_id']}\t{r['video_filename']}\t{r['video_id']}\t"
                f"{r['tier']}\t{r['origin']}\n"
            )

    print(f"[gen-targets] 総数={len(rows)} 欠損(再収集対象)={len(missing_rows)}")
    print(f"[gen-targets] 出力: {OUT_TSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
