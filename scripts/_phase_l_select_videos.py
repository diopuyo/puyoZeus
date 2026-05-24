"""Phase L 動画選定: phase_e_dl_index.tsv のティア確認済から学習候補抽出."""
from __future__ import annotations
import csv
import json
from pathlib import Path


# 既存学習 (= cycle 32c)
TRAINED_VIDEOS = {29, 40, 51, 57, 70, 89, 95, 97}
# 真 holdout 候補 (= cycle 42-45 で v30 試行済)
HOLDOUT_CANDIDATES = {30, 50, 75}


def has_matches(vid: int) -> tuple[str | None, int]:
    """video_X の match_boundaries 存在確認 + 試合数."""
    for ver in ("v5", "v4"):
        p = Path(
            f"data/verify/match_boundaries_{ver}/video_{vid}/matches.tsv"
        )
        if p.is_file():
            with p.open() as f:
                rows = list(csv.reader(f, delimiter="\t"))
            return ver, len(rows) - 1
    return None, 0


def find_best_match(vid: int, version: str) -> dict | None:
    """各動画から最適試合 (= duration 60-150 秒) を選定."""
    p = Path(
        f"data/verify/match_boundaries_{version}/video_{vid}/matches.tsv"
    )
    if not p.is_file():
        return None
    with p.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    best = None
    for r in rows:
        try:
            dur = float(r["duration_sec"])
        except (KeyError, ValueError):
            continue
        if 60 <= dur <= 150:
            if best is None or dur > best["duration"]:
                best = {
                    "idx": int(r["idx"]),
                    "start": float(r["start_sec"]),
                    "end": float(r["end_sec"]),
                    "duration": dur,
                }
    return best


def main():
    print("=" * 80)
    print("Phase L 動画選定 (= ティア確認済 46 動画から)")
    print("=" * 80)
    selected = []
    skipped = []
    for vid in range(29, 99):  # phase_e_dl_index 範囲 (v94 まで)
        ver, n_matches = has_matches(vid)
        if ver is None:
            continue
        best = find_best_match(vid, ver)
        if best is None:
            skipped.append((vid, "no 60-150s match"))
            continue
        category = "train"
        if vid in HOLDOUT_CANDIDATES:
            category = "holdout"
        elif vid in TRAINED_VIDEOS:
            category = "existing_train"
        selected.append({
            "vid": vid,
            "version": ver,
            "match_idx": best["idx"],
            "start_sec": best["start"],
            "end_sec": best["end"],
            "duration": best["duration"],
            "category": category,
        })
    print(f"\n選定動画: {len(selected)} 本 / 候補 46 本")
    print(f"Skipped: {len(skipped)} 本")
    print()
    print(f"{'vid':>5} {'ver':<3} {'match':>5} {'start':>7} {'end':>7} {'dur':>6} {'category':<14}")
    print("-" * 60)
    for s in selected:
        print(f"{s['vid']:>5} {s['version']:<3} {s['match_idx']:>5} "
              f"{s['start_sec']:>7.0f} {s['end_sec']:>7.0f} {s['duration']:>6.0f} "
              f"{s['category']:<14}")
    # JSON 保存
    out = Path("data/verify/phase_l_video_selection.json")
    out.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n→ saved {out}")
    # 統計
    cat_count = {}
    for s in selected:
        cat_count[s["category"]] = cat_count.get(s["category"], 0) + 1
    print()
    print("カテゴリ別:")
    for c, n in sorted(cat_count.items()):
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
