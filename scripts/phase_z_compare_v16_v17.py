"""CNN v16 と v17 の cross_video 結果を比較し、改善幅を集計。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_summary(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            try:
                out[r["video"]] = {
                    "hard": int(r["hard"]),
                    "acc": float(r["est_accuracy"]),
                }
            except (KeyError, ValueError):
                pass
    return out


def main() -> int:
    v16_path = (
        _ROOT / "data/verify/phase_z_review/cross_video/summary.tsv"
    )
    v17_path = (
        _ROOT / "data/verify/phase_z_review/cross_video_v17/summary.tsv"
    )
    v17b_path = (
        _ROOT / "data/verify/phase_z_review/cross_video_v17b/summary.tsv"
    )
    v16 = load_summary(v16_path)
    v17 = load_summary(v17_path)
    v17b = load_summary(v17b_path)
    if not v16:
        print(f"ERROR: v16 summary not found at {v16_path}")
        return 1

    have_v17b = bool(v17b)
    have_v17 = bool(v17)

    if have_v17b:
        print(f"{'video':<5} {'v16':<8} {'v17':<8} {'v17b':<8} "
              f"{'Δv17b':<8}")
        print("-" * 50)
    elif have_v17:
        print(f"{'video':<5} {'v16':<8} {'v17':<8} {'Δv17':<8}")
        print("-" * 35)
    deltas_v17 = []
    deltas_v17b = []
    v16_hards = []
    v17_hards = []
    v17b_hards = []
    for video in sorted(v16.keys()):
        v16_d = v16[video]
        line = f"{video:<5} {v16_d['acc']:<8.3f}"
        if have_v17:
            v17_d = v17.get(video, {"acc": 0.0, "hard": 0})
            d = v17_d["acc"] - v16_d["acc"]
            deltas_v17.append(d)
            v17_hards.append(v17_d["hard"])
            line += f" {v17_d['acc']:<8.3f}"
            if not have_v17b:
                sign = "+" if d >= 0 else ""
                line += f" {sign}{d:<7.3f}"
        if have_v17b:
            v17b_d = v17b.get(video, {"acc": 0.0, "hard": 0})
            d = v17b_d["acc"] - v16_d["acc"]
            deltas_v17b.append(d)
            v17b_hards.append(v17b_d["hard"])
            sign = "+" if d >= 0 else ""
            line += f" {v17b_d['acc']:<8.3f} {sign}{d:<7.3f}"
        v16_hards.append(v16_d["hard"])
        print(line)
    print("-" * (50 if have_v17b else 35))
    if deltas_v17:
        print(f"\nv17 平均改善幅 (vs v16): {sum(deltas_v17) / len(deltas_v17):+.3f}pt")
        print(f"  改善 {sum(1 for d in deltas_v17 if d > 0)} / "
              f"悪化 {sum(1 for d in deltas_v17 if d < 0)}")
    if deltas_v17b:
        print(f"\nv17b 平均改善幅 (vs v16): {sum(deltas_v17b) / len(deltas_v17b):+.3f}pt")
        print(f"  改善 {sum(1 for d in deltas_v17b if d > 0)} / "
              f"悪化 {sum(1 for d in deltas_v17b if d < 0)}")
        v16_total = sum(v16_hards)
        v17b_total = sum(v17b_hards)
        print(f"\nhard 総数: v16={v16_total} → v17b={v17b_total} "
              f"(削減 {v16_total - v17b_total} = "
              f"{100 * (v16_total - v17b_total) / v16_total:+.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
