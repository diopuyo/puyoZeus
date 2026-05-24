"""D-A: intro 区間が試合 1 として誤検出された matches.tsv を補正.

両側 STABLE 率 > 95% の動画は試合 1 が動画冒頭の intro/メニュー区間。
phase_e_recognition_review.tsv を読んで、対象動画の matches.tsv から
idx=1 を削除し、idx を再採番する。winners.tsv も同期更新。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.fix_intro_match_misdetection \
        --review-tsv data/verify/phase_e_recognition_review_v01-40.tsv \
        --review-tsv data/verify/phase_e_recognition_review_v41-94.tsv \
        --threshold 0.95 --apply
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()


def collect_anomalies(
    review_tsvs: list[Path], threshold: float,
) -> list[tuple[int, float]]:
    """両側 STABLE > threshold の動画 ID リスト."""
    out: list[tuple[int, float]] = []
    for tsv in review_tsvs:
        if not tsv.exists():
            print(f"[warn] missing review tsv: {tsv}")
            continue
        with tsv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                bs = float(r["both_stable_rate"])
                if bs >= threshold:
                    out.append((int(r["video_id"]), bs))
    return out


def patch_matches_tsv(
    matches_path: Path, dry_run: bool = True,
) -> tuple[int, int]:
    """matches.tsv の idx=1 を削除して idx を 1..N に振り直す."""
    if not matches_path.exists():
        return (0, 0)
    rows: list[dict] = []
    with matches_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    n_before = len(rows)
    if n_before == 0:
        return (0, 0)
    # idx=1 を削除し、残りを 1..N-1 に振り直す
    new_rows = rows[1:]
    for i, r in enumerate(new_rows, start=1):
        r["idx"] = str(i)
    n_after = len(new_rows)
    if not dry_run and n_after > 0:
        with matches_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=list(rows[0].keys()), delimiter="\t",
            )
            w.writeheader()
            w.writerows(new_rows)
    return (n_before, n_after)


def patch_winners_tsv(
    winners_path: Path, dry_run: bool = True,
) -> tuple[int, int]:
    """winners.tsv の idx=1 を削除して idx を 1..N に振り直す."""
    if not winners_path.exists():
        return (0, 0)
    rows: list[dict] = []
    with winners_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    n_before = len(rows)
    if n_before == 0:
        return (0, 0)
    new_rows = rows[1:]
    for i, r in enumerate(new_rows, start=1):
        r["idx"] = str(i)
    n_after = len(new_rows)
    if not dry_run and n_after > 0:
        with winners_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=list(rows[0].keys()), delimiter="\t",
            )
            w.writeheader()
            w.writerows(new_rows)
    return (n_before, n_after)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review-tsv", type=Path, action="append", required=True,
        help="phase_e_recognition_review_*.tsv (複数指定可)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.95,
        help="両側 STABLE 率の閾値 (この値以上は intro 誤検出と判定)",
    )
    parser.add_argument(
        "--boundary-root", type=Path,
        default=_ROOT / "data" / "verify" / "match_boundaries_v5",
    )
    parser.add_argument(
        "--winners-root", type=Path,
        default=_ROOT / "data" / "verify",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="実際に書き換える (デフォルトは dry-run)",
    )
    args = parser.parse_args()

    anomalies = collect_anomalies(args.review_tsv, args.threshold)
    if not anomalies:
        print("[done] no anomalies above threshold")
        return 0

    print(f"[anomaly] {len(anomalies)} videos with both_stable >= "
          f"{args.threshold:.0%}:")
    for vid, bs in anomalies:
        print(f"  v{vid:02d}: {bs:.1%}")
    print()

    n_changed = 0
    for vid, _ in anomalies:
        m_path = args.boundary_root / f"video_{vid:02d}" / "matches.tsv"
        w_path = args.winners_root / f"match_winners_v{vid:02d}.tsv"
        m_before, m_after = patch_matches_tsv(m_path, dry_run=not args.apply)
        w_before, w_after = patch_winners_tsv(w_path, dry_run=not args.apply)
        if m_before > 0 or w_before > 0:
            print(
                f"v{vid:02d}: matches {m_before} -> {m_after}, "
                f"winners {w_before} -> {w_after}"
            )
            n_changed += 1

    print()
    if args.apply:
        print(f"[applied] {n_changed} videos modified")
    else:
        print(f"[dry-run] {n_changed} videos would be modified. "
              "Add --apply to actually write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
