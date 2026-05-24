"""seed dataset 品質 audit CLI.

使用例:
    PYTHONPATH=. python -m scripts.evaluate_seed_quality \\
        --seed-root data/phase_l/seeds \\
        --report-out data/verify/seed_quality_phase_l.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.seed_quality_evaluator import aggregate_reports, evaluate_seed_dir


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seed-root", type=Path, required=True,
        help="seed dir の親 (= data/phase_l/seeds 等)",
    )
    p.add_argument(
        "--report-out", type=Path, required=True,
        help="集計 JSON 出力 path",
    )
    p.add_argument(
        "--per-video-out", type=Path, default=None,
        help="動画別 report dump (= JSON 1 行ずつ)",
    )
    args = p.parse_args()
    seed_dirs = sorted(
        d for d in args.seed_root.iterdir() if d.is_dir() and (d / "cell.jsonl").exists()
    )
    if not seed_dirs:
        print(f"[error] no seed dirs in {args.seed_root}")
        return 1
    reports = []
    for d in seed_dirs:
        r = evaluate_seed_dir(d)
        reports.append(r)
        c = sum(r.per_color_counts.values())
        print(
            f"  {r.video_id:<20} samples={c:>6} purity={r.overall_purity:.3f} "
            f"per_color={r.per_color_purity}",
        )
    summary = aggregate_reports(reports)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    out_obj = {
        "summary": summary,
        "per_video": [r.to_json() for r in reports],
    }
    args.report_out.write_text(
        json.dumps(out_obj, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"[done] wrote {args.report_out}")
    print(f"=== overall ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.per_video_out:
        with args.per_video_out.open("w", encoding="utf-8") as f:
            for r in reports:
                f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
