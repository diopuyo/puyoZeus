"""強化アナリスト CLI runner (cycle 33+).

visualize_recognition.py の --dump-board-log で生成された JSONL を読み込み、
recognition_evaluator で物理推論ベース自動評価を実行する。

使用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \\
        --board-log logs/cycle_32d_board_log_v89m3.jsonl \\
        --report-out data/verify/cycle_32d_v89m3_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.recognition_evaluator import RecognitionEvaluator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-log", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument(
        "--verbose", action="store_true",
        help="全 violation を stdout に出力 (default: summary のみ)",
    )
    args = parser.parse_args()

    if not args.board_log.is_file():
        print(f"[error] board log not found: {args.board_log}", file=sys.stderr)
        return 1

    evaluator = RecognitionEvaluator()
    evaluator.load_jsonl(args.board_log)
    print(f"[eval] loaded {len(evaluator.entries)} frames")

    report = evaluator.generate_report()

    # summary 出力
    summary = report["summary"]
    print(f"\n=== Recognition Evaluation Report ===")
    print(f"Total frames: {report['total_frames']}")
    print(f"Total violations: {summary['total_violations']}")
    print(f"  critical: {summary['critical']}")
    print(f"  warning:  {summary['warning']}")
    print(f"  info:     {summary['info']}")
    print(f"\nVerdict: {report['verdict']}")
    print(f"\nBy metric (count, critical):")
    for metric, count in sorted(
        summary["by_metric"].items(), key=lambda x: -x[1],
    ):
        crit = summary["by_metric_critical"].get(metric, 0)
        print(f"  {metric:30s} : {count:5d} ({crit} critical)")

    if args.verbose:
        print("\n=== All violations ===")
        for v in report["violations"]:
            print(f"  frame={v['frame_idx']:>5} t={v['t_sec']:6.2f}s "
                  f"{v['side']} {v['metric']:30s} [{v['severity']}] "
                  f"{v['detail']}")

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[done] report → {args.report_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
