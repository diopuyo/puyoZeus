"""3 者一致 DROP 検知 runner スクリプト (2026-06-03).

board_log JSONL に対して check_three_way_sudden_drop を適用し、
動画別・サイド別の発火件数・t_sec・diff を出力する。

使用方法:
    python -m scripts.run_three_way_drop_check [jsonl_path ...]

引数なしの場合は既定 3 本のログを処理する:
    - data/verify/viz/v89_match01_D_2026-06-03.jsonl
    - data/verify/viz/v89_match02_D_2026-06-03.jsonl
    - data/verify/viz/v70_match02_deferred_2026-06-03.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.recognition_evaluator import RecognitionEvaluator, THREE_WAY_DROP_THRESHOLD

# 既定の解析対象 board_log 3 本
DEFAULT_LOGS: list[str] = [
    "data/verify/viz/v89_match01_D_2026-06-03.jsonl",
    "data/verify/viz/v89_match02_D_2026-06-03.jsonl",
    "data/verify/viz/v70_match02_deferred_2026-06-03.jsonl",
]


def run_single_log(log_path: Path) -> None:
    """1 本の board_log を解析し、 発火結果を標準出力に出す。"""
    evaluator = RecognitionEvaluator()
    evaluator.load_jsonl(log_path)
    total_frames = len(evaluator.entries)

    print(f"\n{'='*60}")
    print(f"[解析対象] {log_path.name}  (total frames={total_frames})")
    print(f"[閾値] THREE_WAY_DROP_THRESHOLD = {THREE_WAY_DROP_THRESHOLD}")

    for side in ("1P", "2P"):
        violations = evaluator.check_three_way_sudden_drop(side)
        print(f"\n  {side}: 発火 {len(violations)} 件")
        if violations:
            for v in violations:
                ex = v.extra
                print(
                    f"    t={v.t_sec:7.2f}s  frame={v.frame_idx:5d}  "
                    f"diff={ex['diff']:+3d}  "
                    f"{ex['prev_3way_n']}→{ex['cur_3way_n']}  "
                    f"(prev_frame={ex['prev_frame']})"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="check_three_way_sudden_drop を board_log に適用する runner"
    )
    parser.add_argument(
        "logs",
        nargs="*",
        help="解析する JSONL ファイルのパス (省略時は既定 3 本を使用)",
    )
    args = parser.parse_args()

    log_paths: list[Path]
    if args.logs:
        log_paths = [Path(p) for p in args.logs]
    else:
        log_paths = [_PROJECT_ROOT / p for p in DEFAULT_LOGS]

    for log_path in log_paths:
        if not log_path.exists():
            print(f"[SKIP] ファイルが存在しません: {log_path}", file=sys.stderr)
            continue
        run_single_log(log_path)

    print("\n[完了]")


if __name__ == "__main__":
    main()
