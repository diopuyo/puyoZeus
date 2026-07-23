"""反復6 (2026-07-23): recognition_physics_review.py の per-video JSON をマージする。

3動画を並列プロセス (--video-stem 指定) で実行した結果 JSON を結合し、
_summarize() 相当のサマリを再計算して1つの JSON にまとめる。
src/ は変更しない (読み取り専用の集計スクリプト)。

使い方:
    PYTHONPATH=. python -m scripts._merge_physics_review_json \
        out1.json out2.json out3.json --output merged.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from scripts.recognition_physics_review import (  # noqa: E402
    _print_summary_table, _summarize,
)


def _parse_args() -> argparse.Namespace:
    """CLI引数をパースする。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="マージする per-video JSON パス群")
    ap.add_argument("--output", required=True, help="マージ結果の出力先 JSON パス")
    return ap.parse_args()


def main() -> None:
    """複数の per-video JSON を読み込み、videos を結合してサマリを再計算する。"""
    args = _parse_args()
    merged_videos: list[dict] = []
    for path_str in args.inputs:
        with open(path_str, encoding="utf-8") as f:
            data = json.load(f)
        merged_videos.extend(data["videos"])
        print(f"[merge] {path_str}: {len(data['videos'])} 動画分を取り込み")

    summary = _summarize(merged_videos)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "videos": merged_videos}, f,
                   ensure_ascii=False, indent=2, default=str)
    print(f"\n[DONE] {out_path} に保存しました ({len(merged_videos)} 動画分)")
    _print_summary_table(summary)


if __name__ == "__main__":
    main()
