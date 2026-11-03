"""B3 レビュー動画の CLI: 元動画 + レビュー入力 CSV → データ帯付き動画 + フレーム別 CSV (+ スマホ分割)。

使い方 (WSL):
  PYTHONPATH=. ./venv/bin/python -m scripts.b3_review_render --video V.mp4 --input rows.csv \
      --out-dir D:/.../review_x --start-sec 0 --end-sec 600 --mobile
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.b3_review_input import load_review_input
from src.b3_review_video import make_mobile, render_review


def main(argv: list[str] | None = None) -> int:
    """入力検査 → 描画 → 自己検収 JSON → (任意) スマホ分割。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True, help="docs/B3_REVIEW_INPUT.md の CSV")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stem", default="b3_review")
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, default=None)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--mobile", action="store_true", help="1280縮小・4.5分分割も作る")
    args = ap.parse_args(argv)
    rows = load_review_input(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = args.out_dir / f"{args.stem}.mp4"
    result = render_review(args.video, rows, mp4, args.out_dir / f"{args.stem}_frames.csv",
                           args.start_sec, args.end_sec, with_audio=not args.no_audio)
    if args.mobile:
        result["mobile"] = make_mobile(mp4, args.out_dir / "mobile", args.stem)
    (args.out_dir / f"{args.stem}_complete.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("frames", "fps", "encoder_returncode")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
