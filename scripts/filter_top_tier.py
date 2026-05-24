"""学習 CSV から上級者 (v29-v94) かつ解像度 720p+ のみ抽出する.

phase_e_dl_index.tsv に記載の 45 動画 = マスター級以上 (第2回新おいうリーグ)。
v01-v28 は古い DL 流でティア未確認のため除外する。

2026-05-11 サイクル65: 360p 動画を学習対象外に. 動画 file の実解像度を確認し、
720p 未満は除外. 動画 file が data/frames/video_XX.mp4 にないものはスキップ.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


MIN_RESOLUTION_HEIGHT: int = 720


def _check_video_height(video_path: Path) -> int | None:
    """動画 file の高さ (px) を返す. file 不在なら None."""
    if not video_path.exists():
        return None
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return h if h > 0 else None
    except Exception:
        return None


def filter_csv(
    src: Path, dst: Path, min_id: int = 29,
    video_root: Path = Path("data/frames"),
    require_min_height: int = MIN_RESOLUTION_HEIGHT,
) -> tuple[int, int, int]:
    """video_id 数値 >= min_id かつ 動画解像度 >= require_min_height の行のみコピー."""
    with src.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        kept_rows: list[dict] = []
        n_total = 0
        n_low_res = 0
        for r in reader:
            n_total += 1
            try:
                vid = int(r["video_id"])
            except (ValueError, KeyError):
                continue
            if vid < min_id:
                continue
            # 解像度チェック (file 不在ならスキップでなく warning)
            video_path = video_root / f"video_{vid:02d}.mp4"
            h = _check_video_height(video_path)
            if h is None:
                # 動画 file 不在 → conservative に keep (= DL 後判定)
                kept_rows.append(r)
                continue
            if h < require_min_height:
                n_low_res += 1
                print(
                    f"[filter] excluded v{vid:02d} (height={h} < {require_min_height})",
                    file=sys.stderr,
                )
                continue
            kept_rows.append(r)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)
    return n_total, len(kept_rows), n_low_res


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--min-id", type=int, default=29)
    parser.add_argument("--video-root", type=Path, default=Path("data/frames"))
    parser.add_argument(
        "--require-min-height", type=int, default=MIN_RESOLUTION_HEIGHT,
        help="解像度未満は学習対象から除外 (default: 720)",
    )
    args = parser.parse_args()
    n_total, n_kept, n_low_res = filter_csv(
        args.src, args.dst, args.min_id,
        args.video_root, args.require_min_height,
    )
    print(
        f"[filter] {args.src.name}: {n_total} -> {n_kept} rows "
        f"(min_id={args.min_id}, min_height={args.require_min_height}, "
        f"low_res_excluded={n_low_res})",
    )
    print(f"[save] {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
