"""video_c96 の is_game 時系列から 3 シリーズ区間を推定する.

入力: scripts/_scan_game_screens_c96_2026-08-08.py が出力した
      data/verify/c96_split_2026-08-08/scan_is_game.tsv (t_sec, is_game, panel_score)

処理:
    1. is_game==1 の連続区間 (raw block) を抽出する。
    2. block 間ギャップが GAP_MERGE_SEC 以下なら同一シリーズとして結合する
       (トーク挟みでパネルが一時的に消える場合を吸収)。
    3. 結合後の総スパンが MIN_SERIES_SPAN_SEC 未満の group は
       ハイライトモンタージュ / 順位表画面等のノイズとして除外する。
    4. 残った group の前後に MARGIN_SEC の余白を付けて最終区間とする。

出力: data/verify/c96_split_2026-08-08/series_segments.tsv
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

IN_TSV: Path = (
    _ROOT / "data" / "verify" / "c96_split_2026-08-08" / "scan_is_game.tsv"
)
OUT_TSV: Path = (
    _ROOT / "data" / "verify" / "c96_split_2026-08-08" / "series_segments.tsv"
)

# トーク挟みでギャップができても同一シリーズとして結合する上限 (秒)。
# user 指示「ギャップ≤5分は同一シリーズとして結合」に基づく。
GAP_MERGE_SEC: float = 300.0
# 結合後の総スパンがこれ未満なら (ハイライトモンタージュ/順位表画面等の)
# ノイズ block として除外する。実際の 1 シリーズは 30 本先取で
# 数十分〜2 時間規模になる想定 (user 背景情報より)。
MIN_SERIES_SPAN_SEC: float = 600.0
# 切り出しクリップの前後マージン (秒)。
MARGIN_SEC: float = 30.0


@dataclass
class Block:
    start: float
    end: float  # 最後に is_game=1 だったサンプルの t_sec


def load_samples(path: Path) -> list[tuple[float, bool]]:
    rows: list[tuple[float, bool]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for r in reader:
            if len(r) < 2:
                continue
            rows.append((float(r[0]), bool(int(r[1]))))
    return rows


def extract_raw_blocks(
    samples: list[tuple[float, bool]], sample_interval: float,
) -> list[Block]:
    """is_game==1 の連続区間を抽出する."""
    blocks: list[Block] = []
    cur_start: float | None = None
    prev_t: float | None = None
    for t, is_game in samples:
        if is_game:
            if cur_start is None:
                cur_start = t
            prev_t = t
        else:
            if cur_start is not None:
                blocks.append(Block(start=cur_start, end=prev_t))
                cur_start = None
    if cur_start is not None and prev_t is not None:
        blocks.append(Block(start=cur_start, end=prev_t))
    return blocks


def merge_blocks(blocks: list[Block], gap_merge_sec: float) -> list[Block]:
    """ギャップ <= gap_merge_sec の block を結合する."""
    if not blocks:
        return []
    merged = [blocks[0]]
    for b in blocks[1:]:
        last = merged[-1]
        if b.start - last.end <= gap_merge_sec:
            merged[-1] = Block(start=last.start, end=b.end)
        else:
            merged.append(b)
    return merged


def main() -> int:
    if not IN_TSV.exists():
        print(f"[error] scan tsv not found: {IN_TSV}", file=sys.stderr)
        return 1

    samples = load_samples(IN_TSV)
    if len(samples) < 2:
        print("[error] too few samples", file=sys.stderr)
        return 1
    sample_interval = samples[1][0] - samples[0][0]

    raw_blocks = extract_raw_blocks(samples, sample_interval)
    print(f"[raw blocks] {len(raw_blocks)} 件")
    for b in raw_blocks:
        print(f"  raw: {b.start:8.1f} - {b.end:8.1f}  ({b.end - b.start:7.1f}s)")

    merged = merge_blocks(raw_blocks, GAP_MERGE_SEC)
    print(f"\n[merged blocks (gap<={GAP_MERGE_SEC:.0f}s)] {len(merged)} 件")
    for b in merged:
        span = b.end - b.start
        flag = "OK" if span >= MIN_SERIES_SPAN_SEC else "skip(短すぎ)"
        print(f"  merged: {b.start:8.1f} - {b.end:8.1f}  ({span:7.1f}s)  {flag}")

    series = [b for b in merged if (b.end - b.start) >= MIN_SERIES_SPAN_SEC]
    print(f"\n[series 候補] {len(series)} 件 (期待値=3)")

    video_end = samples[-1][0] + sample_interval

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", encoding="utf-8") as f:
        f.write("idx\tclip_start_sec\tclip_end_sec\tduration_sec\tcore_start_sec\tcore_end_sec\n")
        for i, b in enumerate(series):
            clip_start = max(0.0, b.start - MARGIN_SEC)
            clip_end = min(video_end, b.end + MARGIN_SEC)
            dur = clip_end - clip_start
            f.write(
                f"{i+1}\t{clip_start:.1f}\t{clip_end:.1f}\t{dur:.1f}\t"
                f"{b.start:.1f}\t{b.end:.1f}\n"
            )
            print(
                f"  series {i+1}: clip=[{clip_start:.1f}, {clip_end:.1f}] "
                f"dur={dur:.1f}s ({dur/60:.1f}min)  "
                f"core=[{b.start:.1f}, {b.end:.1f}]"
            )
    print(f"\n出力: {OUT_TSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
