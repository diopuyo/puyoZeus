"""ティア統合台帳の video_c96 行を切り出し 3 本 (c96s1/s2/s3) に分割する.

video_c96 は 3 カード連結の 5.5 時間動画で、 scripts/_cut_c96_series_2026-08-08.sh
により S 級リーグ 3 シリーズへ切り出し済み。 台帳側も 1 行 -> 3 行へ置換して
Phase L の学習構成 (146 -> 148 本) と整合させる。 冪等 (再実行しても増えない)。
"""
from __future__ import annotations

import csv
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
LEDGER: Path = _ROOT / "data" / "video_tier_index_2026-08-07.tsv"
SEGMENTS: Path = (
    _ROOT / "data" / "verify" / "c96_split_2026-08-08" / "series_segments.tsv"
)
# 元タイトルに含まれる 3 カード (切り出し順 = 動画内の登場順で検収済み)。
CARD_TITLES: tuple[str, ...] = (
    "delta vs のらすけ",
    "live vs ちゃるめらー",
    "ともくん vs ゆうき",
)


def _load_durations() -> dict[str, int]:
    """series_segments.tsv から idx -> クリップ長 (秒) を読む."""
    out: dict[str, int] = {}
    with SEGMENTS.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[row["idx"]] = int(float(row["duration_sec"]))
    return out


def main() -> int:
    durations = _load_durations()
    lines = LEDGER.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    replaced = False
    for line in lines:
        cols = line.split("\t")
        if cols[0] != "video_c96":
            # 既に分割済みなら旧 s 行は捨てて作り直す (冪等性確保)
            if cols[0].startswith("video_c96s"):
                continue
            out_lines.append(line)
            continue
        replaced = True
        for i, card in enumerate(CARD_TITLES, start=1):
            new = list(cols)
            new[0] = f"video_c96s{i}"
            new[2] = str(durations.get(str(i), ""))
            new[3] = f"【第2回新おいうリーグS級リーグ最終戦】「{card}」30先 (c96 切り出し{i}/3)"
            out_lines.append("\t".join(new))
    if not replaced:
        print("[split] video_c96 行が見つからない (分割済み?) — 変更なし")
        return 1
    LEDGER.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[split] video_c96 -> c96s1/s2/s3 に分割 (総行数 {len(out_lines)})")
    for line in out_lines:
        if line.startswith("video_c96s"):
            print("   ", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
