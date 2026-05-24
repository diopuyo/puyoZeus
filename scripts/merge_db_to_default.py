"""全動画 DB を merge して未知動画用の default ranges を構築 (Phase I.c 段階 4)。

各動画の per_video_hsv_ranges を読み込み、 各色で H/S/V の min/max を OR で結合。
これで「全既知動画で観測された色範囲の union」 が得られ、 未知動画でも
起動時 inject すれば frame 0 から広範に色判定できる。

2026-05-11 サイクル65 バグ修正: RED は H wraparound (0-10 + 170-180) のため
naive min/max で 0-180 (= 全 hue) に潰れる問題があった。 RED のみは別動画間で
union せず、 各動画の RED 範囲をリストとして保持し、 別 range 集合として使う.

使い方:
    python scripts/merge_db_to_default.py \
        --db-root data/per_video_hsv_ranges \
        --out data/per_video_hsv_ranges/_merged_default.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# 2026-05-11: RED は wraparound のため union しない (色コード 1)
WRAPAROUND_COLORS: set[int] = {1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db-root", type=Path,
                    default=Path("data/per_video_hsv_ranges"))
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    merged: dict[int, list[int]] = {}
    sources: list[str] = []
    for json_path in sorted(args.db_root.glob("*.json")):
        if json_path.name.startswith("_"):
            continue
        with json_path.open() as f:
            d = json.load(f)
        ranges = d.get("per_video_ranges", {})
        if not ranges:
            continue
        sources.append(d.get("video_id", json_path.stem))
        for k, v in ranges.items():
            try:
                color = int(k)
            except (TypeError, ValueError):
                continue
            if len(v) != 6:
                continue
            h_min, h_max, s_min, s_max, v_min, v_max = (int(x) for x in v)
            # 2026-05-11: RED (= wraparound 色) は union 対象外、 DEFAULT_COLOR_RANGES
            # を使う想定で merged には入れない. (= DEFAULT が正しく 2 range 持つ)
            if color in WRAPAROUND_COLORS:
                continue
            if color not in merged:
                merged[color] = [h_min, h_max, s_min, s_max, v_min, v_max]
            else:
                cur = merged[color]
                merged[color] = [
                    min(cur[0], h_min), max(cur[1], h_max),
                    min(cur[2], s_min), max(cur[3], s_max),
                    min(cur[4], v_min), max(cur[5], v_max),
                ]
    payload = {
        "video_id": "_merged_default",
        "sources": sources,
        "n_sources": len(sources),
        "per_video_ranges": {str(k): v for k, v in merged.items()},
        "note": "RED (wraparound) は除外、 DEFAULT_COLOR_RANGES を使う",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[merge] merged from {sources} → {args.out}")
    for k, v in sorted(merged.items()):
        print(
            f"  color {k}: H={v[0]}-{v[1]} S={v[2]}-{v[3]} V={v[4]}-{v[5]}",
        )


if __name__ == "__main__":
    main()
