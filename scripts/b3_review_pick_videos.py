"""B3 レビュー用の未見動画候補を選ぶ (DL しない。メタデータの突合のみ)。

除外: B2 の学習対象 60 本 (targets.json) / zenchi / 封印3本 / OUkub5_6Zts (評価セット参照)。
選別: ティア確認済み (マスター/チャレンジャー/S級/A級) かつ、既存の撃ち合いイベント試作
(EVENT_PILOT) で subset=reserve (旧 R の学習に使っていない) かつ M2 増分 manifest に無い動画。
"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import csv
import json
from pathlib import Path

TARGETS = Path("D:/puyo_analyzer/wt_b2/data/verify/recollect_2026-10-03/targets.json")
EXEV = Path(_public_dev_text('<repo>'))
PILOT = EXEV / "docs/manifests/EVENT_PILOT_2026-08-28.tsv"
M2_MANIFEST = EXEV / "docs/manifests/ADVANTAGE_M2_INCREMENT_S1_S2_2026-09-06.tsv"
TIER_INDEX = Path("data/video_tier_index_2026-08-07.tsv")
FIXED_EXCLUDE = frozenset({"c0BQoMJwwQU", "ApH_cBMWpTU", "Zm9ohkCl2vo", "Ztnh224PmlE", "OUkub5_6Zts"})
TOP_TIERS = ("マスター", "チャレンジャー", "S級", "A級")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def candidates() -> list[dict[str, str]]:
    """旧 R 非学習 (reserve) かつ B3/M2 に未使用の、ティア確認済み動画。"""
    excluded = {t["youtube_id"] for t in json.loads(TARGETS.read_text(encoding="utf-8"))} | FIXED_EXCLUDE
    m2_text = M2_MANIFEST.read_text(encoding="utf-8")
    tiers = {r["video_id"]: r for r in read_tsv(TIER_INDEX) if r["video_id"]}
    out = []
    for r in read_tsv(PILOT):
        vid = r["source_group_id"]
        if r["subset"] != "reserve" or vid in excluded or vid in m2_text:
            continue
        tier = tiers.get(vid, {}).get("tier")
        if tier in TOP_TIERS and r["title"]:
            out.append(dict(video_id=vid, target_id=r["target_id"], tier=tier, title=r["title"]))
    return out


if __name__ == "__main__":
    for c in candidates():
        print(c["target_id"], c["video_id"], c["tier"], c["title"][:70])
