"""148 動画 学習データ再生成の対象マニフェストを生成する (2026-08-11)。

user承認済み (2026-08-11): data/indicators_v2/boards_lean_phase_l_2026-08-07/
の 148 npz を現行認識構成 (production_config + 幻盤面ガード) で再生成する。

本スクリプトは実行 (DL/収集) を一切行わない。対象 148 本それぞれについて
「動画ファイルをどう用意するか (既存 / c96 派生 / 新規DL)」を解決し、
data/verify/regen_2026-08-11_manifest.tsv に書き出す。

名前解決ルール:
- c96s1 / c96s2 / c96s3: video_c96 (5.5時間・3シリーズ連結) からの派生クリップ。
  data/verify/c96_split_2026-08-08/series_segments.tsv の区間で切り出す
  (scripts/_cut_c96_series_2026-08-08.sh と同じロジック)。
- それ以外で data/frames/video_<target>.mp4 が既に存在: 既存動画を再利用
  (削除しない、47本の一部)。
- 上記以外: data/video_tier_index_2026-08-07.tsv (無ければ
  data/_dl_expand2_2026-08-07.tsv) から video_id を引いて新規DL対象とする。
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OLD_NPZ_DIR = PROJECT_ROOT / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-07"
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
TIER_TSV = PROJECT_ROOT / "data" / "video_tier_index_2026-08-07.tsv"
DL_EXPAND2_TSV = PROJECT_ROOT / "data" / "_dl_expand2_2026-08-07.tsv"
OUT_TSV = PROJECT_ROOT / "data" / "verify" / "regen_2026-08-11_manifest.tsv"

# video_c96 (5.5時間・3シリーズ連結) からの派生クリップ名。
C96_SPLIT_TARGETS: tuple[str, ...] = ("c96s1", "c96s2", "c96s3")
C96_SOURCE_TARGET = "c96"


@dataclass
class ManifestRow:
    """1 対象分の解決結果。"""

    target_id: str            # npz stem (例: "c1", "29", "c96s1")
    video_filename: str       # data/frames/ 配下のファイル名
    video_id: str             # yt-dlp 用 video_id ("" = DL不要)
    tier: str                 # ティア (参考情報)
    origin: str               # "preexisting" / "derived_c96" / "download"


def _load_tier_map() -> dict[str, dict[str, str]]:
    """video_tier_index_2026-08-07.tsv を読み込む。"""
    m: dict[str, dict[str, str]] = {}
    if not TIER_TSV.exists():
        return m
    with TIER_TSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            m[row["video_name"]] = row
    return m


def _load_dl_expand2_map() -> dict[str, tuple[str, str]]:
    """_dl_expand2_2026-08-07.tsv (name, id, title) を読み込む (video_c96 用フォールバック)。"""
    m: dict[str, tuple[str, str]] = {}
    if not DL_EXPAND2_TSV.exists():
        return m
    with DL_EXPAND2_TSV.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name, vid, title = parts[0], parts[1], parts[2]
            m[name] = (vid, title)
    return m


def resolve_target(
    target_id: str,
    tier_map: dict[str, dict[str, str]],
    dl_expand2_map: dict[str, tuple[str, str]],
) -> ManifestRow:
    """1 対象の動画所在を解決する。"""
    if target_id in C96_SPLIT_TARGETS:
        return ManifestRow(
            target_id=target_id,
            video_filename=f"video_{target_id}.mp4",
            video_id="",
            tier="S級",  # video_c96 由来 (data/_dl_expand2_2026-08-07.tsv より確認済)
            origin="derived_c96",
        )
    video_filename = f"video_{target_id}.mp4"
    if (FRAMES_DIR / video_filename).exists():
        return ManifestRow(
            target_id=target_id, video_filename=video_filename,
            video_id="", tier="", origin="preexisting",
        )
    vname = f"video_{target_id}"
    row = tier_map.get(vname)
    if row is not None and row.get("video_id"):
        return ManifestRow(
            target_id=target_id, video_filename=video_filename,
            video_id=row["video_id"], tier=row.get("tier", ""),
            origin="download",
        )
    # フォールバック: _dl_expand2 (video_tier_index に未反映のケース)
    fallback = dl_expand2_map.get(vname)
    if fallback is not None:
        vid, _title = fallback
        return ManifestRow(
            target_id=target_id, video_filename=video_filename,
            video_id=vid, tier="", origin="download",
        )
    raise ValueError(f"video_id 解決不能: {target_id} (tier index にも dl_expand2 にも無し)")


def main() -> int:
    targets = sorted(p.stem for p in OLD_NPZ_DIR.glob("*.npz"))
    if not targets:
        print(f"[error] 旧 npz が見つからない: {OLD_NPZ_DIR}", file=sys.stderr)
        return 1

    tier_map = _load_tier_map()
    dl_expand2_map = _load_dl_expand2_map()

    rows: list[ManifestRow] = []
    errors: list[str] = []
    for t in targets:
        try:
            rows.append(resolve_target(t, tier_map, dl_expand2_map))
        except ValueError as e:
            errors.append(str(e))

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", encoding="utf-8", newline="\n") as f:
        f.write("target_id\tvideo_filename\tvideo_id\ttier\torigin\n")
        for r in rows:
            f.write(f"{r.target_id}\t{r.video_filename}\t{r.video_id}\t{r.tier}\t{r.origin}\n")

    n_preexisting = sum(1 for r in rows if r.origin == "preexisting")
    n_derived = sum(1 for r in rows if r.origin == "derived_c96")
    n_download = sum(1 for r in rows if r.origin == "download")
    print(f"[manifest] 総数={len(targets)} 解決={len(rows)} エラー={len(errors)}")
    print(f"[manifest] preexisting={n_preexisting} derived_c96={n_derived} download={n_download}")
    for e in errors:
        print(f"[manifest][ERROR] {e}", file=sys.stderr)
    print(f"[manifest] -> {OUT_TSV}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
