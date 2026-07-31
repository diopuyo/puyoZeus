"""#43 c系: 品質ゲート済み manifest からティア確認済み・正常な動画を N 本選定する。

## ティア除外の根拠 (2026-07-26 手動確認、推測禁止・要 Read 確認済み)
- video_c1-c3: 個別テストDL (_dl_tier_extra.sh/_dl_tier_avc1.sh) 由来で
  タイトル未確認 (video_idx管理外)。CLAUDE.md の「未確認動画は学習から除外」
  方針に倣い、確定するまで除外 (要確認バケット)。
- video_c4-c33: data/pl_new_missing.tsv 由来、全て「チャレンジャー」tier
  (新おいうリーグ・1st edition) -> 採用可。
- video_c34-c81: data/_dl_expand.tsv 由来、全て「マスター」tier
  (第2回新おいうリーグ) -> 採用可。
- video_c82-c84: 同上、「S級決定戦」-> 採用可。
- video_c85-c94: 同上、A/B1/B2/C1/C2/D 級の複合トーナメント (混成、下位級含む)
  -> CLAUDE.md 対象外ティア混入のため除外。
- video_c95: 「マスター進出決定トーナメント」= マスター昇格前の予選、
  マスター確定でないため除外 (要確認)。

上記境界は _ALLOWED_C_INDEX_RANGES として明示定数化 (マジックナンバー禁止対応)。

## 使い方
    python -m scripts.select_labeled_win_videos \\
        --manifest data/verify/labeled_win_c20_2026-07-26/quality_gate_manifest.tsv \\
        --n 20 \\
        --out data/verify/labeled_win_c20_2026-07-26/selected_videos.txt
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# 採用可能な video_cN の連番範囲 (両端含む)。tier確認済み分のみ (上記docstring参照)。
_ALLOWED_C_INDEX_RANGES: tuple[tuple[int, int], ...] = (
    (4, 33),   # チャレンジャー (新おいうリーグ)
    (34, 84),  # マスター + S級決定戦 (第2回新おいうリーグ)
)

_VIDEO_ID_RE = re.compile(r"^video_c(\d+)$")

DEFAULT_MANIFEST_PATH: str = "data/verify/labeled_win_c20_2026-07-26/quality_gate_manifest.tsv"
DEFAULT_OUT_PATH: str = "data/verify/labeled_win_c20_2026-07-26/selected_videos.txt"
DEFAULT_N: int = 20


def _video_index(video_id: str) -> int | None:
    """video_cNN -> NN (int)。マッチしなければ None。"""
    m = _VIDEO_ID_RE.match(video_id)
    return int(m.group(1)) if m else None


def is_tier_allowed(video_id: str) -> bool:
    """ティア確認済み範囲内かどうか。"""
    idx = _video_index(video_id)
    if idx is None:
        return False
    return any(lo <= idx <= hi for lo, hi in _ALLOWED_C_INDEX_RANGES)


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """quality_gate_manifest.tsv を読み込む。"""
    with manifest_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        return list(reader)


def select_videos(rows: list[dict[str, str]], n: int) -> tuple[list[str], list[str]]:
    """
    採用条件 (status=="ok" かつティア確認済み) を満たす video_id を
    連番昇順で先頭 n 本選び、(選定リスト, 除外理由付きリスト) を返す。
    """
    selected: list[str] = []
    excluded_log: list[str] = []
    candidates = sorted(rows, key=lambda r: _video_index(r["video_id"]) or 0)
    for row in candidates:
        vid = row["video_id"]
        if row["status"] != "ok":
            excluded_log.append(f"{vid}: status={row['status']}")
            continue
        if not is_tier_allowed(vid):
            excluded_log.append(f"{vid}: tier未確認/対象外のため除外")
            continue
        selected.append(vid)
        if len(selected) >= n:
            break
    return selected, excluded_log


def main() -> int:
    parser = argparse.ArgumentParser(description="labeled_win 用ティア確認済み動画の選定 (#43 c系)")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_path = Path(args.out)

    rows = load_manifest(manifest_path)
    selected, excluded_log = select_videos(rows, args.n)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")

    print(f"[select] manifest行数: {len(rows)}")
    print(f"[select] 選定: {len(selected)}/{args.n}  -> {out_path}")
    for vid in selected:
        print(f"  + {vid}")
    print(f"[select] 除外: {len(excluded_log)} 件")
    for line in excluded_log:
        print(f"  - {line}")
    if len(selected) < args.n:
        print(f"[select][WARN] 目標本数未達 ({len(selected)}/{args.n})。"
              f"追加完了を待って再実行が必要。")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
