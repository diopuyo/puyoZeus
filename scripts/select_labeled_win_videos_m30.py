"""#43 段階3: マスター級の残り (m20 未使用分) labeled_win 用動画を選定する。

## 背景
- ティア確定: video_c34-c81 = マスター (第2回新おいうリーグ)、
  video_c82-c84 = S級 (本選定では除外、m20 と同じ理由でティア固定比較を守る)。
- m20 (scripts/select_labeled_win_videos_m20.py) で video_c34-c37, c40-c55 の
  20 本を既に収集済み (c38, c39 はソース mp4 未存在のため元々候補外)。
- 本スクリプトは m20 選定ロジック (is_tier_allowed / has_source_video) を
  そのまま再利用し、「マスター級範囲内 かつ status==ok かつ ソース mp4 存在
  かつ m20 で未使用」の動画を連番昇順で選ぶ。m20 が枠 20 本で打ち切った
  video_c56-c81 が主対象になる想定 (実際に c56-c81 は全て mp4 存在・status=ok
  を 2026-07-28 に確認済み)。

## 使い方
    python -m scripts.select_labeled_win_videos_m30 \\
        --manifest data/verify/labeled_win_c20_2026-07-26/quality_gate_manifest.tsv \\
        --used-videos data/verify/labeled_win_m20_2026-07-28/selected_videos_m20.txt \\
        --n 26 \\
        --out data/verify/labeled_win_m30_2026-07-28/selected_videos_m30.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

# m20 選定ロジックをそのまま再利用 (DRY、ロジック分岐の重複を避ける)
from scripts.select_labeled_win_videos_m20 import (
    _video_index,
    has_source_video,
    is_tier_allowed,
    load_manifest,
)

DEFAULT_MANIFEST_PATH: str = "data/verify/labeled_win_c20_2026-07-26/quality_gate_manifest.tsv"
DEFAULT_FRAMES_DIR: str = "data/frames"
DEFAULT_USED_VIDEOS_PATH: str = "data/verify/labeled_win_m20_2026-07-28/selected_videos_m20.txt"
DEFAULT_OUT_PATH: str = "data/verify/labeled_win_m30_2026-07-28/selected_videos_m30.txt"
DEFAULT_N: int = 26


def load_used_videos(used_videos_path: Path) -> frozenset[str]:
    """m20 で既に使用済みの video_id 集合を読み込む (1行1 video_id)。"""
    if not used_videos_path.exists():
        return frozenset()
    lines = used_videos_path.read_text(encoding="utf-8").splitlines()
    return frozenset(line.strip() for line in lines if line.strip())


def select_videos(
    rows: list[dict[str, str]],
    n: int,
    frames_dir: Path,
    used_videos: frozenset[str],
) -> tuple[list[str], list[str]]:
    """
    採用条件 (status=="ok" かつマスター級確認済み範囲内かつソース mp4 存在
    かつ m20 未使用) を満たす video_id を連番昇順で先頭 n 本選び、
    (選定リスト, 除外理由付きリスト) を返す。m20 選定ロジックとの差分は
    「m20 で使用済みならスキップ」の1条件のみ。
    """
    selected: list[str] = []
    excluded_log: list[str] = []
    candidates = sorted(rows, key=lambda r: _video_index(r["video_id"]) or 0)
    for row in candidates:
        vid = row["video_id"]
        idx = _video_index(vid)
        if idx is None:
            continue
        if vid in used_videos:
            continue  # m20 既使用分は選定/除外ログどちらにも出さない (正常スキップ)
        if row["status"] != "ok":
            excluded_log.append(f"{vid}: status={row['status']}")
            continue
        if not is_tier_allowed(vid):
            excluded_log.append(f"{vid}: マスター級範囲外(c34-81でない、S級/対象外)のため除外")
            continue
        if not has_source_video(vid, frames_dir):
            excluded_log.append(f"{vid}: ソースmp4未存在(処理後削除済み)のため除外")
            continue
        selected.append(vid)
        if len(selected) >= n:
            break
    return selected, excluded_log


def main() -> int:
    parser = argparse.ArgumentParser(description="labeled_win マスター級残り本数選定 (#43 段階3)")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--frames-dir", default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--used-videos", default=DEFAULT_USED_VIDEOS_PATH)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    frames_dir = Path(args.frames_dir)
    used_videos_path = Path(args.used_videos)
    out_path = Path(args.out)

    rows = load_manifest(manifest_path)
    used_videos = load_used_videos(used_videos_path)
    selected, excluded_log = select_videos(rows, args.n, frames_dir, used_videos)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")

    print(f"[select_m30] manifest行数: {len(rows)}  m20使用済み: {len(used_videos)}")
    print(f"[select_m30] 選定: {len(selected)}/{args.n}  -> {out_path}")
    for vid in selected:
        print(f"  + {vid}")
    print(f"[select_m30] 除外: {len(excluded_log)} 件")
    for line in excluded_log:
        print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
