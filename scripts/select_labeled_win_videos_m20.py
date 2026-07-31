"""#43 段階2: マスター級のみ (ティア変数固定) の labeled_win 用動画を N 本選定する。

## 背景・ティア範囲の根拠 (2026-07-28、前任確定の対応表を踏襲)
- video_c34-c81: data/_dl_expand.tsv 由来、全て「マスター」tier (第2回新おいうリーグ)。
- video_c82-c84: 同上、「S級決定戦」tier -> 本選定では **除外**。
  (ティア変数を固定してc20=チャレンジャー中心との比較を可能にする狙いのため、
  マスターと異なる S 級を混入させると比較の意味が壊れる。CLAUDE.md 上は
  S級も学習データとして許容ティアだが、本選定の目的= "ティア固定比較" には
  不適合なので range から外す。判断根拠は selected_videos_m20.txt 生成時の
  標準出力ログに明記する)
- video_c38, video_c39: data/frames/ に mp4 が存在しない (処理後削除済み、
  CLAUDE.md「動画ファイルは処理後に削除する」運用による欠落)。ソース動画が
  無いため収集不能 -> 選定候補から除外 (要 mp4 存在確認)。

## 使い方
    python -m scripts.select_labeled_win_videos_m20 \\
        --manifest data/verify/labeled_win_c20_2026-07-26/quality_gate_manifest.tsv \\
        --n 20 \\
        --out data/verify/labeled_win_m20_2026-07-28/selected_videos_m20.txt
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# マスター級のみの連番範囲 (両端含む)。c82-84 (S級) は意図的に除外 (docstring参照)。
_ALLOWED_C_INDEX_RANGES: tuple[tuple[int, int], ...] = (
    (34, 81),  # マスター (第2回新おいうリーグ)
)

_VIDEO_ID_RE = re.compile(r"^video_c(\d+)$")

DEFAULT_MANIFEST_PATH: str = "data/verify/labeled_win_c20_2026-07-26/quality_gate_manifest.tsv"
DEFAULT_FRAMES_DIR: str = "data/frames"
DEFAULT_OUT_PATH: str = "data/verify/labeled_win_m20_2026-07-28/selected_videos_m20.txt"
DEFAULT_N: int = 20


def _video_index(video_id: str) -> int | None:
    """video_cNN -> NN (int)。マッチしなければ None。"""
    m = _VIDEO_ID_RE.match(video_id)
    return int(m.group(1)) if m else None


def is_tier_allowed(video_id: str) -> bool:
    """マスター級確認済み範囲内かどうか。"""
    idx = _video_index(video_id)
    if idx is None:
        return False
    return any(lo <= idx <= hi for lo, hi in _ALLOWED_C_INDEX_RANGES)


def has_source_video(video_id: str, frames_dir: Path) -> bool:
    """ソース mp4 が data/frames/ に存在するか (処理後削除済み動画を除外するため)。"""
    idx = _video_index(video_id)
    if idx is None:
        return False
    return (frames_dir / f"video_c{idx}.mp4").exists()


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """quality_gate_manifest.tsv を読み込む。"""
    with manifest_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        return list(reader)


def select_videos(
    rows: list[dict[str, str]], n: int, frames_dir: Path
) -> tuple[list[str], list[str]]:
    """
    採用条件 (status=="ok" かつマスター級確認済み範囲内かつソース mp4 存在) を
    満たす video_id を連番昇順で先頭 n 本選び、(選定リスト, 除外理由付きリスト) を返す。
    """
    selected: list[str] = []
    excluded_log: list[str] = []
    candidates = sorted(rows, key=lambda r: _video_index(r["video_id"]) or 0)
    for row in candidates:
        vid = row["video_id"]
        idx = _video_index(vid)
        if idx is None:
            continue
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
    parser = argparse.ArgumentParser(description="labeled_win マスター級20本選定 (#43 段階2)")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--frames-dir", default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    frames_dir = Path(args.frames_dir)
    out_path = Path(args.out)

    rows = load_manifest(manifest_path)
    selected, excluded_log = select_videos(rows, args.n, frames_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")

    print(f"[select_m20] manifest行数: {len(rows)}")
    print(f"[select_m20] 選定: {len(selected)}/{args.n}  -> {out_path}")
    for vid in selected:
        print(f"  + {vid}")
    print(f"[select_m20] 除外: {len(excluded_log)} 件")
    for line in excluded_log:
        print(f"  - {line}")
    if len(selected) < args.n:
        print(f"[select_m20][WARN] 目標本数未達 ({len(selected)}/{args.n})。")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
