"""既存 cell.jsonl の label を維持しつつ patch を新 cell_sample_rect 領域で再抽出.

cycle 71i (2026-05-12): image_reader.py の cell_sample_rect が「全 row 中央 sample」
に変更されたため、 既存ラベルの input_data["patch"] が旧領域 (= 上部 row 下寄せ) と
混在している. 全 patch を新領域で再抽出して、 統一された学習データセットを作る.

label (= 色) と metadata (= video_id, frame_idx, row, col, side) は維持する.
input_data["patch"] のみ動画から新領域で再抽出.

使い方:
    PYTHONPATH=. python -m scripts.relabel_patches \\
        --video-id test_v50 \\
        --video-path data/test_unknown/v50_match1_75s_720p.mp4

    PYTHONPATH=. python -m scripts.relabel_patches \\
        --video-id v91_match1 \\
        --video-path data/test_unknown/v91_match1_75s_720p.mp4

注意:
    既存 cell.jsonl を上書きする (= バックアップは自動的に .bak で取られる).
    frame_state.jsonl は変更しない.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.self_supervised.pseudo_label import (
    COMPONENT_CELL, PseudoLabelSample,
)


def relabel_video(
    cell_jsonl: Path,
    video_path: Path,
    backup_suffix: str = ".bak",
) -> tuple[int, int]:
    """1 video 分の cell.jsonl を読み、 patch を新領域で再抽出して上書き.

    Returns:
        (relabeled_count, skipped_count)
    """
    if not cell_jsonl.exists():
        print(f"[skip] cell.jsonl 不在: {cell_jsonl}")
        return 0, 0
    if not video_path.exists():
        print(f"[error] 動画不在: {video_path}")
        return 0, 0

    # バックアップ作成
    backup_path = cell_jsonl.with_suffix(cell_jsonl.suffix + backup_suffix)
    if not backup_path.exists():
        shutil.copy2(cell_jsonl, backup_path)
        print(f"[backup] {backup_path}")

    # 既存ラベルを読み込み
    rows: list[dict[str, Any]] = []
    with cell_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    print(f"[load] {cell_jsonl}: {len(rows)} samples")

    # 動画を開く + frame_idx → frame をキャッシュ
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[error] cannot open: {video_path}")
        return 0, 0
    frame_cache: dict[int, np.ndarray] = {}

    def get_frame(fi: int) -> np.ndarray | None:
        if fi in frame_cache:
            return frame_cache[fi]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            return None
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080))
        frame_cache[fi] = frame
        return frame

    # 再抽出
    relabeled = 0
    skipped = 0
    out_lines: list[str] = []
    for row in rows:
        if row.get("component") != COMPONENT_CELL:
            # cell 以外はそのまま (= 念のため、 通常 cell.jsonl は cell 専用)
            out_lines.append(json.dumps(row, ensure_ascii=False))
            continue
        meta = row.get("metadata", {})
        fi = meta.get("frame_idx")
        side = meta.get("side")
        rrow = meta.get("row")
        rcol = meta.get("col")
        if fi is None or side not in ("1P", "2P") or rrow is None or rcol is None:
            skipped += 1
            out_lines.append(json.dumps(row, ensure_ascii=False))
            continue
        frame = get_frame(int(fi))
        if frame is None:
            skipped += 1
            out_lines.append(json.dumps(row, ensure_ascii=False))
            continue
        region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
        x1, y1, x2, y2 = region.cell_sample_rect(int(rrow), int(rcol))
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            skipped += 1
            out_lines.append(json.dumps(row, ensure_ascii=False))
            continue
        new_patch = frame[y1:y2, x1:x2].copy()
        # PseudoLabelSample の serialize 経由で input_data を更新
        new_sample = PseudoLabelSample(
            component=COMPONENT_CELL,
            timestamp=float(row.get("timestamp", 0.0)),
            input_data={"patch": new_patch},
            label=row.get("label", 0),
            confidence=float(row.get("confidence", 1.0)),
            metadata={**meta, "relabeled_cell_sample_rect": True},
        )
        out_lines.append(json.dumps(new_sample.to_jsonable(), ensure_ascii=False))
        relabeled += 1

    cap.release()

    # 上書き
    with cell_jsonl.open("w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")
    print(f"[done] {cell_jsonl}: relabeled={relabeled} skipped={skipped}")
    return relabeled, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", type=str, required=True)
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument(
        "--store-root", type=Path, default=Path("data/pseudo_labels"),
    )
    args = parser.parse_args()
    cell_jsonl = args.store_root / args.video_id / "cell.jsonl"
    rel, skip = relabel_video(cell_jsonl, args.video_path)
    print(f"\n[summary] video_id={args.video_id} relabeled={rel} skipped={skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
