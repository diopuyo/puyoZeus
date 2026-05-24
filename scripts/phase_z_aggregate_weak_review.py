"""weak_video_extra/*/violations_review/violations.csv の your_answer から訓練 npz を生成。

phase_z_apply_weak_review.py で書き戻された your_answer (E/R/B/G/Y/P/O) を集計し、
violations.csv の time / side / row / col から patch を取り出し、(8x8 BGR, color_idx)
の (X, y) ペアを生成。EX (除外) と ?? (不明) と空白行はスキップ。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_aggregate_weak_review

出力:
    data/training_phase_u/phase_z_gt_weak.npz (X, y)
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.board import (  # noqa: E402
    COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN, COLOR_OJAMA, COLOR_PURPLE,
    COLOR_RED, COLOR_YELLOW, HIDDEN_ROWS,
)
from src.image_reader import (  # noqa: E402
    DEFAULT_P1_REGION, DEFAULT_P2_REGION,
)
from src.patch_classifier import COLOR_TO_CLASS_INDEX  # noqa: E402

LABEL_TO_COLOR = {
    "EM": COLOR_EMPTY, "RED": COLOR_RED, "BLUE": COLOR_BLUE,
    "GRN": COLOR_GREEN, "YEL": COLOR_YELLOW, "PUR": COLOR_PURPLE,
    "OJM": COLOR_OJAMA,
}
COLOR_NAME = {
    COLOR_EMPTY: "EM", COLOR_RED: "RED", COLOR_BLUE: "BLUE",
    COLOR_GREEN: "GRN", COLOR_YELLOW: "YEL", COLOR_PURPLE: "PUR",
    COLOR_OJAMA: "OJM",
}


def extract_patch(
    frame: np.ndarray, side: str, vrow: int, col: int,
) -> np.ndarray:
    region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    h, w = frame.shape[:2]
    row = vrow + HIDDEN_ROWS
    x1, y1, x2, y2 = region.cell_sample_rect(row, col)
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    return frame[y1:y2, x1:x2].copy()


def video_path_for(seg_dir: Path) -> Path | None:
    """seg_dir 名 'v04_m03_1137_1167' から動画 ID を抽出して mp4 path を返す。"""
    name = seg_dir.name
    if not name.startswith("v"):
        return None
    try:
        vid = int(name[1:3])
    except ValueError:
        return None
    p = _ROOT / f"data/frames/video_{vid:02d}.mp4"
    return p if p.exists() else None


def load_violations(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ans = (r.get("your_answer") or "").strip().upper()
            if ans in ("", "EX", "??"):
                continue
            color = LABEL_TO_COLOR.get(ans)
            if color is None:
                continue
            rows.append({
                "time": float(r["time"]),
                "side": r["side"],
                "row": int(r["row"]),
                "col": int(r["col"]),
                "recognized": r["recognized"],
                "answer": color,
            })
    return rows


def collect_patches_from_segment(
    seg_dir: Path,
) -> list[tuple[np.ndarray, int, str]]:
    """1 segment 分の (patch, color, recognized) を取得。"""
    csv_path = seg_dir / "violations_review" / "violations.csv"
    if not csv_path.exists():
        return []
    video_path = video_path_for(seg_dir)
    if video_path is None:
        print(f"[skip] {seg_dir.name}: 動画ファイル未発見")
        return []
    rows = load_violations(csv_path)
    if not rows:
        return []

    by_time: dict[float, list[dict]] = {}
    for r in rows:
        by_time.setdefault(r["time"], []).append(r)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[skip] {seg_dir.name}: video open 失敗")
        return []
    out: list[tuple[np.ndarray, int, str]] = []
    for t, group in sorted(by_time.items()):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        for r in group:
            patch = extract_patch(frame, r["side"], r["row"], r["col"])
            if patch.size == 0:
                continue
            out.append((patch, r["answer"], r["recognized"]))
    cap.release()
    return out


def main() -> int:
    weak_root = _ROOT / "data/verify/phase_z_review/weak_video_extra"
    if not weak_root.exists():
        print(f"ERROR: {weak_root} 不在")
        return 1
    seg_dirs = sorted(
        d for d in weak_root.iterdir()
        if d.is_dir() and d.name.startswith("v")
    )
    if not seg_dirs:
        print("[error] segment 不在")
        return 1
    print(f"対象 segment: {len(seg_dirs)}")

    all_samples: list[tuple[np.ndarray, int]] = []
    per_seg_count: dict[str, int] = {}
    mismatch_count = 0
    match_count = 0
    for seg in seg_dirs:
        items = collect_patches_from_segment(seg)
        per_seg_count[seg.name] = len(items)
        for patch, color, rec in items:
            all_samples.append((patch, color))
            rec_color = LABEL_TO_COLOR.get(rec)
            if rec_color is not None and rec_color == color:
                match_count += 1
            else:
                mismatch_count += 1

    print()
    print(f"取得 cell 数: {len(all_samples)}")
    print(f"  CNN 一致 (false positive): {match_count}")
    print(f"  真の誤り (CNN 修正対象):   {mismatch_count}")
    print()
    print("segment 別取得数:")
    for name, n in per_seg_count.items():
        print(f"  {name}: {n}")
    if not all_samples:
        print("[error] レビュー入力済 cell が無い (your_answer 空)")
        return 1

    counts = Counter(c for _, c in all_samples)
    print()
    print("色別 cell 数:")
    for c, n in sorted(counts.items()):
        print(f"  {COLOR_NAME.get(c, c)}: {n}")

    X = np.array([
        cv2.resize(p, (8, 8), interpolation=cv2.INTER_AREA)
        for p, _ in all_samples
    ], dtype=np.uint8)
    y = np.array(
        [COLOR_TO_CLASS_INDEX[c] for _, c in all_samples],
        dtype=np.int64,
    )
    out_path = _ROOT / "data/training_phase_u/phase_z_gt_weak.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y)
    print(f"\nsaved: {to_windows_path(out_path)} {X.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
