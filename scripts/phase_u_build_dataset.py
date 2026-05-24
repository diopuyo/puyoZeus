"""バッチ 1, 2, 3 のラベル csv から手動ラベル付き訓練データセットを構築。

各 csv の (time, side, row, col, your_answer) を読み、対応する動画フレーム
からパッチを切り出して npz 保存。

your_answer 列が user の真値 (差分修正済み)。空欄なら recognized 列を採用。
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.console_init import init_console  # noqa: E402
init_console()

import cv2
import numpy as np

from src.image_reader import (
    DEFAULT_P1_REGION,
    DEFAULT_P2_REGION,
)
from src.board import HIDDEN_ROWS

# ASCII -> color code (csv の recognized 列の文字列形式)
LABEL_TO_CODE = {
    "EM": 0, "RED": 1, "BLUE": 2, "GRN": 3,
    "YEL": 4, "PUR": 5, "OJM": 9,
}

PATCH_OUT_SIZE = 16


def _extract_patch(
    frame: np.ndarray, side: str, vrow: int, vcol: int,
) -> np.ndarray | None:
    if frame is None:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(
            frame, (1920, 1080), interpolation=cv2.INTER_AREA,
        )
    region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    row = vrow + HIDDEN_ROWS
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region.cell_sample_rect(row, vcol)
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return cv2.resize(
        patch, (PATCH_OUT_SIZE, PATCH_OUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )


def build_dataset(
    base_dirs: list[str],
    video_path: str,
    out_path: str,
) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"video open failed: {video_path}")
        return 1

    patches: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0

    for base in base_dirs:
        base_path = Path(base)
        if not base_path.exists():
            continue
        for sheet_dir in sorted(base_path.iterdir()):
            csv_path = sheet_dir / "labels.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    truth_str = row.get("your_answer", "").strip()
                    if not truth_str:
                        truth_str = row["recognized"]
                    if truth_str not in LABEL_TO_CODE:
                        skipped += 1
                        continue
                    code = LABEL_TO_CODE[truth_str]
                    t = float(row["time"])
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                    ok, fr = cap.read()
                    if not ok or fr is None:
                        skipped += 1
                        continue
                    patch = _extract_patch(
                        fr, row["side"],
                        int(row["row"]), int(row["col"]),
                    )
                    if patch is None:
                        skipped += 1
                        continue
                    patches.append(patch)
                    labels.append(code)

    cap.release()
    if not patches:
        print("no samples")
        return 1

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_p,
        patches=np.array(patches, dtype=np.uint8),
        labels=np.array(labels, dtype=np.int32),
    )
    print(f"saved: {out_p} ({len(patches)} samples, {skipped} skipped)")
    unique, counts = np.unique(labels, return_counts=True)
    for c, n in zip(unique, counts):
        print(f"  code={c}: {n}")
    return 0


def add_empty_samples(
    base_dirs: list[str], video_path: str, out_path: str,
    max_per_sheet: int = 80,
) -> None:
    """各シートで認識結果 EMPTY のセルを EMPTY ラベルとして追加。"""
    from src.image_reader import (
        BOARD_COLS,
        VISIBLE_ROWS,
        ColorClassifier,
        ImageReader,
    )
    from src.board import COLOR_EMPTY
    cap = cv2.VideoCapture(video_path)
    reader = ImageReader(use_ui_mask=False)
    classifier = reader._classifier  # type: ignore[attr-defined]

    # 既存 npz をロード
    data = np.load(out_path)
    patches = list(data["patches"])
    labels = list(data["labels"])
    n_added = 0

    seen_times: set[float] = set()
    for base in base_dirs:
        base_path = Path(base)
        if not base_path.exists():
            continue
        for sheet_dir in sorted(base_path.iterdir()):
            csv_path = sheet_dir / "labels.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, encoding="utf-8") as f:
                reader_csv = csv.DictReader(f)
                times = sorted({float(r["time"]) for r in reader_csv})
            # 各時刻で全 144 セル認識
            empty_cells_added_for_sheet = 0
            for t in times:
                if t in seen_times:
                    continue
                seen_times.add(t)
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, fr = cap.read()
                if not ok or fr is None:
                    continue
                if fr.shape[:2] != (1080, 1920):
                    fr = cv2.resize(
                        fr, (1920, 1080), interpolation=cv2.INTER_AREA,
                    )
                board1, board2 = reader.read_both_boards(fr)
                for side, board in (("1P", board1), ("2P", board2)):
                    for vrow in range(VISIBLE_ROWS):
                        for vcol in range(BOARD_COLS):
                            row = vrow + HIDDEN_ROWS
                            color = int(board.get(row, vcol))
                            if color != COLOR_EMPTY:
                                continue
                            patch = _extract_patch(fr, side, vrow, vcol)
                            if patch is None:
                                continue
                            patches.append(patch)
                            labels.append(0)
                            n_added += 1
                            empty_cells_added_for_sheet += 1
                            if empty_cells_added_for_sheet >= max_per_sheet:
                                break
                        if empty_cells_added_for_sheet >= max_per_sheet:
                            break
                    if empty_cells_added_for_sheet >= max_per_sheet:
                        break
                if empty_cells_added_for_sheet >= max_per_sheet:
                    break
    cap.release()

    np.savez(
        out_path,
        patches=np.array(patches, dtype=np.uint8),
        labels=np.array(labels, dtype=np.int32),
    )
    print(f"added {n_added} EMPTY samples -> total {len(patches)}")


def main() -> int:
    base_dirs = [
        "data/verify/phase_u_batch1",
        "data/verify/phase_u_batch2",
        "data/verify/phase_u_batch3",
        "data/verify/phase_u_batch4",
        "data/verify/phase_u_uncertain_v01",
        "data/verify/phase_u_uncertain_v01_bg",
    ]
    out_path = "data/training_phase_u/manual_labels.npz"
    rc = build_dataset(
        base_dirs=base_dirs,
        video_path="data/frames/video_01.mp4",
        out_path=out_path,
    )
    if rc != 0:
        return rc
    add_empty_samples(base_dirs, "data/frames/video_01.mp4", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
