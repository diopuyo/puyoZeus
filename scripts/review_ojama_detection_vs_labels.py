"""ラベル付き 144 セルに対し、OjamaWarningDetector の判定結果を並べて検証する。

各セルについて:
    - 元画像 (拡大表示)
    - ラベル (人間正解)
    - detector 判定 (現在のテンプレで)
    - 一致 ✓ / 不一致 ✗

セル単位で混同行列とサイド別精度を出力。

出力:
    data/verify/ojama_review_vs_labels_grid.png
    data/verify/ojama_review_vs_labels_summary.txt
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.ojama_warning import (
    BOARD_WIDTH,
    CELL_COUNT,
    CELL_WIDTH,
    P1_BOARD_X,
    P2_BOARD_X,
    WARNING_BOTTOM_Y,
    WARNING_HEIGHT,
    WARNING_TOP_Y,
    OjamaWarningDetector,
)

import argparse

# ユーザラベル名 → 実装側 ICON_* 名のマッピング
LABEL_TO_IMPL: dict[str, str] = {
    "small": "small",
    "large": "line",       # 大ぷよ (= 6 個 = 1 column)
    "rock": "rock",
    "star": "big_crown",   # 星ぷよ (= 180 個)
    "moon": "moon",
    "crown": "crown",
    "empty": "empty",
}
# 互換用 (CLI 引数で上書き可)
DEFAULT_LABELS_PATH = Path("data/verify/ojama_labels.tsv")
DEFAULT_INDEX_PATH = Path("data/verify/ojama_label_index.tsv")
DEFAULT_GRID_OUT = Path("data/verify/ojama_review_vs_labels_grid.png")
DEFAULT_SUMMARY_OUT = Path("data/verify/ojama_review_vs_labels_summary.txt")
VIDEO_DIR = Path("data/frames")
EXPECTED_FRAME_SHAPE: tuple[int, int] = (1080, 1920)
SCALE: int = 3
LABEL_BAR_H: int = 36
CELL_GAP: int = 4
FRAME_GAP: int = 12


def get_frame(video_path: Path, t_sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != EXPECTED_FRAME_SHAPE:
        frame = cv2.resize(
            frame,
            (EXPECTED_FRAME_SHAPE[1], EXPECTED_FRAME_SHAPE[0]),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def annotate_cell(
    cell: np.ndarray,
    truth: str,
    pred: str,
) -> np.ndarray:
    h, w = cell.shape[:2]
    big = cv2.resize(cell, (w * SCALE, h * SCALE),
                     interpolation=cv2.INTER_NEAREST)
    bar = np.zeros((LABEL_BAR_H, big.shape[1], 3), dtype=np.uint8)
    is_match = (truth == pred) or (truth in ("?",))
    color_t = (255, 255, 255)
    color_p = (0, 255, 0) if is_match else (0, 0, 255)
    cv2.putText(bar, f"T:{truth}", (3, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_t, 1)
    cv2.putText(bar, f"P:{pred}", (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_p, 1)
    return np.vstack([bar, big])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--grid-out", type=Path, default=DEFAULT_GRID_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    args = parser.parse_args()

    GRID_OUT = args.grid_out
    SUMMARY_OUT = args.summary_out
    GRID_OUT.parent.mkdir(parents=True, exist_ok=True)

    # インデックス読み込み
    idx: dict[tuple[int, str, int], tuple[float, str]] = {}
    with open(args.index, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            key = (int(r["frame_idx"]), r["side"], int(r["cell_idx"]))
            idx[key] = (float(r["t_sec"]), r["video"])

    # ラベル読み込み
    labels: dict[tuple[int, str, int], str] = {}
    with open(args.labels, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            key = (int(r["frame_idx"]), r["side"], int(r["cell_idx"]))
            labels[key] = r["label"]

    detector = OjamaWarningDetector()
    print(f"templates loaded: {list(detector._templates.keys())}")

    # フレームごとに detect() してセル予測を集める
    n_frames = max(k[0] for k in idx.keys()) + 1
    predictions: dict[tuple[int, str, int], str] = {}
    frame_cache: dict[tuple[str, float], np.ndarray | None] = {}

    for fi in range(n_frames):
        # 1P / 2P それぞれ
        for side in ("1P", "2P"):
            key0 = (fi, side, 0)
            if key0 not in idx:
                continue
            t_sec, vid = idx[key0]
            cache_key = (vid, t_sec)
            if cache_key not in frame_cache:
                video_path = VIDEO_DIR / f"{vid}.mp4"
                frame_cache[cache_key] = get_frame(video_path, t_sec)
            frame = frame_cache[cache_key]
            if frame is None:
                continue
            p1, p2 = detector.detect(frame)
            res = p1 if side == "1P" else p2
            # 6 セルそれぞれを再判定 (detect() は empty を返さない)
            # 内部の _classify_cell 相当を再現するため、生のセル切出 → classify
            base_x = P1_BOARD_X if side == "1P" else P2_BOARD_X
            for ci in range(CELL_COUNT):
                cell = detector._extract_cell(frame, base_x, ci)
                kind = detector._classify_cell(cell)
                predictions[(fi, side, ci)] = kind

    # 集計と grid 画像生成
    panels: list[np.ndarray] = []
    correct = 0
    total = 0
    confusion: dict[tuple[str, str], int] = Counter()

    for fi in range(n_frames):
        if (fi, "1P", 0) not in idx:
            continue
        t_sec, vid = idx[(fi, "1P", 0)]
        frame_cache_key = (vid, t_sec)
        frame = frame_cache.get(frame_cache_key)
        if frame is None:
            continue

        # 1P 6 セル + 2P 6 セルのパネル構築
        annotated_cells: list[np.ndarray] = []
        for side in ("1P", "2P"):
            base_x = P1_BOARD_X if side == "1P" else P2_BOARD_X
            for ci in range(CELL_COUNT):
                raw_truth = labels.get((fi, side, ci), "?")
                truth = LABEL_TO_IMPL.get(raw_truth, raw_truth)
                pred = predictions.get((fi, side, ci), "?")
                cell = frame[
                    WARNING_TOP_Y:WARNING_BOTTOM_Y,
                    base_x + ci * CELL_WIDTH:base_x + (ci + 1) * CELL_WIDTH,
                ]
                annotated_cells.append(annotate_cell(cell, truth, pred))
                # 集計
                if truth not in ("?",):
                    confusion[(truth, pred)] += 1
                    if truth == pred:
                        correct += 1
                    total += 1
            # サイド境界
            if side == "1P":
                # 1P が終わったら太い区切り
                pass

        # 1P 6 + 2P 6 を横に並べる、間に区切り線
        sep = np.full((annotated_cells[0].shape[0], CELL_GAP, 3),
                      60, dtype=np.uint8)
        big_sep = np.full((annotated_cells[0].shape[0], 12, 3),
                          120, dtype=np.uint8)
        parts: list[np.ndarray] = []
        for i, c in enumerate(annotated_cells):
            parts.append(c)
            if i == 5:
                parts.append(big_sep)
            else:
                parts.append(sep)
        body = np.hstack(parts[:-1])

        title_bar = np.zeros((24, body.shape[1], 3), dtype=np.uint8)
        title = f"F{fi}: {vid} t={t_sec:.1f}s   1P [S0..S5]   |   2P [S0..S5]"
        cv2.putText(title_bar, title, (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        panels.append(np.vstack([title_bar, body]))

    # 全パネル縦結合
    sep = np.full((FRAME_GAP, panels[0].shape[1], 3), 30, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for p in panels:
        parts.append(p)
        parts.append(sep)
    grid = np.vstack(parts[:-1])
    cv2.imwrite(str(GRID_OUT), grid)

    # サマリ出力
    accuracy = correct / total if total > 0 else 0.0
    lines: list[str] = []
    lines.append(f"Total cells labeled: {total}")
    lines.append(f"Correct: {correct} ({accuracy:.3f})")
    lines.append("")
    lines.append("=== Confusion Matrix (truth → pred: count) ===")
    truth_classes = sorted({k[0] for k in confusion})
    pred_classes = sorted({k[1] for k in confusion})
    header = "truth\\pred  " + "  ".join(f"{p:>7s}" for p in pred_classes)
    lines.append(header)
    for t in truth_classes:
        row = [f"{t:10s} "]
        for p in pred_classes:
            cnt = confusion.get((t, p), 0)
            row.append(f"{cnt:>7d}")
        lines.append("  ".join(row))

    lines.append("")
    lines.append("=== Per-class accuracy ===")
    truth_total: dict[str, int] = Counter()
    truth_correct: dict[str, int] = Counter()
    for (t, p), c in confusion.items():
        truth_total[t] += c
        if t == p:
            truth_correct[t] += c
    for t in sorted(truth_total):
        n = truth_total[t]
        c = truth_correct[t]
        lines.append(f"  {t:10s} {c}/{n} = {c/n:.3f}")

    summary = "\n".join(lines)
    SUMMARY_OUT.write_text(summary, encoding="utf-8")
    print()
    print(summary)
    print()
    print(f"出力 grid: {GRID_OUT}")
    print(f"出力 summary: {SUMMARY_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
