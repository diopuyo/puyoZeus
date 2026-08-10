"""動画特化デモ実験: video_olRyxDGacbg の試合開始直後フレームから
「空」seed をルールベースで採取する (2026-08-07)。

ルール根拠: ぷよぷよのルール上、試合開始直後の数秒間は両者の盤面が
全マス空であることが保証される (matches.tsv の start_sec を正解源とする)。
CNN/HSV 推論結果には一切依存しない (= 現行モデルが誤認識している
1P 暗赤系キャラ背景セルこそ今回の主目的、推論に頼ると採取できない)。

出力先: data/pseudo_labels_olRyxDGacbg_demo_2026-08-07/olRyxDGacbg_matchstart/cell.jsonl
   (本番 seed ディレクトリ data/pseudo_labels_hsv_seed_no_ojama/ には一切書き込まない)

本番モデル・本番設定には一切触れない使い捨てスクリプト。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, HIDDEN_ROWS
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.self_supervised.label_store import LabelStore
from src.self_supervised.pseudo_label import COMPONENT_CELL, PseudoLabelSample

VIDEO_PATH = Path("data/frames/video_olRyxDGacbg.mp4")
MATCHES_TSV = Path(
    "data/verify/match_boundaries_olRyxDGacbg/video_olRyxDGacbg/matches.tsv",
)
OUT_ROOT = Path("data/pseudo_labels_olRyxDGacbg_demo_2026-08-07")
OUT_VIDEO_ID = "olRyxDGacbg_matchstart"

# 試合開始からのオフセット (秒)。 演出フェード直後〜最初のツモが可視段に
# 降りてくる前の窓を狙う (ルール保証の「全マス空」区間)。
OFFSETS_SEC: tuple[float, ...] = (1.0, 2.0)

TARGET_SIZE = (1920, 1080)  # (width, height)


def _read_matches(path: Path) -> list[float]:
    starts: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            starts.append(float(row["start_sec"]))
    return starts


def _extract_patch(frame: np.ndarray, region, row: int, col: int) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region.cell_sample_rect(row, col)
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(y1 + 1, min(int(y2), h))
    patch = frame[y1:y2, x1:x2]
    return patch.copy() if patch.size > 0 else None


def _collect_frame_samples(
    frame: np.ndarray, region, side: str, match_idx: int, t_sec: float,
) -> list[PseudoLabelSample]:
    out: list[PseudoLabelSample] = []
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        for col in range(BOARD_COLS):
            patch = _extract_patch(frame, region, row, col)
            if patch is None:
                continue
            out.append(PseudoLabelSample(
                component=COMPONENT_CELL,
                timestamp=t_sec,
                input_data={"patch": patch},
                label=COLOR_EMPTY,
                confidence=1.0,
                metadata={
                    "video_id": OUT_VIDEO_ID,
                    "match_idx": match_idx,
                    "row": row, "col": col, "side": side,
                    "ground_truth_rule": "match_start_all_empty",
                },
            ))
    return out


def main() -> int:
    starts = _read_matches(MATCHES_TSV)
    print(f"[extract] {len(starts)} matches, offsets={OFFSETS_SEC}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {VIDEO_PATH}")

    store = LabelStore(video_id=OUT_VIDEO_ID, root=OUT_ROOT)
    total = 0
    for match_idx, start_sec in enumerate(starts, start=1):
        for offset in OFFSETS_SEC:
            t_sec = start_sec + offset
            cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[extract] match={match_idx} t={t_sec:.1f}s read FAILED, skip")
                continue
            if frame.shape[1::-1] != TARGET_SIZE:
                frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            samples = _collect_frame_samples(
                frame, DEFAULT_P1_REGION, "1P", match_idx, t_sec,
            )
            samples.extend(_collect_frame_samples(
                frame, DEFAULT_P2_REGION, "2P", match_idx, t_sec,
            ))
            store.append(samples)
            total += len(samples)
            print(f"[extract] match={match_idx} t={t_sec:.1f}s -> {len(samples)} cells "
                  f"(cum={total})")
    cap.release()
    print(f"[extract] DONE total={total} -> {store.video_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
