"""動画特化デモ実験 v2: video_olRyxDGacbg の試合開始直後フレームから
「空」seed をルールベースで採取する (2026-08-07、v1 の増量版)。

v1 (_finetune_olRyxDGacbg_2026-08-07_extract_seed.py) からの変更点:
    1. OFFSETS_SEC を (1.0, 2.0) → (1.0, 1.5, 2.0, 2.5, 3.0) の5点に増量
       (前回の2.5倍)。
    2. V_std フィルタ (HSV V channel の std ≤ V_STD_MAX) を追加。
       演出フェード残光・グロー境界にかかった patch を除外する
       (src/calibration.py の AUTO_DETECT_V_STD_MAX と同じ考え方)。
    3. 下段3行 (row10-12, 12段目/13段目寄りの可視最下段) を採取対象から
       除外。設置済み1手目がこの帯に混入するリスクを避ける。
       目的 (1P上部の暗赤系キャラ背景セル誤読) には下段は無関係なので
       除外による悪影響なし。

ルール根拠: 試合開始直後の数秒間は両者の盤面が全マス空であることが
保証される (matches.tsv の start_sec を正解源とする)。
CNN/HSV 推論結果には一切依存しない。

出力先: data/pseudo_labels_olRyxDGacbg_demo_2026-08-07/olRyxDGacbg_matchstart_v2/cell.jsonl
   (v1 の olRyxDGacbg_matchstart/ は上書きしない。既存5動画seedにも触れない)

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

from src.board import BOARD_COLS, COLOR_EMPTY, HIDDEN_ROWS
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.self_supervised.label_store import LabelStore
from src.self_supervised.pseudo_label import COMPONENT_CELL, PseudoLabelSample

VIDEO_PATH = Path("data/frames/video_olRyxDGacbg.mp4")
MATCHES_TSV = Path(
    "data/verify/match_boundaries_olRyxDGacbg/video_olRyxDGacbg/matches.tsv",
)
OUT_ROOT = Path("data/pseudo_labels_olRyxDGacbg_demo_2026-08-07")
OUT_VIDEO_ID = "olRyxDGacbg_matchstart_v2"

# 試合開始からのオフセット (秒)。v1 (1.0, 2.0) の2点から5点に増量。
OFFSETS_SEC: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)

# 採取対象の可視行範囲: HIDDEN_ROWS(=1) から ROW_MAX_EXCLUSIVE(=10) まで。
# 下段3行 (row10, row11, row12) は設置済み1手目混入対策として除外
# (目的=1P上部の背景誤読対策なので下段除外は無関係・悪影響なし)。
ROW_MAX_EXCLUSIVE: int = 10

# V_std フィルタ閾値。HSV V channel の std がこれを超える patch は
# 演出残光・グロー境界の疑いがあるため除外する (calibration.py の
# AUTO_DETECT_V_STD_MAX=25 と同系の考え方、今回は厳しめに 20.0)。
V_STD_MAX: float = 20.0

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


def _passes_v_std_filter(patch: np.ndarray) -> bool:
    """HSV V channel の std が V_STD_MAX 以下なら True (=採用)."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return bool(np.std(hsv[:, :, 2]) <= V_STD_MAX)


def _collect_frame_samples(
    frame: np.ndarray, region, side: str, match_idx: int, t_sec: float,
) -> tuple[list[PseudoLabelSample], int]:
    out: list[PseudoLabelSample] = []
    n_rejected_std = 0
    for row in range(HIDDEN_ROWS, ROW_MAX_EXCLUSIVE):
        for col in range(BOARD_COLS):
            patch = _extract_patch(frame, region, row, col)
            if patch is None:
                continue
            if not _passes_v_std_filter(patch):
                n_rejected_std += 1
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
                    "v_std_filter": "le_20.0",
                },
            ))
    return out, n_rejected_std


def main() -> int:
    starts = _read_matches(MATCHES_TSV)
    print(f"[extract_v2] {len(starts)} matches, offsets={OFFSETS_SEC}, "
          f"row_range=[{HIDDEN_ROWS},{ROW_MAX_EXCLUSIVE}), v_std_max={V_STD_MAX}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {VIDEO_PATH}")

    store = LabelStore(video_id=OUT_VIDEO_ID, root=OUT_ROOT)
    total = 0
    total_rejected = 0
    for match_idx, start_sec in enumerate(starts, start=1):
        for offset in OFFSETS_SEC:
            t_sec = start_sec + offset
            cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[extract_v2] match={match_idx} t={t_sec:.1f}s read FAILED, skip")
                continue
            if frame.shape[1::-1] != TARGET_SIZE:
                frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            samples_1p, rej_1p = _collect_frame_samples(
                frame, DEFAULT_P1_REGION, "1P", match_idx, t_sec,
            )
            samples_2p, rej_2p = _collect_frame_samples(
                frame, DEFAULT_P2_REGION, "2P", match_idx, t_sec,
            )
            samples = samples_1p + samples_2p
            store.append(samples)
            total += len(samples)
            total_rejected += rej_1p + rej_2p
            print(f"[extract_v2] match={match_idx} t={t_sec:.1f}s -> {len(samples)} cells "
                  f"(rejected_v_std={rej_1p + rej_2p}, cum={total})")
    cap.release()
    print(f"[extract_v2] DONE total={total} rejected_v_std={total_rejected} "
          f"-> {store.video_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
