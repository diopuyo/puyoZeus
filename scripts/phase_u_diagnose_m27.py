"""m27 (1636-1713s) を 1 秒間隔で診断し、3 つの問題発生時刻を特定する。

1. 右下ぷよ未認識 → 盤面 (visible_row=11, col=5) の認識色を時系列で記録
2. 全認識停止 → match_state.bg_value と認識セル数 (非 EMPTY 数) を記録
3. 相殺エフェクト誤認 → フレーム間差分 / V 平均ジャンプを記録

出力:
    data/verify/m27_diagnose/
      timeline.csv  - 時刻別メトリクス
      frame_TTTs.png - 各時刻のフレーム + 認識オーバーレイ
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

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2
import numpy as np

from src.background_fingerprint import capture_pair_robust
from src.board import (
    BOARD_COLS,
    COLOR_EMPTY,
    HIDDEN_ROWS,
    VISIBLE_ROWS,
)
from src.hybrid_classifier import HybridClassifier
from src.image_reader import (
    DEFAULT_P1_REGION,
    DEFAULT_P2_REGION,
    ImageReader,
)
from src.match_state import MatchStateDetector
from src.patch_classifier import CnnPatchClassifier

VIDEO = "data/frames/video_01.mp4"
START_SEC = 1636.0
END_SEC = 1713.0
INTERVAL_SEC = 1.0
OUT_DIR = Path("data/verify/m27_diagnose_v2")
CNN_MODEL = "models/cnn_phase_u_v6.pt"

COLOR_LABEL = {
    0: "_", 1: "R", 2: "B", 3: "G", 4: "Y", 5: "P", 9: "O", 10: "?",
}
COLOR_BGR = {
    0: (40, 40, 40), 1: (40, 40, 200), 2: (200, 80, 40),
    3: (40, 180, 40), 4: (40, 200, 220), 5: (180, 40, 180),
    9: (170, 170, 170), 10: (80, 80, 120),
}


def overlay_board(frame, board, region) -> None:
    for vrow in range(VISIBLE_ROWS):
        for col in range(BOARD_COLS):
            row = vrow + HIDDEN_ROWS
            color = int(board.get(row, col))
            if color == COLOR_EMPTY:
                continue
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            bgr = COLOR_BGR.get(color, (60, 60, 60))
            sub = frame[y1:y2, x1:x2].astype(np.float32)
            overlay = np.full_like(sub, bgr, dtype=np.float32)
            blended = (sub * 0.6 + overlay * 0.4).astype(np.uint8)
            frame[y1:y2, x1:x2] = blended
            label = COLOR_LABEL.get(color, "?")
            cx = (x1 + x2) // 2 - 8
            cy = (y1 + y2) // 2 + 6
            cv2.putText(frame, label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2, cv2.LINE_AA)


def board_to_str(board) -> str:
    """visible 12行 × 6列 を 1 行で文字列化 (行は | 区切り)。"""
    lines = []
    for vrow in range(VISIBLE_ROWS):
        row = vrow + HIDDEN_ROWS
        chars = [
            COLOR_LABEL.get(int(board.get(row, col)), "?")
            for col in range(BOARD_COLS)
        ]
        lines.append("".join(chars))
    return "|".join(lines)


def count_non_empty(board) -> int:
    n = 0
    for vrow in range(VISIBLE_ROWS):
        row = vrow + HIDDEN_ROWS
        for col in range(BOARD_COLS):
            if int(board.get(row, col)) != COLOR_EMPTY:
                n += 1
    return n


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"video open failed: {VIDEO}")
        return 1

    # CNN
    import torch
    cnn = CnnPatchClassifier()
    state = torch.load(CNN_MODEL, map_location="cpu", weights_only=True)
    cnn._model.load_state_dict(state)
    cnn._model.eval()
    classifier = HybridClassifier(cnn_classifier=cnn)

    reader = ImageReader(
        classifier=classifier,
        use_match_state=True,
        use_ui_mask=True,
    )

    # BG FP は試合開始 +0.5s で取得
    bg_frames = []
    for offset in (-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5):
        t = max(0.0, START_SEC + 0.5 + offset)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fb = cap.read()
        if not ok or fb is None:
            continue
        if fb.shape[:2] != (1080, 1920):
            fb = cv2.resize(fb, (1920, 1080), interpolation=cv2.INTER_AREA)
        bg_frames.append(fb)
    if bg_frames:
        p1_t = (DEFAULT_P1_REGION.x, DEFAULT_P1_REGION.y,
                DEFAULT_P1_REGION.width, DEFAULT_P1_REGION.height)
        p2_t = (DEFAULT_P2_REGION.x, DEFAULT_P2_REGION.y,
                DEFAULT_P2_REGION.width, DEFAULT_P2_REGION.height)
        fp1, fp2 = capture_pair_robust(bg_frames, p1_t, p2_t)
        reader.set_background_fingerprints(fp1, fp2)
        print(f"BG FP set from {len(bg_frames)} frames")

    match_detector = MatchStateDetector.load_default()

    # フレーム間差分用
    prev_frame = None

    rows = []
    t = START_SEC
    while t <= END_SEC:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok or fr is None:
            t += INTERVAL_SEC
            continue
        if fr.shape[:2] != (1080, 1920):
            fr = cv2.resize(fr, (1920, 1080), interpolation=cv2.INTER_AREA)

        # match_state の bg_value
        ms = match_detector.detect(fr)

        # フレーム間差分 (盤面領域)
        diff_p1 = 0.0
        diff_p2 = 0.0
        if prev_frame is not None:
            for region, key in (
                (DEFAULT_P1_REGION, "p1"), (DEFAULT_P2_REGION, "p2"),
            ):
                x, y, w, h = region.x, region.y, region.width, region.height
                a = prev_frame[y:y+h, x:x+w].astype(np.int16)
                b = fr[y:y+h, x:x+w].astype(np.int16)
                d = float(np.mean(np.abs(a - b)))
                if key == "p1":
                    diff_p1 = d
                else:
                    diff_p2 = d
        prev_frame = fr.copy()

        # 認識
        b1, b2 = reader.read_both_boards(fr)
        n1 = count_non_empty(b1)
        n2 = count_non_empty(b2)
        # 右下セル (visible row=11, hidden 1 → row=12, col=5)
        rb_p1 = COLOR_LABEL.get(int(b1.get(VISIBLE_ROWS - 1 + HIDDEN_ROWS, BOARD_COLS - 1)), "?")
        rb_p2 = COLOR_LABEL.get(int(b2.get(VISIBLE_ROWS - 1 + HIDDEN_ROWS, BOARD_COLS - 1)), "?")

        # 静止画 (overlay 付き)
        out_frame = fr.copy()
        overlay_board(out_frame, b1, DEFAULT_P1_REGION)
        overlay_board(out_frame, b2, DEFAULT_P2_REGION)
        # 時刻と V 値を画面に焼き付ける
        cv2.putText(out_frame, f"t={t:.0f}s V={ms.bg_value:.0f} state={ms.state.value}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        out_path = OUT_DIR / f"frame_{int(t):04d}s.png"
        cv2.imwrite(str(out_path), out_frame)

        rows.append({
            "t": f"{t:.1f}",
            "match_state": ms.state.value,
            "bg_v": f"{ms.bg_value:.1f}",
            "bg_s": f"{ms.bg_saturation:.1f}",
            "frame_diff_p1": f"{diff_p1:.2f}",
            "frame_diff_p2": f"{diff_p2:.2f}",
            "n_puyo_p1": n1,
            "n_puyo_p2": n2,
            "rb_p1": rb_p1,
            "rb_p2": rb_p2,
            "p1_board": board_to_str(b1),
            "p2_board": board_to_str(b2),
        })
        if int(t) % 10 == 0:
            print(f"  t={t:.0f}s state={ms.state.value} V={ms.bg_value:.0f} "
                  f"n1={n1} n2={n2} rb1={rb_p1} rb2={rb_p2} diff={diff_p1:.1f}/{diff_p2:.1f}")
        t += INTERVAL_SEC

    cap.release()

    # CSV 書き出し
    out_csv = OUT_DIR / "timeline.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] {to_windows_path(str(out_csv))}")
    print(f"     {to_windows_path(str(OUT_DIR))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
