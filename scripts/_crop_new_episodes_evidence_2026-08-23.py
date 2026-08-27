"""32件の「新規」符号衝突エピソードのうち「真に新規」候補について、実画面を
切り出す (2026-08-23、根治版レビュー用)。

上段=生フレーム全体 (dump 値をテキストで焼き込み)、下段=1P/2P 盤面領域を
並べたもの。幅800px JPEG、日本語ファイル名で出力する
(memory feedback_review_actual_screen_frames_2026-07-24 /
feedback_review_image_links)。

コードは変更しない (動画を読んで切り出すだけ)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402

VIDEO = PROJECT_ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
OUT_DIR = PROJECT_ROOT / "logs/new_episodes_v2_2026-08-23"
OUT_WIDTH = 800


def read_frame(cap: cv2.VideoCapture, t_sec: float) -> "np.ndarray | None":
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame


def crop_board(frame: np.ndarray, region) -> np.ndarray:
    return frame[region.y:region.y + region.height, region.x:region.x + region.width]


def build_evidence(
    cap: cv2.VideoCapture, t_sec: float, label: str, values: dict,
) -> "np.ndarray | None":
    frame = read_frame(cap, t_sec)
    if frame is None:
        print(f"[警告] フレーム読み込み失敗: t={t_sec}")
        return None
    full = frame.copy()
    # 上段: 全画面 + テキスト焼き込み (dump値)
    text_lines = [
        label,
        f"t={values['t_sec']:.2f}s state1={values['state1']} state2={values['state2']}",
        f"pend1={values['pend1']:.0f} pend2={values['pend2']:.0f} "
        f"room1={values['room1']:.0f} room2={values['room2']:.0f}",
        f"adv_raw={values['adv_raw']:.1f} adv_ema={values['adv_ema']:.1f}",
    ]
    bar_h = 30 * len(text_lines) + 10
    bar = np.zeros((bar_h, full.shape[1], 3), dtype=np.uint8)
    for i, line in enumerate(text_lines):
        cv2.putText(bar, line, (10, 28 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
    top = np.vstack([bar, full])

    # 下段: 1P/2P 盤面領域を並べる (3倍拡大)
    p1 = crop_board(frame, DEFAULT_P1_REGION)
    p2 = crop_board(frame, DEFAULT_P2_REGION)
    zoom = 1.5
    p1 = cv2.resize(p1, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    p2 = cv2.resize(p2, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    h = max(p1.shape[0], p2.shape[0])
    pad = np.zeros((h, 20, 3), dtype=np.uint8)
    boards = np.hstack([p1, pad, p2])
    label_bar = np.zeros((30, boards.shape[1], 3), dtype=np.uint8)
    cv2.putText(label_bar, "1P board (left) / 2P board (right)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    bottom = np.vstack([label_bar, boards])

    # 横幅を揃えて縦結合
    w = max(top.shape[1], bottom.shape[1])
    top = cv2.copyMakeBorder(top, 0, 0, 0, w - top.shape[1], cv2.BORDER_CONSTANT, value=0)
    bottom = cv2.copyMakeBorder(
        bottom, 0, 0, 0, w - bottom.shape[1], cv2.BORDER_CONSTANT, value=0)
    combined = np.vstack([top, bottom])

    scale = OUT_WIDTH / combined.shape[1]
    combined = cv2.resize(
        combined, (OUT_WIDTH, int(combined.shape[0] * scale)),
        interpolation=cv2.INTER_AREA)
    return combined


def find_row(dump_path: Path, t_target: float):
    d = np.load(dump_path, allow_pickle=True)
    t = d["t_sec"]
    i = int(np.argmin(np.abs(t - t_target)))
    return dict(
        t_sec=float(t[i]), state1=str(d["state1"][i]), state2=str(d["state2"][i]),
        pend1=float(d["pending_p1"][i]), pend2=float(d["pending_p2"][i]),
        room1=float(d["room1"][i]), room2=float(d["room2"][i]),
        adv_raw=float(d["adv_raw"][i]), adv_ema=float(d["adv_ema"][i]),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dump_dir = PROJECT_ROOT / "data/verify/zenchi_render_slide_exit_guard_v2_2026-08-22"

    # (dump_file, t_sec, 日本語ラベル)
    targets = [
        ("seg01_0_893.7.npz", 366.07, "true_new_seg01_t366_P1連鎖中P1側大量pending"),
        ("seg06_4379.5_5255.6.npz", 4585.60, "true_new_seg06_t4585_連鎖直後の一瞬"),
        ("seg06_4379.5_5255.6.npz", 4609.67, "true_new_seg06_t4609_フラグメント最長12秒"),
        ("seg08_6131.6_7033.6.npz", 6619.87, "true_new_seg08_t6619_P1room1のみ極小"),
        ("seg08_6131.6_7033.6.npz", 6701.63, "true_new_seg08_t6701_符号反転直後"),
    ]

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise SystemExit(f"動画を開けない: {VIDEO}")
    try:
        for fname, t_target, label in targets:
            row = find_row(dump_dir / fname, t_target)
            img = build_evidence(cap, row["t_sec"], label, row)
            if img is None:
                continue
            out_path = OUT_DIR / f"{label}.jpg"
            cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"[出力] {out_path}  (dump値: {row})")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
