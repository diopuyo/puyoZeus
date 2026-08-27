"""is_dead 長時間持続=誤判定 の実画面証拠切り出し (計装専用、本体コード変更なし)。

代表ケースについて、動画 data/frames/video_zenchi_c0BQoMJwwQU.mp4 から
指定 t_sec のフレームを抜き出し、上段=盤面領域(1P/2P矩形+3列目窒息判定セル
のハイライト付き全体フレーム)、下段=該当側3列目(DEATH_COL=2)row0-2 拡大、
を1枚のJPEG(幅800px)に合成して logs/is_dead_persist_2026-08-23/ に保存する。

座標は src/image_reader.py の DEFAULT_P1_REGION / DEFAULT_P2_REGION
(1920x1080キャリブレーション値) をそのまま使う (読むだけ、書き換えない)。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION, HIDDEN_ROWS

VIDEO_PATH = Path("data/frames/video_zenchi_c0BQoMJwwQU.mp4")
OUT_DIR = Path("logs/is_dead_persist_2026-08-23")
FPS = 60.0
DEATH_COL = 2
DEATH_ROW = 1  # フル13行グリッドでの行 (可視row0に相当)


def _region_rect(region) -> tuple[int, int, int, int]:
    return region.x, region.y, region.x + region.width, region.y + region.height


def _death_cell_rect(region) -> tuple[int, int, int, int]:
    """DEATH_ROW/DEATH_COL セルの矩形 (画面座標)。"""
    visible_row = DEATH_ROW - HIDDEN_ROWS  # = 0 (可視最上段)
    x1 = int(region.x + DEATH_COL * region.cell_width)
    x2 = int(region.x + (DEATH_COL + 1) * region.cell_width)
    y1 = int(region.y + visible_row * region.cell_height)
    y2 = int(region.y + (visible_row + 3) * region.cell_height)  # row0-2 (3行分)
    return x1, y1, x2, y2


def extract_case(cap: cv2.VideoCapture, t_sec: float, side: str, label: str, note: str) -> None:
    region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    frame_idx = int(round(t_sec * FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        print(f"[WARN] フレーム読み込み失敗: t={t_sec} idx={frame_idx}")
        return

    overlay = frame.copy()
    # 1P/2P 両方の矩形を描画 (文脈用)
    x1, y1, x2, y2 = _region_rect(DEFAULT_P1_REGION)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
    x1, y1, x2, y2 = _region_rect(DEFAULT_P2_REGION)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
    # 該当側の窒息判定セルを強調(赤)
    dx1, dy1, dx2, dy2 = _death_cell_rect(region)
    cv2.rectangle(overlay, (dx1, dy1), (dx2, dy2), (0, 0, 255), 3)
    cv2.putText(overlay, f"t={t_sec:.2f}s side={side} {note}",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # 下段: 拡大クロップ (少しマージンを付ける)
    margin = 20
    cx1, cy1 = max(0, dx1 - margin), max(0, dy1 - margin)
    cx2, cy2 = min(frame.shape[1], dx2 + margin), min(frame.shape[0], dy2 + margin * 3)
    crop = frame[cy1:cy2, cx1:cx2]
    crop_h = 300
    scale = crop_h / max(1, crop.shape[0])
    crop_resized = cv2.resize(crop, (int(crop.shape[1] * scale), crop_h))

    # 上段リサイズ (幅800にあわせる)
    target_w = 800
    scale_top = target_w / overlay.shape[1]
    top_resized = cv2.resize(overlay, (target_w, int(overlay.shape[0] * scale_top)))

    # 下段をキャンバス中央に配置 (幅をtarget_wに合わせてパディング)
    bottom_canvas = np.full((crop_h, target_w, 3), 40, dtype=np.uint8)
    cw = crop_resized.shape[1]
    x_off = max(0, (target_w - cw) // 2)
    bottom_canvas[:, x_off:x_off + min(cw, target_w)] = crop_resized[:, :max(0, target_w - x_off)]

    canvas = np.vstack([top_resized, bottom_canvas])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{label}.jpg"
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[保存] {out_path}")


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {VIDEO_PATH}")

    cases = [
        # (t_sec, side, label, note)
        (6700.00, "1P", "01_seg08_1P_t6700_連鎖前凍結中", "dead1=True(凍結,STABLE)"),
        (6705.00, "1P", "02_seg08_1P_t6705_連鎖中", "dead1=True(CHAIN継続中)"),
        (6716.00, "1P", "03_seg08_1P_t6716_連鎖中終盤", "dead1=True(GRAVITY_SETTLE手前)"),
        (6717.40, "1P", "04_seg08_1P_t6717_連鎖後解消", "dead1=False(連鎖後盤面更新)"),
        (30.00, "2P", "05_seg01_2P_t30_最長29.7秒区間", "dead2=True(要確認)"),
        (814.00, "1P", "06_seg01_1P_t814_旧誤報告事例", "dead1=True(t=807.667付近、旧報告=誤検証)"),
        (2015.00, "2P", "07_seg03_2P_t2015", "dead2=True(要確認)"),
        (6499.00, "2P", "08_seg08_2P_t6499", "dead2=True(要確認)"),
    ]
    for t_sec, side, label, note in cases:
        extract_case(cap, t_sec, side, label, note)

    cap.release()


if __name__ == "__main__":
    main()
