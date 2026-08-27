"""双方同時窒息(156件, user報告152件相当)の代表エピソードについて、
元動画+オーバーレイ版+窒息判定領域(3列目可視最上段row1)のズームを1枚に
合成した診断画像を作る計装スクリプト (コード変更なし、スマホ閲覧用に軽量化)。

出力: logs/dualdeath_frames_2026-08-22/ 配下、幅960pxのPNG (1エピソード1枚)。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

BASE = Path(__file__).resolve().parent.parent
FF = imageio_ffmpeg.get_ffmpeg_exe()
SRC_VIDEO = BASE / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
OUT_DIR = BASE / "logs/dualdeath_frames_2026-08-22"
TMP_DIR = OUT_DIR / "_tmp"

SEGMENTS = [
    ("seg01", 0.0, 893.7, BASE / "data/verify/zenchi_render_2026-08-21/seg01_0_893.7.mp4"),
    ("seg02", 893.7, 1738.3, BASE / "data/verify/zenchi_render_2026-08-21/seg02_893.7_1738.3.mp4"),
    ("seg03", 1738.3, 2637.3, BASE / "data/verify/zenchi_render_2026-08-21/seg03_1738.3_2637.3.mp4"),
    ("seg04", 2637.3, 3626.0, BASE / "data/verify/zenchi_render_2026-08-21/seg04_2637.3_3626.0.mp4"),
    ("seg05", 3626.0, 4379.5, BASE / "data/verify/zenchi_render_2026-08-21/seg05_3626.0_4379.5.mp4"),
    ("seg06", 4379.5, 5255.6, BASE / "data/verify/zenchi_render_2026-08-21/seg06_4379.5_5255.6.mp4"),
    ("seg07", 5255.6, 6131.6, BASE / "data/verify/zenchi_render_2026-08-21/seg07_5255.6_6131.6.mp4"),
    ("seg08", 6131.6, 7033.6, BASE / "data/verify/zenchi_render_2026-08-21/seg08_6131.6_7033.6.mp4"),
]

# 盤面座標 (userから提示): 1P x282-666,y160-880 / 2P x1258-1642,y160-880 (6列x13行)
BOARD_1P = (282, 160, 666, 880)
BOARD_2P = (1258, 160, 1642, 880)
N_COLS = 6
N_ROWS = 13


def col_row_box(board_xyxy: tuple[int, int, int, int], col: int, row_start: int, row_end: int):
    x0, y0, x1, y1 = board_xyxy
    cw = (x1 - x0) / N_COLS
    rh = (y1 - y0) / N_ROWS
    cx0 = int(x0 + cw * col)
    cx1 = int(x0 + cw * (col + 1))
    cy0 = int(y0 + rh * row_start)
    cy1 = int(y0 + rh * row_end)
    return cx0, cy0, cx1, cy1


def find_segment(t: float):
    for name, s, e, path in SEGMENTS:
        if s <= t <= e:
            return name, s, path
    best = min(SEGMENTS, key=lambda seg: min(abs(t - seg[1]), abs(t - seg[2])))
    return best[0], best[1], best[3]


def extract_frame(video: Path, t_sec: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FF, "-y", "-ss", f"{max(0.0, t_sec):.3f}", "-i", str(video),
           "-frames:v", "1", "-q:v", "2", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def build_composite(label_jp: str, t_global: float, out_name: str) -> Path:
    """1枚の合成画像を作る: [元動画(枠付き)|オーバーレイ動画] 上段960x270、
    下段に1P/2P の窒息判定領域(col2, row0-2)ズーム。
    """
    seg_name, seg_start, seg_path = find_segment(t_global)
    src_png = TMP_DIR / f"src_{t_global:.2f}.png"
    ov_png = TMP_DIR / f"ov_{t_global:.2f}.png"
    extract_frame(SRC_VIDEO, t_global, src_png)
    local_t = t_global - seg_start
    extract_frame(seg_path, local_t, ov_png)

    src = cv2.imread(str(src_png))
    ov = cv2.imread(str(ov_png)) if ov_png.exists() else np.zeros_like(src)

    # 窒息判定領域(3列目=index2, row0隠し段+row1可視最上段+row2まで少し余分に表示)
    box_1p = col_row_box(BOARD_1P, 2, 0, 3)
    box_2p = col_row_box(BOARD_2P, 2, 0, 3)

    src_annot = src.copy()
    cv2.rectangle(src_annot, (box_1p[0], box_1p[1]), (box_1p[2], box_1p[3]), (0, 0, 255), 3)
    cv2.rectangle(src_annot, (box_2p[0], box_2p[1]), (box_2p[2], box_2p[3]), (0, 0, 255), 3)

    def crop_zoom(img, box, scale=4):
        x0, y0, x1, y1 = box
        crop = img[y0:y1, x0:x1]
        return cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

    zoom_1p = crop_zoom(src, box_1p)
    zoom_2p = crop_zoom(src, box_2p)

    # 上段: 元動画(枠付き) + オーバーレイ版、各480幅にリサイズ
    top_w = 480
    src_small = cv2.resize(src_annot, (top_w, int(src_annot.shape[0] * top_w / src_annot.shape[1])))
    ov_small = cv2.resize(ov, (top_w, int(ov.shape[0] * top_w / ov.shape[1])))
    top_h = max(src_small.shape[0], ov_small.shape[0])
    top_row = np.zeros((top_h, top_w * 2, 3), dtype=np.uint8)
    top_row[:src_small.shape[0], :top_w] = src_small
    top_row[:ov_small.shape[0], top_w:] = ov_small
    cv2.putText(top_row, "genga(moto douga)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(top_row, "overlay(ninshiki)", (top_w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 下段: 1P/2Pのズーム(row0-2, col2)を並べる
    zoom_h = max(zoom_1p.shape[0], zoom_2p.shape[0])
    pad = 20
    bottom_row = np.full((zoom_h + 40, top_w * 2, 3), 40, dtype=np.uint8)
    bottom_row[30:30 + zoom_1p.shape[0], pad:pad + zoom_1p.shape[1]] = zoom_1p
    bottom_row[30:30 + zoom_2p.shape[0], top_w + pad:top_w + pad + zoom_2p.shape[1]] = zoom_2p
    cv2.putText(bottom_row, "1P col3 row0-2 zoom", (pad, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(bottom_row, "2P col3 row0-2 zoom", (top_w + pad, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    canvas = np.vstack([top_row, bottom_row])
    label_bar = np.full((30, canvas.shape[1], 3), 0, dtype=np.uint8)
    cv2.putText(label_bar, label_jp, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    canvas = np.vstack([label_bar, canvas])

    out_path = OUT_DIR / out_name
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return out_path


EPISODES = [
    ("A_t16", 16.1, "t=16.1s game_idx=0"),
    ("B_t77", 77.1, "t=77.1s game_idx=0"),
    ("C_t1656", 1656.2, "t=1656.2s game_idx=14"),
    ("D_t4329", 4328.9, "t=4328.9s game_idx=11"),
    ("E_t4896", 4896.4, "t=4896.4s game_idx=9 (2.4s cluster)"),
    ("F_t5228", 5228.7, "t=5228.7s game_idx=0or15"),
    ("G_t5773", 5773.3, "t=5773.3s game_idx=9"),
]

# 最終提出用: 5枚に絞り、日本語ファイル名+JPEG圧縮(スマホ閲覧向け)
FINAL_SELECTION = [
    ("A_t16", "誤認_白フラッシュ_t16s_game0.jpg"),
    ("B_t77", "誤認_灰色ロード画面_t77s_game0.jpg"),
    ("E_t4896", "誤認_試合間結果演出_t4896s_game9_最長2.4秒.jpg"),
    ("F_t5228", "誤認_試合間結果演出_t5228s_gameidx重複疑い.jpg"),
    ("G_t5773", "誤認_試合間結果演出_t5773s_game9_別試合再現.jpg"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    raw_paths: dict[str, Path] = {}
    for label, t, desc in EPISODES:
        out = build_composite(desc, t, f"raw_{label}.png")
        raw_paths[label] = out
        print(f"{label}: {out} ({out.stat().st_size/1024:.0f} KB)")

    print("\n--- 最終選定(縮小+JPEG圧縮、日本語ファイル名) ---")
    for label, jp_name in FINAL_SELECTION:
        img = cv2.imread(str(raw_paths[label]))
        # 幅800pxに縮小(スマホ閲覧・軽量化)
        target_w = 800
        h, w = img.shape[:2]
        img_small = cv2.resize(img, (target_w, int(h * target_w / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img_small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out_path = OUT_DIR / jp_name
        out_path.write_bytes(buf.tobytes())
        print(f"  {out_path} ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
