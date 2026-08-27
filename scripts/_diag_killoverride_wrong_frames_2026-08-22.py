"""kill_override が窒息済み側を勝者として完全上書きする事象の実画面抽出
(計装のみ、本体コード変更なし)。

対象: data/verify/judgment_scan_zenchi_recheck_2026-08-22_fast/d1a_d1b_chain_crosscheck.tsv
     で検出された「片方STABLE(窒息)・もう片方CHAIN/GRAVITY_SETTLE中」の
     kill_override 完全上書き (|adv_ema|>=95 かつ adv_raw と符号反転)。

出力: logs/killoverride_wrong_2026-08-22/ 配下、幅800px JPEG。
  上段: オーバーレイ動画1フレーム (勝率パネル+両者盤面)
  下段: 窒息側の3列目(index2)可視最上段(row1)を含む row0-2 ズーム
  帯: t_sec/game_idx/is_dead1/is_dead2/state1/state2/pending/room/adv_raw/adv_ema/p1/p1_raw
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
OUT_DIR = BASE / "logs/killoverride_wrong_2026-08-22"
TMP_DIR = OUT_DIR / "_tmp"

SEGMENTS = [
    ("seg01", 0.0, 893.7, BASE / "data/verify/zenchi_render_2026-08-21/seg01_0_893.7.mp4",
     BASE / "data/verify/zenchi_render_2026-08-21/seg01_0_893.7.npz"),
    ("seg02", 893.7, 1738.3, BASE / "data/verify/zenchi_render_2026-08-21/seg02_893.7_1738.3.mp4",
     BASE / "data/verify/zenchi_render_2026-08-21/seg02_893.7_1738.3.npz"),
    ("seg03", 1738.3, 2637.3, BASE / "data/verify/zenchi_render_2026-08-21/seg03_1738.3_2637.3.mp4",
     BASE / "data/verify/zenchi_render_2026-08-21/seg03_1738.3_2637.3.npz"),
    ("seg04", 2637.3, 3626.0, BASE / "data/verify/zenchi_render_2026-08-21/seg04_2637.3_3626.0.mp4",
     BASE / "data/verify/zenchi_render_2026-08-21/seg04_2637.3_3626.0.npz"),
    ("seg05", 3626.0, 4379.5, BASE / "data/verify/zenchi_render_2026-08-21/seg05_3626.0_4379.5.mp4",
     BASE / "data/verify/zenchi_render_2026-08-21/seg05_3626.0_4379.5.npz"),
    ("seg06", 4379.5, 5255.6, BASE / "data/verify/zenchi_render_2026-08-21/seg06_4379.5_5255.6.mp4",
     BASE / "data/verify/zenchi_render_2026-08-21/seg06_4379.5_5255.6.npz"),
    ("seg07", 5255.6, 6131.6, BASE / "data/verify/zenchi_render_2026-08-21/seg07_5255.6_6131.6.mp4",
     BASE / "data/verify/zenchi_render_2026-08-21/seg07_5255.6_6131.6.npz"),
    ("seg08", 6131.6, 7033.6, BASE / "data/verify/zenchi_render_2026-08-21/seg08_6131.6_7033.6.mp4",
     BASE / "data/verify/zenchi_render_2026-08-21/seg08_6131.6_7033.6.npz"),
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
    for seg in SEGMENTS:
        name, s, e, path, npz = seg
        if s <= t <= e:
            return seg
    raise ValueError(f"t={t} が全セグメント範囲外")


def extract_frame(video: Path, t_sec: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FF, "-y", "-ss", f"{max(0.0, t_sec):.3f}", "-i", str(video),
           "-frames:v", "1", "-q:v", "2", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def lookup_frame_stats(npz_path: Path, t_global: float) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    t = d["t_sec"]
    idx = int(np.argmin(np.abs(t - t_global)))
    return {
        "t_sec": float(t[idx]),
        "game_idx": int(d["game_idx"][idx]),
        "is_dead1": bool(d["is_dead1"][idx]),
        "is_dead2": bool(d["is_dead2"][idx]),
        "state1": str(d["state1"][idx]),
        "state2": str(d["state2"][idx]),
        "pending_p1": int(d["pending_p1"][idx]),
        "pending_p2": int(d["pending_p2"][idx]),
        "room1": int(d["room1"][idx]),
        "room2": int(d["room2"][idx]),
        "adv_raw": float(d["adv_raw"][idx]),
        "adv_ema": float(d["adv_ema"][idx]),
        "p1": float(d["p1"][idx]),
        "p1_raw": float(d["p1_raw"][idx]),
    }


def build_composite(t_global: float, dead_side: str, out_name: str, label_jp: str) -> tuple[Path, dict]:
    seg_name, seg_start, seg_end, seg_path, npz_path = find_segment(t_global)
    stats = lookup_frame_stats(npz_path, t_global)

    src_png = TMP_DIR / f"src_{t_global:.3f}.png"
    ov_png = TMP_DIR / f"ov_{t_global:.3f}.png"
    extract_frame(SRC_VIDEO, t_global, src_png)
    local_t = t_global - seg_start
    extract_frame(seg_path, local_t, ov_png)

    src = cv2.imread(str(src_png))
    ov = cv2.imread(str(ov_png))
    if ov is None:
        raise RuntimeError(f"overlay frame 抽出失敗: {seg_path} @ {local_t}")

    # 窒息側の col3(index2) row0(隠し段)-row2 を少し広めにズーム
    dead_box_key = BOARD_1P if dead_side == "1P" else BOARD_2P
    box_dead = col_row_box(dead_box_key, 2, 0, 3)

    def crop_zoom(img, box, scale=5):
        x0, y0, x1, y1 = box
        crop = img[y0:y1, x0:x1]
        return cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                           interpolation=cv2.INTER_NEAREST)

    zoom_dead = crop_zoom(src, box_dead)

    # 上段: オーバーレイ版全体 (勝率パネル+両盤面が入るよう幅を確保)
    top_w = 960
    ov_small = cv2.resize(ov, (top_w, int(ov.shape[0] * top_w / ov.shape[1])))

    # 下段: 窒息側ズーム (中央寄せ)
    zoom_h = zoom_dead.shape[0]
    bottom_h = zoom_h + 40
    bottom_row = np.full((bottom_h, top_w, 3), 40, dtype=np.uint8)
    x_off = (top_w - zoom_dead.shape[1]) // 2
    bottom_row[30:30 + zoom_dead.shape[0], x_off:x_off + zoom_dead.shape[1]] = zoom_dead
    side_jp = "1P" if dead_side == "1P" else "2P"
    cv2.putText(bottom_row, f"{side_jp} col3(index2) row0-2 zoom (row1=12段目=窒息判定行)",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    canvas = np.vstack([ov_small, bottom_row])

    # 情報帯 (2行)
    info1 = (f"t={stats['t_sec']:.3f}s game_idx={stats['game_idx']} "
             f"is_dead1={stats['is_dead1']} is_dead2={stats['is_dead2']} "
             f"state1={stats['state1']} state2={stats['state2']}")
    info2 = (f"pending_p1={stats['pending_p1']} pending_p2={stats['pending_p2']} "
             f"room1={stats['room1']} room2={stats['room2']} "
             f"adv_raw={stats['adv_raw']:.1f}(p1_raw={stats['p1_raw']*100:.1f}%) "
             f"adv_disp={stats['adv_ema']:.1f}(p1={stats['p1']*100:.1f}%)")
    label_bar = np.full((70, canvas.shape[1], 3), 0, dtype=np.uint8)
    cv2.putText(label_bar, label_jp, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(label_bar, info1, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(label_bar, info2, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    canvas = np.vstack([label_bar, canvas])

    raw_path = OUT_DIR / f"raw_{out_name}.png"
    cv2.imwrite(str(raw_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    # 縮小+JPEG化
    target_w = 800
    h, w = canvas.shape[:2]
    img_small = cv2.resize(canvas, (target_w, int(h * target_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    jpg_path = OUT_DIR / f"{out_name}.jpg"
    jpg_path.write_bytes(buf.tobytes())
    return jpg_path, stats


EPISODES = [
    # (t_global, 窒息側, ファイル名(拡張子なし), 日本語ラベル)
    (807.667, "1P",
     "安全弁誤上書き_t807_1P窒息なのに1P99%",
     "[必須] t=807.667s game_idx=11: 1P窒息(STABLE)なのに1P勝率99%表示"),
    (4916.133, "1P",
     "安全弁誤上書き_t4916_1P窒息なのに1P99%_最長2.97秒",
     "t=4916.13s game_idx=9: 1P窒息(STABLE)なのに1P勝率99% (継続2.97秒=最長級)"),
    (1032.9, "2P",
     "安全弁誤上書き_t1032_2P窒息なのに2P99%_逆方向",
     "t=1032.9s game_idx=3: 2P窒息(STABLE)なのに2P勝率99% (逆方向、継続2.63秒)"),
    (7018.233, "1P",
     "安全弁誤上書き_t7018_1P窒息なのに1P99%_GRAVITY_SETTLE版",
     "t=7018.23s game_idx=16: 1P窒息(STABLE)なのに1P勝率99% (相手GRAVITY_SETTLE中)"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for t_global, dead_side, out_name, label_jp in EPISODES:
        path, stats = build_composite(t_global, dead_side, out_name, label_jp)
        print(f"{out_name}: {path} ({path.stat().st_size/1024:.0f} KB)")
        print(f"  {stats}")


if __name__ == "__main__":
    main()
