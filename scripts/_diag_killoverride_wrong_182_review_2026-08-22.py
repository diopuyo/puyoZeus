"""kill_override 誤上書き事象 (|adv_disp|>=80、片方STABLE窒息・他方連鎖中・
向き逆転) の182件を連続時刻(gap<=0.5s)でエピソード化し、代表1コマずつ実画面を
生成する計装スクリプト (計装のみ、本体コード変更なし)。

対象データ:
  data/verify/judgment_scan_zenchi_recheck_2026-08-22_fast/suspects.tsv
    (D1a/D1b の evidence テキストから adv_raw/adv_disp を正規表現抽出)
  data/verify/judgment_scan_zenchi_recheck_2026-08-22_fast/d1a_d1b_chain_crosscheck.tsv
    (state1/state2/is_dead1/is_dead2/pending/room を (t_sec,detector,stage,game_idx)
     キーで突合)

抽出条件 (182件を再現する条件、実測で確認済み):
  detector=='D1a' (確定死の無視) かつ side_state=='STABLE' かつ
  other_state in {CHAIN, GRAVITY_SETTLE} (狭義連鎖中) かつ
  side に対応する is_dead が True (フラグ側=窒息側の整合性) かつ
  |adv_disp| >= 80

出力:
  logs/killoverride_wrong_2026-08-22/NN_tXXXX_..._..._X.X秒.jpg (代表1コマ)
  logs/killoverride_wrong_2026-08-22/一覧.tsv
"""
from __future__ import annotations

import csv
import re
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

SUSP_TSV = BASE / "data/verify/judgment_scan_zenchi_recheck_2026-08-22_fast/suspects.tsv"
CROSS_TSV = BASE / "data/verify/judgment_scan_zenchi_recheck_2026-08-22_fast/d1a_d1b_chain_crosscheck.tsv"

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

BOARD_1P = (282, 160, 666, 880)
BOARD_2P = (1258, 160, 1642, 880)
N_COLS = 6
N_ROWS = 13

EVIDENCE_PAT = re.compile(r"adv_raw=([+-]?[0-9.]+)/.*?adv_disp=([+-]?[0-9.]+)/")


def col_row_box(board_xyxy, col, row_start, row_end):
    x0, y0, x1, y1 = board_xyxy
    cw = (x1 - x0) / N_COLS
    rh = (y1 - y0) / N_ROWS
    return (int(x0 + cw * col), int(y0 + rh * row_start),
            int(x0 + cw * (col + 1)), int(y0 + rh * row_end))


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


def load_target_rows() -> list[dict]:
    """182件 (|adv_disp|>=80、片方STABLE窒息・他方連鎖中) を再現する。"""
    susp_rows: dict[tuple, tuple[float, float]] = {}
    with SUSP_TSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["detector"] not in ("D1a", "D1b"):
                continue
            if row["stage"] not in ("display", "both"):
                continue
            m = EVIDENCE_PAT.search(row["evidence"])
            if not m:
                continue
            key = (row["t_sec"], row["detector"], row["stage"], row["game_idx"])
            susp_rows[key] = (float(m.group(1)), float(m.group(2)))

    cross_rows: list[dict] = []
    with CROSS_TSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row["t_sec"], row["detector"], row["stage"], row["game_idx"])
            adv = susp_rows.get(key)
            if adv is None:
                continue
            row["adv_raw"], row["adv_disp"] = adv
            cross_rows.append(row)

    narrow = {"CHAIN", "GRAVITY_SETTLE"}

    def dead_consistent(r: dict) -> bool:
        return (r["is_dead1"] == "True") if r["side"] == "1P" else (r["is_dead2"] == "True")

    profile = [
        r for r in cross_rows
        if r["detector"] == "D1a" and r["side_state"] == "STABLE"
        and r["other_state"] in narrow and dead_consistent(r)
    ]
    big = [r for r in profile if abs(r["adv_disp"]) >= 80]
    big.sort(key=lambda r: float(r["t_sec"]))
    return big


def build_episodes(rows: list[dict]) -> list[list[dict]]:
    """連続時刻 (gap<=0.5秒、game_idx一致) でグルーピングする。"""
    episodes: list[list[dict]] = []
    cur: list[dict] = []
    for r in rows:
        t = float(r["t_sec"])
        if cur and (t - float(cur[-1]["t_sec"]) > 0.5 or r["game_idx"] != cur[-1]["game_idx"]):
            episodes.append(cur)
            cur = []
        cur.append(r)
    if cur:
        episodes.append(cur)
    return episodes


def lookup_npz_stats(t_global: float, game_idx: int) -> dict:
    _, seg_start, seg_end, _, npz_path = find_segment(t_global)
    d = np.load(npz_path, allow_pickle=True)
    t = d["t_sec"]
    idx = int(np.argmin(np.abs(t - t_global)))
    return {
        "p1": float(d["p1"][idx]),
        "p1_raw": float(d["p1_raw"][idx]),
    }


def build_composite(t_global: float, dead_side: str, out_path_png: Path) -> None:
    seg_name, seg_start, seg_end, seg_path, npz_path = find_segment(t_global)

    src_png = TMP_DIR / f"src_{t_global:.3f}.png"
    ov_png = TMP_DIR / f"ov_{t_global:.3f}.png"
    extract_frame(SRC_VIDEO, t_global, src_png)
    local_t = t_global - seg_start
    extract_frame(seg_path, local_t, ov_png)

    src = cv2.imread(str(src_png))
    ov = cv2.imread(str(ov_png))
    if ov is None:
        raise RuntimeError(f"overlay frame 抽出失敗: {seg_path} @ {local_t}")

    dead_box_key = BOARD_1P if dead_side == "1P" else BOARD_2P
    box_dead = col_row_box(dead_box_key, 2, 0, 3)

    def crop_zoom(img, box, scale=5):
        x0, y0, x1, y1 = box
        crop = img[y0:y1, x0:x1]
        return cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                           interpolation=cv2.INTER_NEAREST)

    zoom_dead = crop_zoom(src, box_dead)

    top_w = 960
    ov_small = cv2.resize(ov, (top_w, int(ov.shape[0] * top_w / ov.shape[1])))

    zoom_h = zoom_dead.shape[0]
    bottom_h = zoom_h + 40
    bottom_row = np.full((bottom_h, top_w, 3), 40, dtype=np.uint8)
    x_off = (top_w - zoom_dead.shape[1]) // 2
    bottom_row[30:30 + zoom_dead.shape[0], x_off:x_off + zoom_dead.shape[1]] = zoom_dead
    cv2.putText(bottom_row, f"{dead_side} col3(index2) row0-2 zoom (row1=12dan=death check row)",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    canvas = np.vstack([ov_small, bottom_row])
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path_png), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_target_rows()
    print(f"対象行数 (|adv_disp|>=80、条件一致) = {len(rows)}")
    episodes = build_episodes(rows)
    print(f"エピソード数 = {len(episodes)}")

    manifest_path = OUT_DIR / "一覧.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as mf:
        writer = csv.writer(mf, delimiter="\t")
        writer.writerow([
            "通し番号", "t_sec", "game_idx", "継続秒", "窒息side",
            "is_dead1", "is_dead2", "state1", "state2",
            "pending_p1", "pending_p2", "room1", "room2",
            "adv_raw", "adv_disp", "表示された1P勝率", "画像ファイル名",
        ])

        for i, ep in enumerate(episodes, start=1):
            t0 = float(ep[0]["t_sec"])
            t1 = float(ep[-1]["t_sec"])
            dur = t1 - t0
            rep = max(ep, key=lambda r: abs(r["adv_disp"]))
            dead_side = rep["side"]
            game_idx = int(rep["game_idx"])
            npz_stats = lookup_npz_stats(float(rep["t_sec"]), game_idx)
            p1_disp_pct = npz_stats["p1"] * 100.0

            t_int = int(float(rep["t_sec"]))
            other_side = "2P" if dead_side == "1P" else "1P"
            fname_base = (
                f"{i:02d}_t{t_int}_{dead_side}窒息なのに{dead_side}"
                f"{round(p1_disp_pct if dead_side=='1P' else 100-p1_disp_pct)}%_{dur:.1f}秒"
            )
            png_path = OUT_DIR / f"raw_{fname_base}.png"
            build_composite(float(rep["t_sec"]), dead_side, png_path)

            img = cv2.imread(str(png_path))
            target_w = 800
            h, w = img.shape[:2]
            img_small = cv2.resize(img, (target_w, int(h * target_w / w)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img_small, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpg_name = f"{fname_base}.jpg"
            jpg_path = OUT_DIR / jpg_name
            jpg_path.write_bytes(buf.tobytes())

            writer.writerow([
                i, f"{float(rep['t_sec']):.3f}", game_idx, f"{dur:.2f}", dead_side,
                rep["is_dead1"], rep["is_dead2"], rep["state1"], rep["state2"],
                rep["pending_p1"], rep["pending_p2"], rep["room1"], rep["room2"],
                f"{rep['adv_raw']:.1f}", f"{rep['adv_disp']:.1f}",
                f"{p1_disp_pct:.1f}%", jpg_name,
            ])
            print(f"[{i}] {jpg_path} ({jpg_path.stat().st_size/1024:.0f} KB) "
                  f"t={rep['t_sec']} dur={dur:.2f}s dead={dead_side} "
                  f"adv_raw={rep['adv_raw']:.1f} adv_disp={rep['adv_disp']:.1f}")

    print(f"\n一覧表: {manifest_path}")


if __name__ == "__main__":
    main()
