"""score OCR の精度測定用ラベリング grid を生成する。

戦略:
    - score_series_cache から「OCR 信頼度が高い」フレームを抽出
    - score 表示が安定している瞬間 (連鎖中ではない、メニューでもない)
    - video_01/02/03 各 5 フレームずつ計 15 フレーム
    - 各フレームの 1P/2P の score ROI 部分を拡大表示
    - ユーザは grid を見て真の score 値を読み取り、ラベル付けする

出力:
    data/verify/score_label_grid.png
    data/verify/score_label_index.tsv
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.score_ocr import (
    EXPECTED_FRAME_SHAPE,
    SCORE_1P_REGION,
    SCORE_2P_REGION,
    ScoreOcr,
)

CACHE_PATH = Path("data/training/score_series_cache.json")
OUT_GRID = Path("data/verify/score_label_grid.png")
OUT_INDEX = Path("data/verify/score_label_index.tsv")
VIDEO_DIR = Path("data/frames")

# 動画ごとに採用する候補数
PER_VIDEO_N: int = 5
# OCR 信頼度がこれ以上のフレームから選ぶ (連鎖中・計算式表示を弾くため)
MIN_CONF: float = 0.7
# 多様性: 同じ試合からは最大 2 件
MAX_PER_MATCH: int = 2
# ROI 拡大率
SCALE: int = 2


def select_candidates(cache: dict) -> list[dict]:
    """各動画から信頼度の高い「score 値が多様な」サンプルを選ぶ。

    score 値レンジを 4 ビン (1-100 / 100-10000 / 10000-100000 / 100000+) に
    分けて、各ビンから均等に採用することで上位桁の精度測定もできるようにする。
    """
    bins = [(1, 100), (100, 10000), (10000, 100000), (100000, 100000000)]
    selected: list[dict] = []
    for vid in sorted(cache.keys()):
        # 全有効サンプルを 1 列に集める (試合区切りを跨ぐ)
        all_samples: list[dict] = []
        for midx, samples in cache[vid].items():
            for s in samples:
                if s["1p"] is None or s["2p"] is None:
                    continue
                if s["c1"] is None or s["c2"] is None:
                    continue
                if s["c1"] < MIN_CONF or s["c2"] < MIN_CONF:
                    continue
                all_samples.append({
                    "video": vid, "match": midx,
                    "t_sec": s["t"],
                    "ocr_1p": s["1p"], "ocr_2p": s["2p"],
                    "conf_1p": s["c1"], "conf_2p": s["c2"],
                })

        # ビンごとにサンプルを分類し、信頼度高い順に選ぶ
        per_bin_target = max(1, PER_VIDEO_N // len(bins))
        chosen_keys: set[tuple[str, float]] = set()
        chosen_for_video: list[dict] = []
        for bin_lo, bin_hi in bins:
            in_bin = [
                s for s in all_samples
                if bin_lo <= max(s["ocr_1p"], s["ocr_2p"]) < bin_hi
            ]
            in_bin.sort(
                key=lambda s: (s["conf_1p"] + s["conf_2p"]) / 2,
                reverse=True,
            )
            taken = 0
            for s in in_bin:
                key = (s["match"], round(s["t_sec"], 1))
                if key in chosen_keys:
                    continue
                chosen_keys.add(key)
                chosen_for_video.append(s)
                taken += 1
                if taken >= per_bin_target:
                    break
        # 不足分は残りから補充
        if len(chosen_for_video) < PER_VIDEO_N:
            all_samples.sort(
                key=lambda s: (s["conf_1p"] + s["conf_2p"]) / 2,
                reverse=True,
            )
            for s in all_samples:
                if len(chosen_for_video) >= PER_VIDEO_N:
                    break
                key = (s["match"], round(s["t_sec"], 1))
                if key in chosen_keys:
                    continue
                chosen_keys.add(key)
                chosen_for_video.append(s)

        selected.extend(chosen_for_video[:PER_VIDEO_N])
    return selected


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


def build_panel(idx: int, info: dict, frame: np.ndarray) -> np.ndarray:
    """1 サンプルのパネル: 1P + 2P ROI 拡大 + OCR 出力ラベル。"""
    y1, y2, x1, x2 = SCORE_1P_REGION
    roi_1p = frame[y1:y2, x1:x2]
    y1, y2, x1, x2 = SCORE_2P_REGION
    roi_2p = frame[y1:y2, x1:x2]
    big_1p = cv2.resize(roi_1p,
                         (roi_1p.shape[1] * SCALE, roi_1p.shape[0] * SCALE),
                         interpolation=cv2.INTER_NEAREST)
    big_2p = cv2.resize(roi_2p,
                         (roi_2p.shape[1] * SCALE, roi_2p.shape[0] * SCALE),
                         interpolation=cv2.INTER_NEAREST)
    sep = np.full((big_1p.shape[0], 12, 3), 80, dtype=np.uint8)
    body = np.hstack([big_1p, sep, big_2p])

    # ヘッダ: 識別子 + OCR 出力 (参考、ユーザは目視で真値を読む)
    header_h = 56
    header = np.zeros((header_h, body.shape[1], 3), dtype=np.uint8)
    title = (f"F{idx}: {info['video']} match{info['match']} "
             f"t={info['t_sec']:.1f}s")
    cv2.putText(header, title, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
    line2 = (f"OCR: 1P={info['ocr_1p']:08d} (c={info['conf_1p']:.2f})  "
             f"2P={info['ocr_2p']:08d} (c={info['conf_2p']:.2f})")
    cv2.putText(header, line2, (8, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    note = "Truth -> 目視 1P/2P 8 桁を記録"
    cv2.putText(header, note, (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 100), 1)
    return np.vstack([header, body])


def main() -> int:
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    selected = select_candidates(cache)
    print(f"採用: {len(selected)} サンプル")

    panels: list[np.ndarray] = []
    index_rows: list[dict] = []
    for fi, info in enumerate(selected):
        frame = get_frame(VIDEO_DIR / f"{info['video']}.mp4", info["t_sec"])
        if frame is None:
            print(f"  [skip] F{fi}")
            continue
        panel = build_panel(fi, info, frame)
        panels.append(panel)
        index_rows.append({
            "frame_idx": fi,
            "video": info["video"], "match": info["match"],
            "t_sec": round(info["t_sec"], 2),
            "ocr_1p": info["ocr_1p"], "ocr_2p": info["ocr_2p"],
            "conf_1p": round(info["conf_1p"], 3),
            "conf_2p": round(info["conf_2p"], 3),
        })
        print(f"  [ok] F{fi} {info['video']} m{info['match']} t={info['t_sec']:.1f}s "
              f"OCR={info['ocr_1p']}/{info['ocr_2p']}")

    if not panels:
        print("データなし")
        return 1

    sep = np.full((10, panels[0].shape[1], 3), 30, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for p in panels:
        parts.append(p); parts.append(sep)
    grid = np.vstack(parts[:-1])
    cv2.imwrite(str(OUT_GRID), grid)

    with open(OUT_INDEX, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, delimiter="\t",
            fieldnames=["frame_idx", "video", "match", "t_sec",
                        "ocr_1p", "ocr_2p", "conf_1p", "conf_2p"],
        )
        w.writeheader()
        w.writerows(index_rows)

    print(f"\n出力:")
    print(f"  grid: {OUT_GRID} (shape={grid.shape})")
    print(f"  index: {OUT_INDEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
