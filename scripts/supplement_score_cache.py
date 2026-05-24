"""score_series_cache の None 部分を周辺フレーム探索で補完する。

既存 cache の片側または両側 None だったサンプルだけ再 OCR し、
ScoreOcr.read_with_neighbor_search で ±0.3s 5 フレーム探索する。

入力:
    data/training/score_series_cache.json

出力:
    data/training/score_series_cache.json (in-place 上書き、バックアップ作成)
    data/training/supplement_log.txt
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2

from src.score_ocr import ScoreOcr

CACHE_PATH = Path("data/training/score_series_cache.json")
BACKUP_PATH = Path("data/training/score_series_cache.bak_pre_supplement.json")
LOG_PATH = Path("data/training/supplement_log.txt")
VIDEO_DIR = Path("data/frames")
SEARCH_RADIUS = 0.3
N_SAMPLES = 5


def main() -> int:
    if not CACHE_PATH.is_file():
        print(f"cache なし: {CACHE_PATH}")
        return 1
    if not BACKUP_PATH.is_file():
        shutil.copy(CACHE_PATH, BACKUP_PATH)
        print(f"backup: {BACKUP_PATH}")

    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    ocr = ScoreOcr.load_default()

    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    total_samples = 0
    none_before = 0
    recovered_1p = 0
    recovered_2p = 0
    recovered_both = 0

    for vid in cache:
        video_path = VIDEO_DIR / f"{vid}.mp4"
        if not video_path.is_file():
            log(f"[skip] {vid} 動画なし")
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log(f"[skip] {vid} 動画オープン失敗")
            continue
        log(f"\n=== {vid} ===")
        for midx, samples in cache[vid].items():
            n_supplemented = 0
            for s in samples:
                total_samples += 1
                # None でないなら触らない
                if s["1p"] is not None and s["2p"] is not None:
                    continue
                none_before += 1
                t = s["t"]
                # 周辺探索
                res = ocr.read_with_neighbor_search(
                    cap, t,
                    search_radius_sec=SEARCH_RADIUS,
                    n_samples=N_SAMPLES,
                )
                # 元 None だったサイドが回復したら更新
                if s["1p"] is None and res.score_1p is not None:
                    s["1p"] = int(res.score_1p)
                    s["c1"] = round(float(res.confidence_1p), 3)
                    recovered_1p += 1
                if s["2p"] is None and res.score_2p is not None:
                    s["2p"] = int(res.score_2p)
                    s["c2"] = round(float(res.confidence_2p), 3)
                    recovered_2p += 1
                if (s["1p"] is not None
                        and s["2p"] is not None):
                    # 補完で両側揃った
                    n_supplemented += 1
            recovered_both += n_supplemented
            if n_supplemented > 0:
                log(f"  match {midx}: +{n_supplemented} 両側回復")
        cap.release()

    # 上書き保存
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 統計
    after_both_ok = sum(
        1
        for vid in cache
        for midx in cache[vid]
        for s in cache[vid][midx]
        if s["1p"] is not None and s["2p"] is not None
    )
    log("")
    log(f"total samples: {total_samples}")
    log(f"None before (片側 or 両側 None): {none_before}")
    log(f"recovered 1p: {recovered_1p}")
    log(f"recovered 2p: {recovered_2p}")
    log(f"recovered both-ok matches: {recovered_both}")
    log(f"after: 両側 OK = {after_both_ok} "
        f"({after_both_ok/total_samples:.1%})")

    LOG_PATH.write_text("\n".join(log_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
