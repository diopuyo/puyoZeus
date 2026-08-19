"""match_end NCC 閾値分布の限定スキャン (2026-08-19)。

全フレーム走査は 1 フレーム 200ms かかり非現実的なため、subset50 npz の
match_end_locked==1 行 (= MatchEndDetector が実際に発火した 5 秒ロック
ダウン区間) をエピソード化し、その周辺 [t0-8s, t0+8s] だけを実動画で
再スキャンして発火時の生 NCC 分布を得る。

本物/偽の正解づけ: エピソード終端から +90 秒以内に score OCR が両者
「新規の 0」を読めば本物の試合終了 (次試合開始の物理的証拠)、なければ
偽検出候補 (試合中の全消しテロップ誤検出等)。追跡スキャンは 2fps で軽量。

⚠️ 再DL動画は内容ドリフトの可能性があるため対象外 (元から手元の9本のみ)。

出力: logs/_scan_match_end_ncc_windows_2026-08-19.json
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.match_end_detector import MatchEndDetector  # noqa: E402
from src.score_ocr import ScoreOcr  # noqa: E402

NPZ_DIR = PROJECT_ROOT / "data" / "indicators_v2" / "boards_lean_subset50_2026-08-19"
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
OUT_JSON = PROJECT_ROOT / "logs" / "_scan_match_end_ncc_windows_2026-08-19.json"

EPISODE_GAP_SEC = 15.0
WINDOW_BEFORE_SEC = 8.0
WINDOW_AFTER_SEC = 8.0
SCAN_STRIDE_FRAMES = 3          # 窓内スキャンの間引き
ZERO_FOLLOW_SEC = 90.0          # 本物判定: 終端後この秒数内の両者0読取り
ZERO_SCAN_INTERVAL_SEC = 0.5    # 追跡スキャンの間隔 (2fps)

TARGETS = ["29", "31", "32", "33", "34", "35", "37", "c109", "c132"]


def _episodes_from_npz(stem: str) -> list[tuple[float, float]]:
    d = np.load(NPZ_DIR / f"{stem}.npz", allow_pickle=False)
    t = np.asarray(d["t_sec"], dtype=np.float64)
    locked = np.asarray(d["match_end_locked"]) == 1
    ts = np.unique(np.round(t[locked], 2))
    eps: list[list[float]] = []
    for x in ts:
        if eps and x - eps[-1][1] <= EPISODE_GAP_SEC:
            eps[-1][1] = x
        else:
            eps.append([x, x])
    return [(a, b) for a, b in eps]


def process_video(stem: str) -> dict:
    cv2.setNumThreads(1)
    eps = _episodes_from_npz(stem)
    video = FRAMES_DIR / f"video_{stem}.mp4"
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    me = MatchEndDetector.load_default()
    ocr = ScoreOcr.load_default()
    results = []
    for (t0, t1) in eps:
        # 1) 窓内の NCC 最大値
        f0 = max(0, int((t0 - WINDOW_BEFORE_SEC) * fps))
        f1 = int((t1 + WINDOW_AFTER_SEC) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        best = -1.0
        best_tmpl = ""
        best_t = -1.0
        fi = f0
        while fi < f1:
            ok, frame = cap.read()
            if not ok:
                break
            if (fi - f0) % SCAN_STRIDE_FRAMES == 0:
                if frame.shape[:2] != (1080, 1920):
                    frame = cv2.resize(frame, (1920, 1080))
                r = me.detect(frame)
                if r.score > best:
                    best = float(r.score)
                    best_tmpl = r.template_name or ""
                    best_t = fi / fps
            fi += 1
        # 2) 本物判定: 終端後 +90s の両者フレッシュ0読取り
        followed_zero = False
        tz = t1 + 1.0
        while tz <= t1 + ZERO_FOLLOW_SEC:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(tz * fps))
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080))
            v1, _ = ocr.read_side(frame, "1P")
            v2, _ = ocr.read_side(frame, "2P")
            if v1 == 0 and v2 == 0:
                followed_zero = True
                break
            tz += ZERO_SCAN_INTERVAL_SEC
        results.append({
            "t0": round(t0, 1), "t1": round(t1, 1),
            "max_ncc": round(best, 4), "template": best_tmpl,
            "ncc_t": round(best_t, 1),
            "followed_by_fresh_zero": followed_zero,
        })
        print(f"[{stem}] ep t={t0:.0f}-{t1:.0f} ncc={best:.3f} "
              f"tmpl={best_tmpl} real={followed_zero}", flush=True)
    cap.release()
    return {"video": stem, "episodes": results}


def main() -> None:
    with Pool(processes=3) as pool:
        out = pool.map(process_video, TARGETS)
    OUT_JSON.write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8",
    )
    print("saved", OUT_JSON)


if __name__ == "__main__":
    main()
