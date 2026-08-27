"""(診断) post_match_lockdown_latch が試合を飲み込む機構の計装 (2026-08-19)。

c132 (video_c132.mp4, _qJod5PzH3s) の指定窓を cv2.VideoCapture で全フレーム走査し、
ラッチの入力信号3系統を毎フレーム記録する:
  1. match_end NCC (MatchEndDetector.detect の生スコア、trigger 系)
  2. score_zero NCC 1P/2P (ScoreZeroDetector、解除系の第1条件)
  3. 盤面ROI std 1P/2P (board_motion、解除系の第2条件 = 実ゲームプレイ確認)

さらに本番実装 (recognition_pipeline.py:3896-3925) と同一ロジックでラッチを
オフライン再現し、ON/OFF遷移と解除失敗の原因 (score_zero不成立 / std不成立 /
45秒安全弁のみで解除) をフレーム単位で分類する。

出力: logs/_diag_lockdown_latch_c132_2026-08-19/ 配下
  - signals_<窓>.csv        毎フレーム信号
  - episodes_<窓>.txt       ラッチON区間と解除理由
  - evidence フレーム png   ラッチON中の実画面
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.board_motion import (  # noqa: E402
    REAL_GAMEPLAY_BOARD_STD_THRESHOLD,
    board_roi_gray,
    board_roi_std,
)
from src.match_end_detector import (  # noqa: E402
    DEFAULT_LOCKDOWN_SEC,
    DEFAULT_NCC_THRESHOLD,
    MatchEndDetector,
)
from src.score_zero import ScoreZeroDetector  # noqa: E402

VIDEO = PROJECT_ROOT / "data" / "frames" / "video_c132.mp4"
OUT_DIR = PROJECT_ROOT / "logs" / "_diag_lockdown_latch_c132_2026-08-19"

# 本番定数 (recognition_pipeline.py)
CHAIN_BAN_SEC_AFTER_MATCH_START = 30 / 60  # 0.5s
POST_MATCH_LOCKDOWN_MAX_SEC = 45.0

# 診断窓 (npz timeline から選定):
#   A: 314s 付近で game1 が正常終了 → 325-473s 最初のラッチ飲み込み
#   B: 785s 付近でラッチ解除成功 → 797-949 清浄 → 970 再ラッチ
WINDOWS = [
    ("A_first_swallow", 300.0, 520.0),
    ("B_release_works", 770.0, 1000.0),
]

# 証拠フレーム (npz上「試合中なのに locked=1」の代表時刻)
EVIDENCE_TIMES = [340.0, 371.0, 430.0, 500.0, 976.0, 850.0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"fps={fps:.3f} frames={n_frames} dur={n_frames / fps:.1f}s")

    sz = ScoreZeroDetector.load_default()

    for name, t0, t1 in WINDOWS:
        me = MatchEndDetector.load_default()  # 窓ごとに状態リセット
        rows: list[dict] = []
        f0, f1 = int(t0 * fps), int(t1 * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        # ラッチ状態 (本番 :3896-3925 と同一)
        latch = False
        prev_me_locked = False
        latch_start = -1.0
        raw_since = -1.0
        episodes: list[str] = []
        for fi in range(f0, f1):
            ok, frame = cap.read()
            if not ok:
                break
            t = fi / fps
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080))
            # 信号1: match_end
            me_res = me.detect(frame)
            if me_res.detected:
                me._last_detected_t = t  # update() 相当
            me_locked = me.is_locked(t)
            # 信号2: score_zero
            sz_res = sz.detect(frame)
            # 信号3: 盤面std
            std1 = board_roi_std(board_roi_gray(frame, "1P"))
            std2 = board_roi_std(board_roi_gray(frame, "2P"))
            gameplay = (
                std1 >= REAL_GAMEPLAY_BOARD_STD_THRESHOLD
                and std2 >= REAL_GAMEPLAY_BOARD_STD_THRESHOLD
            )
            # ラッチ再現
            release_reason = ""
            if me_locked and not prev_me_locked:
                if not latch:
                    episodes.append(f"ON  t={t:8.2f} (tmpl={me_res.template_name} ncc={me_res.score:.3f})")
                else:
                    episodes.append(f"RE-TRIGGER t={t:8.2f} (ラッチ継続中に再立ち上がり、valve延長)")
                latch = True
                latch_start = t
                raw_since = -1.0
            prev_me_locked = me_locked
            if latch:
                if sz_res.both_zero:
                    if raw_since < 0.0:
                        raw_since = t
                    persisted = t - raw_since >= CHAIN_BAN_SEC_AFTER_MATCH_START
                    if persisted and gameplay:
                        latch = False
                        release_reason = "normal"
                        episodes.append(f"OFF t={t:8.2f} 正常解除 (score_zero持続+std両側OK)")
                else:
                    raw_since = -1.0
                if latch and latch_start >= 0.0 and t - latch_start >= POST_MATCH_LOCKDOWN_MAX_SEC:
                    latch = False
                    release_reason = "valve45s"
                    episodes.append(f"OFF t={t:8.2f} 45秒安全弁 (解除条件は不成立のまま)")
            rows.append({
                "t": f"{t:.3f}",
                "me_ncc": f"{me_res.score:.4f}",
                "me_tmpl": me_res.template_name or "",
                "me_locked": int(me_locked),
                "sz1": f"{sz_res.score_1p:.4f}",
                "sz2": f"{sz_res.score_2p:.4f}",
                "sz_both": int(sz_res.both_zero),
                "std1": f"{std1:.2f}",
                "std2": f"{std2:.2f}",
                "gameplay": int(gameplay),
                "latch": int(latch),
                "release": release_reason,
            })
        out_csv = OUT_DIR / f"signals_{name}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        ep_txt = OUT_DIR / f"episodes_{name}.txt"
        ep_txt.write_text("\n".join(episodes) + "\n", encoding="utf-8")
        print(f"[{name}] rows={len(rows)} -> {out_csv}")
        for e in episodes:
            print("  ", e)
        # 窓内サマリ
        arr = np.array([[float(r["sz1"]), float(r["sz2"]), float(r["std1"]), float(r["std2"]), float(r["latch"])] for r in rows])
        print(f"  latch率={arr[:, 4].mean() * 100:.1f}%  sz_both率={np.mean([int(r['sz_both']) for r in rows]) * 100:.1f}%  gameplay率={np.mean([int(r['gameplay']) for r in rows]) * 100:.1f}%")

    # 証拠フレーム
    for t in EVIDENCE_TIMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if ok:
            p = OUT_DIR / f"evidence_t{int(t):04d}.png"
            cv2.imwrite(str(p), frame)
            print("saved", p)
    cap.release()


if __name__ == "__main__":
    main()
