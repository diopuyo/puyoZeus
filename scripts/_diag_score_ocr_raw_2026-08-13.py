"""score OCR の生読み値・formula 検知フラグを t=188-210秒 で毎フレーム記録する
軽量診断スクリプト (使い捨て、コミット対象外)。

CNN 盤面認識を一切通さない (ScoreOcr のみ) ため高速。
chain_event.total_score=0 persist の根因が「OCR 自体が読めていない」
(真に formula 表示中で数値が存在しない) か「読めるのに握りつぶしている」
かを独立信号で切り分ける。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from src.score_ocr import (  # noqa: E402
    ScoreOcr, _crop_score_roi, _ensure_1080p, compute_score_roi_ink_ratio,
)

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
START_SEC = 188.0
END_SEC = 210.0
CHAIN_FORMULA_INK_RATIO_MIN = 0.35  # recognition_pipeline.py の既定値と同じにする (下で上書き確認)


def main() -> int:
    from src.recognition_pipeline import (
        CHAIN_FORMULA_INK_RATIO_MIN as REAL_INK_MIN,
    )
    ink_min = REAL_INK_MIN
    print(f"CHAIN_FORMULA_INK_RATIO_MIN(real)={ink_min}")

    ocr = ScoreOcr.load_default()
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(START_SEC * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    last_1p, last_2p = None, None
    n = int((END_SEC - START_SEC) * fps)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        t_sec = START_SEC + i / fps
        f = _ensure_1080p(frame)
        if f is None:
            continue
        s1, c1 = ocr.read_side(f, "1P")
        s2, c2 = ocr.read_side(f, "2P")
        roi1 = _crop_score_roi(f, "1P")
        roi2 = _crop_score_roi(f, "2P")
        ir1 = compute_score_roi_ink_ratio(roi1) if roi1 is not None else -1.0
        ir2 = compute_score_roi_ink_ratio(roi2) if roi2 is not None else -1.0
        formula1 = (s1 is None) and (ir1 > ink_min)
        formula2 = (s2 is None) and (ir2 > ink_min)
        changed = (s1 != last_1p) or (s2 != last_2p) or formula1 or formula2
        if changed or i % (int(fps) * 1) == 0:
            print(
                f"t={t_sec:7.2f} 1P: score={s1!s:>8} conf={c1:.2f} ink={ir1:.2f} "
                f"formula={formula1}  |  2P: score={s2!s:>8} conf={c2:.2f} "
                f"ink={ir2:.2f} formula={formula2}",
                flush=True,
            )
        last_1p, last_2p = s1, s2
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
