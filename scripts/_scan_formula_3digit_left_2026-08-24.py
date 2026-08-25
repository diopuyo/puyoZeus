"""掛け算式 左辺3桁ケースの全編スキャン (2026-08-24 コーダ)。

user 訂正 (2026-08-24):「左辺は 10 個同時消しで 3 桁 (100) になる。
実データで 3 桁のケースを探して実画面と突合せよ」への対応。

video_zenchi_c0BQoMJwwQU.mp4 全編を 0.2 秒間引きで走査し、
ScoreOcr.read_formula_side が valid かつ left >= 100 のフレームを記録、
証拠画像 (score 帯) も保存する。

出力: logs/_scan_formula_3digit_left_2026-08-24/
    hits.tsv      (t, side, left, right, product, mult_ncc)
    all_valid.tsv (全 valid 読取り、後段の分布確認用)
    ev_*.png      (左辺3桁ヒットの score 帯クロップ、最大 40 枚)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

cv2.setNumThreads(2)

from src.score_ocr import ScoreOcr  # noqa: E402

VIDEO = Path("data/frames/video_zenchi_c0BQoMJwwQU.mp4")
OUT = Path("logs/_scan_formula_3digit_left_2026-08-24")
OUT.mkdir(parents=True, exist_ok=True)

STRIDE_SEC = 0.2
MAX_EVIDENCE = 40


def main() -> None:
    ocr = ScoreOcr.load_default()
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(STRIDE_SEC * fps)))
    n_ev = 0
    fi = -1
    hits_f = (OUT / "hits.tsv").open("w", encoding="utf-8")
    valid_f = (OUT / "all_valid.tsv").open("w", encoding="utf-8")
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            fi += 1
            if fi % stride != 0:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            t = fi / fps
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080),
                                   interpolation=cv2.INTER_AREA)
            for side in ("1P", "2P"):
                r = ocr.read_formula_side(frame, side)
                if not r.valid:
                    continue
                valid_f.write(
                    f"{t:.2f}\t{side}\t{r.left}\t{r.right}\t{r.product}"
                    f"\t{r.mult_ncc:.3f}\n"
                )
                if r.left is not None and r.left >= 100:
                    hits_f.write(
                        f"{t:.2f}\t{side}\t{r.left}\t{r.right}\t{r.product}"
                        f"\t{r.mult_ncc:.3f}\n"
                    )
                    hits_f.flush()
                    if n_ev < MAX_EVIDENCE:
                        x0, x1 = (100, 860) if side == "1P" else (1060, 1820)
                        crop = frame[860:980, x0:x1]
                        cv2.imwrite(
                            str(OUT / f"ev_{side}_t{t:.1f}.png"), crop,
                        )
                        n_ev += 1
            if fi % (stride * 500) == 0:
                print(f"[scan] t={t:.0f}s", flush=True)
    finally:
        hits_f.close()
        valid_f.close()
        cap.release()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
