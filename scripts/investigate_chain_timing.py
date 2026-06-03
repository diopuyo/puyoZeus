"""
調査スクリプト 3: v70m2 1P 最初の連鎖 (t≒36.55s) で
ScoreTracker の delta ≥ 80 が最初に観測される frame を特定。

これが「機能B 式検知の場合の最速発火時刻」に対応する。
さらに、score が読めなくなる frame (式表示開始) との比較で
「式検知 vs score_delta 検知」のどちらが何 frame 早いかを実測する。

Usage:
    PYTHONPATH=. venv/bin/python3.12 scripts/investigate_chain_timing.py
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from src.score_ocr import ScoreOcr, SCORE_1P_REGION, SCORE_2P_REGION

VIDEO_PATH: str = "data/evaluation_videos_v2/v70m2_buf15s.mp4"
OUT_DIR: Path = Path("data/investigation/chain_timing")

# 調査区間
T_START: float = 35.0
T_END: float = 42.0
STEP_SEC: float = 1 / 60


def crop_roi(frame: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    y1, y2, x1, x2 = region
    return frame[y1:y2, x1:x2].copy()


def analyze_formula_signature(roi: np.ndarray) -> dict:
    """ROI に掛け算式が表示されているかの特徴量を計算。

    Returns:
        signature (dict):
            'bright_ratio': 輝度>200 の割合
            'green_ratio': 緑背景比率
            'center_std': 横方向輝度 std
            'formula_score': 式表示らしさスコア (0-1, 高いほど式らしい)
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright_ratio = float((gray > 200).mean())

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    green_mask = ((hue >= 40) & (hue <= 90) & (sat > 60) & (val > 60))
    green_ratio = float(green_mask.mean())

    col_means = gray.mean(axis=0)
    center_std = float(col_means.std())

    # 式らしさスコア: 通常スコア (bright_ratio≒0.44, green≒0.15, std≒29)
    #                vs 式表示     (bright_ratio≒0.34, green≒0.20, std≒23)
    # -> bright_ratio が低く green が高く std が低いほど式らしい
    formula_score = (
        (0.44 - bright_ratio) / 0.15  # bright が低いほど +
        + (green_ratio - 0.15) / 0.10  # green が高いほど +
        + (29.0 - center_std) / 8.0    # std が低いほど +
    ) / 3.0
    formula_score = max(0.0, min(1.0, formula_score))

    return {
        "bright_ratio": bright_ratio,
        "green_ratio": green_ratio,
        "center_std": center_std,
        "formula_score": formula_score,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ocr = ScoreOcr.load_default()

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"fps={fps:.2f}")

    last_score: int | None = None
    first_formula_t: float | None = None
    first_large_delta_t: float | None = None
    first_score_after_chain: int | None = None
    first_score_after_chain_t: float | None = None

    print("\n[1P score の frame ごと変化 (t=35-42s)]")
    print(f"{'t_sec':>8s}  {'fi':>5s}  {'score_1p':>10s}  {'conf':>6s}  {'delta':>7s}  {'readable':8s}  {'green_r':7s}  {'formula_score':13s}")

    in_formula = False
    formula_roi_saved = False

    t = T_START
    while t <= T_END:
        fi = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if (h, w) != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        res = ocr.read(frame)
        roi_1p = crop_roi(frame, SCORE_1P_REGION)
        feats = analyze_formula_signature(roi_1p)

        cur_score = res.score_1p
        readable = cur_score is not None
        delta = 0
        if cur_score is not None and last_score is not None:
            delta = cur_score - last_score

        # 式表示開始の検知
        if not readable and first_formula_t is None:
            first_formula_t = t

        # score が復帰したら記録
        if readable and first_formula_t is not None and first_score_after_chain is None:
            first_score_after_chain = cur_score
            first_score_after_chain_t = t

        # 大きな delta の検知 (score_delta ベース 機能B 検知相当)
        if delta >= 40 and first_large_delta_t is None:
            first_large_delta_t = t

        # 式らしさが高い (>0.5) 初回
        formula_flag = feats["formula_score"] > 0.5

        # 前後の状態変化のみ表示
        now_in_formula = not readable or formula_flag
        if now_in_formula != in_formula or (readable and delta != 0):
            marker = ""
            if not readable:
                marker = "*** FORMULA"
            elif delta >= 40:
                marker = "*** DELTA_FIRE"
            print(
                f"{t:8.3f}  {fi:5d}  {str(cur_score):>10s}  "
                f"{res.confidence_1p:6.3f}  {delta:+7d}  "
                f"{'YES' if readable else 'NO':8s}  "
                f"{feats['green_ratio']:7.4f}  "
                f"{feats['formula_score']:13.4f}  {marker}"
            )
        in_formula = now_in_formula

        # 式表示フレームの ROI を保存 (最初の 1 回)
        if not readable and not formula_roi_saved:
            tstr = f"{t:.3f}".replace(".", "p")
            cv2.imwrite(str(OUT_DIR / f"formula_roi1p_t{tstr}.png"), roi_1p)
            small = cv2.resize(frame, (960, 540))
            # ROI に枠
            y1, y2, x1, x2 = SCORE_1P_REGION
            cv2.rectangle(small, (int(x1*0.5), int(y1*0.5)), (int(x2*0.5), int(y2*0.5)), (0,0,255), 2)
            cv2.imwrite(str(OUT_DIR / f"formula_full_t{tstr}.png"), small)
            formula_roi_saved = True

        if cur_score is not None:
            last_score = cur_score
        t += STEP_SEC

    cap.release()

    # サマリ
    print("\n=== タイミング実測結果 ===")
    print(f"  式表示開始 (OCR unreadable):   t={first_formula_t}")
    print(f"  score_delta >= 40 初回:        t={first_large_delta_t}")
    print(f"  スコア復帰 (連鎖後確定):        t={first_score_after_chain_t}  score={first_score_after_chain}")
    if first_formula_t is not None and first_large_delta_t is not None:
        diff_frames = round((first_large_delta_t - first_formula_t) * fps)
        print(f"  式表示 vs score_delta 発火 lag: {diff_frames:+d} frames ({(first_large_delta_t - first_formula_t)*1000:+.0f}ms)")
        print("  (正 = score_delta が式表示より遅い = 式検知が早い)")
    print(f"\n[DONE] 保存先: {OUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
