"""
調査スクリプト 2: v70m2 試合中 (score=0 リセット後) の連鎖式フレームを詳細調査。

試合開始 (score=0, t≒9.25s) 後の最初の 1P unreadable 区間 (t≒36.55s) と
その前後の ROI + full frame を保存し、
- 式の形式 (掛け算記号の種類・位置・背景色)
- 通常スコアとの外観差
- 現状 chain_det が発火する frame との差 (タイミングラグ)
を実測する。

Usage:
    PYTHONPATH=. venv/bin/python3.12 scripts/investigate_chain_formula2.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.score_ocr import (
    ScoreOcr,
    SCORE_1P_REGION,
    SCORE_2P_REGION,
)

VIDEO_PATH: str = "data/evaluation_videos_v2/v70m2_buf15s.mp4"
OUT_DIR: Path = Path("data/investigation/chain_formula_frames2")

# 精細スキャン対象区間: 試合中 1P 最初の連鎖 前後 (秒)
# t=9.25s (score=0) 後の最初の連鎖区間は t≒36.55s が判明済
FINE_SCAN_REGIONS: list[tuple[float, float]] = [
    (34.0, 42.0),   # 1P 最初の連鎖前後
    (38.5, 43.5),   # 2P 最初の連鎖前後
]
FINE_STEP_SEC: float = 1 / 60  # 60fps 1frame ずつ


def crop_roi(frame: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    y1, y2, x1, x2 = region
    return frame[y1:y2, x1:x2].copy()


def save_png(img: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), img)
    print(f"  saved {path.name}")


def add_annotations(
    frame: np.ndarray,
    t_sec: float,
    score_1p: int | None,
    conf_1p: float,
) -> np.ndarray:
    """デバッグ用テキスト注釈を追加した縮小フレームを返す。"""
    small = cv2.resize(frame, (960, 540))
    # 1P ROI を赤枠
    y1, y2, x1, x2 = SCORE_1P_REGION
    sx, sy = 0.5, 0.5
    cv2.rectangle(
        small,
        (int(x1 * sx), int(y1 * sy)),
        (int(x2 * sx), int(y2 * sy)),
        (0, 0, 255), 2,
    )
    # 2P ROI を青枠
    y1b, y2b, x1b, x2b = SCORE_2P_REGION
    cv2.rectangle(
        small,
        (int(x1b * sx), int(y1b * sy)),
        (int(x2b * sx), int(y2b * sy)),
        (255, 0, 0), 2,
    )
    score_str = str(score_1p) if score_1p is not None else "FORMULA"
    label = f"t={t_sec:.3f}s 1P={score_str} conf={conf_1p:.3f}"
    cv2.putText(
        small, label,
        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
    )
    return small


def analyze_roi_for_formula_features(roi: np.ndarray) -> dict:
    """ROI から掛け算式検出に使える特徴量を計算する。

    Returns: {
        'mean_brightness': float,   # 平均輝度
        'bright_ratio': float,      # 輝度 > 200 の割合 (白文字率)
        'blue_ratio': float,        # 青系ピクセル比 (通常スコア背景色)
        'green_ratio': float,       # 緑系ピクセル比 (連鎖背景)
        'non_digit_ink': float,     # 数字 template 非マッチ領域の ink 比率
        'center_x_bright_std': float,  # 横方向輝度分布の標準偏差
    }
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    mean_brightness = float(gray.mean())
    bright_ratio = float((gray > 200).mean())

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # 青系: H 100-140 S>50 V>50
    blue_mask = ((hue >= 100) & (hue <= 140) & (sat > 50) & (val > 50))
    blue_ratio = float(blue_mask.mean())

    # 緑系: H 40-80 S>50 V>50
    green_mask = ((hue >= 40) & (hue <= 80) & (sat > 50) & (val > 50))
    green_ratio = float(green_mask.mean())

    # 横方向輝度の列ごと平均で std を計算
    col_means = gray.mean(axis=0)
    center_x_bright_std = float(col_means.std())

    return {
        "mean_brightness": mean_brightness,
        "bright_ratio": bright_ratio,
        "blue_ratio": blue_ratio,
        "green_ratio": green_ratio,
        "center_x_bright_std": center_x_bright_std,
    }


def scan_fine(
    cap: cv2.VideoCapture,
    ocr: ScoreOcr,
    fps: float,
    t_start: float,
    t_end: float,
    step_sec: float,
) -> list[dict]:
    """精細スキャン: t_start..t_end を step_sec 刻みでフレームごとに解析。"""
    results = []
    t = t_start
    while t <= t_end:
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
        feats = analyze_roi_for_formula_features(roi_1p)
        results.append({
            "t_sec": t,
            "frame_idx": fi,
            "score_1p": res.score_1p,
            "conf_1p": res.confidence_1p,
            "score_2p": res.score_2p,
            "conf_2p": res.confidence_2p,
            "readable_1p": res.score_1p is not None,
            "frame": frame,
            "roi_1p": roi_1p,
            **feats,
        })
        t += step_sec
    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ocr = ScoreOcr.load_default()

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] fps={fps:.2f}")

    # 区間 1: 1P 最初の連鎖 (t=34-42s)
    print("\n=== 区間 1: 1P 最初の連鎖 (t=34-42s) ===")
    region_results = scan_fine(cap, ocr, fps, 34.0, 42.0, FINE_STEP_SEC)

    # score が最後に正常だった frame を特定
    last_readable_t = None
    first_formula_t = None
    first_recovery_t = None
    prev_score: int | None = None
    score_before_chain: int | None = None
    score_after_chain: int | None = None

    for r in region_results:
        if r["readable_1p"]:
            if first_formula_t is None:
                last_readable_t = r["t_sec"]
                score_before_chain = r["score_1p"]
            else:
                if first_recovery_t is None:
                    first_recovery_t = r["t_sec"]
                    score_after_chain = r["score_1p"]
        else:
            if first_formula_t is None:
                first_formula_t = r["t_sec"]

    print(f"  通常スコア最終 readable: t={last_readable_t}")
    print(f"  式表示開始:              t={first_formula_t}")
    print(f"  スコア復帰:              t={first_recovery_t}")
    print(f"  連鎖前スコア: {score_before_chain}")
    print(f"  連鎖後スコア: {score_after_chain}")
    if score_before_chain is not None and score_after_chain is not None:
        delta = score_after_chain - score_before_chain
        print(f"  スコア増分 (delta): {delta}")
    if last_readable_t is not None and first_formula_t is not None:
        lag_ms = (first_formula_t - last_readable_t) * 1000
        print(f"  readable→式表示 lag: {lag_ms:.0f}ms")

    # 状態遷移の詳細を表示
    print("\n  [frame ごとの状態]")
    prev_readable = None
    for r in region_results:
        is_readable = r["readable_1p"]
        if is_readable != prev_readable:
            marker = "SCORE" if is_readable else "FORMULA"
            print(
                f"    t={r['t_sec']:.3f}s fi={r['frame_idx']:5d}  {marker:7s}  "
                f"1P={str(r['score_1p']):>10} conf={r['conf_1p']:.3f}  "
                f"bright={r['bright_ratio']:.3f} green={r['green_ratio']:.3f}"
            )
        prev_readable = is_readable

    # 特徴量の統計 (読める区間 vs 読めない区間)
    readable_feats = [r for r in region_results if r["readable_1p"]]
    formula_feats = [r for r in region_results if not r["readable_1p"]]
    if readable_feats and formula_feats:
        print("\n  [特徴量 readable vs formula]")
        for key in ["mean_brightness", "bright_ratio", "blue_ratio",
                    "green_ratio", "center_x_bright_std"]:
            r_mean = sum(r[key] for r in readable_feats) / len(readable_feats)
            f_mean = sum(r[key] for r in formula_feats) / len(formula_feats)
            print(f"    {key:28s}: readable={r_mean:.4f}  formula={f_mean:.4f}"
                  f"  diff={f_mean - r_mean:+.4f}")

    # キーフレームの PNG 保存
    print("\n  [PNG 保存]")
    # 式開始直前 (通常スコア最終)
    target_frames = []
    for r in region_results:
        if r["t_sec"] >= 36.3 and r["t_sec"] <= 36.6:
            target_frames.append(r)
    for r in target_frames[:3]:
        annotated = add_annotations(r["frame"], r["t_sec"], r["score_1p"], r["conf_1p"])
        tstr = f"{r['t_sec']:.3f}".replace(".", "p")
        save_png(annotated, OUT_DIR / f"before_t{tstr}_full.png")
        save_png(r["roi_1p"], OUT_DIR / f"before_roi1p_t{tstr}.png")

    # 式表示区間 (t=36.5-37.5 の代表 3 frame)
    formula_frames = [r for r in region_results if 36.5 <= r["t_sec"] <= 37.6]
    step = max(1, len(formula_frames) // 3)
    for i, r in enumerate(formula_frames[::step][:3]):
        annotated = add_annotations(r["frame"], r["t_sec"], r["score_1p"], r["conf_1p"])
        tstr = f"{r['t_sec']:.3f}".replace(".", "p")
        save_png(annotated, OUT_DIR / f"formula_t{tstr}_full.png")
        save_png(r["roi_1p"], OUT_DIR / f"formula_roi1p_t{tstr}.png")

    cap.release()
    print(f"\n[DONE] 保存先: {OUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
