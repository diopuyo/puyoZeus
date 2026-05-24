"""score 数値テンプレ (digit_0.png .. digit_9.png) を video_02 から自動生成する。

事前に目視で確認した (時刻, サイド, 8 桁 score 文字列) のラベルテーブルを使い、
各時刻で score ROI を切出 → 各桁画像を保存。各クラスは複数候補から平均化して
最終テンプレを書き出す。

実行:
    PYTHONPATH=. python scripts/build_score_digit_templates.py

出力:
    models/ui_templates/score_digits/digit_0.png ~ digit_9.png
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.score_ocr import (
    DIGIT_COUNT,
    DIGIT_HEIGHT,
    DIGIT_LEFTS_1P,
    DIGIT_LEFTS_2P,
    DIGIT_TOP,
    DIGIT_WIDTH,
    SCORE_1P_REGION,
    SCORE_2P_REGION,
)

VIDEO_PATH: Path = Path("data/frames/video_02.mp4")
OUTPUT_DIR: Path = Path("models/ui_templates/score_digits")
DEBUG_DIR: Path = Path("data/verify/score_dbg/templates")

# (時刻 [秒], サイド, 8 桁 score 文字列) ラベル — 目視で確認済み
# できるだけ多様な数字 (特に 8,9) を含むフレームから採取して精度向上を図る
LABELS: list[tuple[float, str, str]] = [
    # m01 (205-261s)
    (210.0, "1P", "00000037"),
    (215.0, "1P", "00000104"),
    (220.0, "1P", "00000158"),
    (225.0, "1P", "00001724"),
    (230.0, "1P", "00001775"),
    (235.0, "1P", "00003073"),
    (240.0, "1P", "00003120"),
    (250.0, "1P", "00006285"),
    (255.0, "1P", "00006294"),
    # m02 (266-306s) - 7,8,9 大量
    (290.0, "1P", "00007970"),
    (295.0, "1P", "00007998"),
    (300.0, "1P", "00008000"),
    (310.0, "1P", "00000035"),
    (315.0, "1P", "00000102"),
    (320.0, "1P", "00000158"),
    # m05/06 - 4,6,5,2 強化
    (455.0, "1P", "00004610"),
    (460.0, "1P", "00004652"),
    (475.0, "1P", "00000048"),
    (480.0, "1P", "00000113"),
    (485.0, "1P", "00000169"),
    (490.0, "1P", "00000222"),
    # m10 - 1,0,4,7,5 強化
    (665.0, "1P", "00010450"),
    (675.0, "1P", "00010471"),
    # 2P 側 - フォントは同じ。サンプル数増やして平均テンプレを安定化
    (210.0, "2P", "00000037"),
    (220.0, "2P", "00000150"),
    (235.0, "2P", "00001404"),
    (250.0, "2P", "00002352"),
    (290.0, "2P", "00005780"),
    (300.0, "2P", "00011786"),
    (460.0, "2P", "00011267"),
    (480.0, "2P", "00000117"),
    (670.0, "2P", "00016226"),
]


def _read_frame_at(cap: cv2.VideoCapture, t_sec: float) -> np.ndarray | None:
    """1920x1080 BGR で読み取る。"""
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame


def _crop_roi(frame: np.ndarray, side: str) -> np.ndarray:
    region = SCORE_1P_REGION if side == "1P" else SCORE_2P_REGION
    y1, y2, x1, x2 = region
    return frame[y1:y2, x1:x2].copy()


def _crop_cell(roi: np.ndarray, idx: int, side: str) -> np.ndarray:
    lefts = DIGIT_LEFTS_1P if side == "1P" else DIGIT_LEFTS_2P
    x = lefts[idx]
    return roi[DIGIT_TOP:DIGIT_TOP + DIGIT_HEIGHT, x:x + DIGIT_WIDTH].copy()


def _collect_samples() -> dict[int, list[np.ndarray]]:
    """ラベル付きサンプルを集める。"""
    samples: dict[int, list[np.ndarray]] = defaultdict(list)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けない: {VIDEO_PATH}")
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for t_sec, side, score_str in LABELS:
            assert len(score_str) == DIGIT_COUNT, score_str
            frame = _read_frame_at(cap, t_sec)
            if frame is None:
                print(f"[skip] フレーム取得失敗 t={t_sec}")
                continue
            roi = _crop_roi(frame, side)
            for i, ch in enumerate(score_str):
                label = int(ch)
                cell = _crop_cell(roi, i, side)
                samples[label].append(cell)
                # デバッグ用に保存
                dbg_name = f"t{int(t_sec):04d}_{side}_p{i}_label{label}.png"
                cv2.imwrite(str(DEBUG_DIR / dbg_name), cell)
    finally:
        cap.release()
    return samples


def _build_template(crops: list[np.ndarray]) -> np.ndarray:
    """同じラベルの複数 crop を平均してテンプレ化。"""
    arr = np.stack([c.astype(np.float32) for c in crops], axis=0)
    avg = arr.mean(axis=0)
    return np.clip(avg, 0, 255).astype(np.uint8)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = _collect_samples()
    print(f"収集サンプル数: {dict((k, len(v)) for k, v in samples.items())}")

    for label in range(10):
        crops = samples.get(label, [])
        if not crops:
            print(f"[warn] label={label} のサンプル無し → テンプレ未生成")
            continue
        tpl = _build_template(crops)
        out_path = OUTPUT_DIR / f"digit_{label}.png"
        cv2.imwrite(str(out_path), tpl)
        print(f"  digit_{label}.png  ({len(crops)} samples avg)")

    print("\n完了。生成ファイル:")
    for p in sorted(OUTPUT_DIR.glob("digit_*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
