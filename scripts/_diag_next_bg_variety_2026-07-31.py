"""ネクスト表示の背景がキャラ依存でどれだけ変わるかを実測する (2026-07-31)。

## 背景 (user 伝授)

**背景はプレイヤーが選ぶキャラごとに違う。** サイド (1P/2P) では決まらない。

ところが `src/next_detector.py` は
    BG_HUE_RANGES_1P = ((92, 110),)          # 水色
    BG_HUE_RANGES_2P = ((0, 12), (165, 179)) # ピンク/赤系
    HSV_COLOR_RULES_2P (2P だけ彩度閾値を引き上げ)
と **サイド固定でハードコード**している。キャラ依存なら、この決め打ちは
特定のキャラでしか成立しない。

盤面側は `BackgroundFingerprint.capture()` で試合開始時に実フレームから
背景を採取しており **キャラ依存に自動適応**している。両者で設計が割れている。

## なぜ今測るのか

2026-07-31 に実装した「ネクスト裏付け確定」
(`enable_next_corroborated_confirm`) は NextDetector の信頼性を継承する。
**特定の背景で NextDetector が誤読するなら、誤った色をより速く確定させる**
= 最も危険な向きの副作用になる。
「NextDetector 精度 100%」の記録は v89 の 1 動画 48 セルのみで、
**1 つの背景でしか確かめていない。**

## 何を測るか

各動画のネクスト ROI から**ぷよが写っていない背景画素**を採取し、
その色分布が動画間でどれだけ違うかを出す。具体的には:
  1. ROI 内で「中心の内側 crop を除いた外周」= 背景が写る領域をサンプル
  2. H の最頻値・S/V の中央値を動画ごとに求める
  3. ハードコード範囲 (BG_HUE_RANGES_1P/2P) に**入るか外れるか**を判定
  4. 動画をまとめてクラスタし、背景が何種類あるかを見る

認識は一切走らせない (ROI 切り出しと HSV 集計のみ) ので高速。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_next_bg_variety_2026-07-31 \
        --limit 30
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.next_detector import (
    BG_HUE_RANGES_1P,
    BG_HUE_RANGES_2P,
    INNER_CROP_RATIO,
    ROI_1P_NEXT_TOP,
    ROI_2P_NEXT_TOP,
)

TARGET_W, TARGET_H = 1920, 1080
# 各動画から採取するフレーム数 (試合中を狙って等間隔に取る)
SAMPLES_PER_VIDEO: int = 12
# サンプル開始/終了位置 (動画全体に対する割合)。冒頭の演出を避ける。
SAMPLE_FROM_RATIO: float = 0.30
SAMPLE_TO_RATIO: float = 0.80
# 背景とみなす彩度の上限 (ぷよは高彩度。背景は相対的に低彩度) …ではなく、
# ぷよ本体を確実に除くため「内側 crop の外側リング」を空間的に使う。
# ここでは追加のフィルタはかけない (背景が高彩度のキャラもいるため)。


def _bg_ring(patch: np.ndarray) -> np.ndarray:
    """ROI パッチから中心の内側 crop を除いた外周リングを返す。

    ぷよは中心に描画されるので、外周は背景が写る。
    """
    h, w = patch.shape[:2]
    cy, cx = h // 2, w // 2
    rh = max(1, int(h * INNER_CROP_RATIO / 2))
    rw = max(1, int(w * INNER_CROP_RATIO / 2))
    mask = np.ones((h, w), dtype=bool)
    mask[max(0, cy - rh):min(h, cy + rh), max(0, cx - rw):min(w, cx + rw)] = False
    return patch[mask]


def _in_ranges(hue: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    """H がハードコード範囲のいずれかに入るか。"""
    return any(lo <= hue <= hi for lo, hi in ranges)


def _sample_video(path: Path) -> dict[str, tuple[int, int, int]] | None:
    """1 動画から 1P/2P のネクスト背景の代表 HSV を返す。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return None
    idxs = np.linspace(
        total * SAMPLE_FROM_RATIO, total * SAMPLE_TO_RATIO, SAMPLES_PER_VIDEO,
    ).astype(int)
    acc: dict[str, list[np.ndarray]] = {"1P": [], "2P": []}
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(
                frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA,
            )
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for side, roi in (("1P", ROI_1P_NEXT_TOP), ("2P", ROI_2P_NEXT_TOP)):
            y1, y2, x1, x2 = roi
            ring = _bg_ring(hsv[y1:y2, x1:x2])
            if ring.size:
                acc[side].append(ring)
    cap.release()
    out: dict[str, tuple[int, int, int]] = {}
    for side, chunks in acc.items():
        if not chunks:
            continue
        allpx = np.concatenate(chunks, axis=0)
        h_mode = int(Counter(allpx[:, 0].tolist()).most_common(1)[0][0])
        out[side] = (
            h_mode,
            int(np.median(allpx[:, 1])),
            int(np.median(allpx[:, 2])),
        )
    return out or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, default=Path("data/frames"))
    ap.add_argument("--pattern", type=str, default="video_c*.mp4")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    cv2.setNumThreads(1)
    files = sorted(args.video_dir.glob(args.pattern))[: args.limit]
    if not files:
        print(f"動画が無い: {args.video_dir}/{args.pattern}")
        return
    print(f"対象 {len(files)} 動画 / 各 {SAMPLES_PER_VIDEO} フレーム")
    print(f"ハードコード想定: 1P={BG_HUE_RANGES_1P} 2P={BG_HUE_RANGES_2P}\n")
    print(f"{'動画':<14}{'1P (H,S,V)':>18}{'想定内':>8}{'2P (H,S,V)':>18}{'想定内':>8}")
    print("-" * 68)

    hues: dict[str, list[int]] = {"1P": [], "2P": []}
    n_out: dict[str, int] = {"1P": 0, "2P": 0}
    n_have: dict[str, int] = {"1P": 0, "2P": 0}
    for f in files:
        res = _sample_video(f)
        if res is None:
            print(f"{f.stem:<14}  (読めない)")
            continue
        cells = []
        for side, ranges in (("1P", BG_HUE_RANGES_1P), ("2P", BG_HUE_RANGES_2P)):
            v = res.get(side)
            if v is None:
                cells.append(f"{'-':>18}{'-':>8}")
                continue
            inside = _in_ranges(v[0], ranges)
            hues[side].append(v[0])
            n_have[side] += 1
            if not inside:
                n_out[side] += 1
            cells.append(f"{str(v):>18}{('OK' if inside else '★外'):>8}")
        print(f"{f.stem:<14}{''.join(cells)}")

    print("\n=== 集計 ===")
    for side in ("1P", "2P"):
        if not n_have[side]:
            continue
        arr = np.asarray(hues[side])
        uniq = sorted(set(arr.tolist()))
        print(
            f"{side}: ハードコード範囲外 {n_out[side]}/{n_have[side]} "
            f"({100.0 * n_out[side] / n_have[side]:.0f}%)  "
            f"H の実測値 {len(uniq)} 種 (最小{int(arr.min())}/最大{int(arr.max())})"
        )
    print(
        "\n→ 範囲外が多ければ「サイド固定のハードコード」は成立していない。"
        "\n→ H の種類が多ければ背景はキャラ依存 = 背景別サンプルが必要。"
    )


if __name__ == "__main__":
    main()
