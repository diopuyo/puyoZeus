"""1P/2P 盤面領域のアライメント (セル矩形がぷよ中心を捉えているか) を実測する。

## 背景 (2026-07-31)

実測で **2P の反映が systematically 遅い**ことが分かっている
(区間A: 1P 2.0 / 2P 3.0、区間B: 1P 2.0 / 2P 8.0)。
どのパラメータ (stable_frame_count 6/4/3、recovery_min_frames 8/6/4) を
振っても 2P は動かなかったので、構造側に原因がある。

領域定義を検算すると **2P は完全な鏡像から 4px ずれている**:
    P1 中心x = 474.0 / P2 中心x = 1450.0
    960 を軸にした鏡像なら P2 中心は 1446.0 → **+4px**
セル幅 64px に対しサンプル矩形は 38px 幅なので、4px は半幅の約 21%。

ただし「鏡像からずれている」だけでは誤りと言えない (独立に較正した結果かも
しれない)。**実際にぷよの中心を捉えているか**を測って判定する。

## 測り方

各セルについて、セル矩形いっぱい (サンプル矩形ではなく全体) を切り出し、
**高彩度画素 (= ぷよ本体) の重心**がセル中心からどれだけずれているかを測る。
- ぷよが写っていないセルは除外 (高彩度画素が少なすぎるもの)
- 側ごとに dx, dy の中央値を出す

1P がほぼ 0 で 2P に系統的なずれがあれば、領域定義のずれが確定する。
認識は走らせない (切り出しと HSV 集計のみ)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_board_region_alignment_2026-07-31 \
        --videos video_c34 video_c60 video_c56
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.board import BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION

TARGET_W, TARGET_H = 1920, 1080
# 各動画から採取するフレーム数 (試合中を狙って等間隔)
SAMPLES_PER_VIDEO: int = 24
SAMPLE_FROM_RATIO: float = 0.35
SAMPLE_TO_RATIO: float = 0.75
# ぷよ本体とみなす彩度の下限 (背景/枠は相対的に低彩度)
PUYO_SAT_MIN: int = 110
# ぷよが写っていると判定する最小画素比率 (これ未満のセルは空とみなし除外)
MIN_PUYO_PIXEL_RATIO: float = 0.25


def _cell_rect(region, row: int, col: int) -> tuple[int, int, int, int]:
    """セル矩形いっぱい (サンプル矩形ではない) を返す。"""
    cw = region.cell_width
    ch = region.cell_height
    visible_row = row - HIDDEN_ROWS
    x1 = int(region.x + col * cw)
    y1 = int(region.y + visible_row * ch)
    return x1, y1, int(x1 + cw), int(y1 + ch)


def _centroid_offset(hsv_cell: np.ndarray) -> tuple[float, float] | None:
    """セル内の高彩度画素の重心が中心からどれだけずれているかを返す。

    Returns:
        (dx, dy) [px]。ぷよが写っていなければ None。
    """
    if hsv_cell.size == 0:
        return None
    sat = hsv_cell[:, :, 1]
    mask = sat >= PUYO_SAT_MIN
    n = int(mask.sum())
    h, w = mask.shape
    if n < int(h * w * MIN_PUYO_PIXEL_RATIO):
        return None
    ys, xs = np.nonzero(mask)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    return float(xs.mean() - cx), float(ys.mean() - cy)


def _scan(video: Path) -> dict[str, list[tuple[float, float]]]:
    """1 動画を走査して側ごとの (dx, dy) を集める。"""
    out: dict[str, list[tuple[float, float]]] = {"1P": [], "2P": []}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return out
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return out
    idxs = np.linspace(
        total * SAMPLE_FROM_RATIO, total * SAMPLE_TO_RATIO, SAMPLES_PER_VIDEO,
    ).astype(int)
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
        for side, region in (("1P", DEFAULT_P1_REGION), ("2P", DEFAULT_P2_REGION)):
            for row in range(HIDDEN_ROWS, BOARD_ROWS):
                for col in range(BOARD_COLS):
                    x1, y1, x2, y2 = _cell_rect(region, row, col)
                    if x1 < 0 or y1 < 0 or x2 > TARGET_W or y2 > TARGET_H:
                        continue
                    off = _centroid_offset(hsv[y1:y2, x1:x2])
                    if off is not None:
                        out[side].append(off)
    cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--videos", nargs="+", default=["video_c34", "video_c60", "video_c56"],
    )
    ap.add_argument("--video-dir", type=Path, default=Path("data/frames"))
    args = ap.parse_args()

    cv2.setNumThreads(1)
    print(f"P1: x={DEFAULT_P1_REGION.x} 幅={DEFAULT_P1_REGION.width} "
          f"中心x={DEFAULT_P1_REGION.x + DEFAULT_P1_REGION.width / 2}")
    print(f"P2: x={DEFAULT_P2_REGION.x} 幅={DEFAULT_P2_REGION.width} "
          f"中心x={DEFAULT_P2_REGION.x + DEFAULT_P2_REGION.width / 2}")
    print(f"セル幅={DEFAULT_P1_REGION.cell_width:.1f}px  "
          f"高彩度しきい={PUYO_SAT_MIN}\n")
    print(f"{'動画':<14}{'側':>4}{'n':>7}{'dx中央':>9}{'dy中央':>9}{'dx平均':>9}")
    print("-" * 52)

    agg: dict[str, list[tuple[float, float]]] = {"1P": [], "2P": []}
    for name in args.videos:
        path = args.video_dir / f"{name}.mp4"
        if not path.exists():
            print(f"{name:<14}  (動画不在)")
            continue
        res = _scan(path)
        for side in ("1P", "2P"):
            pts = res[side]
            agg[side].extend(pts)
            if not pts:
                print(f"{name:<14}{side:>4}{0:>7}")
                continue
            arr = np.asarray(pts)
            print(
                f"{name:<14}{side:>4}{len(pts):>7}"
                f"{float(np.median(arr[:, 0])):>9.2f}"
                f"{float(np.median(arr[:, 1])):>9.2f}"
                f"{float(arr[:, 0].mean()):>9.2f}"
            )

    print("\n=== 全動画まとめ ===")
    for side in ("1P", "2P"):
        pts = agg[side]
        if not pts:
            continue
        arr = np.asarray(pts)
        print(
            f"{side}: n={len(pts)}  "
            f"dx 中央 {float(np.median(arr[:, 0])):+.2f}px / "
            f"dy 中央 {float(np.median(arr[:, 1])):+.2f}px"
        )
    if agg["1P"] and agg["2P"]:
        a1 = np.asarray(agg["1P"])
        a2 = np.asarray(agg["2P"])
        d = float(np.median(a2[:, 0])) - float(np.median(a1[:, 0]))
        print(
            f"\n→ 2P と 1P の dx 差: {d:+.2f}px "
            f"(領域定義の鏡像からのずれは +4px)"
        )
        print(
            "→ 1P がほぼ 0 で 2P に系統的なずれがあれば領域定義の誤りが確定。"
            "\n→ 両側とも同程度なら領域は正しく、2P の遅さは別の原因。"
        )


if __name__ == "__main__":
    main()
