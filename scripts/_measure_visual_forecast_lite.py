"""STEP 2 軽量版: 視覚予告検出器の精度実測 (RecognitionPipeline不使用).

video_124_4min.mp4 の代表区間で:
    - 各フレームで OjamaWarningDetector.detect() を実行して visual_count を取得
    - glow_guard (compute_glow_score) で発光フレームを除外
    - STABLEフレームの代わりに「盤面変化が小さいフレーム」を代理指標として使用
    - 視覚検出器が正常に動作しているか、どんな値を返すかを確認

2026-07-03 Phase 1b 精度実測 (軽量版)
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")

from src.ojama_warning import (
    COUNT_TABLE,
    OjamaWarningDetector,
    P1_BOARD_X,
    P2_BOARD_X,
    WARNING_BOTTOM_Y,
    WARNING_TOP_Y,
    BOARD_WIDTH,
)

# ============================
# 設定定数
# ============================
VIDEO_PATH: str = "data/frames/video_124_4min.mp4"
START_SEC: float = 0.0
END_SEC: float = 120.0
SAMPLE_INTERVAL: int = 3        # 3フレームごとに1フレーム処理 (10fps相当)
LARGE_DIFF_THRESHOLD: int = 30  # 大乖離閾値 (岩1個分)

# 盤面安定判定: フレーム間差分がこの値未満ならSTABLE代理
BOARD_CHANGE_STABLE_MAX: float = 8.0

# glow判定: 上部ROI (y=105~180, board_x~board_x+BOARD_WIDTH) の輝度閾値
V_HIGH_THRESHOLD: int = 220
GLOW_RATIO_THRESHOLD: float = 0.20


def _compute_glow(frame: np.ndarray, board_x: int) -> float:
    """簡易glow判定: 上部ROIの高輝度ピクセル比率を返す."""
    roi = frame[WARNING_TOP_Y:WARNING_BOTTOM_Y, board_x:board_x + BOARD_WIDTH]
    if roi.size == 0:
        return 0.0
    v = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2].astype(float)
    return float(np.mean(v >= V_HIGH_THRESHOLD))


def main() -> None:
    print("=== STEP 2 軽量版: 視覚予告検出器精度実測 ===")
    print(f"動画: {VIDEO_PATH}")
    print(f"区間: {START_SEC:.0f}s ~ {END_SEC:.0f}s, サンプリング: {SAMPLE_INTERVAL}フレームごと")
    print()

    det = OjamaWarningDetector(use_cnn=False)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    start_frame = int(START_SEC * fps)
    end_frame = int(END_SEC * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 盤面変化追跡用 (STABLE代理)
    prev_board_p1: np.ndarray | None = None
    prev_board_p2: np.ndarray | None = None

    records: list[dict] = []
    frame_idx = start_frame
    processed = 0

    BOARD_Y1 = WARNING_BOTTOM_Y
    BOARD_Y2 = min(1080, WARNING_BOTTOM_Y + 600)

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = frame_idx / fps

        if (frame_idx - start_frame) % SAMPLE_INTERVAL != 0:
            frame_idx += 1
            continue

        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080))

        processed += 1

        # 盤面変化量 (STABLE代理)
        board_p1 = frame[BOARD_Y1:BOARD_Y2, P1_BOARD_X:P1_BOARD_X + BOARD_WIDTH]
        board_p2 = frame[BOARD_Y1:BOARD_Y2, P2_BOARD_X:P2_BOARD_X + BOARD_WIDTH]

        board_change_p1 = 0.0
        board_change_p2 = 0.0
        if prev_board_p1 is not None:
            board_change_p1 = float(np.mean(np.abs(board_p1.astype(float) - prev_board_p1.astype(float))))
        if prev_board_p2 is not None:
            board_change_p2 = float(np.mean(np.abs(board_p2.astype(float) - prev_board_p2.astype(float))))

        prev_board_p1 = board_p1.copy()
        prev_board_p2 = board_p2.copy()

        # glow判定
        glow_p1 = _compute_glow(frame, P1_BOARD_X) >= GLOW_RATIO_THRESHOLD
        glow_p2 = _compute_glow(frame, P2_BOARD_X) >= GLOW_RATIO_THRESHOLD

        # 視覚予告検出
        vis_p1, vis_p2 = det.detect(frame)

        # 記録 (STABLE代理 = board_change < threshold, glow=False)
        p1_stable_proxy = (board_change_p1 < BOARD_CHANGE_STABLE_MAX and not glow_p1)
        p2_stable_proxy = (board_change_p2 < BOARD_CHANGE_STABLE_MAX and not glow_p2)

        records.append({
            "t_sec": t_sec,
            "p1_stable": p1_stable_proxy,
            "p2_stable": p2_stable_proxy,
            "p1_glow": glow_p1,
            "p2_glow": glow_p2,
            "vis_p1": vis_p1.total_count,
            "vis_p2": vis_p2.total_count,
            "p1_icons": [ic.icon_type for ic in vis_p1.icons],
            "p2_icons": [ic.icon_type for ic in vis_p2.icons],
        })

        frame_idx += 1

    cap.release()
    print(f"処理フレーム数: {processed}")
    _print_stats(records)


def _print_stats(records: list[dict]) -> None:
    total = len(records)
    print(f"\n=== 全フレーム統計 ===")
    print(f"  総フレーム数: {total}")

    # STABLE代理かつ非glow
    stable_records = [r for r in records if r["p1_stable"] or r["p2_stable"]]
    stable_nonglow_p1 = [r for r in records if r["p1_stable"] and not r["p1_glow"]]
    stable_nonglow_p2 = [r for r in records if r["p2_stable"] and not r["p2_glow"]]
    print(f"  STABLE代理フレーム: {len(stable_records)} ({100*len(stable_records)/total:.1f}%)")

    # --- 視覚検出統計 ---
    print(f"\n=== STABLE代理(非glow)での視覚検出統計 ===")
    for side_label, side_records, side_key in [
        ("1P", stable_nonglow_p1, "vis_p1"),
        ("2P", stable_nonglow_p2, "vis_p2"),
    ]:
        if not side_records:
            print(f"  {side_label}: データなし")
            continue
        vis_counts = [r[side_key] for r in side_records]
        nonzero = [v for v in vis_counts if v > 0]
        print(f"  {side_label}: n={len(side_records)}, visual>0フレーム: {len(nonzero)} ({100*len(nonzero)/len(side_records):.1f}%)")
        if nonzero:
            print(f"    visual mean(>0時): {statistics.mean(nonzero):.1f}, max: {max(nonzero)}, median: {statistics.median(nonzero):.0f}")
        else:
            print(f"    → 視覚検出器が全フレームで 0 を返しています")

    # --- アイコン種類分布 ---
    print(f"\n=== 検出されたアイコン種類分布 ===")
    icon_counts: dict[str, int] = {}
    for r in records:
        for ic in r["p1_icons"] + r["p2_icons"]:
            icon_counts[ic] = icon_counts.get(ic, 0) + 1

    if icon_counts:
        for kind, cnt in sorted(icon_counts.items(), key=lambda x: x[1], reverse=True):
            val = COUNT_TABLE.get(kind, "?")
            print(f"  {kind} (={val}個): {cnt} 回")
    else:
        print("  アイコンが一切検出されませんでした")

    # --- glow除外効果 ---
    print(f"\n=== glow発光の除外効果 ===")
    glow_p1 = sum(1 for r in records if r["p1_glow"])
    glow_p2 = sum(1 for r in records if r["p2_glow"])
    print(f"  1P glow_active: {glow_p1}/{total} ({100*glow_p1/total:.1f}%)")
    print(f"  2P glow_active: {glow_p2}/{total} ({100*glow_p2/total:.1f}%)")

    # glow中に視覚検出器が誤検出するか
    glow_vis_p1 = [r["vis_p1"] for r in records if r["p1_glow"] and r["vis_p1"] > 0]
    glow_vis_p2 = [r["vis_p2"] for r in records if r["p2_glow"] and r["vis_p2"] > 0]
    if glow_vis_p1:
        print(f"  glow中に1P visual>0: {len(glow_vis_p1)}フレーム (誤検出の可能性)")
    if glow_vis_p2:
        print(f"  glow中に2P visual>0: {len(glow_vis_p2)}フレーム (誤検出の可能性)")
    if not glow_vis_p1 and not glow_vis_p2:
        print(f"  glow中の誤検出: 0件 ✓")

    # --- サンプル: visual>0のフレーム ---
    vis_samples = [r for r in records if r["vis_p1"] > 0 or r["vis_p2"] > 0]
    print(f"\n=== visual>0 のサンプルフレーム (上位20件) ===")
    if vis_samples:
        for r in sorted(vis_samples, key=lambda x: x["vis_p1"] + x["vis_p2"], reverse=True)[:20]:
            print(f"  t={r['t_sec']:.1f}s 1P:{r['vis_p1']}[{','.join(r['p1_icons'])}] 2P:{r['vis_p2']}[{','.join(r['p2_icons'])}] glow={r['p1_glow']}/{r['p2_glow']}")
    else:
        print("  visual>0 のフレームがありません")
        print("  → ROI が実際のアイコン位置をカバーしていない可能性があります")
        print(f"  → 現在: WARNING_TOP_Y={WARNING_TOP_Y}, WARNING_BOTTOM_Y={WARNING_BOTTOM_Y}")
        print(f"  → 実測: アイコン中心はy≈130-145付近 (上記ROIで下端付近にかかる)")

    # --- 所見 ---
    print(f"\n=== 所見サマリー ===")
    all_vis = [r["vis_p1"] + r["vis_p2"] for r in records]
    nonzero_all = [v for v in all_vis if v > 0]
    total_nonzero_rate = len(nonzero_all) / total if total > 0 else 0
    print(f"  視覚検出率(全フレーム): {100*total_nonzero_rate:.1f}%")

    if total_nonzero_rate < 0.01:
        print("  WARNING: 視覚検出率が1%未満 → ROI調整またはテンプレ更新が必要")
    elif total_nonzero_rate < 0.05:
        print("  CAUTION: 視覚検出率が低め (< 5%) → 精度検証に十分なサンプルがない可能性")
    else:
        print("  OK: 視覚検出率は十分")


if __name__ == "__main__":
    main()
