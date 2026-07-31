"""STEP 2: STABLE 時の視覚予告検出器精度実測スクリプト (scratch 扱い).

video_124_4min.mp4 の代表区間 (120s) で:
    - RecognitionPipeline.load_default() で各フレームを処理
    - STABLE かつ非発光 (glow_guard OFF) のフレームを対象に
      OjamaWarningDetector.detect() を実行 → visual_count(1P/2P) を取得
    - OjamaAccountingTracker の forecast_p1/p2 も追跡
    - 乖離統計 (mean/max/大乖離率/方向バイアス) を集計・報告

2026-07-03 Phase 1b 精度実測
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")

from src.board_state_machine import BoardState
from src.ojama_accounting import OjamaAccountingTracker
from src.ojama_warning import OjamaWarningDetector
from src.ojama_warning_glow_guard import (
    GLOW_DETECTION_THRESHOLD,
    GlowGuardState,
    compute_glow_score,
    update_glow_state,
)
from src.recognition_pipeline import RecognitionPipeline

# ============================
# 設定定数
# ============================
VIDEO_PATH: str = "data/frames/video_124_4min.mp4"
# 代表区間: 0~120s (計算時間短縮のため前半)
START_SEC: float = 0.0
END_SEC: float = 120.0
# BoardRegion はデフォルト値を使う
# 大乖離とみなす閾値: 岩1個分
LARGE_DIFF_THRESHOLD: int = 30
# ログ出力レベル
logging.basicConfig(level=logging.WARNING)


def _get_board_region(pipe: RecognitionPipeline) -> tuple:
    """pipeline から 1P/2P の BoardRegion を取得する."""
    try:
        from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
        return DEFAULT_P1_REGION, DEFAULT_P2_REGION
    except Exception:
        return None, None


def main() -> None:
    print(f"=== STEP 2: STABLE 時視覚予告精度実測 ===")
    print(f"動画: {VIDEO_PATH}")
    print(f"区間: {START_SEC:.0f}s ~ {END_SEC:.0f}s")
    print()

    # パイプライン初期化
    print("RecognitionPipeline 初期化中...")
    pipe = RecognitionPipeline.load_default()
    print("  完了")

    # 視覚検出器初期化
    det = OjamaWarningDetector(use_cnn=False)  # CNNなし: テンプレ+HSVのみ

    # 会計トラッカー
    acct = OjamaAccountingTracker()

    # glow_guard state
    p1_glow_state = GlowGuardState()
    p2_glow_state = GlowGuardState()

    # BoardRegion 取得
    r1, r2 = _get_board_region(pipe)

    # 動画読み込み
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    start_frame = int(START_SEC * fps)
    end_frame = int(END_SEC * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 集計用バッファ
    records: list[dict] = []
    frame_idx = start_frame

    print(f"フレーム処理中 (fps={fps:.0f}, フレーム数={end_frame-start_frame})...")

    prev_p1_state = None
    prev_p2_state = None

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = frame_idx / fps

        # 解像度統一
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080))

        # パイプライン更新
        result = pipe.update(frame_idx, t_sec, frame)

        # STABLE 判定
        p1_state = result.p1.state
        p2_state = result.p2.state
        p1_stable = (p1_state == BoardState.STABLE)
        p2_stable = (p2_state == BoardState.STABLE)

        # glow_guard 更新
        p1_glow_score = 0.0
        p2_glow_score = 0.0
        if r1 is not None:
            p1_glow_score = compute_glow_score(frame, r1)
            p2_glow_score = compute_glow_score(frame, r2)

        p1_glow_active = update_glow_state(p1_glow_state, p1_glow_score, frame_idx)
        p2_glow_active = update_glow_state(p2_glow_state, p2_glow_score, frame_idx)

        # 会計トラッカー更新 (score から)
        score_p1 = result.p1.score if hasattr(result.p1, "score") else None
        score_p2 = result.p2.score if hasattr(result.p2, "score") else None

        # on_state_transition 呼び出し (state 遷移時)
        if prev_p1_state is not None and prev_p1_state != p1_state:
            acct.on_state_transition(
                "p1",
                prev_state=prev_p1_state,
                curr_state=p1_state,
                score=score_p1,
                t_sec=t_sec,
            )
        if prev_p2_state is not None and prev_p2_state != p2_state:
            acct.on_state_transition(
                "p2",
                prev_state=prev_p2_state,
                curr_state=p2_state,
                score=score_p2,
                t_sec=t_sec,
            )

        prev_p1_state = p1_state
        prev_p2_state = p2_state

        # STABLE かつ非発光フレームのみ収集
        if not (p1_stable or p2_stable):
            frame_idx += 1
            continue

        # 視覚検出器実行
        vis_p1, vis_p2 = det.detect(frame)
        vis_count_p1 = vis_p1.total_count
        vis_count_p2 = vis_p2.total_count

        # 会計スナップショット
        snap = acct.get_snapshot(t_sec)
        forecast_p1 = snap.forecast_p1
        forecast_p2 = snap.forecast_p2

        # 各フレームを記録
        if p1_stable:
            records.append({
                "t_sec": t_sec,
                "side": "1P",
                "stable": True,
                "glow_active": p1_glow_active,
                "visual": vis_count_p1,
                "forecast": forecast_p1,
                "diff": abs(vis_count_p1 - forecast_p1),
                "signed_diff": vis_count_p1 - forecast_p1,
            })
        if p2_stable:
            records.append({
                "t_sec": t_sec,
                "side": "2P",
                "stable": True,
                "glow_active": p2_glow_active,
                "visual": vis_count_p2,
                "forecast": forecast_p2,
                "diff": abs(vis_count_p2 - forecast_p2),
                "signed_diff": vis_count_p2 - forecast_p2,
            })

        frame_idx += 1

    cap.release()
    print(f"  処理完了: {len(records)} 件の STABLE フレーム収集")

    if not records:
        print("ERROR: STABLE フレームが収集できませんでした")
        return

    _print_stats(records)


def _print_stats(records: list[dict]) -> None:
    """乖離統計を報告する."""
    import statistics

    # --- 全 STABLE フレーム ---
    total = len(records)
    glow_on = sum(1 for r in records if r["glow_active"])
    glow_off_records = [r for r in records if not r["glow_active"]]

    print()
    print("=== 全 STABLE フレーム統計 ===")
    print(f"  合計 STABLE フレーム: {total}")
    print(f"  うち glow_active=True: {glow_on} ({100*glow_on/total:.1f}%)")
    print(f"  うち glow_active=False: {len(glow_off_records)} ({100*len(glow_off_records)/total:.1f}%)")

    # --- glow_active=False フレームの乖離 (本命集計) ---
    if not glow_off_records:
        print("WARNING: glow OFF の STABLE フレームがありません")
        return

    print()
    print("=== STABLE かつ非発光フレームの乖離統計 ===")
    diffs = [r["diff"] for r in glow_off_records]
    signed = [r["signed_diff"] for r in glow_off_records]

    diff_mean = statistics.mean(diffs)
    diff_max = max(diffs)
    large_diff_count = sum(1 for d in diffs if d >= LARGE_DIFF_THRESHOLD)
    large_diff_rate = large_diff_count / len(diffs) if diffs else 0.0
    signed_mean = statistics.mean(signed)

    print(f"  フレーム数: {len(glow_off_records)}")
    print(f"  |visual - forecast| mean: {diff_mean:.2f} 個")
    print(f"  |visual - forecast| max:  {diff_max} 個")
    print(f"  大乖離 (>={LARGE_DIFF_THRESHOLD}個=岩1個分) 件数: {large_diff_count} / {len(diffs)} ({100*large_diff_rate:.1f}%)")
    print(f"  方向バイアス (visual - forecast) mean: {signed_mean:.2f}")
    print(f"    正値 = visual > forecast (視覚が会計より多い)")
    print(f"    負値 = visual < forecast (会計が視覚より多い)")

    # --- forecast>0 or visual>0 のフレームに絞った集計 ---
    active_records = [r for r in glow_off_records if r["visual"] > 0 or r["forecast"] > 0]
    print()
    print(f"=== forecast>0 or visual>0 のフレーム (意味ある集計) ===")
    if active_records:
        a_diffs = [r["diff"] for r in active_records]
        a_signed = [r["signed_diff"] for r in active_records]
        a_large = sum(1 for d in a_diffs if d >= LARGE_DIFF_THRESHOLD)
        print(f"  フレーム数: {len(active_records)}")
        print(f"  |visual - forecast| mean: {statistics.mean(a_diffs):.2f} 個")
        print(f"  |visual - forecast| max:  {max(a_diffs)} 個")
        print(f"  大乖離率 (>={LARGE_DIFF_THRESHOLD}個): {a_large}/{len(a_diffs)} ({100*a_large/len(a_diffs):.1f}%)")
        print(f"  方向バイアス mean: {statistics.mean(a_signed):.2f}")

        # 大乖離サンプル表示
        large_samples = sorted(
            [r for r in active_records if r["diff"] >= LARGE_DIFF_THRESHOLD],
            key=lambda x: x["diff"], reverse=True,
        )[:10]
        if large_samples:
            print()
            print("  大乖離サンプル (diff>=30):")
            for r in large_samples:
                print(f"    t={r['t_sec']:.1f}s {r['side']}: visual={r['visual']} forecast={r['forecast']} diff={r['diff']}")
    else:
        print("  該当フレームなし (全フレームで visual=0 かつ forecast=0)")

    # --- サイド別集計 ---
    print()
    print("=== サイド別 (1P/2P 別) 集計 ===")
    for side in ["1P", "2P"]:
        side_r = [r for r in glow_off_records if r["side"] == side]
        if not side_r:
            continue
        s_diffs = [r["diff"] for r in side_r]
        print(f"  {side}: n={len(side_r)} mean_diff={statistics.mean(s_diffs):.2f} max_diff={max(s_diffs)}")

    # --- visual > 0 のフレームで実際の検出内容サンプル ---
    print()
    print("=== 視覚検出器が0より多い個数を返したフレームのサンプル ===")
    visual_detected = [r for r in glow_off_records if r["visual"] > 0][:15]
    if visual_detected:
        for r in visual_detected:
            print(f"  t={r['t_sec']:.1f}s {r['side']}: visual={r['visual']} forecast={r['forecast']} diff={r['diff']}")
    else:
        print("  視覚検出器が常に 0 を返しています (ROI 外れの可能性)")

    print()
    print("=== 所見サマリー ===")
    if diff_mean < 5:
        print("  視覚 vs 会計の平均乖離が小さく (< 5個)、概ね一致しています")
    elif diff_mean < 30:
        print(f"  中程度の乖離 ({diff_mean:.1f}個) があります。ROI調整や値修正で改善余地あり")
    else:
        print(f"  大きな乖離 ({diff_mean:.1f}個) があります。根本的な認識問題の可能性")

    if all(r["visual"] == 0 for r in glow_off_records):
        print("  WARNING: 視覚検出器が全フレームで 0 を返しています")
        print("  → ROI (WARNING_TOP_Y/BOTTOM_Y) がアイコンをカバーしていない可能性")


if __name__ == "__main__":
    main()
