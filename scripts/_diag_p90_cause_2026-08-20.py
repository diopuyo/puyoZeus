"""p90 が中央値より 10ms 以上遅い原因を特定する (2026-08-20)。

Rust 化後の実測で 1 frame の中央値 26.32ms に対し **p90 が 37.12ms**。
classify 回数による層別では説明できない (どの帯の中央値も最大 27.09ms)。

仮説: **大ROI走査 (match_end 800x600 + telop 720x400) が
LARGE_ROI_THROTTLE_FRAMES=8 フレームに1回だけ走る**ため、12.5% のフレームが
突出して重い。内訳診断の「match_end 1.1ms / telop 0.7ms」は 0.1回/frame の
平均値であり、1回あたりの実コストは約 11ms / 7ms、合計約 18ms と逆算できる。
26.32 + 18 = 44ms で p90 の位置と符合する。

検証方法: フレームを「大ROI走査が走った回」と「走らなかった回」に分けて
update 時間を比べる。走査の有無は `_should_run_large_roi_scan` を計装して
実際の判定結果を記録する (frame_idx % 8 を仮定しない — 実装が別の条件で
判定している可能性を排除する)。

これが確定すれば、リアルタイム用途では間引き幅を広げて p90 を予算内に
入れる選択肢が取れる (試合終了検出が遅れるトレードオフは
src/production_config.py に記録済み)。
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

cv2.setNumThreads(1)

from src.recognition_pipeline import RecognitionPipeline  # noqa: E402


def main() -> None:
    """大ROI走査の有無で update 時間を層別する。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=Path("data/frames/video_c34.mp4"))
    ap.add_argument("--start-sec", type=float, default=600.0)
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--native-hsv", action="store_true", default=True)
    args = ap.parse_args()

    pipe = RecognitionPipeline.load_default(
        enable_native_hsv_classifier=args.native_hsv,
    )
    print(f"[config] native HSV: {getattr(pipe, '_native_hsv_active', None)}")

    # 大ROI走査の判定結果を毎フレーム記録する (実装の条件をそのまま観測)
    scan_flags: list[bool] = []
    orig = type(pipe)._should_run_large_roi_scan

    def spy(self, frame_idx):  # noqa: ANN001, ANN202
        r = orig(self, frame_idx)
        scan_flags.append(bool(r))
        return r

    type(pipe)._should_run_large_roi_scan = spy

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[error] 動画を開けない: {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000.0)

    rows: list[tuple[float, bool]] = []
    try:
        for i in range(args.frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            n_before = len(scan_flags)
            t0 = time.perf_counter()
            pipe.update(i, args.start_sec + i / fps, frame)
            dt = (time.perf_counter() - t0) * 1000.0
            # このフレームで1回でも True が返っていれば「走査あり」
            scanned = any(scan_flags[n_before:])
            rows.append((dt, scanned))
    finally:
        type(pipe)._should_run_large_roi_scan = orig
        cap.release()

    rows = rows[10:] or rows
    if not rows:
        print("[error] フレームを読めなかった")
        return

    allt = [r[0] for r in rows]
    scan = [r[0] for r in rows if r[1]]
    noscan = [r[0] for r in rows if not r[1]]
    allt_s = sorted(allt)
    p90 = allt_s[int(len(allt_s) * 0.9)]

    print(f"\n=== {args.video.name} {args.start_sec:.0f}s から {len(rows)} frame ===")
    print(f"全体: 中央値 {statistics.median(allt):.2f}ms / p90 {p90:.2f}ms\n")
    print(f"{'区分':<24} {'frame数':>8} {'中央値':>9} {'平均':>9} {'最大':>9}")
    print("-" * 62)
    for label, xs in (("大ROI走査あり", scan), ("大ROI走査なし", noscan)):
        if not xs:
            print(f"{label:<24} {'0':>8}  (観測なし)")
            continue
        print(f"{label:<24} {len(xs):8d} {statistics.median(xs):8.2f}ms "
              f"{statistics.mean(xs):8.2f}ms {max(xs):8.2f}ms")
    print("-" * 62)
    if scan and noscan:
        diff = statistics.median(scan) - statistics.median(noscan)
        ratio = len(scan) / len(rows) * 100
        print(f"  走査ありの割合: {ratio:.1f}%  (間引き 1/8 なら 12.5%)")
        print(f"  走査による増分 (中央値差): {diff:+.2f}ms")
        print()
        if diff > 5.0:
            print("  → 仮説を支持: 大ROI走査が p90 を作っている。")
            print("     リアルタイム用途では間引き幅を広げれば p90 を下げられる")
            print("     (試合終了検出が遅れるトレードオフは production_config に記録済み)。")
        else:
            print("  → 仮説を棄却: 大ROI走査では説明できない。別の要因を探す必要がある")
            print("     (状態遷移時の追加処理・bg_fp 更新・GC 等)。")


if __name__ == "__main__":
    main()
