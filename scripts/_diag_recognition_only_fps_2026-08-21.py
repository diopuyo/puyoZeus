"""認識単独fpsの再実測 (2026-08-21)。

背景 (coordinator 依頼、user指摘「認識今40fpsくらいだったと思ってた」への回答用):
    記録に残る値は 単独25.67fps / 4並列12fps / 14並列8-10fps
    (memory project_speed_4to26fps_2026-07-31、video不明・30fps相当・全フレーム処理)。
    本日 (2026-08-20) HSV分類をRustへ移植し
    「1 frame 34.69→29.05ms (1.19倍)、update単体 31.67→26.32ms」という数字が
    src/production_config.py の採用根拠1として記録済みだが、これは video_c34の
    600秒地点から1,990frameの一区間の実測であり、**「単独25.67fps」の再測定
    (同一条件での前後比較) ではない**。両者を同一スクリプト・同一手法で
    揃えて測り直す。

測る内容:
    1. video_c34.mp4 (30fps) で native HSV ON/OFF を単独実行 (並列なし) で
       比較。全フレーム処理 (stride=1、既存25.67fps系列と条件を揃える)。
       - 「1frame」= 前処理(resize) + pipe.update() 実測 (production_config
         の "1 frame" 表記と揃える)
       - 「update単体」= pipe.update() のみの実測 (同上 "update単体" と揃える)
    2. recognition_load_default_kwargs() (本番の RECOGNITION_ADOPTED 全フラグ)
       経由で native HSV が実際に有効化されているか確認 (fail-silent 警戒、
       pipe._native_hsv_active を出力)。
    3. video_zenchi (60fps、本番対象動画) で本番と同じ stride=2 (実効30fps)
       セマンティクスを再現し、「update() が実際に呼ばれた回のみ」のfpsと
       「動画の実時間1秒を処理するための壁時間」を分けて報告する
       (stride skip フレームは cap.read() のみ行う decode コストも計上)。

cProfile 禁止 (memory project_speed_4to26fps_2026-07-31 4.)。time.perf_counter のみ。
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

from src.fps_normalize import resolve_normalize_fps_30_stride  # noqa: E402
from src.production_config import recognition_load_default_kwargs  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

NATIVE_W, NATIVE_H = 1920, 1080


def _bench_allframes(video: Path, start_sec: float, n_frames: int,
                      native_hsv: bool, warmup: int = 10) -> dict:
    """stride=1 (全フレーム) で pipe.update() を単独実行し timing を取る。

    production_config の既存記録 (34.69/29.05ms 系) と条件を揃えるため、
    load_default は他フラグ (RECOGNITION_ADOPTED) を混ぜず native_hsv 単独を
    明示指定する (交絡防止・A/B比較を native_hsv のみの純粋差にする)。
    """
    pipe = RecognitionPipeline.load_default(enable_native_hsv_classifier=native_hsv)
    active = getattr(pipe, "_native_hsv_active", None)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"[error] 動画を開けない: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    frame_ms: list[float] = []   # resize + update
    update_ms: list[float] = []  # update のみ
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t0 = time.perf_counter()
        if frame.shape[:2] != (NATIVE_H, NATIVE_W):
            frame = cv2.resize(frame, (NATIVE_W, NATIVE_H), interpolation=cv2.INTER_AREA)
        t1 = time.perf_counter()
        pipe.update(i, start_sec + i / fps, frame)
        t2 = time.perf_counter()
        frame_ms.append((t2 - t0) * 1000.0)
        update_ms.append((t2 - t1) * 1000.0)
    cap.release()
    frame_ms = frame_ms[warmup:] or frame_ms
    update_ms = update_ms[warmup:] or update_ms
    return {
        "native_hsv_requested": native_hsv,
        "native_hsv_active": active,
        "n": len(frame_ms),
        "frame_median_ms": statistics.median(frame_ms) if frame_ms else float("nan"),
        "frame_mean_ms": statistics.mean(frame_ms) if frame_ms else float("nan"),
        "update_median_ms": statistics.median(update_ms) if update_ms else float("nan"),
        "update_mean_ms": statistics.mean(update_ms) if update_ms else float("nan"),
    }


def _bench_production_stride(video: Path, start_sec: float, duration_sec: float,
                              native_hsv: bool, warmup_frames: int = 20) -> dict:
    """本番 (overlay/collect) と同じ 60fps→stride=2 セマンティクスを再現する。

    decode (cap.read()) は全フレームで行い、stride 対象フレームのみ
    pipe.update() を呼ぶ (visualize_advantage_overlay.py:4359-4369,
    collect_boards_lean.py と同じ「decode毎回・update間引き」方式)。
    """
    pipe = RecognitionPipeline.load_default(enable_native_hsv_classifier=native_hsv)
    active = getattr(pipe, "_native_hsv_active", None)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"[error] 動画を開けない: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = resolve_normalize_fps_30_stride(fps)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    n_frames = int(duration_sec * fps)

    decode_ms: list[float] = []       # stride対象外 (decodeのみ) フレーム
    update_cycle_ms: list[float] = [] # decode+resize+update (stride対象フレーム)
    update_only_ms: list[float] = []
    t_wall0 = time.perf_counter()
    n_decoded = 0
    for i in range(n_frames):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        n_decoded += 1
        if i % stride != 0:
            decode_ms.append((time.perf_counter() - t0) * 1000.0)
            continue
        if frame.shape[:2] != (NATIVE_H, NATIVE_W):
            frame = cv2.resize(frame, (NATIVE_W, NATIVE_H), interpolation=cv2.INTER_AREA)
        t1 = time.perf_counter()
        pipe.update(i, start_sec + i / fps, frame)
        t2 = time.perf_counter()
        update_cycle_ms.append((t2 - t0) * 1000.0)
        update_only_ms.append((t2 - t1) * 1000.0)
    wall_total_sec = time.perf_counter() - t_wall0
    video_sec_covered = n_decoded / fps

    # ウォームアップ分 (呼び出し回数ベース) を落とす
    w = max(1, warmup_frames // stride)
    uc = update_cycle_ms[w:] or update_cycle_ms
    uo = update_only_ms[w:] or update_only_ms

    return {
        "native_hsv_requested": native_hsv,
        "native_hsv_active": active,
        "stride": stride,
        "fps_source": fps,
        "n_decoded": n_decoded,
        "n_update_calls": len(update_cycle_ms),
        "update_cycle_median_ms": statistics.median(uc) if uc else float("nan"),
        "update_cycle_mean_ms": statistics.mean(uc) if uc else float("nan"),
        "update_only_median_ms": statistics.median(uo) if uo else float("nan"),
        "decode_only_median_ms": statistics.median(decode_ms) if decode_ms else float("nan"),
        "wall_total_sec": wall_total_sec,
        "video_sec_covered": video_sec_covered,
        "realtime_multiple": (
            video_sec_covered / wall_total_sec if wall_total_sec > 0 else float("nan")
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-30fps", type=Path, default=Path("data/frames/video_c34.mp4"))
    ap.add_argument("--start-sec-30fps", type=float, default=600.0)
    ap.add_argument("--frames-30fps", type=int, default=800)
    ap.add_argument(
        "--video-60fps", type=Path,
        default=Path("data/frames/video_zenchi_c0BQoMJwwQU.mp4"),
    )
    ap.add_argument("--start-sec-60fps", type=float, default=6733.0)
    ap.add_argument("--duration-sec-60fps", type=float, default=60.0)
    ap.add_argument("--skip-60fps", action="store_true",
                     help="60fps側 (本番動画) 区間はスキップし30fps側のみ測る")
    args = ap.parse_args()

    print("=" * 78)
    print("[1] video_c34 (30fps相当・stride=1・全フレーム) 単独実行、native HSV A/B")
    print("    (memory project_speed_4to26fps_2026-07-31 / production_config "
          "採用根拠1 と同一手法)")
    print("=" * 78)
    for native in (True, False):
        r = _bench_allframes(args.video_30fps, args.start_sec_30fps,
                              args.frames_30fps, native)
        fps_frame = 1000.0 / r["frame_mean_ms"] if r["frame_mean_ms"] else float("nan")
        fps_update = 1000.0 / r["update_mean_ms"] if r["update_mean_ms"] else float("nan")
        print(f"\n--native_hsv={native} (active={r['native_hsv_active']}) n={r['n']}")
        print(f"  1frame(resize+update): 中央値{r['frame_median_ms']:.2f}ms "
              f"平均{r['frame_mean_ms']:.2f}ms -> {fps_frame:.2f}fps")
        print(f"  update単体           : 中央値{r['update_median_ms']:.2f}ms "
              f"平均{r['update_mean_ms']:.2f}ms -> {fps_update:.2f}fps")

    print("\n" + "=" * 78)
    print("[1b] recognition_load_default_kwargs() (本番RECOGNITION_ADOPTED全フラグ) "
          "経由で native HSV が実際に有効か確認")
    print("=" * 78)
    kwargs = recognition_load_default_kwargs()
    print(f"  kwargs = {kwargs}")
    pipe_prod = RecognitionPipeline.load_default(**kwargs)
    print(f"  pipe._native_hsv_active = {getattr(pipe_prod, '_native_hsv_active', None)}")

    if args.skip_60fps:
        return

    print("\n" + "=" * 78)
    print("[2] video_zenchi (60fps・本番stride=2セマンティクス再現) 単独実行、"
          "native HSV A/B")
    print("=" * 78)
    for native in (True, False):
        r = _bench_production_stride(
            args.video_60fps, args.start_sec_60fps, args.duration_sec_60fps, native)
        print(f"\n--native_hsv={native} (active={r['native_hsv_active']}) "
              f"stride={r['stride']} fps_source={r['fps_source']:.2f}")
        print(f"  decode総数={r['n_decoded']} update呼び出し数={r['n_update_calls']}")
        print(f"  update呼び出しサイクル(decode含む): 中央値{r['update_cycle_median_ms']:.2f}ms "
              f"平均{r['update_cycle_mean_ms']:.2f}ms")
        print(f"  update()のみ: 中央値{r['update_only_median_ms']:.2f}ms")
        print(f"  stride対象外decodeのみ: 中央値{r['decode_only_median_ms']:.2f}ms")
        print(f"  壁時間{r['wall_total_sec']:.2f}s で動画実時間{r['video_sec_covered']:.2f}s "
              f"処理 -> 実時間の{r['realtime_multiple']:.3f}倍速 "
              f"(1.0未満=実時間より遅い)")
        if r["realtime_multiple"] > 0:
            print(f"  換算: 実時間1分処理に壁時間{60.0/r['realtime_multiple']:.1f}秒")


if __name__ == "__main__":
    main()
