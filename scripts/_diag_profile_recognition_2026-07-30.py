"""認識パイプラインのプロファイルを取る (何が重いかの実測)。

「matchTemplateが62%」は古い記録からの引用で未実測。どこに時間が使われているか
分からないまま最適化するのは推測なので、まず測る。

本番と同じ設定 (enable_chain_tracker=True) でパイプライン単体を回し、
cProfile の tottime でランキングを出す。他プロセスとのCPU競合で絶対値は膨らむが、
**相対比率は競合の影響を受けにくい**ので「何が重いか」の判断には使える。
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
from collections import defaultdict

import cv2

from src.recognition_pipeline import RecognitionPipeline

TARGET_W, TARGET_H = 1920, 1080
TOP_N = 22


def build_pipeline() -> RecognitionPipeline:
    """本番(レンダ)と同じ設定でパイプラインを作る。"""
    return RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        force_in_match=True,
    )


def run_frames(pipeline: RecognitionPipeline, video: str,
               start_sec: float, n_frames: int) -> int:
    """指定区間を全フレーム処理する。処理できたフレーム数を返す。"""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    done = 0
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H),
                               interpolation=cv2.INTER_AREA)
        fi = start_frame + i
        pipeline.update(fi, fi / fps, frame)
        done += 1
    cap.release()
    return done


def module_rollup(stats: pstats.Stats) -> list[tuple[str, float]]:
    """ファイル単位に tottime を集約して降順で返す。"""
    agg: dict[str, float] = defaultdict(float)
    for (fname, _lineno, _func), (_cc, _nc, tt, _ct, _cs) in stats.stats.items():
        key = fname.split("/")[-1].split("\\")[-1]
        agg[key] += tt
    return sorted(agg.items(), key=lambda kv: -kv[1])


def main() -> None:
    """プロファイルを実行して結果を表示する。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-sec", type=float, default=1450.0)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--cv-threads", type=int, default=-1,
                    help="-1 は既定のまま (通常16)。1 で単スレッド")
    args = ap.parse_args()

    if args.cv_threads >= 0:
        cv2.setNumThreads(args.cv_threads)
    print(f"cv2 threads = {cv2.getNumThreads()}")

    pipeline = build_pipeline()
    # 初回呼び出しのモデルロード等を除くためウォームアップ
    run_frames(pipeline, args.video, args.start_sec, 10)

    prof = cProfile.Profile()
    prof.enable()
    done = run_frames(pipeline, args.video, args.start_sec + 1.0, args.frames)
    prof.disable()

    st = pstats.Stats(prof, stream=io.StringIO())
    total = sum(tt for (_cc, _nc, tt, _ct, _cs) in st.stats.values())
    print(f"\n処理フレーム数 {done} / 総tottime {total:.2f}秒 "
          f"= 1フレーム {total/max(done,1)*1000:.1f}ms "
          f"→ {max(done,1)/total:.1f} fps 相当")

    print(f"\n=== ファイル単位の内訳 (tottime上位) ===")
    for name, tt in module_rollup(st)[:14]:
        print(f"  {tt/total*100:5.1f}%  {tt:7.2f}秒  {name}")

    print(f"\n=== 関数単位 (tottime上位{TOP_N}) ===")
    rows = sorted(st.stats.items(),
                  key=lambda kv: -kv[1][2])[:TOP_N]
    for (fname, lineno, func), (_cc, nc, tt, ct, _cs) in rows:
        short = fname.split("/")[-1].split("\\")[-1]
        print(f"  {tt/total*100:5.1f}%  {tt:7.2f}秒  {nc:>7}回  "
              f"{short}:{lineno} {func}")


if __name__ == "__main__":
    main()
