"""enable_match_start_full_clear の試合中誤発火を診断する使い捨てスクリプト。

video_29 の mid 窓 (--start-sec 1200 --max-sec 360) を新旧2構成で走らせ、
RecognitionPipeline.reset() の呼び出し回数と発生時刻 (t_sec) を計測する。
新構成 (enable_match_start_full_clear=True, 既定) と
旧構成 (enable_match_start_full_clear=False, 明示 False) を比較し、
「試合開始/全消し以外の中盤〜終盤」で reset が多発していないかを確認する。

read-only 診断用。collect_indicators_v2.py の本体ロジックは変更しない。
使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._tmp_reset_probe_2026-07-26
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
cv2.setNumThreads(1)

from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

TARGET_W, TARGET_H = 1920, 1080
DEFAULT_FPS = 30.0

VIDEO_PATH = Path.home() / "frames" / "video_29.mp4"
START_SEC = 1200.0
MAX_SEC = 45.0


def run(enable_full_clear: bool) -> None:
    """指定構成でパイプラインを走らせ、reset 発生時刻と tsumo 推移を記録する。"""
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"[ERROR] cannot open: {VIDEO_PATH}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(START_SEC * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    end_frame = min(total_frames, start_frame + int(MAX_SEC * fps))
    n_frames = max(0, end_frame - start_frame)

    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
        enable_match_start_full_clear=enable_full_clear,
    )
    pipeline.set_video_id("video_29")

    # --- reset() 呼び出しを計装 (monkeypatch, 使い捨て診断専用) ---
    reset_log: list[dict] = []
    last_t_sec_holder = {"t": None}
    orig_reset = pipeline.reset

    def _patched_reset() -> None:
        reset_log.append({
            "t_sec": last_t_sec_holder["t"],
            "prev_score_1p": getattr(pipeline, "_prev_score_for_reset_1p", None),
            "prev_score_2p": getattr(pipeline, "_prev_score_for_reset_2p", None),
        })
        orig_reset()

    pipeline.reset = _patched_reset  # type: ignore[method-assign]

    tsumo_log: list[tuple[float, int, int]] = []
    n_processed = 0
    for local_i in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        fi = start_frame + local_i
        t_sec = fi / fps
        last_t_sec_holder["t"] = t_sec
        pipeline.update(fi, t_sec, frame)
        n_processed += 1
        if local_i % 90 == 0:  # 約3秒おきに tsumo をサンプリング
            tsumo_log.append((t_sec, pipeline.tsumo_count("1P"), pipeline.tsumo_count("2P")))
    cap.release()

    label = "enable_match_start_full_clear=True(新既定)" if enable_full_clear else "enable_match_start_full_clear=False(旧挙動)"
    print(f"\n=== {label} ===", flush=True)
    print(f"  処理フレーム数: {n_processed}  reset回数: {len(reset_log)}")
    for r in reset_log:
        print(f"    reset @ t_sec={r['t_sec']:.2f}  "
              f"prev_score_1p={r['prev_score_1p']}  prev_score_2p={r['prev_score_2p']}")
    print(f"  tsumo推移(約3秒毎、末尾10件): ")
    for t, t1, t2 in tsumo_log[-10:]:
        print(f"    t_sec={t:.1f}  tsumo_1p={t1}  tsumo_2p={t2}")
    print(f"  最終 tsumo: 1P={pipeline.tsumo_count('1P')}  2P={pipeline.tsumo_count('2P')}")


if __name__ == "__main__":
    run(enable_full_clear=True)
    run(enable_full_clear=False)
