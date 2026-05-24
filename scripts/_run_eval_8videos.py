"""eval 8 動画を並列 3 で実行する Python ランナー (シェル変数展開問題を回避)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = Path("/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer")
CNN_MODEL = str(BASE / "models/cnn_phase_b_large_v2.pt")
PYTHON = str(BASE / "venv/bin/python")
MAX_PARALLEL = 3

TASKS = [
    ("v29m2", "data/baseline_videos_v3/v29m2_buf15s.mp4",
     "data/verify/per_video_inject_eval/v29m2.mp4",
     "data/verify/per_video_inject_eval/v29m2.json",
     "logs/per_video_inject/eval_v29m2.jsonl"),
    ("v40m7", "data/baseline_videos_v3/v40m7_buf15s.mp4",
     "data/verify/per_video_inject_eval/v40m7.mp4",
     "data/verify/per_video_inject_eval/v40m7.json",
     "logs/per_video_inject/eval_v40m7.jsonl"),
    ("v51m2", "data/baseline_videos_v3/v51m2_buf15s.mp4",
     "data/verify/per_video_inject_eval/v51m2.mp4",
     "data/verify/per_video_inject_eval/v51m2.json",
     "logs/per_video_inject/eval_v51m2.jsonl"),
    ("v57m2", "data/baseline_videos_v3/v57m2_buf15s.mp4",
     "data/verify/per_video_inject_eval/v57m2.mp4",
     "data/verify/per_video_inject_eval/v57m2.json",
     "logs/per_video_inject/eval_v57m2.jsonl"),
    ("v70m2", "data/baseline_videos_v3/v70m2_buf15s.mp4",
     "data/verify/per_video_inject_eval/v70m2.mp4",
     "data/verify/per_video_inject_eval/v70m2.json",
     "logs/per_video_inject/eval_v70m2.jsonl"),
    ("v89m3", "data/baseline_videos_v3/v89m3_buf15s.mp4",
     "data/verify/per_video_inject_eval/v89m3.mp4",
     "data/verify/per_video_inject_eval/v89m3.json",
     "logs/per_video_inject/eval_v89m3.jsonl"),
    ("v95m15", "data/baseline_videos_v3/v95m15_buf15s.mp4",
     "data/verify/per_video_inject_eval/v95m15.mp4",
     "data/verify/per_video_inject_eval/v95m15.json",
     "logs/per_video_inject/eval_v95m15.jsonl"),
    ("v97m11", "data/baseline_videos_v3/v97m11_buf15s.mp4",
     "data/verify/per_video_inject_eval/v97m11.mp4",
     "data/verify/per_video_inject_eval/v97m11.json",
     "logs/per_video_inject/eval_v97m11.jsonl"),
]


def run_one(key: str, input_mp4: str, output_mp4: str, report: str, board_log: str) -> None:
    """1動画分の viz + evaluate を直列実行."""
    report_path = BASE / report
    input_path = BASE / input_mp4
    if report_path.exists():
        print(f"[skip] {key} (report exists)")
        return
    if not input_path.exists():
        print(f"[skip-missing] {key} ({input_path})")
        return
    log_path = BASE / f"logs/per_video_inject/run_{key}.log"
    env = {"PYTHONPATH": str(BASE)}
    import os
    env = {**os.environ, "PYTHONPATH": str(BASE)}
    with log_path.open("w", encoding="utf-8") as lf:
        # viz
        ret = subprocess.run(
            [PYTHON, "-m", "scripts.visualize_recognition",
             "--video", str(BASE / input_mp4),
             "--output", str(BASE / output_mp4),
             "--cnn-model", CNN_MODEL,
             "--dump-board-log", str(BASE / board_log)],
            cwd=str(BASE), env=env,
            stdout=lf, stderr=subprocess.STDOUT,
        )
        if ret.returncode != 0:
            print(f"[warn] viz failed: {key} (rc={ret.returncode})")
        # evaluate
        ret2 = subprocess.run(
            [PYTHON, "-m", "scripts.evaluate_recognition",
             "--board-log", str(BASE / board_log),
             "--report-out", str(BASE / report)],
            cwd=str(BASE), env=env,
            stdout=lf, stderr=subprocess.STDOUT,
        )
        if ret2.returncode != 0:
            print(f"[warn] eval failed: {key} (rc={ret2.returncode})")
    print(f"[done] {key}")


def main() -> None:
    import concurrent.futures
    (BASE / "data/verify/per_video_inject_eval").mkdir(parents=True, exist_ok=True)
    (BASE / "logs/per_video_inject").mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futs = [
            ex.submit(run_one, key, inp, out, rep, blog)
            for key, inp, out, rep, blog in TASKS
        ]
        for f in concurrent.futures.as_completed(futs):
            f.result()
    print("=== all eval done ===")


if __name__ == "__main__":
    main()
