"""物差し v2 55盤面が属するチャンクのみを 構成b/c で再収集する (2026-08-15)。

`scripts/_collect_yardstick_v2_2026-08-14.py` (構成a=本番採用構成) と同じ
チャンク切り出し方式 (CHUNK_SEC=30秒、CHUNK_OFFSET_FRACTIONS、動画あたり最大3
チャンク) を踏襲するが、55有効盤面が実際に属するチャンクのみに限定し、
以下2構成の追加フラグを付けて再収集する:

    構成b: 本番採用構成 + --stable-majority-window
    構成c: 本番採用構成 + OJAMA_FALL系3フラグ
           (--enable-ojama-fall-placement-override
            --enable-ojama-fall-entry-hardening
            --enable-ojama-fall-scoped-exit)

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._collect_yardstick_v2_bc_2026-08-15 --config b
    PYTHONPATH=. ./venv/bin/python -m scripts._collect_yardstick_v2_bc_2026-08-15 --config c
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.production_config import collect_flags  # noqa: E402

VIDEO_DIR: Path = Path.home() / "frames"
LOG_DIR_ROOT: Path = _ROOT / "logs" / "yardstick_v2_collect_bc_2026-08-15"
PYTHON_BIN: Path = _ROOT / "venv" / "bin" / "python"

# _collect_yardstick_v2_2026-08-14.py と同一の定数 (窓の再現性のため一致必須)
CHUNK_SEC: float = 30.0
CHUNK_OFFSET_FRACTIONS: tuple[float, ...] = (0.08, 0.45, 0.80)
MAX_PARALLEL_WORKERS: int = 8

FLAGS_B: str = "--stable-majority-window"
FLAGS_C: str = (
    "--enable-ojama-fall-placement-override "
    "--enable-ojama-fall-entry-hardening "
    "--enable-ojama-fall-scoped-exit"
)

# scripts/_score_yardstick_v2_2026-08-14.py --list-chunks の出力そのまま
# (video_id, chunk_idx) の28件 (55盤面が属するチャンクのみ)
NEEDED_CHUNKS: tuple[tuple[str, int], ...] = (
    ("c10", 0), ("c10", 1), ("c10", 2),
    ("c109", 0), ("c109", 2),
    ("c11", 0), ("c11", 1),
    ("c12", 1), ("c12", 2),
    ("c13", 0), ("c13", 1), ("c13", 2),
    ("c14", 1),
    ("c16", 1), ("c16", 2),
    ("c17", 0),
    ("c18", 1), ("c18", 2),
    ("c20", 0), ("c20", 1), ("c20", 2),
    ("c21", 1), ("c21", 2),
    ("c22", 0), ("c22", 1),
    ("c23", 1), ("c23", 2),
    ("c96", 2),
)


def video_filename_of(video_id: str) -> str:
    """video_id (例: "c96") から実ファイル名を返す (c96のみ "_hold_" 接頭辞)。"""
    if video_id == "c96":
        return "_hold_video_c96.mp4"
    return f"video_{video_id}.mp4"


def probe_duration_sec(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps if fps > 0 else 0.0


@dataclass(frozen=True)
class ChunkJob:
    video_id: str
    video_path: Path
    chunk_idx: int
    start_sec: float
    out_npz: Path
    log_path: Path
    extra_flags: str


def build_jobs(out_dir: Path, log_dir: Path, extra_flags: str) -> list[ChunkJob]:
    jobs: list[ChunkJob] = []
    duration_cache: dict[str, float] = {}
    for video_id, chunk_idx in NEEDED_CHUNKS:
        filename = video_filename_of(video_id)
        video_path = VIDEO_DIR / filename
        if not video_path.exists():
            print(f"[skip] 動画が無い: {video_path}")
            continue
        if video_id not in duration_cache:
            duration_cache[video_id] = probe_duration_sec(video_path)
        duration = duration_cache[video_id]
        frac = CHUNK_OFFSET_FRACTIONS[chunk_idx]
        start_sec = max(0.0, frac * duration)
        jobs.append(ChunkJob(
            video_id=video_id, video_path=video_path, chunk_idx=chunk_idx,
            start_sec=start_sec,
            out_npz=out_dir / f"{video_id}_chunk{chunk_idx}.npz",
            log_path=log_dir / f"{video_id}_chunk{chunk_idx}.log",
            extra_flags=extra_flags,
        ))
    return jobs


def run_job(job: ChunkJob) -> tuple[str, bool, float]:
    job_name = f"{job.video_id}_chunk{job.chunk_idx}"
    cmd = (
        [str(PYTHON_BIN), "-u", "-m", "scripts._collect_lean_1t",
         "--video", str(job.video_path), "--out-npz", str(job.out_npz),
         "--start-sec", str(job.start_sec), "--max-sec", str(CHUNK_SEC)]
        + collect_flags().split()
        + job.extra_flags.split()
        + ["--with-next"]
    )
    start = time.monotonic()
    with job.log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - start
    ok = proc.returncode == 0 and job.out_npz.exists()
    print(f"[{'OK' if ok else 'FAIL'}] {job_name} ({elapsed:.1f}s)")
    return job_name, ok, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["b", "c"], required=True)
    args = ap.parse_args()

    if args.config == "b":
        out_dir = _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_b_smw_2026-08-15"
        extra_flags = FLAGS_B
    else:
        out_dir = _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_c_ojamafall_2026-08-15"
        extra_flags = FLAGS_C
    log_dir = LOG_DIR_ROOT / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(out_dir, log_dir, extra_flags)
    print(f"[config={args.config}] ジョブ数: {len(jobs)} extra_flags={extra_flags!r}")
    t0 = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as ex:
        for r in ex.map(run_job, jobs):
            results.append(r)
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"完了: {n_ok}/{len(results)} 成功 ({time.monotonic() - t0:.1f}s)")
    if n_ok < len(results):
        print("失敗ジョブ:")
        for name, ok, _ in results:
            if not ok:
                print(f"  {name}")


if __name__ == "__main__":
    main()
