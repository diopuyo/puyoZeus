"""物差し v2 (W8根治版) の STABLE 盤面収集ドライバ (2026-08-14)。

## W8 根治の要点 (docs/KNOWN_WEAKNESSES.md 参照)
旧物差し (board_labels_2026-07-31) は frame_idx が元動画 (削除済) の
タイムラインに紐づいていたため、再DL後に 88.5% のラベルがアンカー不能に
なった。本スクリプトは **収集と同じ実行で参照フレームPNGを同時に保存する**
ことで frame_idx を「今まさに手元にある動画ファイル」に対してのみ有効な
補助キーとして扱い、根本的にズレを起こさない。

## 速度優先のトレードオフ (明示、隠さない)
16本のうち2本 (`_hold_video_c96`=330分, `video_c109`=226分) を含み全長は
数時間規模。全長処理は非現実的なため、動画ごとに `N_CHUNKS_PER_VIDEO` 個の
`CHUNK_SEC` 秒区間 (動画全長に対する相対位置 `CHUNK_OFFSET_FRACTIONS` で
配置、序盤・中盤・終盤に相当する試合を拾う狙い) のみ全フレーム処理する。
チャンク内は一切フレームを間引かない (間引きは状態機械の tsumo_count を
破壊する既知の罠、memory `feedback_indicator_sampling_10frames_2026-07-29`)。

## 使い方
    PYTHONPATH=. ./venv/bin/python -m scripts._collect_yardstick_v2_2026-08-14
"""
from __future__ import annotations

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

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

VIDEO_DIR: Path = Path.home() / "frames"
OUT_NPZ_DIR: Path = _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_2026-08-14"
LOG_DIR: Path = _ROOT / "logs" / "yardstick_v2_collect_2026-08-14"
PYTHON_BIN: Path = _ROOT / "venv" / "bin" / "python"

# 手元実物動画16本 (削除禁止資産、ファイル名は WSL ~/frames/ 実物のまま)
VIDEO_FILENAMES: tuple[str, ...] = (
    "video_c10.mp4", "video_c11.mp4", "video_c12.mp4", "video_c13.mp4",
    "video_c14.mp4", "video_c15.mp4", "video_c16.mp4", "video_c17.mp4",
    "video_c18.mp4", "video_c19.mp4", "video_c20.mp4", "video_c21.mp4",
    "video_c22.mp4", "video_c23.mp4", "_hold_video_c96.mp4", "video_c109.mp4",
)

# 1チャンクの処理長 (秒)。速度優先の明示的トレードオフ。
# 2026-08-14 実測: 本番フラグ (effect-gate/burst-guard-v2等) 有効時の実効
# スループットは8並列下で約7.9生フレーム/秒/プロセス (5分案では1チャンクだけで
# 約38分かかり48ジョブ/8並列で総計約4時間になると判明→再計測の上30秒へ縮小)。
CHUNK_SEC: float = 30.0
# 動画1本あたりのチャンク数 (序盤/中盤/終盤の試合を狙う)
N_CHUNKS_PER_VIDEO: int = 3
# チャンク開始位置 (動画全長に対する相対位置)。冒頭/末尾のロビー・結果画面を避ける。
CHUNK_OFFSET_FRACTIONS: tuple[float, ...] = (0.08, 0.45, 0.80)
# 同時実行プロセス数 (指示上限)
MAX_PARALLEL_WORKERS: int = 8


def video_id_of(filename: str) -> str:
    """ファイル名から video_id (例: "c10") を取り出す。"""
    stem = filename.removeprefix("_hold_").removesuffix(".mp4")
    return stem.removeprefix("video_")


def probe_duration_sec(path: Path) -> float:
    """動画の全長 [秒] を返す (ヘッダ読取のみ、高速)。"""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps if fps > 0 else 0.0


@dataclass(frozen=True)
class ChunkJob:
    """1チャンク分の収集ジョブ。"""

    video_id: str
    video_path: Path
    chunk_idx: int
    start_sec: float
    out_npz: Path
    log_path: Path


def build_jobs() -> list[ChunkJob]:
    """16動画 × N_CHUNKS_PER_VIDEO のジョブ一覧を組み立てる。"""
    jobs: list[ChunkJob] = []
    for filename in VIDEO_FILENAMES:
        vid = video_id_of(filename)
        video_path = VIDEO_DIR / filename
        if not video_path.exists():
            print(f"[skip] 動画が無い: {video_path}")
            continue
        duration = probe_duration_sec(video_path)
        if duration <= 0:
            print(f"[skip] 動画を開けない: {video_path}")
            continue
        for k, frac in enumerate(CHUNK_OFFSET_FRACTIONS[:N_CHUNKS_PER_VIDEO]):
            start_sec = max(0.0, frac * duration)
            jobs.append(ChunkJob(
                video_id=vid, video_path=video_path, chunk_idx=k,
                start_sec=start_sec,
                out_npz=OUT_NPZ_DIR / f"{vid}_chunk{k}.npz",
                log_path=LOG_DIR / f"{vid}_chunk{k}.log",
            ))
    return jobs


def run_job(job: ChunkJob) -> tuple[str, bool, float]:
    """1ジョブを subprocess で実行する。(job_name, success, elapsed_sec) を返す。"""
    job_name = f"{job.video_id}_chunk{job.chunk_idx}"
    cmd = (
        [str(PYTHON_BIN), "-u", "-m", "scripts._collect_lean_1t",
         "--video", str(job.video_path), "--out-npz", str(job.out_npz),
         "--start-sec", str(job.start_sec), "--max-sec", str(CHUNK_SEC)]
        + collect_flags().split()
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
    OUT_NPZ_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    print(f"ジョブ数: {len(jobs)} (動画{len({j.video_id for j in jobs})}本 × 最大{N_CHUNKS_PER_VIDEO}チャンク)")
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
